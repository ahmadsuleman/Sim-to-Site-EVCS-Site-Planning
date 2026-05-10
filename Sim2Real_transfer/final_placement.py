#!/usr/bin/env python3
"""
Final map-balanced EVCS placement inference.

This script turns ranked candidates into final EVCS placement decisions while
adding explicit spatial-distribution constraints derived from the city map.

It is designed for the simulator exports used in the EVCS sim-to-real study.

Inputs:
  - dataset_index.csv with columns: city_id,country,data_mode,seed,data_dir
  - optional ranked_candidates_all_methods.csv from sim-to-real experiment
  - city simulator CSV folders containing candidate_feature_matrix, demand_points,
    graph_nodes, and graph_edges files.

Outputs:
  - final ranked candidates
  - final selected sites
  - zone distribution diagnostics
  - city maps with selected stations
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc

try:
    from scipy.optimize import LinearConstraint, Bounds, milp
    SCIPY_MILP_AVAILABLE = True
except Exception:
    SCIPY_MILP_AVAILABLE = False


# -----------------------------
# Utilities
# -----------------------------

def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dirs(base: Path) -> Dict[str, Path]:
    dirs = {
        "tables": base / "tables",
        "rankings": base / "rankings",
        "selected": base / "selected",
        "figures": base / "figures",
        "logs": base / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def find_file(data_dir: Path, stems: Sequence[str]) -> Optional[Path]:
    """Find a file by stem-like keyword, accepting variants like name(2).csv."""
    if not data_dir.exists():
        return None
    files = sorted(data_dir.glob("*.csv")) + sorted(data_dir.glob("*.json"))
    names = [(p, p.name.lower()) for p in files]
    for stem in stems:
        s = stem.lower()
        # Prefer exact prefix matches.
        for p, name in names:
            if name.startswith(s) and p.suffix.lower() in {".csv", ".json"}:
                return p
        # Fallback substring.
        for p, name in names:
            if s in name and p.suffix.lower() in {".csv", ".json"}:
                return p
    return None


def require_file(data_dir: Path, stems: Sequence[str]) -> Path:
    p = find_file(data_dir, stems)
    if p is None:
        raise FileNotFoundError(f"Could not find any of {stems} in {data_dir}")
    return p


def normalize(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    lo = float(s.min())
    hi = float(s.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    r = 6371.0088
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def infer_lat_lon(df: pd.DataFrame) -> Tuple[str, str]:
    lat_candidates = ["lat", "latitude", "y_lat", "candidate_lat"]
    lon_candidates = ["lon", "lng", "longitude", "x_lon", "candidate_lon"]
    lat = next((c for c in lat_candidates if c in df.columns), None)
    lon = next((c for c in lon_candidates if c in df.columns), None)
    if lat is None or lon is None:
        raise ValueError(f"Could not infer lat/lon columns from: {list(df.columns)}")
    return lat, lon


def parse_linestring_wkt(wkt: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not isinstance(wkt, str) or "LINESTRING" not in wkt.upper():
        return None
    m = re.search(r"LINESTRING\s*\((.*?)\)", wkt, flags=re.IGNORECASE)
    if not m:
        return None
    coords = []
    for token in m.group(1).split(","):
        parts = token.strip().split()
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                coords.append((lon, lat))
            except ValueError:
                continue
    if len(coords) < 2:
        return None
    arr = np.array(coords)
    return arr[:, 0], arr[:, 1]


# -----------------------------
# Data preparation
# -----------------------------

@dataclass
class CityData:
    city_id: str
    country: str
    data_mode: str
    data_dir: Path
    candidates: pd.DataFrame
    demand: pd.DataFrame
    nodes: Optional[pd.DataFrame]
    edges: Optional[pd.DataFrame]


def load_city_data(row: pd.Series) -> CityData:
    data_dir = Path(row["data_dir"])
    cand_path = require_file(data_dir, ["candidate_feature_matrix", "candidate_sites"])
    demand_path = require_file(data_dir, ["demand_points"])
    nodes_path = find_file(data_dir, ["graph_nodes", "nodes"])
    edges_path = find_file(data_dir, ["graph_edges", "edges"])

    candidates = pd.read_csv(cand_path)
    demand = pd.read_csv(demand_path)
    nodes = pd.read_csv(nodes_path) if nodes_path else None
    edges = pd.read_csv(edges_path) if edges_path else None

    if "candidate_id" not in candidates.columns:
        candidates = candidates.copy()
        candidates["candidate_id"] = [f"cand_{row['city_id']}_{i:04d}" for i in range(len(candidates))]

    return CityData(
        city_id=str(row["city_id"]),
        country=str(row.get("country", "")),
        data_mode=str(row["data_mode"]),
        data_dir=data_dir,
        candidates=candidates,
        demand=demand,
        nodes=nodes,
        edges=edges,
    )


def load_rank_scores(
    rankings_path: Optional[Path],
    city_id: str,
    method_id: str,
    shortlist_size: int,
) -> Optional[pd.DataFrame]:
    if rankings_path is None or not rankings_path.exists():
        return None
    ranks = pd.read_csv(rankings_path)
    required = {"city_id", "method", "candidate_id", "rank_score"}
    if not required.issubset(ranks.columns):
        # Older output may use method_id instead of method.
        if "method_id" in ranks.columns:
            ranks = ranks.rename(columns={"method_id": "method"})
        if not required.issubset(ranks.columns):
            return None

    sub = ranks[ranks["city_id"].astype(str).eq(city_id)].copy()
    if "data_mode" in sub.columns:
        sub = sub[sub["data_mode"].astype(str).eq("real")]
    if "method" in sub.columns:
        sub = sub[sub["method"].astype(str).eq(method_id)]
    if "shortlist_size" in sub.columns:
        # Prefer exact size, but if missing fall back to any size for method.
        exact = sub[pd.to_numeric(sub["shortlist_size"], errors="coerce").eq(shortlist_size)]
        if len(exact):
            sub = exact
    if sub.empty:
        return None
    return sub[["candidate_id", "rank", "rank_score"]].drop_duplicates("candidate_id")


def compute_kpi_score(cand: pd.DataFrame, cfg: dict) -> pd.Series:
    # Uses available columns and normalizes weights implicitly through fixed components.
    cols = cand.columns
    score = pd.Series(np.zeros(len(cand)), index=cand.index, dtype=float)
    weight_sum = 0.0

    candidates = [
        ("demand_capture", 0.25),
        ("coverage_gain", 0.15),
        ("accessibility_benefit", 0.15),
        ("grid_feasibility", 0.15),
        ("land_feasibility", 0.10),
        ("cost_efficiency", 0.10),
        ("equity_benefit", 0.10),
        ("candidate_score_for_inclusion", 0.20),
        ("activity_intensity", 0.10),
        ("road_hierarchy_score", 0.10),
    ]
    for col, w in candidates:
        if col in cols:
            score += w * normalize(cand[col])
            weight_sum += w

    if "competition_penalty" in cols:
        score -= 0.05 * normalize(cand["competition_penalty"])
        weight_sum += 0.05

    if weight_sum <= 0:
        return pd.Series(np.ones(len(cand)), index=cand.index)
    return normalize(score / weight_sum)


def add_spatial_features(cand: pd.DataFrame, demand: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cand = cand.copy()
    lat_c, lon_c = infer_lat_lon(cand)
    lat_d, lon_d = infer_lat_lon(demand)

    center_lat = float(cand[lat_c].mean())
    center_lon = float(cand[lon_c].mean())
    cand["distance_to_center_km"] = haversine_km(
        cand[lat_c].values, cand[lon_c].values, center_lat, center_lon
    )

    # nearest candidate distance
    coords_lat = cand[lat_c].values
    coords_lon = cand[lon_c].values
    nearest = []
    for i in range(len(cand)):
        d = haversine_km(coords_lat[i], coords_lon[i], coords_lat, coords_lon)
        d[i] = np.inf
        nearest.append(float(np.min(d)) if len(d) > 1 else np.nan)
    cand["nearest_candidate_distance_km"] = nearest

    # Local demand kernel around each candidate.
    demand_mass_col = "demand_mass" if "demand_mass" in demand.columns else None
    if demand_mass_col is None:
        demand_mass = np.ones(len(demand))
    else:
        demand_mass = pd.to_numeric(demand[demand_mass_col], errors="coerce").fillna(0).values
    kernel_km = float(cfg.get("scoring", {}).get("demand_kernel_km", 2.0))
    local_demand = []
    dlat = demand[lat_d].values
    dlon = demand[lon_d].values
    for _, row in cand.iterrows():
        dist = haversine_km(row[lat_c], row[lon_c], dlat, dlon)
        local_demand.append(float(np.sum(demand_mass * np.exp(-dist / max(kernel_km, 1e-6)))))
    cand["local_demand_score"] = normalize(pd.Series(local_demand, index=cand.index))

    return cand


def assign_grid_zones(cand: pd.DataFrame, rows: int, cols: int) -> pd.DataFrame:
    cand = cand.copy()
    lat_col, lon_col = infer_lat_lon(cand)
    lat = cand[lat_col].astype(float)
    lon = cand[lon_col].astype(float)

    lat_edges = np.linspace(lat.min(), lat.max() + 1e-12, rows + 1)
    lon_edges = np.linspace(lon.min(), lon.max() + 1e-12, cols + 1)
    r = np.clip(np.digitize(lat, lat_edges) - 1, 0, rows - 1)
    c = np.clip(np.digitize(lon, lon_edges) - 1, 0, cols - 1)
    cand["zone_row"] = r
    cand["zone_col"] = c
    cand["zone_id"] = [f"z{int(rr)}_{int(cc)}" for rr, cc in zip(r, c)]
    return cand


def build_final_score(cand: pd.DataFrame, rank_scores: Optional[pd.DataFrame], cfg: dict) -> pd.DataFrame:
    cand = cand.copy()

    if rank_scores is not None:
        cand = cand.merge(rank_scores, on="candidate_id", how="left")
        cand["rank_score_source"] = "provided_rankings"
        cand["rank_score"] = cand["rank_score"].fillna(cand["rank_score"].median())
    else:
        cand["rank_score"] = compute_kpi_score(cand, cfg)
        cand["rank"] = cand["rank_score"].rank(ascending=False, method="first")
        cand["rank_score_source"] = "computed_kpi_fallback"

    cand["rank_score_norm"] = normalize(cand["rank_score"])

    # components
    if "suitability_score" in cand.columns:
        suitability = normalize(cand["suitability_score"])
    else:
        suitability = compute_kpi_score(cand, cfg)
    cand["suitability_component"] = suitability

    if "equity_benefit" in cand.columns:
        equity = normalize(cand["equity_benefit"])
    elif "equity_need_score" in cand.columns:
        equity = normalize(cand["equity_need_score"])
    else:
        equity = pd.Series(np.zeros(len(cand)), index=cand.index)
    cand["equity_component"] = equity

    if "cost_efficiency" in cand.columns:
        cost_eff = normalize(cand["cost_efficiency"])
    elif "land_cost_proxy" in cand.columns:
        cost_eff = 1.0 - normalize(cand["land_cost_proxy"])
    elif "cost_proxy_model" in cand.columns:
        cost_eff = 1.0 - normalize(cand["cost_proxy_model"])
    else:
        cost_eff = pd.Series(np.zeros(len(cand)), index=cand.index)
    cand["cost_efficiency_component"] = cost_eff

    weights = cfg.get("scoring", {})
    w_rank = float(weights.get("rank_weight", 0.45))
    w_dem = float(weights.get("local_demand_weight", 0.25))
    w_suit = float(weights.get("suitability_weight", 0.15))
    w_eq = float(weights.get("equity_weight", 0.10))
    w_cost = float(weights.get("cost_efficiency_weight", 0.05))

    total_w = w_rank + w_dem + w_suit + w_eq + w_cost
    if total_w <= 0:
        total_w = 1.0

    cand["final_rank_score"] = (
        w_rank * cand["rank_score_norm"]
        + w_dem * cand["local_demand_score"]
        + w_suit * cand["suitability_component"]
        + w_eq * cand["equity_component"]
        + w_cost * cand["cost_efficiency_component"]
    ) / total_w
    cand["final_rank"] = cand["final_rank_score"].rank(ascending=False, method="first").astype(int)
    return cand.sort_values("final_rank")


# -----------------------------
# Balanced selection
# -----------------------------

@dataclass
class PlacementResult:
    selected_ids: List[str]
    status: str
    solver: str
    objective: float
    min_active_zones_used: int
    max_per_zone_used: int
    min_separation_km_used: float
    relaxation_notes: str


def solve_balanced_mip(
    cand: pd.DataFrame,
    k: int,
    min_active_zones: int,
    max_per_zone: int,
    min_sep_km: float,
) -> Optional[PlacementResult]:
    if not SCIPY_MILP_AVAILABLE:
        return None
    n = len(cand)
    if n == 0 or k <= 0:
        return None
    k = min(k, n)

    zones = sorted(cand["zone_id"].unique())
    z_index = {z: idx for idx, z in enumerate(zones)}
    z_count = len(zones)
    var_count = n + z_count

    c = np.zeros(var_count)
    c[:n] = -cand["final_rank_score"].astype(float).values  # maximize

    lb = np.zeros(var_count)
    ub = np.ones(var_count)
    integrality = np.ones(var_count)

    constraints = []
    lower = []
    upper = []

    # Sum x = k
    row = np.zeros(var_count)
    row[:n] = 1.0
    constraints.append(row)
    lower.append(k)
    upper.append(k)

    # Zone constraints and links.
    for z in zones:
        idxs = np.where(cand["zone_id"].values == z)[0]
        y_idx = n + z_index[z]

        # max per zone: sum x_i <= max_per_zone
        row = np.zeros(var_count)
        row[idxs] = 1.0
        constraints.append(row)
        lower.append(-np.inf)
        upper.append(max_per_zone)

        # sum x_i <= k * y_z
        row = np.zeros(var_count)
        row[idxs] = 1.0
        row[y_idx] = -k
        constraints.append(row)
        lower.append(-np.inf)
        upper.append(0.0)

        # y_z <= sum x_i  => y_z - sum x_i <= 0
        row = np.zeros(var_count)
        row[idxs] = -1.0
        row[y_idx] = 1.0
        constraints.append(row)
        lower.append(-np.inf)
        upper.append(0.0)

    # Minimum active zones: sum y_z >= min_active_zones
    row = np.zeros(var_count)
    row[n:] = 1.0
    constraints.append(row)
    lower.append(min(min_active_zones, k, z_count))
    upper.append(np.inf)

    # Pairwise minimum separation.
    lat_col, lon_col = infer_lat_lon(cand)
    lat = cand[lat_col].values
    lon = cand[lon_col].values
    pair_count = 0
    if min_sep_km > 0:
        for i in range(n):
            d = haversine_km(lat[i], lon[i], lat[i + 1 :], lon[i + 1 :])
            close_js = np.where(d < min_sep_km)[0] + i + 1
            for j in close_js:
                row = np.zeros(var_count)
                row[i] = 1.0
                row[j] = 1.0
                constraints.append(row)
                lower.append(-np.inf)
                upper.append(1.0)
                pair_count += 1

    A = np.vstack(constraints)
    lc = LinearConstraint(A, np.array(lower, dtype=float), np.array(upper, dtype=float))
    res = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=lc,
        options={"time_limit": 90, "mip_rel_gap": 0.02},
    )
    if not res.success or res.x is None:
        return None
    x = res.x[:n]
    selected = cand.loc[x >= 0.5, "candidate_id"].astype(str).tolist()
    if len(selected) != k:
        return None
    return PlacementResult(
        selected_ids=selected,
        status="optimal_or_feasible",
        solver="scipy_milp",
        objective=float(-res.fun),
        min_active_zones_used=min_active_zones,
        max_per_zone_used=max_per_zone,
        min_separation_km_used=min_sep_km,
        relaxation_notes=f"pairwise_separation_constraints={pair_count}",
    )


def solve_balanced_greedy(
    cand: pd.DataFrame,
    k: int,
    min_active_zones: int,
    max_per_zone: int,
    min_sep_km: float,
) -> PlacementResult:
    cand = cand.sort_values("final_rank_score", ascending=False).reset_index(drop=True)
    lat_col, lon_col = infer_lat_lon(cand)
    selected_idx: List[int] = []
    zone_counts: Dict[str, int] = {}

    def can_add(idx: int, strict: bool = True) -> bool:
        z = str(cand.loc[idx, "zone_id"])
        if zone_counts.get(z, 0) >= max_per_zone:
            return False
        if strict and min_sep_km > 0 and selected_idx:
            d = haversine_km(
                cand.loc[idx, lat_col],
                cand.loc[idx, lon_col],
                cand.loc[selected_idx, lat_col].values,
                cand.loc[selected_idx, lon_col].values,
            )
            if np.any(d < min_sep_km):
                return False
        return True

    # First pass: favor new zones.
    for idx in range(len(cand)):
        if len(selected_idx) >= k:
            break
        z = str(cand.loc[idx, "zone_id"])
        if zone_counts.get(z, 0) == 0 and can_add(idx, strict=True):
            selected_idx.append(idx)
            zone_counts[z] = zone_counts.get(z, 0) + 1
            if len(zone_counts) >= min_active_zones and len(selected_idx) >= min(k, min_active_zones):
                break

    # Second pass: fill while respecting constraints.
    for idx in range(len(cand)):
        if len(selected_idx) >= k:
            break
        if idx in selected_idx:
            continue
        if can_add(idx, strict=True):
            z = str(cand.loc[idx, "zone_id"])
            selected_idx.append(idx)
            zone_counts[z] = zone_counts.get(z, 0) + 1

    # Relax min separation if still not enough.
    if len(selected_idx) < k:
        for idx in range(len(cand)):
            if len(selected_idx) >= k:
                break
            if idx in selected_idx:
                continue
            if can_add(idx, strict=False):
                z = str(cand.loc[idx, "zone_id"])
                selected_idx.append(idx)
                zone_counts[z] = zone_counts.get(z, 0) + 1

    selected_ids = cand.loc[selected_idx, "candidate_id"].astype(str).tolist()
    objective = float(cand.loc[selected_idx, "final_rank_score"].sum()) if selected_idx else 0.0
    return PlacementResult(
        selected_ids=selected_ids,
        status="greedy_feasible" if len(selected_ids) == k else "greedy_partial",
        solver="balanced_greedy",
        objective=objective,
        min_active_zones_used=min_active_zones,
        max_per_zone_used=max_per_zone,
        min_separation_km_used=min_sep_km,
        relaxation_notes="greedy fallback used; separation relaxed only if required to fill K",
    )


def solve_with_relaxation(cand: pd.DataFrame, cfg: dict) -> PlacementResult:
    k = int(cfg.get("k", 10))
    mb = cfg.get("map_balance", {})
    grid_zones = int(cand["zone_id"].nunique())
    min_active = int(math.ceil(float(mb.get("min_active_zones_fraction_of_k", 0.60)) * k))
    min_active = min(min_active, k, grid_zones)
    max_per_zone = int(mb.get("max_per_zone", 2))
    min_sep = float(mb.get("min_separation_km", 1.5))
    allow_relax = bool(mb.get("allow_relaxation", True))

    attempts: List[Tuple[int, int, float]] = [(min_active, max_per_zone, min_sep)]
    if allow_relax:
        attempts += [
            (min_active, max_per_zone, min_sep * 0.75),
            (max(1, min_active - 1), max_per_zone, min_sep * 0.75),
            (max(1, min_active - 2), max_per_zone + 1, min_sep * 0.50),
            (max(1, min_active - 3), max_per_zone + 2, 0.0),
        ]

    notes = []
    for a_min, a_max, a_sep in attempts:
        res = solve_balanced_mip(cand, k, a_min, a_max, a_sep)
        if res is not None:
            if notes:
                res.relaxation_notes += "; relaxed_after=" + " | ".join(notes)
            return res
        notes.append(f"failed(min_zones={a_min},max_per_zone={a_max},sep={a_sep:.2f})")

    # Greedy fallback with last attempted values.
    a_min, a_max, a_sep = attempts[-1]
    res = solve_balanced_greedy(cand, k, a_min, a_max, a_sep)
    res.relaxation_notes += "; " + " | ".join(notes)
    return res


# -----------------------------
# Tables and diagnostics
# -----------------------------

def zone_distribution_table(cand: pd.DataFrame, selected_ids: Sequence[str]) -> pd.DataFrame:
    selected_set = set(map(str, selected_ids))
    tmp = cand.copy()
    tmp["selected"] = tmp["candidate_id"].astype(str).isin(selected_set).astype(int)
    agg = tmp.groupby("zone_id").agg(
        zone_row=("zone_row", "first"),
        zone_col=("zone_col", "first"),
        candidate_count=("candidate_id", "count"),
        selected_count=("selected", "sum"),
        mean_score=("final_rank_score", "mean"),
        max_score=("final_rank_score", "max"),
    ).reset_index()
    return agg.sort_values(["zone_row", "zone_col"])


def summarize_city(city: CityData, cand: pd.DataFrame, result: PlacementResult) -> dict:
    selected = cand[cand["candidate_id"].astype(str).isin(result.selected_ids)]
    return {
        "city_id": city.city_id,
        "country": city.country,
        "data_mode": city.data_mode,
        "candidate_count": len(cand),
        "selected_count": len(selected),
        "k_requested": int(result.selected_ids and len(result.selected_ids) or 0),
        "active_zones_selected": int(selected["zone_id"].nunique()) if len(selected) else 0,
        "candidate_grid_zones": int(cand["zone_id"].nunique()),
        "selection_grid_coverage_rate": float(selected["zone_id"].nunique() / max(1, cand["zone_id"].nunique())) if len(selected) else 0.0,
        "mean_final_rank_score": float(selected["final_rank_score"].mean()) if len(selected) else np.nan,
        "mean_local_demand_score": float(selected["local_demand_score"].mean()) if len(selected) else np.nan,
        "mean_suitability_component": float(selected["suitability_component"].mean()) if len(selected) else np.nan,
        "mean_cost_efficiency_component": float(selected["cost_efficiency_component"].mean()) if len(selected) else np.nan,
        "mean_distance_to_center_km": float(selected["distance_to_center_km"].mean()) if len(selected) else np.nan,
        "mean_nearest_candidate_distance_km": float(selected["nearest_candidate_distance_km"].mean()) if len(selected) else np.nan,
        "solver": result.solver,
        "status": result.status,
        "objective": result.objective,
        "min_active_zones_used": result.min_active_zones_used,
        "max_per_zone_used": result.max_per_zone_used,
        "min_separation_km_used": result.min_separation_km_used,
        "relaxation_notes": result.relaxation_notes,
    }


# -----------------------------
# Plotting
# -----------------------------

def plot_city_map(
    city: CityData,
    cand: pd.DataFrame,
    result: PlacementResult,
    out_base: Path,
    cfg: dict,
    title_suffix: str,
) -> None:
    plots = cfg.get("plots", {})
    dpi = int(plots.get("dpi", 350))
    formats = plots.get("figure_format", ["png"])
    show_grid = bool(plots.get("show_grid", True))
    label_selected = bool(plots.get("label_selected_sites", True))
    max_edges = int(plots.get("max_road_edges", 12000))
    max_demand = int(plots.get("max_demand_points", 5000))

    lat_col, lon_col = infer_lat_lon(cand)
    selected = cand[cand["candidate_id"].astype(str).isin(set(result.selected_ids))].copy()

    fig, ax = plt.subplots(figsize=(9.5, 8.0))

    # Road graph background.
    if city.edges is not None and "geometry_wkt" in city.edges.columns:
        edges = city.edges.head(max_edges)
        for wkt in edges["geometry_wkt"].dropna().values:
            parsed = parse_linestring_wkt(wkt)
            if parsed is not None:
                xs, ys = parsed
                ax.plot(xs, ys, linewidth=0.35, alpha=0.28, zorder=1)
    elif city.nodes is not None:
        nlat, nlon = infer_lat_lon(city.nodes)
        ax.scatter(city.nodes[nlon], city.nodes[nlat], s=0.5, alpha=0.15, zorder=1)

    # Demand points.
    if city.demand is not None and len(city.demand):
        d = city.demand
        if len(d) > max_demand:
            d = d.sample(max_demand, random_state=int(cfg.get("seed", 42)))
        dlat, dlon = infer_lat_lon(d)
        sizes = None
        if "demand_mass" in d.columns:
            sizes = 2 + 10 * normalize(d["demand_mass"]).values
        else:
            sizes = 4
        ax.scatter(d[dlon], d[dlat], s=sizes, alpha=0.18, marker=".", label="Demand points", zorder=2)

    # All candidates.
    ax.scatter(
        cand[lon_col], cand[lat_col],
        s=18, alpha=0.45, marker="o", edgecolors="none", label="Candidate sites", zorder=3,
    )

    # Selected candidates.
    if len(selected):
        ax.scatter(
            selected[lon_col], selected[lat_col],
            s=130, marker="*", edgecolors="black", linewidths=0.7,
            label="Final selected sites", zorder=5,
        )
        if label_selected:
            for _, row in selected.iterrows():
                label = str(int(row.get("final_rank", 0))) if pd.notna(row.get("final_rank", np.nan)) else ""
                ax.text(row[lon_col], row[lat_col], f" {label}", fontsize=8, weight="bold", zorder=6)

    # Grid.
    if show_grid:
        lat_min, lat_max = cand[lat_col].min(), cand[lat_col].max()
        lon_min, lon_max = cand[lon_col].min(), cand[lon_col].max()
        rows = int(cfg.get("map_balance", {}).get("grid_rows", 4))
        cols = int(cfg.get("map_balance", {}).get("grid_cols", 4))
        for x in np.linspace(lon_min, lon_max, cols + 1):
            ax.axvline(x, linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
        for y in np.linspace(lat_min, lat_max, rows + 1):
            ax.axhline(y, linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)

    ax.set_title(f"{city.city_id}: final EVCS placements {title_suffix}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", frameon=True, fontsize=8)
    ax.grid(False)
    fig.tight_layout()

    for fmt in formats:
        fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_ranking_map(city: CityData, cand: pd.DataFrame, out_base: Path, cfg: dict) -> None:
    plots = cfg.get("plots", {})
    dpi = int(plots.get("dpi", 350))
    formats = plots.get("figure_format", ["png"])
    lat_col, lon_col = infer_lat_lon(cand)

    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    if city.edges is not None and "geometry_wkt" in city.edges.columns:
        for wkt in city.edges.head(int(plots.get("max_road_edges", 12000)))["geometry_wkt"].dropna().values:
            parsed = parse_linestring_wkt(wkt)
            if parsed is not None:
                xs, ys = parsed
                ax.plot(xs, ys, linewidth=0.35, alpha=0.20, zorder=1)

    sc = ax.scatter(
        cand[lon_col], cand[lat_col], c=cand["final_rank_score"],
        s=30, alpha=0.85, marker="o", edgecolors="none", zorder=3,
    )
    top = cand.sort_values("final_rank").head(20)
    ax.scatter(top[lon_col], top[lat_col], s=90, marker="o", facecolors="none", edgecolors="black", linewidths=0.9, label="Top-20 ranked", zorder=4)
    for _, row in top.head(10).iterrows():
        ax.text(row[lon_col], row[lat_col], f" {int(row['final_rank'])}", fontsize=7, zorder=5)
    fig.colorbar(sc, ax=ax, label="Final rank score")
    ax.set_title(f"{city.city_id}: ranked candidate map")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", frameon=True, fontsize=8)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main pipeline
# -----------------------------

def process_city(row: pd.Series, cfg: dict, rankings_path: Optional[Path], dirs: Dict[str, Path]) -> dict:
    city = load_city_data(row)
    cand = city.candidates.copy()
    cand = add_spatial_features(cand, city.demand, cfg)
    mb = cfg.get("map_balance", {})
    cand = assign_grid_zones(cand, int(mb.get("grid_rows", 4)), int(mb.get("grid_cols", 4)))

    method_id = str(cfg.get("method_id", "M5_synthetic_to_hybrid_topM_milp"))
    shortlist_size = int(cfg.get("shortlist_size", 40))
    rank_scores = load_rank_scores(rankings_path, city.city_id, method_id, shortlist_size)
    cand = build_final_score(cand, rank_scores, cfg)

    result = solve_with_relaxation(cand, cfg)
    selected_set = set(result.selected_ids)
    cand["final_selected"] = cand["candidate_id"].astype(str).isin(selected_set).astype(int)

    # Exports.
    ranked_cols = [
        "candidate_id", "city_id", "data_mode", "lat", "lon", "final_rank", "final_rank_score",
        "rank_score", "rank_score_source", "local_demand_score", "suitability_component",
        "equity_component", "cost_efficiency_component", "distance_to_center_km",
        "nearest_candidate_distance_km", "zone_id", "zone_row", "zone_col", "final_selected",
    ]
    ranked_cols = [c for c in ranked_cols if c in cand.columns]
    cand.sort_values("final_rank")[ranked_cols].to_csv(
        dirs["rankings"] / f"final_ranked_sites_{city.city_id}.csv", index=False
    )

    selected = cand[cand["final_selected"].eq(1)].sort_values("final_rank")
    selected[ranked_cols].to_csv(dirs["selected"] / f"final_selected_sites_{city.city_id}.csv", index=False)

    zone_tbl = zone_distribution_table(cand, result.selected_ids)
    zone_tbl.to_csv(dirs["tables"] / f"final_zone_distribution_{city.city_id}.csv", index=False)

    plot_city_map(
        city, cand, result,
        dirs["figures"] / f"final_placement_map_{city.city_id}",
        cfg,
        title_suffix=f"(K={cfg.get('k', 10)}, balanced zones)",
    )
    plot_ranking_map(city, cand, dirs["figures"] / f"final_ranking_map_{city.city_id}", cfg)

    summary = summarize_city(city, cand, result)
    summary["method_id_used_for_ranking"] = method_id if rank_scores is not None else "computed_kpi_fallback"
    summary["shortlist_size_used_for_ranking"] = shortlist_size if rank_scores is not None else np.nan
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Map-balanced final EVCS placement inference.")
    parser.add_argument("--dataset-index", required=True, type=Path)
    parser.add_argument("--rankings", type=Path, default=None, help="ranked_candidates_all_methods.csv from sim-to-real experiment")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--city-ids", nargs="*", default=None, help="Optional city IDs to process")
    parser.add_argument("--mode", default=None, help="Override data mode, default from config")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    mode = args.mode or str(cfg.get("mode", "real"))
    dirs = ensure_dirs(args.output_dir)

    idx = pd.read_csv(args.dataset_index)
    required = {"city_id", "data_mode", "data_dir"}
    missing = required - set(idx.columns)
    if missing:
        raise SystemExit(f"dataset_index missing columns: {sorted(missing)}")
    idx = idx[idx["data_mode"].astype(str).eq(mode)].copy()
    if args.city_ids:
        idx = idx[idx["city_id"].astype(str).isin(args.city_ids)].copy()
    if idx.empty:
        raise SystemExit(f"No rows found for mode={mode} and city_ids={args.city_ids}")

    summaries = []
    errors = []
    for _, row in idx.sort_values("city_id").iterrows():
        try:
            print(f"Processing {row['city_id']} ({row['data_mode']})...")
            summaries.append(process_city(row, cfg, args.rankings, dirs))
        except Exception as exc:
            msg = f"{row.get('city_id', 'unknown')}: {type(exc).__name__}: {exc}"
            print("ERROR", msg, file=sys.stderr)
            errors.append(msg)

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(dirs["tables"] / "final_placement_summary.csv", index=False)
        try:
            summary_df.to_excel(dirs["tables"] / "final_placement_summary.xlsx", index=False)
        except Exception:
            pass
    if errors:
        (dirs["logs"] / "errors.txt").write_text("\n".join(errors), encoding="utf-8")
        print(f"Completed with {len(errors)} error(s). See {dirs['logs'] / 'errors.txt'}")
    else:
        print(f"Completed final placement for {len(summaries)} cities. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
