#!/usr/bin/env python3
"""
Distribution-aware final EVCS placement v3.

Adds demand-service map layers:
  - demand nodes
  - served demand nodes
  - unserved demand nodes
  - underserved/proxy-underserved demand nodes
  - selected final EVCS sites
  - top ranked candidates
  - simulator road graph
  - optional OSM basemap

Usage:
    python final_placement_v3.py \
      --dataset-index dataset_index.csv \
      --rankings outputs_sim2real/rankings/ranked_candidates_all_methods.csv \
      --output-dir outputs_final_placement_v3 \
      --config config/final_placement_v3.yaml \
      --city-ids omn_muscat omn_nizwa omn_salalah
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Basic file/schema utilities
# ---------------------------------------------------------------------

def find_file(data_dir: Path, patterns: Iterable[str]) -> Optional[Path]:
    files = list(data_dir.glob("*"))
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        for path in files:
            if regex.search(path.name):
                return path
    return None


def load_optional_csv(data_dir: Path, patterns: Iterable[str]) -> Optional[pd.DataFrame]:
    path = find_file(data_dir, patterns)
    if path is None:
        return None
    return pd.read_csv(path)


def detect_id_col(df: pd.DataFrame, preferred: str = "candidate_id") -> Optional[str]:
    if preferred in df.columns:
        return preferred

    candidates = ["site_id", "candidate_site_id", "id", "node_id", "osmid", "demand_id"]
    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        cl = c.lower()
        if ("candidate" in cl and "id" in cl) or ("site" in cl and "id" in cl) or ("demand" in cl and "id" in cl):
            return c

    return None


def detect_node_id_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["node_id", "osmid", "id", "node"]
    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        cl = c.lower()
        if ("node" in cl and "id" in cl) or cl == "osmid":
            return c
    return None


def detect_lat_lon(df: pd.DataFrame) -> Tuple[str, str]:
    lat_candidates = [
        "lat", "latitude", "y", "node_lat", "candidate_lat",
        "site_lat", "geometry_y", "demand_lat"
    ]
    lon_candidates = [
        "lon", "lng", "longitude", "x", "node_lon", "candidate_lon",
        "site_lon", "geometry_x", "demand_lon"
    ]

    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)

    if lat_col is None or lon_col is None:
        for c in df.columns:
            cl = c.lower()
            if lat_col is None and (cl == "lat" or cl.endswith("_lat") or "latitude" in cl):
                lat_col = c
            if lon_col is None and (cl in {"lon", "lng"} or cl.endswith("_lon") or "longitude" in cl):
                lon_col = c

    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not detect latitude/longitude columns in {list(df.columns)}")

    return lat_col, lon_col


def detect_edge_endpoint_cols(edges: pd.DataFrame) -> Optional[Tuple[str, str]]:
    endpoint_pairs = [
        ("u", "v"),
        ("source", "target"),
        ("from", "to"),
        ("from_node", "to_node"),
        ("from_node_id", "to_node_id"),
        ("start_node", "end_node"),
    ]

    for a, b in endpoint_pairs:
        if a in edges.columns and b in edges.columns:
            return a, b

    lower = {c.lower(): c for c in edges.columns}
    for a, b in endpoint_pairs:
        if a in lower and b in lower:
            return lower[a], lower[b]

    return None


def normalize_series(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)

    s = s.fillna(s.median())
    mn, mx = float(s.min()), float(s.max())

    if abs(mx - mn) < 1e-12:
        out = pd.Series(np.ones(len(s)) * 0.5, index=s.index)
    else:
        out = (s - mn) / (mx - mn)

    if not higher_is_better:
        out = 1.0 - out

    return out.clip(0, 1)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))

    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------
# Coordinate merge and city extent
# ---------------------------------------------------------------------

def load_candidate_sites_with_coords(data_dir: Path) -> pd.DataFrame:
    cand = load_optional_csv(
        data_dir,
        [
            r"candidate_sites.*\.csv$",
            r"candidates?.*\.csv$",
            r".*candidate.*sites.*\.csv$",
        ],
    )

    if cand is None:
        raise ValueError(f"Could not find candidate_sites*.csv in {data_dir}")

    id_col = detect_id_col(cand, "candidate_id")
    if id_col is None:
        raise ValueError(f"Could not detect candidate id column in candidate sites: {list(cand.columns)}")

    lat_col, lon_col = detect_lat_lon(cand)

    keep = [id_col, lat_col, lon_col]
    extra_cols = []
    for c in [
        "road_type", "highway", "zone_id", "activity_intensity",
        "equity_benefit", "equity_need_score", "available_space",
        "land_cost_proxy", "grid_feasibility", "land_feasibility"
    ]:
        if c in cand.columns:
            extra_cols.append(c)

    coords = cand[keep + extra_cols].copy()
    coords = coords.rename(columns={id_col: "candidate_id", lat_col: "lat", lon_col: "lon"})
    coords["candidate_id"] = coords["candidate_id"].astype(str)
    coords = coords.drop_duplicates("candidate_id", keep="first")
    return coords


def ensure_rankings_have_coordinates(rankings: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    df = rankings.copy()

    try:
        lat_col, lon_col = detect_lat_lon(df)
        if lat_col != "lat" or lon_col != "lon":
            df = df.rename(columns={lat_col: "lat", lon_col: "lon"})
        return df
    except Exception:
        pass

    if "candidate_id" not in df.columns:
        raise ValueError("Ranking table lacks lat/lon and candidate_id. Cannot merge coordinates.")

    coords = load_candidate_sites_with_coords(data_dir)
    df["candidate_id"] = df["candidate_id"].astype(str)
    merged = df.merge(coords, on="candidate_id", how="left", validate="many_to_one")

    missing = int(merged["lat"].isna().sum())
    if missing:
        sample_missing = merged.loc[merged["lat"].isna(), "candidate_id"].head(5).tolist()
        raise ValueError(
            f"Missing coordinates for {missing} ranked candidates after merge. "
            f"Sample missing candidate_id values: {sample_missing}."
        )

    return merged


def load_demand_points(data_dir: Path, underserved_quantile: float = 0.75) -> Optional[pd.DataFrame]:
    demand = load_optional_csv(data_dir, [r"demand.*\.csv$"])
    if demand is None or len(demand) == 0:
        return None

    lat_col, lon_col = detect_lat_lon(demand)
    out = demand.copy().rename(columns={lat_col: "lat", lon_col: "lon"})

    demand_id_col = detect_id_col(out, "demand_id")
    if demand_id_col is None:
        out["demand_id"] = [f"demand_{i}" for i in range(len(out))]
    elif demand_id_col != "demand_id":
        out = out.rename(columns={demand_id_col: "demand_id"})

    mass_col = None
    for c in ["demand_mass", "demand", "weight", "population", "activity_intensity"]:
        if c in out.columns:
            mass_col = c
            break

    if mass_col is None:
        out["demand_mass"] = 1.0
    elif mass_col != "demand_mass":
        out["demand_mass"] = pd.to_numeric(out[mass_col], errors="coerce").fillna(0.0)

    out["demand_mass"] = pd.to_numeric(out["demand_mass"], errors="coerce").fillna(0.0)

    # Detect explicit underserved flag.
    flag_cols = [
        "underserved", "underserved_flag", "is_underserved", "proxy_underserved",
        "low_accessibility_flag", "equity_priority_flag"
    ]

    flag_col = next((c for c in flag_cols if c in out.columns), None)

    if flag_col is not None:
        out["underserved_flag"] = pd.to_numeric(out[flag_col], errors="coerce").fillna(0).astype(float) > 0
        out["underserved_source"] = flag_col
    else:
        # Proxy fallback.
        score_col = None
        higher_is_underserved = True

        for c in ["equity_need_score", "underserved_score", "equity_benefit", "accessibility_need"]:
            if c in out.columns:
                score_col = c
                higher_is_underserved = True
                break

        if score_col is None:
            for c in ["reachability_10min", "accessibility_score", "accessibility_benefit"]:
                if c in out.columns:
                    score_col = c
                    higher_is_underserved = False
                    break

        if score_col is not None:
            score = pd.to_numeric(out[score_col], errors="coerce").fillna(pd.to_numeric(out[score_col], errors="coerce").median())
            if higher_is_underserved:
                threshold = score.quantile(float(underserved_quantile))
                out["underserved_flag"] = score >= threshold
            else:
                threshold = score.quantile(1.0 - float(underserved_quantile))
                out["underserved_flag"] = score <= threshold
            out["underserved_source"] = f"proxy:{score_col}"
        else:
            out["underserved_flag"] = False
            out["underserved_source"] = "not_available"

    out = out.dropna(subset=["lat", "lon"])
    return out


def load_city_extent(data_dir: Path, candidate_df: pd.DataFrame, demand_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    pieces = [candidate_df[["lat", "lon"]].copy()]

    if demand_df is not None and len(demand_df):
        pieces.append(demand_df[["lat", "lon"]].copy())

    nodes = load_optional_csv(data_dir, [r"graph_nodes.*\.csv$", r"^nodes.*\.csv$"])
    if nodes is not None and len(nodes):
        try:
            nlat, nlon = detect_lat_lon(nodes)
            pieces.append(nodes[[nlat, nlon]].rename(columns={nlat: "lat", nlon: "lon"}))
        except Exception:
            pass

    coords = pd.concat(pieces, ignore_index=True)
    coords = coords.dropna(subset=["lat", "lon"])
    return coords


def assign_grid_zones(df: pd.DataFrame, city_coords: pd.DataFrame, rows: int, cols: int) -> pd.DataFrame:
    out = df.copy()

    min_lat, max_lat = city_coords["lat"].min(), city_coords["lat"].max()
    min_lon, max_lon = city_coords["lon"].min(), city_coords["lon"].max()

    lat_pad = max((max_lat - min_lat) * 0.001, 1e-9)
    lon_pad = max((max_lon - min_lon) * 0.001, 1e-9)

    lat_bins = np.linspace(min_lat - lat_pad, max_lat + lat_pad, rows + 1)
    lon_bins = np.linspace(min_lon - lon_pad, max_lon + lon_pad, cols + 1)

    out["zone_row"] = np.clip(np.digitize(out["lat"], lat_bins) - 1, 0, rows - 1)
    out["zone_col"] = np.clip(np.digitize(out["lon"], lon_bins) - 1, 0, cols - 1)
    out["zone_id"] = "z" + out["zone_row"].astype(str) + "_" + out["zone_col"].astype(str)
    return out


# ---------------------------------------------------------------------
# Demand service classification
# ---------------------------------------------------------------------

def classify_demand_service(demand: Optional[pd.DataFrame], selected: pd.DataFrame, service_radius_km: float) -> Optional[pd.DataFrame]:
    if demand is None or len(demand) == 0:
        return None

    out = demand.copy()

    if selected is None or len(selected) == 0:
        out["nearest_selected_distance_km"] = np.nan
        out["served_by_final_site"] = False
        out["nearest_selected_candidate_id"] = None
        return out

    nearest_dist = []
    nearest_id = []

    selected_records = selected.to_dict("records")

    for _, d in out.iterrows():
        best_dist = float("inf")
        best_id = None
        for s in selected_records:
            dist = haversine_km(d["lat"], d["lon"], s["lat"], s["lon"])
            if dist < best_dist:
                best_dist = dist
                best_id = s["candidate_id"]
        nearest_dist.append(best_dist)
        nearest_id.append(best_id)

    out["nearest_selected_distance_km"] = nearest_dist
    out["nearest_selected_candidate_id"] = nearest_id
    out["served_by_final_site"] = out["nearest_selected_distance_km"] <= float(service_radius_km)

    return out


def summarize_demand_service(demand_service: Optional[pd.DataFrame]) -> dict:
    if demand_service is None or len(demand_service) == 0:
        return {
            "demand_node_count": 0,
            "served_demand_node_count": 0,
            "unserved_demand_node_count": 0,
            "underserved_demand_node_count": 0,
            "served_underserved_node_count": 0,
            "total_demand_mass": np.nan,
            "served_demand_mass": np.nan,
            "served_demand_mass_rate": np.nan,
            "underserved_demand_mass": np.nan,
            "served_underserved_demand_mass": np.nan,
            "served_underserved_demand_mass_rate": np.nan,
        }

    total_mass = float(demand_service["demand_mass"].sum())
    served = demand_service[demand_service["served_by_final_site"]]
    underserved = demand_service[demand_service["underserved_flag"]]
    served_underserved = demand_service[demand_service["served_by_final_site"] & demand_service["underserved_flag"]]

    served_mass = float(served["demand_mass"].sum())
    underserved_mass = float(underserved["demand_mass"].sum())
    served_underserved_mass = float(served_underserved["demand_mass"].sum())

    return {
        "demand_node_count": int(len(demand_service)),
        "served_demand_node_count": int(len(served)),
        "unserved_demand_node_count": int((~demand_service["served_by_final_site"]).sum()),
        "underserved_demand_node_count": int(len(underserved)),
        "served_underserved_node_count": int(len(served_underserved)),
        "total_demand_mass": total_mass,
        "served_demand_mass": served_mass,
        "served_demand_mass_rate": served_mass / total_mass if total_mass > 0 else np.nan,
        "underserved_demand_mass": underserved_mass,
        "served_underserved_demand_mass": served_underserved_mass,
        "served_underserved_demand_mass_rate": served_underserved_mass / underserved_mass if underserved_mass > 0 else np.nan,
    }


# ---------------------------------------------------------------------
# Local demand, road graph, and map helpers
# ---------------------------------------------------------------------

def add_local_demand_score(candidates: pd.DataFrame, data_dir: Path, radius_km: float = 2.0) -> pd.Series:
    demand = load_demand_points(data_dir)
    if demand is None or len(demand) == 0:
        return pd.Series(np.zeros(len(candidates)), index=candidates.index)

    scores = []
    for _, c in candidates.iterrows():
        total = 0.0
        for _, q in demand.iterrows():
            dist = haversine_km(c["lat"], c["lon"], q["lat"], q["lon"])
            if dist <= radius_km:
                total += float(q["demand_mass"]) * (1 - dist / radius_km)
        scores.append(total)

    return normalize_series(pd.Series(scores, index=candidates.index))


def load_road_graph_for_plot(data_dir: Path, max_edges: int = 15000) -> Optional[pd.DataFrame]:
    nodes = load_optional_csv(data_dir, [r"graph_nodes.*\.csv$", r"^nodes.*\.csv$"])
    edges = load_optional_csv(data_dir, [r"graph_edges.*\.csv$", r"^edges.*\.csv$"])

    if nodes is None or edges is None or len(nodes) == 0 or len(edges) == 0:
        return None

    try:
        node_id_col = detect_node_id_col(nodes)
        nlat, nlon = detect_lat_lon(nodes)
        endpoints = detect_edge_endpoint_cols(edges)
    except Exception:
        return None

    if node_id_col is None or endpoints is None:
        return None

    u_col, v_col = endpoints

    node_lookup = (
        nodes[[node_id_col, nlat, nlon]]
        .dropna()
        .rename(columns={node_id_col: "node_id", nlat: "lat", nlon: "lon"})
        .copy()
    )
    node_lookup["node_id"] = node_lookup["node_id"].astype(str)

    e = edges[[u_col, v_col]].dropna().copy()
    e = e.rename(columns={u_col: "u", v_col: "v"})
    e["u"] = e["u"].astype(str)
    e["v"] = e["v"].astype(str)

    if len(e) > max_edges:
        e = e.sample(n=max_edges, random_state=42)

    e = e.merge(
        node_lookup.rename(columns={"node_id": "u", "lat": "lat1", "lon": "lon1"}),
        on="u",
        how="left",
    )
    e = e.merge(
        node_lookup.rename(columns={"node_id": "v", "lat": "lat2", "lon": "lon2"}),
        on="v",
        how="left",
    )

    e = e.dropna(subset=["lat1", "lon1", "lat2", "lon2"])

    if len(e) == 0:
        return None

    return e[["lon1", "lat1", "lon2", "lat2"]]


def apply_map_extent(ax, coords: pd.DataFrame, padding_fraction: float = 0.04) -> None:
    if coords is None or len(coords) == 0:
        return

    min_lon, max_lon = coords["lon"].min(), coords["lon"].max()
    min_lat, max_lat = coords["lat"].min(), coords["lat"].max()

    lon_pad = max((max_lon - min_lon) * padding_fraction, 0.001)
    lat_pad = max((max_lat - min_lat) * padding_fraction, 0.001)

    ax.set_xlim(min_lon - lon_pad, max_lon + lon_pad)
    ax.set_ylim(min_lat - lat_pad, max_lat + lat_pad)


def add_optional_basemap(ax, cfg: dict) -> None:
    plot_cfg = cfg["final_placement"].get("plots", {})

    if not plot_cfg.get("use_basemap", False):
        return

    try:
        import contextily as ctx

        provider_name = plot_cfg.get("basemap_provider", "OpenStreetMap.Mapnik")
        source = ctx.providers

        for part in provider_name.split("."):
            source = getattr(source, part)

        ctx.add_basemap(
            ax,
            crs="EPSG:4326",
            source=source,
            attribution_size=6,
            alpha=0.85,
        )

    except Exception as exc:
        print(f"  Basemap skipped: {exc}")


# ---------------------------------------------------------------------
# Candidate preparation and selection
# ---------------------------------------------------------------------

def prepare_candidate_pool(city_id: str, data_dir: Path, rankings: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fp_cfg = cfg["final_placement"]

    if "city_id" not in rankings.columns or "method_id" not in rankings.columns:
        raise ValueError("Ranking file must include city_id and method_id columns.")

    city_rankings = rankings[
        (rankings["city_id"] == city_id) &
        (rankings.get("data_mode", "real") == "real")
    ].copy()

    if len(city_rankings) == 0:
        raise ValueError(f"No ranking rows found for city {city_id}")

    method = None
    for m in fp_cfg["preferred_methods"]:
        if m in set(city_rankings["method_id"]):
            method = m
            break

    if method is None:
        available = sorted(city_rankings["method_id"].dropna().unique())
        raise ValueError(f"No preferred method found for {city_id}. Available methods: {available}")

    city_rankings = city_rankings[city_rankings["method_id"] == method].copy()

    if "shortlist_size" in city_rankings.columns and city_rankings["shortlist_size"].notna().any():
        chosen = None
        for s in fp_cfg.get("preferred_shortlist_sizes", [60, 40, 20]):
            subset = city_rankings[city_rankings["shortlist_size"].fillna(-1).astype(float) == float(s)]
            if len(subset):
                city_rankings = subset.copy()
                chosen = s
                break

        if chosen is None:
            max_s = city_rankings["shortlist_size"].dropna().max()
            city_rankings = city_rankings[city_rankings["shortlist_size"] == max_s].copy()

    city_rankings = city_rankings.sort_values("rank").drop_duplicates("candidate_id", keep="first").copy()
    city_rankings = ensure_rankings_have_coordinates(city_rankings, data_dir)

    candidate_pool = fp_cfg.get("candidate_pool", "all")
    if candidate_pool != "all":
        city_rankings = city_rankings.sort_values("rank").head(int(candidate_pool)).copy()

    w = fp_cfg["weights"]

    if "rank_score" in city_rankings.columns:
        city_rankings["rank_score_norm"] = normalize_series(city_rankings["rank_score"])
    else:
        city_rankings["rank_score_norm"] = normalize_series(-city_rankings["rank"])

    radius = float(fp_cfg.get("local_demand_radius_km", 2.0))
    city_rankings["local_demand_norm"] = add_local_demand_score(city_rankings, data_dir, radius_km=radius)

    if "suitability_score" in city_rankings.columns:
        city_rankings["suitability_norm"] = normalize_series(city_rankings["suitability_score"])
    else:
        city_rankings["suitability_norm"] = 0.0

    cost_col = None
    for c in ["cost_proxy_model", "cost_proxy", "land_cost_proxy"]:
        if c in city_rankings.columns:
            cost_col = c
            break

    if cost_col:
        city_rankings["cost_efficiency_norm"] = normalize_series(city_rankings[cost_col], higher_is_better=False)
    else:
        city_rankings["cost_efficiency_norm"] = 0.0

    equity_col = None
    for c in ["equity_component", "equity_benefit", "equity_need_score", "underserved_score"]:
        if c in city_rankings.columns:
            equity_col = c
            break

    if equity_col:
        city_rankings["equity_norm"] = normalize_series(city_rankings[equity_col])
    else:
        city_rankings["equity_norm"] = 0.0

    city_rankings["final_rank_score_v2"] = (
        w.get("rank_score", 0.40) * city_rankings["rank_score_norm"] +
        w.get("local_demand_score", 0.25) * city_rankings["local_demand_norm"] +
        w.get("suitability_score", 0.15) * city_rankings["suitability_norm"] +
        w.get("cost_efficiency", 0.10) * city_rankings["cost_efficiency_norm"] +
        w.get("underserved_or_equity", 0.10) * city_rankings["equity_norm"]
    )

    demand = load_demand_points(data_dir, underserved_quantile=float(fp_cfg.get("underserved_quantile", 0.75)))
    city_coords = load_city_extent(data_dir, city_rankings, demand_df=demand)

    city_rankings = assign_grid_zones(
        city_rankings,
        city_coords,
        rows=int(fp_cfg["grid_rows"]),
        cols=int(fp_cfg["grid_cols"]),
    )

    city_rankings["chosen_ranking_method"] = method
    return city_rankings


def can_add_candidate(row: pd.Series, selected: List[pd.Series], min_sep_km: float) -> bool:
    for s in selected:
        dist = haversine_km(row["lat"], row["lon"], s["lat"], s["lon"])
        if dist < min_sep_km:
            return False
    return True


def select_distribution_aware(candidates: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, dict]:
    fp_cfg = cfg["final_placement"]

    k = int(fp_cfg["k"])
    max_per_zone = int(fp_cfg.get("max_per_zone", 1))
    min_active_zones = int(math.ceil(k * float(fp_cfg.get("min_active_zones_fraction_of_k", 0.7))))

    relaxations = [float(x) for x in fp_cfg.get("min_separation_relaxation", [fp_cfg.get("min_separation_km", 1.5)])]

    zone_priority = (
        candidates
        .groupby("zone_id")
        .agg(
            zone_score=("final_rank_score_v2", "max"),
            zone_mean=("final_rank_score_v2", "mean"),
            zone_count=("candidate_id", "count"),
        )
        .reset_index()
        .sort_values(["zone_score", "zone_mean", "zone_count"], ascending=[False, False, False])
    )

    best_solution = pd.DataFrame()
    best_meta = {"active_zones": 0, "selected_count": 0}

    for min_sep in relaxations:
        selected: List[pd.Series] = []
        zone_counts: Dict[str, int] = {}

        # Phase 1: choose one candidate from as many high-priority zones as possible.
        for _, zr in zone_priority.iterrows():
            if len(selected) >= min(k, min_active_zones):
                break

            z = zr["zone_id"]
            zcands = candidates[candidates["zone_id"] == z].sort_values("final_rank_score_v2", ascending=False)

            for _, cand in zcands.iterrows():
                if can_add_candidate(cand, selected, min_sep):
                    selected.append(cand)
                    zone_counts[z] = zone_counts.get(z, 0) + 1
                    break

        selected_ids = {s["candidate_id"] for s in selected}

        # Phase 2: fill remaining stations while respecting max_per_zone.
        for _, cand in candidates.sort_values("final_rank_score_v2", ascending=False).iterrows():
            if len(selected) >= k:
                break
            if cand["candidate_id"] in selected_ids:
                continue
            if zone_counts.get(cand["zone_id"], 0) >= max_per_zone:
                continue
            if can_add_candidate(cand, selected, min_sep):
                selected.append(cand)
                selected_ids.add(cand["candidate_id"])
                zone_counts[cand["zone_id"]] = zone_counts.get(cand["zone_id"], 0) + 1

        selected_ids = {s["candidate_id"] for s in selected}

        # Phase 3: if still short, relax zone cap but not separation.
        for _, cand in candidates.sort_values("final_rank_score_v2", ascending=False).iterrows():
            if len(selected) >= k:
                break
            if cand["candidate_id"] in selected_ids:
                continue
            if can_add_candidate(cand, selected, min_sep):
                selected.append(cand)
                selected_ids.add(cand["candidate_id"])
                zone_counts[cand["zone_id"]] = zone_counts.get(cand["zone_id"], 0) + 1

        selected_df = pd.DataFrame(selected)
        active_zones = selected_df["zone_id"].nunique() if len(selected_df) else 0

        meta = {
            "requested_k": k,
            "selected_count": int(len(selected_df)),
            "min_separation_used_km": float(min_sep),
            "active_zones": int(active_zones),
            "target_active_zones": int(min_active_zones),
            "max_per_zone": int(max_per_zone),
            "zone_constraint_met": bool(active_zones >= min_active_zones),
            "separation_constraint_relaxed": bool(min_sep < float(fp_cfg.get("min_separation_km", min_sep))),
        }

        if len(selected_df) == k and active_zones >= min_active_zones:
            best_solution, best_meta = selected_df, meta
            break

        if (len(selected_df), active_zones) > (len(best_solution), best_meta.get("active_zones", 0)):
            best_solution, best_meta = selected_df, meta

    best_solution = best_solution.copy()

    if len(best_solution):
        best_solution["final_selected_v2"] = 1
        best_solution["final_rank_v2"] = range(1, len(best_solution) + 1)

    return best_solution, best_meta


def pairwise_selected_stats(selected: pd.DataFrame) -> dict:
    if len(selected) < 2:
        return {
            "min_pairwise_distance_km": np.nan,
            "mean_pairwise_distance_km": np.nan,
            "max_pairwise_distance_km": np.nan,
        }

    rows = selected.to_dict("records")
    dists = []

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            dists.append(haversine_km(rows[i]["lat"], rows[i]["lon"], rows[j]["lat"], rows[j]["lon"]))

    return {
        "min_pairwise_distance_km": float(np.min(dists)),
        "mean_pairwise_distance_km": float(np.mean(dists)),
        "max_pairwise_distance_km": float(np.max(dists)),
    }


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def sample_for_plot(df: pd.DataFrame, max_points: int, seed: int = 42) -> pd.DataFrame:
    if df is None or len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=seed)


def plot_city_placement(
    city_id: str,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    demand_service: Optional[pd.DataFrame],
    output_dir: Path,
    cfg: dict,
    data_dir: Optional[Path] = None,
) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_cfg = cfg["final_placement"].get("plots", {})
    dpi = int(plot_cfg.get("dpi", 300))
    padding = float(plot_cfg.get("map_padding_fraction", 0.04))

    fig, ax = plt.subplots(figsize=(10, 8.5))

    # City extent includes demand when available.
    if data_dir is not None:
        try:
            city_coords = load_city_extent(data_dir, candidates, demand_df=demand_service)
        except Exception:
            city_coords = candidates[["lat", "lon"]].copy()
    else:
        city_coords = candidates[["lat", "lon"]].copy()

    apply_map_extent(ax, city_coords, padding_fraction=padding)

    add_optional_basemap(ax, cfg)

    # Road graph.
    if data_dir is not None and plot_cfg.get("draw_road_graph", True):
        road_segments = load_road_graph_for_plot(
            data_dir=data_dir,
            max_edges=int(plot_cfg.get("max_edges_on_map", 15000)),
        )

        if road_segments is not None and len(road_segments) > 0:
            for _, e in road_segments.iterrows():
                ax.plot(
                    [e["lon1"], e["lon2"]],
                    [e["lat1"], e["lat2"]],
                    linewidth=float(plot_cfg.get("road_linewidth", 0.30)),
                    alpha=float(plot_cfg.get("road_alpha", 0.20)),
                    zorder=1,
                )

    # Demand nodes.
    if demand_service is not None and plot_cfg.get("show_demand_nodes", True):
        max_demand = int(plot_cfg.get("max_demand_points_on_map", 6000))
        dplot = sample_for_plot(demand_service, max_demand)

        served = dplot[dplot["served_by_final_site"]]
        unserved = dplot[~dplot["served_by_final_site"]]
        underserved = dplot[dplot["underserved_flag"]]

        if plot_cfg.get("show_unserved_demand", True) and len(unserved):
            ax.scatter(
                unserved["lon"],
                unserved["lat"],
                s=float(plot_cfg.get("demand_marker_size", 10)),
                alpha=float(plot_cfg.get("demand_alpha", 0.35)),
                marker="x",
                label="unserved demand nodes",
                zorder=2,
            )

        if plot_cfg.get("show_served_demand", True) and len(served):
            ax.scatter(
                served["lon"],
                served["lat"],
                s=float(plot_cfg.get("demand_marker_size", 10)),
                alpha=float(plot_cfg.get("demand_alpha", 0.35)),
                marker=".",
                label="served demand nodes",
                zorder=2,
            )

        if plot_cfg.get("show_underserved_demand", True) and len(underserved):
            ax.scatter(
                underserved["lon"],
                underserved["lat"],
                s=float(plot_cfg.get("underserved_marker_size", 22)),
                alpha=float(plot_cfg.get("underserved_alpha", 0.75)),
                marker="o",
                facecolors="none",
                label="proxy-underserved demand nodes",
                zorder=3,
            )

    # Candidate sites.
    if plot_cfg.get("show_all_candidates", True):
        ax.scatter(
            candidates["lon"],
            candidates["lat"],
            s=16,
            alpha=0.30,
            label="candidate sites",
            zorder=4,
        )

    # Top ranked candidates.
    top_n = int(plot_cfg.get("show_top_ranked_candidates", 40))
    top = candidates.sort_values("final_rank_score_v2", ascending=False).head(top_n)

    ax.scatter(
        top["lon"],
        top["lat"],
        s=34,
        alpha=0.60,
        marker="^",
        label=f"top {top_n} ranked candidates",
        zorder=5,
    )

    # Final selected sites.
    if len(selected):
        ax.scatter(
            selected["lon"],
            selected["lat"],
            s=160,
            marker="*",
            edgecolor="black",
            linewidth=0.9,
            label="final selected EVCS sites",
            zorder=6,
        )

        for _, r in selected.iterrows():
            ax.text(
                r["lon"],
                r["lat"],
                str(int(r["final_rank_v2"])),
                fontsize=8,
                ha="center",
                va="center",
                zorder=7,
            )

    ax.set_title(f"Final EVCS placement with demand-service map: {city_id}", fontsize=13)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.20)
    ax.legend(loc="best", fontsize=8, frameon=True)

    fig.tight_layout()

    for fmt in plot_cfg.get("figure_format", ["png"]):
        fig.savefig(fig_dir / f"final_placement_v3_{city_id}.{fmt}", dpi=dpi, bbox_inches="tight")

    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--city-ids", nargs="*", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    output_dir = args.output_dir
    (output_dir / "selected").mkdir(parents=True, exist_ok=True)
    (output_dir / "rankings").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "demand_service").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(args.dataset_index)
    rankings = pd.read_csv(args.rankings)

    real_rows = index[index["data_mode"].str.lower() == "real"].copy()

    if args.city_ids:
        real_rows = real_rows[real_rows["city_id"].isin(args.city_ids)].copy()

    summaries = []
    fp_cfg = cfg["final_placement"]

    for _, row in real_rows.iterrows():
        city_id = row["city_id"]
        data_dir = Path(row["data_dir"])

        print(f"Processing {city_id} from {data_dir}")

        try:
            candidates = prepare_candidate_pool(city_id, data_dir, rankings, cfg)
            selected, meta = select_distribution_aware(candidates, cfg)

            demand = load_demand_points(
                data_dir,
                underserved_quantile=float(fp_cfg.get("underserved_quantile", 0.75)),
            )

            demand_service = classify_demand_service(
                demand=demand,
                selected=selected,
                service_radius_km=float(fp_cfg.get("demand_service_radius_km", 5.0)),
            )

            stats = pairwise_selected_stats(selected)
            demand_stats = summarize_demand_service(demand_service)

            meta.update(stats)
            meta.update(demand_stats)

            meta.update(
                {
                    "city_id": city_id,
                    "country": row.get("country", ""),
                    "data_mode": "real",
                    "candidate_pool_size": int(len(candidates)),
                    "selected_zone_counts": json.dumps(
                        selected["zone_id"].value_counts().to_dict() if len(selected) else {}
                    ),
                    "ranking_method_used": candidates["chosen_ranking_method"].iloc[0] if len(candidates) else "",
                    "demand_service_radius_km": float(fp_cfg.get("demand_service_radius_km", 5.0)),
                    "underserved_source": (
                        demand_service["underserved_source"].iloc[0]
                        if demand_service is not None and "underserved_source" in demand_service.columns and len(demand_service)
                        else "not_available"
                    ),
                }
            )

            candidates_out = candidates.sort_values("final_rank_score_v2", ascending=False).copy()
            candidates_out["final_rank_v2"] = range(1, len(candidates_out) + 1)
            candidates_out["selected_in_final_v2"] = candidates_out["candidate_id"].isin(
                set(selected["candidate_id"])
            ).astype(int)

            candidates_out.to_csv(output_dir / "rankings" / f"final_ranked_sites_v3_{city_id}.csv", index=False)
            selected.to_csv(output_dir / "selected" / f"final_selected_sites_v3_{city_id}.csv", index=False)

            if demand_service is not None:
                demand_service.to_csv(output_dir / "demand_service" / f"demand_service_v3_{city_id}.csv", index=False)

            plot_city_placement(
                city_id=city_id,
                candidates=candidates_out,
                selected=selected,
                demand_service=demand_service,
                output_dir=output_dir,
                cfg=cfg,
                data_dir=data_dir,
            )

            summaries.append(meta)

            print(
                f"  selected={meta['selected_count']} "
                f"active_zones={meta['active_zones']} "
                f"min_pairwise={meta.get('min_pairwise_distance_km', np.nan):.3f} km "
                f"served_mass_rate={meta.get('served_demand_mass_rate', np.nan):.3f} "
                f"underserved_served_rate={meta.get('served_underserved_demand_mass_rate', np.nan):.3f}"
            )

        except Exception as exc:
            summaries.append(
                {
                    "city_id": city_id,
                    "error": str(exc),
                    "selected_count": 0,
                }
            )
            print(f"ERROR for {city_id}: {exc}")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "tables" / "final_placement_v3_summary.csv", index=False)

    print(f"Done. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
