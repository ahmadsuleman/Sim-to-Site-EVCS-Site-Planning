#!/usr/bin/env python3
"""
Plot actual EVCS placements from existing analysis outputs.

This script does NOT run a new placement algorithm.

It reads:
  - dataset_index.csv
  - T8_final_selected_sites.csv or outputs/tables/T8_final_selected_sites.csv
  - candidate_sites*.csv from each city folder
  - demand_points*.csv from each city folder
  - graph_nodes*.csv and graph_edges*.csv for road graph plotting

It outputs:
  - actual selected-site maps with demand-service layers
  - demand-service CSV per city
  - actual placement summary CSV

Usage:
    python plot_actual_placements.py \
      --dataset-index dataset_index.csv \
      --selected-sites T8_final_selected_sites.csv \
      --output-dir outputs_actual_placement_maps \
      --config config/actual_placement_map.yaml \
      --city-ids omn_muscat omn_nizwa omn_salalah

Important:
    Demand-service classification on the map uses geographic distance to selected sites.
    If exact MILP assignment files exist, they should be used for formal service metrics in the paper.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt


# -----------------------------
# Generic helpers
# -----------------------------

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


def detect_lat_lon(df: pd.DataFrame) -> Tuple[str, str]:
    lat_candidates = [
        "lat", "latitude", "y", "node_lat", "candidate_lat", "site_lat", "demand_lat", "geometry_y"
    ]
    lon_candidates = [
        "lon", "lng", "longitude", "x", "node_lon", "candidate_lon", "site_lon", "demand_lon", "geometry_x"
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


def detect_id_col(df: pd.DataFrame, preferred: str) -> Optional[str]:
    if preferred in df.columns:
        return preferred

    candidates = ["candidate_id", "site_id", "candidate_site_id", "demand_id", "node_id", "id", "osmid"]
    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        cl = c.lower()
        if "id" in cl and any(t in cl for t in ["candidate", "site", "demand", "node"]):
            return c

    return None


def detect_node_id_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["node_id", "osmid", "id", "node"]:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = c.lower()
        if ("node" in cl and "id" in cl) or cl == "osmid":
            return c
    return None


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


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def apply_map_extent(ax, coords: pd.DataFrame, padding_fraction: float = 0.04) -> None:
    coords = coords.dropna(subset=["lat", "lon"])
    if len(coords) == 0:
        return

    min_lon, max_lon = coords["lon"].min(), coords["lon"].max()
    min_lat, max_lat = coords["lat"].min(), coords["lat"].max()

    lon_pad = max((max_lon - min_lon) * padding_fraction, 0.001)
    lat_pad = max((max_lat - min_lat) * padding_fraction, 0.001)

    ax.set_xlim(min_lon - lon_pad, max_lon + lon_pad)
    ax.set_ylim(min_lat - lat_pad, max_lat + lat_pad)


def add_optional_basemap(ax, cfg: dict) -> None:
    plot_cfg = cfg["actual_placement_map"].get("plots", {})
    if not plot_cfg.get("use_basemap", False):
        return

    try:
        import contextily as ctx

        provider_name = plot_cfg.get("basemap_provider", "OpenStreetMap.Mapnik")
        source = ctx.providers
        for part in provider_name.split("."):
            source = getattr(source, part)

        ctx.add_basemap(ax, crs="EPSG:4326", source=source, attribution_size=6, alpha=0.85)
    except Exception as exc:
        print(f"  Basemap skipped: {exc}")


# -----------------------------
# Data loading
# -----------------------------

def load_candidates(data_dir: Path) -> Optional[pd.DataFrame]:
    cand = load_optional_csv(
        data_dir,
        [
            r"candidate_sites.*\.csv$",
            r"candidates?.*\.csv$",
            r".*candidate.*sites.*\.csv$",
        ],
    )
    if cand is None:
        return None

    id_col = detect_id_col(cand, "candidate_id")
    lat_col, lon_col = detect_lat_lon(cand)

    cand = cand.copy().rename(columns={id_col: "candidate_id", lat_col: "lat", lon_col: "lon"})
    cand["candidate_id"] = cand["candidate_id"].astype(str)
    return cand


def load_demand_points(data_dir: Path, underserved_quantile: float = 0.75) -> Optional[pd.DataFrame]:
    demand = load_optional_csv(data_dir, [r"demand.*\.csv$"])
    if demand is None or len(demand) == 0:
        return None

    lat_col, lon_col = detect_lat_lon(demand)
    out = demand.copy().rename(columns={lat_col: "lat", lon_col: "lon"})

    did_col = detect_id_col(out, "demand_id")
    if did_col is None:
        out["demand_id"] = [f"demand_{i}" for i in range(len(out))]
    elif did_col != "demand_id":
        out = out.rename(columns={did_col: "demand_id"})

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

    # Underserved detection: explicit flag first.
    flag_cols = [
        "underserved", "underserved_flag", "is_underserved", "proxy_underserved",
        "low_accessibility_flag", "equity_priority_flag"
    ]
    flag_col = next((c for c in flag_cols if c in out.columns), None)

    if flag_col:
        out["underserved_flag"] = pd.to_numeric(out[flag_col], errors="coerce").fillna(0).astype(float) > 0
        out["underserved_source"] = flag_col
    else:
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

        if score_col:
            score = pd.to_numeric(out[score_col], errors="coerce")
            score = score.fillna(score.median())
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

    return out.dropna(subset=["lat", "lon"]).copy()


def load_road_graph_for_plot(data_dir: Path, max_edges: int) -> Optional[pd.DataFrame]:
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

    node_lookup = nodes[[node_id_col, nlat, nlon]].dropna().rename(
        columns={node_id_col: "node_id", nlat: "lat", nlon: "lon"}
    )
    node_lookup["node_id"] = node_lookup["node_id"].astype(str)

    e = edges[[u_col, v_col]].dropna().rename(columns={u_col: "u", v_col: "v"})
    e["u"] = e["u"].astype(str)
    e["v"] = e["v"].astype(str)

    if len(e) > max_edges:
        e = e.sample(n=max_edges, random_state=42)

    e = e.merge(node_lookup.rename(columns={"node_id": "u", "lat": "lat1", "lon": "lon1"}), on="u", how="left")
    e = e.merge(node_lookup.rename(columns={"node_id": "v", "lat": "lat2", "lon": "lon2"}), on="v", how="left")
    e = e.dropna(subset=["lat1", "lon1", "lat2", "lon2"])

    if len(e) == 0:
        return None
    return e[["lon1", "lat1", "lon2", "lat2"]]


# -----------------------------
# Actual selected-site extraction
# -----------------------------

def choose_actual_selected_sites(
    selected_all: pd.DataFrame,
    city_id: str,
    cfg: dict,
) -> Tuple[pd.DataFrame, dict]:
    """Choose already-computed selected sites from T8, without re-optimization."""
    ap_cfg = cfg["actual_placement_map"]

    df = selected_all[(selected_all["city_id"] == city_id) & (selected_all.get("data_mode", "real") == "real")].copy()
    if len(df) == 0:
        raise ValueError(f"No selected sites found in selected-sites table for {city_id}")

    chosen_method = None
    for method in ap_cfg["preferred_methods"]:
        if method in set(df["method_id"]):
            chosen_method = method
            break
    if chosen_method is None:
        available = sorted(df["method_id"].dropna().unique())
        raise ValueError(f"No preferred method available for {city_id}. Available: {available}")

    df = df[df["method_id"] == chosen_method].copy()

    chosen_shortlist = None
    if "shortlist_size" in df.columns and df["shortlist_size"].notna().any():
        for s in ap_cfg.get("preferred_shortlist_sizes", [60, 40, 20]):
            subset = df[df["shortlist_size"].fillna(-1).astype(float) == float(s)]
            if len(subset):
                chosen_shortlist = s
                df = subset.copy()
                break

        if chosen_shortlist is None:
            max_s = df["shortlist_size"].dropna().max()
            df = df[df["shortlist_size"] == max_s].copy()
            chosen_shortlist = max_s

    lat_col, lon_col = detect_lat_lon(df)
    if lat_col != "lat" or lon_col != "lon":
        df = df.rename(columns={lat_col: "lat", lon_col: "lon"})

    if "candidate_id" not in df.columns:
        raise ValueError("Selected-sites table must include candidate_id")

    # Deduplicate if needed.
    df["candidate_id"] = df["candidate_id"].astype(str)
    df = df.drop_duplicates("candidate_id", keep="first").copy()

    # Prefer readable order: suitability or candidate score if rank unavailable.
    sort_col = None
    for c in ["final_rank_v2", "rank", "suitability_score", "candidate_score_for_inclusion", "demand_capture"]:
        if c in df.columns:
            sort_col = c
            break

    if sort_col:
        ascending = sort_col in {"final_rank_v2", "rank"}
        df = df.sort_values(sort_col, ascending=ascending).copy()

    df["actual_selected_rank"] = range(1, len(df) + 1)

    meta = {
        "method_id": chosen_method,
        "shortlist_size": chosen_shortlist,
        "selected_count": int(len(df)),
    }

    return df, meta


# -----------------------------
# Demand service classification
# -----------------------------

def classify_demand_service(demand: Optional[pd.DataFrame], selected: pd.DataFrame, service_radius_km: float) -> Optional[pd.DataFrame]:
    if demand is None or len(demand) == 0:
        return None

    out = demand.copy()

    if selected is None or len(selected) == 0:
        out["nearest_selected_distance_km"] = np.nan
        out["served_by_actual_site"] = False
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
    out["served_by_actual_site"] = out["nearest_selected_distance_km"] <= float(service_radius_km)

    return out


def summarize_actual_placement(selected: pd.DataFrame, demand_service: Optional[pd.DataFrame]) -> dict:
    # Pairwise selected distances.
    rows = selected.to_dict("records")
    dists = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            dists.append(haversine_km(rows[i]["lat"], rows[i]["lon"], rows[j]["lat"], rows[j]["lon"]))

    out = {
        "min_pairwise_distance_km": float(np.min(dists)) if dists else np.nan,
        "mean_pairwise_distance_km": float(np.mean(dists)) if dists else np.nan,
        "max_pairwise_distance_km": float(np.max(dists)) if dists else np.nan,
    }

    if demand_service is None or len(demand_service) == 0:
        out.update({
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
            "underserved_source": "not_available",
        })
        return out

    total_mass = float(demand_service["demand_mass"].sum())
    served = demand_service[demand_service["served_by_actual_site"]]
    unserved = demand_service[~demand_service["served_by_actual_site"]]
    underserved = demand_service[demand_service["underserved_flag"]]
    served_underserved = demand_service[demand_service["served_by_actual_site"] & demand_service["underserved_flag"]]

    served_mass = float(served["demand_mass"].sum())
    underserved_mass = float(underserved["demand_mass"].sum())
    served_underserved_mass = float(served_underserved["demand_mass"].sum())

    out.update({
        "demand_node_count": int(len(demand_service)),
        "served_demand_node_count": int(len(served)),
        "unserved_demand_node_count": int(len(unserved)),
        "underserved_demand_node_count": int(len(underserved)),
        "served_underserved_node_count": int(len(served_underserved)),
        "total_demand_mass": total_mass,
        "served_demand_mass": served_mass,
        "served_demand_mass_rate": served_mass / total_mass if total_mass > 0 else np.nan,
        "underserved_demand_mass": underserved_mass,
        "served_underserved_demand_mass": served_underserved_mass,
        "served_underserved_demand_mass_rate": served_underserved_mass / underserved_mass if underserved_mass > 0 else np.nan,
        "underserved_source": (
            demand_service["underserved_source"].iloc[0]
            if "underserved_source" in demand_service.columns and len(demand_service)
            else "not_available"
        ),
    })

    return out


# -----------------------------
# Plotting
# -----------------------------

def sample_for_plot(df: pd.DataFrame, max_points: int, seed: int = 42) -> pd.DataFrame:
    if df is None or len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=seed)


def plot_actual_city_map(
    city_id: str,
    data_dir: Path,
    selected: pd.DataFrame,
    candidates: Optional[pd.DataFrame],
    demand_service: Optional[pd.DataFrame],
    output_dir: Path,
    cfg: dict,
) -> None:
    plot_cfg = cfg["actual_placement_map"]["plots"]
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8.5))

    # Build city extent from all available geographic layers.
    extent_parts = [selected[["lat", "lon"]]]
    if candidates is not None:
        extent_parts.append(candidates[["lat", "lon"]])
    if demand_service is not None:
        extent_parts.append(demand_service[["lat", "lon"]])
    coords = pd.concat(extent_parts, ignore_index=True)

    apply_map_extent(ax, coords, padding_fraction=float(plot_cfg.get("map_padding_fraction", 0.04)))
    add_optional_basemap(ax, cfg)

    # Road graph.
    if plot_cfg.get("draw_road_graph", True):
        road_segments = load_road_graph_for_plot(data_dir, int(plot_cfg.get("max_edges_on_map", 15000)))
        if road_segments is not None and len(road_segments):
            for _, e in road_segments.iterrows():
                ax.plot(
                    [e["lon1"], e["lon2"]],
                    [e["lat1"], e["lat2"]],
                    linewidth=float(plot_cfg.get("road_linewidth", 0.30)),
                    alpha=float(plot_cfg.get("road_alpha", 0.22)),
                    zorder=1,
                )

    # Demand layers.
    if demand_service is not None and plot_cfg.get("show_demand_nodes", True):
        dplot = sample_for_plot(demand_service, int(plot_cfg.get("max_demand_points_on_map", 6000)))

        served = dplot[dplot["served_by_actual_site"]]
        unserved = dplot[~dplot["served_by_actual_site"]]
        underserved = dplot[dplot["underserved_flag"]]

        if plot_cfg.get("show_unserved_demand", True) and len(unserved):
            ax.scatter(
                unserved["lon"], unserved["lat"],
                s=float(plot_cfg.get("demand_marker_size", 10)),
                alpha=float(plot_cfg.get("demand_alpha", 0.35)),
                marker="x",
                label="unserved demand nodes",
                zorder=2,
            )

        if plot_cfg.get("show_served_demand", True) and len(served):
            ax.scatter(
                served["lon"], served["lat"],
                s=float(plot_cfg.get("demand_marker_size", 10)),
                alpha=float(plot_cfg.get("demand_alpha", 0.35)),
                marker=".",
                label="served demand nodes",
                zorder=2,
            )

        if plot_cfg.get("show_underserved_demand", True) and len(underserved):
            ax.scatter(
                underserved["lon"], underserved["lat"],
                s=float(plot_cfg.get("underserved_marker_size", 24)),
                alpha=float(plot_cfg.get("underserved_alpha", 0.85)),
                marker="o",
                facecolors="none",
                label="proxy-underserved demand nodes",
                zorder=3,
            )

    # Candidate layer.
    if candidates is not None and plot_cfg.get("show_all_candidates", True):
        ax.scatter(
            candidates["lon"], candidates["lat"],
            s=float(plot_cfg.get("candidate_marker_size", 14)),
            alpha=float(plot_cfg.get("candidate_alpha", 0.25)),
            label="candidate sites",
            zorder=4,
        )

    # Actual selected placements.
    ax.scatter(
        selected["lon"], selected["lat"],
        s=float(plot_cfg.get("selected_marker_size", 180)),
        marker="*",
        edgecolor="black",
        linewidth=0.9,
        label="actual selected EVCS sites",
        zorder=6,
    )

    for _, r in selected.iterrows():
        ax.text(
            r["lon"], r["lat"],
            str(int(r["actual_selected_rank"])),
            fontsize=8,
            ha="center",
            va="center",
            zorder=7,
        )

    ax.set_title(f"Actual analysis EVCS placements with demand service: {city_id}", fontsize=13)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.20)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()

    for fmt in plot_cfg.get("figure_format", ["png"]):
        fig.savefig(fig_dir / f"actual_placement_map_{city_id}.{fmt}", dpi=int(plot_cfg.get("dpi", 300)), bbox_inches="tight")

    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--selected-sites", type=Path, required=True, help="Usually T8_final_selected_sites.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--city-ids", nargs="*", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ap_cfg = cfg["actual_placement_map"]

    output_dir = args.output_dir
    for sub in ["figures", "tables", "selected", "demand_service"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(args.dataset_index)
    selected_all = pd.read_csv(args.selected_sites)

    real_rows = index[index["data_mode"].str.lower() == "real"].copy()
    if args.city_ids:
        real_rows = real_rows[real_rows["city_id"].isin(args.city_ids)].copy()

    summaries = []

    for _, row in real_rows.iterrows():
        city_id = row["city_id"]
        data_dir = Path(row["data_dir"])

        print(f"Processing actual placements for {city_id} from {data_dir}")

        try:
            selected, meta = choose_actual_selected_sites(selected_all, city_id, cfg)
            candidates = load_candidates(data_dir)
            demand = load_demand_points(data_dir, underserved_quantile=float(ap_cfg.get("underserved_quantile", 0.75)))

            demand_service = classify_demand_service(
                demand=demand,
                selected=selected,
                service_radius_km=float(ap_cfg.get("demand_service_radius_km", 5.0)),
            )

            summary = {}
            summary.update(meta)
            summary.update(summarize_actual_placement(selected, demand_service))
            summary.update({
                "city_id": city_id,
                "country": row.get("country", ""),
                "data_mode": "real",
                "expected_k": int(ap_cfg.get("expected_k", 10)),
                "selected_count_matches_expected": int(len(selected)) == int(ap_cfg.get("expected_k", 10)),
                "demand_service_radius_km": float(ap_cfg.get("demand_service_radius_km", 5.0)),
            })

            summaries.append(summary)

            selected.to_csv(output_dir / "selected" / f"actual_selected_sites_{city_id}.csv", index=False)
            if demand_service is not None:
                demand_service.to_csv(output_dir / "demand_service" / f"actual_demand_service_{city_id}.csv", index=False)

            plot_actual_city_map(
                city_id=city_id,
                data_dir=data_dir,
                selected=selected,
                candidates=candidates,
                demand_service=demand_service,
                output_dir=output_dir,
                cfg=cfg,
            )

            print(
                f"  method={meta['method_id']} shortlist={meta['shortlist_size']} "
                f"selected={len(selected)} served_mass_rate={summary.get('served_demand_mass_rate', np.nan):.3f} "
                f"underserved_served_rate={summary.get('served_underserved_demand_mass_rate', np.nan):.3f}"
            )

        except Exception as exc:
            print(f"ERROR for {city_id}: {exc}")
            summaries.append({"city_id": city_id, "error": str(exc), "selected_count": 0})

    pd.DataFrame(summaries).to_csv(output_dir / "tables" / "actual_placement_summary.csv", index=False)
    print(f"Done. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
