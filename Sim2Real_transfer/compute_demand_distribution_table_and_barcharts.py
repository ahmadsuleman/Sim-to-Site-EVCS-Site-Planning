#!/usr/bin/env python3
"""
Compute fulfilled / unfulfilled / underserved demand metrics for planning-inference ablations
and generate paper-ready bar charts.

This script is designed for the three-row paper table:

| Variant                   | Fulfilled demand % | Fulfilled demand mass % | Unfulfilled demand % |
| Fulfilled underserved % | Active zones | Min distance km |

It does NOT estimate these demand metrics from the inference summary alone.
It computes them from:
  1. selected EVCS sites for each ablation variant
  2. each city's demand-node file
  3. the service coverage radius

Expected selected-site files:
  <ablation-dir>/selected/A0_learned_topk_<city_id>.csv
  <ablation-dir>/selected/A1_planning_score_topk_<city_id>.csv
  <ablation-dir>/selected/A2_distribution_aware_<city_id>.csv

Expected inference summary:
  <ablation-dir>/tables/planning_inference_ablation_summary.csv

Main outputs:
  <output-dir>/tables/demand_distribution_metrics_by_city_variant.csv
  <output-dir>/tables/paper_ready_demand_distribution_table_macro_mean.csv
  <output-dir>/tables/paper_ready_demand_distribution_table_micro_total.csv

Figures:
  <output-dir>/figures/bar_paper_percentage_metrics_macro_mean.png
  <output-dir>/figures/bar_active_zones_macro_mean.png
  <output-dir>/figures/bar_min_distance_macro_mean.png

Usage:
  python compute_demand_distribution_table_and_barcharts.py \
    --dataset-index dataset_index.csv \
    --ablation-dir outputs_planning_ablation \
    --output-dir outputs_planning_ablation \
    --coverage-radius-km 2.0 \
    --city-ids omn_muscat omn_nizwa omn_salalah

If you have a real underserved column in demand files:
  python compute_demand_distribution_table_and_barcharts.py \
    --dataset-index dataset_index.csv \
    --ablation-dir outputs_planning_ablation \
    --output-dir outputs_planning_ablation \
    --coverage-radius-km 2.0 \
    --underserved-col underserved_score
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_VARIANTS = [
    "A0_learned_topk",
    "A1_planning_score_topk",
    "A2_distribution_aware",
    "A3_equal_weight_distribution_aware",
]

VARIANT_LABELS = {
    "A0_learned_topk": "A0 Learned Top-K",
    "A1_planning_score_topk": "A1 Planning Score Top-K",
    "A2_distribution_aware": "A2 Distribution-Aware",
    "A3_equal_weight_distribution_aware": "A3 Equal-Weight Distribution-Aware",
}

# SHORT_VARIANT_LABELS = {
#     "A0_learned_topk": "A0 Learned\nTop-K",
#     "A1_planning_score_topk": "A1 Planning\nScore Top-K",
#     "A2_distribution_aware": "A2 Distribution-\nAware",
#     "A3_equal_weight_distribution_aware": "A3 Equal-Weight\nDistribution-Aware",
# }
SHORT_VARIANT_LABELS = {
    "A0_learned_topk": "L-TopK",
    "A1_planning_score_topk": "P-TopK",
    "A2_distribution_aware": "DA-Inf",
    "A3_equal_weight_distribution_aware": "EW-DA",
}


# ---------------------------------------------------------------------
# File and schema helpers
# ---------------------------------------------------------------------

def find_file(data_dir: Path, patterns: Iterable[str]) -> Optional[Path]:
    files = list(data_dir.glob("*"))
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        for path in files:
            if regex.search(path.name):
                return path
    return None


def detect_lat_lon(df: pd.DataFrame) -> Tuple[str, str]:
    lat_candidates = [
        "lat", "latitude", "y", "node_lat", "demand_lat", "candidate_lat",
        "site_lat", "geometry_y"
    ]
    lon_candidates = [
        "lon", "lng", "longitude", "x", "node_lon", "demand_lon", "candidate_lon",
        "site_lon", "geometry_x"
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
        raise ValueError(f"Could not detect latitude/longitude columns in: {list(df.columns)}")

    return lat_col, lon_col


def detect_id_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["demand_id", "node_id", "id", "osmid"]
    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        cl = c.lower()
        if "demand" in cl and "id" in cl:
            return c
        if "node" in cl and "id" in cl:
            return c

    return None


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_demand_nodes(data_dir: Path) -> pd.DataFrame:
    demand_path = find_file(data_dir, [r"demand.*\.csv$"])
    if demand_path is None:
        raise FileNotFoundError(f"Could not find demand*.csv in {data_dir}")

    demand = pd.read_csv(demand_path)
    if len(demand) == 0:
        raise ValueError(f"Demand file is empty: {demand_path}")

    lat_col, lon_col = detect_lat_lon(demand)
    id_col = detect_id_col(demand)

    out = demand.copy()
    out = out.rename(columns={lat_col: "lat", lon_col: "lon"})
    out = out.dropna(subset=["lat", "lon"]).copy()

    if id_col is not None:
        out = out.rename(columns={id_col: "demand_id"})
    else:
        out["demand_id"] = [f"demand_{i}" for i in range(len(out))]

    mass_col = None
    for c in [
        "demand_mass", "demand", "weight", "population", "activity_intensity",
        "trip_count", "traffic_volume", "charging_demand", "ev_demand"
    ]:
        if c in out.columns:
            mass_col = c
            break

    if mass_col is None:
        out["demand_mass"] = 1.0
        out["_demand_mass_source"] = "uniform_1"
    else:
        out["demand_mass"] = pd.to_numeric(out[mass_col], errors="coerce").fillna(0.0)
        out["_demand_mass_source"] = mass_col

    # Avoid all-zero mass because rates become undefined.
    if float(out["demand_mass"].sum()) <= 0:
        out["demand_mass"] = 1.0
        out["_demand_mass_source"] = "uniform_1_due_to_zero_mass"

    return out


def choose_underserved_indicator(
    demand: pd.DataFrame,
    underserved_col: Optional[str],
    underserved_quantile: float,
) -> Tuple[pd.Series, str, float]:
    """
    Choose underserved/high-need demand nodes.

    Priority:
      1. user-provided underserved_col
      2. common underserved/equity/access-gap columns
      3. proxy: top quantile of demand_mass
    """
    out = demand.copy()

    if underserved_col:
        if underserved_col not in out.columns:
            raise ValueError(f"--underserved-col '{underserved_col}' not found in demand columns.")
        values = pd.to_numeric(out[underserved_col], errors="coerce")
        threshold = float(values.quantile(underserved_quantile))
        return (values >= threshold).fillna(False), underserved_col, threshold

    candidates = [
        "underserved_score",
        "equity_need_score",
        "equity_score",
        "low_access_score",
        "unserved_score",
        "service_gap_score",
        "access_gap_score",
        "underserved",
        "equity_need",
    ]

    for c in candidates:
        if c in out.columns:
            values = pd.to_numeric(out[c], errors="coerce")
            # Binary underserved column.
            unique_non_na = set(values.dropna().unique().tolist())
            if unique_non_na.issubset({0, 1}):
                return (values == 1).fillna(False), c, 0.5

            threshold = float(values.quantile(underserved_quantile))
            return (values >= threshold).fillna(False), c, threshold

    # Proxy high-need demand when no real underserved field exists.
    values = pd.to_numeric(out["demand_mass"], errors="coerce").fillna(0.0)
    threshold = float(values.quantile(underserved_quantile))
    return values >= threshold, "proxy_high_demand_quantile", threshold


def load_selected_sites(selected_path: Path) -> pd.DataFrame:
    if not selected_path.exists():
        raise FileNotFoundError(f"Selected-site file not found: {selected_path}")

    selected = pd.read_csv(selected_path)
    if len(selected) == 0:
        return selected

    lat_col, lon_col = detect_lat_lon(selected)
    selected = selected.rename(columns={lat_col: "lat", lon_col: "lon"}).copy()

    if "candidate_id" not in selected.columns:
        selected["candidate_id"] = [f"selected_{i}" for i in range(len(selected))]

    return selected.dropna(subset=["lat", "lon"]).copy()


# ---------------------------------------------------------------------
# Demand fulfillment computation
# ---------------------------------------------------------------------

def classify_demand(
    demand: pd.DataFrame,
    selected: pd.DataFrame,
    coverage_radius_km: float,
    underserved_col: Optional[str],
    underserved_quantile: float,
) -> pd.DataFrame:
    out = demand.copy()

    if selected is None or len(selected) == 0:
        out["nearest_selected_distance_km"] = np.nan
        out["nearest_selected_candidate_id"] = ""
        out["fulfilled"] = 0
    else:
        selected_records = selected[["candidate_id", "lat", "lon"]].to_dict("records")

        nearest_distances = []
        nearest_ids = []

        for _, drow in out.iterrows():
            best_dist = np.inf
            best_id = ""

            for srow in selected_records:
                dist = haversine_km(drow["lat"], drow["lon"], srow["lat"], srow["lon"])
                if dist < best_dist:
                    best_dist = dist
                    best_id = str(srow["candidate_id"])

            nearest_distances.append(float(best_dist))
            nearest_ids.append(best_id)

        out["nearest_selected_distance_km"] = nearest_distances
        out["nearest_selected_candidate_id"] = nearest_ids
        out["fulfilled"] = (out["nearest_selected_distance_km"] <= coverage_radius_km).astype(int)

    out["unfulfilled"] = 1 - out["fulfilled"]

    underserved_mask, source, threshold = choose_underserved_indicator(
        out,
        underserved_col=underserved_col,
        underserved_quantile=underserved_quantile,
    )

    out["underserved_or_high_need"] = underserved_mask.astype(int)
    out["underserved_source"] = source
    out["underserved_threshold"] = threshold
    out["fulfilled_underserved_or_high_need"] = (
        (out["fulfilled"] == 1) & (out["underserved_or_high_need"] == 1)
    ).astype(int)
    out["unfulfilled_underserved_or_high_need"] = (
        (out["unfulfilled"] == 1) & (out["underserved_or_high_need"] == 1)
    ).astype(int)

    return out


def summarize_classified_demand(
    city_id: str,
    country: str,
    variant_id: str,
    variant_label: str,
    classified: pd.DataFrame,
    selected_count: int,
    coverage_radius_km: float,
) -> Dict[str, object]:
    total_nodes = int(len(classified))
    total_mass = float(classified["demand_mass"].sum())

    fulfilled_nodes = int(classified["fulfilled"].sum())
    unfulfilled_nodes = int(classified["unfulfilled"].sum())

    fulfilled_mass = float(classified.loc[classified["fulfilled"] == 1, "demand_mass"].sum())
    unfulfilled_mass = float(classified.loc[classified["unfulfilled"] == 1, "demand_mass"].sum())

    underserved = classified[classified["underserved_or_high_need"] == 1]
    underserved_nodes = int(len(underserved))
    underserved_mass = float(underserved["demand_mass"].sum())

    fulfilled_underserved_nodes = int(underserved["fulfilled"].sum()) if underserved_nodes else 0
    unfulfilled_underserved_nodes = int(underserved["unfulfilled"].sum()) if underserved_nodes else 0

    fulfilled_underserved_mass = float(
        underserved.loc[underserved["fulfilled"] == 1, "demand_mass"].sum()
    ) if underserved_nodes else 0.0

    unfulfilled_underserved_mass = float(
        underserved.loc[underserved["unfulfilled"] == 1, "demand_mass"].sum()
    ) if underserved_nodes else 0.0

    distances = pd.to_numeric(classified["nearest_selected_distance_km"], errors="coerce")

    return {
        "city_id": city_id,
        "country": country,
        "variant_id": variant_id,
        "variant_label": variant_label,
        "selected_count": selected_count,
        "coverage_radius_km": coverage_radius_km,
        "total_demand_nodes": total_nodes,
        "fulfilled_demand_nodes": fulfilled_nodes,
        "unfulfilled_demand_nodes": unfulfilled_nodes,
        "fulfilled_demand_pct": 100.0 * fulfilled_nodes / total_nodes if total_nodes else np.nan,
        "unfulfilled_demand_pct": 100.0 * unfulfilled_nodes / total_nodes if total_nodes else np.nan,
        "total_demand_mass": total_mass,
        "fulfilled_demand_mass": fulfilled_mass,
        "unfulfilled_demand_mass": unfulfilled_mass,
        "fulfilled_demand_mass_pct": 100.0 * fulfilled_mass / total_mass if total_mass else np.nan,
        "unfulfilled_demand_mass_pct": 100.0 * unfulfilled_mass / total_mass if total_mass else np.nan,
        "underserved_or_high_need_nodes": underserved_nodes,
        "fulfilled_underserved_nodes": fulfilled_underserved_nodes,
        "unfulfilled_underserved_nodes": unfulfilled_underserved_nodes,
        "fulfilled_underserved_pct": 100.0 * fulfilled_underserved_nodes / underserved_nodes if underserved_nodes else np.nan,
        "unfulfilled_underserved_pct": 100.0 * unfulfilled_underserved_nodes / underserved_nodes if underserved_nodes else np.nan,
        "underserved_or_high_need_mass": underserved_mass,
        "fulfilled_underserved_mass": fulfilled_underserved_mass,
        "unfulfilled_underserved_mass": unfulfilled_underserved_mass,
        "fulfilled_underserved_mass_pct": 100.0 * fulfilled_underserved_mass / underserved_mass if underserved_mass else np.nan,
        "unfulfilled_underserved_mass_pct": 100.0 * unfulfilled_underserved_mass / underserved_mass if underserved_mass else np.nan,
        "mean_nearest_selected_distance_km": float(distances.mean()) if distances.notna().any() else np.nan,
        "median_nearest_selected_distance_km": float(distances.median()) if distances.notna().any() else np.nan,
        "p90_nearest_selected_distance_km": float(distances.quantile(0.90)) if distances.notna().any() else np.nan,
        "underserved_source": str(classified["underserved_source"].iloc[0]) if len(classified) else "",
        "underserved_threshold": float(classified["underserved_threshold"].iloc[0]) if len(classified) else np.nan,
        "demand_mass_source": str(classified["_demand_mass_source"].iloc[0]) if len(classified) else "",
    }


def pairwise_selected_stats(selected: pd.DataFrame) -> Dict[str, float]:
    if selected is None or len(selected) < 2:
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


def compute_active_zone_stats(selected: pd.DataFrame) -> Dict[str, float]:
    if selected is None or len(selected) == 0 or "zone_id" not in selected.columns:
        return {
            "active_zones": np.nan,
            "max_sites_per_zone": np.nan,
        }

    counts = selected["zone_id"].value_counts()
    return {
        "active_zones": int(counts.shape[0]),
        "max_sites_per_zone": int(counts.max()),
    }


# ---------------------------------------------------------------------
# Aggregation and plotting
# ---------------------------------------------------------------------

def merge_inference_metrics(demand_summary: pd.DataFrame, inference_summary: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = demand_summary.copy()

    if inference_summary is None or len(inference_summary) == 0:
        return out

    keep_cols = [
        "city_id", "variant_id",
        "active_zones", "max_sites_per_zone", "min_pairwise_distance_km",
        "mean_pairwise_distance_km", "selected_count", "requested_k",
        "overlap_fraction_with_A0", "separation_constraint_relaxed",
        "zone_constraint_met",
    ]

    keep_cols = [c for c in keep_cols if c in inference_summary.columns]
    inf = inference_summary[keep_cols].copy()

    # Avoid duplicate selected_count if already present.
    if "selected_count" in out.columns and "selected_count" in inf.columns:
        inf = inf.rename(columns={"selected_count": "selected_count_inference"})

    out = out.merge(inf, on=["city_id", "variant_id"], how="left")
    return out


def make_macro_table(df: pd.DataFrame, variant_order: List[str]) -> pd.DataFrame:
    metric_cols = [
        "fulfilled_demand_pct",
        "fulfilled_demand_mass_pct",
        "unfulfilled_demand_pct",
        "fulfilled_underserved_pct",
        "active_zones",
        "min_pairwise_distance_km",
    ]
    

    available = [c for c in metric_cols if c in df.columns]
    rows = []

    for variant_id in variant_order:
        sub = df[df["variant_id"] == variant_id].copy()
        if len(sub) == 0:
            continue

        row = {
            "variant_id": variant_id,
            "variant_label": VARIANT_LABELS.get(variant_id, variant_id),
            "num_cities": int(sub["city_id"].nunique()),
        }

        for col in available:
            values = pd.to_numeric(sub[col], errors="coerce")
            row[col] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{col}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def make_micro_table(df: pd.DataFrame, variant_order: List[str]) -> pd.DataFrame:
    rows = []

    for variant_id in variant_order:
        sub = df[df["variant_id"] == variant_id].copy()
        if len(sub) == 0:
            continue

        total_nodes = float(pd.to_numeric(sub["total_demand_nodes"], errors="coerce").sum())
        fulfilled_nodes = float(pd.to_numeric(sub["fulfilled_demand_nodes"], errors="coerce").sum())
        unfulfilled_nodes = float(pd.to_numeric(sub["unfulfilled_demand_nodes"], errors="coerce").sum())

        total_mass = float(pd.to_numeric(sub["total_demand_mass"], errors="coerce").sum())
        fulfilled_mass = float(pd.to_numeric(sub["fulfilled_demand_mass"], errors="coerce").sum())

        underserved_nodes = float(pd.to_numeric(sub["underserved_or_high_need_nodes"], errors="coerce").sum())
        fulfilled_underserved_nodes = float(pd.to_numeric(sub["fulfilled_underserved_nodes"], errors="coerce").sum())

        row = {
            "variant_id": variant_id,
            "variant_label": VARIANT_LABELS.get(variant_id, variant_id),
            "num_cities": int(sub["city_id"].nunique()),
            "fulfilled_demand_pct": 100.0 * fulfilled_nodes / total_nodes if total_nodes else np.nan,
            "fulfilled_demand_mass_pct": 100.0 * fulfilled_mass / total_mass if total_mass else np.nan,
            "unfulfilled_demand_pct": 100.0 * unfulfilled_nodes / total_nodes if total_nodes else np.nan,
            "fulfilled_underserved_pct": 100.0 * fulfilled_underserved_nodes / underserved_nodes if underserved_nodes else np.nan,
            "active_zones": float(pd.to_numeric(sub["active_zones"], errors="coerce").mean()) if "active_zones" in sub.columns else np.nan,
            "min_pairwise_distance_km": float(pd.to_numeric(sub["min_pairwise_distance_km"], errors="coerce").mean()) if "min_pairwise_distance_km" in sub.columns else np.nan,
        }

        rows.append(row)

    return pd.DataFrame(rows)


def plot_percentage_metrics(table: pd.DataFrame, output_dir: Path, formats: List[str]) -> None:
    metrics = [
    ("fulfilled_demand_pct", "FD ↑"),
    ("fulfilled_demand_mass_pct", "FDM ↑"),
    ("unfulfilled_demand_pct", "UD ↓"),
    ("fulfilled_underserved_pct", "FUD ↑"),
]

    available = [(c, label) for c, label in metrics if c in table.columns]
    if not available:
        return

    variants = table["variant_id"].tolist()
    x = np.arange(len(available))
    n_variants = max(len(variants), 1)
    width = min(0.8 / n_variants, 0.22)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    for idx, variant_id in enumerate(variants):
        row = table[table["variant_id"] == variant_id].iloc[0]
        values = [float(row[c]) if pd.notna(row[c]) else np.nan for c, _ in available]
        offset = (idx - (n_variants - 1) / 2) * width
        ax.bar(x + offset, values, width, label=SHORT_VARIANT_LABELS.get(variant_id, variant_id))

    # ax.set_title("Demand fulfillment metrics by inference variant")
    ax.set_ylabel("Rate (%)", fontsize=20)
    # ax.set_xlabel("Metric")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in available], rotation=0, ha="center", fontsize=20)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=16)
    fig.tight_layout()

    for fmt in formats:
        path = output_dir / f"bar_paper_percentage_metrics_macro_mean.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Wrote {path}")

    plt.close(fig)


def plot_single_metric(table: pd.DataFrame, metric: str, ylabel: str, title: str, filename: str, output_dir: Path, formats: List[str]) -> None:
    if metric not in table.columns:
        return

    plot_df = table.dropna(subset=[metric]).copy()
    if len(plot_df) == 0:
        return

    x = np.arange(len(plot_df))
    values = pd.to_numeric(plot_df[metric], errors="coerce").values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Variant")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [SHORT_VARIANT_LABELS.get(v, v) for v in plot_df["variant_id"]],
        rotation=0,
        ha="center",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    for fmt in formats:
        path = output_dir / f"{filename}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Wrote {path}")

    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inference-summary-csv", type=Path, default=None)
    parser.add_argument("--city-ids", nargs="*", default=None)
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    parser.add_argument("--coverage-radius-km", type=float, default=2.0)
    parser.add_argument("--underserved-col", type=str, default=None)
    parser.add_argument("--underserved-quantile", type=float, default=0.75)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()

    tables_dir = args.output_dir / "tables"
    nodes_dir = args.output_dir / "demand_nodes"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    nodes_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(args.dataset_index)
    if "data_mode" in index.columns:
        real_rows = index[index["data_mode"].astype(str).str.lower() == "real"].copy()
    else:
        real_rows = index.copy()

    if args.city_ids:
        real_rows = real_rows[real_rows["city_id"].isin(args.city_ids)].copy()

    if len(real_rows) == 0:
        raise ValueError("No city rows found in dataset index after filtering.")

    inference_csv = args.inference_summary_csv
    if inference_csv is None:
        candidate = args.ablation_dir / "tables" / "planning_inference_ablation_summary.csv"
        inference_csv = candidate if candidate.exists() else None

    inference_summary = None
    if inference_csv is not None and Path(inference_csv).exists():
        inference_summary = pd.read_csv(inference_csv)
        print(f"Loaded inference summary: {inference_csv}")
    else:
        print("No inference summary found. Active zones and min-distance will be computed from selected files where possible.")

    summary_rows = []

    for _, city_row in real_rows.iterrows():
        city_id = str(city_row["city_id"])
        country = str(city_row.get("country", ""))
        data_dir = Path(city_row["data_dir"])

        print(f"Processing demand fulfillment for {city_id}")

        demand = load_demand_nodes(data_dir)

        for variant_id in args.variants:
            selected_path = args.ablation_dir / "selected" / f"{variant_id}_{city_id}.csv"

            try:
                selected = load_selected_sites(selected_path)
            except Exception as exc:
                print(f"  Skipping {variant_id} for {city_id}: {exc}")
                continue

            classified = classify_demand(
                demand=demand,
                selected=selected,
                coverage_radius_km=args.coverage_radius_km,
                underserved_col=args.underserved_col,
                underserved_quantile=args.underserved_quantile,
            )

            # Add/overwrite metadata columns safely. Some demand files already contain city_id.
            classified = classified.copy()
            classified["city_id"] = city_id
            classified["variant_id"] = variant_id

            # Move metadata columns to the front without triggering duplicate-column insert errors.
            front_cols = ["city_id", "variant_id"]
            classified = classified[front_cols + [c for c in classified.columns if c not in front_cols]]

            node_path = nodes_dir / f"{variant_id}_demand_fulfillment_{city_id}.csv"
            classified.to_csv(node_path, index=False)

            row = summarize_classified_demand(
                city_id=city_id,
                country=country,
                variant_id=variant_id,
                variant_label=VARIANT_LABELS.get(variant_id, variant_id),
                classified=classified,
                selected_count=int(len(selected)),
                coverage_radius_km=float(args.coverage_radius_km),
            )

            # Compute fallback spatial metrics from selected file. Inference summary will override where available.
            row.update(pairwise_selected_stats(selected))
            row.update(compute_active_zone_stats(selected))

            summary_rows.append(row)

    demand_summary = pd.DataFrame(summary_rows)

    if len(demand_summary) == 0:
        raise RuntimeError("No demand fulfillment rows were computed. Check selected-site filenames and city IDs.")

    merged = merge_inference_metrics(demand_summary, inference_summary)

    # Prefer inference-summary active_zones/min distance if merge created duplicates.
    # If pandas creates _x/_y variants, clean them.
    for col in ["active_zones", "max_sites_per_zone", "min_pairwise_distance_km"]:
        x_col = f"{col}_x"
        y_col = f"{col}_y"
        if x_col in merged.columns and y_col in merged.columns:
            merged[col] = merged[y_col].combine_first(merged[x_col])
            merged = merged.drop(columns=[x_col, y_col])

    by_city_path = tables_dir / "demand_distribution_metrics_by_city_variant.csv"
    merged.to_csv(by_city_path, index=False)
    print(f"Wrote {by_city_path}")

    macro_table = make_macro_table(merged, args.variants)
    macro_path = tables_dir / "paper_ready_demand_distribution_table_macro_mean.csv"
    macro_table.to_csv(macro_path, index=False)
    print(f"Wrote {macro_path}")

    micro_table = make_micro_table(merged, args.variants)
    micro_path = tables_dir / "paper_ready_demand_distribution_table_micro_total.csv"
    micro_table.to_csv(micro_path, index=False)
    print(f"Wrote {micro_path}")

    plot_percentage_metrics(macro_table, figures_dir, args.formats)
    plot_single_metric(
        macro_table,
        metric="active_zones",
        ylabel="Active zones",
        title="Active zones by inference variant",
        filename="bar_active_zones_macro_mean",
        output_dir=figures_dir,
        formats=args.formats,
    )
    plot_single_metric(
        macro_table,
        metric="min_pairwise_distance_km",
        ylabel="Minimum pairwise distance (km)",
        title="Minimum pairwise distance by inference variant",
        filename="bar_min_distance_macro_mean",
        output_dir=figures_dir,
        formats=args.formats,
    )

    print("Done.")


if __name__ == "__main__":
    main()