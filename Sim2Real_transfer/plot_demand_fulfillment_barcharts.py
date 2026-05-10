#!/usr/bin/env python3
"""
Create bar-chart plots for demand-fulfillment results from planning-inference ablation.

Expected input:
  outputs_planning_ablation/tables/demand_fulfillment_summary.csv

or:
  outputs_planning_ablation/tables/planning_inference_ablation_with_demand_fulfillment.csv

The script creates one separate PNG/PDF per metric:
  - fulfilled demand rate by nodes
  - unfulfilled demand rate by nodes
  - fulfilled demand-mass rate
  - fulfilled underserved/high-need demand rate
  - selected count, if available
  - active zones, if available

Usage:
    python plot_demand_fulfillment_barcharts.py \
      --summary-csv outputs_planning_ablation/tables/planning_inference_ablation_with_demand_fulfillment.csv \
      --output-dir outputs_planning_ablation/figures/bar_charts

If you only have demand_fulfillment_summary.csv:
    python plot_demand_fulfillment_barcharts.py \
      --summary-csv outputs_planning_ablation/tables/demand_fulfillment_summary.csv \
      --output-dir outputs_planning_ablation/figures/bar_charts
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


VARIANT_ORDER = [
    "A0_learned_topk",
    "A1_planning_score_topk",
    "A2_distribution_aware",
    "A3_equal_weight_distribution_aware",
]

VARIANT_LABELS = {
    "A0_learned_topk": "A0 Learned\nTop-K",
    "A1_planning_score_topk": "A1 Planning\nScore Top-K",
    "A2_distribution_aware": "A2 Distribution\nAware",
    "A3_equal_weight_distribution_aware": "A3 Equal-Weight\nDistribution Aware",
}


METRIC_SPECS = {
    "fulfilled_demand_rate_nodes": {
        "title": "Fulfilled demand nodes",
        "ylabel": "Fulfilled demand nodes (%)",
        "scale": 100.0,
        "filename": "bar_fulfilled_demand_nodes",
    },
    "unfulfilled_demand_rate_nodes": {
        "title": "Unfulfilled demand nodes",
        "ylabel": "Unfulfilled demand nodes (%)",
        "scale": 100.0,
        "filename": "bar_unfulfilled_demand_nodes",
    },
    "fulfilled_demand_rate_mass": {
        "title": "Fulfilled demand mass",
        "ylabel": "Fulfilled demand mass (%)",
        "scale": 100.0,
        "filename": "bar_fulfilled_demand_mass",
    },
    "unfulfilled_demand_rate_mass": {
        "title": "Unfulfilled demand mass",
        "ylabel": "Unfulfilled demand mass (%)",
        "scale": 100.0,
        "filename": "bar_unfulfilled_demand_mass",
    },
    "fulfilled_underserved_or_high_need_rate_nodes": {
        "title": "Fulfilled underserved/high-need demand nodes",
        "ylabel": "Fulfilled underserved/high-need nodes (%)",
        "scale": 100.0,
        "filename": "bar_fulfilled_underserved_high_need_nodes",
    },
    "unfulfilled_underserved_or_high_need_rate_nodes": {
        "title": "Unfulfilled underserved/high-need demand nodes",
        "ylabel": "Unfulfilled underserved/high-need nodes (%)",
        "scale": 100.0,
        "filename": "bar_unfulfilled_underserved_high_need_nodes",
    },
    "selected_count": {
        "title": "Selected EVCS sites",
        "ylabel": "Selected sites",
        "scale": 1.0,
        "filename": "bar_selected_count",
    },
    "active_zones": {
        "title": "Active spatial zones",
        "ylabel": "Active zones",
        "scale": 1.0,
        "filename": "bar_active_zones",
    },
    "min_pairwise_distance_km": {
        "title": "Minimum pairwise distance",
        "ylabel": "Minimum pairwise distance (km)",
        "scale": 1.0,
        "filename": "bar_min_pairwise_distance",
    },
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize likely column aliases from different summary versions."""
    out = df.copy()

    aliases = {
        "coverage_rate_nodes": "fulfilled_demand_rate_nodes",
        "covered_demand_rate_nodes": "fulfilled_demand_rate_nodes",
        "uncovered_demand_rate_nodes": "unfulfilled_demand_rate_nodes",
        "coverage_rate_mass": "fulfilled_demand_rate_mass",
        "covered_demand_rate_mass": "fulfilled_demand_rate_mass",
        "uncovered_demand_rate_mass": "unfulfilled_demand_rate_mass",
        "covered_underserved_or_high_need_rate_nodes": "fulfilled_underserved_or_high_need_rate_nodes",
        "uncovered_underserved_or_high_need_rate_nodes": "unfulfilled_underserved_or_high_need_rate_nodes",
    }

    for old, new in aliases.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]

    if "variant_id" not in out.columns:
        raise ValueError("Input CSV must contain a 'variant_id' column.")

    if "city_id" not in out.columns:
        raise ValueError("Input CSV must contain a 'city_id' column.")

    if "variant_label" not in out.columns:
        out["variant_label"] = out["variant_id"].map(VARIANT_LABELS).fillna(out["variant_id"])

    return out


def order_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows using preferred ablation order."""
    out = df.copy()
    order_map = {v: i for i, v in enumerate(VARIANT_ORDER)}
    out["_variant_order"] = out["variant_id"].map(order_map).fillna(999).astype(int)
    return out.sort_values(["city_id", "_variant_order", "variant_id"]).drop(columns=["_variant_order"])


def make_metric_bar_chart(
    df: pd.DataFrame,
    metric: str,
    spec: Dict[str, object],
    output_dir: Path,
    formats: List[str],
) -> None:
    """Create grouped bar chart for one metric across cities and variants."""
    if metric not in df.columns:
        print(f"Skipping {metric}: column not found.")
        return

    plot_df = df[["city_id", "variant_id", "variant_label", metric]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce") * float(spec["scale"])
    plot_df = plot_df.dropna(subset=[metric])

    if len(plot_df) == 0:
        print(f"Skipping {metric}: no valid numeric values.")
        return

    cities = list(plot_df["city_id"].drop_duplicates())
    variants = [v for v in VARIANT_ORDER if v in set(plot_df["variant_id"])]
    extra_variants = [v for v in plot_df["variant_id"].drop_duplicates() if v not in variants]
    variants += extra_variants

    x = np.arange(len(cities))
    n_variants = max(len(variants), 1)
    width = min(0.8 / n_variants, 0.24)

    fig_width = max(7.5, 1.5 * len(cities) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))

    for idx, variant_id in enumerate(variants):
        values = []
        for city in cities:
            row = plot_df[(plot_df["city_id"] == city) & (plot_df["variant_id"] == variant_id)]
            if len(row):
                values.append(float(row[metric].iloc[0]))
            else:
                values.append(np.nan)

        offset = (idx - (n_variants - 1) / 2) * width
        label = VARIANT_LABELS.get(variant_id, variant_id)
        ax.bar(x + offset, values, width, label=label)

    ax.set_title(str(spec["title"]))
    ax.set_ylabel(str(spec["ylabel"]))
    ax.set_xlabel("City")
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)

    if str(spec["ylabel"]).endswith("(%)"):
        ymax = plot_df[metric].max()
        ax.set_ylim(0, min(105, max(10, ymax * 1.15)))

    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()

    for fmt in formats:
        path = output_dir / f"{spec['filename']}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Wrote {path}")

    plt.close(fig)


def make_mean_metric_bar_chart(
    df: pd.DataFrame,
    metric: str,
    spec: Dict[str, object],
    output_dir: Path,
    formats: List[str],
) -> None:
    """Create one averaged bar chart across cities for one metric."""
    if metric not in df.columns:
        return

    plot_df = df[["variant_id", "variant_label", metric]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce") * float(spec["scale"])
    plot_df = plot_df.dropna(subset=[metric])

    if len(plot_df) == 0:
        return

    grouped = (
        plot_df.groupby("variant_id", as_index=False)
        .agg(mean_value=(metric, "mean"), std_value=(metric, "std"), n=("variant_id", "count"))
    )

    grouped["sem"] = grouped["std_value"] / np.sqrt(grouped["n"].clip(lower=1))
    grouped["variant_label"] = grouped["variant_id"].map(VARIANT_LABELS).fillna(grouped["variant_id"])

    order_map = {v: i for i, v in enumerate(VARIANT_ORDER)}
    grouped["_order"] = grouped["variant_id"].map(order_map).fillna(999).astype(int)
    grouped = grouped.sort_values(["_order", "variant_id"])

    x = np.arange(len(grouped))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(x, grouped["mean_value"], yerr=grouped["sem"].fillna(0.0), capsize=4)

    ax.set_title(f"{spec['title']} averaged across cities")
    ax.set_ylabel(str(spec["ylabel"]))
    ax.set_xlabel("Variant")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["variant_label"], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)

    if str(spec["ylabel"]).endswith("(%)"):
        ymax = grouped["mean_value"].max()
        ax.set_ylim(0, min(105, max(10, ymax * 1.15)))

    fig.tight_layout()

    for fmt in formats:
        path = output_dir / f"{spec['filename']}_mean_across_cities.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Wrote {path}")

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], help="Output formats, e.g. png pdf")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional metric columns to plot. Defaults to all available standard metrics.",
    )
    parser.add_argument(
        "--mean-across-cities",
        action="store_true",
        help="Also create mean-across-cities bar charts with SEM error bars.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary_csv)
    df = normalize_columns(df)
    df = order_variants(df)

    metrics = args.metrics if args.metrics else list(METRIC_SPECS.keys())

    for metric in metrics:
        spec = METRIC_SPECS.get(
            metric,
            {
                "title": metric.replace("_", " ").title(),
                "ylabel": metric.replace("_", " ").title(),
                "scale": 1.0,
                "filename": f"bar_{metric}",
            },
        )
        make_metric_bar_chart(df, metric, spec, args.output_dir, args.formats)

        if args.mean_across_cities:
            make_mean_metric_bar_chart(df, metric, spec, args.output_dir, args.formats)

    print("Done.")


if __name__ == "__main__":
    main()
