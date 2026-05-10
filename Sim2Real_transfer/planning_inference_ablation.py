#!/usr/bin/env python3
"""
Minimal ablation analysis for distribution-aware EVCS planning inference.

This script is designed to sit in the same directory as final_placement_v2.py.
It reuses the existing data loading, candidate preparation, scoring components,
spatial selection, and map helper functions from final_placement_v2.py.

It generates the compact post-hoc planning ablation:

  A0: learned_topk
      Uses only the learned normalized score u_i.
      No planning-score correction and no spatial constraints.

  A1: planning_score_topk
      Uses the full planning score psi_i.
      No spatial constraints.

  A2: distribution_aware
      Uses the full planning score psi_i.
      Applies the grid-zone capacity and minimum-separation constraints.

Optional:

  A3: equal_weight_distribution_aware
      Uses equal weights over all five normalized components.
      Applies the same distribution-aware spatial constraints.

Outputs:

  outputs_planning_ablation/
    selected/
      A0_learned_topk_<city_id>.csv
      A1_planning_score_topk_<city_id>.csv
      A2_distribution_aware_<city_id>.csv

    rankings/
      A0_learned_topk_ranked_sites_<city_id>.csv
      A1_planning_score_topk_ranked_sites_<city_id>.csv
      A2_distribution_aware_ranked_sites_<city_id>.csv

    tables/
      planning_inference_ablation_summary.csv

    figures/
      A0_learned_topk_<city_id>.png
      A1_planning_score_topk_<city_id>.png
      A2_distribution_aware_<city_id>.png
      ablation_three_panel_<city_id>.png

Example:

  python planning_inference_ablation.py \
    --dataset-index dataset_index.csv \
    --rankings outputs_sim2real/rankings/ranked_candidates_all_methods.csv \
    --config config/final_placement_v2.yaml \
    --output-dir outputs_planning_ablation \
    --city-ids omn_muscat omn_nizwa omn_salalah
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

try:
    from final_placement_v2 import (
        prepare_candidate_pool,
        select_distribution_aware,
        pairwise_selected_stats,
        load_city_extent,
        apply_map_extent,
        add_optional_basemap,
        load_road_graph_for_plot,
    )
except ImportError as exc:
    raise SystemExit(
        "Could not import final_placement_v2.py. Place this script in the same "
        "directory as final_placement_v2.py, or update the import block at the top "
        "of this file.\n"
        f"Original error: {exc}"
    )


COMPONENT_COLUMNS = [
    "rank_score_norm",
    "local_demand_norm",
    "suitability_norm",
    "cost_efficiency_norm",
    "equity_norm",
]

VARIANT_SPECS = [
    {
        "variant_id": "A0_learned_topk",
        "variant_label": "Learned Top-K",
        "score_type": "learned_only",
        "spatial_constraints": False,
    },
    {
        "variant_id": "A1_planning_score_topk",
        "variant_label": "Planning Score Top-K",
        "score_type": "default_planning",
        "spatial_constraints": False,
    },
    {
        "variant_id": "A2_distribution_aware",
        "variant_label": "Distribution-Aware Inference",
        "score_type": "default_planning",
        "spatial_constraints": True,
    },
]


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------


def ensure_output_dirs(output_dir: Path) -> None:
    for sub in ["selected", "rankings", "tables", "figures", "logs"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)



def get_default_weights(cfg: dict) -> Dict[str, float]:
    """Read planning-score weights from config, with paper defaults as fallback."""
    w = cfg.get("final_placement", {}).get("weights", {})
    return {
        "rank_score": float(w.get("rank_score", 0.45)),
        "local_demand_score": float(w.get("local_demand_score", 0.25)),
        "suitability_score": float(w.get("suitability_score", 0.15)),
        "cost_efficiency": float(w.get("cost_efficiency", 0.10)),
        "underserved_or_equity": float(w.get("underserved_or_equity", 0.05)),
    }



def normalized_equal_weight_score(df: pd.DataFrame) -> pd.Series:
    """Equal-weight score over the five normalized planning components."""
    cols = [c for c in COMPONENT_COLUMNS if c in df.columns]
    if not cols:
        raise ValueError("No normalized component columns are available for equal-weight scoring.")
    return df[cols].mean(axis=1)



def add_score_columns(candidates: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Preserve the default planning score from prepare_candidate_pool and add
    explicit named score columns for ablation variants.
    """
    out = candidates.copy()

    if "final_rank_score_v2" not in out.columns:
        raise ValueError("Expected final_rank_score_v2 from prepare_candidate_pool.")

    if "rank_score_norm" not in out.columns:
        raise ValueError("Expected rank_score_norm from prepare_candidate_pool.")

    out["score_learned_only"] = out["rank_score_norm"].astype(float)
    out["score_default_planning"] = out["final_rank_score_v2"].astype(float)
    out["score_equal_weight"] = normalized_equal_weight_score(out)

    # Keep a copy under a paper-readable name.
    out["planning_score_default"] = out["score_default_planning"]

    weights = get_default_weights(cfg)
    out["alpha_rank_score"] = weights["rank_score"]
    out["alpha_local_demand_score"] = weights["local_demand_score"]
    out["alpha_suitability_score"] = weights["suitability_score"]
    out["alpha_cost_efficiency"] = weights["cost_efficiency"]
    out["alpha_underserved_or_equity"] = weights["underserved_or_equity"]

    return out



def get_variant_score_col(score_type: str) -> str:
    if score_type == "learned_only":
        return "score_learned_only"
    if score_type == "default_planning":
        return "score_default_planning"
    if score_type == "equal_weight":
        return "score_equal_weight"
    raise ValueError(f"Unknown score_type: {score_type}")



def select_topk_no_constraints(
    candidates: pd.DataFrame,
    score_col: str,
    k: int,
) -> Tuple[pd.DataFrame, dict]:
    """Select top K sites by score, without spatial constraints."""
    selected = candidates.sort_values(score_col, ascending=False).head(k).copy()
    selected["final_selected_v2"] = 1
    selected["final_rank_v2"] = range(1, len(selected) + 1)

    meta = {
        "requested_k": int(k),
        "selected_count": int(len(selected)),
        "active_zones": int(selected["zone_id"].nunique()) if len(selected) and "zone_id" in selected.columns else 0,
        "target_active_zones": np.nan,
        "max_per_zone": np.nan,
        "zone_constraint_met": np.nan,
        "min_separation_used_km": 0.0,
        "separation_constraint_relaxed": False,
    }
    return selected, meta



def select_variant(
    candidates: pd.DataFrame,
    cfg: dict,
    score_type: str,
    spatial_constraints: bool,
) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Run one ablation variant.

    Returns:
      selected_df, meta, ranked_candidates_df
    """
    k = int(cfg["final_placement"]["k"])
    score_col = get_variant_score_col(score_type)

    cands = candidates.copy()
    cands["variant_score"] = cands[score_col].astype(float)

    # Reuse final_rank_score_v2 because the existing distribution-aware selector
    # expects this column.
    cands["final_rank_score_v2"] = cands["variant_score"]

    ranked = cands.sort_values("variant_score", ascending=False).copy()
    ranked["variant_rank"] = range(1, len(ranked) + 1)

    if spatial_constraints:
        selected, meta = select_distribution_aware(cands, cfg)
        selected = selected.copy()
        selected["variant_score"] = selected[score_col].astype(float)
    else:
        selected, meta = select_topk_no_constraints(cands, "variant_score", k)

    selected["selection_score_type"] = score_type
    selected["spatial_constraints"] = bool(spatial_constraints)

    selected_ids = set(selected["candidate_id"].astype(str)) if len(selected) else set()
    ranked["selected_in_variant"] = ranked["candidate_id"].astype(str).isin(selected_ids).astype(int)

    return selected, meta, ranked


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def safe_mean(df: pd.DataFrame, col: str) -> float:
    if df is None or len(df) == 0 or col not in df.columns:
        return float("nan")
    return float(pd.to_numeric(df[col], errors="coerce").mean())



def safe_sum(df: pd.DataFrame, col: str) -> float:
    if df is None or len(df) == 0 or col not in df.columns:
        return float("nan")
    return float(pd.to_numeric(df[col], errors="coerce").sum())



def zone_entropy(selected: pd.DataFrame) -> Tuple[float, float]:
    """Return raw and normalized zone entropy for selected sites."""
    if selected is None or len(selected) == 0 or "zone_id" not in selected.columns:
        return float("nan"), float("nan")

    counts = selected["zone_id"].value_counts()
    if len(counts) == 0:
        return float("nan"), float("nan")

    p = counts / counts.sum()
    entropy = float(-(p * np.log(p)).sum())

    # Normalize by maximum possible entropy for the observed selected count.
    denom = math.log(min(len(selected), counts.size)) if min(len(selected), counts.size) > 1 else 0.0
    entropy_norm = float(entropy / denom) if denom > 0 else float("nan")

    return entropy, entropy_norm



def summarize_selection(
    city_id: str,
    country: str,
    variant_id: str,
    variant_label: str,
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
    meta: dict,
    baseline_ids: Optional[set],
    cfg: dict,
) -> dict:
    """Build one summary row for the ablation table."""
    k = int(cfg["final_placement"]["k"])
    selected_ids = set(selected["candidate_id"].astype(str)) if len(selected) else set()

    if baseline_ids is None:
        overlap = float("nan")
        overlap_count = np.nan
    else:
        overlap_count = len(selected_ids.intersection(baseline_ids))
        overlap = float(overlap_count / k) if k > 0 else float("nan")

    pair_stats = pairwise_selected_stats(selected)
    ent, ent_norm = zone_entropy(selected)

    zone_counts = selected["zone_id"].value_counts().to_dict() if len(selected) and "zone_id" in selected.columns else {}
    max_sites_per_zone = int(max(zone_counts.values())) if zone_counts else 0

    row = {
        "city_id": city_id,
        "country": country,
        "variant_id": variant_id,
        "variant_label": variant_label,
        "requested_k": int(k),
        "selected_count": int(len(selected)),
        "mean_variant_score": safe_mean(selected, "variant_score"),
        "sum_variant_score": safe_sum(selected, "variant_score"),
        "mean_default_planning_score": safe_mean(selected, "planning_score_default"),
        "mean_u_score": safe_mean(selected, "rank_score_norm"),
        "mean_local_demand": safe_mean(selected, "local_demand_norm"),
        "mean_suitability": safe_mean(selected, "suitability_norm"),
        "mean_cost_efficiency": safe_mean(selected, "cost_efficiency_norm"),
        "mean_underserved_equity": safe_mean(selected, "equity_norm"),
        "active_zones": int(selected["zone_id"].nunique()) if len(selected) and "zone_id" in selected.columns else 0,
        "max_sites_per_zone": max_sites_per_zone,
        "zone_entropy": ent,
        "normalized_zone_entropy": ent_norm,
        "min_pairwise_distance_km": pair_stats.get("min_pairwise_distance_km", np.nan),
        "mean_pairwise_distance_km": pair_stats.get("mean_pairwise_distance_km", np.nan),
        "max_pairwise_distance_km": pair_stats.get("max_pairwise_distance_km", np.nan),
        "overlap_count_with_A0": overlap_count,
        "overlap_fraction_with_A0": overlap,
        "min_separation_used_km": meta.get("min_separation_used_km", np.nan),
        "separation_constraint_relaxed": meta.get("separation_constraint_relaxed", np.nan),
        "zone_constraint_met": meta.get("zone_constraint_met", np.nan),
        "target_active_zones": meta.get("target_active_zones", np.nan),
        "max_per_zone_config": meta.get("max_per_zone", np.nan),
        "selected_zone_counts": json.dumps(zone_counts),
    }

    # Store configured default alphas for traceability.
    weights = get_default_weights(cfg)
    row.update(
        {
            "alpha_1_u": weights["rank_score"],
            "alpha_2_L": weights["local_demand_score"],
            "alpha_3_S": weights["suitability_score"],
            "alpha_4_cost_efficiency": weights["cost_efficiency"],
            "alpha_5_E": weights["underserved_or_equity"],
        }
    )

    return row


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def _draw_road_graph(ax, data_dir: Path, cfg: dict) -> None:
    plot_cfg = cfg["final_placement"].get("plots", {})
    if not plot_cfg.get("draw_road_graph", True):
        return

    road_segments = load_road_graph_for_plot(
        data_dir=data_dir,
        max_edges=int(plot_cfg.get("max_edges_on_map", 15000)),
    )

    if road_segments is None or len(road_segments) == 0:
        return

    for _, e in road_segments.iterrows():
        ax.plot(
            [e["lon1"], e["lon2"]],
            [e["lat1"], e["lat2"]],
            linewidth=float(plot_cfg.get("road_linewidth", 0.35)),
            alpha=float(plot_cfg.get("road_alpha", 0.25)),
            zorder=1,
        )



def draw_map_layers(
    ax,
    city_id: str,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    data_dir: Path,
    cfg: dict,
    title: str,
    show_legend: bool = True,
) -> None:
    """Draw one ablation map on an existing axis."""
    plot_cfg = cfg["final_placement"].get("plots", {})
    padding = float(plot_cfg.get("map_padding_fraction", 0.04))

    try:
        city_coords = load_city_extent(data_dir, candidates)
    except Exception:
        city_coords = candidates[["lat", "lon"]].copy()

    apply_map_extent(ax, city_coords, padding_fraction=padding)
    add_optional_basemap(ax, cfg)
    _draw_road_graph(ax, data_dir, cfg)

    # Candidate sites.
    ax.scatter(
        candidates["lon"],
        candidates["lat"],
        s=70,
        alpha=0.30,
        label="Cand. Site",
        zorder=3,
    )

    # Top candidates under the variant score.
    top_n = int(plot_cfg.get("show_top_ranked_candidates", 40))
    sort_col = "variant_score" if "variant_score" in candidates.columns else "final_rank_score_v2"
    top = candidates.sort_values(sort_col, ascending=False).head(top_n)

    ax.scatter(
        top["lon"],
        top["lat"],
        s=70,
        alpha=0.55,
        marker="^",
        label=f"Top {top_n} Sites",
        zorder=4,
    )

    # Selected sites.
    if selected is not None and len(selected):
        ax.scatter(
            selected["lon"],
            selected["lat"],
            s=145,
            marker="*",
            edgecolor="black",
            linewidth=0.9,
            label="EVCS",
            zorder=5,
        )

        for _, r in selected.iterrows():
            rank_value = int(r.get("final_rank_v2", 0))
            if rank_value > 0:
                ax.text(
                    r["lon"],
                    r["lat"],
                    str(rank_value),
                    fontsize=18,
                    ha="center",
                    va="center",
                    zorder=6,
                )

    # ax.set_title(title, fontsize=18)
    ax.set_xlabel("Longitude", fontsize=20)
    ax.set_ylabel("Latitude", fontsize=20)
    ax.grid(True, alpha=0.20)

    if show_legend:
        ax.legend(loc="best", fontsize=20)



def plot_individual_variant_map(
    city_id: str,
    variant_id: str,
    variant_label: str,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    data_dir: Path,
    output_dir: Path,
    cfg: dict,
) -> None:
    plot_cfg = cfg["final_placement"].get("plots", {})
    dpi = int(plot_cfg.get("dpi", 300))
    formats = plot_cfg.get("figure_format", ["png"])

    fig, ax = plt.subplots(figsize=(9, 8))
    draw_map_layers(
        ax=ax,
        city_id=city_id,
        candidates=candidates,
        selected=selected,
        data_dir=data_dir,
        cfg=cfg,
        title=f"{variant_label}: {city_id}",
        show_legend=True,
    )
    fig.tight_layout()

    for fmt in formats:
        fig.savefig(
            output_dir / "figures" / f"{variant_id}_{city_id}.{fmt}",
            dpi=dpi,
            bbox_inches="tight",
        )

    plt.close(fig)



def plot_three_panel_map(
    city_id: str,
    variant_outputs: Sequence[dict],
    data_dir: Path,
    output_dir: Path,
    cfg: dict,
) -> None:
    """Create the main paper figure: A0, A1, A2 side-by-side."""
    plot_cfg = cfg["final_placement"].get("plots", {})
    dpi = int(plot_cfg.get("dpi", 300))
    formats = plot_cfg.get("figure_format", ["png"])

    main_variants = [v for v in variant_outputs if v["variant_id"] in {
        "A0_learned_topk",
        "A1_planning_score_topk",
        "A2_distribution_aware",
    }]

    if len(main_variants) == 0:
        return

    n = len(main_variants)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 5.3), squeeze=False)
    axes = axes[0]

    for idx, (ax, v) in enumerate(zip(axes, main_variants)):
        draw_map_layers(
            ax=ax,
            city_id=city_id,
            candidates=v["ranked"],
            selected=v["selected"],
            data_dir=data_dir,
            cfg=cfg,
            title=f"({chr(97 + idx)}) {v['variant_label']}",
            show_legend=(idx == 0),
        )

    fig.suptitle(f"Planning inference ablation: {city_id}", fontsize=13)
    fig.tight_layout()

    for fmt in formats:
        fig.savefig(
            output_dir / "figures" / f"ablation_three_panel_{city_id}.{fmt}",
            dpi=dpi,
            bbox_inches="tight",
        )

    plt.close(fig)


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------


def run_city(
    city_id: str,
    country: str,
    data_dir: Path,
    rankings: pd.DataFrame,
    cfg: dict,
    output_dir: Path,
    include_equal_weight: bool = False,
    make_individual_maps: bool = True,
    make_three_panel: bool = True,
) -> List[dict]:
    print(f"Processing ablations for {city_id} from {data_dir}")

    candidates = prepare_candidate_pool(city_id, data_dir, rankings, cfg)
    candidates = add_score_columns(candidates, cfg)
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)

    specs = list(VARIANT_SPECS)
    if include_equal_weight:
        specs.append(
            {
                "variant_id": "A3_equal_weight_distribution_aware",
                "variant_label": "Equal-Weight Distribution-Aware",
                "score_type": "equal_weight",
                "spatial_constraints": True,
            }
        )

    summaries: List[dict] = []
    variant_outputs: List[dict] = []
    baseline_ids: Optional[set] = None

    for spec in specs:
        variant_id = spec["variant_id"]
        variant_label = spec["variant_label"]
        score_type = spec["score_type"]
        spatial_constraints = bool(spec["spatial_constraints"])

        selected, meta, ranked = select_variant(
            candidates=candidates,
            cfg=cfg,
            score_type=score_type,
            spatial_constraints=spatial_constraints,
        )

        selected["city_id"] = city_id
        selected["variant_id"] = variant_id
        selected["variant_label"] = variant_label

        ranked["city_id"] = city_id
        ranked["variant_id"] = variant_id
        ranked["variant_label"] = variant_label

        # Set A0 as the baseline for overlap.
        if variant_id == "A0_learned_topk":
            baseline_ids = set(selected["candidate_id"].astype(str))

        summary = summarize_selection(
            city_id=city_id,
            country=country,
            variant_id=variant_id,
            variant_label=variant_label,
            selected=selected,
            candidates=candidates,
            meta=meta,
            baseline_ids=baseline_ids,
            cfg=cfg,
        )
        summaries.append(summary)

        selected.to_csv(output_dir / "selected" / f"{variant_id}_{city_id}.csv", index=False)
        ranked.to_csv(output_dir / "rankings" / f"{variant_id}_ranked_sites_{city_id}.csv", index=False)

        if make_individual_maps:
            plot_individual_variant_map(
                city_id=city_id,
                variant_id=variant_id,
                variant_label=variant_label,
                candidates=ranked,
                selected=selected,
                data_dir=data_dir,
                output_dir=output_dir,
                cfg=cfg,
            )

        variant_outputs.append(
            {
                "variant_id": variant_id,
                "variant_label": variant_label,
                "selected": selected,
                "ranked": ranked,
                "meta": meta,
            }
        )

        print(
            f"  {variant_id}: selected={len(selected)} "
            f"active_zones={summary['active_zones']} "
            f"min_dist={summary['min_pairwise_distance_km']:.3f} "
            f"overlap_A0={summary['overlap_fraction_with_A0']}"
        )

    if make_three_panel:
        plot_three_panel_map(
            city_id=city_id,
            variant_outputs=variant_outputs,
            data_dir=data_dir,
            output_dir=output_dir,
            cfg=cfg,
        )

    return summaries



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--city-ids", nargs="*", default=None)
    parser.add_argument(
        "--include-equal-weight",
        action="store_true",
        help="Also run A3 equal-weight distribution-aware robustness variant.",
    )
    parser.add_argument(
        "--no-individual-maps",
        action="store_true",
        help="Skip individual variant maps.",
    )
    parser.add_argument(
        "--no-three-panel",
        action="store_true",
        help="Skip the three-panel paper figure.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir
    ensure_output_dirs(output_dir)

    index = pd.read_csv(args.dataset_index)
    rankings = pd.read_csv(args.rankings)

    if "data_mode" not in index.columns:
        raise ValueError("dataset_index.csv must include a data_mode column.")

    real_rows = index[index["data_mode"].astype(str).str.lower() == "real"].copy()

    if args.city_ids:
        real_rows = real_rows[real_rows["city_id"].isin(args.city_ids)].copy()

    if len(real_rows) == 0:
        raise ValueError("No real data rows found for the requested city IDs.")

    all_summaries: List[dict] = []

    for _, row in real_rows.iterrows():
        city_id = str(row["city_id"])
        country = str(row.get("country", ""))
        data_dir = Path(row["data_dir"])

        try:
            summaries = run_city(
                city_id=city_id,
                country=country,
                data_dir=data_dir,
                rankings=rankings,
                cfg=cfg,
                output_dir=output_dir,
                include_equal_weight=bool(args.include_equal_weight),
                make_individual_maps=not bool(args.no_individual_maps),
                make_three_panel=not bool(args.no_three_panel),
            )
            all_summaries.extend(summaries)
        except Exception as exc:
            print(f"ERROR for {city_id}: {exc}")
            all_summaries.append(
                {
                    "city_id": city_id,
                    "country": country,
                    "variant_id": "ERROR",
                    "variant_label": "ERROR",
                    "error": str(exc),
                }
            )

    summary_df = pd.DataFrame(all_summaries)
    summary_path = output_dir / "tables" / "planning_inference_ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Done. Summary written to {summary_path}")
    print(f"Figures written to {output_dir / 'figures'}")
    print(f"Selected-site CSVs written to {output_dir / 'selected'}")


if __name__ == "__main__":
    main()
