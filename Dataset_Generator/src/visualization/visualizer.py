from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Visualizer:
    def __init__(self, plot_dir: Path):
        self.plot_dir = plot_dir
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def build_all(self, nodes: pd.DataFrame, edges: pd.DataFrame, candidates: pd.DataFrame, demand: pd.DataFrame,
                  kpis: pd.DataFrame, matrix: pd.DataFrame, provenance: pd.DataFrame, validation_metrics: pd.DataFrame) -> None:
        self._road_graph(edges, nodes, "01_road_graph_map.png")
        self._scatter_categorical(nodes, "land_use_type", "02_node_type_map.png", "Node land/use type")
        self._candidate_map(nodes, candidates, "03_candidate_sites_map.png")
        self._demand_map(demand, "04_demand_points_map.png")
        self._scatter_categorical(nodes, "land_use_type", "05_land_use_map.png", "Land use")
        for idx, feature in enumerate([
            "population_density", "activity_intensity", "traffic_flow_proxy", "grid_access_score",
            "available_space", "nearby_charger_pressure", "equity_need_score", "service_coverage_gain",
        ], start=6):
            self._scatter_feature(nodes, feature, f"{idx:02d}_{feature}_map.png", feature)
        self._kpi_distribution(kpis, "14_candidate_kpi_distribution.png")
        self._feature_histograms(matrix, "15_feature_histograms.png")
        self._correlation_heatmap(matrix, "16_feature_correlation_heatmap.png")
        self._boxplots_by_land_use(matrix, "17_boxplots_by_land_use.png")
        self._bar(candidates, "candidate_source", "18_candidate_source_breakdown.png", "Candidate source breakdown")
        self._provenance_breakdown(provenance, "19_feature_provenance_breakdown.png")
        self._scatter_feature(nodes, "uncertainty_score", "20_uncertainty_map.png", "Uncertainty score")
        self._dashboard(validation_metrics, "21_dataset_validation_dashboard.png")

    def _road_graph(self, edges: pd.DataFrame, nodes: pd.DataFrame, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 7))
        pos = nodes.set_index("node_id")[["lon", "lat"]].to_dict("index")
        for r in edges.itertuples():
            a, b = pos.get(str(r.source)), pos.get(str(r.target))
            if a and b:
                ax.plot([a["lon"], b["lon"]], [a["lat"], b["lat"]], linewidth=0.5, alpha=0.45)
        ax.scatter(nodes["lon"], nodes["lat"], s=3, alpha=0.6)
        ax.set_title("Road graph")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.tight_layout()
        fig.savefig(self.plot_dir / filename, dpi=160)
        plt.close(fig)

    def _scatter_feature(self, df: pd.DataFrame, feature: str, filename: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 7))
        if df.empty or feature not in df.columns:
            ax.text(0.5, 0.5, f"{feature} unavailable", ha="center")
        else:
            sc = ax.scatter(df["lon"], df["lat"], c=df[feature], s=14, alpha=0.8)
            fig.colorbar(sc, ax=ax, shrink=0.7)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.tight_layout()
        fig.savefig(self.plot_dir / filename, dpi=160)
        plt.close(fig)

    def _scatter_categorical(self, df: pd.DataFrame, column: str, filename: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 7))
        if df.empty or column not in df.columns:
            ax.text(0.5, 0.5, f"{column} unavailable", ha="center")
        else:
            codes = pd.Categorical(df[column]).codes
            sc = ax.scatter(df["lon"], df["lat"], c=codes, s=12, alpha=0.8)
            labels = list(pd.Categorical(df[column]).categories)
            # Compact legend for readability.
            handles = [plt.Line2D([], [], marker='o', linestyle='', label=lab) for lab in labels[:12]]
            ax.legend(handles=handles, fontsize=7, loc="best")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.tight_layout()
        fig.savefig(self.plot_dir / filename, dpi=160)
        plt.close(fig)

    def _candidate_map(self, nodes: pd.DataFrame, candidates: pd.DataFrame, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(nodes["lon"], nodes["lat"], s=4, alpha=0.25)
        if not candidates.empty:
            ax.scatter(candidates["lon"], candidates["lat"], s=30, alpha=0.85)
        ax.set_title("Candidate sites")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.tight_layout()
        fig.savefig(self.plot_dir / filename, dpi=160)
        plt.close(fig)

    def _demand_map(self, demand: pd.DataFrame, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 7))
        if not demand.empty:
            sc = ax.scatter(demand["lon"], demand["lat"], c=demand["demand_mass"], s=20, alpha=0.85)
            fig.colorbar(sc, ax=ax, shrink=0.7)
        else:
            ax.text(0.5, 0.5, "No demand points", ha="center")
        ax.set_title("Demand points")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.tight_layout()
        fig.savefig(self.plot_dir / filename, dpi=160)
        plt.close(fig)

    def _kpi_distribution(self, kpis: pd.DataFrame, filename: str) -> None:
        features = [c for c in ["demand_capture", "coverage_gain", "accessibility_benefit", "grid_feasibility", "land_feasibility", "cost_efficiency", "equity_benefit", "competition_penalty"] if c in kpis.columns]
        self._multi_hist(kpis, features, filename, "Candidate KPI distributions")

    def _feature_histograms(self, matrix: pd.DataFrame, filename: str) -> None:
        features = [c for c in ["population_density", "activity_intensity", "traffic_flow_proxy", "grid_access_score", "available_space", "nearby_charger_pressure", "equity_need_score", "demand_capture", "coverage_gain"] if c in matrix.columns]
        self._multi_hist(matrix, features, filename, "Feature histograms")

    def _multi_hist(self, df: pd.DataFrame, features: List[str], filename: str, title: str) -> None:
        n = max(len(features), 1)
        cols = 3
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
        axes = np.array(axes).reshape(-1)
        for ax, feature in zip(axes, features):
            ax.hist(pd.to_numeric(df[feature], errors="coerce").dropna(), bins=20)
            ax.set_title(feature, fontsize=9)
        for ax in axes[len(features):]:
            ax.axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(self.plot_dir / filename, dpi=160)
        plt.close(fig)

    def _correlation_heatmap(self, matrix: pd.DataFrame, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(9, 8))
        numeric = matrix.select_dtypes(include="number")
        if numeric.empty:
            ax.text(0.5, 0.5, "No numeric features", ha="center")
        else:
            corr = numeric.corr().fillna(0)
            im = ax.imshow(corr.values, vmin=-1, vmax=1)
            fig.colorbar(im, ax=ax, shrink=0.7)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
            ax.set_yticklabels(corr.columns, fontsize=6)
        ax.set_title("Feature correlation heatmap")
        fig.tight_layout()
        fig.savefig(self.plot_dir / filename, dpi=160)
        plt.close(fig)

    def _boxplots_by_land_use(self, matrix: pd.DataFrame, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        if matrix.empty or "land_use_type" not in matrix.columns or "demand_capture" not in matrix.columns:
            ax.text(0.5, 0.5, "Land-use boxplot unavailable", ha="center")
        else:
            groups = [g["demand_capture"].dropna().values for _, g in matrix.groupby("land_use_type")]
            labels = [str(k) for k, _ in matrix.groupby("land_use_type")]
            ax.boxplot(groups, labels=labels, vert=True)
            ax.tick_params(axis="x", rotation=45)
        ax.set_title("Demand capture by land use")
        fig.tight_layout()
        fig.savefig(self.plot_dir / filename, dpi=160)
        plt.close(fig)

    def _bar(self, df: pd.DataFrame, column: str, filename: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        if df.empty or column not in df.columns:
            ax.text(0.5, 0.5, f"{column} unavailable", ha="center")
        else:
            counts = df[column].value_counts()
            ax.bar(counts.index.astype(str), counts.values)
            ax.tick_params(axis="x", rotation=45)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(self.plot_dir / filename, dpi=160)
        plt.close(fig)

    def _provenance_breakdown(self, provenance: pd.DataFrame, filename: str) -> None:
        self._bar(provenance, "source_type", filename, "Feature provenance breakdown")

    def _dashboard(self, metrics: pd.DataFrame, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        if metrics.empty:
            ax.text(0.5, 0.5, "Validation metrics unavailable", ha="center")
        else:
            display = metrics.copy()
            if "metric" in display.columns and "value" in display.columns:
                subset = display[display["value"].apply(lambda x: isinstance(x, (int, float, np.integer, np.floating)))].head(12)
                ax.barh(subset["metric"].astype(str), subset["value"].astype(float))
            else:
                ax.text(0.5, 0.5, "Validation metrics exported as CSV", ha="center")
        ax.set_title("Dataset validation dashboard")
        fig.tight_layout()
        fig.savefig(self.plot_dir / filename, dpi=160)
        plt.close(fig)

    def build_city_comparisons(self, run_dirs: List[Path], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summaries = []
        for run_dir in run_dirs:
            p = run_dir / "csv" / "simulation_summary.csv"
            if p.exists():
                summaries.append(pd.read_csv(p))
        if not summaries:
            return
        df = pd.concat(summaries, ignore_index=True)
        for metric, fname in [
            ("node_count", "city_comparison_node_counts.png"),
            ("candidate_count", "city_comparison_candidate_counts.png"),
            ("mean_feature_confidence", "city_comparison_data_quality.png"),
            ("mean_uncertainty_score", "city_comparison_uncertainty.png"),
        ]:
            if metric in df.columns:
                fig, ax = plt.subplots(figsize=(12, 5))
                labels = df["city_id"].astype(str) + "_" + df["data_mode"].astype(str)
                ax.bar(labels, df[metric].astype(float))
                ax.tick_params(axis="x", rotation=80)
                ax.set_title(metric)
                fig.tight_layout()
                fig.savefig(output_dir / fname, dpi=160)
                plt.close(fig)
