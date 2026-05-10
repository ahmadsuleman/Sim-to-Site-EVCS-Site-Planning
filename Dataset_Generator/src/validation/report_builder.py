from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import pandas as pd

from src.common import CityConfig, write_json


class ReportBuilder:
    def build_reports(
        self,
        report_dir: Path,
        city: CityConfig,
        data_mode: str,
        seed: int,
        graph_meta: Dict[str, str],
        validation_metrics: Dict[str, Any],
        provenance_summary: Dict[str, Any],
        feature_quality: pd.DataFrame,
        correlation_report: pd.DataFrame,
    ) -> None:
        report_dir.mkdir(parents=True, exist_ok=True)
        validation_report = {
            "city_id": city.city_id,
            "city_name": city.name,
            "country": city.country,
            "data_mode": data_mode,
            "seed": seed,
            "graph_metadata": graph_meta,
            "graph_validation": validation_metrics,
            "provenance_summary": provenance_summary,
            "feature_quality_pass_rate": float(feature_quality["range_check_pass"].mean()) if not feature_quality.empty else 0.0,
            "correlation_checks": correlation_report.to_dict(orient="records") if not correlation_report.empty else [],
        }
        write_json(report_dir / "validation_report.json", validation_report)
        (report_dir / "run_summary.md").write_text(self._run_summary(city, data_mode, seed, graph_meta, validation_metrics, provenance_summary), encoding="utf-8")
        (report_dir / "dataset_card.md").write_text(self._dataset_card(city, data_mode, graph_meta, provenance_summary), encoding="utf-8")

    def _run_summary(self, city: CityConfig, data_mode: str, seed: int, graph_meta: Dict[str, str], metrics: Dict[str, Any], prov: Dict[str, Any]) -> str:
        return f"""# Run summary

City: {city.name} ({city.city_id})  
Country: {city.country}  
Mode: {data_mode}  
Seed: {seed}

## Graph source

- Source: {graph_meta.get('graph_source', 'unknown')}
- Fallback used: {graph_meta.get('fallback_used', 'false')}
- Fallback reason: {graph_meta.get('fallback_reason', 'none')}

## Dataset sizes

- Nodes: {metrics.get('node_count', 0)}
- Edges: {metrics.get('edge_count', 0)}
- Candidates: {metrics.get('candidate_count', 0)}
- Demand nodes: {metrics.get('demand_node_count', 0)}

## Provenance summary

- Real features: {prov.get('real_feature_percent', 0):.2f}%
- Proxy features: {prov.get('proxy_feature_percent', 0):.2f}%
- Randomized features: {prov.get('randomized_feature_percent', 0):.2f}%
- Mean feature confidence: {prov.get('mean_feature_confidence', 0):.3f}
- Mean uncertainty score: {prov.get('mean_uncertainty_score', 0):.3f}

## Important limitation

This dataset is intended for training, simulation, benchmarking, and downstream optimization. It does not provide final EV station placement recommendations.
"""

    def _dataset_card(self, city: CityConfig, data_mode: str, graph_meta: Dict[str, str], prov: Dict[str, Any]) -> str:
        return f"""# Dataset card

## Dataset purpose

This dataset provides graph-structured EV charging infrastructure features and candidate KPIs for research, training, simulation, benchmarking, and downstream optimization.

## City included

- {city.name}, {city.country} ({city.city_id})

## Data mode generated

- {data_mode}

## Data sources used

- Road graph source: {graph_meta.get('graph_source', 'unknown')}
- Feature source mix: real {prov.get('real_feature_percent', 0):.2f}%, proxy {prov.get('proxy_feature_percent', 0):.2f}%, randomized {prov.get('randomized_feature_percent', 0):.2f}%

## Feature schema

The run exports stable node, edge, candidate, demand, KPI, provenance, quality, validation, and ML-ready candidate feature matrix CSV files.

## Missing data handling

When direct real data is unavailable, the pipeline computes proxy values where possible. If a feature is weakly observable, it uses domain randomization and records the sampled value, bounds, confidence, seed, and reason.

## Known limitations

- Equity features are proxy indicators and should not be interpreted as true socioeconomic equity unless official socioeconomic microdata is added.
- Hybrid and real runs may use synthetic fallback when OSMnx, internet access, or local source files are unavailable.
- Candidate KPIs are labels and explanatory features. They are not final siting recommendations.

## Recommended downstream use

- GNN, VGAE, ranking, optimization, simulation, and ablation studies.
- Benchmarking feature robustness across cities and data modes.

## Not recommended use

- Do not treat candidate rows as approved final EV charging station sites.
- Do not use the outputs for policy decisions without local validation and authoritative infrastructure data.

## Required limitation statement

This dataset is intended for training, simulation, benchmarking, and downstream optimization. It does not provide final EV station placement recommendations.
"""
