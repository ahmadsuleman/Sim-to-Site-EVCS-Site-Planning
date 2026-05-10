from __future__ import annotations

from typing import Any, Dict, Tuple
import pandas as pd
import networkx as nx

from src.validation.graph_validation import GraphValidator
from src.validation.feature_validation import FeatureValidator
from src.validation.spatial_validation import SpatialValidator
from src.validation.kpi_validation import KPIValidator
from src.validation.provenance_validation import ProvenanceValidator


class PipelineValidator:
    def validate(self, G: nx.Graph, nodes: pd.DataFrame, edges: pd.DataFrame, candidates: pd.DataFrame,
                 demand: pd.DataFrame, matrix: pd.DataFrame, provenance: pd.DataFrame,
                 randomization_log: pd.DataFrame) -> Dict[str, Any]:
        graph = GraphValidator().validate(G, nodes, edges, candidates, demand)
        feature_quality = FeatureValidator().validate(matrix)
        kpi_validator = KPIValidator()
        corr_matrix = kpi_validator.correlation_matrix(matrix)
        corr_report = kpi_validator.correlation_report(matrix)
        spatial = SpatialValidator().neighbor_autocorrelation(G, nodes, [
            "population_density", "activity_intensity", "traffic_flow_proxy", "nearby_charger_pressure", "service_coverage_gain"
        ])
        provenance_summary = ProvenanceValidator().summarize(provenance, randomization_log)
        return {
            "graph_metrics": graph,
            "feature_quality": feature_quality,
            "correlation_matrix": corr_matrix,
            "correlation_report": corr_report,
            "spatial_autocorrelation": spatial,
            "provenance_summary": provenance_summary,
        }


def metrics_dict_to_frame(metrics: Dict[str, Any], extra: Dict[str, Any] | None = None) -> pd.DataFrame:
    rows = []
    extra = extra or {}
    for k, v in {**extra, **metrics}.items():
        rows.append({"metric": k, "value": v})
    return pd.DataFrame(rows)
