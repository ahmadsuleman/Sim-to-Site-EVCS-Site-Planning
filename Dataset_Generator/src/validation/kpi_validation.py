from __future__ import annotations

from typing import Dict, List
import pandas as pd
from src.common import safe_corr


EXPECTED_CORRELATIONS = [
    ("activity_intensity", "traffic_flow_proxy", "positive"),
    ("population_density", "od_outflow_proxy", "positive"),
    ("activity_intensity", "od_inflow_proxy", "positive"),
    ("road_hierarchy_score", "traffic_flow_proxy", "positive"),
    ("nearby_charger_pressure", "coverage_gain", "negative_or_weak"),
    ("available_space", "land_cost_proxy", "weak_or_city_dependent"),
    ("reachability_10min", "demand_capture", "positive"),
]


class KPIValidator:
    def correlation_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        numeric = matrix.select_dtypes(include="number")
        return numeric.corr()

    def correlation_report(self, matrix: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for a, b, expectation in EXPECTED_CORRELATIONS:
            if a not in matrix.columns or b not in matrix.columns:
                rows.append({"feature_a": a, "feature_b": b, "expectation": expectation, "correlation": None, "pass": False})
                continue
            corr = safe_corr(matrix[a], matrix[b])
            if expectation == "positive":
                ok = corr >= 0.05 if pd.notna(corr) else False
            elif expectation == "negative_or_weak":
                ok = corr <= 0.20 if pd.notna(corr) else False
            else:
                ok = True if pd.notna(corr) else False
            rows.append({"feature_a": a, "feature_b": b, "expectation": expectation, "correlation": corr, "pass": bool(ok)})
        return pd.DataFrame(rows)
