from __future__ import annotations

from typing import Dict, Any
import pandas as pd


class ProvenanceValidator:
    def summarize(self, provenance: pd.DataFrame, randomization_log: pd.DataFrame) -> Dict[str, Any]:
        if provenance.empty:
            return {}
        total = len(provenance)
        source_counts = provenance["source_type"].value_counts().to_dict()
        return {
            "real_feature_percent": 100.0 * source_counts.get("real", 0) / total,
            "proxy_feature_percent": 100.0 * source_counts.get("proxy", 0) / total,
            "randomized_feature_percent": 100.0 * (source_counts.get("randomized", 0) + provenance["was_randomized"].sum()) / max(total, 1),
            "synthetic_feature_percent": 100.0 * source_counts.get("synthetic", 0) / total,
            "mean_feature_confidence": float(provenance["confidence_score"].mean()),
            "mean_uncertainty_score": float(provenance["uncertainty_score"].mean()),
            "randomized_features_by_feature_name": randomization_log.get("feature_name", pd.Series(dtype=str)).value_counts().to_dict() if not randomization_log.empty else {},
        }
