from __future__ import annotations

from typing import Any, Dict, List, Tuple
import pandas as pd

from src.common import clamp01, stable_rng


class DomainRandomizer:
    """Samples weakly observable features and logs every sample."""

    def __init__(self, rules: Dict[str, Any], seed: int, city_id: str, data_mode: str):
        self.rules = rules
        self.seed = seed
        self.city_id = city_id
        self.data_mode = data_mode
        self.records: List[Dict[str, Any]] = []

    def sample(
        self,
        feature_name: str,
        land_use_type: str,
        entity_id: str,
        reason: str,
        default_low: float = 0.0,
        default_high: float = 1.0,
        default_confidence: float = 0.30,
    ) -> Tuple[float, float]:
        feature_rules = self.rules.get(feature_name, {})
        bounds = feature_rules.get(land_use_type) or feature_rules.get("mixed") or {}
        low = float(bounds.get("low", default_low))
        high = float(bounds.get("high", default_high))
        confidence = float(bounds.get("confidence", default_confidence))
        rng = stable_rng(self.seed, self.city_id, self.data_mode, entity_id, feature_name)
        value = float(rng.uniform(low, high))
        value = float(clamp01(value))
        self.records.append({
            "city_id": self.city_id,
            "data_mode": self.data_mode,
            "entity_id": entity_id,
            "feature_name": feature_name,
            "low_bound": low,
            "high_bound": high,
            "sampled_value": value,
            "random_seed": self.seed,
            "randomization_reason": reason,
            "confidence_score": confidence,
        })
        return value, confidence

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)
