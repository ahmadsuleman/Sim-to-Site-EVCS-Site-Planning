from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import pandas as pd


@dataclass
class FeatureValue:
    value: float
    source_type: str
    source_name: str
    confidence_score: float
    was_randomized: bool


class ProvenanceTracker:
    def __init__(self, city_id: str, data_mode: str):
        self.city_id = city_id
        self.data_mode = data_mode
        self.records: List[Dict[str, Any]] = []

    def add(self, entity_type: str, entity_id: str, feature_name: str, value: Any,
            source_type: str, source_name: str, confidence_score: float,
            was_randomized: bool) -> None:
        confidence = float(confidence_score)
        self.records.append({
            "city_id": self.city_id,
            "data_mode": self.data_mode,
            "entity_type": entity_type,
            "entity_id": str(entity_id),
            "feature_name": feature_name,
            "value": value,
            "source_type": source_type,
            "source_name": source_name,
            "confidence_score": confidence,
            "was_randomized": bool(was_randomized),
            "uncertainty_score": 1.0 - confidence,
        })

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def feature_confidence_mean(self, entity_type: str, entity_id: str) -> float:
        df = self.dataframe()
        if df.empty:
            return 0.0
        sub = df[(df["entity_type"] == entity_type) & (df["entity_id"] == str(entity_id))]
        if sub.empty:
            return 0.0
        return float(sub["confidence_score"].mean())
