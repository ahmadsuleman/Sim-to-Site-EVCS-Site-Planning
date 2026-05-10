from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd

from src.candidates.candidate_filters import enforce_candidate_bounds
from src.common import CityConfig, clamp01, stable_rng
from src.features.provenance import ProvenanceTracker


SOURCE_BY_LAND_USE = {
    "commercial": ["parking", "mall", "retail_center", "large_commercial_poi"],
    "residential": ["parking", "public_building", "synthetic_fallback_candidate"],
    "industrial": ["industrial_area", "fuel_station", "parking"],
    "transit": ["transit_station", "parking", "public_building"],
    "tourism": ["hotel", "tourism_poi", "parking"],
    "religious": ["public_building", "parking", "large_commercial_poi"],
    "mixed": ["parking", "retail_center", "public_building"],
    "public": ["public_building", "parking"],
    "highway": ["highway_service_area", "fuel_station"],
    "open_space": ["synthetic_fallback_candidate", "parking"],
}


class CandidateGenerator:
    def __init__(self, city: CityConfig, data_mode: str, seed: int, config: Dict[str, Any]):
        self.city = city
        self.data_mode = data_mode
        self.seed = seed
        self.defaults = config["pipeline_config"]["defaults"]
        self.rng = stable_rng(seed, city.city_id, data_mode, "candidates")

    def build(self, nodes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = nodes.copy()
        low_existing_charger_pressure = 1 - df["nearby_charger_pressure"]
        road_accessibility = 0.55 * df["road_hierarchy_score"] + 0.45 * df["reachability_10min"]
        df["candidate_score_for_inclusion"] = clamp01(
            0.25 * df["available_space"]
            + 0.20 * road_accessibility
            + 0.20 * df["activity_intensity"]
            + 0.15 * df["grid_access_score"]
            + 0.10 * df["parking_dwell_score"]
            + 0.10 * low_existing_charger_pressure
        )
        threshold = float(df["candidate_score_for_inclusion"].quantile(float(self.defaults.get("candidate_quantile_threshold", 0.72))))
        candidates = df[df["candidate_score_for_inclusion"] >= threshold].copy()
        min_count = int(self.defaults.get("synthetic_candidate_min", 35))
        max_count = int(self.defaults.get("synthetic_candidate_max", 140))
        candidates = enforce_candidate_bounds(candidates, "candidate_score_for_inclusion", min_count, max_count)
        records = []
        prov_rows = []
        for i, row in candidates.iterrows():
            land = str(row["land_use_type"])
            source_options = SOURCE_BY_LAND_USE.get(land, ["synthetic_fallback_candidate"])
            source = str(self.rng.choice(source_options))
            candidate_id = f"cand_{self.city.city_id}_{self.data_mode}_{len(records):04d}"
            confidence = float(row.get("feature_confidence_mean", 0.5))
            uncertainty = 1.0 - confidence
            records.append({
                "candidate_id": candidate_id,
                "city_id": self.city.city_id,
                "data_mode": self.data_mode,
                "node_id": row["node_id"],
                "lat": row["lat"],
                "lon": row["lon"],
                "candidate_source": source,
                "land_use_type": land,
                "available_space": row["available_space"],
                "parking_dwell_score": row["parking_dwell_score"],
                "grid_access_score": row["grid_access_score"],
                "land_cost_proxy": row["land_cost_proxy"],
                "nearby_charger_pressure": row["nearby_charger_pressure"],
                "candidate_score_for_inclusion": row["candidate_score_for_inclusion"],
                "candidate_confidence": confidence,
                "candidate_uncertainty": uncertainty,
            })
            for feature in ["candidate_source", "candidate_score_for_inclusion", "candidate_confidence", "candidate_uncertainty"]:
                prov_rows.append({
                    "city_id": self.city.city_id,
                    "data_mode": self.data_mode,
                    "entity_type": "candidate",
                    "entity_id": candidate_id,
                    "feature_name": feature,
                    "value": source if feature == "candidate_source" else records[-1][feature],
                    "source_type": "proxy",
                    "source_name": "candidate_inclusion_rule",
                    "confidence_score": confidence,
                    "was_randomized": False,
                    "uncertainty_score": uncertainty,
                })
        return pd.DataFrame(records), pd.DataFrame(prov_rows)
