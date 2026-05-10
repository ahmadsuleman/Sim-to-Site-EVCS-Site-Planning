from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd

from src.common import CityConfig, clamp01, normalize_series


class CandidateKPIEngine:
    """Exports candidate KPIs as learning targets or explanatory labels, not placement decisions."""

    def __init__(self, city: CityConfig, data_mode: str):
        self.city = city
        self.data_mode = data_mode

    def build(self, candidates: pd.DataFrame, nodes: pd.DataFrame, demand_points: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            return pd.DataFrame()
        node_lookup = nodes.set_index("node_id")
        rows = []
        for _, cand in candidates.iterrows():
            crow = node_lookup.loc[cand["node_id"]]
            demand_capture_raw = self._decayed_sum(cand, demand_points, "demand_mass", radius_m=5000)
            unmet = demand_points.copy()
            # Use a city-wide pressure proxy when a precise node mapping is not available.
            unmet["unmet_demand_mass"] = unmet["demand_mass"] * (1 - float(cand["nearby_charger_pressure"]))
            coverage_gain_raw = self._decayed_sum(cand, unmet, "unmet_demand_mass", radius_m=5000)
            accessibility_benefit = float(clamp01(
                0.45 * crow["reachability_10min"]
                + 0.30 * crow["road_hierarchy_score"]
                + 0.25 * crow["betweenness_centrality"]
            ))
            grid_feasibility = float(cand["grid_access_score"])
            land_feasibility = float(clamp01(0.60 * cand["available_space"] + 0.40 * cand["parking_dwell_score"]))
            cost_efficiency = float(clamp01(1 - cand["land_cost_proxy"]))
            equity_benefit = float(crow["equity_need_score"])
            competition_penalty = float(cand["nearby_charger_pressure"])
            conf = float(cand.get("candidate_confidence", crow.get("feature_confidence_mean", 0.5)))
            rows.append({
                "candidate_id": cand["candidate_id"],
                "city_id": self.city.city_id,
                "data_mode": self.data_mode,
                "demand_capture_raw": demand_capture_raw,
                "coverage_gain_raw": coverage_gain_raw,
                "accessibility_benefit": accessibility_benefit,
                "grid_feasibility": grid_feasibility,
                "land_feasibility": land_feasibility,
                "cost_efficiency": cost_efficiency,
                "equity_benefit": equity_benefit,
                "competition_penalty": competition_penalty,
                "service_coverage_gain": float(crow["service_coverage_gain"]),
                "kpi_confidence_mean": conf,
                "kpi_uncertainty_score": 1.0 - conf,
            })
        df = pd.DataFrame(rows)
        df["demand_capture"] = normalize_series(df["demand_capture_raw"])
        df["coverage_gain"] = normalize_series(df["coverage_gain_raw"])
        order = [
            "candidate_id", "city_id", "data_mode", "demand_capture", "coverage_gain", "accessibility_benefit",
            "grid_feasibility", "land_feasibility", "cost_efficiency", "equity_benefit", "competition_penalty",
            "service_coverage_gain", "kpi_confidence_mean", "kpi_uncertainty_score"
        ]
        return df[order]

    def _decayed_sum(self, cand: pd.Series, demand: pd.DataFrame, value_col: str, radius_m: float) -> float:
        if demand.empty:
            return 0.0
        dx = demand["x"].values - float(cand.get("x", 0.0) if "x" in cand else 0.0)
        dy = demand["y"].values - float(cand.get("y", 0.0) if "y" in cand else 0.0)
        # Candidate table does not carry x/y by default. Fallback to lat/lon pseudo-distance if needed.
        if np.allclose(dx, demand["x"].values) and np.allclose(dy, demand["y"].values):
            dx = (demand["lon"].values - float(cand["lon"])) * 111000.0
            dy = (demand["lat"].values - float(cand["lat"])) * 111000.0
        dist = np.sqrt(dx * dx + dy * dy)
        mask = dist <= radius_m
        if not mask.any():
            return 0.0
        decay = 1.0 / (1.0 + dist[mask] / 1000.0)
        return float(np.sum(demand.loc[mask, value_col].values * decay))
