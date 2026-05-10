from __future__ import annotations

from typing import Any, Dict
import pandas as pd
from src.common import CityConfig
from src.demand.demand_mass import compute_demand_mass


class DemandNodeGenerator:
    def __init__(self, city: CityConfig, data_mode: str, config: Dict[str, Any]):
        self.city = city
        self.data_mode = data_mode
        self.defaults = config["pipeline_config"]["defaults"]

    def build(self, nodes: pd.DataFrame) -> pd.DataFrame:
        df = nodes.copy()
        df["demand_mass"] = compute_demand_mass(df)
        threshold = float(df["demand_mass"].quantile(float(self.defaults.get("demand_quantile_threshold", 0.60))))
        demand = df[df["demand_mass"] >= threshold].copy()
        return pd.DataFrame({
            "demand_id": [f"dem_{self.city.city_id}_{self.data_mode}_{i:04d}" for i in range(len(demand))],
            "city_id": self.city.city_id,
            "data_mode": self.data_mode,
            "node_id": demand["node_id"].values,
            "lat": demand["lat"].values,
            "lon": demand["lon"].values,
            "x": demand["x"].values,
            "y": demand["y"].values,
            "population_density": demand["population_density"].values,
            "activity_intensity": demand["activity_intensity"].values,
            "od_inflow_proxy": demand["od_inflow_proxy"].values,
            "od_outflow_proxy": demand["od_outflow_proxy"].values,
            "demand_mass": demand["demand_mass"].values,
        })
