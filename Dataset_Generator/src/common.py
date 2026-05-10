from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml

NORMALIZED_FEATURES = [
    "degree_norm",
    "betweenness_centrality",
    "closeness_centrality",
    "road_hierarchy_score",
    "reachability_5min",
    "reachability_10min",
    "reachability_15min",
    "population_density",
    "activity_intensity",
    "traffic_flow_proxy",
    "od_inflow_proxy",
    "od_outflow_proxy",
    "parking_dwell_score",
    "grid_access_score",
    "land_cost_proxy",
    "available_space",
    "nearby_charger_pressure",
    "equity_need_score",
    "service_coverage_gain",
    "demand_capture",
    "coverage_gain",
    "accessibility_benefit",
    "grid_feasibility",
    "land_feasibility",
    "cost_efficiency",
    "equity_benefit",
    "competition_penalty",
    "feature_confidence_mean",
    "uncertainty_score",
]

NODE_SCORE_FEATURES = [
    "population_density",
    "activity_intensity",
    "traffic_flow_proxy",
    "od_inflow_proxy",
    "od_outflow_proxy",
    "parking_dwell_score",
    "grid_access_score",
    "land_cost_proxy",
    "available_space",
    "nearby_charger_pressure",
    "equity_need_score",
    "service_coverage_gain",
]

ROAD_TYPE_SCORES = {
    "motorway": 1.00,
    "trunk": 0.95,
    "primary": 0.85,
    "secondary": 0.70,
    "tertiary": 0.55,
    "residential": 0.35,
    "service": 0.25,
    "unclassified": 0.30,
}

ROAD_TYPE_SPEEDS = {
    "motorway": 100,
    "trunk": 90,
    "primary": 75,
    "secondary": 60,
    "tertiary": 45,
    "residential": 30,
    "service": 20,
    "unclassified": 35,
}

LAND_USE_TYPES = [
    "commercial",
    "residential",
    "industrial",
    "transit",
    "tourism",
    "religious",
    "mixed",
    "public",
    "highway",
    "open_space",
]

ARCHETYPE_LAND_USE_WEIGHTS = {
    "capital_inland":        [0.20, 0.35, 0.08, 0.07, 0.04, 0.04, 0.12, 0.05, 0.03, 0.02],
    "coastal_port":          [0.18, 0.30, 0.12, 0.06, 0.08, 0.03, 0.12, 0.04, 0.04, 0.03],
    "religious_tourism":     [0.18, 0.30, 0.04, 0.08, 0.10, 0.14, 0.10, 0.03, 0.02, 0.01],
    "coastal_industrial":    [0.14, 0.28, 0.20, 0.06, 0.05, 0.03, 0.10, 0.04, 0.07, 0.03],
    "coastal_commercial":    [0.24, 0.30, 0.08, 0.07, 0.08, 0.03, 0.12, 0.04, 0.02, 0.02],
    "coastal_capital":       [0.20, 0.32, 0.08, 0.08, 0.08, 0.03, 0.12, 0.05, 0.02, 0.02],
    "coastal_tourism":       [0.17, 0.28, 0.06, 0.06, 0.16, 0.04, 0.12, 0.04, 0.03, 0.04],
    "inland_medium_density": [0.15, 0.38, 0.06, 0.04, 0.05, 0.06, 0.12, 0.05, 0.03, 0.06],
}

@dataclass
class CityConfig:
    city_id: str
    name: str
    country: str
    osm_place: str
    lat: float
    lon: float
    bbox_km: float
    city_archetype: str
    mode_enabled: List[str]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config() -> Dict[str, Any]:
    root = project_root()
    return {
        "cities": load_yaml(root / "config" / "cities.yaml"),
        "feature_schema": load_yaml(root / "config" / "feature_schema.yaml"),
        "domain_randomization": load_yaml(root / "config" / "domain_randomization.yaml"),
        "poi_weights": load_yaml(root / "config" / "poi_weights.yaml"),
        "validation_thresholds": load_yaml(root / "config" / "validation_thresholds.yaml"),
        "pipeline_config": load_yaml(root / "config" / "pipeline_config.yaml"),
    }


def get_city(config: Dict[str, Any], city_id: str) -> CityConfig:
    for item in config["cities"]["cities"]:
        if item["city_id"] == city_id:
            return CityConfig(**item)
    available = ", ".join(c["city_id"] for c in config["cities"]["cities"])
    raise ValueError(f"Unknown city_id '{city_id}'. Available: {available}")


def list_cities(config: Dict[str, Any]) -> List[CityConfig]:
    return [CityConfig(**item) for item in config["cities"]["cities"]]


def ensure_run_dirs(output_dir: Path) -> Dict[str, Path]:
    dirs = {
        "root": output_dir,
        "csv": output_dir / "csv",
        "plots": output_dir / "plots",
        "graph": output_dir / "graph",
        "reports": output_dir / "reports",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def stable_rng(seed: int, *parts: Any) -> np.random.Generator:
    text = "|".join(str(p) for p in (seed, *parts))
    value = abs(hash(text)) % (2**32)
    return np.random.default_rng(value)


def clamp01(x: Any) -> Any:
    return np.clip(x, 0.0, 1.0)


def normalize_series(s: pd.Series, fill: float = 0.0) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.isna().all():
        return pd.Series(fill, index=s.index, dtype=float)
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or abs(mx - mn) < 1e-12:
        return pd.Series(fill, index=s.index, dtype=float)
    return ((s - mn) / (mx - mn)).fillna(fill).clip(0, 1)


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 3 or df["a"].std() < 1e-12 or df["b"].std() < 1e-12:
        return float("nan")
    return float(df["a"].corr(df["b"]))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return 2 * r * math.asin(math.sqrt(a))


def xy_from_latlon(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    x = haversine_m(lat0, lon0, lat0, lon)
    if lon < lon0:
        x *= -1
    y = haversine_m(lat0, lon0, lat, lon0)
    if lat < lat0:
        y *= -1
    return x, y


def line_wkt(points: Iterable[Tuple[float, float]]) -> str:
    coords = ", ".join(f"{lon:.7f} {lat:.7f}" for lat, lon in points)
    return f"LINESTRING ({coords})"


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    def default(o):
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")
