from __future__ import annotations

from typing import Dict, Tuple
import math

import networkx as nx
import numpy as np
import pandas as pd

from src.common import (
    ARCHETYPE_LAND_USE_WEIGHTS,
    LAND_USE_TYPES,
    ROAD_TYPE_SCORES,
    ROAD_TYPE_SPEEDS,
    CityConfig,
    haversine_m,
    line_wkt,
    stable_rng,
    xy_from_latlon,
)


class SyntheticGraphBuilder:
    """Creates a city-like road graph with reproducible spatial structure."""

    def __init__(self, city: CityConfig, seed: int, node_count: int = 324):
        self.city = city
        self.seed = seed
        self.node_count = node_count
        self.rng = stable_rng(seed, city.city_id, "synthetic_graph")

    def build(self, data_mode: str = "synthetic") -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
        n_side = int(math.ceil(math.sqrt(self.node_count)))
        total_nodes = n_side * n_side
        spacing_km = self.city.bbox_km / max(n_side - 1, 1)
        deg_lat_per_km = 1 / 111.0
        deg_lon_per_km = 1 / (111.0 * max(math.cos(math.radians(self.city.lat)), 0.2))
        half = self.city.bbox_km / 2

        G = nx.Graph()
        land_weights = ARCHETYPE_LAND_USE_WEIGHTS.get(self.city.city_archetype, ARCHETYPE_LAND_USE_WEIGHTS["capital_inland"])
        node_records = []

        for i in range(n_side):
            for j in range(n_side):
                node_id = f"n_{i}_{j}"
                dx_km = -half + j * spacing_km + self.rng.normal(0, spacing_km * 0.08)
                dy_km = -half + i * spacing_km + self.rng.normal(0, spacing_km * 0.08)
                lat = self.city.lat + dy_km * deg_lat_per_km
                lon = self.city.lon + dx_km * deg_lon_per_km
                x, y = xy_from_latlon(lat, lon, self.city.lat, self.city.lon)

                radial = math.sqrt((dx_km / max(half, 1))**2 + (dy_km / max(half, 1))**2)
                adjusted = np.array(land_weights, dtype=float)
                if radial < 0.30:
                    adjusted[LAND_USE_TYPES.index("commercial")] += 0.12
                    adjusted[LAND_USE_TYPES.index("mixed")] += 0.08
                if radial > 0.75:
                    adjusted[LAND_USE_TYPES.index("residential")] += 0.08
                    adjusted[LAND_USE_TYPES.index("industrial")] += 0.04
                    adjusted[LAND_USE_TYPES.index("open_space")] += 0.04
                adjusted = adjusted / adjusted.sum()
                land_use = str(self.rng.choice(LAND_USE_TYPES, p=adjusted))
                G.add_node(node_id, lat=lat, lon=lon, x=x, y=y, node_type="intersection", land_use_type=land_use)
                node_records.append({
                    "node_id": node_id,
                    "city_id": self.city.city_id,
                    "country": self.city.country,
                    "data_mode": data_mode,
                    "lat": lat,
                    "lon": lon,
                    "x": x,
                    "y": y,
                    "node_type": "intersection",
                    "land_use_type": land_use,
                })

        def road_type_for(i: int, j: int, orientation: str) -> str:
            centerline = abs(i - n_side // 2) <= 1 or abs(j - n_side // 2) <= 1
            ring = i in (1, n_side - 2) or j in (1, n_side - 2)
            if centerline and self.rng.random() < 0.65:
                return str(self.rng.choice(["primary", "secondary", "trunk"], p=[0.45, 0.35, 0.20]))
            if ring and self.rng.random() < 0.45:
                return str(self.rng.choice(["secondary", "tertiary", "primary"], p=[0.45, 0.35, 0.20]))
            return str(self.rng.choice(["residential", "tertiary", "service", "unclassified"], p=[0.50, 0.25, 0.15, 0.10]))

        edge_records = []
        edge_idx = 0
        for i in range(n_side):
            for j in range(n_side):
                u = f"n_{i}_{j}"
                for ni, nj, orient in [(i + 1, j, "v"), (i, j + 1, "h")]:
                    if ni < n_side and nj < n_side:
                        v = f"n_{ni}_{nj}"
                        rt = road_type_for(i, j, orient)
                        length_m = haversine_m(G.nodes[u]["lat"], G.nodes[u]["lon"], G.nodes[v]["lat"], G.nodes[v]["lon"])
                        speed = ROAD_TYPE_SPEEDS[rt]
                        travel_time_min = length_m / 1000.0 / speed * 60.0
                        score = ROAD_TYPE_SCORES[rt]
                        G.add_edge(u, v, length_m=length_m, travel_time_min=travel_time_min, road_type=rt,
                                   road_hierarchy_score=score, speed_kph=speed, capacity_score=score)
                        edge_records.append(self._edge_record(edge_idx, u, v, data_mode, length_m, travel_time_min, rt, score, speed))
                        edge_idx += 1

                # Sparse diagonal links and highways create more realistic shortest paths.
                if i + 1 < n_side and j + 1 < n_side and self.rng.random() < 0.10:
                    v = f"n_{i+1}_{j+1}"
                    rt = str(self.rng.choice(["tertiary", "secondary", "primary"], p=[0.55, 0.35, 0.10]))
                    length_m = haversine_m(G.nodes[u]["lat"], G.nodes[u]["lon"], G.nodes[v]["lat"], G.nodes[v]["lon"])
                    speed = ROAD_TYPE_SPEEDS[rt]
                    travel_time_min = length_m / 1000.0 / speed * 60.0
                    score = ROAD_TYPE_SCORES[rt]
                    G.add_edge(u, v, length_m=length_m, travel_time_min=travel_time_min, road_type=rt,
                               road_hierarchy_score=score, speed_kph=speed, capacity_score=score)
                    edge_records.append(self._edge_record(edge_idx, u, v, data_mode, length_m, travel_time_min, rt, score, speed))
                    edge_idx += 1

        for rec in edge_records:
            u, v = rec["source"], rec["target"]
            rec["geometry_wkt"] = line_wkt([
                (float(G.nodes[u]["lat"]), float(G.nodes[u]["lon"])),
                (float(G.nodes[v]["lat"]), float(G.nodes[v]["lon"])),
            ])
        return G, pd.DataFrame(node_records), pd.DataFrame(edge_records), {"graph_source": "synthetic", "fallback_used": "false"}

    def _edge_record(self, edge_idx: int, u: str, v: str, data_mode: str, length_m: float,
                     travel_time_min: float, road_type: str, score: float, speed: float) -> Dict[str, object]:
        return {
            "edge_id": f"e_{edge_idx}",
            "city_id": self.city.city_id,
            "data_mode": data_mode,
            "source": u,
            "target": v,
            "length_m": length_m,
            "travel_time_min": travel_time_min,
            "road_type": road_type,
            "road_hierarchy_score": score,
            "speed_kph": speed,
            "capacity_score": score,
            "geometry_wkt": line_wkt([
                (float(self.city.lat), float(self.city.lon)),
            ]),
        }
