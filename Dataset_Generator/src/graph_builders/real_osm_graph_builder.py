from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple
import warnings
from pathlib import Path

import networkx as nx
import pandas as pd

from src.common import ROAD_TYPE_SCORES, ROAD_TYPE_SPEEDS, CityConfig, haversine_m, line_wkt, xy_from_latlon
from src.graph_builders.synthetic_graph_builder import SyntheticGraphBuilder


class RealOSMGraphBuilder:
    """Downloads and cleans an OSM drive network via OSMnx.

    The cleaner is intentionally strict about connectivity. OSM place polygons can
    contain many small disconnected fragments, and arbitrary node caps can break a
    connected road graph into many visible islands. The builder now:

    1. keeps the largest connected road component before sampling;
    2. applies the node cap with a network-distance expansion from the city center;
    3. rechecks the largest connected component after sampling.

    This preserves a contiguous graph for feature engineering, shortest paths,
    candidate KPIs, and the `01_road_graph_map.png` visualization.
    """

    def __init__(self, city: CityConfig, seed: int, pipeline_defaults: Dict[str, object]):
        self.city = city
        self.seed = seed
        self.defaults = pipeline_defaults
        self.cache_dir = Path("data_raw/osm")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cached_osm_graph(self) -> Tuple[nx.MultiDiGraph, Dict[str, str]]:
        import osmnx as ox  # type: ignore
        cache_path = self.cache_dir / f"{self.city.city_id}.graphml"
        meta: Dict[str, str] = {
            "graph_cache_path": str(cache_path),
            "osm_place": self.city.osm_place,
        }
        if cache_path.exists():
            G = ox.load_graphml(cache_path)
            meta["graph_cache_hit"] = "true"
            meta["osm_query_mode"] = "cached"
            meta["osm_radius_m"] = "n/a"
            return G, meta

        # Download
        radius_m = int(self.city.bbox_km * 500)  # approximate meters
        meta["osm_radius_m"] = str(radius_m)
        try:
            # Try point-based query first
            G = ox.graph_from_point(
                (self.city.lat, self.city.lon),
                dist=radius_m,
                network_type=str(self.defaults.get("real_network_type", "drive")),
                simplify=bool(self.defaults.get("real_simplify", True)),
                retain_all=bool(self.defaults.get("real_retain_all", True)),
                truncate_by_edge=bool(self.defaults.get("real_truncate_by_edge", True)),
            )
            meta["osm_query_mode"] = "point"
        except Exception:
            # Fallback to place-based query
            G = ox.graph_from_place(
                self.city.osm_place,
                network_type=str(self.defaults.get("real_network_type", "drive")),
                simplify=bool(self.defaults.get("real_simplify", True)),
                retain_all=bool(self.defaults.get("real_retain_all", True)),
                truncate_by_edge=bool(self.defaults.get("real_truncate_by_edge", True)),
            )
            meta["osm_query_mode"] = "place"

        ox.save_graphml(G, cache_path)
        meta["graph_cache_hit"] = "false"
        return G, meta

    def build(self, data_mode: str = "real") -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
        try:
            import osmnx as ox  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional install
            return self._fallback(data_mode, f"osmnx unavailable: {exc}")

        try:  # pragma: no cover - requires network access
            G_multi, graph_meta = self._get_cached_osm_graph()
            G_multi = ox.add_edge_speeds(G_multi, fallback=45)
            G_multi = ox.add_edge_travel_times(G_multi)
            G_multi = ox.project_graph(G_multi, to_crs="EPSG:4326")

            G_multi, clean_meta = self._clean_osm_graph(G_multi)
            G, nodes, edges = self._to_dataset_frames(G_multi, data_mode)

            final_ratio = self._largest_component_ratio(G)
            clean_meta["final_largest_connected_component_ratio"] = f"{final_ratio:.6f}"
            return G, nodes, edges, {
                "graph_source": "osm",
                "fallback_used": "false",
                **graph_meta,
                **clean_meta,
            }
        except Exception as exc:  # pragma: no cover - requires network access
            return self._fallback(data_mode, f"osm download failed: {exc}")

    def _clean_osm_graph(self, G_multi: nx.MultiDiGraph) -> Tuple[nx.MultiDiGraph, Dict[str, str]]:
        """Return a connected OSM graph and metadata about the cleanup.

        The previous implementation capped large OSM graphs with
        `list(component)[:max_nodes]`. Because NetworkX node iteration order is not
        spatial or topological, that produced maps with many disconnected patches.
        This method replaces that with connected component filtering plus a
        Dijkstra-distance sample.
        """
        max_nodes = int(self.defaults.get("real_max_nodes", 2500))
        meta: Dict[str, str] = {
            "connectivity_cleaning": "true",
            "original_node_count": str(G_multi.number_of_nodes()),
            "original_edge_count": str(G_multi.number_of_edges()),
        }

        G_multi = self._keep_largest_component(G_multi, meta, stage="before_sampling")

        if max_nodes > 0 and G_multi.number_of_nodes() > max_nodes:
            selected = self._network_distance_sample(G_multi, max_nodes)
            meta["node_cap_applied"] = "true"
            meta["node_cap"] = str(max_nodes)
            meta["nodes_before_cap"] = str(G_multi.number_of_nodes())
            meta["sampling_strategy"] = "network_distance_from_city_center"
            G_multi = G_multi.subgraph(selected).copy()
            G_multi = self._keep_largest_component(G_multi, meta, stage="after_sampling")
        else:
            meta["node_cap_applied"] = "false"
            meta["node_cap"] = str(max_nodes)

        meta["cleaned_node_count"] = str(G_multi.number_of_nodes())
        meta["cleaned_edge_count"] = str(G_multi.number_of_edges())
        return G_multi, meta

    def _keep_largest_component(self, G_multi: nx.MultiDiGraph, meta: Dict[str, str], stage: str) -> nx.MultiDiGraph:
        if G_multi.number_of_nodes() == 0:
            meta[f"component_count_{stage}"] = "0"
            meta[f"largest_component_ratio_{stage}"] = "0.000000"
            return G_multi

        undirected = nx.Graph(G_multi.to_undirected())
        components = sorted(nx.connected_components(undirected), key=len, reverse=True)
        component_count = len(components)
        largest_nodes = set(components[0]) if components else set()
        largest_ratio = len(largest_nodes) / max(G_multi.number_of_nodes(), 1)

        meta[f"component_count_{stage}"] = str(component_count)
        meta[f"largest_component_ratio_{stage}"] = f"{largest_ratio:.6f}"

        if component_count <= 1:
            return G_multi

        dropped = G_multi.number_of_nodes() - len(largest_nodes)
        meta[f"dropped_small_component_nodes_{stage}"] = str(dropped)
        return G_multi.subgraph(largest_nodes).copy()

    def _network_distance_sample(self, G_multi: nx.MultiDiGraph, max_nodes: int) -> set[Any]:
        if G_multi.number_of_nodes() <= max_nodes:
            return set(G_multi.nodes())

        seed_node = self._center_node(G_multi)
        simple = nx.Graph()
        for node_id, attrs in G_multi.nodes(data=True):
            simple.add_node(node_id, **attrs)
        for u, v, attrs in G_multi.edges(data=True):
            length = float(attrs.get("length", 1.0) or 1.0)
            if simple.has_edge(u, v):
                if length < simple[u][v].get("length", float("inf")):
                    simple[u][v]["length"] = length
            else:
                simple.add_edge(u, v, length=max(length, 0.001))

        lengths = nx.single_source_dijkstra_path_length(simple, seed_node, weight="length")
        ordered = sorted(lengths.items(), key=lambda item: item[1])
        selected = {node for node, _ in ordered[:max_nodes]}

        # The distance-ball sample should be connected. Keep this final guard to
        # handle any malformed OSM edges or zero-length anomalies.
        sampled = simple.subgraph(selected).copy()
        if sampled.number_of_nodes() == 0:
            return set(list(G_multi.nodes())[:max_nodes])
        largest = max(nx.connected_components(sampled), key=len)
        return set(largest)

    def _center_node(self, G_multi: nx.MultiDiGraph) -> Any:
        def dist(node_id: Any) -> float:
            attrs = G_multi.nodes[node_id]
            lat = float(attrs.get("y", self.city.lat))
            lon = float(attrs.get("x", self.city.lon))
            return haversine_m(lat, lon, self.city.lat, self.city.lon)

        return min(G_multi.nodes(), key=dist)

    def _to_dataset_frames(self, G_multi: nx.MultiDiGraph, data_mode: str) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
        G = nx.Graph()
        node_records = []

        for node_id, attrs in G_multi.nodes(data=True):
            lat = float(attrs.get("y", self.city.lat))
            lon = float(attrs.get("x", self.city.lon))
            x, y = xy_from_latlon(lat, lon, self.city.lat, self.city.lon)
            nid = str(node_id)
            G.add_node(nid, lat=lat, lon=lon, x=x, y=y, node_type="intersection", land_use_type="mixed")
            node_records.append({
                "node_id": nid,
                "city_id": self.city.city_id,
                "country": self.city.country,
                "data_mode": data_mode,
                "lat": lat,
                "lon": lon,
                "x": x,
                "y": y,
                "node_type": "intersection",
                "land_use_type": "mixed",
            })

        edge_records = []
        edge_idx = 0
        for u_raw, v_raw, attrs in G_multi.edges(data=True):
            u, v = str(u_raw), str(v_raw)
            if u == v or u not in G.nodes or v not in G.nodes:
                continue

            highway = attrs.get("highway", "unclassified")
            if isinstance(highway, list):
                highway = highway[0]
            road_type = str(highway)
            score = float(ROAD_TYPE_SCORES.get(road_type, 0.35))
            length_m = float(attrs.get("length", 0.0) or 0.0)
            if length_m <= 0:
                continue

            speed = float(attrs.get("speed_kph", ROAD_TYPE_SPEEDS.get(road_type, 35)) or ROAD_TYPE_SPEEDS.get(road_type, 35))
            travel_time_min = float(attrs.get("travel_time", length_m / 1000 / max(speed, 1) * 3600) or 0.0) / 60.0
            travel_time_min = max(travel_time_min, 0.01)

            # Keep the lowest travel time when parallel OSM edges collapse into
            # one NetworkX Graph edge, while still exporting every OSM edge row.
            if G.has_edge(u, v):
                if travel_time_min < G[u][v].get("travel_time_min", float("inf")):
                    G[u][v].update(
                        length_m=length_m,
                        travel_time_min=travel_time_min,
                        road_type=road_type,
                        road_hierarchy_score=score,
                        speed_kph=speed,
                        capacity_score=score,
                    )
            else:
                G.add_edge(
                    u,
                    v,
                    length_m=length_m,
                    travel_time_min=travel_time_min,
                    road_type=road_type,
                    road_hierarchy_score=score,
                    speed_kph=speed,
                    capacity_score=score,
                )

            p1 = (G.nodes[u]["lat"], G.nodes[u]["lon"])
            p2 = (G.nodes[v]["lat"], G.nodes[v]["lon"])
            edge_records.append({
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
                "geometry_wkt": line_wkt([p1, p2]),
            })
            edge_idx += 1

        # In rare cases, dropping zero-length/self-loop OSM edges can fragment the
        # simple graph. Keep only the final largest component and sync the frames.
        if G.number_of_nodes() > 0 and not nx.is_connected(G):
            largest = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest).copy()
            node_records = [r for r in node_records if r["node_id"] in largest]
            edge_records = [r for r in edge_records if r["source"] in largest and r["target"] in largest]

        return G, pd.DataFrame(node_records), pd.DataFrame(edge_records)

    def _largest_component_ratio(self, G: nx.Graph) -> float:
        if G.number_of_nodes() == 0:
            return 0.0
        if nx.is_connected(G):
            return 1.0
        largest = max((len(c) for c in nx.connected_components(G)), default=0)
        return largest / max(G.number_of_nodes(), 1)

    def _fallback(self, data_mode: str, reason: str):
        allow = bool(self.defaults.get("allow_real_fallback", True))
        if not allow:
            raise RuntimeError(reason)
        warnings.warn(f"Real graph unavailable for {self.city.city_id}; using synthetic fallback. Reason: {reason}")
        node_count = int(self.defaults.get("synthetic_node_count", 324))
        G, nodes, edges, meta = SyntheticGraphBuilder(self.city, self.seed, node_count=node_count).build(data_mode=data_mode)
        meta.update({"graph_source": "synthetic_fallback", "fallback_used": "true", "fallback_reason": reason})
        return G, nodes, edges, meta
