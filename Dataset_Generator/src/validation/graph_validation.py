from __future__ import annotations

from typing import Any, Dict
import networkx as nx
import numpy as np
import pandas as pd


class GraphValidator:
    def validate(self, G: nx.Graph, nodes: pd.DataFrame, edges: pd.DataFrame, candidates: pd.DataFrame, demand: pd.DataFrame) -> Dict[str, Any]:
        if len(G) == 0:
            return {
                "node_count": 0,
                "edge_count": 0,
                "component_count": 0,
                "is_connected": False,
                "largest_connected_component_ratio": 0.0,
            }

        components = list(nx.connected_components(G)) if not nx.is_directed(G) else list(nx.weakly_connected_components(G))
        largest = max((len(c) for c in components), default=0)
        lcc_ratio = float(largest / max(len(G), 1))
        degrees = [d for _, d in G.degree()]
        lengths = pd.to_numeric(edges.get("length_m", pd.Series(dtype=float)), errors="coerce")
        sampled_path = self._average_shortest_path_sample(G)
        isolated = list(nx.isolates(G))

        return {
            "node_count": int(len(nodes)),
            "edge_count": int(len(edges)),
            "component_count": int(len(components)),
            "is_connected": bool(len(components) == 1),
            "isolated_node_count": int(len(isolated)),
            "connectivity_pass": bool(lcc_ratio >= 0.90),
            "candidate_count": int(len(candidates)),
            "demand_node_count": int(len(demand)),
            "average_degree": float(np.mean(degrees)) if degrees else 0.0,
            "median_degree": float(np.median(degrees)) if degrees else 0.0,
            "degree_std": float(np.std(degrees)) if degrees else 0.0,
            "largest_connected_component_ratio": lcc_ratio,
            "total_road_length_km": float(lengths.sum(skipna=True) / 1000.0),
            "mean_edge_length_m": float(lengths.mean(skipna=True)) if len(lengths) else 0.0,
            "median_edge_length_m": float(lengths.median(skipna=True)) if len(lengths) else 0.0,
            "road_type_distribution": edges.get("road_type", pd.Series(dtype=str)).value_counts(normalize=True).to_dict(),
            "network_density": float(nx.density(G)),
            "clustering_coefficient": float(nx.average_clustering(G)) if len(G) > 1 else 0.0,
            "average_shortest_path_sample": sampled_path,
            "missing_edge_length_percent": float(lengths.isna().mean() * 100) if len(lengths) else 100.0,
        }

    def _average_shortest_path_sample(self, G: nx.Graph, sample_size: int = 50) -> float:
        nodes = list(G.nodes())
        if len(nodes) < 2:
            return 0.0

        # Draw the sample from the largest component so this metric reflects the
        # usable road network rather than being biased by disconnected fragments.
        if not nx.is_connected(G):
            largest = max(nx.connected_components(G), key=len)
            nodes = [n for n in nodes if n in largest]

        nodes = nodes[: min(sample_size, len(nodes))]
        vals = []
        for i, u in enumerate(nodes):
            lengths = nx.single_source_dijkstra_path_length(G, u, cutoff=60, weight="travel_time_min")
            for v in nodes[i+1:]:
                if v in lengths:
                    vals.append(lengths[v])
        return float(np.mean(vals)) if vals else 0.0
