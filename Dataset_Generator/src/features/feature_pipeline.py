from __future__ import annotations

from typing import Any, Dict, List, Tuple
import math

import networkx as nx
import numpy as np
import pandas as pd

from src.common import CityConfig, NODE_SCORE_FEATURES, clamp01, normalize_series, safe_corr, stable_rng
from src.features.provenance import ProvenanceTracker
from src.randomization.domain_randomizer import DomainRandomizer


class FeaturePipeline:
    """Builds node features, provenance records, and uncertainty metadata."""

    def __init__(self, city: CityConfig, data_mode: str, seed: int, config: Dict[str, Any], graph_meta: Dict[str, str]):
        self.city = city
        self.data_mode = data_mode
        self.seed = seed
        self.config = config
        self.graph_meta = graph_meta
        self.randomizer = DomainRandomizer(config["domain_randomization"], seed, city.city_id, data_mode)
        self.provenance = ProvenanceTracker(city.city_id, data_mode)
        self.rng = stable_rng(seed, city.city_id, data_mode, "features")

    def build(self, G: nx.Graph, nodes: pd.DataFrame, edges: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        nodes = nodes.copy()
        self._structural_features(G, nodes)
        self._landuse_features(nodes)
        self._synthetic_or_proxy_features(G, nodes)
        self._formula_features(nodes)
        self._confidence_uncertainty(nodes)
        self._fill_and_clip(nodes)
        return nodes, self.provenance.dataframe(), self.randomizer.dataframe()

    def _structural_features(self, G: nx.Graph, nodes: pd.DataFrame) -> None:
        degree = dict(G.degree())
        nodes["degree"] = nodes["node_id"].map(degree).fillna(0).astype(float)
        nodes["degree_norm"] = normalize_series(nodes["degree"])

        # Use exact centrality for small graphs; sampled approximation for larger graphs.
        if len(G) <= 700:
            bet = nx.betweenness_centrality(G, weight="travel_time_min", normalized=True)
        else:
            sample = min(250, len(G))
            bet = nx.betweenness_centrality(G, k=sample, seed=self.seed, weight="travel_time_min", normalized=True)
        clo = nx.closeness_centrality(G, distance="travel_time_min") if len(G) > 1 else {}
        nodes["betweenness_centrality"] = normalize_series(nodes["node_id"].map(bet).fillna(0.0))
        nodes["closeness_centrality"] = normalize_series(nodes["node_id"].map(clo).fillna(0.0))

        road_score = {}
        for node_id in nodes["node_id"]:
            vals = [float(d.get("road_hierarchy_score", 0.25)) for _, _, d in G.edges(str(node_id), data=True)]
            road_score[str(node_id)] = max(vals) if vals else 0.0
        nodes["road_hierarchy_score"] = nodes["node_id"].map(road_score).fillna(0.0).clip(0, 1)

        reach = self._reachability(G, nodes["node_id"].tolist(), thresholds=[5, 10, 15])
        for threshold in [5, 10, 15]:
            nodes[f"reachability_{threshold}min"] = nodes["node_id"].map(reach[threshold]).fillna(0.0)

        graph_source = self.graph_meta.get("graph_source", "unknown")
        source_type = "real" if graph_source == "osm" else "synthetic" if self.data_mode == "synthetic" else "proxy"
        confidence = 0.90 if graph_source == "osm" else 0.70 if self.data_mode == "synthetic" else 0.50
        for _, row in nodes.iterrows():
            for feature in ["degree", "degree_norm", "betweenness_centrality", "closeness_centrality", "road_hierarchy_score", "reachability_5min", "reachability_10min", "reachability_15min"]:
                self.provenance.add("node", row["node_id"], feature, row[feature], source_type, graph_source, confidence, False)

    def _reachability(self, G: nx.Graph, node_ids: List[str], thresholds: List[int]) -> Dict[int, Dict[str, float]]:
        result = {t: {} for t in thresholds}
        n = max(len(node_ids) - 1, 1)
        for node_id in node_ids:
            lengths = nx.single_source_dijkstra_path_length(G, str(node_id), cutoff=max(thresholds), weight="travel_time_min")
            for t in thresholds:
                result[t][str(node_id)] = max((sum(1 for d in lengths.values() if 0 < d <= t) / n), 0.0)
        return result

    def _landuse_features(self, nodes: pd.DataFrame) -> None:
        # OSM land use adapter can overwrite this in real deployments. Current version records proxy when OSM land use is absent.
        source_type = "real" if self.graph_meta.get("graph_source") == "osm" and self.data_mode == "real" else "synthetic" if self.data_mode == "synthetic" else "proxy"
        confidence = 0.70 if source_type == "synthetic" else 0.55 if source_type == "proxy" else 0.80
        for _, row in nodes.iterrows():
            self.provenance.add("node", row["node_id"], "land_use_type", row["land_use_type"], source_type, "landuse_adapter", confidence, False)

    def _synthetic_or_proxy_features(self, G: nx.Graph, nodes: pd.DataFrame) -> None:
        for idx, row in nodes.iterrows():
            entity_id = str(row["node_id"])
            land_use = str(row.get("land_use_type", "mixed"))
            for feature in ["population_density", "activity_intensity", "parking_dwell_score", "available_space", "nearby_charger_pressure"]:
                if self.data_mode in ["synthetic", "hybrid"]:
                    value, confidence = self.randomizer.sample(feature, land_use, entity_id, f"{self.data_mode}_mode_domain_randomization")
                    source_type, source_name, randomized = "synthetic", "domain_randomization", True
                elif self.graph_meta.get("graph_source") == "osm" and feature in {"activity_intensity"}:
                    # Real POI extraction hook. Current lightweight implementation uses graph/land-use proxy.
                    base = self._proxy_value_from_context(feature, row)
                    noise = self.rng.normal(0, 0.08)
                    value = float(clamp01(base + noise))
                    confidence = 0.55
                    source_type, source_name, randomized = "proxy", "osm_graph_context_proxy", False
                else:
                    value, confidence = self.randomizer.sample(feature, land_use, entity_id, f"{self.data_mode}_missing_direct_real_data")
                    source_type, source_name, randomized = "randomized", "domain_randomization", True
                nodes.at[idx, feature] = value
                self.provenance.add("node", entity_id, feature, value, source_type, source_name, confidence, randomized)

    def _proxy_value_from_context(self, feature: str, row: pd.Series) -> float:
        land = str(row.get("land_use_type", "mixed"))
        land_bonus = {
            "commercial": 0.30, "transit": 0.25, "tourism": 0.22, "religious": 0.18,
            "mixed": 0.15, "residential": 0.08, "industrial": 0.05, "public": 0.10,
            "highway": 0.08, "open_space": -0.10,
        }.get(land, 0.0)
        return float(clamp01(0.30 + land_bonus + 0.25 * row.get("road_hierarchy_score", 0) + 0.20 * row.get("degree_norm", 0)))

    def _formula_features(self, nodes: pd.DataFrame) -> None:
        nodes["traffic_flow_proxy"] = clamp01(
            0.35 * nodes["road_hierarchy_score"]
            + 0.25 * nodes["betweenness_centrality"]
            + 0.20 * nodes["activity_intensity"]
            + 0.10 * nodes["population_density"]
            + 0.10 * nodes["reachability_10min"]
        )
        nodes["residential_landuse_score"] = (nodes["land_use_type"] == "residential").astype(float)
        nodes["commercial_or_industrial_landuse"] = nodes["land_use_type"].isin(["commercial", "industrial", "mixed"]).astype(float)
        nodes["commercial_landuse_score"] = (nodes["land_use_type"] == "commercial").astype(float)
        nodes["urban_infrastructure_density"] = clamp01(0.55 * nodes["activity_intensity"] + 0.45 * nodes["road_hierarchy_score"])
        nodes["existing_charger_proximity"] = nodes["nearby_charger_pressure"]
        nodes["centrality_score"] = clamp01(0.5 * nodes["betweenness_centrality"] + 0.5 * nodes["closeness_centrality"])

        nodes["od_inflow_proxy"] = clamp01(
            0.45 * nodes["activity_intensity"]
            + 0.25 * nodes["reachability_10min"]
            + 0.20 * nodes["traffic_flow_proxy"]
            + 0.10 * nodes["road_hierarchy_score"]
        )
        nodes["od_outflow_proxy"] = clamp01(
            0.45 * nodes["population_density"]
            + 0.25 * nodes["residential_landuse_score"]
            + 0.20 * nodes["reachability_10min"]
            + 0.10 * nodes["traffic_flow_proxy"]
        )
        # Proxy grid access is formula-based unless a real electrical grid adapter is added.
        nodes["grid_access_score"] = clamp01(
            0.30 * nodes["road_hierarchy_score"]
            + 0.25 * nodes["urban_infrastructure_density"]
            + 0.20 * nodes["commercial_or_industrial_landuse"]
            + 0.15 * nodes["existing_charger_proximity"]
            + 0.10 * nodes["population_density"]
        )
        nodes["land_cost_proxy"] = clamp01(
            0.35 * nodes["activity_intensity"]
            + 0.25 * nodes["population_density"]
            + 0.20 * nodes["commercial_landuse_score"]
            + 0.10 * nodes["road_hierarchy_score"]
            + 0.10 * nodes["centrality_score"]
        )
        low_charger_access = 1 - nodes["nearby_charger_pressure"]
        peripheral_access_gap = 1 - nodes["reachability_10min"]
        lower_activity_accessibility = 1 - nodes["activity_intensity"]
        nodes["equity_need_score"] = clamp01(
            0.35 * nodes["population_density"]
            + 0.25 * low_charger_access
            + 0.20 * peripheral_access_gap
            + 0.20 * lower_activity_accessibility
        )
        nodes["service_coverage_gain"] = clamp01(
            0.45 * nodes["population_density"]
            + 0.25 * nodes["activity_intensity"]
            + 0.20 * low_charger_access
            + 0.10 * nodes["reachability_10min"]
        )

        for _, row in nodes.iterrows():
            for feature in ["traffic_flow_proxy", "od_inflow_proxy", "od_outflow_proxy", "grid_access_score", "land_cost_proxy", "equity_need_score", "service_coverage_gain"]:
                self.provenance.add("node", row["node_id"], feature, row[feature], "proxy", "feature_formula", 0.60, False)

        nodes.drop(columns=[
            "residential_landuse_score", "commercial_or_industrial_landuse", "commercial_landuse_score",
            "urban_infrastructure_density", "existing_charger_proximity", "centrality_score"
        ], inplace=True, errors="ignore")

    def _confidence_uncertainty(self, nodes: pd.DataFrame) -> None:
        prov = self.provenance.dataframe()
        if prov.empty:
            nodes["feature_confidence_mean"] = 0.0
            nodes["uncertainty_score"] = 1.0
            return
        agg = prov[prov["entity_type"] == "node"].groupby("entity_id")["confidence_score"].mean()
        nodes["feature_confidence_mean"] = nodes["node_id"].astype(str).map(agg).fillna(0.0).clip(0, 1)
        nodes["uncertainty_score"] = 1 - nodes["feature_confidence_mean"]

    def _fill_and_clip(self, nodes: pd.DataFrame) -> None:
        for col in NODE_SCORE_FEATURES + ["degree_norm", "betweenness_centrality", "closeness_centrality", "road_hierarchy_score", "reachability_5min", "reachability_10min", "reachability_15min", "feature_confidence_mean", "uncertainty_score"]:
            if col in nodes.columns:
                nodes[col] = pd.to_numeric(nodes[col], errors="coerce").fillna(0.0).clip(0, 1)
