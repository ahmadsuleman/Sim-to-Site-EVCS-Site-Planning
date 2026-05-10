from __future__ import annotations

from typing import Dict, Tuple
import networkx as nx
import pandas as pd

from src.common import CityConfig
from src.graph_builders.real_osm_graph_builder import RealOSMGraphBuilder


class HybridGraphBuilder:
    """Uses real graph data when available, with intentionally mixed features for hybrid scenarios."""

    def __init__(self, city: CityConfig, seed: int, pipeline_defaults: Dict[str, object]):
        self.real_builder = RealOSMGraphBuilder(city, seed, pipeline_defaults)

    def build(self, data_mode: str = "hybrid") -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
        return self.real_builder.build(data_mode=data_mode)
