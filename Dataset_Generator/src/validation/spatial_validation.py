from __future__ import annotations

from typing import Dict, Iterable
import networkx as nx
import numpy as np
import pandas as pd


class SpatialValidator:
    def neighbor_autocorrelation(self, G: nx.Graph, nodes: pd.DataFrame, features: Iterable[str]) -> Dict[str, float]:
        lookup = nodes.set_index("node_id")
        result = {}
        for feature in features:
            pairs = []
            for u, v in G.edges():
                if str(u) in lookup.index and str(v) in lookup.index and feature in lookup.columns:
                    pairs.append((float(lookup.loc[str(u), feature]), float(lookup.loc[str(v), feature])))
            if len(pairs) < 3:
                result[feature] = float("nan")
                continue
            a = np.array([p[0] for p in pairs])
            b = np.array([p[1] for p in pairs])
            if a.std() < 1e-12 or b.std() < 1e-12:
                result[feature] = 0.0
            else:
                result[feature] = float(np.corrcoef(a, b)[0, 1])
        return result
