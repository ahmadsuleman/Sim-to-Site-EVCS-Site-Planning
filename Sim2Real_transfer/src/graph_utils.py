from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd

from .io_utils import normalize_series


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame, distance_metric: str = 'length_m', directed: bool = False) -> nx.Graph:
    G = nx.DiGraph() if directed else nx.Graph()
    nodes = nodes.copy()
    for _, r in nodes.iterrows():
        nid = str(r['node_id']) if 'node_id' in r else str(r.name)
        attrs = {k: r[k] for k in nodes.columns if k != 'node_id'}
        G.add_node(nid, **attrs)
    if distance_metric not in edges.columns:
        if distance_metric == 'length_m' and 'travel_time_min' in edges.columns:
            distance_metric = 'travel_time_min'
        else:
            numeric = edges.select_dtypes(include='number').columns.tolist()
            distance_metric = numeric[0] if numeric else None
    for _, r in edges.iterrows():
        u = str(r['source'])
        v = str(r['target'])
        w = float(r[distance_metric]) if distance_metric and pd.notna(r[distance_metric]) else 1.0
        attrs = {k: r[k] for k in edges.columns}
        attrs['weight'] = max(w, 1e-9)
        G.add_edge(u, v, **attrs)
    return G


def _nearest_node_by_xy(points: pd.DataFrame, nodes: pd.DataFrame) -> pd.Series:
    if not {'x','y'}.issubset(points.columns) or not {'x','y','node_id'}.issubset(nodes.columns):
        raise ValueError('Cannot snap to nearest node: x/y coordinates are missing.')
    node_xy = nodes[['node_id','x','y']].dropna().copy()
    ids = node_xy['node_id'].astype(str).to_numpy()
    xy = node_xy[['x','y']].to_numpy(float)
    out = []
    for _, r in points.iterrows():
        p = np.array([float(r['x']), float(r['y'])])
        d2 = ((xy - p)**2).sum(axis=1)
        out.append(ids[int(np.argmin(d2))])
    return pd.Series(out, index=points.index)


def ensure_node_ids(points: pd.DataFrame, nodes: pd.DataFrame, id_col: str = 'node_id') -> pd.DataFrame:
    df = points.copy()
    if id_col in df.columns and df[id_col].notna().any():
        df[id_col] = df[id_col].astype(str)
        return df
    df[id_col] = _nearest_node_by_xy(df, nodes)
    return df


def compute_reachable_pairs(
    G: nx.Graph,
    candidates: pd.DataFrame,
    demand: pd.DataFrame,
    radius_km: float,
    distance_metric: str = 'length_m',
    cache_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return pd.read_csv(cache_path)
    cutoff = radius_km * 1000.0 if distance_metric == 'length_m' else radius_km
    if distance_metric == 'travel_time_min':
        # In this case radius_km is treated by caller as minutes only when configured as time. Keep name for simplicity.
        cutoff = radius_km
    cand_nodes = candidates[['candidate_id','node_id']].dropna().copy()
    dem = demand[['demand_id','node_id','demand_mass']].dropna().copy()
    demand_node_to_rows: Dict[str, list] = {}
    for idx, r in dem.iterrows():
        demand_node_to_rows.setdefault(str(r['node_id']), []).append((str(r['demand_id']), float(r.get('demand_mass', 1.0))))
    rows = []
    for _, cr in cand_nodes.iterrows():
        cid = str(cr['candidate_id'])
        cnode = str(cr['node_id'])
        if cnode not in G:
            continue
        lengths = nx.single_source_dijkstra_path_length(G, cnode, cutoff=cutoff, weight='weight')
        for dnode, dist in lengths.items():
            if dnode in demand_node_to_rows:
                for did, q in demand_node_to_rows[dnode]:
                    rows.append({
                        'candidate_id': cid,
                        'demand_id': did,
                        'distance': float(dist),
                        'distance_km': float(dist) / 1000.0 if distance_metric == 'length_m' else np.nan,
                        'travel_time_min': float(dist) if distance_metric == 'travel_time_min' else np.nan,
                        'demand_mass': q,
                    })
    pairs = pd.DataFrame(rows)
    if not pairs.empty:
        pairs['candidate_id'] = pairs['candidate_id'].astype(str)
        pairs['demand_id'] = pairs['demand_id'].astype(str)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(cache_path, index=False)
    return pairs


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))


def compute_euclidean_pairs(candidates: pd.DataFrame, demand: pd.DataFrame, radius_km: float) -> pd.DataFrame:
    rows = []
    cdf = candidates[['candidate_id','lat','lon']].dropna().copy()
    ddf = demand[['demand_id','lat','lon','demand_mass']].dropna().copy()
    for _, c in cdf.iterrows():
        dist = haversine_km(float(c['lat']), float(c['lon']), ddf['lat'].astype(float).to_numpy(), ddf['lon'].astype(float).to_numpy())
        mask = dist <= radius_km
        for did, q, dkm in zip(ddf.loc[mask,'demand_id'], ddf.loc[mask,'demand_mass'], dist[mask]):
            rows.append({'candidate_id': str(c['candidate_id']), 'demand_id': str(did), 'distance': float(dkm*1000), 'distance_km': float(dkm), 'travel_time_min': np.nan, 'demand_mass': float(q)})
    return pd.DataFrame(rows)
