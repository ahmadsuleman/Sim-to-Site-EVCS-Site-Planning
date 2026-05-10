from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, distance_matrix


def _entropy(counts: np.ndarray) -> float:
    counts = counts.astype(float)
    total = counts.sum()
    if total <= 0:
        return np.nan
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum() / np.log(len(counts))) if len(counts) > 1 else 0.0


def candidate_distribution_metrics(candidates: pd.DataFrame, grid_size: int = 5) -> dict:
    c = candidates.dropna(subset=['lat','lon']).copy()
    n = len(c)
    if n == 0:
        return {'candidate_count': 0}
    lat = c['lat'].astype(float).to_numpy()
    lon = c['lon'].astype(float).to_numpy()
    xy = np.column_stack([lon, lat])
    hull_area = np.nan
    if n >= 3:
        try:
            hull = ConvexHull(xy)
            hull_area = float(hull.volume)  # degree^2 diagnostic only
        except Exception:
            hull_area = np.nan
    if n >= 2:
        D = distance_matrix(xy, xy)
        np.fill_diagonal(D, np.inf)
        nn = D.min(axis=1)
        nn_mean = float(np.mean(nn))
        nn_median = float(np.median(nn))
    else:
        nn_mean = nn_median = np.nan
    H, _, _ = np.histogram2d(lat, lon, bins=grid_size)
    occupied = int((H > 0).sum())
    total_cells = int(grid_size * grid_size)
    center_lat, center_lon = float(np.mean(lat)), float(np.mean(lon))
    # Approx central/peripheral diagnostic in degree units; maps use actual coordinates.
    dcenter = np.sqrt((lat-center_lat)**2 + (lon-center_lon)**2)
    road_entropy = np.nan
    if 'road_type' in c.columns:
        road_entropy = _entropy(c['road_type'].value_counts().to_numpy())
    elif 'land_use_type' in c.columns:
        road_entropy = _entropy(c['land_use_type'].value_counts().to_numpy())
    underserved_share = np.nan
    if 'equity_need_score' in c.columns:
        threshold = pd.to_numeric(c['equity_need_score'], errors='coerce').quantile(0.75)
        underserved_share = float((pd.to_numeric(c['equity_need_score'], errors='coerce') >= threshold).mean())
    return {
        'candidate_count': n,
        'convex_hull_area_degree2': hull_area,
        'nearest_neighbor_mean_degree': nn_mean,
        'nearest_neighbor_median_degree': nn_median,
        'spatial_entropy': _entropy(H.ravel()),
        'grid_occupied_cells': occupied,
        'grid_total_cells': total_cells,
        'grid_coverage_rate': occupied / total_cells if total_cells else np.nan,
        'distance_to_center_mean_degree': float(np.mean(dcenter)),
        'distance_to_center_median_degree': float(np.median(dcenter)),
        'road_or_landuse_entropy': road_entropy,
        'underserved_zone_candidate_share': underserved_share,
    }


def add_spatial_diagnostics(candidates: pd.DataFrame) -> pd.DataFrame:
    c = candidates.copy()
    if {'lat','lon'}.issubset(c.columns) and len(c):
        lat = c['lat'].astype(float).to_numpy()
        lon = c['lon'].astype(float).to_numpy()
        c['distance_to_center_degree'] = np.sqrt((lat-lat.mean())**2 + (lon-lon.mean())**2)
        xy = np.column_stack([lon, lat])
        if len(c) >= 2:
            D = distance_matrix(xy, xy)
            np.fill_diagonal(D, np.inf)
            c['nearest_candidate_distance_degree'] = D.min(axis=1)
        else:
            c['nearest_candidate_distance_degree'] = np.nan
    return c
