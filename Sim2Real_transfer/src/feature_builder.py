from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .io_utils import normalize_series


ID_COLS = {'candidate_id','city_id','country','data_mode','node_id','lat','lon','land_use_type','candidate_source'}


def merge_candidate_tables(candidates: pd.DataFrame, features: pd.DataFrame, kpis: pd.DataFrame) -> pd.DataFrame:
    base = candidates.copy()
    for df in [features, kpis]:
        if df is None or df.empty or 'candidate_id' not in df.columns:
            continue
        cols = [c for c in df.columns if c not in base.columns or c == 'candidate_id']
        base = base.merge(df[cols], on='candidate_id', how='left')
    base['candidate_id'] = base['candidate_id'].astype(str)
    if 'node_id' in base.columns:
        base['node_id'] = base['node_id'].astype(str)
    return base


def compute_suitability_and_cost(candidates: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    c = candidates.copy()
    scoring = cfg.get('scoring', {})
    pos = scoring.get('positive_features', {})
    neg = scoring.get('negative_features', {})
    weighted = []
    weight_sum = 0.0
    for col, w in pos.items():
        if col in c.columns:
            weighted.append(normalize_series(c[col]) * float(w))
            weight_sum += float(w)
    for col, w in neg.items():
        if col in c.columns:
            weighted.append(normalize_series(c[col], invert=True) * float(w))
            weight_sum += float(w)
    if weighted and weight_sum > 0:
        c['suitability_score'] = sum(weighted) / weight_sum
    elif 'candidate_score_for_inclusion' in c.columns:
        c['suitability_score'] = normalize_series(c['candidate_score_for_inclusion'])
    else:
        numeric = c.select_dtypes(include='number').columns.difference(['lat','lon']).tolist()
        c['suitability_score'] = c[numeric].apply(pd.to_numeric, errors='coerce').mean(axis=1) if numeric else 0.5
        c['suitability_score'] = normalize_series(c['suitability_score'])

    cost_candidates = scoring.get('cost_columns_priority', ['land_cost_proxy','cost_proxy'])
    if any(col in c.columns for col in cost_candidates):
        first = next(col for col in cost_candidates if col in c.columns)
        c['cost_proxy_model'] = normalize_series(c[first])
    elif 'cost_efficiency' in c.columns:
        c['cost_proxy_model'] = normalize_series(c['cost_efficiency'], invert=True)
    else:
        c['cost_proxy_model'] = 0.5

    if 'uncertainty_score' not in c.columns:
        if 'candidate_uncertainty' in c.columns:
            c['uncertainty_score'] = c['candidate_uncertainty']
        elif 'kpi_uncertainty_score' in c.columns:
            c['uncertainty_score'] = c['kpi_uncertainty_score']
        else:
            c['uncertainty_score'] = np.nan
    c['uncertainty_score'] = pd.to_numeric(c['uncertainty_score'], errors='coerce').fillna(c['uncertainty_score'].median() if c['uncertainty_score'].notna().any() else 0.0)
    return c


def compute_capacity(candidates: pd.DataFrame, total_demand: float, cfg: Dict, multiplier: float = 1.0) -> pd.DataFrame:
    c = candidates.copy()
    cap_cfg = cfg.get('capacity', {})
    mode = cap_cfg.get('mode', 'hybrid')
    base = float(cap_cfg.get('base_fraction_of_total_demand', 0.12)) * float(total_demand) * float(multiplier)
    min_cap = float(cap_cfg.get('min_fraction_of_total_demand', 0.03)) * float(total_demand)
    max_cap = float(cap_cfg.get('max_fraction_of_total_demand', 0.25)) * float(total_demand)
    factors = pd.Series(np.ones(len(c)), index=c.index, dtype=float)
    road = normalize_series(c['road_hierarchy_score']) if 'road_hierarchy_score' in c.columns else pd.Series(0.5, index=c.index)
    land = normalize_series(c['available_space']) if 'available_space' in c.columns else pd.Series(0.5, index=c.index)
    if mode == 'constant':
        factors = pd.Series(1.0, index=c.index)
    elif mode == 'road_based':
        factors = 0.75 + 0.5 * road
    elif mode == 'land_based':
        factors = 0.75 + 0.5 * land
    else:
        factors = 0.75 + 0.25 * road + 0.25 * land
    c['capacity_model'] = (base * factors).clip(min_cap, max_cap)
    return c


def candidate_feature_columns(candidates: pd.DataFrame, label_cols: List[str] | None = None) -> List[str]:
    label_cols = set(label_cols or [])
    banned = ID_COLS | label_cols | {'geometry_wkt'}
    cols = []
    for col in candidates.columns:
        if col in banned:
            continue
        if pd.api.types.is_numeric_dtype(candidates[col]):
            cols.append(col)
    return cols


def compute_kpi_score(candidates: pd.DataFrame, cfg: Dict) -> pd.Series:
    c = compute_suitability_and_cost(candidates, cfg)
    return c['suitability_score']
