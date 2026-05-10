from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def recall_at_m(shortlist: set, oracle_selected: set) -> float:
    if not oracle_selected:
        return np.nan
    return len(shortlist & oracle_selected) / len(oracle_selected)


def precision_at_m(shortlist: set, oracle_selected: set) -> float:
    if not shortlist:
        return np.nan
    return len(shortlist & oracle_selected) / len(shortlist)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b) if (a | b) else np.nan


def ndcg_at_m(ranked_ids, relevance: dict, m: int) -> float:
    def dcg(ids):
        s = 0.0
        for idx, cid in enumerate(ids[:m], start=1):
            rel = float(relevance.get(str(cid), 0.0))
            s += (2**rel - 1) / np.log2(idx + 1)
        return s
    actual = dcg(ranked_ids)
    ideal_ids = [k for k, _ in sorted(relevance.items(), key=lambda kv: kv[1], reverse=True)]
    ideal = dcg(ideal_ids)
    return actual / ideal if ideal > 0 else np.nan


def spearman_rank_agreement(df: pd.DataFrame, score_col: str, oracle_col: str = 'oracle_assigned_demand') -> float:
    if score_col not in df.columns or oracle_col not in df.columns or len(df) < 3:
        return np.nan
    a = pd.to_numeric(df[score_col], errors='coerce')
    b = pd.to_numeric(df[oracle_col], errors='coerce')
    mask = a.notna() & b.notna()
    if mask.sum() < 3:
        return np.nan
    return float(spearmanr(a[mask], b[mask]).correlation)


def compare_to_oracle(method_metrics: dict, oracle_metrics: dict) -> dict:
    oracle_obj = float(oracle_metrics.get('objective_value', np.nan))
    method_obj = float(method_metrics.get('objective_value', np.nan))
    oracle_cov = float(oracle_metrics.get('covered_demand', np.nan))
    method_cov = float(method_metrics.get('covered_demand', np.nan))
    oracle_runtime = float(oracle_metrics.get('runtime_seconds', np.nan))
    method_runtime = float(method_metrics.get('runtime_seconds', np.nan))
    return {
        'objective_ratio': method_obj / oracle_obj if np.isfinite(oracle_obj) and abs(oracle_obj) > 1e-12 else np.nan,
        'coverage_ratio': method_cov / oracle_cov if np.isfinite(oracle_cov) and abs(oracle_cov) > 1e-12 else np.nan,
        'regret': oracle_obj - method_obj if np.isfinite(oracle_obj) and np.isfinite(method_obj) else np.nan,
        'runtime_speedup': oracle_runtime / method_runtime if np.isfinite(oracle_runtime) and np.isfinite(method_runtime) and method_runtime > 0 else np.nan,
    }
