from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

CANONICAL_PATTERNS = {
    'nodes': ['graph_nodes*.csv', 'nodes*.csv'],
    'edges': ['graph_edges*.csv', 'edges*.csv'],
    'candidates': ['candidate_sites*.csv'],
    'features': ['candidate_feature_matrix*.csv'],
    'kpis': ['candidate_kpis*.csv'],
    'demand': ['demand_points*.csv'],
    'provenance': ['feature_provenance*.csv'],
    'quality': ['feature_quality_report*.csv'],
    'validation': ['validation_report.json'],
    'summary': ['run_summary.md'],
}


def find_file(data_dir: str | Path, logical_name: str, required: bool = True) -> Optional[Path]:
    data_dir = Path(data_dir)
    patterns = CANONICAL_PATTERNS.get(logical_name, [logical_name])
    matches: List[Path] = []
    for pat in patterns:
        matches.extend(sorted(data_dir.glob(pat)))
    matches = [m for m in matches if m.is_file()]
    if matches:
        # Prefer names without parentheses when available.
        matches.sort(key=lambda p: ('(' in p.name, len(p.name), p.name))
        return matches[0]
    if required:
        raise FileNotFoundError(f'Missing {logical_name} file in {data_dir}. Tried patterns: {patterns}')
    return None


def read_csv_if_exists(path: Optional[str | Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 1:
        return pd.DataFrame()
    return pd.read_csv(p)


def read_json_if_exists(path: Optional[str | Path]) -> Dict:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 1:
        return {}
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def standardize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col.endswith('_id') or col in {'node_id', 'source', 'target'}:
            df[col] = df[col].astype(str)
    return df


def choose_existing_column(df: pd.DataFrame, candidates: Iterable[str], default: Optional[str] = None) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return default


def normalize_series(s: pd.Series, invert: bool = False) -> pd.Series:
    x = pd.to_numeric(s, errors='coerce').astype(float)
    if x.notna().sum() == 0:
        out = pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    else:
        lo, hi = x.min(skipna=True), x.max(skipna=True)
        if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
            out = pd.Series(np.full(len(s), 0.5), index=s.index, dtype=float)
        else:
            out = (x - lo) / (hi - lo)
            out = out.fillna(out.median())
    if invert:
        out = 1.0 - out
    return out.clip(0, 1)


def safe_to_latex(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    try:
        df.to_latex(path, index=False, escape=True, **kwargs)
    except Exception:
        # LaTeX export is useful but not critical.
        pass


def slugify(text: str) -> str:
    text = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(text)).strip('_')
    return text[:150]
