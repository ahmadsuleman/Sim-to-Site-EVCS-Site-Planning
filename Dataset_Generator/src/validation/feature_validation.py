from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np
import pandas as pd

from src.common import NORMALIZED_FEATURES


class FeatureValidator:
    def validate(self, df: pd.DataFrame, normalized_features: Iterable[str] = NORMALIZED_FEATURES) -> pd.DataFrame:
        rows = []
        for feature in normalized_features:
            if feature not in df.columns:
                continue
            s = pd.to_numeric(df[feature], errors="coerce")
            rows.append({
                "feature_name": feature,
                "min": float(s.min(skipna=True)) if not s.dropna().empty else np.nan,
                "max": float(s.max(skipna=True)) if not s.dropna().empty else np.nan,
                "mean": float(s.mean(skipna=True)) if not s.dropna().empty else np.nan,
                "std": float(s.std(skipna=True)) if s.dropna().shape[0] > 1 else 0.0,
                "p05": float(s.quantile(0.05)) if not s.dropna().empty else np.nan,
                "p25": float(s.quantile(0.25)) if not s.dropna().empty else np.nan,
                "median": float(s.median(skipna=True)) if not s.dropna().empty else np.nan,
                "p75": float(s.quantile(0.75)) if not s.dropna().empty else np.nan,
                "p95": float(s.quantile(0.95)) if not s.dropna().empty else np.nan,
                "skewness": float(s.skew(skipna=True)) if s.dropna().shape[0] > 2 else 0.0,
                "kurtosis": float(s.kurt(skipna=True)) if s.dropna().shape[0] > 3 else 0.0,
                "missing_percent": float(s.isna().mean() * 100),
                "zero_variance_flag": bool(s.std(skipna=True) < 1e-9 if s.dropna().shape[0] > 1 else True),
                "outlier_percent": self._outlier_percent(s),
                "range_check_pass": bool((s.dropna().between(0, 1)).all()) if not s.dropna().empty else False,
            })
        return pd.DataFrame(rows)

    def _outlier_percent(self, s: pd.Series) -> float:
        s = s.dropna()
        if len(s) < 4:
            return 0.0
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 1e-12:
            return 0.0
        return float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() * 100)
