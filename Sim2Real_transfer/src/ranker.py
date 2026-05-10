from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class RankerBundle:
    method: str
    feature_columns: List[str]
    model: object
    residual_model: Optional[object] = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xn = X.reindex(columns=self.feature_columns).copy()
        y = np.asarray(self.model.predict(Xn), dtype=float)
        if self.residual_model is not None:
            y = y + np.asarray(self.residual_model.predict(Xn), dtype=float)
        return y

    def save(self, path):
        joblib.dump(self, path)


def make_regressor(seed: int = 42):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', HistGradientBoostingRegressor(random_state=seed, max_iter=250, learning_rate=0.05, l2_regularization=0.01)),
    ])


def train_ranker(train_df: pd.DataFrame, feature_columns: List[str], target_col: str, seed: int, method: str) -> RankerBundle:
    if train_df.empty:
        # Constant fallback model via RF on dummy data is overkill; use simple mean wrapper.
        raise ValueError(f'No training rows for {method}.')
    X = train_df.reindex(columns=feature_columns)
    y = pd.to_numeric(train_df[target_col], errors='coerce').fillna(0.0).astype(float)
    model = make_regressor(seed)
    model.fit(X, y)
    return RankerBundle(method=method, feature_columns=feature_columns, model=model)


def train_synthetic_to_hybrid(synth_df: pd.DataFrame, hybrid_df: pd.DataFrame, feature_columns: List[str], target_col: str, seed: int) -> RankerBundle:
    base = train_ranker(synth_df, feature_columns, target_col, seed, method='synthetic_to_hybrid').model
    Xh = hybrid_df.reindex(columns=feature_columns)
    yh = pd.to_numeric(hybrid_df[target_col], errors='coerce').fillna(0.0).astype(float)
    base_pred = np.asarray(base.predict(Xh), dtype=float)
    residual = yh.to_numpy(float) - base_pred
    residual_model = make_regressor(seed + 17)
    residual_model.fit(Xh, residual)
    return RankerBundle(method='synthetic_to_hybrid', feature_columns=feature_columns, model=base, residual_model=residual_model)


def permutation_feature_importance(bundle: RankerBundle, df: pd.DataFrame, target_col: str, seed: int = 42, n_repeats: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if df.empty:
        return pd.DataFrame(columns=['feature','importance'])
    X = df.reindex(columns=bundle.feature_columns).copy()
    y = pd.to_numeric(df[target_col], errors='coerce').fillna(0.0).to_numpy(float)
    base_pred = bundle.predict(X)
    base_mse = float(np.mean((y-base_pred)**2))
    rows = []
    for col in bundle.feature_columns:
        losses = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            pred = bundle.predict(Xp)
            losses.append(float(np.mean((y-pred)**2)) - base_mse)
        rows.append({'feature': col, 'importance': float(np.mean(losses))})
    out = pd.DataFrame(rows).sort_values('importance', ascending=False)
    return out
