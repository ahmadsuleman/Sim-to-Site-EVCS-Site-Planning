from __future__ import annotations

import pandas as pd
from src.common import clamp01, normalize_series


def compute_demand_mass(nodes: pd.DataFrame) -> pd.Series:
    raw = (
        0.55 * nodes["population_density"]
        + 0.25 * nodes["activity_intensity"]
        + 0.20 * nodes["od_outflow_proxy"]
    )
    return normalize_series(pd.Series(clamp01(raw), index=nodes.index))
