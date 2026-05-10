from __future__ import annotations

import pandas as pd


def enforce_candidate_bounds(candidates: pd.DataFrame, score_col: str, min_count: int, max_count: int) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    ranked = candidates.sort_values(score_col, ascending=False).copy()
    count = max(min_count, min(max_count, len(ranked)))
    count = min(count, len(ranked))
    return ranked.head(count).reset_index(drop=True)
