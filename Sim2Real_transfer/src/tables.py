from __future__ import annotations

from pathlib import Path
import pandas as pd

from .io_utils import safe_to_latex


def export_table(df: pd.DataFrame, out_dir: str | Path, name: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f'{name}.csv', index=False)
    try:
        df.to_excel(out_dir / f'{name}.xlsx', index=False)
    except Exception:
        pass
    safe_to_latex(df, out_dir / f'{name}.tex')


def method_definitions() -> pd.DataFrame:
    rows = [
        ('M0_random_topM_milp', 'Random top-M + MILP', 'Lower-bound random shortlist, refined by graph MILP.'),
        ('M1_kpi_topM_milp', 'KPI top-M + MILP', 'Explainable non-learning baseline using KPI/suitability score.'),
        ('M2_greedy_graph', 'Greedy graph coverage', 'Fast graph heuristic baseline.'),
        ('M3_synthetic_ranker_topM_milp', 'Synthetic-trained ranker + MILP', 'Surrogate trained on synthetic simulator outputs.'),
        ('M4_hybrid_ranker_topM_milp', 'Hybrid-trained ranker + MILP', 'Surrogate trained on hybrid simulator outputs.'),
        ('M5_synthetic_to_hybrid_topM_milp', 'Synthetic→Hybrid ranker + MILP', 'Synthetic pretraining plus hybrid residual refinement.'),
        ('M6_full_real_oracle', 'Full real-mode MILP oracle', 'Upper bound using all candidates in the target real-mode dataset.'),
    ]
    return pd.DataFrame(rows, columns=['method_id','method_name','definition'])
