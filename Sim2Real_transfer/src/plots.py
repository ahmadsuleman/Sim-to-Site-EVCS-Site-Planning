from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save(fig, out_dir: Path, name: str, formats=('png','pdf'), dpi=300):
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f'{name}.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_framework(out_dir: str | Path, formats=('png','pdf'), dpi=300):
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis('off')
    boxes = [
        ('Simulator modes\nsynthetic | hybrid | real', 0.05),
        ('Oracle graph MILP\nfull candidates', 0.22),
        ('Candidate labels\nselected + assigned demand', 0.39),
        ('Transfer ranker\nKPI / synthetic / hybrid', 0.56),
        ('Top-M shortlist\nM=20,40,60', 0.73),
        ('Shortlisted MILP\nfinal selected stations', 0.90),
    ]
    for text, x in boxes:
        ax.text(x, 0.55, text, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.45', facecolor='white', edgecolor='black'))
    for (_, x1), (_, x2) in zip(boxes[:-1], boxes[1:]):
        ax.annotate('', xy=(x2-0.07, 0.55), xytext=(x1+0.07, 0.55), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.set_title('Simulator-assisted graph optimization workflow', fontsize=14, pad=16)
    _save(fig, out_dir, 'F1_framework_pipeline', formats, dpi)


def plot_dataset_mode_summary(summary: pd.DataFrame, out_dir: str | Path, formats=('png','pdf'), dpi=300):
    if summary.empty:
        return
    out_dir = Path(out_dir)
    agg = summary.groupby('data_mode', as_index=False).agg(
        cities=('city_id','nunique'),
        candidates=('candidate_count','sum'),
        demand_points=('demand_count','sum'),
    )
    x = np.arange(len(agg))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x-width, agg['cities'], width, label='Cities')
    ax.bar(x, agg['candidates'], width, label='Candidates')
    ax.bar(x+width, agg['demand_points'], width, label='Demand points')
    ax.set_xticks(x)
    ax.set_xticklabels(agg['data_mode'])
    ax.set_ylabel('Count')
    ax.set_title('Dataset mode summary')
    ax.legend()
    _save(fig, out_dir, 'F2_dataset_mode_summary', formats, dpi)


def plot_candidate_distribution(nodes: pd.DataFrame, edges: pd.DataFrame, candidates: pd.DataFrame, demand: pd.DataFrame, out_dir: str | Path, name='F3_candidate_distribution_map', formats=('png','pdf'), dpi=300, max_edges=10000, max_demand=5000):
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(7,7))
    if {'source','target'}.issubset(edges.columns) and {'node_id','lon','lat'}.issubset(nodes.columns):
        coord = nodes.set_index(nodes['node_id'].astype(str))[['lon','lat']].to_dict('index')
        es = edges.head(max_edges)
        for _, e in es.iterrows():
            u, v = str(e['source']), str(e['target'])
            if u in coord and v in coord:
                ax.plot([coord[u]['lon'], coord[v]['lon']], [coord[u]['lat'], coord[v]['lat']], linewidth=0.25, alpha=0.25)
    if {'lon','lat'}.issubset(demand.columns):
        dd = demand.head(max_demand)
        ax.scatter(dd['lon'], dd['lat'], s=5, alpha=0.25, label='Demand')
    if {'lon','lat'}.issubset(candidates.columns):
        ax.scatter(candidates['lon'], candidates['lat'], s=28, marker='^', label='Candidates')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Candidate and demand distribution')
    ax.legend(loc='best')
    ax.set_aspect('equal', adjustable='datalim')
    _save(fig, out_dir, name, formats, dpi)


def plot_objective_ratio(results: pd.DataFrame, out_dir: str | Path, formats=('png','pdf'), dpi=300):
    if results.empty or 'objective_ratio' not in results.columns:
        return
    df = results.copy()
    df = df[df['method_id'] != 'M6_full_real_oracle']
    agg = df.groupby('method_id', as_index=False)['objective_ratio'].mean().sort_values('objective_ratio')
    fig, ax = plt.subplots(figsize=(10,5))
    ax.barh(agg['method_id'], agg['objective_ratio'])
    ax.axvline(1.0, linestyle='--', linewidth=1)
    ax.set_xlabel('Objective ratio relative to full MILP oracle')
    ax.set_ylabel('Method')
    ax.set_title('Transfer performance by method')
    _save(fig, Path(out_dir), 'F4_objective_ratio_by_method', formats, dpi)


def plot_coverage_by_shortlist(results: pd.DataFrame, out_dir: str | Path, formats=('png','pdf'), dpi=300):
    if results.empty or 'coverage_ratio' not in results.columns or 'shortlist_size' not in results.columns:
        return
    df = results.dropna(subset=['shortlist_size']).copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8,5))
    for method, grp in df.groupby('method_id'):
        g = grp.groupby('shortlist_size', as_index=False)['coverage_ratio'].mean().sort_values('shortlist_size')
        ax.plot(g['shortlist_size'], g['coverage_ratio'], marker='o', label=method)
    ax.set_xlabel('Shortlist size M')
    ax.set_ylabel('Coverage ratio relative to oracle')
    ax.set_title('Coverage ratio by shortlist size')
    ax.legend(fontsize=8)
    _save(fig, Path(out_dir), 'F5_coverage_ratio_by_shortlist_size', formats, dpi)


def plot_recall_at_m(results: pd.DataFrame, out_dir: str | Path, formats=('png','pdf'), dpi=300):
    if results.empty or 'recall_at_M' not in results.columns or 'shortlist_size' not in results.columns:
        return
    df = results.dropna(subset=['shortlist_size']).copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8,5))
    for method, grp in df.groupby('method_id'):
        g = grp.groupby('shortlist_size', as_index=False)['recall_at_M'].mean().sort_values('shortlist_size')
        ax.plot(g['shortlist_size'], g['recall_at_M'], marker='o', label=method)
    ax.set_xlabel('Shortlist size M')
    ax.set_ylabel('Oracle recall@M')
    ax.set_title('Recall of oracle-selected candidates')
    ax.legend(fontsize=8)
    _save(fig, Path(out_dir), 'F6_recall_at_M', formats, dpi)


def plot_runtime_quality(results: pd.DataFrame, out_dir: str | Path, formats=('png','pdf'), dpi=300):
    if results.empty or 'runtime_speedup' not in results.columns or 'objective_ratio' not in results.columns:
        return
    df = results.replace([np.inf, -np.inf], np.nan).dropna(subset=['runtime_speedup','objective_ratio'])
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8,5))
    for method, grp in df.groupby('method_id'):
        ax.scatter(grp['runtime_speedup'], grp['objective_ratio'], label=method, s=40, alpha=0.8)
    ax.set_xlabel('Runtime speedup vs full oracle')
    ax.set_ylabel('Objective ratio')
    ax.set_title('Runtime-quality trade-off')
    ax.legend(fontsize=8)
    _save(fig, Path(out_dir), 'F7_runtime_quality_tradeoff', formats, dpi)


def plot_ranked_candidate_comparison(rankings: pd.DataFrame, out_dir: str | Path, formats=('png','pdf'), dpi=300):
    if rankings.empty:
        return
    df = rankings.copy()
    if 'oracle_assigned_demand' not in df.columns or 'rank_score' not in df.columns:
        return
    # Use one city/mode/M for readability.
    if 'city_id' in df.columns:
        df = df[df['city_id'] == df['city_id'].iloc[0]]
    if 'method_id' in df.columns:
        method = df['method_id'].drop_duplicates().iloc[0]
        df = df[df['method_id'] == method]
    df = df.sort_values('rank').head(50)
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(df['rank'], df['rank_score'], marker='o', label='Method rank score')
    ax.plot(df['rank'], df['oracle_assigned_demand'], marker='s', label='Oracle assigned demand')
    ax.set_xlabel('Candidate rank')
    ax.set_ylabel('Score / oracle utility')
    ax.set_title('Ranked candidate comparison')
    ax.legend()
    _save(fig, Path(out_dir), 'F8_ranked_candidate_comparison', formats, dpi)


def plot_selected_sites_map(nodes: pd.DataFrame, edges: pd.DataFrame, candidates: pd.DataFrame, oracle_selected: set, method_selected: set, out_dir: str | Path, formats=('png','pdf'), dpi=300, max_edges=10000):
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(7,7))
    if {'source','target'}.issubset(edges.columns) and {'node_id','lon','lat'}.issubset(nodes.columns):
        coord = nodes.set_index(nodes['node_id'].astype(str))[['lon','lat']].to_dict('index')
        for _, e in edges.head(max_edges).iterrows():
            u, v = str(e['source']), str(e['target'])
            if u in coord and v in coord:
                ax.plot([coord[u]['lon'], coord[v]['lon']], [coord[u]['lat'], coord[v]['lat']], linewidth=0.25, alpha=0.25)
    c = candidates.copy()
    c['candidate_id'] = c['candidate_id'].astype(str)
    ax.scatter(c['lon'], c['lat'], s=15, alpha=0.4, label='Candidates')
    o = c[c['candidate_id'].isin(oracle_selected)]
    m = c[c['candidate_id'].isin(method_selected)]
    if len(o): ax.scatter(o['lon'], o['lat'], s=70, marker='*', label='Oracle selected')
    if len(m): ax.scatter(m['lon'], m['lat'], s=55, marker='o', facecolors='none', edgecolors='black', label='Method selected')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Oracle vs proposed selected sites')
    ax.legend()
    ax.set_aspect('equal', adjustable='datalim')
    _save(fig, out_dir, 'F9_selected_sites_map', formats, dpi)


def plot_sensitivity_summary(sens: pd.DataFrame, out_dir: str | Path, formats=('png','pdf'), dpi=300):
    if sens.empty:
        return
    fig, ax = plt.subplots(figsize=(10,5))
    if 'parameter' in sens.columns and 'objective_ratio' in sens.columns:
        for p, grp in sens.groupby('parameter'):
            ax.plot(range(len(grp)), grp['objective_ratio'], marker='o', label=p)
        ax.set_ylabel('Objective ratio')
        ax.set_xlabel('Sensitivity scenario index')
    else:
        numeric = sens.select_dtypes(include='number')
        if numeric.empty: return
        ax.plot(numeric.iloc[:,0].to_numpy(), marker='o')
        ax.set_ylabel(numeric.columns[0])
    ax.set_title('Sensitivity summary')
    ax.legend(fontsize=8)
    _save(fig, Path(out_dir), 'F10_sensitivity_summary', formats, dpi)


def plot_feature_importance(fi: pd.DataFrame, out_dir: str | Path, formats=('png','pdf'), dpi=300):
    if fi.empty or 'importance' not in fi.columns:
        return
    df = fi.sort_values('importance', ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(df['feature'], df['importance'])
    ax.set_xlabel('Permutation importance')
    ax.set_title('Ranker feature importance')
    _save(fig, Path(out_dir), 'D1_feature_importance', formats, dpi)
