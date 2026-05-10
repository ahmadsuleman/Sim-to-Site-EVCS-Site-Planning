from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .candidate_distribution import add_spatial_diagnostics, candidate_distribution_metrics
from .config import ensure_dirs
from .data_loader import DatasetInstance, load_dataset_index, load_instance, validate_index_coverage
from .feature_builder import candidate_feature_columns, compute_capacity, compute_suitability_and_cost, merge_candidate_tables
from .graph_utils import build_graph, compute_euclidean_pairs, compute_reachable_pairs, ensure_node_ids
from .metrics import compare_to_oracle, jaccard, ndcg_at_m, precision_at_m, recall_at_m, spearman_rank_agreement
from .optimizer import OptimizationOptions, solve_location_allocation, greedy_coverage
from .ranker import permutation_feature_importance, train_ranker, train_synthetic_to_hybrid
from .tables import export_table, method_definitions
from .plots import (
    plot_framework, plot_dataset_mode_summary, plot_candidate_distribution, plot_objective_ratio,
    plot_coverage_by_shortlist, plot_recall_at_m, plot_runtime_quality, plot_ranked_candidate_comparison,
    plot_selected_sites_map, plot_sensitivity_summary, plot_feature_importance,
)


def _opt_options(cfg: Dict, model_id: str, k=None, radius=None, beta=None, mu=None, capacity_multiplier=1.0) -> OptimizationOptions:
    oc = cfg.get('optimization', {})
    return OptimizationOptions(
        model_id=model_id,
        k=int(k if k is not None else oc.get('k', 10)),
        radius_km=float(radius if radius is not None else oc.get('radius_km', 5.0)),
        coverage_weight=float(oc.get('coverage_weight', 1.0)),
        suitability_weight=float(beta if beta is not None else oc.get('suitability_weight', 0.5)),
        cost_weight=float(mu if mu is not None else oc.get('cost_weight', 0.25)),
        use_capacity=bool(oc.get('use_capacity', True)),
        solver=str(oc.get('solver', 'scipy_milp')),
        time_limit_seconds=int(oc.get('time_limit_seconds', 120)),
        mip_gap=float(oc.get('mip_gap', 0.03)),
        fallback_to_greedy=bool(oc.get('fallback_to_greedy', True)),
    )


def prepare_instance(inst: DatasetInstance, cfg: Dict, out_dirs: Dict, radius_km: float | None = None, capacity_multiplier: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    radius_km = float(radius_km if radius_km is not None else cfg.get('optimization', {}).get('radius_km', 5.0))
    metric = cfg.get('optimization', {}).get('distance_metric', 'length_m')
    nodes = inst.nodes.copy()
    edges = inst.edges.copy()
    candidates = merge_candidate_tables(inst.candidates, inst.features, inst.kpis)
    candidates = compute_suitability_and_cost(candidates, cfg)
    candidates = add_spatial_diagnostics(candidates)
    demand = inst.demand.copy()
    if 'demand_mass' not in demand.columns:
        demand['demand_mass'] = 1.0
    candidates = ensure_node_ids(candidates, nodes)
    demand = ensure_node_ids(demand, nodes)
    total_demand = float(pd.to_numeric(demand['demand_mass'], errors='coerce').fillna(0).sum())
    candidates = compute_capacity(candidates, total_demand, cfg, multiplier=capacity_multiplier)
    G = build_graph(nodes, edges, distance_metric=metric)
    cache_name = f'{inst.instance_id}_R{radius_km:g}_{metric}_pairs.csv'.replace('.', 'p')
    pairs = compute_reachable_pairs(G, candidates, demand, radius_km, metric, cache_path=out_dirs['cache'] / cache_name)
    return candidates, demand, pairs, nodes


def run_oracle_for_instance(inst: DatasetInstance, cfg: Dict, out_dirs: Dict, radius_km: float | None = None, k: int | None = None, beta=None, mu=None, capacity_multiplier: float = 1.0, model_id_suffix: str = '') -> Tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates, demand, pairs, nodes = prepare_instance(inst, cfg, out_dirs, radius_km, capacity_multiplier)
    opts = _opt_options(cfg, f'M6_full_real_oracle{model_id_suffix}', k=k, radius=radius_km, beta=beta, mu=mu)
    result = solve_location_allocation(candidates, demand, pairs, opts)
    result.metrics.update({'city_id': inst.city_id, 'country': inst.country, 'data_mode': inst.data_mode, 'seed': inst.seed})
    labels = build_oracle_labels(candidates, result.assignments, result.selected_candidate_ids, inst)
    return result.metrics, labels, candidates, demand, pairs, nodes


def build_oracle_labels(candidates: pd.DataFrame, assignments: pd.DataFrame, selected_ids: List[str], inst: DatasetInstance) -> pd.DataFrame:
    c = candidates.copy()
    c['candidate_id'] = c['candidate_id'].astype(str)
    assigned = assignments.groupby('candidate_id')['demand_mass'].sum() if not assignments.empty else pd.Series(dtype=float)
    c['oracle_assigned_demand'] = c['candidate_id'].map(assigned).fillna(0.0).astype(float)
    c['oracle_selected'] = c['candidate_id'].isin(set(map(str, selected_ids))).astype(int)
    c['oracle_rank'] = c['oracle_assigned_demand'].rank(ascending=False, method='dense').astype(int)
    c['city_id'] = inst.city_id
    c['country'] = inst.country
    c['data_mode'] = inst.data_mode
    c['seed'] = inst.seed
    return c


def build_dataset_summary(index: pd.DataFrame, instances: List[DatasetInstance], cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    dist_rows = []
    for inst in instances:
        candidates = merge_candidate_tables(inst.candidates, inst.features, inst.kpis)
        candidates = compute_suitability_and_cost(candidates, cfg)
        dist = candidate_distribution_metrics(candidates, grid_size=int(cfg.get('candidate_distribution', {}).get('grid_size', 5)))
        dist.update({'city_id': inst.city_id, 'country': inst.country, 'data_mode': inst.data_mode, 'seed': inst.seed})
        dist_rows.append(dist)
        val = inst.validation or {}
        graph_val = val.get('graph_validation', val)
        prov = val.get('provenance_summary', {})
        rows.append({
            'city_id': inst.city_id, 'country': inst.country, 'data_mode': inst.data_mode, 'seed': inst.seed,
            'node_count': len(inst.nodes), 'edge_count': len(inst.edges), 'candidate_count': len(inst.candidates), 'demand_count': len(inst.demand),
            'is_connected': graph_val.get('is_connected'),
            'isolated_node_count': graph_val.get('isolated_node_count'),
            'largest_connected_component_ratio': graph_val.get('largest_connected_component_ratio'),
            'real_feature_percent': prov.get('real_feature_percent'),
            'proxy_feature_percent': prov.get('proxy_feature_percent'),
            'synthetic_feature_percent': prov.get('synthetic_feature_percent'),
            'mean_feature_confidence': prov.get('mean_feature_confidence'),
            'mean_uncertainty_score': prov.get('mean_uncertainty_score'),
        })
    return pd.DataFrame(rows), pd.DataFrame(dist_rows)


def _rank_candidates(method_id: str, candidates: pd.DataFrame, scores: np.ndarray, oracle_labels: pd.DataFrame | None, selected_by_method: set | None, shortlist_size: int | None) -> pd.DataFrame:
    df = candidates.copy()
    df['candidate_id'] = df['candidate_id'].astype(str)
    df['method_id'] = method_id
    df['shortlist_size'] = shortlist_size
    df['rank_score'] = scores
    df['rank'] = pd.Series(scores).rank(ascending=False, method='first').astype(int).to_numpy()
    if oracle_labels is not None and not oracle_labels.empty:
        keep = ['candidate_id','oracle_selected','oracle_assigned_demand','oracle_rank']
        df = df.merge(oracle_labels[keep], on='candidate_id', how='left')
    df['selected_by_method'] = df['candidate_id'].isin(selected_by_method or set()).astype(int)
    cols = ['city_id','country','data_mode','seed','method_id','shortlist_size','candidate_id','rank','rank_score','suitability_score','cost_proxy_model','uncertainty_score','distance_to_center_degree','nearest_candidate_distance_degree','oracle_selected','oracle_assigned_demand','oracle_rank','selected_by_method']
    return df[[c for c in cols if c in df.columns]].sort_values(['method_id','shortlist_size','rank'])


def run_shortlist_method(inst: DatasetInstance, cfg: Dict, out_dirs: Dict, candidates: pd.DataFrame, demand: pd.DataFrame, pairs: pd.DataFrame, oracle_metrics: dict, oracle_labels: pd.DataFrame, method_id: str, scores: np.ndarray, shortlist_size: int, direct_topk: bool = False) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    scores = np.asarray(scores, dtype=float)
    ordered_idx = np.argsort(-scores)
    selected_pool = candidates.iloc[ordered_idx[:shortlist_size]].copy()
    if direct_topk:
        selected_pool = candidates.iloc[ordered_idx[:int(cfg.get('optimization', {}).get('k', 10))]].copy()
        # Evaluate direct selection through greedy assignment, without solving MILP.
        from .optimizer import _greedy_assign, _evaluate_solution
        opts = _opt_options(cfg, method_id)
        assignments = _greedy_assign(candidates, demand, pairs, selected_pool['candidate_id'].astype(str).tolist(), opts.use_capacity)
        res = _evaluate_solution(candidates, demand, pairs, selected_pool['candidate_id'].astype(str).tolist(), assignments, opts, 'Feasible', 'direct_topk', float(assignments['demand_mass'].sum()) if not assignments.empty else 0.0, 0.0)
    else:
        p = pairs[pairs['candidate_id'].astype(str).isin(set(selected_pool['candidate_id'].astype(str)))].copy()
        opts = _opt_options(cfg, method_id)
        res = solve_location_allocation(selected_pool, demand, p, opts)
    metrics = dict(res.metrics)
    metrics.update({'city_id': inst.city_id, 'country': inst.country, 'data_mode': inst.data_mode, 'seed': inst.seed, 'method_id': method_id, 'shortlist_size': shortlist_size})
    metrics.update(compare_to_oracle(metrics, oracle_metrics))
    shortlist_set = set(selected_pool['candidate_id'].astype(str))
    oracle_set = set(oracle_labels.loc[oracle_labels['oracle_selected'] == 1, 'candidate_id'].astype(str))
    method_set = set(map(str, res.selected_candidate_ids))
    oracle_final_set = oracle_set
    relevance = oracle_labels.set_index(oracle_labels['candidate_id'].astype(str))['oracle_assigned_demand'].fillna(0.0).to_dict()
    ranked_ids = candidates.iloc[ordered_idx]['candidate_id'].astype(str).tolist()
    metrics.update({
        'recall_at_M': recall_at_m(shortlist_set, oracle_set),
        'precision_at_M': precision_at_m(shortlist_set, oracle_set),
        'ndcg_at_M': ndcg_at_m(ranked_ids, relevance, shortlist_size),
        'topK_jaccard': jaccard(method_set, oracle_final_set),
    })
    rank_df = _rank_candidates(method_id, candidates, scores, oracle_labels, method_set, shortlist_size)
    selected_df = candidates[candidates['candidate_id'].astype(str).isin(method_set)].copy()
    selected_df['method_id'] = method_id
    selected_df['shortlist_size'] = shortlist_size
    return metrics, rank_df, selected_df


def run_experiment(dataset_index_path: str | Path, output_dir: str | Path, cfg: Dict, run_sensitivity: bool = True) -> None:
    out_dirs = ensure_dirs(output_dir)
    seed = int(cfg.get('seed', 42))
    rng = np.random.default_rng(seed)
    index = load_dataset_index(dataset_index_path, expected_seed=seed)
    mode_report = validate_index_coverage(index, cfg.get('input', {}).get('required_modes', ['synthetic','hybrid','real']), bool(cfg.get('input', {}).get('allow_single_city_pilot', True)))
    mode_report.to_csv(out_dirs['metadata'] / 'input_validation_report.csv', index=False)
    instances = [load_instance(row) for _, row in index.iterrows()]
    ds_summary, dist_summary = build_dataset_summary(index, instances, cfg)
    ds_summary.to_csv(out_dirs['metadata'] / 'dataset_inventory.csv', index=False)
    export_table(ds_summary, out_dirs['tables'], 'T1_dataset_summary')
    export_table(dist_summary, out_dirs['tables'], 'T2_candidate_distribution')
    export_table(method_definitions(), out_dirs['tables'], 'T3_method_definitions')

    # Plot first available real instance candidate map.
    plot_framework(out_dirs['figures'], cfg.get('plots', {}).get('figure_format', ['png','pdf']), int(cfg.get('plots', {}).get('dpi', 300)))
    plot_dataset_mode_summary(ds_summary, out_dirs['figures'], cfg.get('plots', {}).get('figure_format', ['png','pdf']), int(cfg.get('plots', {}).get('dpi', 300)))
    first_real = next((i for i in instances if i.data_mode == 'real'), instances[0])
    plot_candidate_distribution(first_real.nodes, first_real.edges, first_real.candidates, first_real.demand, out_dirs['figures'], formats=cfg.get('plots', {}).get('figure_format', ['png','pdf']), dpi=int(cfg.get('plots', {}).get('dpi', 300)), max_edges=int(cfg.get('plots', {}).get('max_edges_on_map', 10000)))

    # Oracle labels for all instances.
    oracle_metrics_rows = []
    all_labels = []
    prepared = {}
    for inst in instances:
        metrics, labels, candidates, demand, pairs, nodes = run_oracle_for_instance(inst, cfg, out_dirs)
        oracle_metrics_rows.append(metrics)
        all_labels.append(labels)
        prepared[inst.instance_id] = {'candidates': candidates, 'demand': demand, 'pairs': pairs, 'nodes': nodes, 'labels': labels, 'metrics': metrics, 'inst': inst}
        labels.to_csv(out_dirs['labels'] / f'{inst.instance_id}_oracle_labels.csv', index=False)
    oracle_df = pd.DataFrame(oracle_metrics_rows)
    oracle_df.to_csv(out_dirs['oracle'] / 'oracle_results.csv', index=False)
    labels_all = pd.concat(all_labels, ignore_index=True) if all_labels else pd.DataFrame()
    labels_all.to_csv(out_dirs['labels'] / 'oracle_candidate_labels.csv', index=False)

    # Feature columns from all labels.
    feature_cols = candidate_feature_columns(labels_all, label_cols=['oracle_selected','oracle_assigned_demand','oracle_rank'])
    if not feature_cols:
        raise ValueError('No numeric feature columns found for ranker training.')

    result_rows = []
    ranking_rows = []
    selected_rows = []
    fi_rows = []
    real_instances = [i for i in instances if i.data_mode == 'real']
    for target in real_instances:
        target_pack = prepared[target.instance_id]
        source_labels = labels_all[labels_all['city_id'] != target.city_id].copy()
        if source_labels.empty and bool(cfg.get('input', {}).get('allow_single_city_pilot', True)):
            source_labels = labels_all[(labels_all['city_id'] == target.city_id) & (labels_all['data_mode'] != 'real')].copy()
        synth = source_labels[source_labels['data_mode'] == 'synthetic'].copy()
        hybrid = source_labels[source_labels['data_mode'] == 'hybrid'].copy()
        rankers = {}
        if not synth.empty:
            rankers['M3_synthetic_ranker_topM_milp'] = train_ranker(synth, feature_cols, cfg.get('transfer', {}).get('target_column', 'oracle_assigned_demand'), seed, 'synthetic')
        if not hybrid.empty:
            rankers['M4_hybrid_ranker_topM_milp'] = train_ranker(hybrid, feature_cols, cfg.get('transfer', {}).get('target_column', 'oracle_assigned_demand'), seed, 'hybrid')
        if not synth.empty and not hybrid.empty:
            rankers['M5_synthetic_to_hybrid_topM_milp'] = train_synthetic_to_hybrid(synth, hybrid, feature_cols, cfg.get('transfer', {}).get('target_column', 'oracle_assigned_demand'), seed)
        for mid, bundle in rankers.items():
            bundle.save(out_dirs['models'] / f'{target.city_id}_{mid}.joblib')
            fi = permutation_feature_importance(bundle, source_labels, cfg.get('transfer', {}).get('target_column', 'oracle_assigned_demand'), seed)
            fi['method_id'] = mid; fi['target_city_id'] = target.city_id
            fi_rows.append(fi)

        candidates = target_pack['candidates']; demand = target_pack['demand']; pairs = target_pack['pairs']; oracle_labels = target_pack['labels']; oracle_metrics = target_pack['metrics']
        # Oracle row.
        oracle_row = dict(oracle_metrics)
        oracle_row.update({'method_id': 'M6_full_real_oracle', 'shortlist_size': np.nan, 'objective_ratio': 1.0, 'coverage_ratio': 1.0, 'regret': 0.0, 'runtime_speedup': 1.0, 'recall_at_M': 1.0, 'precision_at_M': 1.0, 'ndcg_at_M': 1.0, 'topK_jaccard': 1.0})
        result_rows.append(oracle_row)
        # KPI baseline.
        kpi_scores = candidates['suitability_score'].to_numpy(float)
        for M in cfg.get('transfer', {}).get('shortlist_sizes', [20,40,60]):
            metrics, ranks, sels = run_shortlist_method(target, cfg, out_dirs, candidates, demand, pairs, oracle_metrics, oracle_labels, 'M1_kpi_topM_milp', kpi_scores, int(M))
            result_rows.append(metrics); ranking_rows.append(ranks); selected_rows.append(sels)
            # Random baselines.
            for rep in range(int(cfg.get('baselines', {}).get('random_repeats', 3))):
                random_scores = rng.random(len(candidates))
                metrics, ranks, sels = run_shortlist_method(target, cfg, out_dirs, candidates, demand, pairs, oracle_metrics, oracle_labels, f'M0_random_topM_milp_rep{rep+1}', random_scores, int(M))
                result_rows.append(metrics); ranking_rows.append(ranks); selected_rows.append(sels)
            # Learned rankers.
            for mid, bundle in rankers.items():
                scores = bundle.predict(candidates)
                metrics, ranks, sels = run_shortlist_method(target, cfg, out_dirs, candidates, demand, pairs, oracle_metrics, oracle_labels, mid, scores, int(M))
                result_rows.append(metrics); ranking_rows.append(ranks); selected_rows.append(sels)
                if cfg.get('transfer', {}).get('direct_topk_comparison', True) and int(M) == int(cfg.get('transfer', {}).get('shortlist_sizes', [20,40,60])[1]):
                    dm_id = mid.replace('_topM_milp','_direct_topK')
                    m2, r2, s2 = run_shortlist_method(target, cfg, out_dirs, candidates, demand, pairs, oracle_metrics, oracle_labels, dm_id, scores, int(M), direct_topk=True)
                    result_rows.append(m2); ranking_rows.append(r2); selected_rows.append(s2)
        # Greedy graph baseline once.
        if cfg.get('baselines', {}).get('include_greedy_graph', True):
            opts = _opt_options(cfg, 'M2_greedy_graph')
            res = greedy_coverage(candidates, demand, pairs, opts)
            m = dict(res.metrics); m.update({'city_id': target.city_id, 'country': target.country, 'data_mode': target.data_mode, 'seed': target.seed, 'method_id': 'M2_greedy_graph', 'shortlist_size': np.nan})
            m.update(compare_to_oracle(m, oracle_metrics))
            m['topK_jaccard'] = jaccard(set(res.selected_candidate_ids), set(oracle_labels.loc[oracle_labels['oracle_selected']==1,'candidate_id'].astype(str)))
            result_rows.append(m)
            sels = candidates[candidates['candidate_id'].astype(str).isin(set(res.selected_candidate_ids))].copy(); sels['method_id']='M2_greedy_graph'; sels['shortlist_size']=np.nan
            selected_rows.append(sels)

        # Map oracle vs best proposed if available.
        try:
            results_tmp = pd.DataFrame(result_rows)
            target_results = results_tmp[(results_tmp['city_id'] == target.city_id) & (results_tmp['method_id'].str.contains('synthetic_to_hybrid', na=False))]
            if not target_results.empty:
                best = target_results.sort_values('objective_ratio', ascending=False).iloc[0]
                best_sel = pd.concat(selected_rows, ignore_index=True)
                best_set = set(best_sel[(best_sel['method_id'] == best['method_id']) & (best_sel['shortlist_size'] == best['shortlist_size'])]['candidate_id'].astype(str))
                oracle_set = set(oracle_labels.loc[oracle_labels['oracle_selected']==1,'candidate_id'].astype(str))
                plot_selected_sites_map(target.nodes, target.edges, candidates, oracle_set, best_set, out_dirs['figures'], formats=cfg.get('plots', {}).get('figure_format', ['png','pdf']), dpi=int(cfg.get('plots', {}).get('dpi', 300)), max_edges=int(cfg.get('plots', {}).get('max_edges_on_map', 10000)))
        except Exception:
            pass

    results = pd.DataFrame(result_rows)
    results.to_csv(out_dirs['transfer'] / 'transfer_results.csv', index=False)
    export_table(results, out_dirs['tables'], 'T4_main_results')

    rankings = pd.concat(ranking_rows, ignore_index=True) if ranking_rows else pd.DataFrame()
    if not rankings.empty:
        rankings.to_csv(out_dirs['rankings'] / 'ranked_candidates_all_methods.csv', index=False)
        real_ranks = rankings[rankings.get('data_mode', '') == 'real'] if 'data_mode' in rankings.columns else rankings
        real_ranks.to_csv(out_dirs['rankings'] / 'ranked_candidates_real_mode.csv', index=False)
        top20 = rankings[rankings['rank'] <= 20].copy()
        top20.to_csv(out_dirs['rankings'] / 'top20_ranked_sites_by_method.csv', index=False)
        export_table(top20, out_dirs['tables'], 'T7_top20_ranked_sites')
        plot_ranked_candidate_comparison(rankings, out_dirs['figures'], cfg.get('plots', {}).get('figure_format', ['png','pdf']), int(cfg.get('plots', {}).get('dpi', 300)))

    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    if not selected.empty:
        selected.to_csv(out_dirs['transfer'] / 'selected_sites_by_method.csv', index=False)
        export_table(selected, out_dirs['tables'], 'T8_final_selected_sites')

    fi_all = pd.concat(fi_rows, ignore_index=True) if fi_rows else pd.DataFrame()
    if not fi_all.empty:
        fi_all.to_csv(out_dirs['tables'] / 'T10_feature_importance.csv', index=False)
        plot_feature_importance(fi_all.groupby('feature', as_index=False)['importance'].mean(), out_dirs['figures'], cfg.get('plots', {}).get('figure_format', ['png','pdf']), int(cfg.get('plots', {}).get('dpi', 300)))

    # Ablation summary as selected slices of main results.
    ablation = results.copy()
    export_table(ablation, out_dirs['tables'], 'T5_ablation_results')

    if run_sensitivity and cfg.get('sensitivity', {}).get('enabled', True):
        sens = run_sensitivity_block(instances, prepared, cfg, out_dirs, results)
        export_table(sens, out_dirs['tables'], 'T6_sensitivity_results')
        sens.to_csv(out_dirs['sensitivity'] / 'sensitivity_results.csv', index=False)
        plot_sensitivity_summary(sens, out_dirs['figures'], cfg.get('plots', {}).get('figure_format', ['png','pdf']), int(cfg.get('plots', {}).get('dpi', 300)))

    # Main plots.
    plot_objective_ratio(results, out_dirs['figures'], cfg.get('plots', {}).get('figure_format', ['png','pdf']), int(cfg.get('plots', {}).get('dpi', 300)))
    plot_coverage_by_shortlist(results, out_dirs['figures'], cfg.get('plots', {}).get('figure_format', ['png','pdf']), int(cfg.get('plots', {}).get('dpi', 300)))
    plot_recall_at_m(results, out_dirs['figures'], cfg.get('plots', {}).get('figure_format', ['png','pdf']), int(cfg.get('plots', {}).get('dpi', 300)))
    plot_runtime_quality(results, out_dirs['figures'], cfg.get('plots', {}).get('figure_format', ['png','pdf']), int(cfg.get('plots', {}).get('dpi', 300)))


def run_sensitivity_block(instances: List[DatasetInstance], prepared: Dict, cfg: Dict, out_dirs: Dict, main_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    real_instances = [i for i in instances if i.data_mode == 'real']
    if not real_instances:
        return pd.DataFrame()
    # Keep sensitivity modest: oracle only; and KPI/best proposed if rankings available are represented in main experiment.
    for inst in real_instances:
        for k in cfg.get('sensitivity', {}).get('k_values', [5,10,20]):
            metrics, *_ = run_oracle_for_instance(inst, cfg, out_dirs, k=int(k), model_id_suffix=f'_sens_K{k}')
            metrics.update({'parameter': 'K', 'parameter_value': k, 'method_id': 'oracle'})
            rows.append(metrics)
        for r in cfg.get('sensitivity', {}).get('radius_values_km', [3.0,5.0,8.0]):
            metrics, *_ = run_oracle_for_instance(inst, cfg, out_dirs, radius_km=float(r), model_id_suffix=f'_sens_R{r}')
            metrics.update({'parameter': 'R', 'parameter_value': r, 'method_id': 'oracle'})
            rows.append(metrics)
        for cm in cfg.get('sensitivity', {}).get('capacity_multipliers', [0.75,1.0,1.25]):
            metrics, *_ = run_oracle_for_instance(inst, cfg, out_dirs, capacity_multiplier=float(cm), model_id_suffix=f'_sens_C{cm}')
            metrics.update({'parameter': 'capacity_multiplier', 'parameter_value': cm, 'method_id': 'oracle'})
            rows.append(metrics)
        for pair in cfg.get('sensitivity', {}).get('weight_pairs', [[0.5,0.25]]):
            beta, mu = float(pair[0]), float(pair[1])
            metrics, *_ = run_oracle_for_instance(inst, cfg, out_dirs, beta=beta, mu=mu, model_id_suffix=f'_sens_B{beta}_M{mu}')
            metrics.update({'parameter': 'weights_beta_mu', 'parameter_value': f'{beta},{mu}', 'method_id': 'oracle'})
            rows.append(metrics)
    return pd.DataFrame(rows)
