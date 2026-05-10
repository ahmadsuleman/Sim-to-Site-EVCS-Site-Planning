from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
    SCIPY_MILP_AVAILABLE = True
except Exception:  # pragma: no cover
    SCIPY_MILP_AVAILABLE = False


@dataclass
class OptimizationOptions:
    model_id: str
    k: int
    radius_km: float
    coverage_weight: float = 1.0
    suitability_weight: float = 0.5
    cost_weight: float = 0.25
    use_capacity: bool = True
    solver: str = 'scipy_milp'
    time_limit_seconds: int = 120
    mip_gap: float = 0.03
    fallback_to_greedy: bool = True


@dataclass
class OptimizationResult:
    model_id: str
    status: str
    solver_kind: str
    objective_value: float
    runtime_seconds: float
    selected_candidate_ids: List[str]
    assignments: pd.DataFrame
    metrics: Dict


def _evaluate_solution(candidates: pd.DataFrame, demand: pd.DataFrame, pairs: pd.DataFrame, selected: Sequence[str], assignments: pd.DataFrame, options: OptimizationOptions, status: str, solver_kind: str, objective_value: float, runtime: float) -> OptimizationResult:
    total_demand = float(pd.to_numeric(demand.get('demand_mass', pd.Series(np.ones(len(demand)))), errors='coerce').fillna(0).sum())
    selected = [str(x) for x in selected]
    selected_df = candidates[candidates['candidate_id'].astype(str).isin(selected)].copy()
    covered_demand = float(assignments['demand_mass'].sum()) if not assignments.empty and 'demand_mass' in assignments.columns else 0.0
    coverage_rate = covered_demand / total_demand if total_demand > 0 else np.nan
    mean_dist = float(assignments['distance'].mean()) if not assignments.empty and 'distance' in assignments.columns else np.nan
    max_dist = float(assignments['distance'].max()) if not assignments.empty and 'distance' in assignments.columns else np.nan
    sel_suit = float(selected_df['suitability_score'].mean()) if 'suitability_score' in selected_df else np.nan
    sel_cost = float(selected_df['cost_proxy_model'].mean()) if 'cost_proxy_model' in selected_df else np.nan
    sel_unc = float(selected_df['uncertainty_score'].mean()) if 'uncertainty_score' in selected_df else np.nan
    cap_violation_count = 0
    cap_violation_demand = 0.0
    min_slack = np.nan
    mean_slack = np.nan
    if options.use_capacity and 'capacity_model' in selected_df.columns:
        assigned_by_c = assignments.groupby('candidate_id')['demand_mass'].sum() if not assignments.empty else pd.Series(dtype=float)
        slacks = []
        for _, r in selected_df.iterrows():
            assigned = float(assigned_by_c.get(str(r['candidate_id']), 0.0))
            cap = float(r['capacity_model'])
            slack = cap - assigned
            slacks.append(slack)
            if slack < -1e-6:
                cap_violation_count += 1
                cap_violation_demand += -slack
        if slacks:
            min_slack = float(np.min(slacks))
            mean_slack = float(np.mean(slacks))
    # Proxy underserved metric: bottom accessibility / high equity_need if columns available in demand.
    underserved_coverage_rate = np.nan
    underserved_total = np.nan
    underserved_covered = np.nan
    if len(demand) and {'demand_id'}.issubset(demand.columns):
        d = demand.copy()
        if 'equity_need_score' in d.columns:
            threshold = pd.to_numeric(d['equity_need_score'], errors='coerce').quantile(0.75)
            d['underserved_flag'] = pd.to_numeric(d['equity_need_score'], errors='coerce') >= threshold
        elif 'activity_intensity' in d.columns:
            threshold = pd.to_numeric(d['activity_intensity'], errors='coerce').quantile(0.25)
            d['underserved_flag'] = pd.to_numeric(d['activity_intensity'], errors='coerce') <= threshold
        else:
            d['underserved_flag'] = False
        underserved_ids = set(d.loc[d['underserved_flag'], 'demand_id'].astype(str))
        underserved_total = float(d.loc[d['underserved_flag'], 'demand_mass'].sum()) if 'demand_mass' in d else 0.0
        if not assignments.empty:
            underserved_covered = float(assignments.loc[assignments['demand_id'].astype(str).isin(underserved_ids), 'demand_mass'].sum())
        else:
            underserved_covered = 0.0
        underserved_coverage_rate = underserved_covered / underserved_total if underserved_total > 0 else np.nan
    metrics = {
        'metrics_valid': status in {'Optimal', 'Feasible', 'Greedy'},
        'selected_count': len(selected),
        'k': options.k,
        'radius_km': options.radius_km,
        'total_demand': total_demand,
        'covered_demand': covered_demand,
        'coverage_rate': coverage_rate,
        'underserved_total_demand': underserved_total,
        'underserved_covered_demand': underserved_covered,
        'underserved_coverage_rate': underserved_coverage_rate,
        'mean_assigned_distance': mean_dist,
        'max_assigned_distance': max_dist,
        'selected_mean_suitability': sel_suit,
        'selected_mean_cost_proxy': sel_cost,
        'selected_mean_uncertainty': sel_unc,
        'capacity_violation_count': cap_violation_count,
        'capacity_violation_demand': cap_violation_demand,
        'min_capacity_slack': min_slack,
        'mean_capacity_slack': mean_slack,
        'runtime_seconds': runtime,
        'objective_value': objective_value,
        'solver_kind': solver_kind,
        'status': status,
        'model_id': options.model_id,
    }
    return OptimizationResult(options.model_id, status, solver_kind, objective_value, runtime, selected, assignments, metrics)


def _greedy_assign(candidates: pd.DataFrame, demand: pd.DataFrame, pairs: pd.DataFrame, selected: Sequence[str], use_capacity: bool = True) -> pd.DataFrame:
    if pairs.empty or not selected:
        return pd.DataFrame(columns=['candidate_id','demand_id','distance','demand_mass'])
    selected = set(map(str, selected))
    p = pairs[pairs['candidate_id'].astype(str).isin(selected)].copy()
    if p.empty:
        return pd.DataFrame(columns=['candidate_id','demand_id','distance','demand_mass'])
    cap = candidates.set_index(candidates['candidate_id'].astype(str)).get('capacity_model', pd.Series(np.inf, index=candidates['candidate_id'].astype(str))).to_dict()
    used = {cid: 0.0 for cid in selected}
    rows = []
    # Assign each demand to nearest selected site that has capacity.
    for did, grp in p.sort_values('distance').groupby('demand_id'):
        q = float(grp['demand_mass'].iloc[0])
        for _, r in grp.iterrows():
            cid = str(r['candidate_id'])
            if (not use_capacity) or (used.get(cid, 0.0) + q <= float(cap.get(cid, np.inf)) + 1e-9):
                used[cid] = used.get(cid, 0.0) + q
                rows.append({'candidate_id': cid, 'demand_id': str(did), 'distance': float(r['distance']), 'demand_mass': q})
                break
    return pd.DataFrame(rows)


def greedy_coverage(candidates: pd.DataFrame, demand: pd.DataFrame, pairs: pd.DataFrame, options: OptimizationOptions) -> OptimizationResult:
    start = time.time()
    selected = []
    covered = set()
    pair_by_c = {cid: set(g['demand_id'].astype(str)) for cid, g in pairs.groupby('candidate_id')} if not pairs.empty else {}
    demand_mass = demand.set_index(demand['demand_id'].astype(str))['demand_mass'].astype(float).to_dict() if 'demand_mass' in demand else {}
    candidates_ids = candidates['candidate_id'].astype(str).tolist()
    for _ in range(min(options.k, len(candidates_ids))):
        best_c = None
        best_score = -np.inf
        for cid in candidates_ids:
            if cid in selected:
                continue
            new = pair_by_c.get(cid, set()) - covered
            gain = sum(float(demand_mass.get(d, 0.0)) for d in new)
            row = candidates.loc[candidates['candidate_id'].astype(str) == cid]
            suit = float(row['suitability_score'].iloc[0]) if len(row) and 'suitability_score' in row else 0.0
            cost = float(row['cost_proxy_model'].iloc[0]) if len(row) and 'cost_proxy_model' in row else 0.0
            score = gain + options.suitability_weight * suit - options.cost_weight * cost
            if score > best_score:
                best_score = score
                best_c = cid
        if best_c is None:
            break
        selected.append(best_c)
        covered |= pair_by_c.get(best_c, set())
    assignments = _greedy_assign(candidates, demand, pairs, selected, use_capacity=options.use_capacity)
    runtime = time.time() - start
    objective = float(assignments['demand_mass'].sum()) if not assignments.empty else 0.0
    return _evaluate_solution(candidates, demand, pairs, selected, assignments, options, 'Greedy', 'greedy', objective, runtime)


def solve_location_allocation(candidates: pd.DataFrame, demand: pd.DataFrame, pairs: pd.DataFrame, options: OptimizationOptions) -> OptimizationResult:
    candidates = candidates.copy()
    candidates['candidate_id'] = candidates['candidate_id'].astype(str)
    demand = demand.copy()
    demand['demand_id'] = demand['demand_id'].astype(str)
    if len(candidates) <= options.k:
        selected = candidates['candidate_id'].astype(str).tolist()
        assignments = _greedy_assign(candidates, demand, pairs, selected, options.use_capacity)
        return _evaluate_solution(candidates, demand, pairs, selected, assignments, options, 'Feasible', 'trivial', float(assignments['demand_mass'].sum()), 0.0)
    if options.solver != 'scipy_milp' or not SCIPY_MILP_AVAILABLE or pairs.empty:
        return greedy_coverage(candidates, demand, pairs, options)
    start = time.time()
    cids = candidates['candidate_id'].astype(str).tolist()
    cid_to_i = {c:i for i,c in enumerate(cids)}
    pairs = pairs[pairs['candidate_id'].astype(str).isin(cid_to_i)].copy()
    pairs = pairs[pairs['demand_id'].astype(str).isin(set(demand['demand_id'].astype(str)))].copy()
    if pairs.empty:
        return greedy_coverage(candidates, demand, pairs, options)
    pairs['candidate_id'] = pairs['candidate_id'].astype(str)
    pairs['demand_id'] = pairs['demand_id'].astype(str)
    n = len(cids)
    m = len(pairs)
    total_vars = n + m
    total_demand = float(demand['demand_mass'].astype(float).sum()) if 'demand_mass' in demand else 1.0
    total_demand = max(total_demand, 1e-9)
    # Objective: maximize normalized coverage + avg selected suitability - avg selected cost.
    c = np.zeros(total_vars, dtype=float)
    suit = candidates.set_index('candidate_id').get('suitability_score', pd.Series(0.0, index=cids)).reindex(cids).fillna(0).to_numpy(float)
    cost = candidates.set_index('candidate_id').get('cost_proxy_model', pd.Series(0.0, index=cids)).reindex(cids).fillna(0).to_numpy(float)
    c[:n] = -(options.suitability_weight * suit / max(options.k, 1) - options.cost_weight * cost / max(options.k, 1))
    q = pairs['demand_mass'].astype(float).to_numpy()
    c[n:] = -(options.coverage_weight * q / total_demand)
    integrality = np.ones(total_vars, dtype=int)
    bounds = Bounds(lb=np.zeros(total_vars), ub=np.ones(total_vars))
    rows = []
    lbs = []
    ubs = []
    row_idx = 0
    # sum x = k
    rows_i = list(range(n)); cols_i = list(range(n)); data_i = [1.0]*n
    row_blocks = [(rows_i, cols_i, data_i, options.k, options.k)]
    # y <= x: y_p - x_i <= 0
    rp, cp, dp = [], [], []
    for p_idx, r in enumerate(pairs.itertuples(index=False)):
        cid = str(getattr(r, 'candidate_id'))
        i = cid_to_i[cid]
        rp.extend([p_idx, p_idx]); cp.extend([n+p_idx, i]); dp.extend([1.0, -1.0])
    # Demand assigned once
    dem_groups = pairs.groupby('demand_id').indices
    # Capacity constraints
    capacity_rows = []
    # Build sparse constraints incrementally.
    A_rows=[]; A_cols=[]; A_data=[]; lb=[]; ub=[]; rr=0
    A_rows.extend([rr]*n); A_cols.extend(range(n)); A_data.extend([1.0]*n); lb.append(options.k); ub.append(options.k); rr += 1
    for p_idx, r in enumerate(pairs.itertuples(index=False)):
        i = cid_to_i[str(getattr(r, 'candidate_id'))]
        A_rows.extend([rr, rr]); A_cols.extend([n+p_idx, i]); A_data.extend([1.0, -1.0]); lb.append(-np.inf); ub.append(0.0); rr += 1
    for did, idxs in dem_groups.items():
        idxs = list(idxs)
        A_rows.extend([rr]*len(idxs)); A_cols.extend([n+i for i in idxs]); A_data.extend([1.0]*len(idxs)); lb.append(-np.inf); ub.append(1.0); rr += 1
    if options.use_capacity and 'capacity_model' in candidates.columns:
        cap = candidates.set_index('candidate_id')['capacity_model'].reindex(cids).fillna(total_demand).to_numpy(float)
        for cid, idxs in pairs.groupby('candidate_id').indices.items():
            i = cid_to_i[str(cid)]
            idxs = list(idxs)
            qs = pairs.iloc[idxs]['demand_mass'].astype(float).to_numpy()
            A_rows.append(rr); A_cols.append(i); A_data.append(-float(cap[i]))
            A_rows.extend([rr]*len(idxs)); A_cols.extend([n+j for j in idxs]); A_data.extend(qs.tolist())
            lb.append(-np.inf); ub.append(0.0); rr += 1
    A = sparse.coo_matrix((A_data, (A_rows, A_cols)), shape=(rr, total_vars)).tocsr()
    constraints = LinearConstraint(A, np.array(lb), np.array(ub))
    try:
        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints, options={'time_limit': options.time_limit_seconds, 'mip_rel_gap': options.mip_gap, 'disp': False})
        runtime = time.time() - start
        status = 'Optimal' if res.success else f'MILP_{res.message}'
        if not res.success or res.x is None:
            if options.fallback_to_greedy:
                fallback = greedy_coverage(candidates, demand, pairs, options)
                fallback.status = f'Fallback_after_{status}'
                fallback.metrics['status'] = fallback.status
                return fallback
            return _evaluate_solution(candidates, demand, pairs, [], pd.DataFrame(), options, status, 'scipy_milp', np.nan, runtime)
        x = res.x[:n]
        y = res.x[n:]
        selected = [cids[i] for i, val in enumerate(x) if val >= 0.5]
        assign_rows = []
        for p_idx, val in enumerate(y):
            if val >= 0.5:
                r = pairs.iloc[p_idx]
                assign_rows.append({'candidate_id': str(r['candidate_id']), 'demand_id': str(r['demand_id']), 'distance': float(r['distance']), 'demand_mass': float(r['demand_mass'])})
        assignments = pd.DataFrame(assign_rows)
        objective = float(-res.fun) if np.isfinite(res.fun) else np.nan
        return _evaluate_solution(candidates, demand, pairs, selected, assignments, options, status, 'scipy_milp', objective, runtime)
    except Exception as e:
        runtime = time.time() - start
        if options.fallback_to_greedy:
            fallback = greedy_coverage(candidates, demand, pairs, options)
            fallback.status = f'Fallback_exception_{type(e).__name__}'
            fallback.metrics['status'] = fallback.status
            return fallback
        return _evaluate_solution(candidates, demand, pairs, [], pd.DataFrame(), options, f'Exception_{type(e).__name__}', 'scipy_milp', np.nan, runtime)
