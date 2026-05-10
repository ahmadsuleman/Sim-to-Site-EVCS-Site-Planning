from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .io_utils import find_file, read_csv_if_exists, read_json_if_exists, standardize_id_columns


@dataclass
class DatasetInstance:
    city_id: str
    country: str
    data_mode: str
    seed: int
    data_dir: Path
    nodes: pd.DataFrame
    edges: pd.DataFrame
    candidates: pd.DataFrame
    features: pd.DataFrame
    kpis: pd.DataFrame
    demand: pd.DataFrame
    provenance: pd.DataFrame
    quality: pd.DataFrame
    validation: Dict

    @property
    def instance_id(self) -> str:
        return f'{self.city_id}_{self.data_mode}_seed{self.seed}'


def load_dataset_index(path: str | Path, expected_seed: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'city_id', 'country', 'data_mode', 'seed', 'data_dir'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'dataset_index is missing required columns: {missing}')
    df['data_mode'] = df['data_mode'].str.lower().str.strip()
    df['city_id'] = df['city_id'].astype(str)
    df['seed'] = df['seed'].astype(int)
    if expected_seed is not None:
        df = df[df['seed'] == int(expected_seed)].copy()
    if df.empty:
        raise ValueError('dataset_index has no rows after seed filtering.')
    return df.reset_index(drop=True)


def load_instance(row: pd.Series) -> DatasetInstance:
    data_dir = Path(row['data_dir'])
    files = {
        name: find_file(data_dir, name, required=(name in {'nodes','edges','candidates','features','demand'}))
        for name in ['nodes','edges','candidates','features','kpis','demand','provenance','quality','validation']
    }
    nodes = standardize_id_columns(read_csv_if_exists(files['nodes']))
    edges = standardize_id_columns(read_csv_if_exists(files['edges']))
    candidates = standardize_id_columns(read_csv_if_exists(files['candidates']))
    features = standardize_id_columns(read_csv_if_exists(files['features']))
    kpis = standardize_id_columns(read_csv_if_exists(files['kpis']))
    demand = standardize_id_columns(read_csv_if_exists(files['demand']))
    provenance = standardize_id_columns(read_csv_if_exists(files['provenance']))
    quality = standardize_id_columns(read_csv_if_exists(files['quality']))
    validation = read_json_if_exists(files['validation'])
    return DatasetInstance(
        city_id=str(row['city_id']),
        country=str(row.get('country', '')),
        data_mode=str(row['data_mode']).lower(),
        seed=int(row['seed']),
        data_dir=data_dir,
        nodes=nodes,
        edges=edges,
        candidates=candidates,
        features=features,
        kpis=kpis,
        demand=demand,
        provenance=provenance,
        quality=quality,
        validation=validation,
    )


def validate_index_coverage(index: pd.DataFrame, required_modes: List[str], allow_single_city_pilot: bool) -> pd.DataFrame:
    rows = []
    for city, grp in index.groupby('city_id'):
        modes = set(grp['data_mode'].str.lower())
        missing = sorted(set(required_modes) - modes)
        rows.append({
            'city_id': city,
            'available_modes': ','.join(sorted(modes)),
            'missing_modes': ','.join(missing),
            'has_all_required_modes': len(missing) == 0,
            'row_count': len(grp),
        })
    report = pd.DataFrame(rows)
    if (not allow_single_city_pilot) and (not report['has_all_required_modes'].all()):
        bad = report.loc[~report['has_all_required_modes'], ['city_id', 'missing_modes']]
        raise ValueError(f'Some cities are missing required modes:\n{bad.to_string(index=False)}')
    return report
