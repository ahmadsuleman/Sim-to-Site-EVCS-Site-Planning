#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pandas as pd
from src.config import load_config, ensure_dirs
from src.data_loader import load_dataset_index, load_instance
from src.pipeline import run_oracle_for_instance


def main():
    p = argparse.ArgumentParser(description='Run full MILP/greedy oracles for all dataset instances.')
    p.add_argument('--dataset-index', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--config', default='config/experiment_config.yaml')
    args = p.parse_args()
    cfg = load_config(args.config)
    out_dirs = ensure_dirs(args.output_dir)
    index = load_dataset_index(args.dataset_index, expected_seed=int(cfg.get('seed', 42)))
    rows = []
    for _, row in index.iterrows():
        inst = load_instance(row)
        metrics, labels, *_ = run_oracle_for_instance(inst, cfg, out_dirs)
        rows.append(metrics)
        labels.to_csv(out_dirs['labels'] / f'{inst.instance_id}_oracle_labels.csv', index=False)
    pd.DataFrame(rows).to_csv(out_dirs['oracle'] / 'oracle_results.csv', index=False)
    print(f'Oracle runs complete: {args.output_dir}')

if __name__ == '__main__':
    main()
