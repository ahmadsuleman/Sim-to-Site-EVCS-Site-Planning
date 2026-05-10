#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.config import load_config
from src.pipeline import run_experiment


def main():
    parser = argparse.ArgumentParser(description='Fixed-seed sim-to-real EVCS siting experiment.')
    parser.add_argument('--dataset-index', required=True, help='CSV with city_id,country,data_mode,seed,data_dir')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--config', default='config/experiment_config.yaml', help='YAML config')
    parser.add_argument('--skip-sensitivity', action='store_true', help='Skip sensitivity analysis')
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_experiment(args.dataset_index, args.output_dir, cfg, run_sensitivity=not args.skip_sensitivity)
    print(f'Completed EVCS sim-to-real experiment. Outputs written to: {args.output_dir}')


if __name__ == '__main__':
    main()
