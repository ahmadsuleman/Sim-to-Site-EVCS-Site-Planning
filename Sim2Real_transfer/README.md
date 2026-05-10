# EVCS Simulator-to-Real Transfer Optimization

This package implements a fixed-seed simulator-to-real EV charging station siting experiment.
It evaluates whether synthetic and hybrid simulator outputs can train a candidate-ranking surrogate
that shortlists sites before a graph-based MILP optimizer makes the final station-placement decision.

## Core methods

- Random top-M + MILP
- KPI top-M + MILP
- Greedy graph coverage
- Synthetic-trained ranker + top-M MILP
- Hybrid-trained ranker + top-M MILP
- Synthetic→Hybrid ranker + top-M MILP
- Full real-mode MILP oracle

## Input

Create a `dataset_index.csv`:

```csv
city_id,country,data_mode,seed,data_dir
omn_nizwa,Oman,real,42,data/omn_nizwa/real
omn_nizwa,Oman,hybrid,42,data/omn_nizwa/hybrid
omn_nizwa,Oman,synthetic,42,data/omn_nizwa/synthetic
```

Each `data_dir` should contain simulator exports such as:

- `graph_nodes.csv` or `nodes*.csv`
- `graph_edges.csv` or `edges*.csv`
- `candidate_sites*.csv`
- `candidate_feature_matrix*.csv`
- `candidate_kpis*.csv`
- `demand_points*.csv`
- `validation_report.json`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python run_experiment.py \
  --dataset-index dataset_index.csv \
  --output-dir outputs_sim2real \
  --config config/experiment_config.yaml
```

To skip sensitivity while debugging:

```bash
python run_experiment.py \
  --dataset-index dataset_index.csv \
  --output-dir outputs_debug \
  --config config/experiment_config.yaml \
  --skip-sensitivity
```

## Outputs

The pipeline exports:

- `outputs/metadata`: dataset inventory and validation reports
- `outputs/oracle`: full MILP oracle results
- `outputs/labels`: candidate-level oracle labels
- `outputs/rankings`: ranked candidate lists for all methods
- `outputs/transfer`: selected sites and transfer metrics
- `outputs/tables`: CSV, XLSX, and LaTeX tables
- `outputs/figures`: PNG and PDF figures

## Notes

- Seed is fixed at 42.
- Candidate ranking is explainable through exported ranked tables and feature importance.
- The learned model only shortlists candidates. Final site placement is made by graph optimization.
- If the MILP solver fails or times out, the result is marked and can fall back to a greedy graph solver depending on config.

## Optional: make a dataset index automatically

If your exports are arranged under one root folder, try:

```bash
python make_dataset_index.py --root data --output dataset_index.csv --seed 42
```

Check the generated CSV before running the experiment.
