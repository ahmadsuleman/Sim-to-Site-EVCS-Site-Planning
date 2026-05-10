from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

# Support: python src/run_pipeline.py ...
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.common import ensure_run_dirs, get_city, list_cities, load_config
from src.graph_builders.synthetic_graph_builder import SyntheticGraphBuilder
from src.graph_builders.real_osm_graph_builder import RealOSMGraphBuilder
from src.graph_builders.hybrid_graph_builder import HybridGraphBuilder
from src.features.feature_pipeline import FeaturePipeline
from src.candidates.candidate_generator import CandidateGenerator
from src.demand.demand_node_generator import DemandNodeGenerator
from src.kpis.candidate_kpi_engine import CandidateKPIEngine
from src.exporters.csv_exporter import CSVExporter
from src.exporters.graph_exporter import GraphExporter
from src.validation.pipeline_validator import PipelineValidator, metrics_dict_to_frame
from src.validation.report_builder import ReportBuilder
from src.visualization.visualizer import Visualizer


def build_graph(city, mode: str, seed: int, config: Dict[str, Any]):
    defaults = config["pipeline_config"]["defaults"]
    if mode == "synthetic":
        return SyntheticGraphBuilder(city, seed, int(defaults.get("synthetic_node_count", 324))).build(data_mode=mode)
    if mode == "real":
        return RealOSMGraphBuilder(city, seed, defaults).build(data_mode=mode)
    if mode == "hybrid":
        return HybridGraphBuilder(city, seed, defaults).build(data_mode=mode)
    raise ValueError(f"Unsupported mode: {mode}")


def build_candidate_feature_matrix(city, mode: str, nodes: pd.DataFrame, candidates: pd.DataFrame, kpis: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
    node_cols = [
        "node_id", "degree_norm", "betweenness_centrality", "closeness_centrality", "road_hierarchy_score",
        "reachability_5min", "reachability_10min", "reachability_15min", "population_density", "activity_intensity",
        "traffic_flow_proxy", "od_inflow_proxy", "od_outflow_proxy", "parking_dwell_score", "grid_access_score",
        "land_cost_proxy", "available_space", "nearby_charger_pressure", "equity_need_score", "service_coverage_gain",
        "feature_confidence_mean", "uncertainty_score"
    ]
    merged = candidates.merge(nodes[node_cols], on="node_id", suffixes=("", "_node"), how="left")
    # Prefer candidate-level features where duplicated.
    for col in ["available_space", "parking_dwell_score", "grid_access_score", "land_cost_proxy", "nearby_charger_pressure"]:
        node_col = f"{col}_node"
        if node_col in merged.columns:
            merged[col] = merged[col].combine_first(merged[node_col])
            merged.drop(columns=[node_col], inplace=True)
    merged = merged.merge(kpis, on=["candidate_id", "city_id", "data_mode"], how="left", suffixes=("", "_kpi"))
    merged["country"] = city.country
    expected = schema["candidate_feature_matrix"]
    for col in expected:
        if col not in merged.columns:
            merged[col] = None
    return merged[expected]


def run_one(city_id: str, mode: str, seed: int, output_dir: Path, config: Dict[str, Any]) -> Path:
    city = get_city(config, city_id)
    if mode not in city.mode_enabled:
        raise ValueError(f"Mode {mode} is not enabled for {city_id}")
    dirs = ensure_run_dirs(output_dir)

    G, nodes_base, edges, graph_meta = build_graph(city, mode, seed, config)
    nodes, provenance, randomization_log = FeaturePipeline(city, mode, seed, config, graph_meta).build(G, nodes_base, edges)

    candidates, candidate_prov = CandidateGenerator(city, mode, seed, config).build(nodes)
    if not candidate_prov.empty:
        provenance = pd.concat([provenance, candidate_prov], ignore_index=True)

    demand = DemandNodeGenerator(city, mode, config).build(nodes)
    kpis = CandidateKPIEngine(city, mode).build(candidates, nodes, demand)
    matrix = build_candidate_feature_matrix(city, mode, nodes, candidates, kpis, config["feature_schema"])

    validation = PipelineValidator().validate(G, nodes, edges, candidates, demand, matrix, provenance, randomization_log)
    graph_metrics = validation["graph_metrics"]
    feature_quality = validation["feature_quality"]
    corr_matrix = validation["correlation_matrix"]
    corr_report = validation["correlation_report"]
    provenance_summary = validation["provenance_summary"]

    extra = {
        "city_id": city.city_id,
        "country": city.country,
        "data_mode": mode,
        "seed": seed,
        "graph_source": graph_meta.get("graph_source", "unknown"),
        "fallback_used": graph_meta.get("fallback_used", "false"),
    }
    validation_metrics_frame = metrics_dict_to_frame(graph_metrics, extra=extra)
    spatial_rows = [{"metric": f"neighbor_autocorrelation_{k}", "value": v} for k, v in validation["spatial_autocorrelation"].items()]
    if spatial_rows:
        validation_metrics_frame = pd.concat([validation_metrics_frame, pd.DataFrame(spatial_rows)], ignore_index=True)

    simulation_summary = pd.DataFrame([{**extra, **graph_metrics, **provenance_summary}])

    csv = CSVExporter(dirs["csv"])
    csv.write_many({
        "nodes.csv": nodes,
        "edges.csv": edges,
        "candidate_sites.csv": candidates,
        "demand_points.csv": demand,
        "candidate_kpis.csv": kpis,
        "candidate_feature_matrix.csv": matrix,
        "feature_provenance.csv": provenance,
        "feature_quality_report.csv": feature_quality,
        "domain_randomization_log.csv": randomization_log,
        "validation_metrics.csv": validation_metrics_frame,
        "correlation_matrix.csv": corr_matrix.reset_index().rename(columns={"index": "feature"}) if not corr_matrix.empty else pd.DataFrame(),
        "correlation_validity_report.csv": corr_report,
        "simulation_summary.csv": simulation_summary,
    })

    GraphExporter(dirs["graph"]).export(
        G,
        nodes,
        edges,
        export_graphml=bool(config["pipeline_config"]["defaults"].get("export_graphml", True)),
        export_gpkg_if_available=bool(config["pipeline_config"]["defaults"].get("export_gpkg_if_available", True)),
    )

    ReportBuilder().build_reports(
        dirs["reports"], city, mode, seed, graph_meta, graph_metrics, provenance_summary, feature_quality, corr_report
    )

    if bool(config["pipeline_config"]["defaults"].get("output_plots", True)):
        Visualizer(dirs["plots"]).build_all(nodes, edges, candidates, demand, kpis, matrix, provenance, validation_metrics_frame)

    return output_dir


def validate_only(input_dir: Path) -> None:
    csv_dir = input_dir / "csv"
    required = ["candidate_feature_matrix.csv", "feature_provenance.csv", "domain_randomization_log.csv"]
    missing = [name for name in required if not (csv_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required validation files in {csv_dir}: {missing}")
    matrix = pd.read_csv(csv_dir / "candidate_feature_matrix.csv")
    provenance = pd.read_csv(csv_dir / "feature_provenance.csv")
    randomization_log = pd.read_csv(csv_dir / "domain_randomization_log.csv")
    from src.validation.feature_validation import FeatureValidator
    from src.validation.kpi_validation import KPIValidator
    from src.validation.provenance_validation import ProvenanceValidator
    feature_quality = FeatureValidator().validate(matrix)
    corr = KPIValidator().correlation_matrix(matrix)
    corr_report = KPIValidator().correlation_report(matrix)
    prov = ProvenanceValidator().summarize(provenance, randomization_log)
    feature_quality.to_csv(csv_dir / "feature_quality_report.csv", index=False)
    corr.reset_index().rename(columns={"index": "feature"}).to_csv(csv_dir / "correlation_matrix.csv", index=False)
    corr_report.to_csv(csv_dir / "correlation_validity_report.csv", index=False)
    pd.DataFrame([prov]).to_csv(csv_dir / "provenance_validation_summary.csv", index=False)
    print(f"Validation refreshed for {input_dir}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="EV graph-structured dataset generation pipeline")
    parser.add_argument("--city", help="City id from config/cities.yaml")
    parser.add_argument("--mode", choices=["synthetic", "real", "hybrid"], default="synthetic")
    parser.add_argument("--modes", nargs="+", choices=["synthetic", "real", "hybrid"], default=None)
    parser.add_argument("--all_cities", action="store_true")
    parser.add_argument("--task", default="generate_dataset", choices=["generate_dataset"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/run"))
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--input_dir", type=Path)
    args = parser.parse_args(argv)

    config = load_config()
    if args.validate_only:
        if not args.input_dir:
            raise ValueError("--input_dir is required with --validate_only")
        validate_only(args.input_dir)
        return

    run_dirs = []
    if args.all_cities:
        modes = args.modes or [args.mode]
        for city in list_cities(config):
            for mode in modes:
                run_dir = args.output_dir / f"{city.city_id}_{mode}_seed{args.seed}"
                print(f"Generating {city.city_id} / {mode} -> {run_dir}")
                run_dirs.append(run_one(city.city_id, mode, args.seed, run_dir, config))
        if run_dirs:
            Visualizer(args.output_dir / "plots").build_city_comparisons(run_dirs, args.output_dir / "plots")
        print(f"Generated {len(run_dirs)} runs under {args.output_dir}")
    else:
        if not args.city:
            raise ValueError("--city is required unless --all_cities is used")
        out = run_one(args.city, args.mode, args.seed, args.output_dir, config)
        print(f"Generated dataset at {out}")


if __name__ == "__main__":
    main()
