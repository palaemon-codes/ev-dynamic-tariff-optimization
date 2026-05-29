import argparse
from typing import Any

from .data import build_analytics_base
from .modeling import run_demand_prediction_agent
from .pricing import optimize_dynamic_tariffs
from .reporting import save_outputs


def run_pipeline(use_synthetic_if_missing: bool = True, skip_figures: bool = False) -> dict[str, Any]:
    analytics_df, source_summary = build_analytics_base(use_synthetic_if_missing=use_synthetic_if_missing)
    prediction_metrics, predictions, modeling_df = run_demand_prediction_agent(analytics_df)
    recommendation_details, recommendation_metrics, monitoring_feedback = optimize_dynamic_tariffs(modeling_df, predictions)

    output_paths = save_outputs(
        analytics_df=analytics_df,
        prediction_metrics=prediction_metrics,
        predictions=predictions,
        recommendation_metrics=recommendation_metrics,
        recommendation_details=recommendation_details,
        monitoring_feedback=monitoring_feedback,
        source_summary=source_summary,
        generate_figures=not skip_figures,
    )
    return output_paths


def cli() -> int:
    parser = argparse.ArgumentParser(description="Run the EV tariff optimization project pipeline.")
    parser.add_argument("--no-synthetic", action="store_true", help="Fail if no official raw dataset is present.")
    parser.add_argument("--skip-figures", action="store_true", help="Keep the output generation narrow during smoke runs.")
    args = parser.parse_args()

    output_paths = run_pipeline(use_synthetic_if_missing=not args.no_synthetic, skip_figures=args.skip_figures)
    print("Pipeline completed.")
    for label, path in output_paths.items():
        print(f"{label}: {path}")
    return 0
