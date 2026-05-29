import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, OUTPUT_DIR, PROCESSED_DIR


def save_outputs(
    analytics_df: pd.DataFrame,
    prediction_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    recommendation_metrics: pd.DataFrame,
    recommendation_details: pd.DataFrame,
    monitoring_feedback: pd.DataFrame,
    source_summary: dict,
    generate_figures: bool = True,
) -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    analytics_path = PROCESSED_DIR / "unified_ev_analytics.csv"
    forecast_path = OUTPUT_DIR / "demand_forecast.csv"
    recommendation_path = OUTPUT_DIR / "dynamic_tariff_recommendations.csv"
    feedback_path = OUTPUT_DIR / "monitoring_feedback.csv"
    metrics_path = OUTPUT_DIR / "metrics_summary.csv"
    summary_path = OUTPUT_DIR / "run_summary.json"

    analytics_df.to_csv(analytics_path, index=False)
    predictions.to_csv(forecast_path, index=False)
    recommendation_details.to_csv(recommendation_path, index=False)
    monitoring_feedback.to_csv(feedback_path, index=False)
    metrics_summary = pd.concat([prediction_metrics, recommendation_metrics], ignore_index=True)
    metrics_summary.to_csv(metrics_path, index=False)

    summary = {
        "rows_processed": int(len(analytics_df)),
        "stations_covered": int(analytics_df["station_id"].nunique()),
        "sources": sorted(analytics_df["source"].dropna().unique().tolist()),
        **source_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    if generate_figures:
        create_figures(analytics_df, recommendation_details)
    return {
        "analytics_path": str(analytics_path),
        "forecast_path": str(forecast_path),
        "recommendation_path": str(recommendation_path),
        "feedback_path": str(feedback_path),
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
    }


def create_figures(analytics_df: pd.DataFrame, recommendation_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    hourly_profile = analytics_df.groupby(["hour", "source"], as_index=False)["demand_kwh"].mean()
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=hourly_profile, x="hour", y="demand_kwh", hue="source", marker="o")
    plt.title("Average Charging Demand by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Demand (kWh)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hourly_demand_profile.png", dpi=180)
    plt.close()

    heatmap_data = analytics_df.pivot_table(index="day_of_week", columns="hour", values="utilization_rate", aggfunc="mean")
    plt.figure(figsize=(12, 5))
    sns.heatmap(heatmap_data, cmap="YlOrRd")
    plt.title("Average Utilization Heatmap")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "utilization_heatmap.png", dpi=180)
    plt.close()

    station_utilization = (
        analytics_df.groupby("station_id", as_index=False)["utilization_rate"].mean().sort_values("utilization_rate", ascending=False).head(15)
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=station_utilization, x="utilization_rate", y="station_id", hue="station_id", palette="crest", legend=False)
    plt.title("Top 15 Stations by Average Utilization")
    plt.xlabel("Average Utilization Rate")
    plt.ylabel("Station ID")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "station_utilization_rank.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plot_df = recommendation_df.sample(min(1500, len(recommendation_df)), random_state=42) if len(recommendation_df) > 1500 else recommendation_df
    sns.scatterplot(
        data=plot_df,
        x="predicted_utilization_rate",
        y="recommended_tariff_per_kwh",
        hue="tariff_signal",
        alpha=0.7,
    )
    plt.axhline(15.0, linestyle="--", color="black", linewidth=1)
    plt.title("Dynamic Tariff Recommendation by Predicted Utilization")
    plt.xlabel("Predicted Utilization Rate")
    plt.ylabel("Recommended Tariff (Rs./kWh)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tariff_response_map.png", dpi=180)
    plt.close()
