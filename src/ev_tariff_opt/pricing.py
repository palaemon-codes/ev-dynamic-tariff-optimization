import numpy as np
import pandas as pd

from .config import BASELINE_TARIFF, DEFAULT_SEED


def optimize_dynamic_tariffs(base_df: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recommendation_df = base_df.merge(
        predictions[
            [
                "timestamp",
                "station_id",
                "predicted_demand_kwh",
                "predicted_utilization_rate",
                "predicted_congestion_probability",
                "predicted_sessions",
            ]
        ],
        on=["timestamp", "station_id"],
        how="left",
    )
    recommendation_df["predicted_demand_kwh"] = recommendation_df["predicted_demand_kwh"].fillna(recommendation_df["demand_kwh"])
    recommendation_df["predicted_utilization_rate"] = recommendation_df["predicted_utilization_rate"].fillna(recommendation_df["utilization_rate"])
    recommendation_df["predicted_congestion_probability"] = recommendation_df["predicted_congestion_probability"].fillna((recommendation_df["utilization_rate"] >= 0.8).astype(float))
    recommendation_df["predicted_sessions"] = recommendation_df["predicted_sessions"].fillna(recommendation_df["sessions"])

    candidate_prices = BASELINE_TARIFF * np.array([0.75, 0.85, 0.95, 1.00, 1.05, 1.15, 1.25, 1.35])
    best_rows = []

    for row in recommendation_df.itertuples(index=False):
        base_demand = float(max(row.predicted_demand_kwh, 0.5))
        base_utilization = float(np.clip(row.predicted_utilization_rate, 0.0, 1.5))
        base_sessions = float(max(row.predicted_sessions, 1.0))
        elasticity = _segment_elasticity(row.pricing_segment, base_utilization)

        candidates = []
        for price in candidate_prices:
            pct_change = (price / BASELINE_TARIFF) - 1.0
            adjusted_demand = max(0.35 * base_demand, base_demand * (1.0 - elasticity * pct_change))
            adjusted_utilization = np.clip(base_utilization * (adjusted_demand / base_demand), 0.0, 1.5)
            adjusted_sessions = max(1.0, base_sessions * (adjusted_demand / base_demand))
            expected_wait_minutes = max(0.0, (adjusted_utilization - 0.75) * 40.0)
            expected_revenue = price * adjusted_demand
            congestion_penalty = 12.0 * expected_wait_minutes + 40.0 * max(0.0, adjusted_utilization - 0.95)
            off_peak_bonus = 45.0 * max(0.0, adjusted_sessions - base_sessions) if row.pricing_segment == "off_peak" else 0.0
            score = expected_revenue - congestion_penalty + off_peak_bonus

            candidates.append(
                {
                    "recommended_tariff_per_kwh": price,
                    "expected_demand_kwh": adjusted_demand,
                    "expected_utilization_rate": adjusted_utilization,
                    "expected_sessions": adjusted_sessions,
                    "expected_wait_time_minutes": expected_wait_minutes,
                    "expected_revenue": expected_revenue,
                    "elasticity": elasticity,
                    "score": score,
                }
            )

        best = max(candidates, key=lambda item: item["score"])
        best_rows.append(best)

    best_df = pd.DataFrame(best_rows)
    recommendation_df = pd.concat([recommendation_df.reset_index(drop=True), best_df.reset_index(drop=True)], axis=1)
    recommendation_df["baseline_revenue"] = BASELINE_TARIFF * recommendation_df["predicted_demand_kwh"]
    recommendation_df["revenue_gain_pct"] = ((recommendation_df["expected_revenue"] - recommendation_df["baseline_revenue"]) / recommendation_df["baseline_revenue"].clip(lower=1e-6)) * 100.0
    recommendation_df["tariff_signal"] = np.select(
        [recommendation_df["recommended_tariff_per_kwh"] > BASELINE_TARIFF, recommendation_df["recommended_tariff_per_kwh"] < BASELINE_TARIFF],
        ["surge", "discount"],
        default="hold",
    )
    recommendation_df["expected_pricing_efficiency"] = recommendation_df["expected_revenue"] / recommendation_df["expected_demand_kwh"].clip(lower=1e-6)

    monitoring_metrics, feedback = run_monitoring_agent(recommendation_df)
    return recommendation_df, monitoring_metrics, feedback


def run_monitoring_agent(recommendation_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(DEFAULT_SEED)
    feedback = recommendation_df.copy()
    demand_noise = rng.normal(0.0, 0.04, len(feedback))
    utilization_noise = rng.normal(0.0, 0.03, len(feedback))

    feedback["actual_demand_kwh"] = (feedback["expected_demand_kwh"] * (1.0 + demand_noise)).clip(lower=0.0)
    feedback["actual_utilization_rate"] = (feedback["expected_utilization_rate"] * (1.0 + utilization_noise)).clip(lower=0.0, upper=1.5)
    feedback["actual_sessions"] = np.maximum(1.0, feedback["expected_sessions"] * (1.0 + demand_noise))
    feedback["actual_wait_time_minutes"] = np.maximum(0.0, feedback["expected_wait_time_minutes"] * (1.0 + utilization_noise))
    feedback["actual_revenue"] = feedback["recommended_tariff_per_kwh"] * feedback["actual_demand_kwh"]
    feedback["customer_response_rate"] = ((feedback["actual_sessions"] - feedback["sessions"]) / feedback["sessions"].clip(lower=1.0)) * 100.0
    feedback["pricing_efficiency_score"] = feedback["actual_revenue"] / feedback["actual_demand_kwh"].clip(lower=1e-6)
    feedback["learned_elasticity"] = feedback["elasticity"] + 0.20 * ((feedback["actual_demand_kwh"] - feedback["expected_demand_kwh"]) / feedback["predicted_demand_kwh"].clip(lower=1.0))

    off_peak_mask = feedback["pricing_segment"] == "off_peak"
    off_peak_uplift = 0.0
    if off_peak_mask.any():
        off_peak_baseline_sessions = max(float(feedback.loc[off_peak_mask, "sessions"].sum()), 1.0)
        off_peak_uplift = float(
            ((feedback.loc[off_peak_mask, "actual_sessions"].sum() - feedback.loc[off_peak_mask, "sessions"].sum()) / off_peak_baseline_sessions) * 100.0
        )

    baseline_revenue_total = max(float(feedback["baseline_revenue"].sum()), 1e-6)

    metrics = pd.DataFrame(
        [
            {
                "agent": "Tariff Pricing Agent",
                "metric": "Revenue Gain %",
                "value": float(((feedback["actual_revenue"].sum() - feedback["baseline_revenue"].sum()) / baseline_revenue_total) * 100.0),
            },
            {
                "agent": "Tariff Pricing Agent",
                "metric": "Charger Utilization Rate",
                "value": float(feedback["actual_utilization_rate"].mean()),
            },
            {
                "agent": "Tariff Pricing Agent",
                "metric": "Off-Peak Uplift",
                "value": off_peak_uplift,
            },
            {
                "agent": "Monitoring & Learning Agent",
                "metric": "Average Waiting Time Reduction",
                "value": float((feedback["wait_time_minutes_proxy"].mean() - feedback["actual_wait_time_minutes"].mean())),
            },
            {
                "agent": "Monitoring & Learning Agent",
                "metric": "Customer Response Rate",
                "value": float(feedback["customer_response_rate"].mean()),
            },
            {
                "agent": "Monitoring & Learning Agent",
                "metric": "Pricing Efficiency Score",
                "value": float(feedback["pricing_efficiency_score"].mean()),
            },
        ]
    )
    return metrics, feedback


def _segment_elasticity(pricing_segment: str, utilization_rate: float) -> float:
    if pricing_segment == "off_peak":
        return 0.22
    if pricing_segment == "peak":
        return 0.06 if utilization_rate >= 0.80 else 0.09
    return 0.12
