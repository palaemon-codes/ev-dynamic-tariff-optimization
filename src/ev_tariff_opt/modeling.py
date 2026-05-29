import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import DEFAULT_SEED


MAX_TRAIN_ROWS = 60000


NUMERIC_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak_hour",
    "is_off_peak",
    "capacity_units",
    "busy_count",
    "avg_session_duration_hours",
    "avg_energy_kwh",
    "tariff_per_kwh",
    "queue_length_proxy",
    "wait_time_minutes_proxy",
    "occupancy_density",
    "demand_lag_1",
    "demand_lag_24",
    "demand_lag_168",
    "utilization_lag_1",
    "utilization_lag_24",
    "utilization_lag_168",
    "demand_roll_mean_24",
    "demand_roll_std_24",
    "utilization_roll_mean_24",
]

CATEGORICAL_FEATURES = ["station_id", "site_id", "source", "pricing_segment"]


def run_demand_prediction_agent(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    modeling_df = df.sort_values("timestamp").reset_index(drop=True).copy()
    modeling_df["congestion_flag"] = (modeling_df["utilization_rate"] >= 0.80).astype(int)

    numeric_features = [column for column in NUMERIC_FEATURES if column in modeling_df.columns]
    categorical_features = [column for column in CATEGORICAL_FEATURES if column in modeling_df.columns]
    feature_columns = numeric_features + categorical_features

    split_index = max(int(len(modeling_df) * 0.80), 1)
    train_df = modeling_df.iloc[:split_index].copy()
    test_df = modeling_df.iloc[split_index:].copy()
    if test_df.empty:
        test_df = train_df.copy()

    train_fit_df = _sample_training_frame(train_df, max_rows=MAX_TRAIN_ROWS)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    demand_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                Ridge(alpha=1.0),
            ),
        ]
    )
    utilization_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                Ridge(alpha=1.0),
            ),
        ]
    )
    congestion_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1200,
                    solver="liblinear",
                    random_state=DEFAULT_SEED + 2,
                ),
            ),
        ]
    )

    demand_model.fit(train_fit_df[feature_columns], train_fit_df["demand_kwh"])
    utilization_model.fit(train_fit_df[feature_columns], train_fit_df["utilization_rate"])
    congestion_model.fit(train_fit_df[feature_columns], train_fit_df["congestion_flag"])

    demand_test_pred = demand_model.predict(test_df[feature_columns])
    util_test_pred = utilization_model.predict(test_df[feature_columns])
    congestion_test_prob = _predict_positive_probability(congestion_model, test_df[feature_columns])

    metrics = pd.DataFrame(
        [
            {
                "agent": "Demand Prediction Agent",
                "metric": "RMSE",
                "value": float(np.sqrt(mean_squared_error(test_df["demand_kwh"], demand_test_pred))),
            },
            {
                "agent": "Demand Prediction Agent",
                "metric": "MAE",
                "value": float(mean_absolute_error(test_df["demand_kwh"], demand_test_pred)),
            },
            {
                "agent": "Demand Prediction Agent",
                "metric": "R2 Score",
                "value": float(r2_score(test_df["demand_kwh"], demand_test_pred)),
            },
            {
                "agent": "Demand Prediction Agent",
                "metric": "Utilization RMSE",
                "value": float(np.sqrt(mean_squared_error(test_df["utilization_rate"], util_test_pred))),
            },
            {
                "agent": "Demand Prediction Agent",
                "metric": "Mean Congestion Probability",
                "value": float(np.mean(congestion_test_prob)),
            },
            {
                "agent": "Demand Prediction Agent",
                "metric": "Training Rows Used",
                "value": float(len(train_fit_df)),
            },
        ]
    )

    all_predictions = modeling_df[["timestamp", "station_id", "site_id", "source", "pricing_segment", "demand_kwh", "utilization_rate", "sessions"]].copy()
    all_predictions["predicted_demand_kwh"] = demand_model.predict(modeling_df[feature_columns])
    all_predictions["predicted_utilization_rate"] = utilization_model.predict(modeling_df[feature_columns]).clip(0.0, 1.5)
    all_predictions["predicted_congestion_probability"] = _predict_positive_probability(congestion_model, modeling_df[feature_columns]).clip(0.0, 1.0)
    all_predictions["predicted_sessions"] = np.maximum(1.0, np.round(modeling_df["sessions"] * (all_predictions["predicted_demand_kwh"] / modeling_df["demand_kwh"].clip(lower=1.0))))
    all_predictions["dataset_split"] = np.where(all_predictions.index < split_index, "train", "test")

    return metrics, all_predictions, modeling_df


def _predict_positive_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    estimator = model.named_steps["model"]
    if len(getattr(estimator, "classes_", [])) < 2:
        return np.zeros(len(features), dtype=float)
    return model.predict_proba(features)[:, 1]


def _sample_training_frame(train_df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(train_df) <= max_rows:
        return train_df
    return train_df.sample(n=max_rows, random_state=DEFAULT_SEED).sort_index()
