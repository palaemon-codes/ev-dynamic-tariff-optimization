import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BASELINE_TARIFF, DEFAULT_SEED, FIGURES_DIR, OUTPUT_DIR, PROCESSED_DIR, RAW_ACN_DIR, RAW_URBANEV_DIR


def ensure_directories() -> None:
    for directory in (RAW_ACN_DIR, RAW_URBANEV_DIR, PROCESSED_DIR, OUTPUT_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def build_analytics_base(use_synthetic_if_missing: bool = True) -> tuple[pd.DataFrame, dict]:
    ensure_directories()

    acn_hourly = load_acn_hourly()
    urbanev_hourly = load_urbanev_hourly()

    frames = [frame for frame in (acn_hourly, urbanev_hourly) if not frame.empty]
    source_summary = {
        "acn_rows": int(len(acn_hourly)),
        "urbanev_rows": int(len(urbanev_hourly)),
        "used_synthetic_data": False,
    }

    if not frames:
        if not use_synthetic_if_missing:
            raise FileNotFoundError("No raw datasets were found in data/raw/acn or data/raw/urbanev.")
        hourly = generate_synthetic_hourly_data()
        source_summary["used_synthetic_data"] = True
    else:
        hourly = pd.concat(frames, ignore_index=True, sort=False)

    hourly = finalize_analytics_base(hourly)
    return hourly, source_summary


def load_acn_hourly() -> pd.DataFrame:
    json_files = sorted(RAW_ACN_DIR.rglob("*.json"))
    if not json_files:
        return pd.DataFrame()

    records: list[dict] = []
    for file_path in json_files:
        payload = _load_json_payload_tolerant(file_path)
        if payload is None:
            continue
        records.extend(_extract_acn_session_records(payload))

    if not records:
        return pd.DataFrame()

    sessions = pd.json_normalize(records)
    rename_map = {
        "sessionID": "session_id",
        "sessionId": "session_id",
        "stationID": "station_id",
        "stationId": "station_id",
        "spaceID": "space_id",
        "spaceId": "space_id",
        "siteID": "site_id",
        "siteId": "site_id",
        "site": "site_id",
        "userID": "user_id",
        "userId": "user_id",
        "kWhDelivered": "energy_kwh",
        "kwhDelivered": "energy_kwh",
        "connectionTime": "connection_time",
        "disconnectTime": "disconnect_time",
        "doneChargingTime": "done_charging_time",
    }
    sessions = sessions.rename(columns=rename_map)

    for timestamp_col in ("connection_time", "disconnect_time", "done_charging_time"):
        if timestamp_col in sessions.columns:
            sessions[timestamp_col] = pd.to_datetime(sessions[timestamp_col], errors="coerce", utc=True).dt.tz_localize(None)

    if "connection_time" not in sessions.columns:
        return pd.DataFrame()

    sessions = sessions.dropna(subset=["connection_time"]).copy()
    sessions["station_id"] = sessions.get("station_id", "acn_station_unknown").astype(str)
    sessions["site_id"] = sessions.get("site_id", "acn_site_unknown").astype(str)
    sessions["session_id"] = sessions.get("session_id", pd.RangeIndex(len(sessions))).astype(str)
    sessions["energy_kwh"] = pd.to_numeric(sessions.get("energy_kwh", 0.0), errors="coerce").fillna(0.0)

    disconnect = sessions.get("disconnect_time")
    done_charging = sessions.get("done_charging_time")
    if disconnect is None:
        disconnect = sessions["connection_time"] + pd.to_timedelta(np.maximum(sessions["energy_kwh"] / 6.6, 0.5), unit="h")
    disconnect = disconnect.fillna(sessions["connection_time"] + pd.to_timedelta(np.maximum(sessions["energy_kwh"] / 6.6, 0.5), unit="h"))

    if done_charging is None:
        done_charging = sessions["connection_time"] + pd.to_timedelta(np.maximum(sessions["energy_kwh"] / 7.2, 0.25), unit="h")
    done_charging = done_charging.fillna(sessions["connection_time"] + pd.to_timedelta(np.maximum(sessions["energy_kwh"] / 7.2, 0.25), unit="h"))

    sessions["session_duration_hours"] = (disconnect - sessions["connection_time"]).dt.total_seconds().div(3600).clip(lower=0.05)
    sessions["charging_duration_hours"] = (done_charging - sessions["connection_time"]).dt.total_seconds().div(3600).clip(lower=0.05)
    sessions["timestamp"] = sessions["connection_time"].dt.floor("h")
    sessions["baseline_tariff_per_kwh"] = BASELINE_TARIFF

    hourly = (
        sessions.groupby(["timestamp", "station_id", "site_id"], dropna=False)
        .agg(
            sessions=("session_id", "nunique"),
            demand_kwh=("energy_kwh", "sum"),
            duration_hours=("charging_duration_hours", "sum"),
            avg_session_duration_hours=("session_duration_hours", "mean"),
            avg_energy_kwh=("energy_kwh", "mean"),
        )
        .reset_index()
    )
    hourly["capacity_units"] = 1.0
    hourly["busy_count"] = hourly["sessions"].clip(upper=1)
    hourly["tariff_per_kwh"] = BASELINE_TARIFF
    hourly["e_price_per_kwh"] = np.nan
    hourly["s_price_per_kwh"] = np.nan
    hourly["latitude"] = np.nan
    hourly["longitude"] = np.nan
    hourly["source"] = "ACN-Data"
    return hourly


def _load_json_payload_tolerant(file_path: Path) -> dict | list | None:
    raw_text = file_path.read_text()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    stripped = raw_text.rstrip()
    if '"_items"' not in stripped:
        return None

    repaired = stripped.rstrip(",\n\r\t ")
    repair_candidates = [
        repaired + "\n  ]\n}",
        repaired + "\n]\n}",
    ]
    for candidate in repair_candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _extract_acn_session_records(payload: object) -> list[dict]:
    records: list[dict] = []

    def walk(node: object) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return

        lowered_keys = {str(key).lower() for key in node.keys()}
        if {"connectiontime", "stationid"}.issubset(lowered_keys) or "kwhdelivered" in lowered_keys:
            records.append(node)
            return

        for value in node.values():
            if isinstance(value, (list, dict)):
                walk(value)

    walk(payload)
    return records


def load_urbanev_hourly() -> pd.DataFrame:
    volume_path = RAW_URBANEV_DIR / "volume.csv"
    duration_path = RAW_URBANEV_DIR / "duration.csv"
    occupancy_path = RAW_URBANEV_DIR / "occupancy.csv"
    e_price_path = RAW_URBANEV_DIR / "e_price.csv"
    s_price_path = RAW_URBANEV_DIR / "s_price.csv"
    legacy_price_path = RAW_URBANEV_DIR / "price.csv"
    info_path = RAW_URBANEV_DIR / "inf.csv"
    legacy_info_path = RAW_URBANEV_DIR / "information.csv"
    time_path = RAW_URBANEV_DIR / "time.csv"

    if not e_price_path.exists() and legacy_price_path.exists():
        e_price_path = legacy_price_path
    if not info_path.exists() and legacy_info_path.exists():
        info_path = legacy_info_path

    timestamp_index = _load_time_index(time_path)
    dataset_label = "UrbanEV" if (RAW_URBANEV_DIR / "inf.csv").exists() or (RAW_URBANEV_DIR / "e_price.csv").exists() else "ST-EVCDP"

    available = [path for path in (volume_path, duration_path, occupancy_path, e_price_path, s_price_path) if path.exists()]
    if not available:
        return pd.DataFrame()

    merged = None
    for path, value_name in (
        (volume_path, "demand_kwh"),
        (duration_path, "duration_hours"),
        (occupancy_path, "busy_count"),
        (e_price_path, "e_price_per_kwh"),
        (s_price_path, "s_price_per_kwh"),
    ):
        if not path.exists():
            continue
        current = _load_wide_timeseries(path, value_name=value_name, timestamp_index=timestamp_index)
        if current.empty:
            continue
        if merged is None:
            merged = current
        else:
            merged = merged.merge(current, on=["timestamp", "station_id"], how="outer")

    if merged is None or merged.empty:
        return pd.DataFrame()

    if info_path.exists():
        metadata = _load_station_metadata(info_path)
        merged = merged.merge(metadata, on="station_id", how="left")

    merged["demand_kwh"] = pd.to_numeric(merged.get("demand_kwh", 0.0), errors="coerce").fillna(0.0)
    merged["duration_hours"] = pd.to_numeric(merged.get("duration_hours", 0.0), errors="coerce").fillna(0.0)
    merged["busy_count"] = pd.to_numeric(merged.get("busy_count", 0.0), errors="coerce").fillna(0.0)
    merged["e_price_per_kwh"] = pd.to_numeric(merged.get("e_price_per_kwh", np.nan), errors="coerce")
    merged["s_price_per_kwh"] = pd.to_numeric(merged.get("s_price_per_kwh", np.nan), errors="coerce")

    merged["capacity_units"] = pd.to_numeric(merged.get("capacity_units", 8.0), errors="coerce").fillna(8.0).clip(lower=1.0)
    merged["avg_session_duration_hours"] = np.where(
        merged["busy_count"] > 0,
        merged["duration_hours"] / merged["busy_count"].clip(lower=1.0),
        merged["duration_hours"],
    )
    merged["sessions"] = np.maximum(1.0, np.round(merged["busy_count"].clip(lower=0.0))).astype(int)
    merged["site_id"] = merged.get("district", merged.get("area_type", "urban_zone_unknown")).fillna("urban_zone_unknown")
    if legacy_price_path.exists() and not s_price_path.exists():
        merged["s_price_per_kwh"] = merged["s_price_per_kwh"].fillna(0.0)
        merged["tariff_per_kwh"] = merged["e_price_per_kwh"].fillna(BASELINE_TARIFF)
    else:
        merged["tariff_per_kwh"] = (
            merged["e_price_per_kwh"].fillna(BASELINE_TARIFF * 0.55)
            + merged["s_price_per_kwh"].fillna(BASELINE_TARIFF * 0.45)
        )
    merged["avg_energy_kwh"] = np.where(
        merged["sessions"] > 0,
        merged["demand_kwh"] / merged["sessions"].clip(lower=1),
        merged["demand_kwh"],
    )
    merged = _aggregate_to_hourly(merged)
    merged["source"] = dataset_label
    return merged


def _load_wide_timeseries(path: Path, value_name: str, timestamp_index: pd.Series | None = None) -> pd.DataFrame:
    wide = pd.read_csv(path)
    if wide.empty or wide.shape[1] < 2:
        return pd.DataFrame()

    wide.columns = [str(column).strip() for column in wide.columns]
    first_col = wide.columns[0]

    if timestamp_index is not None and len(timestamp_index) == len(wide):
        leading_values = pd.to_numeric(wide[first_col], errors="coerce")
        if first_col.lower() in {"timestamp", "index", "unnamed: 0"} or leading_values.notna().mean() >= 0.95:
            wide = wide.drop(columns=[first_col])
        wide = pd.concat(
            [pd.DataFrame({"timestamp": pd.Series(timestamp_index).reset_index(drop=True)}), wide.reset_index(drop=True)],
            axis=1,
        )
    else:
        parsed = pd.to_datetime(wide[first_col], errors="coerce")
        if parsed.notna().mean() >= 0.8:
            wide = wide.rename(columns={first_col: "timestamp"})
            wide["timestamp"] = parsed.dt.tz_localize(None) if getattr(parsed.dt, "tz", None) is not None else parsed
        else:
            wide.insert(0, "timestamp", pd.date_range("2022-09-01", periods=len(wide), freq="h"))

    id_vars = ["timestamp"]
    value_vars = [column for column in wide.columns if column not in id_vars]
    long = wide.melt(id_vars=id_vars, value_vars=value_vars, var_name="station_id", value_name=value_name)
    long["station_id"] = long["station_id"].astype(str)
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    long = long.dropna(subset=[value_name])
    return long


def _load_time_index(path: Path) -> pd.Series | None:
    if not path.exists():
        return None

    time_df = pd.read_csv(path)
    if time_df.empty:
        return None

    time_df.columns = [_normalize_column_name(column) for column in time_df.columns]
    component_cols = {"year", "month", "day", "hour", "minute", "second"}
    if component_cols.issubset(set(time_df.columns)):
        timestamps = pd.to_datetime(
            {
                "year": time_df["year"],
                "month": time_df["month"],
                "day": time_df["day"],
                "hour": time_df["hour"],
                "minute": time_df["minute"],
                "second": time_df["second"],
            },
            errors="coerce",
        )
        return pd.Series(timestamps)

    parsed = pd.to_datetime(time_df.iloc[:, 0], errors="coerce")
    if parsed.notna().any():
        return pd.Series(parsed)
    return None


def _aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    aggregated = df.copy()
    aggregated["timestamp"] = pd.to_datetime(aggregated["timestamp"], errors="coerce")
    aggregated = aggregated.dropna(subset=["timestamp"])
    aggregated["timestamp"] = aggregated["timestamp"].dt.floor("h")

    group_cols = ["timestamp", "station_id"]
    agg_spec = {
        "site_id": ("site_id", "first"),
        "demand_kwh": ("demand_kwh", "sum"),
        "duration_hours": ("duration_hours", "sum"),
        "busy_count": ("busy_count", "mean"),
        "tariff_per_kwh": ("tariff_per_kwh", "mean"),
        "e_price_per_kwh": ("e_price_per_kwh", "mean"),
        "s_price_per_kwh": ("s_price_per_kwh", "mean"),
        "capacity_units": ("capacity_units", "first"),
        "latitude": ("latitude", "first"),
        "longitude": ("longitude", "first"),
        "district": ("district", "first"),
        "area_type": ("area_type", "first"),
    }
    agg_spec = {target: spec for target, spec in agg_spec.items() if spec[0] in aggregated.columns}
    grouped = aggregated.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()
    grouped["sessions"] = np.maximum(1.0, np.round(grouped["busy_count"].clip(lower=0.0))).astype(int)
    grouped["avg_session_duration_hours"] = np.where(
        grouped["sessions"] > 0,
        grouped["duration_hours"] / grouped["sessions"].clip(lower=1),
        grouped["duration_hours"],
    )
    grouped["avg_energy_kwh"] = np.where(
        grouped["sessions"] > 0,
        grouped["demand_kwh"] / grouped["sessions"].clip(lower=1),
        grouped["demand_kwh"],
    )
    return grouped


def _load_station_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path)
    metadata.columns = [_normalize_column_name(column) for column in metadata.columns]

    station_col = next((column for column in ("station_id", "grid", "num", "csid", "id", "station", "name") if column in metadata.columns), metadata.columns[0])
    metadata = metadata.rename(columns={station_col: "station_id"})

    selected = pd.DataFrame({"station_id": metadata["station_id"].astype(str)})
    column_candidates = {
        "latitude": ["latitude", "lat", "la", "y"],
        "longitude": ["longitude", "lng", "lon", "x"],
        "capacity_units": ["count", "charging_piles", "pile_count", "num_piles", "capacity_units", "piles", "cp_num"],
        "district": ["district", "region", "zone", "city_zone", "grid"],
        "area_type": ["cbd", "dynamic_pricing", "area_type", "is_cbd", "business_district", "area"],
    }
    for target, candidates in column_candidates.items():
        source = next((candidate for candidate in candidates if candidate in metadata.columns), None)
        if source is not None:
            selected[target] = metadata[source]

    return selected.drop_duplicates(subset=["station_id"])


def _normalize_column_name(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def generate_synthetic_hourly_data(days: int = 90, stations: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(DEFAULT_SEED)
    timestamps = pd.date_range("2025-01-01", periods=24 * days, freq="h")
    records: list[dict] = []

    for station_idx in range(stations):
        station_id = f"STN_{station_idx + 1:03d}"
        site_id = f"SITE_{station_idx % 4 + 1}"
        capacity_units = int(rng.integers(4, 11))
        station_factor = 1.0 + (station_idx % 6) * 0.08
        base_demand = float(rng.uniform(18.0, 40.0))
        latitude = 28.50 + rng.normal(0, 0.08)
        longitude = 77.10 + rng.normal(0, 0.08)

        for ts in timestamps:
            hour = ts.hour
            day_of_week = ts.dayofweek
            morning_peak = np.exp(-((hour - 9.0) / 2.6) ** 2)
            evening_peak = np.exp(-((hour - 18.0) / 2.8) ** 2)
            shoulder = np.exp(-((hour - 13.0) / 4.0) ** 2)
            weekend_factor = 0.84 if day_of_week >= 5 else 1.0
            noise = rng.normal(0, 2.8)

            demand_kwh = max(0.0, base_demand * station_factor * weekend_factor * (0.55 + 0.65 * morning_peak + 0.80 * evening_peak + 0.25 * shoulder) + noise)
            duration_hours = min(capacity_units * 1.05, max(0.1, demand_kwh / 7.2))
            utilization_rate = duration_hours / capacity_units
            sessions = max(1, int(round(duration_hours * rng.uniform(0.9, 1.3))))
            busy_count = min(capacity_units, max(1, int(round(utilization_rate * capacity_units + rng.normal(0, 0.5)))))
            avg_session_duration_hours = duration_hours / max(sessions, 1)
            avg_energy_kwh = demand_kwh / max(sessions, 1)

            records.append(
                {
                    "timestamp": ts,
                    "station_id": station_id,
                    "site_id": site_id,
                    "sessions": sessions,
                    "demand_kwh": demand_kwh,
                    "duration_hours": duration_hours,
                    "avg_session_duration_hours": avg_session_duration_hours,
                    "avg_energy_kwh": avg_energy_kwh,
                    "capacity_units": capacity_units,
                    "busy_count": busy_count,
                    "tariff_per_kwh": BASELINE_TARIFF,
                    "e_price_per_kwh": BASELINE_TARIFF * 0.55,
                    "s_price_per_kwh": BASELINE_TARIFF * 0.45,
                    "latitude": latitude,
                    "longitude": longitude,
                    "source": "Synthetic-Demo",
                }
            )

    return pd.DataFrame.from_records(records)


def finalize_analytics_base(hourly: pd.DataFrame) -> pd.DataFrame:
    df = hourly.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["timestamp", "station_id"]).reset_index(drop=True)

    defaults = {
        "site_id": "unknown_site",
        "source": "unknown_source",
        "sessions": 1,
        "demand_kwh": 0.0,
        "duration_hours": 0.0,
        "avg_session_duration_hours": 0.0,
        "avg_energy_kwh": 0.0,
        "capacity_units": 1.0,
        "busy_count": 0.0,
        "tariff_per_kwh": BASELINE_TARIFF,
        "e_price_per_kwh": np.nan,
        "s_price_per_kwh": np.nan,
        "latitude": np.nan,
        "longitude": np.nan,
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    numeric_cols = [
        "sessions",
        "demand_kwh",
        "duration_hours",
        "avg_session_duration_hours",
        "avg_energy_kwh",
        "capacity_units",
        "busy_count",
        "tariff_per_kwh",
        "e_price_per_kwh",
        "s_price_per_kwh",
        "latitude",
        "longitude",
    ]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["station_id"] = df["station_id"].astype(str)
    df["site_id"] = df["site_id"].fillna("unknown_site").astype(str)
    df["source"] = df["source"].fillna("unknown_source").astype(str)
    df["sessions"] = df["sessions"].fillna(1).clip(lower=1)
    df["demand_kwh"] = df["demand_kwh"].fillna(0.0).clip(lower=0.0)
    df["duration_hours"] = df["duration_hours"].fillna(0.0).clip(lower=0.0)
    df["capacity_units"] = df["capacity_units"].fillna(1.0).clip(lower=1.0)
    df["busy_count"] = df["busy_count"].fillna(df["sessions"]).clip(lower=0.0)
    df["avg_session_duration_hours"] = df["avg_session_duration_hours"].fillna(df["duration_hours"] / df["sessions"].clip(lower=1))
    df["avg_energy_kwh"] = df["avg_energy_kwh"].fillna(df["demand_kwh"] / df["sessions"].clip(lower=1))
    df["tariff_per_kwh"] = df["tariff_per_kwh"].fillna(BASELINE_TARIFF)
    df["revenue_baseline"] = df["demand_kwh"] * BASELINE_TARIFF
    df["utilization_rate"] = (df["duration_hours"] / df["capacity_units"]).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.5)
    df["occupancy_density"] = (df["busy_count"] / df["capacity_units"]).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.5)
    df["queue_length_proxy"] = np.maximum(0.0, (df["occupancy_density"] - 0.80) * df["capacity_units"])
    df["wait_time_minutes_proxy"] = np.maximum(0.0, (df["utilization_rate"] - 0.75) * 45.0)
    df["pricing_efficiency_baseline"] = df["revenue_baseline"] / df["demand_kwh"].clip(lower=1e-6)

    df = add_time_features(df)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["hour"] = enriched["timestamp"].dt.hour
    enriched["day_of_week"] = enriched["timestamp"].dt.dayofweek
    enriched["month"] = enriched["timestamp"].dt.month
    enriched["is_weekend"] = (enriched["day_of_week"] >= 5).astype(int)
    enriched["is_peak_hour"] = enriched["hour"].isin([8, 9, 10, 17, 18, 19]).astype(int)
    enriched["is_off_peak"] = enriched["hour"].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)
    enriched["pricing_segment"] = np.select(
        [enriched["is_peak_hour"] == 1, enriched["is_off_peak"] == 1],
        ["peak", "off_peak"],
        default="shoulder",
    )

    station_group = enriched.groupby("station_id", sort=False)
    for lag in (1, 24, 168):
        enriched[f"demand_lag_{lag}"] = station_group["demand_kwh"].shift(lag)
        enriched[f"utilization_lag_{lag}"] = station_group["utilization_rate"].shift(lag)

    enriched["demand_roll_mean_24"] = station_group["demand_kwh"].transform(lambda series: series.shift(1).rolling(24, min_periods=1).mean())
    enriched["demand_roll_std_24"] = station_group["demand_kwh"].transform(lambda series: series.shift(1).rolling(24, min_periods=1).std())
    enriched["utilization_roll_mean_24"] = station_group["utilization_rate"].transform(lambda series: series.shift(1).rolling(24, min_periods=1).mean())

    for column in [
        "demand_lag_1",
        "demand_lag_24",
        "demand_lag_168",
        "utilization_lag_1",
        "utilization_lag_24",
        "utilization_lag_168",
        "demand_roll_mean_24",
        "demand_roll_std_24",
        "utilization_roll_mean_24",
    ]:
        enriched[column] = enriched[column].fillna(enriched.groupby("station_id")[column].transform("median"))
        enriched[column] = enriched[column].fillna(enriched[column].median())

    enriched["demand_roll_std_24"] = enriched["demand_roll_std_24"].fillna(0.0)
    return enriched
