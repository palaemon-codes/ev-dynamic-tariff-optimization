# Agentic AI-Based Dynamic Tariff Optimization for EV Charging Networks

**SOC/BIZ Open Project 2026**  
Praneshwar Kannan Kommiya — B.Tech (Mechanical Engineering), 3rd Year, IIT Roorkee

---

This project builds an end-to-end pipeline that uses real EV charging session data to model demand, recommend dynamic per-kWh tariffs, and evaluate the resulting revenue and utilisation outcomes through a simulated monitoring loop.

The core question: can demand-responsive pricing unlock meaningful revenue and reduce network congestion — without adding any physical infrastructure?

## Datasets used

- **ACN-Data** — session-level charging logs from Caltech and JPL campuses, Pasadena CA (Apr–Nov 2018), exported via the ACN-Data API.
- **ST-EVCDP** — 5-minute interval occupancy, volume, duration, and price records across 247 grid zones in Shenzhen, China (Jun–Jul 2022). Source: [IntelligentSystemsLab/ST-EVCDP](https://github.com/IntelligentSystemsLab/ST-EVCDP).

Combined: **184,113 station-hour observations across 351 charging stations**.

## What the pipeline does

1. Loads and cleans both datasets; aggregates 5-minute records to hourly station-level observations.
2. Engineers 23 features — demand lags (1h, 24h, 168h), rolling statistics, utilisation proxies, time-of-day flags.
3. Trains a ridge regression demand model and a logistic congestion classifier on an 80/20 chronological split.
4. Runs a tariff agent that tests 8 candidate price points per record and picks the highest-scoring tariff based on expected revenue, congestion risk, and segment elasticity.
5. Simulates post-pricing feedback to compute revenue gain, session response rate, and wait-time reduction.

## Key results

| Metric | Value |
|---|---|
| Demand model R² (holdout) | 0.971 |
| Revenue uplift vs. Rs.15/kWh baseline | +28.5% |
| Customer session response | +11.1% |
| Avg. wait time reduction | 0.23 min |

## Project layout

```
src/ev_tariff_opt/   core pipeline modules
notebooks/           walkthrough notebook
data/raw/            place ACN JSON + ST-EVCDP CSV files here (not committed)
data/processed/      generated unified analytical base (not committed)
outputs/             model outputs and figures (not committed)
docs/                presentation deck
```



## Outputs generated

- `data/processed/unified_ev_analytics.csv`
- `outputs/demand_forecast.csv`
- `outputs/dynamic_tariff_recommendations.csv`
- `outputs/monitoring_feedback.csv`
- `outputs/metrics_summary.csv`
- `outputs/run_summary.json`
- `outputs/figures/*.png`

## Core assumptions

- Baseline tariff is fixed at Rs.15/kWh.
- If official data is missing, the pipeline uses a synthetic dataset with realistic daily peaks and station heterogeneity.
- Wait time is modeled as a congestion proxy derived from utilization because direct queue labels are not provided.
- Customer price response is approximated through segment-level elasticity rules and refined by the monitoring agent.


