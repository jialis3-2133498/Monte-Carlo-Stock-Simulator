# Monte Carlo Equity Risk Analytics | AMZN

**Python · MySQL · SQL · Tableau · Monte Carlo Simulation · Risk Analytics**

An end-to-end equity risk analytics project that combines a Geometric Brownian Motion (GBM) Monte Carlo engine, rolling out-of-sample model evaluation, SQL-based analytics, and interactive Tableau dashboards. Using Amazon (AMZN) as a case study, the project quantifies five-year downside exposure and compares risk under three controlled volatility assumptions.

> **Modeling scope:** These are scenario-conditional simulation outcomes, not investment recommendations or calibrated predictions of future market prices.

## Highlights

- **Monte Carlo forecasting:** 1,000 reproducible GBM paths across 1,260 trading days (~5 years), using maximum-likelihood estimates of historical daily log-return mean and volatility.
- **Downside risk:** Probability of loss, lower/upper terminal price percentiles, 5% Value at Risk (VaR), and 5% Expected Shortfall (ES).
- **Model validation:** Rolling backtest with a 756-day training window, 63-day forecast horizon, and 21-day step; coverage, interval width, probability integral transform (PIT), and exception drilldown.
- **Data pipeline:** Python-generated price, simulation, metric, and backtest data loaded into six MySQL tables through repeatable, upsert-based ETL; SQL views support dashboard reporting.
- **Volatility sensitivity:** Base (1.0×), High Volatility (1.5×), and Stress (2.0×) scenarios holding estimated daily log-return mean, initial price, forecast horizon, and random seed constant.
- **Interactive analysis:** Tableau dashboard for forecast/model validation and a second dashboard for volatility sensitivity (locally developed; public-view link to be added once published).

## Results

### Baseline five-year simulation

| Metric | Result |
|---|---:|
| Initial adjusted price | $219.39 |
| Mean terminal price | $683.23 |
| Median terminal price | $499.55 |
| 5th / 95th percentile terminal price | $133.76 / $1,790.02 |
| Probability of terminal loss | 15.60% |
| 5% VaR (simple-return lower-tail quantile) | −39.03% |
| 5% ES (mean simple return in the lower 5% tail) | −57.90% |

### Volatility sensitivity

| Metric | Base (1.0×) | High (1.5×) | Stress (2.0×) |
|---|---:|---:|---:|
| Daily volatility | 2.26% | 3.40% | 4.53% |
| Probability of loss | 15.60% | 24.70% | 30.60% |
| 5% VaR (simple return) | −39.03% | −68.72% | −83.95% |
| 5% ES (simple return) | −57.90% | −81.56% | −91.81% |
| Mean terminal price | $683.23 | $980.93 | $1,591.58 |
| Median terminal price | $499.55 | $495.28 | $491.05 |

**Interpretation:** Doubling historical daily volatility raises the simulated five-year probability of loss by **15.0 percentage points** (15.6% → 30.6%). Despite higher modeled mean terminal prices under fixed log-return drift, median outcomes are nearly unchanged and downside tail-risk measures worsen. This is a controlled volatility sensitivity exercise, not a comprehensive market-crisis stress test.

### Rolling backtest

- **43 overlapping windows** through **October 9, 2026**, of which **42** realized returns fell within the nominal 90% prediction interval (**97.67% observed coverage**).
- One upside-band exception occurred in the window ending **August 4, 2023**.
- Overlapping windows are **not independent**, so the observed coverage should not be interpreted as strong statistical proof of superior calibration.

## Architecture

```text
Yahoo Finance / yfinance
         |
         v
Python data preparation and daily log returns
         |
         +--> GBM Monte Carlo + risk metrics
         +--> Rolling backtest + PIT/coverage
         +--> Volatility scenario analysis
         |
         v
CSV outputs + Python MySQL ETL (upserts)
         |
         v
MySQL: historical_prices | simulation_runs | simulation_price_summary
       simulation_metrics | backtest_results | scenario_metrics
         |
         v
SQL analytical views and comparison queries
         |
         v
Tableau: Forecast & Model Validation | Volatility Stress & Risk Sensitivity
```

## Modeling methodology

Daily log returns are calculated as

$$
r_t = \ln(S_t / S_{t-1}), \quad r_t \overset{iid}{\sim} \mathcal{N}(\mu_d,\sigma_d^2).
$$

Daily historical estimates use the sample mean and population-form standard deviation (`ddof=0`), consistent with a normal-distribution MLE. Simulated prices are recovered from cumulative daily log returns:

$$
S_{t} = S_0\exp\bigg(\sum_{i=1}^{t} r_i\bigg).
$$

For scenarios, only `sigma_d` changes by a multiplier; the random seed is held constant to facilitate comparable simulations (common random numbers).

## MySQL and Tableau

Six relational reporting tables provide primary-key and foreign-key relationships and support SQL analytical views, including:

- `vw_forecast_summary` — mean/median and P05–P95 path summaries
- `vw_backtest_annual` — annual interval coverage and diagnostics
- `vw_backtest_detail` — individual backtest windows and exceptions
- `vw_risk_metrics` — baseline return-risk KPIs
- `vw_scenario_comparison` — comparative volatility-scenario metrics

The local Tableau workbook uses **live MySQL connections**. Tableau Public does not support live connections to local MySQL, so any separately published dashboard must use an approved public-data extract/CSV workflow. **Public Dashboard:** _link pending_.

## Run locally

**Prerequisites:** Python 3.10+ (check your installed library compatibility), MySQL 8+, and a MySQL account with privileges on the `monte_carlo_analytics` schema.

1. Create and activate a Python virtual environment; install NumPy, pandas, SciPy, Matplotlib, yfinance, and `mysql-connector-python` (and other dependencies required by your environment).
2. Set `MYSQL_USER` and `MYSQL_PASSWORD` environment variables; optionally configure `MYSQL_HOST` and `MYSQL_PORT`. **Never commit credentials.**
3. Create/select the `monte_carlo_analytics` database and execute `sql/schema.sql` to create tables.
4. Run `python main.py` to regenerate outputs and load data; then execute `sql/views.sql` to create or refresh reporting views and `sql/analysis.sql` to inspect the results.
5. In Tableau, connect to the MySQL reporting views to reproduce the local dashboards.

> **Reproducibility note:** The baseline simulation is deterministic conditional on fixed historical training data, parameter estimates, random seed, and dependency versions. Historical market-data downloads and rolling-backtest counts may change as the data end date advances. The 2026 backtest result refers to the Oct 28, 2026 project run.

## Repository guide

```text
main.py                  # Orchestrates ingestion, simulation, ETL and outputs
src/                     # Data, returns, simulation, scenarios, metrics,
                         # rolling backtest, plotting and MySQL ETL modules
sql/schema.sql           # Database schema
sql/views.sql            # Tableau-facing analytical views
sql/analysis.sql         # Risk and model-evaluation SQL queries
outputs/                 # Generated CSV tables and static charts
Report.md                # Technical methodology, findings and limitations
```

## Key limitations

- Normal i.i.d. daily log returns exclude volatility clustering, fat tails, jumps, regime changes, and structural breaks.
- Scenario multipliers are hypothetical sensitivity settings, **not probabilities of different market regimes**.
- 1,000 paths imply Monte Carlo sampling error in tail estimates; no confidence intervals are reported.
- Overlapping 63-day test periods complicate inference about backtest coverage.
- The model should not be used alone for trading or investment decisions.

## Dashboard previews

_Add vetted screenshots here after exporting from Tableau. Public Tableau link pending._

## Portfolio links

- **Tableau Public:** Add the verified public dashboard URL after publishing.
- **Technical report:** [Report.md](Report.md)
