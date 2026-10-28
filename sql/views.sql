
USE monte_carlo_analytics;

-- ============================================================
-- 1. Backtest Detail View
-- ============================================================
-- Provide window-level backtest results and exception labels
-- for Tableau model calibration visualizations.

CREATE OR REPLACE VIEW vw_backtest_detail AS
SELECT
    train_start,
    train_end,
    test_start,
    test_end,
    horizon_days,
    realized_cumlog,
    p05,
    p95,
    inside_band,
    band_width,
    pit,
    CASE
        WHEN realized_cumlog < p05 THEN 'Below Lower Bound'
        WHEN realized_cumlog > p95 THEN 'Above Upper Bound'
        ELSE 'Inside Interval'
    END AS prediction_outcome
FROM backtest_results;


-- ============================================================
-- 2. Annual Backtest Performance View
-- ============================================================
-- Summarize empirical coverage and interval width by year.

CREATE OR REPLACE VIEW vw_backtest_annual AS
SELECT
    YEAR(test_end) AS backtest_year,
    COUNT(*) AS total_backtests,
    SUM(inside_band) AS covered_backtests,
    AVG(inside_band) * 100 AS coverage_pct,
    AVG(band_width) AS avg_interval_width,
    AVG(pit) AS avg_pit
FROM backtest_results
GROUP BY YEAR(test_end);


-- ============================================================
-- 3. Risk Metrics View
-- ============================================================
-- Combine simulation metadata and downside risk measures.

CREATE OR REPLACE VIEW vw_risk_metrics AS
SELECT
    r.run_id,
    r.ticker,
    r.model,
    r.forecast_horizon,
    r.n_sims,
    r.seed,
    r.train_start,
    r.train_end,
    m.initial_price,
    m.mean_end_price,
    m.median_end_price,
    m.p05_end_price,
    m.p95_end_price,
    m.prob_loss * 100 AS prob_loss_pct,
    m.var_simple * 100 AS var_simple_pct,
    m.es_simple * 100 AS es_simple_pct,
    m.alpha
FROM simulation_runs AS r
JOIN simulation_metrics AS m
    ON r.run_id = m.run_id;


-- ============================================================
-- 4. Forecast Summary View
-- ============================================================
-- Provide daily Monte Carlo summary statistics and price
-- interval widths for interactive Tableau forecasts.

CREATE OR REPLACE VIEW vw_forecast_summary AS
SELECT
    r.run_id,
    r.ticker,
    r.model,
    s.day,
    s.day / 252.0 AS forecast_year,
    s.mean_price,
    s.median_price,
    s.p05_price,
    s.p95_price,
    s.p95_price - s.p05_price AS interval_width
FROM simulation_runs AS r
JOIN simulation_price_summary AS s
    ON r.run_id = s.run_id;


CREATE OR REPLACE VIEW vw_scenario_comparison AS
SELECT
    run_id,
    scenario,
    volatility_multiplier,
    ROUND(daily_volatility * 100, 2) AS daily_volatility_pct,
    horizon,
    n_sims,
    ROUND(probability_of_loss * 100, 2) AS loss_probability_pct,
    ROUND(var_simple * 100, 2) AS var_pct,
    ROUND(es_simple * 100, 2) AS es_pct,
    ROUND(mean_end_price, 2) AS mean_terminal_price,
    ROUND(median_end_price, 2) AS median_terminal_price,
    ROUND(p05_end_price, 2) AS p05_terminal_price,
    ROUND(p95_end_price, 2) AS p95_terminal_price
FROM scenario_metrics;
