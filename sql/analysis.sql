
-- ============================================================
-- Monte Carlo Stock Simulator - SQL Analytics
-- Database: monte_carlo_analytics
-- Purpose: Evaluate rolling backtest performance and calibration
-- ============================================================


-- ============================================================
-- 1. Overall Backtest Performance
-- ============================================================
-- Calculate the total number of rolling backtests, empirical
-- coverage rate of the 90% prediction interval, and average
-- prediction interval width.
-- Also identify the earliest and latest backtest end dates.

SELECT
    COUNT(*) AS total_backtests,
    SUM(inside_band) AS successful_predictions,
    ROUND(AVG(inside_band) * 100, 2) AS coverage_pct,
    ROUND(AVG(band_width), 4) AS avg_interval_width,
    MIN(test_end) AS earliest_test_end,
    MAX(test_end) AS latest_test_end
FROM monte_carlo_analytics.backtest_results;


-- ============================================================
-- 2. Prediction Interval Exception Analysis
-- ============================================================
-- Identify backtest windows where realized cumulative log
-- returns fall outside the simulated 90% prediction interval.
-- Classify exceptions as below the lower bound or above
-- the upper bound.

SELECT
    br.test_start,
    br.test_end,
    br.realized_cumlog,
    br.p05,
    br.p95,
    br.band_width,
    br.pit,
    CASE
        WHEN br.realized_cumlog < br.p05 THEN 'Below Lower Bound'
        WHEN br.realized_cumlog > br.p95 THEN 'Above Upper Bound'
        ELSE 'Inside Interval'
    END AS prediction_outcome
FROM monte_carlo_analytics.backtest_results AS br
WHERE br.inside_band = 0
ORDER BY br.test_end;


-- ============================================================
-- 3. Annual Backtest Performance
-- ============================================================
-- Aggregate rolling backtest results by evaluation end year.
-- Compare empirical coverage, average prediction interval
-- width, and average Probability Integral Transform (PIT).
-- Note: Backtest windows overlap and are not independent.

SELECT
    YEAR(test_end) AS backtest_year,
    COUNT(*) AS total_backtests,
    SUM(inside_band) AS covered_backtests,
    ROUND(AVG(inside_band) * 100, 2) AS coverage_pct,
    ROUND(AVG(band_width), 4) AS avg_interval_width,
    ROUND(AVG(pit), 4) AS avg_pit
FROM monte_carlo_analytics.backtest_results
GROUP BY YEAR(test_end)
ORDER BY backtest_year;


-- ============================================================
-- 4. Monte Carlo Risk Metrics Summary
-- ============================================================
-- Combine simulation metadata with terminal price statistics
-- and downside risk measures.
-- Report Value-at-Risk and Expected Shortfall on the
-- simple-return scale.

SELECT
    r.run_id,
    r.ticker,
    r.model,
    r.forecast_horizon,
    r.n_sims,
    m.initial_price,
    m.mean_end_price,
    m.median_end_price,
    m.p05_end_price,
    m.p95_end_price,
    ROUND(m.prob_loss * 100, 2) AS prob_loss_pct,
    ROUND(m.var_simple * 100, 2) AS var_5pct,
    ROUND(m.es_simple * 100, 2) AS es_5pct
FROM monte_carlo_analytics.simulation_runs AS r
JOIN monte_carlo_analytics.simulation_metrics AS m
    ON r.run_id = m.run_id;


-- ============================================================
-- 5. Terminal Price Consistency Check
-- ============================================================
-- Extract the final simulation day using the forecast horizon
-- stored in simulation_runs.
-- Compare the endpoint statistics against simulation_metrics.

WITH terminal_prices AS (
    SELECT
        r.run_id,
        s.day,
        s.mean_price,
        s.median_price,
        s.p05_price,
        s.p95_price
    FROM monte_carlo_analytics.simulation_runs AS r
    JOIN monte_carlo_analytics.simulation_price_summary AS s
        ON r.run_id = s.run_id
    WHERE s.day = r.forecast_horizon
)
SELECT
    t.run_id,
    t.day,
    ROUND(t.mean_price - m.mean_end_price, 8)
        AS mean_difference,
    ROUND(t.median_price - m.median_end_price, 8)
        AS median_difference,
    ROUND(t.p05_price - m.p05_end_price, 8)
        AS p05_difference,
    ROUND(t.p95_price - m.p95_end_price, 8)
        AS p95_difference
FROM terminal_prices AS t
JOIN monte_carlo_analytics.simulation_metrics AS m
    ON t.run_id = m.run_id;



-- ============================================================
-- 6. Forecast Horizon and Uncertainty Analysis
-- ============================================================
-- Summarize forecast uncertainty at annual milestones.
-- Compare predicted mean prices and 90% prediction interval
-- widths across the five-year simulation horizon.

WITH forecast_summary AS (
    SELECT
        r.run_id,
        r.ticker,
        s.day,
        s.mean_price,
        s.median_price,
        s.p05_price,
        s.p95_price,
        s.p95_price - s.p05_price AS interval_width
    FROM monte_carlo_analytics.simulation_runs AS r
    JOIN monte_carlo_analytics.simulation_price_summary AS s
        ON r.run_id = s.run_id
)
SELECT
    ticker,
    day,
    ROUND(day / 252.0, 1) AS forecast_year,
    ROUND(mean_price, 2) AS mean_price,
    ROUND(median_price, 2) AS median_price,
    ROUND(p05_price, 2) AS p05_price,
    ROUND(p95_price, 2) AS p95_price,
    ROUND(interval_width, 2) AS interval_width
FROM forecast_summary
WHERE day % 252 = 0
ORDER BY ticker, day;

-- Compare risk metrics across volatility assumptions
SELECT
    scenario,
    volatility_multiplier,
    loss_probability_pct,
    var_pct,
    es_pct,
    mean_terminal_price,
    median_terminal_price
FROM vw_scenario_comparison
WHERE run_id = 'AMZN_GBM_2025_422'
ORDER BY volatility_multiplier;
