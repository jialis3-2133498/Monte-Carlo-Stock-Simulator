import os
import mysql.connector
from mysql.connector import Error


def connect_database():
    """
    """
    connection = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        database="monte_carlo_analytics"
    )
    return connection


def load_historical_prices(df):
    """Insert or update historical stock prices in MySQL."""

    sql = """
        INSERT INTO historical_prices
            (ticker, trade_date, adjusted_close)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
            adjusted_close = VALUES(adjusted_close)
    """

    rows = [
        (
            str(row.ticker),
            row.trade_date,
            float(row.adjusted_close)
        )
        for row in df.itertuples(index=False)
    ]

    conn = connect_database()

    try:
        cursor = conn.cursor()
        cursor.executemany(sql, rows)
        conn.commit()
        print(f"Historical prices loaded: {len(rows)} rows")
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def load_simulation_runs(df):
    """Insert or update Monte Carlo simulation metadata."""

    sql = """
        INSERT INTO simulation_runs (
            run_id, ticker, model, train_start, train_end,
            forecast_horizon, n_sims, seed, initial_price
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            ticker = VALUES(ticker),
            model = VALUES(model),
            train_start = VALUES(train_start),
            train_end = VALUES(train_end),
            forecast_horizon = VALUES(forecast_horizon),
            n_sims = VALUES(n_sims),
            seed = VALUES(seed),
            initial_price = VALUES(initial_price)
    """

    rows = [
        (
            str(row.run_id),
            str(row.ticker),
            str(row.model),
            row.train_start,
            row.train_end,
            int(row.forecast_horizon),
            int(row.n_sims),
            int(row.seed),
            float(row.initial_price)
        )
        for row in df.itertuples(index=False)
    ]

    conn = connect_database()
    cursor = None

    try:
        cursor = conn.cursor()
        cursor.executemany(sql, rows)
        conn.commit()
        print(f"Simulation runs loaded: {len(rows)} rows")
    except Exception:
        conn.rollback()
        raise
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


def load_simulation_price_summary(df):
    """Insert or update daily Monte Carlo forecast statistics."""

    sql = """
        INSERT INTO simulation_price_summary (
            run_id, day, mean_price, median_price,
            p05_price, p95_price
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            mean_price = VALUES(mean_price),
            median_price = VALUES(median_price),
            p05_price = VALUES(p05_price),
            p95_price = VALUES(p95_price)
    """

    rows = [
        (
            str(row.run_id),
            int(row.day),
            float(row.mean_price),
            float(row.median_price),
            float(row.p05_price),
            float(row.p95_price)
        )
        for row in df.itertuples(index=False)
    ]

    conn = connect_database()
    cursor = None

    try:
        cursor = conn.cursor()
        cursor.executemany(sql, rows)
        conn.commit()
        print(f"Simulation price summary loaded: {len(rows)} rows")
    except Exception:
        conn.rollback()
        raise
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()



def load_simulation_metrics(df):
    """Insert or update Monte Carlo risk metrics."""

    sql = """
        INSERT INTO simulation_metrics (
            run_id,
            initial_price,
            mean_end_price,
            median_end_price,
            p05_end_price,
            p95_end_price,
            prob_loss,
            log_loss,
            simple_loss,
            alpha,
            var_log,
            es_log,
            var_simple,
            es_simple
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            initial_price = VALUES(initial_price),
            mean_end_price = VALUES(mean_end_price),
            median_end_price = VALUES(median_end_price),
            p05_end_price = VALUES(p05_end_price),
            p95_end_price = VALUES(p95_end_price),
            prob_loss = VALUES(prob_loss),
            log_loss = VALUES(log_loss),
            simple_loss = VALUES(simple_loss),
            alpha = VALUES(alpha),
            var_log = VALUES(var_log),
            es_log = VALUES(es_log),
            var_simple = VALUES(var_simple),
            es_simple = VALUES(es_simple)
    """

    columns = [
        "run_id", "initial_price", "mean_end_price",
        "median_end_price", "p05_end_price", "p95_end_price",
        "prob_loss", "log_loss", "simple_loss", "alpha",
        "var_log", "es_log", "var_simple", "es_simple"
    ]

    rows = [
        tuple(
            str(value) if column == "run_id" else float(value)
            for column, value in zip(columns, row)
        )
        for row in df[columns].itertuples(index=False, name=None)
    ]

    conn = connect_database()
    cursor = None

    try:
        cursor = conn.cursor()
        cursor.executemany(sql, rows)
        conn.commit()
        print(f"Simulation metrics loaded: {len(rows)} rows")
    except Exception:
        conn.rollback()
        raise
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


def load_backtest_results(df):
    """Insert or update rolling backtest results."""

    sql = """
        INSERT INTO backtest_results (
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
            pit
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            train_end = VALUES(train_end),
            test_end = VALUES(test_end),
            realized_cumlog = VALUES(realized_cumlog),
            p05 = VALUES(p05),
            p95 = VALUES(p95),
            inside_band = VALUES(inside_band),
            band_width = VALUES(band_width),
            pit = VALUES(pit)
    """

    rows = [
        (
            row.train_start,
            row.train_end,
            row.test_start,
            row.test_end,
            int(row.horizon_days),
            float(row.realized_cumlog),
            float(row.p05),
            float(row.p95),
            int(row.inside_band),
            float(row.band_width),
            float(row.pit)
        )
        for row in df.itertuples(index=False)
    ]

    conn = connect_database()
    cursor = None

    try:
        cursor = conn.cursor()
        cursor.executemany(sql, rows)
        conn.commit()
        print(f"Backtest results loaded: {len(rows)} rows")
    except Exception:
        conn.rollback()
        raise
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


def load_scenario_metrics(df):
    """Insert or update volatility scenario risk metrics."""

    sql = """
        INSERT INTO scenario_metrics (
            run_id,
            scenario,
            volatility_multiplier,
            daily_volatility,
            horizon,
            n_sims,
            seed,
            alpha,
            mean_end_price,
            median_end_price,
            p05_end_price,
            p95_end_price,
            probability_of_loss,
            var_simple,
            es_simple
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            volatility_multiplier = VALUES(volatility_multiplier),
            daily_volatility = VALUES(daily_volatility),
            horizon = VALUES(horizon),
            n_sims = VALUES(n_sims),
            seed = VALUES(seed),
            alpha = VALUES(alpha),
            mean_end_price = VALUES(mean_end_price),
            median_end_price = VALUES(median_end_price),
            p05_end_price = VALUES(p05_end_price),
            p95_end_price = VALUES(p95_end_price),
            probability_of_loss = VALUES(probability_of_loss),
            var_simple = VALUES(var_simple),
            es_simple = VALUES(es_simple)
    """

    rows = [
        (
            str(row.run_id),
            str(row.scenario),
            float(row.volatility_multiplier),
            float(row.daily_volatility),
            int(row.horizon),
            int(row.n_sims),
            int(row.seed),
            float(row.alpha),
            float(row.mean_end_price),
            float(row.median_end_price),
            float(row.p05_end_price),
            float(row.p95_end_price),
            float(row.probability_of_loss),
            float(row.var_simple),
            float(row.es_simple)
        )
        for row in df.itertuples(index=False)
    ]

    conn = connect_database()
    cursor = None

    try:
        cursor = conn.cursor()
        cursor.executemany(sql, rows)
        conn.commit()
        print(f"Scenario metrics loaded: {len(rows)} rows")
    except Exception:
        conn.rollback()
        raise
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()
