CREATE DATABASE IF NOT EXISTS monte_carlo_analytics;
use monte_carlo_analytics;

create table historical_prices (
	ticker VARCHAR(10) NOT NULL,
	trade_date DATE NOT NULL,
	adjusted_close DOUBLE NOT NULL,
	PRIMARY KEY (ticker, trade_date),
	check (adjusted_close > 0)
);

create table simulation_runs (
	run_id varchar(100) primary key,
	ticker varchar(10) not null,
	model varchar(30) not null,
	train_start date not null,
	train_end date not null,
	forecast_horizon int not null,
	n_sims int not null,
	seed int not null,
	initial_price double not null,
	check (train_end >= train_start),
	check (forecast_horizon > 0),
	check (n_sims > 0),
	check (initial_price > 0)
);

create table simulation_price_summary (
	run_id varchar(100) not null,
	day int not null,
	mean_price double not null,
	median_price double not null,
	p05_price double not null,
	p95_price double not null,
	primary key (run_id, day),
	foreign key (run_id)
		references simulation_runs(run_id),
	check (day >= 0),
	check (mean_price > 0),
	check (median_price > 0),
	check (p05_price > 0),
	check (p95_price > 0),
	check (p05_price <= p95_price)
);

CREATE TABLE simulation_metrics (
    run_id VARCHAR(100) PRIMARY KEY,
    initial_price DOUBLE NOT NULL,
    mean_end_price DOUBLE NOT NULL,
    median_end_price DOUBLE NOT NULL,
    p05_end_price DOUBLE NOT NULL,
    p95_end_price DOUBLE NOT NULL,
    prob_loss DOUBLE NOT NULL,
    log_loss DOUBLE NOT NULL,
    simple_loss DOUBLE NOT NULL,
    alpha DOUBLE NOT NULL,
    var_log DOUBLE NOT NULL,
    es_log DOUBLE NOT NULL,
    var_simple DOUBLE NOT NULL,
    es_simple DOUBLE NOT NULL,

    FOREIGN KEY (run_id)
        REFERENCES simulation_runs(run_id),

    CHECK (initial_price > 0),
    CHECK (mean_end_price > 0),
    CHECK (prob_loss BETWEEN 0 AND 1),
    CHECK (alpha > 0 AND alpha < 1)
);

CREATE TABLE backtest_results (
    train_start DATE NOT NULL,
    train_end DATE NOT NULL,
    test_start DATE NOT NULL,
    test_end DATE NOT NULL,
    horizon_days INT NOT NULL,
    realized_cumlog DOUBLE NOT NULL,
    p05 DOUBLE NOT NULL,
    p95 DOUBLE NOT NULL,
    inside_band BOOLEAN NOT NULL,
    band_width DOUBLE NOT NULL,
    pit DOUBLE NOT NULL,

    PRIMARY KEY (train_start, test_start, horizon_days),

    CHECK (train_end >= train_start),
    CHECK (test_end >= test_start),
    CHECK (test_start > train_end),
    CHECK (horizon_days > 0),
    CHECK (p05 <= p95),
    CHECK (band_width >= 0),
    CHECK (pit BETWEEN 0 AND 1)
);