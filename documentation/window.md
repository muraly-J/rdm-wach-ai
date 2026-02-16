# AHU Energy Forecasting – XGBoost Model
# Project Overview

This project implements a time-series energy forecasting model for AHU (Air Handling Unit) systems using:

15-minute interval data

Feature engineering (time features + lag + rolling statistics)

Walk-forward (rolling window) validation

XGBoost regression model

The objective is to predict:

power_total (kW)

# Data Description

Data Frequency: 15-minute intervals

Rows per day: 96

One year of historical data

Target Variable: power_total

# Key Feature Types
A. Electrical Features

power_l1, power_l2, power_l3

apparent_power_total

current_l1, current_l2, current_l3

volts_l1_n, volts_l2_n, volts_l3_n

power_factor_avg

B. Time-Based Features

hour

dayofweek

month

is_weekend

C. Lag Features

lag_1 (15 min)

lag_4 (1 hour)

lag_96 (1 day)

D. Rolling Statistics

rolling_mean_4

rolling_std_4

<!-- # Data Cleaning -->

<!-- Out-of-range values were replaced with NaN based on domain thresholds:

Feature Type	Range Applied
Power	0 – 10 kW
Current	0 – 20 A
Voltage	200 – 260 V
Power Factor	0 – 1

Rows with missing lag/rolling values were dropped before modeling. -->

# Validation Strategy

We used Rolling Window Backtesting (Walk-Forward Validation).

Parameters:

Training Window: 60 days

Testing Window: 7 days

Rows per day: 96

Number of splits: 5

This approach ensures:

No data leakage

Realistic production simulation

# Model Configuration

Model used: XGBoost Regressor

n_estimators = 500
max_depth = 5
learning_rate = 0.05
objective = reg:squarederror
random_state = 42

## Backtesting Results
## Overall Performance (Average Across Splits)


| Metric     | Value   |
|------------|---------|
| Mean MAE   | 0.1028  |
| Mean RMSE  | 0.1456  |
| Mean MAPE  | 2.92%   |
| Mean R²    | 0.8231  |


##  Rolling Window Backtesting Results

| Split | MAE    | RMSE   | MAPE  | R²     |
|-------|--------|--------|-------|--------|
| 1     | 0.2137 | 0.2710 | 6.06% | 0.4191 |
| 2     | 0.1830 | 0.2607 | 4.89% | 0.7933 |
| 3     | 0.0634 | 0.1046 | 2.19% | 0.9231 |
| 4     | 0.0202 | 0.0375 | 0.64% | 0.9885 |
| 5     | 0.0334 | 0.0541 | 0.84% | 0.9915 |
