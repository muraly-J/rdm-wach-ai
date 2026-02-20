# Feature Documentation for Power Prediction Model

**Used for controllers:** `e0202`, `e0207`, `e0211`

This document describes the features used as input for the XGBoost regression model that predicts `power_total` for the specified controllers.

##  List of Features

| Feature Name      | Type     | Description                                                                 
|------------------|---------|----------------------------------------------------------------|
| power_factor_avg  | numeric | Average power factor at the timestamp.                                      |
| current_l1        | numeric | Current measured on phase L1.                                              |
| current_l2        | numeric | Current measured on phase L2.                                              |
| current_l3        | numeric | Current measured on phase L3.                                              |
| volts_l1_n        | numeric | Voltage from L1 to neutral.                                                |
| volts_l2_n        | numeric | Voltage from L2 to neutral.                                                |
| volts_l3_n        | numeric | Voltage from L3 to neutral.                                                |
| hour              | integer | Hour of the day (0–23) derived from timestamp.                             |
| dayofweek         | integer | Day of the week (0=Monday, 6=Sunday) derived from timestamp.               |
| month             | integer | Month of the year (1–12) derived from timestamp.                            |
| is_weekend        | binary  | 1 if the day is Saturday or Sunday, 0 otherwise.                            |
| lag_1             | numeric | `power_total` value from the previous timestep (15 minutes ago).           |
| lag_4             | numeric | `power_total` value from 1 hour ago (4 timesteps of 15 minutes each).      |
| lag_96            | numeric | `power_total` value from 1 day ago (96 timesteps of 15 minutes each).      |
| rolling_mean_4    | numeric | Rolling mean of `power_total` over the last 4 timesteps (1 hour).          |
| rolling_std_4     | numeric | Rolling standard deviation of `power_total` over the last 4 timesteps (1 hour). |