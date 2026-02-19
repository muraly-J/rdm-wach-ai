## Project Overview

This project involves rolling window training and evaluation of controllers for predicting system metrics (like power, current, etc.) using historical data. The goal is to analyze controller performance, compare errors, and identify which controllers perform reliably.

We performed rolling training for multiple controllers using train-test splits and calculated standard performance metrics:

 MAE (Mean Absolute Error) – average magnitude of errors.

 RMSE (Root Mean Squared Error) – penalizes larger errors more than MAE.

 MAPE (Mean Absolute Percentage Error) – error as a percentage.

 R² (Coefficient of Determination) – indicates goodness of fit; 1 is perfect, negative means poor fit.


## Data & Controllers

Number of controllers trained: 14

e0201, e0202, e0203

raw_data_e0201 → raw_data_e0211 (11 controllers)

 Training data: Each split had ~5760 rows and 16–20 features depending on the controller.

 Testing data: Each split had ~672 rows.

 Rolling splits: 3 splits for e0201–e0203, 5 splits for raw_data controllers.

## Goal: Compare model performance and stability across splits.


## Steps Performed

1) Data Preparation

   Selected features from raw controller data (16–20 features depending on controller).

    Prepared train-test splits for rolling window evaluation.

2) Rolling Window Training

    For each split: trained the model on the train set and predicted on the test set.

    Recorded MAE, RMSE, MAPE, R² for each split.

3) Summary Calculation

    Computed mean and standard deviation of metrics across splits.

    Stored results for all controllers.

4) Analysis

## Compared mean metrics across controllers.

Checked for anomalies (e.g., negative R² or very high MAPE).


## Controller Performance Overview (Color-Coded)

## Controller Performance Overview (Compact)

| Controller     | Spl | MAE    | σMAE  | RMSE   | σRMSE | MAPE   | σMAPE  | R²      | σR²    | Status     |
|---------------|----:|-------:|------:|-------:|------:|-------:|-------:|--------:|-------:|-----------|
| raw_data_e0201|  5  | 0.064  | 0.033 | 0.117  | 0.055 | 2.71%  | 1.43%  | 0.929   | 0.070  | ⚠️ Warning |
| raw_data_e0202|  5  | 0.007  | 0.001 | 0.013  | 0.003 | 0.33%  | 0.04%  | 0.962   | 0.024  | ✅ Good    |
| raw_data_e0203|  5  | 0.107  | 0.214 | 0.135  | 0.220 | 5.68%  | 11.34% | -1.046  | 2.729  | ❌ Bad     |
| raw_data_e0204|  5  | 0.009  | 0.001 | 0.021  | 0.013 | 0.24%  | 0.04%  | -0.003  | 2.104  | ⚠️ Warning |
| raw_data_e0205|  5  | 0.007  | 0.007 | 0.016  | 0.022 | 0.32%  | 0.44%  | 0.789   | 0.140  | ⚠️ Warning |
| raw_data_e0206|  5  | 0.163  | 0.163 | 0.221  | 0.188 | 4.63%  | 4.56%  | 0.470   | 0.970  | ❌ Bad     |
| raw_data_e0207|  5  | 0.005  | 0.001 | 0.014  | 0.013 | 0.20%  | 0.08%  | 0.918   | 0.065  | ✅ Good    |
| raw_data_e0208|  5  | 0.017  | 0.007 | 0.065  | 0.046 | 0.34%  | 0.13%  | -0.470  | 2.426  | ❌ Bad     |
| raw_data_e0209|  5  | 0.025  | 0.007 | 0.094  | 0.078 | 1.64%  | 2.97%  | 0.712   | 0.430  | ⚠️ Warning |
| raw_data_e0210|  5  | 0.020  | 0.018 | 0.067  | 0.061 | 9417358% | 17994759% | 0.998 | 0.004  | ❌ Bad     |
| raw_data_e0211|  5  | 0.032  | 0.005 | 0.064  | 0.016 | 0.60%  | 0.14%  | 0.994   | 0.004  | ✅ Good    |



## Observations

    Controllers e0201, e0202, raw_data_e0202, raw_data_e0207 show stable and low errors.

    Controllers like e0203, raw_data_e0203, raw_data_e0206, raw_data_e0208, raw_data_e0210 show high variability or negative R², indicating poor predictive performance or data issues.

    Very high MAPE in raw_data_e0210 indicates likely division by near-zero values or outliers in the data.

    Rolling window analysis helps identify controllers that are consistently performing well versus controllers that are unstable across time.