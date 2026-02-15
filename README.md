# rdm-wach-ai

AI-driven facility operations intelligence platform.

Initial focus:
- AHU energy forecasting (N-BEATS)
- Residual-based alerting
- InfluxDB + PostgreSQL integration

Roadmap:
- Agentic workflows
- Operations chatbot

# AHU Energy Forecasting
- XGBoost-Based Power Prediction with Residual-Based Alerting

# Project Overview

This project implements a machine learning-based energy forecasting system for an Air Handling Unit (AHU).

The system:

Predicts total AHU power consumption (power_total)

Evaluates performance against baseline models

Detects abnormal behavior using residual-based alerting (3-sigma rule)

The goal is to:

Accurately forecast energy usage

Detect faults, inefficiencies, or abnormal energy spikes

# Methodology
1️ Forecasting Model

Used:

XGBoost Regressor

Chronological 80/20 train-test split

Target variable: power_total

Model configuration:

n_estimators=500

max_depth=5

learning_rate=0.05

objective="reg:squarederror"

# Model Performance

Train performance
- MAE  : 0.0031
- RMSE : 0.0049
- R²   : 1.0000
- MAPE : 0.10%

Test performance 
- MAE  : 0.0613
- RMSE : 0.0888
- R²   : 0.9725
- MAPE : 1.64%

# Baseline Comparisons
 Naive Baseline 
- MAE  : 0.2723
- RMSE : 0.3925
- R²   : 0.4639
- MAPE : 7.86%

 Seasonal Naive (1-dat lag)
- MAE  : 0.3949
- RMSE : 0.5227
- R²   : 0.0492
- MAPE : 11.20%

# Interpretation

The XGBoost model significantly outperforms naive baselines

Test R² = 0.9725 → Very strong predictive performance

MAPE = 1.64% → Excellent forecasting accuracy