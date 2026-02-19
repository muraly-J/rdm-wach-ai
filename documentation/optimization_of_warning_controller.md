# XGBoost Model Evaluation – AHU Controllers

This document summarizes the evaluation metrics for XGBoost models on different controllers (`e0201`, `e0204`, `e0205`, `e0209`) after feature engineering, hyperparameter tuning, lag and rolling window adjustments, and outlier handling.

## Controller: e0201

model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    objective='reg:squarederror',
    random_state=42
)
| Split | MAE    | RMSE   | MAPE  | R²     |
| ----- | ------ | ------ | ----- | ------ |
| 1     | 0.0306 | 0.0658 | 0.92% | 0.9358 |
| 2     | 0.0864 | 0.1777 | 2.89% | 0.8224 |
| 3     | 0.1062 | 0.1664 | 4.82% | 0.9314 |
| 4     | 0.0477 | 0.0851 | 2.56% | 0.9814 |
| 5     | 0.0420 | 0.0793 | 2.12% | 0.9873 |

Model 2: Tuned Hyperparameters

model = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    objective='reg:squarederror',
    random_state=42
)
| Split | MAE    | RMSE   | MAPE  | R²     |
| ----- | ------ | ------ | ----- | ------ |
| 1     | 0.0307 | 0.0631 | 0.93% | 0.9409 |
| 2     | 0.0826 | 0.1703 | 2.75% | 0.8369 |
| 3     | 0.1067 | 0.1733 | 4.88% | 0.9256 |
| 4     | 0.0471 | 0.0841 | 2.55% | 0.9819 |
| 5     | 0.0413 | 0.0773 | 2.21% | 0.9879 |

Observation: Hyperparameter tuning mostly affects R² slightly; errors (MAE, RMSE, MAPE) remain similar.

Feature Engineering & Rolling (Lag=1, Rolling=3)
| Split | MAE    | RMSE   | MAPE  | R²     |
| ----- | ------ | ------ | ----- | ------ |
| 1     | 0.0346 | 0.0652 | 1.05% | 0.9383 |
| 2     | 0.0843 | 0.1717 | 2.80% | 0.8025 |
| 3     | 0.0940 | 0.1635 | 4.28% | 0.9354 |
| 4     | 0.0580 | 0.0978 | 2.87% | 0.9761 |
| 5     | 0.0308 | 0.0613 | 1.29% | 0.9887 |

Notes:

Outliers increase MAPE dramatically. After removing 1% spikes:
| Split | MAE    | RMSE   | MAPE  | R²     |
| ----- | ------ | ------ | ----- | ------ |
| 1     | 0.0813 | 0.1724 | 2.69% | 0.6569 |
| 2     | 0.0528 | 0.0898 | 1.80% | 0.9654 |
| 3     | 0.0810 | 0.1409 | 4.21% | 0.9488 |
| 4     | 0.0382 | 0.0665 | 1.82% | 0.9884 |
| 5     | 0.0425 | 0.0859 | 2.39% | 0.9848 |

Final After Feature Engineering
| Split | MAE    | RMSE   | MAPE  | R²     |
| ----- | ------ | ------ | ----- | ------ |
| 1     | 0.0366 | 0.0700 | 1.18% | 0.9434 |
| 2     | 0.0352 | 0.0652 | 1.20% | 0.9817 |
| 3     | 0.0556 | 0.1100 | 2.91% | 0.9688 |
| 4     | 0.0344 | 0.0647 | 1.69% | 0.9890 |
| 5     | 0.0355 | 0.0694 | 1.92% | 0.9901 |

# Mean Values:

MAE: 0.0395

RMSE: 0.0759

MAPE: 1.78% (previously 2.71%)

R²: 0.9746

##  Controller: e0204 
| Split | MAE     | RMSE    | MAPE  | R²      |
| ----- | ------- | ------- | ----- | ------- |
| 1     | 0.00688 | 0.00894 | 0.18% | 0.9768  |
| 2     | 0.00991 | 0.03404 | 0.28% | 0.9812  |
| 3     | 0.01049 | 0.03559 | 0.28% | -3.7653 |
| 4     | 0.00849 | 0.01286 | 0.22% | 0.8446  |
| 5     | 0.00851 | 0.01169 | 0.23% | 0.9494  |

After Feature Engineering & Hyperparameter Tuning

| Split | MAE    | RMSE   | MAPE  | R²      |
| ----- | ------ | ------ | ----- | ------- |
| 1     | 0.0068 | 0.0086 | 0.18% | 0.9786  |
| 2     | 0.0108 | 0.0264 | 0.32% | 0.9887  |
| 3     | 0.0123 | 0.0484 | 0.32% | -7.7897 |
| 4     | 0.0075 | 0.0097 | 0.20% | 0.9110  |
| 5     | 0.0079 | 0.0102 | 0.21% | 0.9616  |

Note: Split 3 contains very low variance week → negative R² lowers mean R².

## Controller: e0205
| Split | MAE    | RMSE   | MAPE  | R²      |
| ----- | ------ | ------ | ----- | ------- |
| 1     | 0.0037 | 0.0047 | 0.11% | 0.7973  |
| 2     | 0.0199 | 0.0561 | 1.07% | 0.9530  |
| 3     | 0.0064 | 0.0192 | 0.19% | -0.6146 |
| 4     | 0.0033 | 0.0043 | 0.10% | 0.7773  |
| 5     | 0.0034 | 0.0048 | 0.10% | 0.9029  |

Observation: Negative R² in Split 3, but small MAE/RMSE → do not change, model is acceptable for production.

## Controller: e0209

Before Feature Engineering

| Split | MAE     | RMSE    | MAPE  | R²      |
| ----- | ------- | ------- | ----- | ------- |
| 1     | 0.01697 | 0.02101 | 0.22% | 0.9243  |
| 2     | 0.03639 | 0.21306 | 6.95% | 0.8229  |
| 3     | 0.02510 | 0.12735 | 0.35% | -0.0518 |
| 4     | 0.02663 | 0.07081 | 0.38% | 0.9545  |
| 5     | 0.02204 | 0.03670 | 0.29% | 0.9090  |

After Feature Engineering & Rolling
| Split | MAE    | RMSE   | MAPE  | R²     |
| ----- | ------ | ------ | ----- | ------ |
| 1     | 0.0146 | 0.0183 | 0.19% | 0.9428 |
| 2     | 0.0300 | 0.2098 | 6.57% | 0.8281 |
| 3     | 0.0177 | 0.0513 | 0.24% | 0.8292 |
| 4     | 0.0284 | 0.0952 | 0.47% | 0.9178 |

## Observation:

 Feature engineering fixes instability in Split 3 (negative R² before). Outliers in Split 2 still increase MAPE.

Feature engineering and rolling windows significantly improve stability and R², especially in splits with previously negative or very low R².

Hyperparameter tuning alone does not significantly reduce MAE or RMSE, mostly affects R² slightly.

Outlier removal reduces MAPE spikes and improves reliability in production.

Controllers e0205 and e0209 show that even with small negative R² in a split, low MAE/RMSE can make the model acceptable for production.