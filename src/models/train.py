# train_rolling_multi_controller_fixed.py

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- EVALUATION ----------------
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return mae, rmse, r2, mape

# ---------------- ROLLING TRAINING ----------------
def rolling_train(df, controller_name,
                  target_col="power_total",
                  rows_per_day=96,
                  train_days=60,
                  test_days=7):

    print(f"\nStarting rolling training for {controller_name}")

    # DROP UNNECESSARY COLUMNS
    drop_cols = ['result', 'table', 'controller', 'site', 'units']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # SELECT NUMERIC FEATURES ONLY
    feature_cols = df.drop(columns=[target_col], errors="ignore").select_dtypes(include=[np.number]).columns
    print(f"Features used for training ({len(feature_cols)}): {list(feature_cols)}")

    train_size = train_days * rows_per_day
    test_size = test_days * rows_per_day

    metrics_list = []
    all_test_outputs = []
    last_model = None
    split_counter = 1

    for start in range(0, len(df) - train_size - test_size + 1, test_size):
        train_idx = slice(start, start + train_size)
        test_idx = slice(start + train_size, start + train_size + test_size)

        X_train = df.iloc[train_idx][feature_cols]
        y_train = df.iloc[train_idx][target_col]
        X_test = df.iloc[test_idx][feature_cols]
        y_test = df.iloc[test_idx][target_col]

        print(f"\nSplit {split_counter} | Train: {X_train.shape} | Test: {X_test.shape}")

        # SAFETY CHECKS
        if len(X_train) == 0 or len(X_test) == 0:
            print("Skipped: Empty train/test window")
            split_counter += 1
            continue
        if y_train.nunique() <= 1:
            print("Skipped: Target is constant")
            split_counter += 1
            continue
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(0)
        if np.isinf(X_train).sum().sum() > 0:
            X_train = X_train.replace([np.inf, -np.inf], 0)

        # TRAIN MODEL
        try:
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            last_model = model
        except Exception as e:
            print(f"Split {split_counter} failed: {e}")
            split_counter += 1
            continue

        # EVALUATION
        mae, rmse, r2, mape = evaluate(y_test, y_pred)
        print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f} | MAPE: {mape:.2f}%")

        metrics_list.append({
            "split": split_counter,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2
        })

        split_df = pd.DataFrame({
            "split": split_counter,
            "time": y_test.index,
            "actual": y_test.values,
            "predicted": y_pred
        })
        all_test_outputs.append(split_df)
        split_counter += 1

    if len(metrics_list) == 0:
        raise ValueError("No valid splits were trained.")

    all_test_outputs_df = pd.concat(all_test_outputs, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_list)

    overall_metrics = {
        "num_splits": len(metrics_df),
        "mean_mae": metrics_df["mae"].mean(),
        "mean_rmse": metrics_df["rmse"].mean(),
        "mean_mape": metrics_df["mape"].mean(),
        "mean_r2": metrics_df["r2"].mean(),
        "std_mae": metrics_df["mae"].std(),
        "std_rmse": metrics_df["rmse"].std(),
        "std_mape": metrics_df["mape"].std(),
        "std_r2": metrics_df["r2"].std()
    }

    return all_test_outputs_df, metrics_df, overall_metrics, last_model

# ---------------- SAVE RESULTS ----------------
def save_results(test_output_df, metrics_df, overall_metrics,
                 final_model, controller_name):

    output_folder = os.path.join(r"D:\AHU\rdm-wach-ai\paraquet_data\models", controller_name)
    os.makedirs(output_folder, exist_ok=True)

    test_output_df.to_csv(os.path.join(output_folder, f"{controller_name}_test_predictions.csv"), index=False)
    metrics_df.to_csv(os.path.join(output_folder, f"{controller_name}_metrics_per_split.csv"), index=False)
    pd.DataFrame([overall_metrics]).to_csv(os.path.join(output_folder, f"{controller_name}_metrics_summary.csv"), index=False)
    joblib.dump(final_model, os.path.join(output_folder, f"{controller_name}_model.pkl"))

    print(f"\nSaved results AND model for {controller_name} at {output_folder}")

# ---------------- MAIN ----------------
if __name__ == "__main__":

    gold_folder = r"D:\AHU\rdm-wach-ai\paraquet_data\gold"
    feature_files = [f for f in os.listdir(gold_folder) if f.endswith(".parquet") and f.startswith("features_")]

    if not feature_files:
        print("No feature datasets found. Exiting.")
        exit()

    for file in feature_files:
        controller_name = file.replace("features_", "").replace(".parquet", "")
        path = os.path.join(gold_folder, file)
        print(f"\n==============================\nTraining controller: {controller_name}\n==============================")

        try:
            df = pd.read_parquet(path, engine="pyarrow")
            test_output_df, metrics_df, overall_metrics, final_model = rolling_train(df, controller_name)

            print("\nRolling Summary:")
            for k, v in overall_metrics.items():
                print(f"{k}: {v}")

            save_results(test_output_df, metrics_df, overall_metrics, final_model, controller_name)

        except Exception as e:
            print(f"Controller {controller_name} FAILED. Reason: {e}")
            continue

    print("\nAll controllers processed!")