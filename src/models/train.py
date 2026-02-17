# train_rolling_multi_controller.py

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#  EVALUATION FUNCTION
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return mae, rmse, r2, mape


# ROLLING WINDOW TRAINING
def rolling_train(df, target_col="power_total", rows_per_day=96, train_days=60, test_days=7):
    feature_cols = df.drop(columns=[target_col, "controller"]).columns  # Exclude target & controller

    train_size = train_days * rows_per_day
    test_size = test_days * rows_per_day

    metrics_list = []
    all_test_outputs = []

    for start in range(0, len(df) - train_size - test_size + 1, test_size):
        split_num = start // test_size + 1
        train_idx = slice(start, start + train_size)
        test_idx = slice(start + train_size, start + train_size + test_size)

        X_train = df.iloc[train_idx][feature_cols]
        y_train = df.iloc[train_idx][target_col]

        X_test = df.iloc[test_idx][feature_cols]
        y_test = df.iloc[test_idx][target_col]

        # Train XGBoost
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            objective='reg:squarederror',
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate
        mae, rmse, r2, mape = evaluate(y_test, y_pred)

        # Store per-split metrics
        metrics_list.append({
            "split": split_num,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2
        })

        # Store test predictions
        split_df = pd.DataFrame({
            "split": split_num,
            "time": y_test.index,
            "actual": y_test.values,
            "predicted": y_pred
        })
        all_test_outputs.append(split_df)

    # Combine all test predictions
    all_test_outputs_df = pd.concat(all_test_outputs, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_list)

    # Overall metrics
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

    return all_test_outputs_df, metrics_df, overall_metrics


#  SAVE RESULTS PER CONTROLLER
def save_results(test_output_df, metrics_df, overall_metrics, project_root, controller_name):
    output_folder = os.path.join(project_root, "models", controller_name)
    os.makedirs(output_folder, exist_ok=True)

    # Save test predictions
    test_csv_path = os.path.join(output_folder, f"{controller_name}_test_predictions.csv")
    test_output_df.to_csv(test_csv_path, index=False)

    # Save per-split metrics
    metrics_csv_path = os.path.join(output_folder, f"{controller_name}_metrics_per_split.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    # Save overall summary metrics
    overall_csv_path = os.path.join(output_folder, f"{controller_name}_metrics_summary.csv")
    pd.DataFrame([overall_metrics]).to_csv(overall_csv_path, index=False)

    print(f"Saved predictions and metrics for {controller_name} in {output_folder}")
    return test_csv_path, metrics_csv_path, overall_csv_path


#  MAIN PROCESS
if __name__ == "__main__":
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    gold_folder = os.path.join(project_root, "paraquet_data", "gold")

    # Get all per-controller feature datasets
    feature_files = [f for f in os.listdir(gold_folder) if f.endswith(".parquet") and f.startswith("features_")]
    if not feature_files:
        print("No per-controller feature datasets found. Run feature_engineering.py first.")
        exit()

    for file in feature_files:
        controller_name = file.replace("features_", "").replace(".parquet", "")
        path = os.path.join(gold_folder, file)
        print(f"\n--- Training controller: {controller_name} ---")

        df = pd.read_parquet(path, engine="pyarrow")

        # Rolling train
        test_output_df, metrics_df, overall_metrics = rolling_train(df)

        # Show summary
        print(f"\nController {controller_name} Rolling Window Summary:")
        for k, v in overall_metrics.items():
            print(f"{k}: {v}")

        # Save per-controller results
        save_results(test_output_df, metrics_df, overall_metrics, project_root, controller_name)

    print("\nAll controllers trained and evaluated successfully!")
