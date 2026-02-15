# train.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, r2, mape

def train_model(df, target_col="power_total"):
    df = df.copy()
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Chronological split
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_metrics = evaluate(y_train, y_train_pred)
    test_metrics = evaluate(y_test, y_test_pred)
    
    print("Train Metrics: MAE={:.3f}, RMSE={:.3f}, R2={:.3f}, MAPE={:.2f}%".format(*train_metrics))
    print("Test  Metrics: MAE={:.3f}, RMSE={:.3f}, R2={:.3f}, MAPE={:.2f}%".format(*test_metrics))
    
    # Create test output dataframe
    test_output = pd.DataFrame({
        "actual": y_test.values,
        "predicted": y_test_pred
    }, index=y_test.index)  # Keep original indices
    
    return model, X_test, y_test, y_test_pred, test_output, train_metrics, test_metrics

def save_model(model, project_root, filename="xgb_ahu_model.pkl"):
    models_folder = os.path.join(project_root, "models")
    os.makedirs(models_folder, exist_ok=True)
    path = os.path.join(models_folder, filename)
    joblib.dump(model, path)
    print(f"Saved model to {path}")
    return path

def save_test_output(df_output, train_metrics, test_metrics, project_root, filename="xgb_ahu_test_predictions.csv"):
    """
    Saves test predictions along with Train/Test metrics at the top of the CSV.
    """
    output_folder = os.path.join(project_root, "models")
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, filename)
    
    # Prepare metrics as a small dataframe
    metrics_df = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R2", "MAPE"],
        "Train": [f"{train_metrics[0]:.3f}", f"{train_metrics[1]:.3f}", f"{train_metrics[2]:.3f}", f"{train_metrics[3]:.2f}%"],
        "Test":  [f"{test_metrics[0]:.3f}",  f"{test_metrics[1]:.3f}",  f"{test_metrics[2]:.3f}",  f"{test_metrics[3]:.2f}%"]
    })
    
    # Write metrics first, then append predictions
    with open(path, 'w') as f:
        f.write("# Train/Test Metrics\n")
        metrics_df.to_csv(f, index=False)
        f.write("\n# Test Predictions\n")
        df_output.to_csv(f, index=True)
    
    print(f"Saved test predictions with metrics to {path}")
    return path

# ---- Example usage ----
if __name__ == "__main__":
    project_root = r"D:/AHU/rdm-wach-ai/paraquet_data"
    gold_path = os.path.join(project_root,"gold","ahu_features.parquet")
    
    # Load data
    df = pd.read_parquet(gold_path, engine="pyarrow")
    
    # Train model and get test predictions
    model, X_test, y_test, y_pred, test_output, train_metrics, test_metrics = train_model(df)
    
    # Save trained model
    save_model(model, project_root)
    
    # Save test predictions CSV with metrics
    save_test_output(test_output, train_metrics, test_metrics, project_root)
