# evaluate.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def residuals(y_true, y_pred):
    return y_true - y_pred

def compute_alerts(residuals, sigma=3):
    threshold = sigma * residuals.std()
    alerts = np.abs(residuals) > threshold
    return alerts, threshold

def plot_residuals(residuals, alerts, threshold, title="Residual-based Alerting", save_path=None):
    plt.figure(figsize=(15,5))
    plt.plot(residuals.index, residuals, label="Residuals")
    plt.axhline(threshold, color='red', linestyle='--', label="Upper Threshold")
    plt.axhline(-threshold, color='red', linestyle='--', label="Lower Threshold")
    plt.scatter(residuals.index[alerts], residuals[alerts], color='black', label="Alerts")
    plt.legend()
    plt.title(title)
    plt.xlabel("Index / Time")
    plt.ylabel("Residual")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Residual plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def evaluate_from_csv(csv_path, sigma=3):
    # Find '# Test Predictions' section
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    start_line = 0
    for i, line in enumerate(lines):
        if line.strip() == "# Test Predictions":
            start_line = i + 1
            break

    # Read only test predictions
    df = pd.read_csv(csv_path, skiprows=start_line)

    if "Unnamed: 0" in df.columns:
        df = df.set_index("Unnamed: 0")

    if "actual" not in df.columns or "predicted" not in df.columns:
        raise ValueError(f"CSV must contain 'actual' and 'predicted' columns. Found: {df.columns.tolist()}")

    y_true = df["actual"]
    y_pred = df["predicted"]

    res = residuals(y_true, y_pred)
    alerts, threshold = compute_alerts(res, sigma=sigma)

    print(f"Residual std: {res.std():.3f}")
    print(f"Alert threshold ({sigma}-sigma): ±{threshold:.3f}")
    print(f"Number of alerts: {alerts.sum()} out of {len(res)}")

    # Save plot
    plot_path = os.path.join(os.path.dirname(csv_path), "residual_plot.png")
    plot_residuals(res, alerts, threshold, save_path=plot_path)

    return res, alerts, threshold

# ---- Example usage ----
if __name__ == "__main__":
    project_root = r"D:/AHU/rdm-wach-ai/paraquet_data"
    csv_path = os.path.join(project_root, "models", "xgb_ahu_test_predictions.csv")

    residuals_values, alert_flags, alert_threshold = evaluate_from_csv(csv_path)
