# evaluate_with_plots.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# HELPER FUNCTION TO LOAD CONTROLLER RESULTS
def load_controller_results(models_folder, controller_name):
    """Load test predictions and metrics for a controller"""
    output_folder = os.path.join(models_folder, controller_name)
    
    try:
        predictions_path = os.path.join(output_folder, f"{controller_name}_test_predictions.csv")
        per_split_path = os.path.join(output_folder, f"{controller_name}_metrics_per_split.csv")
        summary_path = os.path.join(output_folder, f"{controller_name}_metrics_summary.csv")

        test_df = pd.read_csv(predictions_path)
        metrics_df = pd.read_csv(per_split_path)
        overall_df = pd.read_csv(summary_path)

        return test_df, metrics_df, overall_df
    except FileNotFoundError:
        print(f"⚠️  Controller {controller_name} results not found. Skipping...")
        return None, None, None

# PLOT PER-CONTROLLER METRICS
def plot_controller_metrics(controller_name, metrics_df, save_folder):
    """Plot MAE, RMSE, R2 per split for a controller"""
    if metrics_df is None or metrics_df.empty:
        return None

    plt.figure(figsize=(10, 5))
    plt.plot(metrics_df["split"], metrics_df["mae"], marker='o', label="MAE")
    plt.plot(metrics_df["split"], metrics_df["rmse"], marker='s', label="RMSE")
    plt.plot(metrics_df["split"], metrics_df["r2"], marker='^', label="R²")
    plt.xlabel("Split Number")
    plt.ylabel("Metric Value")
    plt.title(f"Controller: {controller_name} - Metrics per Split")
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_folder, exist_ok=True)
    img_path = os.path.join(save_folder, f"{controller_name}_metrics.png")
    plt.savefig(img_path)
    plt.close()
    print(f"Saved plot for {controller_name} → {img_path}")
    return img_path

# PLOT COMBINED METRICS SUMMARY
def plot_combined_metrics(combined_df, save_folder):
    """Create a combined metrics plot for all controllers"""
    plt.figure(figsize=(12, 6))
    metrics = ["mean_mae", "mean_rmse", "mean_r2"]
    for metric in metrics:
        sns.barplot(x="controller", y=metric, data=combined_df, label=metric)
    plt.xticks(rotation=45)
    plt.ylabel("Metric Value")
    plt.title("Combined Metrics Across Controllers")
    plt.legend()
    plt.tight_layout()

    combined_img_path = os.path.join(save_folder, "combined_metrics.png")
    plt.savefig(combined_img_path)
    plt.close()
    print(f" Saved combined metrics plot → {combined_img_path}")
    return combined_img_path

# MAIN PROCESS
if __name__ == "__main__":

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    models_folder = os.path.join(project_root, "models")
    plot_folder = os.path.join(models_folder, "plots")
    os.makedirs(plot_folder, exist_ok=True)

    # Get all controller folders
    controller_folders = [f for f in os.listdir(models_folder)
                          if os.path.isdir(os.path.join(models_folder, f))]

    combined_summary = []

    for controller_name in controller_folders:
        test_df, metrics_df, overall_df = load_controller_results(models_folder, controller_name)

        if test_df is None:
            continue  # skip controllers without results

        # Plot individual controller metrics
        plot_controller_metrics(controller_name, metrics_df, plot_folder)

        # Append for combined CSV and overall metrics
        if overall_df is not None and not overall_df.empty:
            overall_metrics = overall_df.copy()
            overall_metrics["controller"] = controller_name
            combined_summary.append(overall_metrics)

    # Create combined summary plot
    if combined_summary:
        combined_df = pd.concat(combined_summary, ignore_index=True)
        combined_summary_csv = os.path.join(models_folder, "combined_metrics_summary.csv")
        combined_df.to_csv(combined_summary_csv, index=False)
        print(f"\n Combined summary metrics saved to: {combined_summary_csv}")

        # Save combined plot
        plot_combined_metrics(combined_df, plot_folder)
    else:
        print("\nNo controller metrics found to combine.")

    print("\n Evaluation with plots completed for all controllers!")
