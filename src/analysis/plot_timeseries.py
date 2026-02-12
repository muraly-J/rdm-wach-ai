import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

report_path = "../../paraquet_data/reports/"
plot_path = report_path + "plots/"
os.makedirs(plot_path, exist_ok=True)

files = glob.glob("../../paraquet_data/raw/*.parquet")

for file in files:

    df = pd.read_parquet(file)
    print("Processing:", file)

    # Detect Time Column Safely
    possible_time_cols = ["time", "_time", "timestamp", "datetime"]

    time_col = None
    for col in possible_time_cols:
        if col in df.columns:
            time_col = col
            break

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)
        df.set_index(time_col, inplace=True)

    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) == 0:
        continue

    #  TIME SERIES (Top 5 columns)
    top_cols = numeric_cols[:5]

    plt.figure(figsize=(14,6))
    for col in top_cols:
        plt.plot(df.index, df[col], label=col)

    plt.legend()
    plt.title("Time Series Trend")
    plt.tight_layout()
    plt.savefig(plot_path + "timeseries.png")
    plt.close()

    #  ROLLING MEAN & STD (Fluctuation)
    col = numeric_cols[0]

    rolling_mean = df[col].rolling(window=50).mean()
    rolling_std = df[col].rolling(window=50).std()

    plt.figure(figsize=(14,6))
    plt.plot(df[col], label="Original")
    plt.plot(rolling_mean, label="Rolling Mean")
    plt.plot(rolling_std, label="Rolling Std")
    plt.legend()
    plt.title(f"Rolling Statistics - {col}")
    plt.tight_layout()
    plt.savefig(plot_path + "rolling_stats.png")
    plt.close()

    #  CORRELATION HEATMAP
    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(12,10))
    sns.heatmap(corr_matrix, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(plot_path + "correlation_heatmap.png")
    plt.close()

    #  SCATTER vs TARGET
    target = "power_total"
    if target in numeric_cols:

        top_corr = corr_matrix[target].abs().sort_values(ascending=False).index[1:6]

        for col in top_corr:
            plt.figure(figsize=(6,4))
            plt.scatter(df[col], df[target])
            plt.xlabel(col)
            plt.ylabel(target)
            plt.title(f"{col} vs {target}")
            plt.tight_layout()
            plt.savefig(plot_path + f"scatter_{col}.png")
            plt.close()

    # HISTOGRAM DISTRIBUTION
        plt.figure(figsize=(6,4))
        df[col].hist(bins=50)
        plt.title(f"Distribution - {col}")
        plt.tight_layout()
        plt.savefig(plot_path + f"hist_{col}.png")
        plt.close()

    # BOXPLOT (Outlier Detection)
    plt.figure(figsize=(14,6))
    df[numeric_cols[:10]].boxplot(rot=90)
    plt.title("Boxplot - Outlier Detection")
    plt.tight_layout()
    plt.savefig(plot_path + "boxplot.png")
    plt.close()
    
    # AUTOCORRELATION (For Forecasting)
    from pandas.plotting import autocorrelation_plot

    col = numeric_cols[0]

    plt.figure(figsize=(8,5))
    autocorrelation_plot(df[col])
    plt.title(f"Autocorrelation - {col}")
    plt.tight_layout()
    plt.savefig(plot_path + "autocorrelation.png")
    plt.close()

print("Full EDA plots generated successfully.")
