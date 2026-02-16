"""
BACKTEST DIAGNOSTICS AND STABILITY ANALYSIS
Investigate why models collapse on certain windows

This script performs:
1. Target variance analysis per window
2. Baseline comparison (persistence)
3. Visual inspection of problem windows
4. Distribution drift detection
5. Feature importance analysis

Usage: python backtest_diagnostics.py --ahu_id e0206
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path("/Users/rdmasia/Documents/JINENDRA/rdm-wach-ai")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")


# ============================================================================
# PATHS
# ============================================================================

DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
OUTPUT_DIR = PROJECT_ROOT / "models" / "outputs" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# PHASE 1: TARGET VARIANCE ANALYSIS
# ============================================================================

def analyze_target_variance(df, window_config):
    """
    Compute target statistics per window to check for metric illusions
    
    Low target variance → R² becomes unreliable
    """
    print("\n" + "="*80)
    print("PHASE 1: TARGET VARIANCE ANALYSIS")
    print("="*80)
    print("\nGoal: Identify if R² collapse is due to low target variance")
    
    results = []
    
    train_days = window_config['train_days']
    test_days = window_config['test_days']
    step_days = window_config['step_days']
    
    start_date = df['time'].min()
    end_date = df['time'].max()
    
    window_idx = 0
    current_start = start_date
    
    while True:
        train_end = current_start + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        
        if test_end > end_date:
            break
        
        window_idx += 1
        
        # Filter data
        train_df = df[(df['time'] >= current_start) & (df['time'] < train_end)]
        test_df = df[(df['time'] >= train_end) & (df['time'] < test_end)]
        
        if len(test_df) == 0:
            break
        
        # Compute statistics
        train_power = train_df['power_total'].values
        test_power = test_df['power_total'].values
        
        results.append({
            'window': window_idx,
            'train_start': current_start.date(),
            'test_start': train_end.date(),
            'train_mean': train_power.mean(),
            'train_std': train_power.std(),
            'train_var': train_power.var(),
            'test_mean': test_power.mean(),
            'test_std': test_power.std(),
            'test_var': test_power.var(),
            'mean_shift': abs(test_power.mean() - train_power.mean()),
            'std_ratio': test_power.std() / train_power.std() if train_power.std() > 0 else 0
        })
        
        current_start += timedelta(days=step_days)
    
    results_df = pd.DataFrame(results)
    
    print("\nTarget Statistics Per Window:")
    print(results_df.to_string(index=False))
    
    # Identify problematic windows
    print("\n" + "-"*80)
    print("DIAGNOSTIC FINDINGS:")
    print("-"*80)
    
    # Check for low variance
    low_var_threshold = results_df['test_var'].mean() * 0.5
    low_var_windows = results_df[results_df['test_var'] < low_var_threshold]
    
    if len(low_var_windows) > 0:
        print(f"\nWARNING: {len(low_var_windows)} windows with unusually low test variance:")
        for _, row in low_var_windows.iterrows():
            print(f"  Window {row['window']}: test_var = {row['test_var']:.4f} (avg: {results_df['test_var'].mean():.4f})")
        print("  → Low variance makes R² unstable - even small errors destroy R²")
    
    # Check for distribution shift
    large_shift_threshold = results_df['mean_shift'].mean() + 2 * results_df['mean_shift'].std()
    shift_windows = results_df[results_df['mean_shift'] > large_shift_threshold]
    
    if len(shift_windows) > 0:
        print(f"\nWARNING: {len(shift_windows)} windows with large mean shift:")
        for _, row in shift_windows.iterrows():
            print(f"  Window {row['window']}: shift = {row['mean_shift']:.4f} kW")
        print("  → Distribution shift - model trained on different regime")
    
    # Check for variance change
    unstable_var = results_df[(results_df['std_ratio'] < 0.7) | (results_df['std_ratio'] > 1.3)]
    
    if len(unstable_var) > 0:
        print(f"\nWARNING: {len(unstable_var)} windows with variance instability:")
        for _, row in unstable_var.iterrows():
            print(f"  Window {row['window']}: std_ratio = {row['std_ratio']:.2f} (1.0 = stable)")
        print("  → Volatility changed - model may not adapt")
    
    # Save
    results_df.to_csv(OUTPUT_DIR / 'variance_analysis.csv', index=False)
    
    return results_df


# ============================================================================
# PHASE 2: BASELINE COMPARISON
# ============================================================================

def compare_with_baseline(df, window_config):
    """
    Compare model performance against persistence baseline
    
    If model barely beats baseline → feature problem
    """
    print("\n" + "="*80)
    print("PHASE 2: BASELINE COMPARISON")
    print("="*80)
    print("\nGoal: Check if models actually beat simple persistence")
    
    results = []
    
    train_days = window_config['train_days']
    test_days = window_config['test_days']
    step_days = window_config['step_days']
    
    start_date = df['time'].min()
    end_date = df['time'].max()
    
    window_idx = 0
    current_start = start_date
    
    while True:
        train_end = current_start + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        
        if test_end > end_date:
            break
        
        window_idx += 1
        
        train_df = df[(df['time'] >= current_start) & (df['time'] < train_end)]
        test_df = df[(df['time'] >= train_end) & (df['time'] < test_end)]
        
        if len(test_df) == 0:
            break
        
        y_test = test_df['power_total'].values
        
        # Persistence baseline
        last_train_value = train_df['power_total'].iloc[-1]
        y_pred_persistence = np.repeat(last_train_value, len(y_test))
        
        # Rolling mean baseline
        rolling_mean = train_df['power_total'].tail(96).mean()  # Last 24h
        y_pred_rolling = np.repeat(rolling_mean, len(y_test))
        
        # Metrics
        results.append({
            'window': window_idx,
            'persistence_mae': mean_absolute_error(y_test, y_pred_persistence),
            'persistence_r2': r2_score(y_test, y_pred_persistence),
            'rolling_mean_mae': mean_absolute_error(y_test, y_pred_rolling),
            'rolling_mean_r2': r2_score(y_test, y_pred_rolling)
        })
        
        current_start += timedelta(days=step_days)
    
    results_df = pd.DataFrame(results)
    
    print("\nBaseline Performance Per Window:")
    print(results_df.to_string(index=False))
    
    print("\n" + "-"*80)
    print("BASELINE SUMMARY:")
    print("-"*80)
    print(f"Persistence Baseline:")
    print(f"  Mean MAE: {results_df['persistence_mae'].mean():.4f} kW")
    print(f"  Mean R²:  {results_df['persistence_r2'].mean():.4f}")
    print(f"\nRolling Mean Baseline:")
    print(f"  Mean MAE: {results_df['rolling_mean_mae'].mean():.4f} kW")
    print(f"  Mean R²:  {results_df['rolling_mean_r2'].mean():.4f}")
    
    results_df.to_csv(OUTPUT_DIR / 'baseline_comparison.csv', index=False)
    
    return results_df


# ============================================================================
# PHASE 3: VISUAL INSPECTION OF PROBLEM WINDOWS
# ============================================================================

def visualize_problem_window(df, window_num, window_config):
    """
    Deep dive into a specific problematic window
    """
    print("\n" + "="*80)
    print(f"PHASE 3: VISUAL INSPECTION - WINDOW {window_num}")
    print("="*80)
    
    train_days = window_config['train_days']
    test_days = window_config['test_days']
    step_days = window_config['step_days']
    
    start_date = df['time'].min()
    current_start = start_date + timedelta(days=step_days * (window_num - 1))
    train_end = current_start + timedelta(days=train_days)
    test_end = train_end + timedelta(days=test_days)
    
    train_df = df[(df['time'] >= current_start) & (df['time'] < train_end)]
    test_df = df[(df['time'] >= train_end) & (df['time'] < test_end)]
    
    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    
    # Panel 1: Full train + test
    ax1 = axes[0]
    ax1.plot(train_df['time'], train_df['power_total'], 'o-', 
             label='Train', markersize=2, linewidth=0.5, alpha=0.7)
    ax1.plot(test_df['time'], test_df['power_total'], 's-', 
             label='Test', markersize=2, linewidth=0.5, color='red', alpha=0.7)
    ax1.axvline(x=train_end, color='black', linestyle='--', linewidth=2, label='Train/Test Split')
    ax1.set_ylabel('Power (kW)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Window {window_num}: Full Context', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Last 14 days of train + test
    last_14_train = train_df.tail(14*96)  # 14 days * 96 intervals/day
    ax2 = axes[1]
    ax2.plot(last_14_train['time'], last_14_train['power_total'], 'o-',
             label='Train (last 14 days)', markersize=2, linewidth=0.5)
    ax2.plot(test_df['time'], test_df['power_total'], 's-',
             label='Test (7 days)', markersize=2, linewidth=0.5, color='red')
    ax2.axvline(x=train_end, color='black', linestyle='--', linewidth=2)
    ax2.set_ylabel('Power (kW)', fontsize=12, fontweight='bold')
    ax2.set_title('Zoomed: Last 14 Days Train + Test Week', fontweight='bold', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Distribution comparison
    ax3 = axes[2]
    ax3.hist(train_df['power_total'], bins=50, alpha=0.5, label='Train', density=True)
    ax3.hist(test_df['power_total'], bins=50, alpha=0.5, label='Test', density=True, color='red')
    ax3.set_xlabel('Power (kW)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax3.set_title('Distribution Comparison', fontweight='bold', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Statistics
    ax4 = axes[3]
    ax4.axis('off')
    
    stats_text = f"""
    WINDOW {window_num} STATISTICS:
    
    Train Period: {current_start.date()} to {train_end.date()}
    Test Period:  {train_end.date()} to {test_end.date()}
    
    TRAIN STATISTICS:
      Mean:     {train_df['power_total'].mean():.3f} kW
      Std:      {train_df['power_total'].std():.3f} kW
      Min:      {train_df['power_total'].min():.3f} kW
      Max:      {train_df['power_total'].max():.3f} kW
      Range:    {train_df['power_total'].max() - train_df['power_total'].min():.3f} kW
    
    TEST STATISTICS:
      Mean:     {test_df['power_total'].mean():.3f} kW
      Std:      {test_df['power_total'].std():.3f} kW
      Min:      {test_df['power_total'].min():.3f} kW
      Max:      {test_df['power_total'].max():.3f} kW
      Range:    {test_df['power_total'].max() - test_df['power_total'].min():.3f} kW
    
    DISTRIBUTION SHIFT:
      Mean Shift:       {abs(test_df['power_total'].mean() - train_df['power_total'].mean()):.3f} kW
      Std Ratio:        {test_df['power_total'].std() / train_df['power_total'].std():.3f}
      Range Change:     {(test_df['power_total'].max() - test_df['power_total'].min()) - (train_df['power_total'].max() - train_df['power_total'].min()):.3f} kW
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'window_{window_num}_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nWindow {window_num} visualization saved to: {OUTPUT_DIR}")
    print(stats_text)


# ============================================================================
# PHASE 4: DISTRIBUTION DRIFT DETECTION
# ============================================================================

def detect_distribution_drift(df, window_config):
    """
    Systematic check for train/test distribution drift across all windows
    """
    print("\n" + "="*80)
    print("PHASE 4: DISTRIBUTION DRIFT DETECTION")
    print("="*80)
    
    variance_df = pd.read_csv(OUTPUT_DIR / 'variance_analysis.csv')
    
    # Categorize windows
    high_drift = variance_df[variance_df['mean_shift'] > variance_df['mean_shift'].mean() + variance_df['mean_shift'].std()]
    unstable_var = variance_df[(variance_df['std_ratio'] < 0.8) | (variance_df['std_ratio'] > 1.2)]
    
    print("\nDRIFT SUMMARY:")
    print(f"  High mean shift windows: {len(high_drift)}")
    print(f"  Unstable variance windows: {len(unstable_var)}")
    
    if len(high_drift) > 0:
        print("\nHigh Drift Windows:")
        print(high_drift[['window', 'mean_shift', 'train_mean', 'test_mean']].to_string(index=False))
    
    if len(unstable_var) > 0:
        print("\nUnstable Variance Windows:")
        print(unstable_var[['window', 'std_ratio', 'train_std', 'test_std']].to_string(index=False))
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    
    if len(high_drift) > 2:
        print("\n1. CRITICAL: Strong distribution drift detected")
        print("   → Add robust normalization (per-window scaling)")
        print("   → Add trend/detrending features")
        print("   → Consider online learning / adaptive models")
    
    if len(unstable_var) > 2:
        print("\n2. CRITICAL: Variance instability detected")
        print("   → Add rolling std features")
        print("   → Use heteroskedastic loss (weighted by uncertainty)")
        print("   → Consider quantile regression")
    
    if variance_df['test_var'].min() < variance_df['test_var'].mean() * 0.3:
        print("\n3. WARNING: Some test periods have very low variance")
        print("   → R² metric becomes unreliable")
        print("   → Focus on MAE/RMSE for these windows")
        print("   → Consider switching to MAPE or normalized metrics")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("BACKTEST DIAGNOSTICS AND STABILITY ANALYSIS")
    print("="*80)
    print("\nInvestigating model instability across rolling windows...")
    
    # Load data
    data_file = DATA_DIR / "clean_longest.csv"
    df = pd.read_csv(data_file)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    window_config = {
        'train_days': 60,
        'test_days': 7,
        'step_days': 7
    }
    
    # Phase 1: Variance analysis
    variance_df = analyze_target_variance(df, window_config)
    
    # Phase 2: Baseline comparison
    baseline_df = compare_with_baseline(df, window_config)
    
    # Phase 3: Visualize problem windows
    # Focus on windows 2, 3, 4 (the ones with low R²)
    for window_num in [2, 3, 4]:
        visualize_problem_window(df, window_num, window_config)
    
    # Phase 4: Drift detection
    detect_distribution_drift(df, window_config)
    
    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - variance_analysis.csv")
    print("  - baseline_comparison.csv")
    print("  - window_2_analysis.png")
    print("  - window_3_analysis.png")
    print("  - window_4_analysis.png")
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review window visualizations for regime changes")
    print("2. Compare model R² vs baseline R²")
    print("3. If baseline also fails → inherently hard periods")
    print("4. If model worse than baseline → feature engineering needed")
    print("5. Implement recommended fixes based on findings")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()