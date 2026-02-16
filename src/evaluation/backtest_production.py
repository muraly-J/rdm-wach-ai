"""
PRODUCTION ROLLING WINDOW BACKTESTING
TRUE retraining per window - Industrial Grade

This script performs proper time-series cross-validation by:
1. Creating rolling time windows
2. Training a NEW model on each window
3. Evaluating on future data
4. Computing robust statistics

NO DATA LEAKAGE. NO PERSISTENCE BASELINE. REAL TRAINING.

Location: src/evaluation/backtest.py
Usage: python backtest.py --model transformer --ahu_id e0206
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
PROJECT_ROOT = Path("/Users/rdmasia/Documents/JINENDRA/rdm-wach-ai")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import our modular components
from models.transformer import TransformerModel
from models.enhanced_nbeats import EnhancedNBeatsModel
from training.dataset import PowerDataset, engineer_features, create_sequences, get_default_features
from training.train_utils import train_model, predict

sns.set_style("whitegrid")


# ============================================================================
# PATHS
# ============================================================================

DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
OUTPUT_DIR = PROJECT_ROOT / "models" / "outputs" / "backtesting_production"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = {
    'transformer': {
        'input_size': 12,
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.2,
        'window_size': 192,
        'epochs': 50,  # Reduced for backtesting speed
        'learning_rate': 0.0001,
        'weight_decay': 1e-5,
        'patience': 10,
        'use_hybrid_loss': True,
        'mse_weight': 0.5,
        'huber_weight': 0.5,
        'batch_size': 64
    },
    'nbeats': {
        'input_size': 12,
        'window_size': 192,
        'num_stacks': 5,
        'num_blocks_per_stack': 4,
        'hidden_size': 256,
        'num_layers': 4,
        'dropout': 0.1,
        'share_weights': True,
        'polynomial_degree': 5,
        'num_harmonics': 20,
        'epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'patience': 10,
        'use_hybrid_loss': False,
        'batch_size': 64
    }
}


# ============================================================================
# BACKTESTING CONFIGURATION
# ============================================================================

BACKTEST_CONFIG = {
    'train_days': 60,
    'test_days': 7,
    'step_days': 7,
    'min_train_samples': 5000,
    'features': get_default_features()
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_data(ahu_id):
    """Load and prepare data for specific AHU"""
    data_file = DATA_DIR / f"clean_longest.csv"  # Adjust filename as needed
    
    df = pd.read_csv(data_file)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df = engineer_features(df)
    
    print(f"\nLoaded data for {ahu_id}:")
    print(f"  Samples: {len(df):,}")
    print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"  Duration: {(df['time'].max() - df['time'].min()).days} days")
    
    return df


def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
    
    r2 = r2_score(y_true, y_pred)
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}


def get_model(model_type, config):
    """Instantiate model based on type"""
    if model_type == 'transformer':
        return TransformerModel(config)
    elif model_type == 'nbeats':
        return EnhancedNBeatsModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# ROLLING WINDOW BACKTESTING
# ============================================================================

def rolling_window_backtest(df, model_type, model_config, backtest_config, device):
    """
    Perform rolling window backtesting with TRUE retraining
    
    CRITICAL: This function trains a NEW model for EACH window.
    No data leakage. No pretrained models. Real evaluation.
    """
    print("\n" + "="*80)
    print(f"ROLLING WINDOW BACKTEST: {model_type.upper()}")
    print("="*80)
    print("Methodology: Train new model per window")
    print(f"Train window: {backtest_config['train_days']} days")
    print(f"Test window: {backtest_config['test_days']} days")
    print(f"Step size: {backtest_config['step_days']} days")
    print("="*80)
    
    # Create sequences
    window_size = model_config['window_size']
    features = backtest_config['features']
    
    X, y, timestamps = create_sequences(df, window_size, features)
    
    seq_df = pd.DataFrame({
        'time': timestamps,
        'y_true': y.flatten()
    })
    
    # Calculate windows
    start_date = df['time'].min()
    end_date = df['time'].max()
    
    train_days = backtest_config['train_days']
    test_days = backtest_config['test_days']
    step_days = backtest_config['step_days']
    
    results = []
    all_predictions = []
    window_idx = 0
    current_start = start_date
    
    while True:
        train_end = current_start + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        
        if test_end > end_date:
            break
        
        window_idx += 1
        
        print(f"\n{'='*80}")
        print(f"WINDOW {window_idx}")
        print(f"{'='*80}")
        print(f"Train: {current_start.date()} to {train_end.date()}")
        print(f"Test:  {train_end.date()} to {test_end.date()}")
        
        # Filter data
        train_mask = (seq_df['time'] >= current_start) & (seq_df['time'] < train_end)
        test_mask = (seq_df['time'] >= train_end) & (seq_df['time'] < test_end)
        
        train_indices = seq_df[train_mask].index.tolist()
        test_indices = seq_df[test_mask].index.tolist()
        
        if len(train_indices) < backtest_config['min_train_samples']:
            print(f"SKIP: Insufficient training samples ({len(train_indices)})")
            current_start += timedelta(days=step_days)
            continue
        
        if len(test_indices) == 0:
            print(f"SKIP: No test samples")
            current_start += timedelta(days=step_days)
            continue
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        test_times = seq_df.loc[test_indices, 'time'].tolist()
        
        print(f"Train samples: {len(X_train):,}")
        print(f"Test samples:  {len(X_test):,}")
        
        # Scale data
        n_train, seq_len, n_features = X_train.shape
        scaler = StandardScaler()
        scaler.fit(X_train.reshape(-1, n_features))
        
        X_train_scaled = scaler.transform(X_train.reshape(-1, n_features)).reshape(n_train, seq_len, n_features)
        X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
        
        # Create datasets
        train_dataset = PowerDataset(X_train_scaled, y_train)
        
        # Split training into train/val (80/20)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_subset, batch_size=model_config['batch_size'], shuffle=False)
        val_loader = DataLoader(val_subset, batch_size=model_config['batch_size'], shuffle=False)
        test_dataset = PowerDataset(X_test_scaled, y_test)
        test_loader = DataLoader(test_dataset, batch_size=model_config['batch_size'], shuffle=False)
        
        # CRITICAL: Instantiate FRESH model
        print(f"\nInstantiating new {model_type} model...")
        model = get_model(model_type, model_config)
        
        # CRITICAL: Train model on THIS window
        print(f"Training model on window {window_idx}...")
        model, history = train_model(model, train_loader, val_loader, model_config, device)
        
        print(f"Training complete. Best val loss: {min(history['val_loss']):.6f}")
        
        # Predict on test set
        print("Generating predictions...")
        y_pred = predict(model, test_loader, device)
        
        # Compute metrics
        metrics = compute_metrics(y_test.flatten(), y_pred)
        
        print(f"\nWindow {window_idx} Results:")
        print(f"  MAE:  {metrics['mae']:.4f} kW")
        print(f"  RMSE: {metrics['rmse']:.4f} kW")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  R²:   {metrics['r2']:.4f}")
        
        # Store results
        results.append({
            'window': window_idx,
            'model': model_type,
            'train_start': current_start,
            'train_end': train_end,
            'test_start': train_end,
            'test_end': test_end,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'mape': metrics['mape'],
            'r2': metrics['r2'],
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        })
        
        # Store predictions
        for i, (time, true_val, pred_val) in enumerate(zip(test_times, y_test.flatten(), y_pred)):
            all_predictions.append({
                'window': window_idx,
                'model': model_type,
                'time': time,
                'y_true': true_val,
                'y_pred': pred_val,
                'error': abs(true_val - pred_val)
            })
        
        current_start += timedelta(days=step_days)
    
    return pd.DataFrame(results), pd.DataFrame(all_predictions)


# ============================================================================
# ANALYSIS AND VISUALIZATION
# ============================================================================

def analyze_and_visualize(results_df, predictions_df, model_type, ahu_id):
    """Analyze and visualize backtest results"""
    print("\n" + "="*80)
    print(f"BACKTEST SUMMARY: {model_type.upper()} - {ahu_id}")
    print("="*80)
    
    print(f"\nWindows tested: {len(results_df)}")
    print(f"\nAggregate Metrics:")
    print(f"  Mean MAE:  {results_df['mae'].mean():.4f} ± {results_df['mae'].std():.4f} kW")
    print(f"  Mean RMSE: {results_df['rmse'].mean():.4f} ± {results_df['rmse'].std():.4f} kW")
    print(f"  Mean MAPE: {results_df['mape'].mean():.2f} ± {results_df['mape'].std():.2f}%")
    print(f"  Mean R²:   {results_df['r2'].mean():.4f} ± {results_df['r2'].std():.4f}")
    
    print(f"\nRobustness Metrics:")
    print(f"  Best R²:   {results_df['r2'].max():.4f} (window {results_df.loc[results_df['r2'].idxmax(), 'window']})")
    print(f"  Worst R²:  {results_df['r2'].min():.4f} (window {results_df.loc[results_df['r2'].idxmin(), 'window']})")
    print(f"  Stability: {results_df['r2'].mean() - results_df['r2'].std():.4f} (mean - std)")
    
    # Plot
    model_output_dir = OUTPUT_DIR / model_type / ahu_id
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # R² per window
    ax1 = axes[0]
    ax1.plot(results_df['window'], results_df['r2'], 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.axhline(y=results_df['r2'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean = {results_df["r2"].mean():.3f}')
    ax1.fill_between(results_df['window'],
                      results_df['r2'].mean() - results_df['r2'].std(),
                      results_df['r2'].mean() + results_df['r2'].std(),
                      alpha=0.2, color='red')
    ax1.set_ylabel('R²', fontsize=12, fontweight='bold')
    ax1.set_title(f'{model_type.upper()} - R² Across Rolling Windows', fontweight='bold', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # MAE per window
    ax2 = axes[1]
    ax2.plot(results_df['window'], results_df['mae'], 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax2.axhline(y=results_df['mae'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean = {results_df["mae"].mean():.3f} kW')
    ax2.set_ylabel('MAE (kW)', fontsize=12, fontweight='bold')
    ax2.set_title('MAE Across Rolling Windows', fontweight='bold', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Predictions sample
    ax3 = axes[2]
    sample = predictions_df.sample(min(1000, len(predictions_df))).sort_values('time')
    ax3.plot(sample['time'], sample['y_true'], 'o-', label='Actual',
             markersize=2, linewidth=1, alpha=0.7, color='black')
    ax3.plot(sample['time'], sample['y_pred'], 's-', label='Predicted',
             markersize=2, linewidth=1, alpha=0.7, color='#F18F01')
    ax3.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Power (kW)', fontsize=12, fontweight='bold')
    ax3.set_title('Predictions vs Actual (Sample)', fontweight='bold', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(model_output_dir / 'backtest_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to: {model_output_dir}")
    
    return {
        'mean_mae': results_df['mae'].mean(),
        'std_mae': results_df['mae'].std(),
        'mean_r2': results_df['r2'].mean(),
        'std_r2': results_df['r2'].std(),
        'worst_r2': results_df['r2'].min(),
        'stability': results_df['r2'].mean() - results_df['r2'].std()
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Rolling Window Backtesting')
    parser.add_argument('--model', type=str, default='transformer',
                       choices=['transformer', 'nbeats'],
                       help='Model type to test')
    parser.add_argument('--ahu_id', type=str, default='e0206',
                       help='AHU identifier')
    parser.add_argument('--all_models', action='store_true',
                       help='Test all models')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("PRODUCTION ROLLING WINDOW BACKTESTING")
    print("="*80)
    print("TRUE RETRAINING PER WINDOW")
    print("NO DATA LEAKAGE")
    print("="*80)
    
    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    df = load_data(args.ahu_id)
    
    # Test models
    models_to_test = ['transformer', 'nbeats'] if args.all_models else [args.model]
    
    all_summaries = []
    
    for model_type in models_to_test:
        model_config = MODEL_CONFIGS[model_type]
        
        # Run backtest
        results_df, predictions_df = rolling_window_backtest(
            df, model_type, model_config, BACKTEST_CONFIG, device
        )
        
        # Analyze
        summary = analyze_and_visualize(results_df, predictions_df, model_type, args.ahu_id)
        summary['model'] = model_type
        all_summaries.append(summary)
        
        # Save results
        output_dir = OUTPUT_DIR / model_type / args.ahu_id
        results_df.to_csv(output_dir / 'results.csv', index=False)
        predictions_df.to_csv(output_dir / 'predictions.csv', index=False)
    
    # Final comparison
    if len(all_summaries) > 1:
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        comparison_df = pd.DataFrame(all_summaries)
        print("\n", comparison_df.to_string(index=False))
        comparison_df.to_csv(OUTPUT_DIR / f'{args.ahu_id}_comparison.csv', index=False)
    
    print("\n" + "="*80)
    print("BACKTESTING COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()