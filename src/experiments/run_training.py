"""
EXPERIMENT RUNNER
Train models once and save for deployment

Location: src/experiments/run_training.py
Usage: python run_training.py --model transformer --ahu_id e0206
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path("/Users/rdmasia/Documents/JINENDRA/rdm-wach-ai")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import joblib
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.transformer import TransformerModel
from models.enhanced_nbeats import EnhancedNBeatsModel
from training.dataset import PowerDataset, engineer_features, create_sequences, get_default_features
from training.train_utils import train_model, predict

sns.set_style("whitegrid")


# ============================================================================
# PATHS
# ============================================================================

DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training_experiments"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# MODEL CONFIGS
# ============================================================================

CONFIGS = {
    'transformer': {
        'input_size': 12,
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.2,
        'window_size': 192,
        'epochs': 150,
        'learning_rate': 0.0001,
        'weight_decay': 1e-5,
        'patience': 20,
        'use_hybrid_loss': True,
        'mse_weight': 0.5,
        'huber_weight': 0.5,
        'batch_size': 64,
        'train_ratio': 0.70,
        'val_ratio': 0.15
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
        'epochs': 250,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'patience': 25,
        'use_hybrid_loss': False,
        'batch_size': 64,
        'train_ratio': 0.70,
        'val_ratio': 0.15
    }
}


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train_and_save_model(model_type, ahu_id, device):
    """
    Complete training pipeline:
    1. Load data
    2. Create sequences
    3. Train model
    4. Evaluate
    5. Save artifacts
    """
    print("\n" + "="*80)
    print(f"TRAINING {model_type.upper()} MODEL")
    print(f"AHU: {ahu_id}")
    print("="*80)
    
    config = CONFIGS[model_type]
    
    # Load data
    data_file = DATA_DIR / "clean_longest.csv"
    df = pd.read_csv(data_file)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df = engineer_features(df)
    
    print(f"\nData loaded: {len(df):,} samples")
    
    # Create sequences
    X, y, timestamps = create_sequences(df, config['window_size'], get_default_features())
    print(f"Sequences created: {len(X):,}")
    
    # Split data
    n = len(X)
    train_end = int(n * config['train_ratio'])
    val_end = int(n * (config['train_ratio'] + config['val_ratio']))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train):,}")
    print(f"  Val:   {len(X_val):,}")
    print(f"  Test:  {len(X_test):,}")
    
    # Scale
    n_train, seq_len, n_features = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))
    
    X_train_scaled = scaler.transform(X_train.reshape(-1, n_features)).reshape(n_train, seq_len, n_features)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    
    # Datasets
    train_dataset = PowerDataset(X_train_scaled, y_train)
    val_dataset = PowerDataset(X_val_scaled, y_val)
    test_dataset = PowerDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize model
    if model_type == 'transformer':
        model = TransformerModel(config)
    elif model_type == 'nbeats':
        model = EnhancedNBeatsModel(config)
    else:
        raise ValueError(f"Unknown model: {model_type}")
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\nTraining...")
    model, history = train_model(model, train_loader, val_loader, config, device)
    
    print(f"\nTraining complete!")
    print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"  Best val loss: {min(history['val_loss']):.6f}")
    
    # Evaluate on test
    print("\nEvaluating on test set...")
    y_pred = predict(model, test_loader, device)
    y_test_flat = y_test.flatten()
    
    mae = mean_absolute_error(y_test_flat, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred))
    mape = np.mean(np.abs((y_test_flat - y_pred) / y_test_flat)) * 100
    r2 = r2_score(y_test_flat, y_pred)
    
    print("\nTest Metrics:")
    print(f"  MAE:  {mae:.4f} kW")
    print(f"  RMSE: {rmse:.4f} kW")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")
    
    # Save artifacts
    model_save_dir = MODEL_DIR / model_type
    model_save_dir.mkdir(exist_ok=True)
    
    torch.save(model.state_dict(), model_save_dir / f"{ahu_id}.pth")
    joblib.dump(scaler, model_save_dir / f"{ahu_id}_scaler.pkl")
    
    with open(model_save_dir / f"{ahu_id}_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nModel saved to: {model_save_dir}")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_type.upper()} Training History', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    sample_size = min(500, len(y_test_flat))
    axes[1].plot(y_test_flat[:sample_size], 'o-', label='Actual', markersize=2, linewidth=1)
    axes[1].plot(y_pred[:sample_size], 's-', label='Predicted', markersize=2, linewidth=1)
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Power (kW)')
    axes[1].set_title('Predictions vs Actual', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_plot_dir = OUTPUT_DIR / model_type
    output_plot_dir.mkdir(exist_ok=True)
    plt.savefig(output_plot_dir / f'{ahu_id}_training.png', dpi=150)
    plt.close()
    
    print(f"Training plots saved to: {output_plot_dir}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train and save models')
    parser.add_argument('--model', type=str, required=True,
                       choices=['transformer', 'nbeats'],
                       help='Model type')
    parser.add_argument('--ahu_id', type=str, default='e0206',
                       help='AHU identifier')
    
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    metrics = train_and_save_model(args.model, args.ahu_id, device)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nFinal Test Metrics:")
    print(f"  MAE:  {metrics['mae']:.4f} kW")
    print(f"  R²:   {metrics['r2']:.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()