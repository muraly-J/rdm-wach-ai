"""
LSTM POWER FORECASTING MODEL - e0206 AHU
Deep Learning Time Series Forecasting with LSTM Networks

Architecture:
- Bidirectional LSTM layers for forward/backward temporal context
- Dropout for regularization
- Dense layers for final prediction
- GPU-accelerated training (MPS for Mac Studio)

Input: 96 timesteps x 12 features
Output: 1-step ahead power prediction (15 minutes)

Dataset: ~9,480 samples (98 days) - sufficient for deep learning
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Absolute paths for Mac
PROJECT_ROOT = Path("/Users/rdmasia/Documents/JINENDRA/rdm-wach-ai")
DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "lstm_training"

# Create directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Files
DATA_FILE = DATA_DIR / "clean_longest.csv"
MODEL_FILE = MODEL_DIR / "lstm_e0206.pth"
SCALER_FILE = MODEL_DIR / "lstm_scaler_e0206.pkl"
CONFIG_FILE = MODEL_DIR / "lstm_config_e0206.json"


# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

CONFIG = {
    # Data parameters
    'window_size': 96,          # 24 hours lookback
    'forecast_horizon': 1,      # 1-step ahead (15 min)
    'train_ratio': 0.70,        # 70% train
    'val_ratio': 0.15,          # 15% validation
    'test_ratio': 0.15,         # 15% test
    
    # Model architecture
    'input_size': 12,           # Number of features
    'hidden_size': 128,         # LSTM hidden units
    'num_layers': 3,            # Number of LSTM layers
    'dropout': 0.2,             # Dropout rate
    'bidirectional': True,      # Use bidirectional LSTM
    
    # Training parameters
    'batch_size': 64,
    'epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'patience': 15,             # Early stopping patience
    
    # Features
    'features': [
        'power_total',
        'power_factor_avg',
        'current_l1', 'current_l2', 'current_l3',
        'voltage_avg',
        'apparent_power_total',
        'hour', 'day_of_week', 'weekend',
        'rolling_mean_4h', 'power_delta'
    ]
}


# ============================================================================
# LSTM MODEL ARCHITECTURE
# ============================================================================

class LSTMForecaster(nn.Module):
    """
    Bidirectional LSTM network for power forecasting
    
    Architecture:
    - Input: [batch, seq_len=96, features=12]
    - 3x Bidirectional LSTM layers (128 hidden units each)
    - Dropout layers for regularization
    - Fully connected layers for prediction
    - Output: [batch, 1] - next power value
    """
    
    def __init__(self, config):
        super(LSTMForecaster, self).__init__()
        
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.bidirectional = config['bidirectional']
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [batch, seq_len, features]
        
        Returns:
            output: [batch, 1]
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(last_output))
        out = self.dropout_layer(out)
        out = self.relu(self.fc2(out))
        out = self.dropout_layer(out)
        out = self.fc3(out)
        
        return out


# ============================================================================
# DATA PREPARATION
# ============================================================================

class PowerDataset(Dataset):
    """PyTorch Dataset for power forecasting"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_prepare_data(filepath):
    """Load CSV and prepare features"""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    print(f"Loaded {len(df):,} samples")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    
    return df


def engineer_features(df):
    """Engineer features for LSTM model"""
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    # Time features
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Voltage average
    df['voltage_avg'] = df[['volts_l1_n', 'volts_l2_n', 'volts_l3_n']].mean(axis=1)
    
    # Rolling features
    df['rolling_mean_4h'] = df['power_total'].rolling(window=16, min_periods=1).mean()
    df['power_delta'] = df['power_total'].diff().fillna(0)
    
    print("Features engineered successfully")
    return df


def create_sequences(df, config):
    """Create sequences for LSTM training"""
    print("\n" + "="*80)
    print("CREATING SEQUENCES")
    print("="*80)
    
    features = config['features']
    window_size = config['window_size']
    
    # Extract feature columns
    data = df[features].values
    
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, :])
        y.append(data[i, 0])  # power_total is first feature
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    print(f"Created {len(X):,} sequences")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    return X, y


def split_data(X, y, config):
    """Split data into train/val/test"""
    print("\n" + "="*80)
    print("SPLITTING DATA")
    print("="*80)
    
    n = len(X)
    train_end = int(n * config['train_ratio'])
    val_end = int(n * (config['train_ratio'] + config['val_ratio']))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"Train: {len(X_train):,} samples ({len(X_train)/n*100:.1f}%)")
    print(f"Val:   {len(X_val):,} samples ({len(X_val)/n*100:.1f}%)")
    print(f"Test:  {len(X_test):,} samples ({len(X_test)/n*100:.1f}%)")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def scale_data(train_data, val_data, test_data):
    """Scale features using StandardScaler"""
    print("\n" + "="*80)
    print("SCALING FEATURES")
    print("="*80)
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # Reshape for scaling
    n_train, seq_len, n_features = X_train.shape
    
    # Fit scaler on train data
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, n_features)
    scaler.fit(X_train_reshaped)
    
    # Transform all sets
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(n_train, seq_len, n_features)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    
    print("Scaling complete")
    
    return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test), scaler


# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    """LSTM model trainer"""
    
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader, criterion):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def fit(self, train_loader, val_loader):
        """Full training loop"""
        print("\n" + "="*80)
        print("TRAINING LSTM MODEL")
        print("="*80)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.validate(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:3d}/{self.config['epochs']} | "
                      f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), MODEL_FILE)
            else:
                patience_counter += 1
            
            if patience_counter >= self.config['patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        self.model.load_state_dict(torch.load(MODEL_FILE))
        print(f"\nBest validation loss: {best_val_loss:.6f}")
        
        return self.history
    
    def predict(self, dataloader):
        """Generate predictions"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for X_batch, _ in dataloader:
                X_batch = X_batch.to(self.device)
                preds = self.model(X_batch)
                predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions).flatten()


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(y_true, y_pred, name="Model"):
    """Calculate evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'model': name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2
    }


def plot_results(history, y_test, y_pred):
    """Plot training history and predictions"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Training history
    ax1 = axes[0]
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('LSTM Training History', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Predictions vs Actual
    ax2 = axes[1]
    test_indices = range(len(y_test))
    ax2.plot(test_indices, y_test, 'o-', label='Actual', 
             markersize=2, linewidth=1, alpha=0.7)
    ax2.plot(test_indices, y_pred, 's-', label='Predicted', 
             markersize=2, linewidth=1, alpha=0.7)
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Power (kW)')
    ax2.set_title('LSTM Predictions vs Actual', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lstm_results.png', dpi=150)
    plt.close()
    
    print(f"Results plot saved to: {OUTPUT_DIR / 'lstm_results.png'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("LSTM POWER FORECASTING - e0206 AHU")
    print("="*80)
    
    # Check GPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"\nUsing device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nUsing device: CUDA GPU")
    else:
        device = torch.device('cpu')
        print(f"\nUsing device: CPU")
    
    # Load and prepare data
    df = load_and_prepare_data(DATA_FILE)
    df = engineer_features(df)
    
    # Create sequences
    X, y = create_sequences(df, CONFIG)
    
    # Split data
    train_data, val_data, test_data = split_data(X, y, CONFIG)
    
    # Scale data
    train_data, val_data, test_data, scaler = scale_data(train_data, val_data, test_data)
    
    # Create datasets
    train_dataset = PowerDataset(*train_data)
    val_dataset = PowerDataset(*val_data)
    test_dataset = PowerDataset(*test_data)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Initialize model
    model = LSTMForecaster(CONFIG)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    # Train model
    trainer = Trainer(model, device, CONFIG)
    history = trainer.fit(train_loader, val_loader)
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    y_pred = trainer.predict(test_loader)
    y_test = test_data[1].flatten()
    
    metrics = evaluate_model(y_test, y_pred, "LSTM")
    print(f"\nTest Metrics:")
    print(f"  MAE:  {metrics['MAE']:.4f} kW")
    print(f"  RMSE: {metrics['RMSE']:.4f} kW")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  R²:   {metrics['R²']:.4f}")
    
    # Save artifacts
    import joblib
    joblib.dump(scaler, SCALER_FILE)
    
    with open(CONFIG_FILE, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    # Plot results
    plot_results(history, y_test, y_pred)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nSaved artifacts:")
    print(f"  Model: {MODEL_FILE}")
    print(f"  Scaler: {SCALER_FILE}")
    print(f"  Config: {CONFIG_FILE}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()