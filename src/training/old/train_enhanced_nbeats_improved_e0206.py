"""
IMPROVED ENHANCED N-BEATS - e0206 AHU
Maximum Capacity Seasonal Decomposition Model

Improvements:
- Increased Fourier harmonics: 20 (vs 10)
- Increased polynomial degree: 5 (vs 3)
- Extended training: 250 epochs
- Weight sharing enabled
- Extended patience: 25

Expected: MAE < 0.25 kW, R² > 0.65
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path("/Users/rdmasia/Documents/JINENDRA/rdm-wach-ai")
DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
OUTPUT_DIR = PROJECT_ROOT / "models" / "outputs" / "nbeats_improved_training"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "clean_longest.csv"
MODEL_FILE = MODEL_DIR / "nbeats_improved_e0206.pth"
SCALER_FILE = MODEL_DIR / "nbeats_improved_scaler_e0206.pkl"
CONFIG_FILE = MODEL_DIR / "nbeats_improved_config_e0206.json"


# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    'window_size': 192,          # INCREASED: 48 hours context
    'forecast_horizon': 1,
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    
    # Improved N-BEATS architecture
    'input_size': 12,
    'num_stacks': 5,
    'num_blocks_per_stack': 4,
    'hidden_size': 256,
    'num_layers': 4,
    'dropout': 0.1,
    'share_weights': True,        # ENABLED weight sharing
    'polynomial_degree': 5,       # INCREASED from 3
    'num_harmonics': 20,          # INCREASED from 10
    
    'batch_size': 64,
    'epochs': 250,                # INCREASED from 100
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'patience': 25,               # INCREASED from 15
    
    'features': [
        'power_total', 'power_factor_avg',
        'current_l1', 'current_l2', 'current_l3',
        'voltage_avg', 'apparent_power_total',
        'hour', 'day_of_week', 'weekend',
        'rolling_mean_4h', 'power_delta'
    ]
}


# ============================================================================
# N-BEATS BLOCKS
# ============================================================================

class NBeatsBlock(nn.Module):
    """Generic N-BEATS block"""
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(NBeatsBlock, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.fc_layers = nn.Sequential(*layers)
        self.backcast_linear = nn.Linear(hidden_size, input_size)
        self.forecast_linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h = self.fc_layers(x)
        backcast = self.backcast_linear(h)
        forecast = self.forecast_linear(h)
        return backcast, forecast


class TrendBlock(nn.Module):
    """Trend block with HIGHER polynomial degree"""
    def __init__(self, input_size, hidden_size, num_layers, dropout, degree=5):
        super(TrendBlock, self).__init__()
        
        self.degree = degree
        self.input_size = input_size
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.fc_layers = nn.Sequential(*layers)
        self.backcast_coef = nn.Linear(hidden_size, degree + 1)
        self.forecast_coef = nn.Linear(hidden_size, degree + 1)
    
    def forward(self, x):
        h = self.fc_layers(x)
        backcast_theta = self.backcast_coef(h)
        backcast = self._trend_basis(backcast_theta, self.input_size)
        forecast_theta = self.forecast_coef(h)
        forecast = self._trend_basis(forecast_theta, 1)
        return backcast, forecast
    
    def _trend_basis(self, theta, T):
        batch_size = theta.size(0)
        t = torch.arange(0, T, dtype=torch.float, device=theta.device) / T
        t = t.unsqueeze(0).repeat(batch_size, 1)
        basis = torch.stack([t ** i for i in range(self.degree + 1)], dim=2)
        trend = torch.sum(basis * theta.unsqueeze(1), dim=2)
        return trend


class SeasonalityBlock(nn.Module):
    """Seasonality block with MORE harmonics"""
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_harmonics=20):
        super(SeasonalityBlock, self).__init__()
        
        self.num_harmonics = num_harmonics
        self.input_size = input_size
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.fc_layers = nn.Sequential(*layers)
        self.backcast_coef = nn.Linear(hidden_size, 2 * num_harmonics)
        self.forecast_coef = nn.Linear(hidden_size, 2 * num_harmonics)
    
    def forward(self, x):
        h = self.fc_layers(x)
        backcast_theta = self.backcast_coef(h)
        backcast = self._seasonality_basis(backcast_theta, self.input_size)
        forecast_theta = self.forecast_coef(h)
        forecast = self._seasonality_basis(forecast_theta, 1)
        return backcast, forecast
    
    def _seasonality_basis(self, theta, T):
        batch_size = theta.size(0)
        t = torch.arange(0, T, dtype=torch.float, device=theta.device) / T
        t = t.unsqueeze(0).repeat(batch_size, 1)
        
        basis_list = []
        for i in range(1, self.num_harmonics + 1):
            basis_list.append(torch.sin(2 * np.pi * i * t))
            basis_list.append(torch.cos(2 * np.pi * i * t))
        
        basis = torch.stack(basis_list, dim=2)
        seasonality = torch.sum(basis * theta.unsqueeze(1), dim=2)
        return seasonality


# ============================================================================
# IMPROVED N-BEATS MODEL
# ============================================================================

class ImprovedNBeats(nn.Module):
    """
    Improved N-BEATS with:
    - Weight sharing across blocks
    - Higher polynomial degree (5)
    - More harmonics (20)
    - Extended context (192)
    """
    def __init__(self, config):
        super(ImprovedNBeats, self).__init__()
        
        self.input_size = config['input_size'] * config['window_size']
        self.share_weights = config['share_weights']
        
        # Generic stacks (2 stacks)
        if self.share_weights:
            # Create shared block
            generic_block = NBeatsBlock(
                self.input_size, config['hidden_size'],
                config['num_layers'], config['dropout']
            )
            self.generic_stacks = nn.ModuleList([
                nn.ModuleList([generic_block for _ in range(config['num_blocks_per_stack'])])
                for _ in range(2)
            ])
        else:
            self.generic_stacks = nn.ModuleList([
                nn.ModuleList([
                    NBeatsBlock(self.input_size, config['hidden_size'],
                               config['num_layers'], config['dropout'])
                    for _ in range(config['num_blocks_per_stack'])
                ]) for _ in range(2)
            ])
        
        # Trend stacks (2 stacks)
        if self.share_weights:
            trend_block = TrendBlock(
                self.input_size, config['hidden_size'],
                config['num_layers'], config['dropout'],
                degree=config['polynomial_degree']
            )
            self.trend_stacks = nn.ModuleList([
                nn.ModuleList([trend_block for _ in range(config['num_blocks_per_stack'])])
                for _ in range(2)
            ])
        else:
            self.trend_stacks = nn.ModuleList([
                nn.ModuleList([
                    TrendBlock(self.input_size, config['hidden_size'],
                              config['num_layers'], config['dropout'],
                              degree=config['polynomial_degree'])
                    for _ in range(config['num_blocks_per_stack'])
                ]) for _ in range(2)
            ])
        
        # Seasonal stack (1 stack)
        if self.share_weights:
            seasonal_block = SeasonalityBlock(
                self.input_size, config['hidden_size'],
                config['num_layers'], config['dropout'],
                num_harmonics=config['num_harmonics']
            )
            self.seasonal_stack = nn.ModuleList([seasonal_block for _ in range(config['num_blocks_per_stack'])])
        else:
            self.seasonal_stack = nn.ModuleList([
                SeasonalityBlock(self.input_size, config['hidden_size'],
                                config['num_layers'], config['dropout'],
                                num_harmonics=config['num_harmonics'])
                for _ in range(config['num_blocks_per_stack'])
            ])
    
    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)
        
        residual = x_flat
        forecast_sum = 0
        
        # Generic stacks
        for stack in self.generic_stacks:
            for block in stack:
                backcast, forecast = block(residual)
                residual = residual - backcast
                forecast_sum = forecast_sum + forecast
        
        # Trend stacks
        for stack in self.trend_stacks:
            for block in stack:
                backcast, forecast = block(residual)
                residual = residual - backcast
                forecast_sum = forecast_sum + forecast
        
        # Seasonal stack
        for block in self.seasonal_stack:
            backcast, forecast = block(residual)
            residual = residual - backcast
            forecast_sum = forecast_sum + forecast
        
        return forecast_sum


# ============================================================================
# DATA & TRAINING
# ============================================================================

class PowerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'])
    return df.sort_values('time').reset_index(drop=True)


def engineer_features(df):
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['voltage_avg'] = df[['volts_l1_n', 'volts_l2_n', 'volts_l3_n']].mean(axis=1)
    df['rolling_mean_4h'] = df['power_total'].rolling(window=16, min_periods=1).mean()
    df['power_delta'] = df['power_total'].diff().fillna(0)
    return df


def create_sequences(df, config):
    print(f"\nCreating sequences with window size: {config['window_size']}")
    data = df[config['features']].values
    X, y = [], []
    for i in range(config['window_size'], len(data)):
        X.append(data[i-config['window_size']:i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y).reshape(-1, 1)


def split_scale_data(X, y, config):
    n = len(X)
    train_end = int(n * config['train_ratio'])
    val_end = int(n * (config['train_ratio'] + config['val_ratio']))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    n_train, seq_len, n_features = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))
    
    X_train_scaled = scaler.transform(X_train.reshape(-1, n_features)).reshape(n_train, seq_len, n_features)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    
    return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test), scaler


class Trainer:
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            optimizer.zero_grad()
            loss = criterion(self.model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)
    
    def validate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                loss = criterion(self.model(X_batch), y_batch)
                total_loss += loss.item()
        return total_loss / len(dataloader)
    
    def fit(self, train_loader, val_loader):
        print("\n" + "="*80)
        print("TRAINING IMPROVED N-BEATS")
        print("="*80)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.validate(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{self.config['epochs']} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
            
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
        print(f"\nBest val loss: {best_val_loss:.6f}")
        return self.history
    
    def predict(self, dataloader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for X_batch, _ in dataloader:
                predictions.extend(self.model(X_batch.to(self.device)).cpu().numpy())
        return np.array(predictions).flatten()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("IMPROVED N-BEATS - e0206 AHU")
    print("="*80)
    print("Enhancements:")
    print("  - Extended context: 192 timesteps")
    print("  - Polynomial degree: 5")
    print("  - Fourier harmonics: 20")
    print("  - Weight sharing: Enabled")
    print("  - Extended training: 250 epochs")
    print("="*80)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    df = load_and_prepare_data(DATA_FILE)
    df = engineer_features(df)
    X, y = create_sequences(df, CONFIG)
    train_data, val_data, test_data, scaler = split_scale_data(X, y, CONFIG)
    
    train_loader = DataLoader(PowerDataset(*train_data), CONFIG['batch_size'], shuffle=False)
    val_loader = DataLoader(PowerDataset(*val_data), CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(PowerDataset(*test_data), CONFIG['batch_size'], shuffle=False)
    
    model = ImprovedNBeats(CONFIG)
    print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = Trainer(model, device, CONFIG)
    history = trainer.fit(train_loader, val_loader)
    
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    y_pred = trainer.predict(test_loader)
    y_test = test_data[1].flatten()
    
    print(f"\nMetrics:")
    print(f"  MAE:  {mean_absolute_error(y_test, y_pred):.4f} kW")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f} kW")
    print(f"  MAPE: {np.mean(np.abs((y_test - y_pred) / y_test)) * 100:.2f}%")
    print(f"  R²:   {r2_score(y_test, y_pred):.4f}")
    
    import joblib
    joblib.dump(scaler, SCALER_FILE)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0].set_title('Improved N-BEATS Training', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(y_test[:500], 'o-', label='Actual', markersize=2, linewidth=1)
    axes[1].plot(y_pred[:500], 's-', label='Predicted', markersize=2, linewidth=1)
    axes[1].set_title('Predictions', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'nbeats_improved_results.png', dpi=150)
    plt.close()
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nSaved: {MODEL_FILE}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()