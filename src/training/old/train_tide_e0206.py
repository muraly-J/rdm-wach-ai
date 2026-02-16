"""
TiDE (TIME SERIES DENSE ENCODER) - e0206 AHU
Modern, Efficient Time Series Forecasting (Google Research 2023)

Architecture:
- Dense encoder-decoder structure
- Residual connections
- Feature projection
- Very parameter efficient
- Fast training and inference

Advantages:
- Fewer parameters than Transformer/LSTM
- Fast inference
- Strong empirical performance
- Simple yet effective

Input: 96 timesteps x 12 features
Output: 1-step ahead prediction
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
OUTPUT_DIR = PROJECT_ROOT / "models" / "outputs" / "tide_training"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "clean_longest.csv"
MODEL_FILE = MODEL_DIR / "tide_e0206.pth"
SCALER_FILE = MODEL_DIR / "tide_scaler_e0206.pkl"
CONFIG_FILE = MODEL_DIR / "tide_config_e0206.json"


# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    'window_size': 96,
    'forecast_horizon': 1,
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    
    # TiDE architecture
    'input_size': 12,
    'hidden_size': 256,
    'num_encoder_layers': 2,
    'num_decoder_layers': 2,
    'dropout': 0.3,
    'temporal_decoder_hidden': 128,
    
    'batch_size': 64,
    'epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'patience': 15,
    
    'features': [
        'power_total', 'power_factor_avg',
        'current_l1', 'current_l2', 'current_l3',
        'voltage_avg', 'apparent_power_total',
        'hour', 'day_of_week', 'weekend',
        'rolling_mean_4h', 'power_delta'
    ]
}


# ============================================================================
# TiDE MODEL
# ============================================================================

class ResidualBlock(nn.Module):
    """Dense residual block"""
    def __init__(self, hidden_size, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        residual = x
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x


class DenseEncoder(nn.Module):
    """Dense encoder with residual blocks"""
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(DenseEncoder, self).__init__()
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return x


class DenseDecoder(nn.Module):
    """Dense decoder with residual blocks"""
    def __init__(self, hidden_size, output_size, num_layers, dropout):
        super(DenseDecoder, self).__init__()
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)
        return x


class TiDEForecaster(nn.Module):
    """
    TiDE: Time series Dense Encoder
    
    Architecture:
    - Feature projection
    - Dense encoder (residual blocks)
    - Temporal aggregation
    - Dense decoder
    - Output projection
    """
    def __init__(self, config):
        super(TiDEForecaster, self).__init__()
        
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.window_size = config['window_size']
        
        # Feature encoder (per timestep)
        self.feature_encoder = DenseEncoder(
            self.input_size,
            self.hidden_size,
            config['num_encoder_layers'],
            config['dropout']
        )
        
        # Temporal aggregation
        self.temporal_proj = nn.Linear(self.window_size * self.hidden_size, 
                                       config['temporal_decoder_hidden'])
        
        # Decoder
        self.decoder = DenseDecoder(
            config['temporal_decoder_hidden'],
            config['forecast_horizon'],
            config['num_decoder_layers'],
            config['dropout']
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config['dropout'])
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, features]
        Returns:
            output: [batch, 1]
        """
        batch_size = x.size(0)
        
        # Encode each timestep
        encoded = []
        for t in range(x.size(1)):
            encoded_t = self.feature_encoder(x[:, t, :])
            encoded.append(encoded_t)
        
        # Stack and flatten
        encoded = torch.stack(encoded, dim=1)  # [batch, seq_len, hidden]
        encoded_flat = encoded.reshape(batch_size, -1)  # [batch, seq_len*hidden]
        
        # Temporal aggregation
        temporal_repr = self.relu(self.temporal_proj(encoded_flat))
        temporal_repr = self.dropout(temporal_repr)
        
        # Decode
        output = self.decoder(temporal_repr)
        
        return output


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
    df = df.sort_values('time').reset_index(drop=True)
    return df


def engineer_features(df):
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['voltage_avg'] = df[['volts_l1_n', 'volts_l2_n', 'volts_l3_n']].mean(axis=1)
    df['rolling_mean_4h'] = df['power_total'].rolling(window=16, min_periods=1).mean()
    df['power_delta'] = df['power_total'].diff().fillna(0)
    return df


def create_sequences(df, config):
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
            predictions = self.model(X_batch)
            loss = criterion(predictions, y_batch)
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
        print("TRAINING TiDE MODEL")
        print("="*80)
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.validate(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            scheduler.step(val_loss)
            
            if (epoch + 1) % 5 == 0:
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
    print("TiDE POWER FORECASTING - e0206 AHU")
    print("="*80)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    df = load_and_prepare_data(DATA_FILE)
    df = engineer_features(df)
    X, y = create_sequences(df, CONFIG)
    train_data, val_data, test_data, scaler = split_scale_data(X, y, CONFIG)
    
    train_loader = DataLoader(PowerDataset(*train_data), CONFIG['batch_size'], shuffle=False)
    val_loader = DataLoader(PowerDataset(*val_data), CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(PowerDataset(*test_data), CONFIG['batch_size'], shuffle=False)
    
    model = TiDEForecaster(CONFIG)
    print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = Trainer(model, device, CONFIG)
    history = trainer.fit(train_loader, val_loader)
    
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    y_pred = trainer.predict(test_loader)
    y_test = test_data[1].flatten()
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nMetrics:")
    print(f"  MAE:  {mae:.4f} kW")
    print(f"  RMSE: {rmse:.4f} kW")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")
    
    import joblib
    joblib.dump(scaler, SCALER_FILE)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0].set_title('TiDE Training', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(y_test[:500], 'o-', label='Actual', markersize=2, linewidth=1)
    axes[1].plot(y_pred[:500], 's-', label='Predicted', markersize=2, linewidth=1)
    axes[1].set_title('TiDE Predictions', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tide_results.png', dpi=150)
    plt.close()
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nSaved: {MODEL_FILE}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()