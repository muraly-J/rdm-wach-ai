"""
IMPROVED TRANSFORMER - e0206 AHU
Enhanced Architecture for Superior Forecasting Performance

Improvements:
- Extended context window: 192 timesteps (48 hours)
- Stronger positional encoding
- Hybrid loss: MSE + Huber (robust to outliers)
- Increased model capacity
- Better regularization

Expected Performance: MAE < 0.25 kW, R² > 0.65
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
OUTPUT_DIR = PROJECT_ROOT / "models" / "outputs" / "transformer_improved_training"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "clean_longest.csv"
MODEL_FILE = MODEL_DIR / "transformer_improved_e0206.pth"
SCALER_FILE = MODEL_DIR / "transformer_improved_scaler_e0206.pkl"
CONFIG_FILE = MODEL_DIR / "transformer_improved_config_e0206.json"


# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    'window_size': 192,          # INCREASED: 48 hours context
    'forecast_horizon': 1,
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    
    # Improved Transformer architecture
    'input_size': 12,
    'd_model': 256,              # INCREASED model dimension
    'nhead': 8,
    'num_encoder_layers': 6,     # INCREASED depth
    'dim_feedforward': 1024,     # INCREASED capacity
    'dropout': 0.2,
    
    # Hybrid loss weights
    'mse_weight': 0.5,
    'huber_weight': 0.5,
    'huber_delta': 1.0,
    
    'batch_size': 64,
    'epochs': 150,               # INCREASED epochs
    'learning_rate': 0.0001,
    'weight_decay': 1e-5,
    'patience': 20,              # INCREASED patience
    
    'features': [
        'power_total', 'power_factor_avg',
        'current_l1', 'current_l2', 'current_l3',
        'voltage_avg', 'apparent_power_total',
        'hour', 'day_of_week', 'weekend',
        'rolling_mean_4h', 'power_delta'
    ]
}


# ============================================================================
# ENHANCED POSITIONAL ENCODING
# ============================================================================

class EnhancedPositionalEncoding(nn.Module):
    """
    Stronger positional encoding with learnable components
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(EnhancedPositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Fixed sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        # Learnable position embedding (additive)
        self.learnable_pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        # Combine fixed and learnable positional encodings
        x = x + self.pe[:, :seq_len, :] + self.learnable_pe[:, :seq_len, :]
        return self.dropout(x)


# ============================================================================
# IMPROVED TRANSFORMER
# ============================================================================

class ImprovedTransformerForecaster(nn.Module):
    """
    Enhanced Transformer with:
    - Stronger positional encoding
    - Deeper architecture
    - Layer normalization
    - Residual connections
    """
    def __init__(self, config):
        super(ImprovedTransformerForecaster, self).__init__()
        
        self.input_size = config['input_size']
        self.d_model = config['d_model']
        
        # Input projection with layer norm
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_size, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(config['dropout'])
        )
        
        # Enhanced positional encoding
        self.pos_encoder = EnhancedPositionalEncoding(
            self.d_model,
            dropout=config['dropout']
        )
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=config['num_encoder_layers'],
            norm=nn.LayerNorm(config['d_model'])
        )
        
        # Output projection with residual
        self.output_layers = nn.Sequential(
            nn.Linear(config['d_model'], 128),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, features]
        Returns:
            output: [batch, 1]
        """
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Global average pooling + max pooling
        avg_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        combined = avg_pool + max_pool
        
        # Output projection
        output = self.output_layers(combined)
        
        return output


# ============================================================================
# HYBRID LOSS
# ============================================================================

class HybridLoss(nn.Module):
    """
    Combined MSE + Huber Loss
    - MSE: Good for normal errors
    - Huber: Robust to outliers
    """
    def __init__(self, mse_weight=0.5, huber_weight=0.5, huber_delta=1.0):
        super(HybridLoss, self).__init__()
        self.mse_weight = mse_weight
        self.huber_weight = huber_weight
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=huber_delta)
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        huber_loss = self.huber(pred, target)
        return self.mse_weight * mse_loss + self.huber_weight * huber_loss


# ============================================================================
# DATA PREPARATION
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
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    print(f"Loaded {len(df):,} samples")
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
    print(f"\nCreating sequences with window size: {config['window_size']}")
    data = df[config['features']].values
    X, y = [], []
    for i in range(config['window_size'], len(data)):
        X.append(data[i-config['window_size']:i, :])
        y.append(data[i, 0])
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    print(f"Created {len(X):,} sequences")
    return X, y


def split_data(X, y, config):
    n = len(X)
    train_end = int(n * config['train_ratio'])
    val_end = int(n * (config['train_ratio'] + config['val_ratio']))
    return (X[:train_end], y[:train_end]), (X[train_end:val_end], y[train_end:val_end]), (X[val_end:], y[val_end:])


def scale_data(train_data, val_data, test_data):
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    n_train, seq_len, n_features = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))
    
    X_train_scaled = scaler.transform(X_train.reshape(-1, n_features)).reshape(n_train, seq_len, n_features)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    
    return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test), scaler


# ============================================================================
# TRAINER
# ============================================================================

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
        print("TRAINING IMPROVED TRANSFORMER")
        print("="*80)
        
        criterion = HybridLoss(
            mse_weight=self.config['mse_weight'],
            huber_weight=self.config['huber_weight'],
            huber_delta=self.config['huber_delta']
        )
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5)
        
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
    print("IMPROVED TRANSFORMER - e0206 AHU")
    print("="*80)
    print("Enhancements:")
    print("  - Extended context: 192 timesteps (48h)")
    print("  - Enhanced positional encoding")
    print("  - Hybrid MSE + Huber loss")
    print("  - Deeper architecture (6 layers)")
    print("="*80)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    df = load_and_prepare_data(DATA_FILE)
    df = engineer_features(df)
    X, y = create_sequences(df, CONFIG)
    train_data, val_data, test_data = split_data(X, y, CONFIG)
    train_data, val_data, test_data, scaler = scale_data(train_data, val_data, test_data)
    
    train_loader = DataLoader(PowerDataset(*train_data), CONFIG['batch_size'], shuffle=False)
    val_loader = DataLoader(PowerDataset(*val_data), CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(PowerDataset(*test_data), CONFIG['batch_size'], shuffle=False)
    
    model = ImprovedTransformerForecaster(CONFIG)
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
    axes[0].set_title('Improved Transformer Training', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(y_test[:500], 'o-', label='Actual', markersize=2, linewidth=1)
    axes[1].plot(y_pred[:500], 's-', label='Predicted', markersize=2, linewidth=1)
    axes[1].set_title('Predictions', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'transformer_improved_results.png', dpi=150)
    plt.close()
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nSaved: {MODEL_FILE}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()