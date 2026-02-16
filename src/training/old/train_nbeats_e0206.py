"""
N-BEATS TRAINING SCRIPT — e0206 Cafeteria AHU
Offline 96-Step Forecast Demonstration

This script implements the N-BEATS architecture as specified in the XML diagram
and produces a 96-step rolling forecast for model validation.

Architecture:
- 1 Generic Stack
- 3 Blocks per stack
- 4 FC layers per block (128 units, ReLU, Dropout 0.1)
- Backcast length: 96 timesteps
- Forecast length: 1 timestep
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. Using NumPy-based implementation.")

sns.set_style("whitegrid")


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Get the project root directory (assuming script is in src/training/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Define paths
DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
OUTPUT_DIR = PROJECT_ROOT / "models" / "outputs" / "nbeats_training"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Data file
DATA_FILE = DATA_DIR / "clean_longest.parquet"


# ============================================================================
# N-BEATS MODEL ARCHITECTURE
# ============================================================================

class NBeatsBlock(nn.Module):
    """
    Single N-BEATS block (Generic)
    
    Architecture from XML:
    - FC Layer 1: [1152 → 128] + ReLU
    - FC Layer 2: [128 → 128] + ReLU + Dropout(0.1)
    - FC Layer 3: [128 → 128] + ReLU
    - FC Layer 4: [128 → 128] + ReLU
    - Backcast: [128 → 1152]
    - Forecast: [128 → 1]
    """
    
    def __init__(self, input_size=1152, hidden_units=128, dropout=0.1):
        super(NBeatsBlock, self).__init__()
        
        self.input_size = input_size
        self.hidden_units = hidden_units
        
        # Fully connected layers (4 layers as per spec)
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.fc4 = nn.Linear(hidden_units, hidden_units)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
        self.backcast_linear = nn.Linear(hidden_units, input_size)
        self.forecast_linear = nn.Linear(hidden_units, 1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, input_size]
        
        Returns:
            backcast: Reconstruction [batch_size, input_size]
            forecast: Prediction [batch_size, 1]
        """
        # Layer 1
        h = self.relu(self.fc1(x))
        
        # Layer 2 (with dropout)
        h = self.relu(self.fc2(h))
        h = self.dropout(h)
        
        # Layer 3
        h = self.relu(self.fc3(h))
        
        # Layer 4
        h = self.relu(self.fc4(h))
        
        # Outputs
        backcast = self.backcast_linear(h)
        forecast = self.forecast_linear(h)
        
        return backcast, forecast


class NBeatsStack(nn.Module):
    """
    N-BEATS Stack containing multiple blocks
    
    From specs:
    - 3 blocks
    - Generic stack (no trend/seasonality decomposition)
    - Residual connections between blocks
    """
    
    def __init__(self, num_blocks=3, input_size=1152, hidden_units=128, dropout=0.1):
        super(NBeatsStack, self).__init__()
        
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, hidden_units, dropout) 
            for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        """
        Forward pass through stack with residual connections
        
        Args:
            x: Input tensor [batch_size, input_size]
        
        Returns:
            forecast: Sum of all block forecasts [batch_size, 1]
        """
        residual = x
        forecast_sum = 0
        
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast  # Residual connection
            forecast_sum = forecast_sum + forecast  # Accumulate forecasts
        
        return forecast_sum


class NBeatsFull(nn.Module):
    """
    Complete N-BEATS model for e0206
    
    Single generic stack with 3 blocks
    """
    
    def __init__(self, num_blocks=3, input_size=1152, hidden_units=128, dropout=0.1):
        super(NBeatsFull, self).__init__()
        
        self.stack = NBeatsStack(num_blocks, input_size, hidden_units, dropout)
    
    def forward(self, x):
        return self.stack(x)


# ============================================================================
# DATA PREPARATION
# ============================================================================

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series windowing"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class E0206DataPreparation:
    """Data preparation for N-BEATS training"""
    
    def __init__(self, window_size=96):
        self.window_size = window_size
        self.scaler = StandardScaler()
        
        # Feature columns
        self.target = 'power_total'  # Was 'kw'
        self.electrical_features = [
            'power_l1', 'power_l2', 'power_l3',     # Was 'p_l1', 'p_l2', 'p_l3'
            'current_l1', 'current_l2', 'current_l3', # Was 'i_l1', 'i_l2', 'i_l3'
            'volts_l1_n', 'volts_l2_n', 'volts_l3_n', # Was 'v_l1', 'v_l2', 'v_l3'
            'power_factor_avg',                       # Was 'pf'
            'reactive_power_total'                    # Was 'kvar_tot'
        ]
        # Note: If you have hour_of_day, etc., ensure they are in your dataframe 
        # or remove them from self.time_features
        self.time_features = [] # Leave empty if not yet engineered in the parquet
        
        # Total: 1 target history + 11 electrical + 3 time = 15 features per timestep
        # But we only use target history (kw) + 11 electrical = 12 features
        # 12 features × 96 timesteps = 1,152 input dimensions
    
    def create_supervised_windows(self, df):
        """
        Create supervised learning windows
        
        Input: 96 timesteps × 12 features = 1,152 values
        Output: 1 timestep ahead (kw prediction)
        
        Args:
            df: DataFrame with cleaned data
        
        Returns:
            X: Input windows [samples, 1152]
            y: Target values [samples, 1]
        """
        print("\nCreating supervised windows...")
        
        # Select features (kw history + electrical features)
        feature_cols = [self.target] + self.electrical_features
        data = df[feature_cols].values
        
        # Add time features separately (they don't need history)
        time_data = df[self.time_features].values
        
        X_windows = []
        y_targets = []
        time_contexts = []
        
        # Create windows
        for i in range(self.window_size, len(data)):
            # Input: historical window of kw + electrical features
            window = data[i - self.window_size:i, :].flatten()  # 96 × 12 = 1,152
            
            # Target: next kw value
            target = data[i, 0]  # kw is first column
            
            # Time context at prediction time
            time_ctx = time_data[i, :]
            
            X_windows.append(window)
            y_targets.append(target)
            time_contexts.append(time_ctx)
        
        X = np.array(X_windows)
        y = np.array(y_targets).reshape(-1, 1)
        
        print(f"   Created {len(X)} windows")
        print(f"   Input shape: {X.shape}")
        print(f"   Output shape: {y.shape}")
        
        return X, y
    
    def split_data(self, X, y, df):
        """
        Time-based split (NO SHUFFLING)
        
        From specs:
        - Train: Days 1-10 (960 intervals = first 67%)
        - Validation: Days 11-13 (288 intervals = next 20%)
        - Test: Days 14-15 (192 intervals = last 13%)
        
        But we need to account for the 96-step window offset
        """
        print("\nSplitting data (time-ordered)...")
        
        total_samples = len(X)
        
        # Account for window offset
        # Total clean rows ≈ 1,357
        # After windowing: 1,357 - 96 = 1,261 samples
        
        # Split points (approximate)
        train_end = int(total_samples * 0.67)  # ~845 samples
        val_end = int(total_samples * 0.87)    # ~1,096 samples
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        print(f"   Train: {len(X_train)} samples ({len(X_train)/total_samples*100:.1f}%)")
        print(f"   Val:   {len(X_val)} samples ({len(X_val)/total_samples*100:.1f}%)")
        print(f"   Test:  {len(X_test)} samples ({len(X_test)/total_samples*100:.1f}%)")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def scale_data(self, train_data, val_data, test_data):
        """
        Scale features using StandardScaler fitted ONLY on training data
        
        Critical: No data leakage from val/test into scaler
        """
        print("\nScaling features...")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # Fit scaler on TRAIN only
        self.scaler.fit(X_train)
        
        # Transform all sets
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"   Scaler fitted on train set only")
        print(f"   Mean: {self.scaler.mean_[:5]} ... (first 5)")
        print(f"   Std:  {self.scaler.scale_[:5]} ... (first 5)")
        
        return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test)


# ============================================================================
# BASELINE MODELS
# ============================================================================

def compute_baseline_persistence(y_test, X_test_original, window_size=96):
    """
    Baseline 1: Persistence
    Prediction: kw(t) = kw(t-1)
    
    Args:
        y_test: Actual values
        X_test_original: Original (unscaled) test windows
        window_size: Window size (96)
    
    Returns:
        predictions: Persistence predictions
    """
    # Last value from each window is kw(t-1)
    # Window structure: [kw(t-96), kw(t-95), ..., kw(t-1), other features...]
    # kw(t-1) is at position window_size-1
    
    predictions = X_test_original[:, window_size - 1]  # Last kw value in window
    return predictions


def compute_baseline_rolling_mean(X_test_original, window_size=96):
    """
    Baseline 2: Rolling Mean
    Prediction: kw(t) = mean(kw(t-4:t-1))  # Last 4 steps = 1 hour
    
    Args:
        X_test_original: Original (unscaled) test windows
        window_size: Window size (96)
    
    Returns:
        predictions: Rolling mean predictions
    """
    predictions = []
    
    for window in X_test_original:
        # Extract last 4 kw values from window
        # kw values are at positions: 0, 12, 24, ..., 96*12-12
        # We want the last 4: positions (92*12, 93*12, 94*12, 95*12)
        last_4_kw = [window[i * 12] for i in range(92, 96)]
        pred = np.mean(last_4_kw)
        predictions.append(pred)
    
    return np.array(predictions)


def evaluate_model(y_true, y_pred, model_name="Model"):
    """Calculate evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2
    }


# ============================================================================
# TRAINING
# ============================================================================

class NBEATSTrainer:
    """Training manager for N-BEATS model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            predictions = self.model(X_batch)
            loss = criterion(predictions, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(dataloader)
    
    def validate(self, dataloader, criterion):
        """Validate model"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                
                val_loss += loss.item()
        
        return val_loss / len(dataloader)
    
    def fit(self, train_loader, val_loader, epochs=100, lr=1e-3, patience=10):
        """
        Full training loop with early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum epochs
            lr: Learning rate
            patience: Early stopping patience
        """
        print("\n" + "="*80)
        print("TRAINING N-BEATS MODEL")
        print("="*80)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Define model save path
        model_save_path = MODEL_DIR / 'nbeats_best_e0206.pth'
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss = self.validate(val_loader, criterion)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), model_save_path)
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"   Best validation loss: {best_val_loss:.6f}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load(model_save_path))
        
        print("\nTraining complete!")
        print(f"   Best validation loss: {best_val_loss:.6f}")
        
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
# VISUALIZATION
# ============================================================================

def plot_training_history(history):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('N-BEATS Training History - e0206', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'e0206_training_history.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"   Training history saved to: {output_path}")


def plot_96step_forecast(y_actual, y_nbeats, y_persist, y_rolling):
    """
    Plot 96-step offline forecast comparison
    
    This is the REQUIRED outcome artifact from the spec
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Take final 96 steps from test set
    steps = min(96, len(y_actual))
    x_axis = range(steps)
    
    actual = y_actual[-steps:]
    nbeats = y_nbeats[-steps:]
    persist = y_persist[-steps:]
    rolling = y_rolling[-steps:]
    
    # Plot 1: All models
    ax1 = axes[0]
    ax1.plot(x_axis, actual, 'o-', label='Actual', linewidth=2, markersize=4, color='black', alpha=0.8)
    ax1.plot(x_axis, nbeats, 's-', label='N-BEATS', linewidth=2, markersize=3, color='red', alpha=0.7)
    ax1.plot(x_axis, persist, '^--', label='Persistence', linewidth=1.5, markersize=3, color='blue', alpha=0.6)
    ax1.plot(x_axis, rolling, 'v--', label='Rolling Mean', linewidth=1.5, markersize=3, color='green', alpha=0.6)
    
    ax1.set_xlabel('Timestep (15-min intervals)', fontsize=11)
    ax1.set_ylabel('Power (kW)', fontsize=11)
    ax1.set_title('Offline 96-Step Forecast - e0206 Cafeteria AHU', fontweight='bold', fontsize=13)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Errors
    ax2 = axes[1]
    nbeats_error = np.abs(actual - nbeats)
    persist_error = np.abs(actual - persist)
    rolling_error = np.abs(actual - rolling)
    
    ax2.plot(x_axis, nbeats_error, 's-', label='N-BEATS Error', linewidth=2, markersize=3, color='red', alpha=0.7)
    ax2.plot(x_axis, persist_error, '^--', label='Persistence Error', linewidth=1.5, markersize=3, color='blue', alpha=0.6)
    ax2.plot(x_axis, rolling_error, 'v--', label='Rolling Mean Error', linewidth=1.5, markersize=3, color='green', alpha=0.6)
    
    ax2.set_xlabel('Timestep (15-min intervals)', fontsize=11)
    ax2.set_ylabel('Absolute Error (kW)', fontsize=11)
    ax2.set_title('Prediction Errors', fontweight='bold', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'e0206_96step_forecast.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n96-step forecast plot saved to: {output_path}")


def print_comparison_table(results):
    """Print formatted comparison table"""
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    df_results = pd.DataFrame(results)
    
    print("\n" + df_results.to_string(index=False))
    
    # Calculate improvements
    print("\n" + "="*80)
    print("N-BEATS vs BASELINES")
    print("="*80)
    
    nbeats_mae = df_results[df_results['model'] == 'N-BEATS']['MAE'].values[0]
    persist_mae = df_results[df_results['model'] == 'Persistence']['MAE'].values[0]
    rolling_mae = df_results[df_results['model'] == 'Rolling Mean']['MAE'].values[0]
    
    improve_vs_persist = ((persist_mae - nbeats_mae) / persist_mae) * 100
    improve_vs_rolling = ((rolling_mae - nbeats_mae) / rolling_mae) * 100
    
    print(f"\nImprovement vs Persistence: {improve_vs_persist:+.1f}%")
    print(f"Improvement vs Rolling Mean: {improve_vs_rolling:+.1f}%")
    
    # Verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    
    if improve_vs_persist >= 20:
        print("EXCELLENT: N-BEATS beats persistence by >=20%")
        print("   -> Forecasting adds significant value")
    elif improve_vs_persist >= 10:
        print("GOOD: N-BEATS beats persistence by 10-20%")
        print("   -> Forecasting adds marginal value")
    elif improve_vs_persist >= 0:
        print("MARGINAL: N-BEATS beats persistence by <10%")
        print("   -> Persistence likely sufficient for this device")
    else:
        print("WORSE: N-BEATS does not beat persistence")
        print("   -> Use persistence baseline for this device")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("N-BEATS BASELINE TRAINING - e0206")
    print("="*80)
    print("\nArchitecture:")
    print("  - 1 Generic Stack")
    print("  - 3 Blocks")
    print("  - 4 FC Layers per block (128 units)")
    print("  - Input: 96 timesteps x 12 features = 1,152 dims")
    print("  - Output: 1-step ahead prediction")
    print("="*80)
    
    # Check PyTorch
    if not PYTORCH_AVAILABLE:
        print("\nERROR: PyTorch is required for this script")
        print("   Install with: pip install torch")
        return
    
    # Check for GPU availability (CUDA for NVIDIA, MPS for Apple Silicon)
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"\nUsing device: {device}")
        print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print(f"\nUsing device: {device} (Apple Metal Performance Shaders)")
        print(f"   GPU: Apple Silicon")
    else:
        print("\nERROR: No GPU available")
        print("   This script requires GPU for training")
        print("   Supported: CUDA (NVIDIA) or MPS (Apple Silicon)")
        return
    
    # Load cleaned data
    print(f"\nLoading cleaned dataset from: {DATA_FILE}")
    try:
        df = pd.read_parquet(DATA_FILE)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"   Loaded {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {DATA_FILE}")
        print("   Please ensure the cleaned_longest.parquet file exists in data/cleaned/")
        return
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return
    
    # Prepare data
    prep = E0206DataPreparation(window_size=96)
    X, y = prep.create_supervised_windows(df)
    
    # Split data
    train_data, val_data, test_data = prep.split_data(X, y, df)
    
    # Store original test data for baselines
    X_test_original = test_data[0].copy()
    y_test = test_data[1]
    
    # Scale data
    train_data, val_data, test_data = prep.scale_data(train_data, val_data, test_data)
    
    # Create PyTorch datasets
    train_dataset = TimeSeriesDataset(train_data[0], train_data[1])
    val_dataset = TimeSeriesDataset(val_data[0], val_data[1])
    test_dataset = TimeSeriesDataset(test_data[0], test_data[1])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # No shuffling!
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    print("\nBuilding N-BEATS model...")
    model = NBeatsFull(
        num_blocks=3,
        input_size=1152,
        hidden_units=128,
        dropout=0.1
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Train model on GPU
    trainer = NBEATSTrainer(model, device=device)
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        lr=1e-3,
        patience=10
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Generate predictions
    print("\nGenerating predictions...")
    y_nbeats_pred = trainer.predict(test_loader)
    
    # Compute baselines
    print("\nComputing baseline models...")
    y_persist_pred = compute_baseline_persistence(y_test, X_test_original, window_size=96)
    y_rolling_pred = compute_baseline_rolling_mean(X_test_original, window_size=96)
    
    # Evaluate all models
    results = []
    
    results.append(evaluate_model(y_test, y_nbeats_pred, "N-BEATS"))
    results.append(evaluate_model(y_test, y_persist_pred, "Persistence"))
    results.append(evaluate_model(y_test, y_rolling_pred, "Rolling Mean"))
    
    # Print comparison
    print_comparison_table(results)
    
    # Plot 96-step forecast
    print("\nCreating 96-step forecast visualization...")
    plot_96step_forecast(
        y_test.flatten(),
        y_nbeats_pred,
        y_persist_pred,
        y_rolling_pred
    )
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / 'e0206_model_comparison.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "="*80)
    print("TRAINING AND EVALUATION COMPLETE")
    print("="*80)
    print("\nDeliverables:")
    print(f"  1. Trained N-BEATS model ({MODEL_DIR / 'nbeats_best_e0206.pth'})")
    print(f"  2. 96-step forecast plot ({OUTPUT_DIR / 'e0206_96step_forecast.png'})")
    print(f"  3. Training history ({OUTPUT_DIR / 'e0206_training_history.png'})")
    print(f"  4. Performance comparison ({OUTPUT_DIR / 'e0206_model_comparison.csv'})")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()