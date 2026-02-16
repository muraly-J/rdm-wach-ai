"""
ENHANCED DATASET WITH AUTOREGRESSIVE FEATURES
Adds lag features and rolling statistics for improved stability

Key additions:
- Lag features: t-1, t-4, t-96, t-672
- Rolling statistics: 24h mean/std, 7-day mean
- Difference features: short-term and long-term changes

Location: src/training/dataset_enhanced.py
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PowerDataset(Dataset):
    """PyTorch Dataset for power forecasting"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def engineer_features_enhanced(df):
    """
    Enhanced feature engineering with autoregressive components
    
    CRITICAL FOR STABILITY: Adding lag and rolling features
    dramatically improves temporal robustness
    
    Args:
        df: DataFrame with columns: time, power_total, current_l1/l2/l3,
            volts_l1_n/l2_n/l3_n
    
    Returns:
        df: DataFrame with comprehensive feature set
    """
    print("\n" + "="*80)
    print("ENHANCED FEATURE ENGINEERING")
    print("="*80)
    print("Adding autoregressive features for improved stability...")
    
    # Time features
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Voltage average
    df['voltage_avg'] = df[['volts_l1_n', 'volts_l2_n', 'volts_l3_n']].mean(axis=1)
    
    # LAG FEATURES (CRITICAL)
    print("  Adding lag features...")
    df['power_lag_1'] = df['power_total'].shift(1)        # 15 min ago
    df['power_lag_4'] = df['power_total'].shift(4)        # 1 hour ago
    df['power_lag_96'] = df['power_total'].shift(96)      # 24 hours ago
    df['power_lag_672'] = df['power_total'].shift(672)    # 1 week ago
    
    # ROLLING STATISTICS (CRITICAL)
    print("  Adding rolling statistics...")
    df['rolling_mean_1h'] = df['power_total'].rolling(window=4, min_periods=1).mean()
    df['rolling_std_1h'] = df['power_total'].rolling(window=4, min_periods=1).std()
    df['rolling_mean_24h'] = df['power_total'].rolling(window=96, min_periods=1).mean()
    df['rolling_std_24h'] = df['power_total'].rolling(window=96, min_periods=1).std()
    df['rolling_mean_7d'] = df['power_total'].rolling(window=672, min_periods=1).mean()
    
    # DIFFERENCE FEATURES (CRITICAL)
    print("  Adding difference features...")
    df['power_delta_1'] = df['power_total'].diff(1)       # Change from last period
    df['power_delta_96'] = df['power_total'].diff(96)     # Change from 24h ago
    df['power_delta_672'] = df['power_total'].diff(672)   # Change from 1 week ago
    
    # RATIO FEATURES
    print("  Adding ratio features...")
    df['power_to_rolling_mean_24h'] = df['power_total'] / (df['rolling_mean_24h'] + 1e-6)
    df['current_volatility_1h'] = df['rolling_std_1h'] / (df['rolling_mean_1h'] + 1e-6)
    
    # Fill NaN values created by lag/rolling operations
    # Use forward fill then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    print(f"\nTotal features created: {len(df.columns) - 2}")  # -2 for time and power_total
    print("  Time features: 3")
    print("  Voltage features: 1")
    print("  Lag features: 4 (CRITICAL for stability)")
    print("  Rolling features: 5 (CRITICAL for adaptation)")
    print("  Difference features: 3 (CRITICAL for trend)")
    print("  Ratio features: 2")
    
    return df


def get_enhanced_features():
    """
    Return comprehensive feature list for enhanced modeling
    
    IMPORTANT: This list is MUCH more robust than the basic version
    """
    return [
        # Target and basic
        'power_total',
        'power_factor_avg',
        'current_l1',
        'current_l2',
        'current_l3',
        'voltage_avg',
        'apparent_power_total',
        
        # Time features
        'hour',
        'day_of_week',
        'weekend',
        
        # LAG FEATURES (Key for stability)
        'power_lag_1',
        'power_lag_4',
        'power_lag_96',
        'power_lag_672',
        
        # ROLLING STATISTICS (Key for adaptation)
        'rolling_mean_1h',
        'rolling_std_1h',
        'rolling_mean_24h',
        'rolling_std_24h',
        'rolling_mean_7d',
        
        # DIFFERENCE FEATURES (Key for trend)
        'power_delta_1',
        'power_delta_96',
        'power_delta_672',
        
        # RATIO FEATURES (Key for normalization)
        'power_to_rolling_mean_24h',
        'current_volatility_1h'
    ]


def create_sequences(df, window_size, features):
    """
    Create sequences for time series forecasting
    
    Args:
        df: DataFrame with engineered features
        window_size: Number of timesteps to look back
        features: List of feature column names
    
    Returns:
        X: np.array of shape [n_samples, window_size, n_features]
        y: np.array of shape [n_samples, 1]
        timestamps: List of timestamps for each sequence
    """
    data = df[features].values
    X, y, timestamps = [], [], []
    
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, :])
        y.append(data[i, 0])  # power_total is first feature
        timestamps.append(df['time'].iloc[i])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    return X, y, timestamps


def compare_feature_sets(df, window_size=192):
    """
    Compare basic vs enhanced feature sets
    
    Shows the difference in feature richness
    """
    print("\n" + "="*80)
    print("FEATURE SET COMPARISON")
    print("="*80)
    
    basic_features = [
        'power_total', 'power_factor_avg',
        'current_l1', 'current_l2', 'current_l3',
        'voltage_avg', 'apparent_power_total',
        'hour', 'day_of_week', 'weekend',
        'rolling_mean_1h', 'power_delta_1'
    ]
    
    enhanced_features = get_enhanced_features()
    
    print(f"\nBASIC FEATURE SET:")
    print(f"  Total features: {len(basic_features)}")
    print(f"  Input dimension: {window_size} x {len(basic_features)} = {window_size * len(basic_features):,}")
    print(f"  Autoregressive signals: WEAK (only 1h rolling mean)")
    
    print(f"\nENHANCED FEATURE SET:")
    print(f"  Total features: {len(enhanced_features)}")
    print(f"  Input dimension: {window_size} x {len(enhanced_features)} = {window_size * len(enhanced_features):,}")
    print(f"  Autoregressive signals: STRONG")
    print(f"    - 4 lag features (15min, 1h, 24h, 7d)")
    print(f"    - 5 rolling statistics (adaptive to regime)")
    print(f"    - 3 difference features (trend capture)")
    print(f"    - 2 ratio features (normalization)")
    
    print(f"\nEXPECTED IMPROVEMENT:")
    print(f"  Stability (Std R²): MAJOR reduction")
    print(f"  Robustness (Worst R²): MAJOR improvement")
    print(f"  Regime adaptation: MAJOR enhancement")
    
    print(f"\nWHY THIS MATTERS:")
    print(f"  - Lag features provide direct autoregressive signal")
    print(f"  - Rolling stats adapt to changing baselines")
    print(f"  - Difference features capture trend direction")
    print(f"  - Result: Model sees 'context' not just 'levels'")


# Backward compatibility
def engineer_features(df):
    """Wrapper for backward compatibility"""
    return engineer_features_enhanced(df)


def get_default_features():
    """Wrapper for backward compatibility"""
    return get_enhanced_features()