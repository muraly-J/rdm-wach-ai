"""
DATASET AND PREPROCESSING UTILITIES
Reusable across training and backtesting

Location: src/training/dataset.py
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


def engineer_features(df):
    """
    Add engineered features to dataframe
    
    Args:
        df: DataFrame with columns: time, power_total, current_l1/l2/l3,
            volts_l1_n/l2_n/l3_n
    
    Returns:
        df: DataFrame with added features
    """
    # Time features
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Voltage average
    df['voltage_avg'] = df[['volts_l1_n', 'volts_l2_n', 'volts_l3_n']].mean(axis=1)
    
    # Rolling features
    df['rolling_mean_4h'] = df['power_total'].rolling(window=16, min_periods=1).mean()
    df['power_delta'] = df['power_total'].diff().fillna(0)
    
    return df


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


def get_default_features():
    """Return default feature list"""
    return [
        'power_total',
        'power_factor_avg',
        'current_l1',
        'current_l2',
        'current_l3',
        'voltage_avg',
        'apparent_power_total',
        'hour',
        'day_of_week',
        'weekend',
        'rolling_mean_4h',
        'power_delta'
    ]