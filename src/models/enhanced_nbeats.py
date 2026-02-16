"""
ENHANCED N-BEATS MODEL ARCHITECTURE
Pure model definition - no training, no data loading

Location: src/models/enhanced_nbeats.py
"""

import torch
import torch.nn as nn
import numpy as np


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
    """Trend block with polynomial basis"""
    
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
    """Seasonality block with Fourier basis"""
    
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


class EnhancedNBeatsModel(nn.Module):
    """
    Enhanced N-BEATS with seasonal decomposition
    
    Args:
        config: Dict with keys:
            - input_size: Number of input features
            - window_size: Lookback window
            - num_stacks: Number of stacks (5)
            - num_blocks_per_stack: Blocks per stack
            - hidden_size: Hidden dimension
            - num_layers: FC layers per block
            - dropout: Dropout rate
            - share_weights: Weight sharing flag
            - polynomial_degree: Polynomial degree for trend
            - num_harmonics: Fourier harmonics for seasonality
    """
    
    def __init__(self, config):
        super(EnhancedNBeatsModel, self).__init__()
        
        self.input_size = config['input_size'] * config['window_size']
        self.share_weights = config['share_weights']
        
        # Generic stacks (2)
        if self.share_weights:
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
        
        # Trend stacks (2)
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
        
        # Seasonal stack (1)
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
        """
        Args:
            x: [batch, seq_len, features]
        Returns:
            forecast: [batch, 1]
        """
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