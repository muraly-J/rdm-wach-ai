"""
TRANSFORMER MODEL ARCHITECTURE
Pure model definition - no training, no data loading

Location: src/models/transformer.py
"""

import torch
import torch.nn as nn
import numpy as np


class EnhancedPositionalEncoding(nn.Module):
    """Enhanced positional encoding with learnable components"""
    
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
        
        # Learnable position embedding
        self.learnable_pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :] + self.learnable_pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Improved Transformer for time series forecasting
    
    Args:
        config: Dict with keys:
            - input_size: Number of input features
            - d_model: Model dimension
            - nhead: Number of attention heads
            - num_encoder_layers: Number of transformer layers
            - dim_feedforward: Feedforward dimension
            - dropout: Dropout rate
    """
    
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        
        self.input_size = config['input_size']
        self.d_model = config['d_model']
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_size, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(config['dropout'])
        )
        
        # Positional encoding
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
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=config['num_encoder_layers'],
            norm=nn.LayerNorm(config['d_model'])
        )
        
        # Output layers
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
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Dual pooling
        avg_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        combined = avg_pool + max_pool
        
        output = self.output_layers(combined)
        return output