"""
TRAINING UTILITIES
Reusable training logic

Location: src/training/train_utils.py
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class HybridLoss(nn.Module):
    """Combined MSE + Huber Loss"""
    
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


def train_model(model, train_loader, val_loader, config, device):
    """
    Train a model with early stopping
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: Dict with training parameters:
            - epochs: Max epochs
            - learning_rate: Learning rate
            - weight_decay: L2 regularization
            - patience: Early stopping patience
            - use_hybrid_loss: Whether to use hybrid loss
        device: torch.device
    
    Returns:
        model: Trained model (loaded with best weights)
        history: Dict with 'train_loss' and 'val_loss' lists
    """
    model = model.to(device)
    
    # Loss function
    if config.get('use_hybrid_loss', False):
        criterion = HybridLoss(
            mse_weight=config.get('mse_weight', 0.5),
            huber_weight=config.get('huber_weight', 0.5),
            huber_delta=config.get('huber_delta', 1.0)
        )
    else:
        criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config.get('scheduler_patience', 5),
        factor=0.5,
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, history


def predict(model, dataloader, device):
    """
    Generate predictions from a trained model
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader
        device: torch.device
    
    Returns:
        predictions: np.array of predictions
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            predictions.extend(preds.cpu().numpy())
    
    return torch.tensor(predictions).numpy().flatten()