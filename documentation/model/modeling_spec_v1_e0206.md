# N-BEATS Modeling Specification v1 — e0206 (Cafeteria AHU)

## Document Purpose

This specification defines the complete supervised learning setup for forecasting power consumption using N-BEATS on the e0206 Cafeteria AHU device.

**Binding for:** Model implementation, training pipeline, evaluation framework

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Device** | e0206 (Cafeteria AHU) |
| **Model Type** | N-BEATS (Neural Basis Expansion Analysis) |
| **Task** | Time series forecasting (regression) |
| **Status** | Ready for implementation |
| **Created** | 2026-02-12 |

---

## 1. Modeling Objective

### 1.1 Primary Goal

**Predict:** Total active power consumption (`kw`) at time `t`  
**Using:** Historical power data and electrical features from time `t-96` to `t-1`  
**Horizon:** Single-step ahead (15 minutes)

### 1.2 Success Criteria

| Tier | MAE Threshold | RMSE Threshold | MAPE Threshold | Status |
|------|---------------|----------------|----------------|--------|
| **Minimum Viable** | < 0.240 kW | < 0.350 kW | < 7.0% | 10% better than persistence |
| **Target** | < 0.215 kW | < 0.320 kW | < 6.0% | 20% better than persistence |
| **Excellent** | < 0.190 kW | < 0.290 kW | < 5.5% | 30% better than persistence |

**Baseline to beat:** Persistence model (MAE = 0.268 kW, MAPE = 7.53%)

### 1.3 Secondary Objectives

- Validate data extraction → model → evaluation pipeline
- Benchmark N-BEATS on low-complexity signal
- Identify minimum architecture for stable devices
- Establish feature importance baseline

---

## 2. Data Specification

### 2.1 Source Data

| Parameter | Value |
|-----------|-------|
| **Device ID** | e0206 |
| **Bucket** | default |
| **Measurement pattern** | `wach_e0206_*` |
| **Date range** | 2026-01-28 00:00 to 2026-02-11 23:45 |
| **Total duration** | 15 days |
| **Sampling frequency** | 15 minutes |
| **Total samples** | 1,440 |

### 2.2 Data Processing Pipeline

**Input:** Raw InfluxDB measurements  
**Processing steps:**

1. Extract using validated spec (`data_cleaning_spec_v1_e0206_validated.md`)
2. Apply column renaming (InfluxDB → model variables)
3. Remove flatline periods (83 intervals)
4. Apply range validation thresholds
5. Add temporal features
6. Standardize features (fit on train only)

**Output:** Clean tabular dataset

**Expected gold dataset size:** ~1,357 rows (94.2% of raw)

### 2.3 Train/Validation/Test Split

**Strategy:** Strict time-ordered split (NO random shuffle)

| Split | Duration | Intervals | Date Range | Purpose |
|-------|----------|-----------|------------|---------|
| **Train** | 9 days | 864 | 2026-01-28 to 2026-02-06 | Model fitting |
| **Validation** | 2 days | 192 | 2026-02-06 to 2026-02-08 | Hyperparameter tuning |
| **Test** | 3 days | 288 | 2026-02-08 to 2026-02-11 | Final evaluation |

**Rationale:**
- Time-ordered preserves temporal causality
- 64% / 13% / 23% split
- Validation set for early stopping
- Test set held out completely

**After flatline removal:**
- Train: ~811 intervals
- Val: ~180 intervals  
- Test: ~270 intervals

---

## 3. Feature Engineering

### 3.1 Target Variable

| Variable | Description | Source | Type | Nullable |
|----------|-------------|--------|------|----------|
| `kw` | Total active power | `power_total` | float | No |

**Unit:** kW  
**Range:** [2.77, 4.94] (observed)  
**Missing handling:** Drop row if target is NaN

### 3.2 Raw Electrical Features

| Feature | Description | InfluxDB Source | Unit | Correlation with kw |
|---------|-------------|-----------------|------|---------------------|
| `p_l1` | Active power phase 1 | `power_l1` | kW | 0.932 *** |
| `p_l2` | Active power phase 2 | `power_l2` | kW | 0.280 |
| `p_l3` | Active power phase 3 | `power_l3` | kW | 0.796 ** |
| `i_l1` | Current phase 1 | `current_l1` | A | 0.922 *** |
| `i_l2` | Current phase 2 | `current_l2` | A | 0.207 |
| `i_l3` | Current phase 3 | `current_l3` | A | 0.799 ** |
| `v_l1` | Voltage phase 1 | `volts_l1_n` | V | Low |
| `v_l2` | Voltage phase 2 | `volts_l2_n` | V | Low |
| `v_l3` | Voltage phase 3 | `volts_l3_n` | V | Low |
| `pf` | Power factor | `power_factor_avg` | - | 0.957 *** |
| `kvar_tot` | Apparent power | `apparent_power_total` | kVA | 0.991 *** |

**Total:** 11 electrical features

**Feature Selection Note:**
- `kvar_tot` has extreme correlation (0.991) → monitor for redundancy
- Voltage features have low correlation → may be excluded in ablation study
- L2 metrics (power, current) have lowest correlation due to phase stability

### 3.3 Temporal Features (Derived)

| Feature | Description | Values | Type | Rationale |
|---------|-------------|--------|------|-----------|
| `hour` | Hour of day | 0-23 | int | Weak diurnal pattern (7.9% variation) |
| `day_of_week` | Day of week | 0-6 | int | Weak weekly pattern (13.8% variation) |
| `is_weekend` | Weekend flag | 0, 1 | int | Potential occupancy proxy |

**Encoding:** Use as-is (ordinal for hour/dow, binary for weekend)  
**Alternative:** Cyclical encoding (sin/cos) if patterns emerge

**Expected impact:** LOW (patterns are weak)

### 3.4 Feature Window

**For each prediction at time `t`:**

```
Input features:
  - kw[t-96 : t-1]         (96 historical target values)
  - p_l1[t-96 : t-1]       (96 values)
  - p_l2[t-96 : t-1]       (96 values)
  - ... (all 11 electrical features over 96 timesteps)
  - hour[t]                (1 value, current hour)
  - day_of_week[t]         (1 value)
  - is_weekend[t]          (1 value)

Output:
  - kw[t]                  (single value to predict)
```

**Total input dimension per sample:**
- Historical window: 96 steps × 12 features (target + 11 electrical) = 1,152 values
- Temporal context: 3 features = 3 values
- **Total:** 1,155 input features

### 3.5 Preprocessing

**Standardization (Z-score normalization):**

```python
from sklearn.preprocessing import StandardScaler

# Fit scaler on TRAIN set only
scaler = StandardScaler()
scaler.fit(X_train)

# Transform all sets
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save scaler for inference
joblib.dump(scaler, 'scaler_e0206.pkl')
```

**Applied to:** All electrical features and target  
**NOT applied to:** Temporal features (already bounded)

**Rationale:**
- Neural networks benefit from normalized inputs
- Prevents features with large magnitude (voltage ~235V) from dominating
- Standardized residuals easier to interpret

---

## 4. N-BEATS Architecture

### 4.1 Architecture Philosophy

**For e0206 (low-complexity signal):**
- Use **generic blocks only** (no interpretable seasonal/trend stacks)
- Minimal depth (3 blocks)
- Moderate width (128 units)
- Focus on learning residuals efficiently

**Rationale:**
- Signal has weak seasonality (7.9% hourly variation)
- High autocorrelation (0.751) favors simple patterns
- Avoid over-parameterization on small dataset

### 4.2 Model Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Number of stacks** | 1 | Generic stack only |
| **Blocks per stack** | 3 | Minimal for multi-scale learning |
| **FC layers per block** | 4 | Standard N-BEATS depth |
| **Hidden units** | 128 | Moderate capacity |
| **Input window** | 96 steps | 24 hours lookback |
| **Output horizon** | 1 step | 15 minutes ahead |
| **Activation** | ReLU | Standard, works well |
| **Batch normalization** | Optional | Test both |
| **Dropout** | 0.1 | Light regularization |

### 4.3 Detailed Layer Specifications

**Stack 1: Generic Blocks (Residual)**

```
Block 1:
  Input: [batch, 96, num_features]
  FC1: [96 * num_features] → 128, ReLU
  FC2: 128 → 128, ReLU, Dropout(0.1)
  FC3: 128 → 128, ReLU
  FC4: 128 → 128, ReLU
  
  Backcast: 128 → 96 (reconstructs input window)
  Forecast: 128 → 1 (predicts next step)
  
  Output: residual = input - backcast

Block 2:
  Input: residual from Block 1
  [Same architecture as Block 1]
  Output: residual from Block 2

Block 3:
  Input: residual from Block 2
  [Same architecture as Block 1]
  Output: final residual
  
Final Forecast:
  Sum of all block forecasts
```

**Total trainable parameters:** ~150K (lightweight)

### 4.4 Alternative Architectures (Ablation Studies)

| Variant | Modification | When to Use |
|---------|--------------|-------------|
| **Shallow** | 2 blocks instead of 3 | If 3-block overfits |
| **Deep** | 5 blocks | If underfitting (unlikely) |
| **Wide** | 256 units | If capacity limited |
| **No temporal features** | Drop hour/dow/weekend | If they don't help |
| **Simplified features** | Only kw, p_l*, i_l* | Reduce redundancy |

---

## 5. Training Specification

### 5.1 Loss Function

**Primary:** Mean Squared Error (MSE)

```python
loss = torch.nn.MSELoss()
```

**Why MSE:**
- Penalizes large errors heavily
- Standard for regression
- Differentiable everywhere

**Alternative:** MAE (L1 loss)
- More robust to outliers
- Test if MSE leads to instability

### 5.2 Optimization

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Optimizer** | Adam | Adaptive learning rate |
| **Learning rate** | 1e-3 | Standard starting point |
| **Weight decay** | 1e-5 | Light L2 regularization |
| **LR scheduler** | ReduceLROnPlateau | Reduce on val loss plateau |
| **Patience (scheduler)** | 10 epochs | Wait before reducing LR |
| **LR reduction factor** | 0.5 | Halve LR when plateau detected |
| **Min LR** | 1e-6 | Stop reducing beyond this |

### 5.3 Training Loop

```python
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=1e-3, 
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=10
)

for epoch in range(max_epochs):
    # Training
    model.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        predictions = model(batch['X'])
        loss = criterion(predictions, batch['y'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            predictions = model(batch['X'])
            loss = criterion(predictions, batch['y'])
            val_loss += loss.item()
    
    # LR scheduling
    scheduler.step(val_loss)
    
    # Early stopping check
    if early_stopping.should_stop(val_loss):
        break
```

### 5.4 Regularization & Convergence

| Technique | Value | Purpose |
|-----------|-------|---------|
| **Batch size** | 32 | Balance speed & generalization |
| **Dropout** | 0.1 | Prevent overfitting |
| **Weight decay** | 1e-5 | L2 regularization |
| **Gradient clipping** | 1.0 | Prevent exploding gradients |
| **Early stopping** | Patience = 20 epochs | Stop if val loss doesn't improve |
| **Max epochs** | 200 | Upper limit |

**Expected convergence:** 50-100 epochs

### 5.5 Data Loading

```python
from torch.utils.data import DataLoader, TensorDataset

# Create sequences
def create_sequences(df, lookback=96, horizon=1):
    X, y = [], []
    for i in range(lookback, len(df) - horizon + 1):
        X.append(df.iloc[i-lookback:i].values)  # All features
        y.append(df.iloc[i]['kw'])  # Target only
    return np.array(X), np.array(y)

# Create dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True,  # Shuffle within train only
    num_workers=4
)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False
)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(
    test_dataset, 
    batch_size=32, 
    shuffle=False
)
```

---

## 6. Evaluation Framework

### 6.1 Metrics

**Primary Metrics:**

| Metric | Formula | Target | Unit |
|--------|---------|--------|------|
| **MAE** | mean(\|y_true - y_pred\|) | < 0.240 | kW |
| **RMSE** | sqrt(mean((y_true - y_pred)²)) | < 0.350 | kW |
| **MAPE** | mean(\|y_true - y_pred\| / y_true) × 100 | < 7.0 | % |

**Secondary Metrics:**

| Metric | Purpose |
|--------|---------|
| **R²** | Variance explained |
| **Max Error** | Worst-case performance |
| **Median AE** | Robust central tendency |

### 6.2 Baseline Comparisons

**Must compare against:**

1. **Persistence Model**
   ```python
   y_pred = y_true.shift(1)  # kw(t) = kw(t-1)
   ```
   Expected MAE: 0.268 kW

2. **Rolling Mean (4 intervals)**
   ```python
   y_pred = y_true.rolling(4).mean().shift(1)
   ```
   Expected MAE: 0.289 kW

3. **Historical Average (by hour)**
   ```python
   y_pred = df.groupby('hour')['kw'].transform('mean')
   ```
   Expected MAE: ~0.30 kW (worse due to weak patterns)

### 6.3 Evaluation Procedure

```python
# On test set
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for batch in test_loader:
        pred = model(batch['X'])
        predictions.extend(pred.cpu().numpy())
        actuals.extend(batch['y'].cpu().numpy())

predictions = np.array(predictions)
actuals = np.array(actuals)

# Inverse transform (if standardized)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
actuals = scaler.inverse_transform(actuals.reshape(-1, 1))

# Calculate metrics
mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
r2 = r2_score(actuals, predictions)

# Compare to baselines
baseline_mae = mean_absolute_error(actuals[1:], actuals[:-1])
improvement = (baseline_mae - mae) / baseline_mae * 100

print(f"MAE: {mae:.4f} kW")
print(f"RMSE: {rmse:.4f} kW")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2:.4f}")
print(f"Improvement over persistence: {improvement:.1f}%")
```

### 6.4 Diagnostic Plots

**Required visualizations:**

1. **Predictions vs Actuals (time series)**
   - Full test period
   - Zoom on 48-hour window

2. **Residuals over time**
   - Check for systematic bias
   - Look for autocorrelation in errors

3. **Residual distribution**
   - Histogram
   - Q-Q plot (should be normal)

4. **Error by hour of day**
   - Identify if errors concentrate at specific times

5. **Scatter: Predicted vs Actual**
   - Should fall on y=x line
   - R² reference

---

## 7. Hyperparameter Tuning

### 7.1 Search Space

| Hyperparameter | Options | Priority |
|----------------|---------|----------|
| **Learning rate** | [1e-4, 5e-4, 1e-3, 5e-3] | HIGH |
| **Batch size** | [16, 32, 64] | MEDIUM |
| **Hidden units** | [64, 128, 256] | MEDIUM |
| **Dropout** | [0.0, 0.1, 0.2] | LOW |
| **Number of blocks** | [2, 3, 4] | MEDIUM |
| **Lookback window** | [48, 96, 192] | LOW |

### 7.2 Search Strategy

**Phase 1:** Manual sweep (small dataset)
- Train with default config
- Adjust LR if not converging
- Test 2 vs 3 vs 4 blocks

**Phase 2:** Targeted grid search (if needed)
- Fix architecture from Phase 1
- Search over LR × batch size

**Budget:** Max 10 training runs

---

## 8. Implementation Checklist

### 8.1 Data Pipeline

- [ ] Load cleaned dataset from extraction pipeline
- [ ] Apply flatline filtering (remove 83 intervals)
- [ ] Create train/val/test splits (time-ordered)
- [ ] Generate sequences (96-step lookback)
- [ ] Standardize features (fit on train only)
- [ ] Create PyTorch DataLoaders
- [ ] Verify no data leakage (test dates > train dates)

### 8.2 Model Implementation

- [ ] Implement N-BEATS generic block
- [ ] Stack 3 blocks with residual connections
- [ ] Add batch normalization (optional)
- [ ] Add dropout layers
- [ ] Verify forward pass output shape
- [ ] Count trainable parameters (~150K expected)

### 8.3 Training Infrastructure

- [ ] Set up loss function (MSE)
- [ ] Configure Adam optimizer
- [ ] Implement learning rate scheduler
- [ ] Add gradient clipping
- [ ] Implement early stopping
- [ ] Set up logging (tensorboard/wandb)
- [ ] Save best model checkpoint

### 8.4 Evaluation

- [ ] Compute metrics on test set
- [ ] Generate baseline comparisons
- [ ] Create diagnostic plots
- [ ] Calculate improvement percentage
- [ ] Document results

---

## 9. Expected Results

### 9.1 Performance Expectations

| Scenario | MAE | RMSE | Likelihood | Interpretation |
|----------|-----|------|------------|----------------|
| **Optimistic** | 0.19 | 0.29 | 20% | 30% better than baseline — excellent |
| **Realistic** | 0.22 | 0.32 | 50% | 18% better — good, justifies model |
| **Pessimistic** | 0.25 | 0.36 | 30% | 7% better — marginal, may not justify complexity |

### 9.2 Feature Importance (Expected)

| Feature | Expected Importance | Rationale |
|---------|---------------------|-----------|
| `kw` (lag-1 to lag-4) | HIGH | Strong autocorr (0.75) |
| `pf` | HIGH | 0.957 correlation |
| `kvar_tot` | HIGH | 0.991 correlation (may be redundant) |
| `i_l1`, `p_l1` | MEDIUM | Strong phase predictors |
| `i_l3`, `p_l3` | MEDIUM | Moderate correlation |
| `hour`, `day_of_week` | LOW | Weak temporal patterns |
| Voltage features | LOW | Stable, low correlation |

### 9.3 Training Convergence

**Expected behavior:**
- Rapid initial loss drop (first 10 epochs)
- Slower convergence after epoch 30
- Convergence by epoch 80-100
- Val loss tracking train loss closely (low variance signal)

**Red flags:**
- Val loss >> train loss (overfitting)
- No improvement after 50 epochs (underfitting or bad LR)
- Loss oscillating wildly (LR too high)

---

## 10. Code Template

### 10.1 Model Definition (PyTorch)

```python
import torch
import torch.nn as nn

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, hidden_size=128, num_layers=4):
        super().__init__()
        
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        
        self.layers = nn.Sequential(*layers)
        self.backcast = nn.Linear(hidden_size, input_size)
        self.forecast = nn.Linear(hidden_size, 1)  # 1-step ahead
        
    def forward(self, x):
        # x shape: [batch, lookback, features]
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)  # Flatten
        
        h = self.layers(x_flat)
        backcast = self.backcast(h)
        forecast = self.forecast(h)
        
        return backcast, forecast

class NBeatsModel(nn.Module):
    def __init__(self, input_size, num_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size) for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        # x shape: [batch, lookback, features]
        batch_size, lookback, features = x.shape
        input_size = lookback * features
        
        residual = x
        forecast = torch.zeros(batch_size, 1).to(x.device)
        
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            
            # Reshape backcast to original shape
            backcast = backcast.reshape(batch_size, lookback, features)
            residual = residual - backcast
            forecast = forecast + block_forecast
        
        return forecast.squeeze()

# Initialize
model = NBeatsModel(input_size=96 * 12, num_blocks=3)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 10.2 Training Loop

```python
import torch.optim as optim
from torch.utils.data import DataLoader

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

# Training
best_val_loss = float('inf')
patience_counter = 0
max_patience = 20

for epoch in range(200):
    # Train
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    # Logging
    print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    # LR scheduling
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_e0206.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Load best model
model.load_state_dict(torch.load('best_model_e0206.pt'))
```

---

## 11. Deliverables

### 11.1 Code Artifacts

- [ ] `model_nbeats.py` - Model definition
- [ ] `train_e0206.py` - Training script
- [ ] `evaluate_e0206.py` - Evaluation script
- [ ] `data_loader.py` - Data preparation
- [ ] `config_e0206.yaml` - Hyperparameters
- [ ] `requirements.txt` - Dependencies

### 11.2 Model Artifacts

- [ ] `best_model_e0206.pt` - Trained weights
- [ ] `scaler_e0206.pkl` - Feature scaler
- [ ] `training_history.csv` - Loss curves

### 11.3 Results

- [ ] `evaluation_report_e0206.md` - Metrics & analysis
- [ ] `predictions_vs_actuals.png` - Time series plot
- [ ] `residuals_analysis.png` - Diagnostic plots
- [ ] `feature_importance.csv` - If extractable

---

## Document Status

**Version:** 1.0  
**Status:** Ready for implementation  
**Dependencies:** `data_cleaning_spec_v1_e0206_validated.md`  
**Next:** Begin model development
