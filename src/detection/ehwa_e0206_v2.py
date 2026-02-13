"""
E0206 AHU HEALTH MONITORING SYSTEM
Industrial-Grade Anomaly Detection with EWMA + Isolation Forest

This script implements a hospital-grade health monitoring system using:
- Layered EWMA (Exponentially Weighted Moving Average) monitoring
- Multivariate drift detection
- Residual anomaly detection
- Isolation Forest multivariate anomaly detection (if trained model available)
- Composite health scoring (0-100)
- 96-step baseline forecast visualization with anomaly risk scoring

Workflow:
1. Load trained Isolation Forest model (optional)
2. EWMA-based drift monitoring (deterministic)
3. Isolation Forest anomaly scoring (if available)
4. Forecast generation + anomaly risk evaluation
5. Composite health score with all risk factors

No retraining required - model loaded from disk.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import timedelta
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Get the project root directory (assuming script is in src/detection/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Define paths
DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
OUTPUT_DIR = PROJECT_ROOT / "detections" / "outputs" / "health_monitoring"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data file
DATA_FILE = DATA_DIR / "clean_longest.parquet"

# Model files
MODEL_FILE = MODEL_DIR / "isolation_e0206.pkl"
SCALER_FILE = MODEL_DIR / "isolation_scaler_e0206.pkl"
FEATURES_FILE = MODEL_DIR / "isolation_features_e0206.json"


# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# EWMA parameters
EWMA_LAMBDA = 0.2  # Smoothing parameter (0 < lambda <= 1)
CONTROL_LIMIT = 3  # Number of standard deviations for control limits

# Baseline window (use first 60% of data for establishing normal behavior)
BASELINE_FRACTION = 0.6

# Composite health score weights (updated with Isolation Forest)
WEIGHTS = {
    'kw_drift': 0.30,            # Power consumption drift (reduced from 0.40)
    'power_factor_drift': 0.15,  # Power factor degradation (reduced from 0.20)
    'imbalance_drift': 0.15,     # Phase imbalance issues (reduced from 0.20)
    'residual_spike': 0.15,      # Unexpected residual spikes (reduced from 0.20)
    'iso_risk': 0.25             # Isolation Forest multivariate anomaly (NEW)
}

# Health score thresholds
HEALTH_EXCELLENT = 90  # Above this: excellent health
HEALTH_GOOD = 70       # 70-90: good health
HEALTH_WARNING = 50    # 50-70: warning zone
# Below 50: critical zone


# ============================================================================
# COLUMN MAPPING
# ============================================================================

# Map parquet columns to expected names
COLUMN_MAP = {
    'power_total': 'power_total',        # May also be 'kw' or 'power_demand'
    'power_factor_avg': 'power_factor_avg',
    'current_l1': 'current_l1',
    'current_l2': 'current_l2',
    'current_l3': 'current_l3',
    'time': 'timestamp'                   # Index column
}


# ============================================================================
# ISOLATION FOREST MODEL LOADING
# ============================================================================

def load_isolation_forest_model():
    """
    Load trained Isolation Forest model and artifacts
    
    Returns:
        iso_model: Trained Isolation Forest
        iso_scaler: Fitted StandardScaler
        iso_features: List of feature names
        model_loaded: Boolean indicating successful load
    """
    print("\n" + "="*80)
    print("LOADING ISOLATION FOREST MODEL")
    print("="*80)
    
    try:
        # Load model
        iso_model = joblib.load(MODEL_FILE)
        print(f"   Model loaded from: {MODEL_FILE}")
        
        # Load scaler
        iso_scaler = joblib.load(SCALER_FILE)
        print(f"   Scaler loaded from: {SCALER_FILE}")
        
        # Load feature names
        with open(FEATURES_FILE, 'r') as f:
            iso_features = json.load(f)
        print(f"   Features loaded: {len(iso_features)} features")
        print(f"   Feature list: {iso_features}")
        
        print("\nIsolation Forest model ready for anomaly scoring")
        return iso_model, iso_scaler, iso_features, True
        
    except FileNotFoundError as e:
        print(f"\nWARNING: Isolation Forest model not found")
        print(f"   Missing file: {e.filename}")
        print(f"   Please run train_iso_forest_e0206.py first")
        print(f"   Continuing with EWMA monitoring only...")
        return None, None, None, False
    
    except Exception as e:
        print(f"\nWARNING: Error loading Isolation Forest model: {e}")
        print(f"   Continuing with EWMA monitoring only...")
        return None, None, None, False


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_prepare_data(filepath):
    """
    Load parquet file and prepare for monitoring
    
    Args:
        filepath: Path to cleaned parquet file
    
    Returns:
        df: Prepared DataFrame with timestamp index
    """
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load parquet
    print(f"Loading data from: {filepath}")
    df = pd.read_parquet(filepath)
    
    print(f"   Loaded {len(df)} rows")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   Duration: {(df.index[-1] - df.index[0]).days} days")
    
    # Ensure timestamp index
    if 'time' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('time')
    
    # Sort by time
    df = df.sort_index()
    
    # Verify 15-minute frequency
    freq = pd.infer_freq(df.index)
    print(f"   Detected frequency: {freq}")
    
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def compute_current_imbalance(df):
    """
    Compute current imbalance ratio and additional features for Isolation Forest
    
    Features computed:
    - Current imbalance: (max_phase - min_phase) / mean_phase
    - Rolling mean (1 hour): 4-interval rolling average
    - Power delta: Rate of power change
    
    CRITICAL: These features must match training script exactly
    
    Args:
        df: DataFrame with current_l1, current_l2, current_l3, power_total
    
    Returns:
        df: DataFrame with added features
    """
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    print("Computing current imbalance...")
    
    # Compute phase statistics
    df['current_mean'] = df[['current_l1', 'current_l2', 'current_l3']].mean(axis=1)
    df['current_max'] = df[['current_l1', 'current_l2', 'current_l3']].max(axis=1)
    df['current_min'] = df[['current_l1', 'current_l2', 'current_l3']].min(axis=1)
    
    # Compute imbalance ratio (avoid division by zero)
    df['current_imbalance'] = np.where(
        df['current_mean'] > 0,
        (df['current_max'] - df['current_min']) / df['current_mean'],
        0
    )
    
    print(f"   Mean imbalance: {df['current_imbalance'].mean():.4f}")
    print(f"   Max imbalance: {df['current_imbalance'].max():.4f}")
    
    # Additional features for Isolation Forest
    print("Computing additional features for Isolation Forest...")
    
    # Rolling mean (1 hour = 4 intervals for 15-min data)
    df['rolling_mean_1h'] = df['power_total'].rolling(window=4).mean()
    print(f"   Rolling mean computed")
    
    # Power delta (rate of change)
    df['power_delta'] = df['power_total'].diff()
    print(f"   Power delta computed")
    
    return df


# ============================================================================
# BASELINE ESTABLISHMENT
# ============================================================================

def establish_baseline(df, fraction=0.6):
    """
    Establish normal baseline statistics from first portion of data
    
    Args:
        df: Full DataFrame
        fraction: Fraction of data to use for baseline (0.6 = 60%)
    
    Returns:
        baseline_stats: Dictionary of baseline statistics
        baseline_df: DataFrame containing baseline period
    """
    print("\n" + "="*80)
    print("ESTABLISHING NORMAL BASELINE")
    print("="*80)
    
    # Split data
    split_idx = int(len(df) * fraction)
    baseline_df = df.iloc[:split_idx].copy()
    
    print(f"Using first {fraction*100:.0f}% of data ({len(baseline_df)} samples)")
    print(f"   Baseline period: {baseline_df.index[0]} to {baseline_df.index[-1]}")
    
    # Compute baseline statistics
    baseline_stats = {
        'kw_mean': baseline_df['power_total'].mean(),
        'kw_std': baseline_df['power_total'].std(),
        'pf_mean': baseline_df['power_factor_avg'].mean(),
        'pf_std': baseline_df['power_factor_avg'].std(),
        'imbalance_mean': baseline_df['current_imbalance'].mean(),
        'imbalance_std': baseline_df['current_imbalance'].std(),
    }
    
    print("\nBaseline Statistics:")
    print(f"   Power (kW):        {baseline_stats['kw_mean']:.2f} +/- {baseline_stats['kw_std']:.2f}")
    print(f"   Power Factor:      {baseline_stats['pf_mean']:.4f} +/- {baseline_stats['pf_std']:.4f}")
    print(f"   Current Imbalance: {baseline_stats['imbalance_mean']:.4f} +/- {baseline_stats['imbalance_std']:.4f}")
    
    return baseline_stats, baseline_df


# ============================================================================
# EWMA MONITORING
# ============================================================================

def compute_ewma(series, lam=0.2):
    """
    Compute Exponentially Weighted Moving Average
    
    Args:
        series: Time series data
        lam: Smoothing parameter (lambda)
    
    Returns:
        ewma: EWMA series
    """
    return series.ewm(alpha=lam, adjust=False).mean()


def compute_ewma_zscore(ewma_series, baseline_mean, baseline_std, lam=0.2):
    """
    Compute Z-score for EWMA monitoring
    
    Z = (EWMA - mu) / (sigma * sqrt(lambda / (2 - lambda)))
    
    Args:
        ewma_series: EWMA time series
        baseline_mean: Baseline mean
        baseline_std: Baseline standard deviation
        lam: EWMA lambda parameter
    
    Returns:
        z_scores: Standardized EWMA scores
    """
    # EWMA standard error
    ewma_factor = np.sqrt(lam / (2 - lam))
    
    # Compute Z-scores
    z_scores = (ewma_series - baseline_mean) / (baseline_std * ewma_factor)
    
    return z_scores


def apply_ewma_monitoring(df, baseline_stats, lam=0.2):
    """
    Apply EWMA monitoring to all tracked metrics
    
    Args:
        df: DataFrame with features
        baseline_stats: Baseline statistics
        lam: EWMA lambda parameter
    
    Returns:
        df: DataFrame with EWMA metrics added
    """
    print("\n" + "="*80)
    print("APPLYING EWMA MONITORING")
    print("="*80)
    
    print(f"Using lambda = {lam}")
    
    # Compute EWMA for each metric
    df['ewma_kw'] = compute_ewma(df['power_total'], lam)
    df['ewma_pf'] = compute_ewma(df['power_factor_avg'], lam)
    df['ewma_imbalance'] = compute_ewma(df['current_imbalance'], lam)
    
    # Compute EWMA Z-scores
    df['z_kw'] = compute_ewma_zscore(
        df['ewma_kw'], 
        baseline_stats['kw_mean'], 
        baseline_stats['kw_std'], 
        lam
    )
    
    df['z_pf'] = compute_ewma_zscore(
        df['ewma_pf'], 
        baseline_stats['pf_mean'], 
        baseline_stats['pf_std'], 
        lam
    )
    
    df['z_imbalance'] = compute_ewma_zscore(
        df['ewma_imbalance'], 
        baseline_stats['imbalance_mean'], 
        baseline_stats['imbalance_std'], 
        lam
    )
    
    print("   EWMA computed for: power, power_factor, current_imbalance")
    print("   Z-scores computed for drift detection")
    
    return df


# ============================================================================
# RESIDUAL MONITORING
# ============================================================================

def compute_residual_monitoring(df, baseline_df, window=4):
    """
    Compute residual-based anomaly detection
    
    Uses rolling mean as baseline predictor:
    residual = actual - rolling_mean
    
    Args:
        df: Full DataFrame
        baseline_df: Baseline period DataFrame
        window: Rolling window size (4 = 1 hour for 15-min data)
    
    Returns:
        df: DataFrame with residual metrics
    """
    print("\n" + "="*80)
    print("RESIDUAL MONITORING")
    print("="*80)
    
    # Compute rolling mean baseline predictor
    df['rolling_mean'] = df['power_total'].rolling(window=window).mean()
    
    # Compute residuals
    df['residual'] = df['power_total'] - df['rolling_mean']
    
    # Establish baseline residual statistics
    baseline_residual_std = baseline_df['power_total'].diff().std()
    
    print(f"   Rolling window: {window} intervals (1 hour)")
    print(f"   Baseline residual std: {baseline_residual_std:.4f}")
    
    # Compute residual Z-scores
    df['z_residual'] = df['residual'] / baseline_residual_std
    
    return df


# ============================================================================
# ANOMALY SCORE NORMALIZATION
# ============================================================================

def normalize_anomaly_scores(df, control_limit=3):
    """
    Convert Z-scores to bounded risk scores [0, 1]
    
    Risk = min(|Z| / control_limit, 1.0)
    
    Args:
        df: DataFrame with Z-scores
        control_limit: Z-score threshold (typically 3)
    
    Returns:
        df: DataFrame with normalized risk scores
    """
    print("\n" + "="*80)
    print("NORMALIZING ANOMALY SCORES")
    print("="*80)
    
    print(f"Using control limit: {control_limit} sigma")
    
    # Compute bounded risk scores
    df['risk_kw'] = np.minimum(np.abs(df['z_kw']) / control_limit, 1.0)
    df['risk_pf'] = np.minimum(np.abs(df['z_pf']) / control_limit, 1.0)
    df['risk_imbalance'] = np.minimum(np.abs(df['z_imbalance']) / control_limit, 1.0)
    df['risk_residual'] = np.minimum(np.abs(df['z_residual']) / control_limit, 1.0)
    
    # Summary statistics
    print("\nRisk Score Statistics:")
    print(f"   Power drift risk:      mean={df['risk_kw'].mean():.3f}, max={df['risk_kw'].max():.3f}")
    print(f"   PF drift risk:         mean={df['risk_pf'].mean():.3f}, max={df['risk_pf'].max():.3f}")
    print(f"   Imbalance drift risk:  mean={df['risk_imbalance'].mean():.3f}, max={df['risk_imbalance'].max():.3f}")
    print(f"   Residual spike risk:   mean={df['risk_residual'].mean():.3f}, max={df['risk_residual'].max():.3f}")
    
    return df


# ============================================================================
# ISOLATION FOREST ANOMALY SCORING
# ============================================================================

def score_isolation_forest(df, iso_model, iso_scaler, iso_features):
    """
    Score historical data using trained Isolation Forest model
    
    Args:
        df: DataFrame with engineered features
        iso_model: Trained Isolation Forest
        iso_scaler: Fitted StandardScaler
        iso_features: List of feature names (exact order)
    
    Returns:
        df: DataFrame with iso_score and iso_risk columns
    """
    print("\n" + "="*80)
    print("ISOLATION FOREST ANOMALY SCORING")
    print("="*80)
    
    # Check if all required features exist
    missing_features = [f for f in iso_features if f not in df.columns]
    if missing_features:
        print(f"WARNING: Missing features for Isolation Forest: {missing_features}")
        print("   Creating dummy risk scores...")
        df['iso_score'] = 0.0
        df['iso_risk'] = 0.0
        return df
    
    # Extract feature matrix (must match training order exactly)
    X_live = df[iso_features].values
    
    # Handle NaN values (from rolling operations)
    nan_mask = np.isnan(X_live).any(axis=1)
    n_nan = nan_mask.sum()
    if n_nan > 0:
        print(f"   Found {n_nan} rows with NaN values, filling with forward fill...")
        df_temp = df[iso_features].fillna(method='ffill').fillna(method='bfill')
        X_live = df_temp.values
    
    print(f"   Feature matrix shape: {X_live.shape}")
    print(f"   Features: {iso_features}")
    
    # Scale features (same transformation as training)
    X_live_scaled = iso_scaler.transform(X_live)
    print(f"   Features scaled using saved scaler")
    
    # Score with Isolation Forest
    # decision_function returns anomaly score (lower = more abnormal)
    iso_scores = iso_model.decision_function(X_live_scaled)
    df['iso_score'] = iso_scores
    
    print(f"   Anomaly scores computed")
    print(f"   Score range: [{iso_scores.min():.4f}, {iso_scores.max():.4f}]")
    
    # Normalize to risk score [0, 1]
    # 0 = normal, 1 = highly abnormal
    score_min = iso_scores.min()
    score_max = iso_scores.max()
    score_range = score_max - score_min
    
    if score_range > 0:
        df['iso_risk'] = 1 - ((iso_scores - score_min) / score_range)
    else:
        df['iso_risk'] = 0.0
    
    print(f"   Risk normalized to [0, 1]")
    print(f"   Mean iso_risk: {df['iso_risk'].mean():.3f}")
    print(f"   Max iso_risk: {df['iso_risk'].max():.3f}")
    
    # Identify top anomalies
    n_anomalies = (df['iso_risk'] > 0.8).sum()
    print(f"   High-risk anomalies (>0.8): {n_anomalies}")
    
    return df


def score_forecast_with_isolation_forest(forecast_df, historical_df, 
                                         iso_model, iso_scaler, iso_features):
    """
    Score 96-step forecast using Isolation Forest
    
    Workflow:
    1. Take forecast power values
    2. Compute required features (imbalance, rolling_mean, delta)
    3. Use last known values for phase currents and power factor
    4. Transform and score
    
    Args:
        forecast_df: DataFrame with forecast values
        historical_df: Historical data for feature context
        iso_model: Trained Isolation Forest
        iso_scaler: Fitted StandardScaler
        iso_features: List of feature names
    
    Returns:
        forecast_df: DataFrame with iso_risk column
    """
    print("\n" + "="*80)
    print("SCORING FORECAST WITH ISOLATION FOREST")
    print("="*80)
    
    # Get last known values from historical data for static features
    last_pf = historical_df['power_factor_avg'].iloc[-1]
    last_i1 = historical_df['current_l1'].iloc[-1]
    last_i2 = historical_df['current_l2'].iloc[-1]
    last_i3 = historical_df['current_l3'].iloc[-1]
    last_imbalance = historical_df['current_imbalance'].iloc[-1]
    
    print(f"   Using last known values for static features")
    print(f"   Last power factor: {last_pf:.4f}")
    print(f"   Last currents: L1={last_i1:.2f}, L2={last_i2:.2f}, L3={last_i3:.2f}")
    
    # Create feature matrix for forecast
    # Assume static features remain constant in forecast horizon
    forecast_df['power_factor_avg'] = last_pf
    forecast_df['current_l1'] = last_i1
    forecast_df['current_l2'] = last_i2
    forecast_df['current_l3'] = last_i3
    forecast_df['current_imbalance'] = last_imbalance
    
    # Compute rolling mean and delta for forecast
    # Use last values from historical as seed
    last_rolling = historical_df['rolling_mean_1h'].iloc[-1]
    forecast_df['rolling_mean_1h'] = forecast_df['forecast'].rolling(window=4, min_periods=1).mean()
    forecast_df['rolling_mean_1h'].iloc[0] = last_rolling  # Seed with historical
    
    forecast_df['power_delta'] = forecast_df['forecast'].diff()
    forecast_df['power_delta'].iloc[0] = 0  # No delta for first forecast point
    
    # Rename forecast column to power_total for consistency
    forecast_df['power_total'] = forecast_df['forecast']
    
    # Extract feature matrix
    X_forecast = forecast_df[iso_features].fillna(method='ffill').values
    
    print(f"   Forecast feature matrix shape: {X_forecast.shape}")
    
    # Scale and score
    X_forecast_scaled = iso_scaler.transform(X_forecast)
    forecast_scores = iso_model.decision_function(X_forecast_scaled)
    
    # Normalize using historical score range
    hist_scores = historical_df['iso_score'].values
    score_min = hist_scores.min()
    score_max = hist_scores.max()
    score_range = score_max - score_min
    
    if score_range > 0:
        forecast_df['iso_risk'] = 1 - ((forecast_scores - score_min) / score_range)
    else:
        forecast_df['iso_risk'] = 0.0
    
    forecast_df['iso_risk'] = forecast_df['iso_risk'].clip(0, 1)
    
    print(f"   Forecast anomaly risk computed")
    print(f"   Mean forecast risk: {forecast_df['iso_risk'].mean():.3f}")
    print(f"   Max forecast risk: {forecast_df['iso_risk'].max():.3f}")
    
    return forecast_df


# ============================================================================
# COMPOSITE HEALTH SCORE
# ============================================================================

def compute_composite_health_score(df, weights, use_iso_forest=True):
    """
    Compute weighted composite health score (0-100)
    
    Health = 100 * (1 - composite_risk)
    
    Args:
        df: DataFrame with risk scores
        weights: Dictionary of risk weights
        use_iso_forest: Whether to include Isolation Forest risk
    
    Returns:
        df: DataFrame with composite health score
    """
    print("\n" + "="*80)
    print("COMPUTING COMPOSITE HEALTH SCORE")
    print("="*80)
    
    print("Weights:")
    for key, value in weights.items():
        print(f"   {key}: {value*100:.0f}%")
    
    # Compute weighted composite risk
    if use_iso_forest and 'iso_risk' in df.columns:
        print("\nIncluding Isolation Forest anomaly risk")
        df['composite_risk'] = (
            weights['kw_drift'] * df['risk_kw'] +
            weights['power_factor_drift'] * df['risk_pf'] +
            weights['imbalance_drift'] * df['risk_imbalance'] +
            weights['residual_spike'] * df['risk_residual'] +
            weights['iso_risk'] * df['iso_risk']
        )
    else:
        print("\nUsing EWMA monitoring only (no Isolation Forest)")
        # Redistribute weights if not using Isolation Forest
        total_weight = (weights['kw_drift'] + weights['power_factor_drift'] + 
                       weights['imbalance_drift'] + weights['residual_spike'])
        df['composite_risk'] = (
            (weights['kw_drift'] / total_weight) * df['risk_kw'] +
            (weights['power_factor_drift'] / total_weight) * df['risk_pf'] +
            (weights['imbalance_drift'] / total_weight) * df['risk_imbalance'] +
            (weights['residual_spike'] / total_weight) * df['risk_residual']
        )
    
    # Convert to health score (0-100)
    df['health_score'] = 100 * (1 - df['composite_risk'])
    
    # Bound between 0 and 100
    df['health_score'] = df['health_score'].clip(0, 100)
    
    # Summary
    print(f"\nHealth Score Statistics:")
    print(f"   Mean: {df['health_score'].mean():.1f}")
    print(f"   Min:  {df['health_score'].min():.1f}")
    print(f"   Max:  {df['health_score'].max():.1f}")
    
    # Health distribution
    excellent = (df['health_score'] >= HEALTH_EXCELLENT).sum()
    good = ((df['health_score'] >= HEALTH_GOOD) & (df['health_score'] < HEALTH_EXCELLENT)).sum()
    warning = ((df['health_score'] >= HEALTH_WARNING) & (df['health_score'] < HEALTH_GOOD)).sum()
    critical = (df['health_score'] < HEALTH_WARNING).sum()
    
    total = len(df)
    print(f"\nHealth Distribution:")
    print(f"   Excellent (>=90): {excellent:6d} ({excellent/total*100:5.1f}%)")
    print(f"   Good (70-90):     {good:6d} ({good/total*100:5.1f}%)")
    print(f"   Warning (50-70):  {warning:6d} ({warning/total*100:5.1f}%)")
    print(f"   Critical (<50):   {critical:6d} ({critical/total*100:5.1f}%)")
    
    return df


# ============================================================================
# BASELINE FORECAST (VISUALIZATION ONLY)
# ============================================================================

def generate_96step_forecast(df):
    """
    Generate simple 96-step (24-hour) baseline forecast
    
    Uses persistence model: forecast = last observed value
    This is for visualization only, not predictive modeling
    
    Args:
        df: DataFrame with historical data
    
    Returns:
        forecast_df: DataFrame with 96-step forecast
        actual_96: Last 96 actual values for comparison
    """
    print("\n" + "="*80)
    print("GENERATING 96-STEP BASELINE FORECAST")
    print("="*80)
    
    # Get last value
    last_value = df['power_total'].iloc[-1]
    last_timestamp = df.index[-1]
    
    print(f"   Last observed value: {last_value:.2f} kW")
    print(f"   Last timestamp: {last_timestamp}")
    
    # Generate forecast timestamps (96 steps = 24 hours at 15-min intervals)
    forecast_timestamps = pd.date_range(
        start=last_timestamp + timedelta(minutes=15),
        periods=96,
        freq='15min'
    )
    
    # Simple persistence forecast
    forecast_values = np.repeat(last_value, 96)
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'timestamp': forecast_timestamps,
        'forecast': forecast_values
    }).set_index('timestamp')
    
    # Get last 96 actual values for comparison
    actual_96 = df['power_total'].iloc[-96:].copy()
    
    print(f"   Forecast horizon: 24 hours (96 intervals)")
    print(f"   Forecast method: Persistence (last value)")
    
    return forecast_df, actual_96


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_ewma_monitoring(df, baseline_stats, control_limit=3):
    """
    Plot EWMA monitoring charts with control limits
    
    Creates 3-panel plot:
    - Power (kW) with EWMA and control bands
    - Power Factor with EWMA
    - Current Imbalance with EWMA
    
    Args:
        df: DataFrame with EWMA metrics
        baseline_stats: Baseline statistics
        control_limit: Control limit in sigmas
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Panel 1: Power monitoring
    ax1 = axes[0]
    ax1.plot(df.index, df['power_total'], 'o-', 
             linewidth=0.5, markersize=1, alpha=0.5, 
             label='Actual Power', color='gray')
    ax1.plot(df.index, df['ewma_kw'], '-', 
             linewidth=2, label='EWMA', color='blue')
    
    # Control limits
    ewma_factor = np.sqrt(EWMA_LAMBDA / (2 - EWMA_LAMBDA))
    ucl = baseline_stats['kw_mean'] + control_limit * baseline_stats['kw_std'] * ewma_factor
    lcl = baseline_stats['kw_mean'] - control_limit * baseline_stats['kw_std'] * ewma_factor
    
    ax1.axhline(y=baseline_stats['kw_mean'], color='green', linestyle='--', 
                linewidth=1, label='Baseline Mean', alpha=0.7)
    ax1.axhline(y=ucl, color='red', linestyle='--', 
                linewidth=1, label=f'Control Limits (+/-{control_limit}σ)', alpha=0.7)
    ax1.axhline(y=lcl, color='red', linestyle='--', 
                linewidth=1, alpha=0.7)
    
    ax1.set_ylabel('Power (kW)', fontsize=11)
    ax1.set_title('EWMA Power Monitoring - e0206', fontweight='bold', fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Power Factor monitoring
    ax2 = axes[1]
    ax2.plot(df.index, df['power_factor_avg'], 'o-', 
             linewidth=0.5, markersize=1, alpha=0.5, 
             label='Actual PF', color='gray')
    ax2.plot(df.index, df['ewma_pf'], '-', 
             linewidth=2, label='EWMA', color='orange')
    ax2.axhline(y=baseline_stats['pf_mean'], color='green', linestyle='--', 
                linewidth=1, label='Baseline Mean', alpha=0.7)
    
    ax2.set_ylabel('Power Factor', fontsize=11)
    ax2.set_title('EWMA Power Factor Monitoring', fontweight='bold', fontsize=12)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Current Imbalance monitoring
    ax3 = axes[2]
    ax3.plot(df.index, df['current_imbalance'], 'o-', 
             linewidth=0.5, markersize=1, alpha=0.5, 
             label='Actual Imbalance', color='gray')
    ax3.plot(df.index, df['ewma_imbalance'], '-', 
             linewidth=2, label='EWMA', color='purple')
    ax3.axhline(y=baseline_stats['imbalance_mean'], color='green', linestyle='--', 
                linewidth=1, label='Baseline Mean', alpha=0.7)
    
    ax3.set_xlabel('Time', fontsize=11)
    ax3.set_ylabel('Current Imbalance Ratio', fontsize=11)
    ax3.set_title('EWMA Current Imbalance Monitoring', fontweight='bold', fontsize=12)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'e0206_ewma_monitoring.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   EWMA monitoring plot saved to: {output_path}")


def plot_composite_health(df):
    """
    Plot composite health score over time
    
    Args:
        df: DataFrame with health_score
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot health score
    ax.plot(df.index, df['health_score'], '-', 
            linewidth=1.5, color='navy', alpha=0.8)
    ax.fill_between(df.index, df['health_score'], 0, 
                     alpha=0.3, color='navy')
    
    # Threshold lines
    ax.axhline(y=HEALTH_EXCELLENT, color='green', linestyle='--', 
               linewidth=2, label=f'Excellent (>={HEALTH_EXCELLENT})', alpha=0.7)
    ax.axhline(y=HEALTH_GOOD, color='orange', linestyle='--', 
               linewidth=2, label=f'Warning (<{HEALTH_GOOD})', alpha=0.7)
    ax.axhline(y=HEALTH_WARNING, color='red', linestyle='--', 
               linewidth=2, label=f'Critical (<{HEALTH_WARNING})', alpha=0.7)
    
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Health Score (0-100)', fontsize=11)
    ax.set_title('Composite Health Score - e0206 AHU', fontweight='bold', fontsize=13)
    ax.set_ylim([0, 105])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'e0206_composite_health_plot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Composite health plot saved to: {output_path}")


def plot_96step_forecast(forecast_df, actual_96):
    """
    Plot 96-step forecast with last 96 actual values
    
    Args:
        forecast_df: DataFrame with forecast
        actual_96: Series with last 96 actual values
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot last 96 actual values
    ax.plot(actual_96.index, actual_96.values, 'o-', 
            linewidth=2, markersize=4, label='Actual (Last 24h)', 
            color='black', alpha=0.8)
    
    # Plot 96-step forecast
    ax.plot(forecast_df.index, forecast_df['forecast'], 's--', 
            linewidth=2, markersize=3, label='Forecast (Next 24h)', 
            color='red', alpha=0.7)
    
    # Add vertical line at forecast boundary
    ax.axvline(x=actual_96.index[-1], color='gray', 
               linestyle=':', linewidth=2, alpha=0.5)
    ax.text(actual_96.index[-1], ax.get_ylim()[1]*0.95, 
            'Forecast Start', rotation=90, va='top', ha='right', 
            fontsize=9, alpha=0.7)
    
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Power (kW)', fontsize=11)
    ax.set_title('96-Step Baseline Forecast - e0206 AHU', fontweight='bold', fontsize=13)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'e0206_96step_forecast.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   96-step forecast plot saved to: {output_path}")


# ============================================================================
# OUTPUT SAVING
# ============================================================================

def save_health_timeseries(df):
    """
    Save enriched health monitoring time series to CSV
    
    Args:
        df: DataFrame with all monitoring metrics
    """
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)
    
    # Select columns to save
    output_cols = [
        'power_total',
        'power_factor_avg',
        'current_l1', 'current_l2', 'current_l3',
        'current_mean', 'current_imbalance',
        'rolling_mean_1h', 'power_delta',
        'ewma_kw', 'ewma_pf', 'ewma_imbalance',
        'z_kw', 'z_pf', 'z_imbalance', 'z_residual',
        'risk_kw', 'risk_pf', 'risk_imbalance', 'risk_residual',
        'iso_score', 'iso_risk',
        'composite_risk',
        'health_score'
    ]
    
    # Filter to columns that exist
    available_cols = [col for col in output_cols if col in df.columns]
    
    output_df = df[available_cols].copy()
    
    # Save to CSV
    output_path = OUTPUT_DIR / 'e0206_health_timeseries.csv'
    output_df.to_csv(output_path)
    
    print(f"   Health time series saved to: {output_path}")
    print(f"   Columns saved: {len(available_cols)}")
    print(f"   Rows saved: {len(output_df)}")
    
    return output_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("E0206 AHU HEALTH MONITORING SYSTEM")
    print("="*80)
    print("\nIndustrial-Grade Anomaly Detection")
    print("EWMA Monitoring + Isolation Forest Anomaly Detection")
    print("="*80)
    
    # Step 1: Load Isolation Forest model (if available)
    iso_model, iso_scaler, iso_features, model_loaded = load_isolation_forest_model()
    
    # Step 2: Load data
    df = load_and_prepare_data(DATA_FILE)
    
    # Step 3: Feature engineering (includes features for Isolation Forest)
    df = compute_current_imbalance(df)
    
    # Step 4: Establish baseline
    baseline_stats, baseline_df = establish_baseline(df, BASELINE_FRACTION)
    
    # Step 5: Apply EWMA monitoring
    df = apply_ewma_monitoring(df, baseline_stats, EWMA_LAMBDA)
    
    # Step 6: Residual monitoring
    df = compute_residual_monitoring(df, baseline_df, window=4)
    
    # Step 7: Normalize anomaly scores
    df = normalize_anomaly_scores(df, CONTROL_LIMIT)
    
    # Step 8: Score with Isolation Forest (if model loaded)
    if model_loaded:
        df = score_isolation_forest(df, iso_model, iso_scaler, iso_features)
    else:
        print("\nSkipping Isolation Forest scoring (model not available)")
        df['iso_score'] = 0.0
        df['iso_risk'] = 0.0
    
    # Step 9: Compute composite health score
    df = compute_composite_health_score(df, WEIGHTS, use_iso_forest=model_loaded)
    
    # Step 10: Generate 96-step forecast
    forecast_df, actual_96 = generate_96step_forecast(df)
    
    # Step 11: Score forecast with Isolation Forest (if model loaded)
    if model_loaded:
        forecast_df = score_forecast_with_isolation_forest(
            forecast_df, df, iso_model, iso_scaler, iso_features
        )
    
    # Step 12: Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_ewma_monitoring(df, baseline_stats, CONTROL_LIMIT)
    plot_composite_health(df)
    plot_96step_forecast(forecast_df, actual_96)
    
    # Step 13: Save outputs
    csv_path = save_health_timeseries(df)
    
    # Final summary
    print("\n" + "="*80)
    print("MONITORING COMPLETE")
    print("="*80)
    print("\nAll outputs saved to:")
    print(f"   {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("   1. e0206_health_timeseries.csv")
    print("   2. e0206_composite_health_plot.png")
    print("   3. e0206_ewma_monitoring.png")
    print("   4. e0206_96step_forecast.png")
    
    if model_loaded:
        print("\n" + "="*80)
        print("MONITORING MODE: EWMA + ISOLATION FOREST")
        print("="*80)
        print("Multivariate anomaly detection active")
        print(f"Anomaly risk range: [{df['iso_risk'].min():.3f}, {df['iso_risk'].max():.3f}]")
        if 'iso_risk' in forecast_df.columns:
            print(f"Forecast risk range: [{forecast_df['iso_risk'].min():.3f}, {forecast_df['iso_risk'].max():.3f}]")
    else:
        print("\n" + "="*80)
        print("MONITORING MODE: EWMA ONLY")
        print("="*80)
        print("Isolation Forest not available")
        print("Run train_iso_forest_e0206.py to enable multivariate anomaly detection")
    
    print("\n" + "="*80)
    print("System Status: HOSPITAL-GRADE MONITORING ACTIVE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()