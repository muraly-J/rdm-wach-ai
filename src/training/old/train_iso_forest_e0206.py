"""
ISOLATION FOREST TRAINING SCRIPT - e0206 AHU
Train Once, Deploy Everywhere

This script trains an Isolation Forest model for unsupervised anomaly detection
on the e0206 AHU system. The model learns "normal behavior fingerprints" from
historical data and can be reused for real-time anomaly scoring without retraining.

Purpose:
- Load historical cleaned data
- Engineer multivariate features
- Train Isolation Forest model
- Fit StandardScaler
- Save model, scaler, and feature configuration for production use

Industrial Design:
- No retraining required in production
- Deterministic feature engineering
- Portable model artifacts
- Separation of training and inference
"""

import os
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Get the project root directory (assuming script is in src/training/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Define paths
DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"

# Create model directory if it doesn't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Data file
DATA_FILE = DATA_DIR / "clean_longest.parquet"

# Output model files
MODEL_FILE = MODEL_DIR / "isolation_e0206.pkl"
SCALER_FILE = MODEL_DIR / "isolation_scaler_e0206.pkl"
FEATURES_FILE = MODEL_DIR / "isolation_features_e0206.json"
METADATA_FILE = MODEL_DIR / "isolation_metadata_e0206.json"


# ============================================================================
# ISOLATION FOREST CONFIGURATION
# ============================================================================

# Model hyperparameters (industrial settings)
ISO_CONFIG = {
    'n_estimators': 300,           # Number of trees (more = better pattern learning)
    'max_samples': 'auto',         # Auto-select optimal sample size
    'contamination': 0.02,         # Assume 2% of data are anomalies
    'max_features': 1.0,           # Use all features
    'bootstrap': False,            # No bootstrap sampling
    'n_jobs': -1,                  # Use all CPU cores
    'random_state': 42,            # Reproducibility
    'verbose': 0
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'rolling_window': 4,           # 4 intervals = 1 hour for 15-min data
    'use_power_delta': True,       # Include power change rate
    'use_imbalance': True,         # Include phase imbalance
    'use_rolling_mean': True       # Include short-term baseline
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(filepath):
    """
    Load and prepare data for training
    
    Args:
        filepath: Path to cleaned parquet file
    
    Returns:
        df: DataFrame with timestamp index
    """
    print("\n" + "="*80)
    print("LOADING DATA FOR TRAINING")
    print("="*80)
    
    print(f"Loading data from: {filepath}")
    df = pd.read_parquet(filepath)
    
    # Ensure timestamp index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            df = df.set_index('time')
        else:
            raise ValueError("No timestamp column found in data")
    
    # Sort by time
    df = df.sort_index()
    
    print(f"   Loaded {len(df)} rows")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   Duration: {(df.index[-1] - df.index[0]).days} days")
    print(f"   Columns: {len(df.columns)}")
    
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df, config):
    """
    Engineer features for Isolation Forest training
    
    CRITICAL: These features must be reproduced EXACTLY in production
    
    Features:
    - power_total: Raw power consumption
    - power_factor_avg: Power factor
    - current_l1, current_l2, current_l3: Phase currents
    - current_imbalance: Phase imbalance ratio
    - rolling_mean_1h: 1-hour rolling average
    - power_delta: Rate of power change
    
    Args:
        df: Raw DataFrame
        config: Feature engineering configuration
    
    Returns:
        df: DataFrame with engineered features
        feature_names: List of feature names for model input
    """
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # 1. Current Imbalance (CRITICAL: Same formula as EWMA script)
    if config['use_imbalance']:
        print("Computing current imbalance...")
        df['current_mean'] = df[['current_l1', 'current_l2', 'current_l3']].mean(axis=1)
        df['current_max'] = df[['current_l1', 'current_l2', 'current_l3']].max(axis=1)
        df['current_min'] = df[['current_l1', 'current_l2', 'current_l3']].min(axis=1)
        
        # Imbalance = (max - min) / mean
        df['current_imbalance'] = np.where(
            df['current_mean'] > 0,
            (df['current_max'] - df['current_min']) / df['current_mean'],
            0
        )
        print(f"   Mean imbalance: {df['current_imbalance'].mean():.4f}")
    
    # 2. Rolling Mean (1 hour baseline)
    if config['use_rolling_mean']:
        print(f"Computing rolling mean (window={config['rolling_window']})...")
        df['rolling_mean_1h'] = df['power_total'].rolling(
            window=config['rolling_window']
        ).mean()
        print(f"   Rolling mean computed")
    
    # 3. Power Delta (rate of change)
    if config['use_power_delta']:
        print("Computing power delta...")
        df['power_delta'] = df['power_total'].diff()
        print(f"   Mean abs delta: {df['power_delta'].abs().mean():.4f}")
    
    # Define feature names in exact order
    feature_names = [
        'power_total',
        'power_factor_avg',
        'current_l1',
        'current_l2',
        'current_l3',
        'current_imbalance',
        'rolling_mean_1h',
        'power_delta'
    ]
    
    # Check all features exist
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    
    print(f"\nFeature matrix shape before cleaning: {df[feature_names].shape}")
    
    # Drop rows with NaN (from rolling operations)
    df_clean = df.dropna(subset=feature_names)
    
    print(f"Feature matrix shape after cleaning: {df_clean[feature_names].shape}")
    print(f"Rows dropped: {len(df) - len(df_clean)}")
    
    return df_clean, feature_names


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_isolation_forest(X, config):
    """
    Train Isolation Forest model
    
    Args:
        X: Feature matrix (numpy array)
        config: Model configuration dictionary
    
    Returns:
        model: Trained Isolation Forest
        training_stats: Dictionary of training statistics
    """
    print("\n" + "="*80)
    print("TRAINING ISOLATION FOREST")
    print("="*80)
    
    print("Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Initialize model
    model = IsolationForest(**config)
    
    # Train (fit) the model
    print("\nTraining in progress...")
    model.fit(X)
    
    # Compute training statistics
    scores = model.decision_function(X)
    predictions = model.predict(X)
    
    n_anomalies = (predictions == -1).sum()
    contamination_actual = n_anomalies / len(predictions)
    
    training_stats = {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_anomalies_detected': int(n_anomalies),
        'contamination_actual': float(contamination_actual),
        'score_mean': float(scores.mean()),
        'score_std': float(scores.std()),
        'score_min': float(scores.min()),
        'score_max': float(scores.max())
    }
    
    print("\nTraining complete!")
    print(f"   Samples trained: {training_stats['n_samples']:,}")
    print(f"   Features: {training_stats['n_features']}")
    print(f"   Anomalies detected: {training_stats['n_anomalies_detected']} ({contamination_actual*100:.2f}%)")
    print(f"   Score range: [{training_stats['score_min']:.4f}, {training_stats['score_max']:.4f}]")
    
    return model, training_stats


# ============================================================================
# SCALING
# ============================================================================

def fit_scaler(X):
    """
    Fit StandardScaler on training data
    
    CRITICAL: Isolation Forest is distance-based, so scaling is essential
    
    Args:
        X: Feature matrix
    
    Returns:
        scaler: Fitted StandardScaler
        X_scaled: Scaled feature matrix
    """
    print("\n" + "="*80)
    print("FITTING SCALER")
    print("="*80)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("StandardScaler fitted")
    print(f"   Feature means: {scaler.mean_[:3]} ... (first 3)")
    print(f"   Feature stds:  {scaler.scale_[:3]} ... (first 3)")
    
    return scaler, X_scaled


# ============================================================================
# MODEL SAVING
# ============================================================================

def save_artifacts(model, scaler, feature_names, training_stats, config):
    """
    Save all model artifacts for production deployment
    
    Saves:
    - Trained Isolation Forest model
    - Fitted StandardScaler
    - Feature names (exact order)
    - Training metadata
    
    Args:
        model: Trained Isolation Forest
        scaler: Fitted StandardScaler
        feature_names: List of feature names
        training_stats: Training statistics dictionary
        config: Model configuration
    """
    print("\n" + "="*80)
    print("SAVING MODEL ARTIFACTS")
    print("="*80)
    
    # Save model
    joblib.dump(model, MODEL_FILE)
    print(f"   Model saved to: {MODEL_FILE}")
    
    # Save scaler
    joblib.dump(scaler, SCALER_FILE)
    print(f"   Scaler saved to: {SCALER_FILE}")
    
    # Save feature names (JSON)
    with open(FEATURES_FILE, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"   Features saved to: {FEATURES_FILE}")
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'model_type': 'IsolationForest',
        'sklearn_version': joblib.__version__,
        'model_config': config,
        'feature_config': FEATURE_CONFIG,
        'training_stats': training_stats,
        'feature_names': feature_names,
        'n_features': len(feature_names)
    }
    
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata saved to: {METADATA_FILE}")
    
    print("\nAll artifacts saved successfully!")


# ============================================================================
# VALIDATION
# ============================================================================

def validate_saved_model():
    """
    Validate that saved artifacts can be loaded correctly
    
    This ensures the model is deployable to production
    """
    print("\n" + "="*80)
    print("VALIDATING SAVED ARTIFACTS")
    print("="*80)
    
    try:
        # Load model
        model = joblib.load(MODEL_FILE)
        print("   Model loaded successfully")
        
        # Load scaler
        scaler = joblib.load(SCALER_FILE)
        print("   Scaler loaded successfully")
        
        # Load features
        with open(FEATURES_FILE, 'r') as f:
            features = json.load(f)
        print(f"   Features loaded successfully ({len(features)} features)")
        
        # Load metadata
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        print(f"   Metadata loaded successfully")
        
        print("\nValidation passed! Model is ready for production deployment.")
        return True
        
    except Exception as e:
        print(f"\nValidation failed: {e}")
        return False


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("\n" + "="*80)
    print("ISOLATION FOREST TRAINING PIPELINE - e0206 AHU")
    print("="*80)
    print("\nTrain Once, Deploy Everywhere")
    print("Unsupervised Anomaly Detection Model")
    print("="*80)
    
    # Step 1: Load data
    df = load_data(DATA_FILE)
    
    # Step 2: Engineer features
    df_features, feature_names = engineer_features(df, FEATURE_CONFIG)
    
    # Step 3: Extract feature matrix
    X = df_features[feature_names].values
    print(f"\nFeature matrix shape: {X.shape}")
    
    # Step 4: Fit scaler
    scaler, X_scaled = fit_scaler(X)
    
    # Step 5: Train Isolation Forest
    model, training_stats = train_isolation_forest(X_scaled, ISO_CONFIG)
    
    # Step 6: Save all artifacts
    save_artifacts(model, scaler, feature_names, training_stats, ISO_CONFIG)
    
    # Step 7: Validate saved model
    validation_success = validate_saved_model()
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    if validation_success:
        print("\nModel Status: READY FOR PRODUCTION")
        print("\nSaved artifacts:")
        print(f"   1. Model:    {MODEL_FILE}")
        print(f"   2. Scaler:   {SCALER_FILE}")
        print(f"   3. Features: {FEATURES_FILE}")
        print(f"   4. Metadata: {METADATA_FILE}")
        print("\nUsage in production:")
        print("   import joblib, json")
        print(f"   model = joblib.load('{MODEL_FILE}')")
        print(f"   scaler = joblib.load('{SCALER_FILE}')")
        print(f"   features = json.load(open('{FEATURES_FILE}'))")
        print("\nNext steps:")
        print("   1. Integrate model into health monitoring system")
        print("   2. Load model for real-time anomaly scoring")
        print("   3. No retraining required unless load profile changes")
    else:
        print("\nModel Status: VALIDATION FAILED")
        print("Please check error messages above")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
