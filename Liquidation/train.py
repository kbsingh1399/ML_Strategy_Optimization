import os
import sys
import time
import json
import hashlib
import polars as pl
import numpy as np
import pickle
from sklearn.metrics import classification_report, accuracy_score

# Fix Windows cp1252 UnicodeEncodeError for any non-ASCII print output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from features import load_and_merge_data, compute_rolling_features, generate_labels

# Model classes
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Define active symbols for multi-asset training
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT",
    "LTCUSDT", "NEARUSDT", "SUIUSDT", "TRXUSDT"
]

FEATURE_COLS = [
    "trigger_type", "poc_pos", "delta_usd_ratio", "size_ratio", 
    "trade_ratio", "liq_long_z_50", "liq_short_z_50", 
    "liq_long_z_200", "liq_short_z_200", "cvd_z_10", 
    "cvd_z_50", "cvd_z_200", "atr_ratio", "close_to_ema_200"
]

def load_multi_asset_data(symbols):
    """Loads and computes features for all symbols, returns concatenated dataset."""
    import gc
    all_dfs = []
    for sym in symbols:
        print(f"Processing data for {sym}...")
        try:
            df = load_and_merge_data(sym)
            df = compute_rolling_features(df)
            df = generate_labels(df)
            all_dfs.append(df)
            gc.collect()
        except Exception as e:
            print(f"Error loading {sym}: {e}")
            
    # Concatenate all symbols
    full_df = pl.concat(all_dfs)
    gc.collect()
    return full_df

def prepare_ml_dataset(df):
    """Filter to trigger events and split into features and targets."""
    # Filter to only active triggers
    df_trigger = df.filter(pl.col("trigger_type") != 0)
    
    # Features & Labels
    X = df_trigger.select(FEATURE_COLS).to_pandas()
    y = df_trigger["target_label"].to_pandas()
    
    # Map target labels from {-1, 0, 1} to {0, 1, 2} for multi-class classification
    # 0 = Breakout (-1)
    # 1 = Hold/No Trade (0)
    # 2 = Reversal (1)
    y_mapped = y.map({-1: 0, 0: 1, 1: 2})
    
    # Keep datetime and symbol for sorting/splitting
    meta = df_trigger.select(["Symbol", "datetime"]).to_pandas()
    return X, y_mapped, meta

def train_ensemble():
    print("Step 1: Loading and engineering features for all assets...")
    df = load_multi_asset_data(SYMBOLS)
    
    print("\nStep 2: Preparing dataset...")
    X, y, meta = prepare_ml_dataset(df)
    
    import pandas as pd
    from datetime import timedelta
    
    # LIVE PRODUCTION MODE: Dynamic Window Selection
    print("\n--- Dynamic Window Selection (Recent 14-day OOS) ---")
    max_date = meta['datetime'].max()
    test_start_date = max_date - timedelta(days=14)
    
    # Create the evaluation sets (last 14 days)
    mask_test = meta['datetime'] >= test_start_date
    X_eval, y_eval = X[mask_test], y[mask_test]
    
    windows = [30, 60, 90, 180, None] # None = ALL data
    best_f1 = -1
    best_window = None
    
    from sklearn.metrics import f1_score
    
    for w in windows:
        if w is not None:
            train_start = test_start_date - timedelta(days=w)
            mask_train = (meta['datetime'] >= train_start) & (meta['datetime'] < test_start_date)
        else:
            mask_train = meta['datetime'] < test_start_date
            
        X_t, y_t = X[mask_train], y[mask_train]
        
        if len(set(y_t)) < 3:
            continue # Not enough classes
            
        # Quick eval with LGBM
        eval_model = lgb.LGBMClassifier(n_estimators=50, max_depth=3, num_leaves=15, random_state=42, verbosity=-1)
        eval_model.fit(X_t, y_t)
        preds = eval_model.predict(X_eval)
        
        # We care about F1 macro for reversal/breakout
        score = f1_score(y_eval, preds, average='macro')
        w_str = f"{w} days" if w else "ALL data"
        print(f"Window {w_str}: 14-day OOS F1 = {score:.4f}")
        
        if score > best_f1:
            best_f1 = score
            best_window = w
            
    win_str = f"{best_window} days" if best_window else "ALL data"
    print(f"\n[OK] Selected optimal lookback window: {win_str} (F1: {best_f1:.4f})")
    
    # Now retrain FINAL models on the best window (including the last 14 days!)
    if best_window is not None:
        final_train_start = max_date - timedelta(days=best_window)
        final_mask = meta['datetime'] >= final_train_start
    else:
        final_mask = pd.Series(True, index=meta.index)
        
    X_train, y_train = X[final_mask], y[final_mask]
    X_test, y_test = X_train, y_train # Dummy test for logging
    
    print(f"\nFinal Train samples: {len(X_train)} (starting {meta['datetime'][final_mask].min()}, ending {max_date})")
    
    # Check class distribution
    print("\nTrain class distribution:\n", y_train.value_counts())
    
    # Create models directory absolute to this file's folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Train LightGBM
    print("\n--- Training LightGBM Classifier ---")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_train, y_train)
    
    # 2. Train XGBoost
    print("\n--- Training XGBoost Classifier ---")
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=5,
        random_state=42,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)
    
    # 3. Train CatBoost
    print("\n--- Training CatBoost Classifier ---")
    cb_model = cb.CatBoostClassifier(
        iterations=150,
        learning_rate=0.03,
        depth=5,
        random_state=42,
        verbose=0
    )
    cb_model.fit(X_train, y_train)
    
    # Save models atomically using tmp + replace
    lgb_tmp = os.path.join(models_dir, "lgb_model.pkl.tmp")
    xgb_tmp = os.path.join(models_dir, "xgb_model.pkl.tmp")
    cb_tmp = os.path.join(models_dir, "cb_model.pkl.tmp")
    features_tmp = os.path.join(models_dir, "features.pkl.tmp")
    
    with open(lgb_tmp, "wb") as f:
        pickle.dump(lgb_model, f)
    with open(xgb_tmp, "wb") as f:
        pickle.dump(xgb_model, f)
    with open(cb_tmp, "wb") as f:
        pickle.dump(cb_model, f)
    with open(features_tmp, "wb") as f:
        pickle.dump(FEATURE_COLS, f)

    os.replace(lgb_tmp, os.path.join(models_dir, "lgb_model.pkl"))
    os.replace(xgb_tmp, os.path.join(models_dir, "xgb_model.pkl"))
    os.replace(cb_tmp, os.path.join(models_dir, "cb_model.pkl"))
    os.replace(features_tmp, os.path.join(models_dir, "features.pkl"))

    # Write manifest.json atomically
    import hashlib
    
    def get_sha256(path):
        if not os.path.exists(path):
            return ""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    manifest = {
        "timestamp": time.time(),
        "files": {
            "lgb_model.pkl": get_sha256(os.path.join(models_dir, "lgb_model.pkl")),
            "xgb_model.pkl": get_sha256(os.path.join(models_dir, "xgb_model.pkl")),
            "cb_model.pkl": get_sha256(os.path.join(models_dir, "cb_model.pkl")),
            "features.pkl": get_sha256(os.path.join(models_dir, "features.pkl"))
        }
    }
    manifest_tmp = os.path.join(models_dir, "manifest.json.tmp")
    with open(manifest_tmp, "w") as f:
        json.dump(manifest, f, indent=4)
    os.replace(manifest_tmp, os.path.join(models_dir, "manifest.json"))
    
    print("All models trained, saved atomically, and manifest.json written successfully!")

if __name__ == "__main__":
    train_ensemble()
