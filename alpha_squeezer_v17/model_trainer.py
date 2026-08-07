"""
Alpha Squeezer V17 — Model Trainer
===================================
Walk-Forward LightGBM training pipeline.
Run weekly (or on-demand) to retrain models on latest data.
Uses exact parameters from final_wfo_params.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

_DIR = os.path.dirname(os.path.abspath(__file__))
_STRAT_DIR = os.path.dirname(_DIR)
_CORE_DIR = os.path.join(_STRAT_DIR, 'Core')
sys.path.insert(0, _STRAT_DIR)
sys.path.insert(0, _CORE_DIR)

from unified_backtest import load_asset, prep, NUM_LEAVES, MAX_BARS
from fast_optuna import build_labels_fast

_GDRIVE_PARQUET   = r"G:\My Drive\_Trading_Data\15m\parquet"
_LOCAL_PARQUET    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backtesting_data"))
DATA_DIR = _GDRIVE_PARQUET if os.path.exists(_GDRIVE_PARQUET) else _LOCAL_PARQUET
MODEL_DIR = os.path.join(_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Training parameters
# TRAIN_WINDOW_MONTHS will be loaded per-asset from optimal_windows.json
LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.05,
    'num_leaves': NUM_LEAVES,
    'verbose': -1,
    'n_jobs': 1,
}

def train_models():
    print(f"\n{'='*60}")
    print(f"ALPHA SQUEEZER V17 — LIVE MODEL TRAINER")
    print(f"{'='*60}")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Model dir: {MODEL_DIR}")
    
    target_assets = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT',
        'TRXUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'NEARUSDT', 'SUIUSDT',
        'XAGUSDT', 'XAUUSDT'
    ]
    print(f"  Assets to train: {len(target_assets)}")

    # Load BTC reference for CVD
    print(f"\n[1/4] Loading BTC reference...")
    btc = load_asset('BTCUSDT')
    if btc.empty:
        print("ERROR: BTC data required for cross-asset features.")
        return
    btc_ref = btc[['Close', 'CVD']].copy()
    btc_ref.columns = ['btc_Close', 'btc_CVD']

    configs_dir = os.path.join(_DIR, "agent5_configs")

    for sym in target_assets:
        print(f"\nProcessing {sym}...")
        cfg_path = os.path.join(configs_dir, f"{sym}.json")
        if not os.path.exists(cfg_path):
            print(f"  Skipping {sym} - no config file found at {cfg_path}")
            continue
            
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
            
        if cfg.get('score', -99999) <= 0:
            print(f"  Skipping {sym} - optimizer score is non-positive ({cfg.get('score')})")
            continue

        df = load_asset(sym)
        if df.empty:
            print(f"  Skipping {sym} - no data")
            continue
            
        sl_mult = cfg.get('sl_mult', 1.0)
        tp_mult = cfg.get('tp_mult', 3.0)
        window_months = cfg.get('window_size', 3)
        
        # 2. Feature engineering
        df, feats = prep(df.copy(), btc_ref)
        df = df.reset_index()
        
        # 3. Build labels using Optuna params
        df['lab_long'] = build_labels_fast(df, 1, tp_mult, sl_mult)
        df['lab_short'] = build_labels_fast(df, -1, tp_mult, sl_mult)
        
        # 4. Training Window (Dynamic Optimal Window)
        cutoff_start = df['ts'].max() - pd.DateOffset(months=window_months)
        train_mask = (df['ts'] >= cutoff_start)
        
        X_train = df.loc[train_mask, feats]
        y_train_long = df.loc[train_mask, 'lab_long']
        y_train_short = df.loc[train_mask, 'lab_short']
        
        if len(X_train) < 500:
            print(f"  {sym}: Insufficient training data ({len(X_train)} rows).")
            continue
            
        print(f"  Training on {len(X_train)} bars from {cutoff_start.date()} to {df['ts'].max().date()} (Window: {window_months}M)...")
        
        # 5. Train & Save Models
        # 5a. LightGBM
        ml = lgb.train(LGB_PARAMS, lgb.Dataset(X_train, y_train_long), 120)
        ms = lgb.train(LGB_PARAMS, lgb.Dataset(X_train, y_train_short), 120)
        
        # 5b. XGBoost
        xgb_l = xgb.XGBClassifier(n_estimators=120, learning_rate=0.05, max_depth=5, eval_metric='logloss', tree_method='hist', device='cuda', verbosity=0)
        xgb_s = xgb.XGBClassifier(n_estimators=120, learning_rate=0.05, max_depth=5, eval_metric='logloss', tree_method='hist', device='cuda', verbosity=0)
        xgb_l.fit(X_train, y_train_long)
        xgb_s.fit(X_train, y_train_short)
        
        # 5c. CatBoost
        cb_l = CatBoostClassifier(iterations=120, learning_rate=0.05, depth=5, verbose=0, thread_count=1)
        cb_s = CatBoostClassifier(iterations=120, learning_rate=0.05, depth=5, verbose=0, thread_count=1)
        cb_l.fit(X_train, y_train_long)
        cb_s.fit(X_train, y_train_short)
        
        # Save all 3 models to tmp first, then replace atomically
        lgb_long_tmp = os.path.join(MODEL_DIR, f"{sym}_long_lgb.txt.tmp")
        lgb_short_tmp = os.path.join(MODEL_DIR, f"{sym}_short_lgb.txt.tmp")
        xgb_long_tmp = os.path.join(MODEL_DIR, f"{sym}_long_xgb_tmp.json")
        xgb_short_tmp = os.path.join(MODEL_DIR, f"{sym}_short_xgb_tmp.json")
        cb_long_tmp = os.path.join(MODEL_DIR, f"{sym}_long_cb.cbm.tmp")
        cb_short_tmp = os.path.join(MODEL_DIR, f"{sym}_short_cb.cbm.tmp")

        ml.save_model(lgb_long_tmp)
        ms.save_model(lgb_short_tmp)
        xgb_l.save_model(xgb_long_tmp)
        xgb_s.save_model(xgb_short_tmp)
        cb_l.save_model(cb_long_tmp)
        cb_s.save_model(cb_short_tmp)
        
        os.replace(lgb_long_tmp, os.path.join(MODEL_DIR, f"{sym}_long_lgb.txt"))
        os.replace(lgb_short_tmp, os.path.join(MODEL_DIR, f"{sym}_short_lgb.txt"))
        os.replace(xgb_long_tmp, os.path.join(MODEL_DIR, f"{sym}_long_xgb.json"))
        os.replace(xgb_short_tmp, os.path.join(MODEL_DIR, f"{sym}_short_xgb.json"))
        os.replace(cb_long_tmp, os.path.join(MODEL_DIR, f"{sym}_long_cb.cbm"))
        os.replace(cb_short_tmp, os.path.join(MODEL_DIR, f"{sym}_short_cb.cbm"))
        print(f"  Saved {sym} ensemble models to {MODEL_DIR}")
        
    # Generate and write manifest.json atomically at the very end
    import time
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
        "files": {}
    }
    for sym in target_assets:
        cfg_path = os.path.join(configs_dir, f"{sym}.json")
        if not os.path.exists(cfg_path):
            continue
        lgb_long_path = os.path.join(MODEL_DIR, f"{sym}_long_lgb.txt")
        if os.path.exists(lgb_long_path):
            for suffix in ["long_lgb.txt", "short_lgb.txt", "long_xgb.json", "short_xgb.json", "long_cb.cbm", "short_cb.cbm"]:
                fn = f"{sym}_{suffix}"
                manifest["files"][fn] = get_sha256(os.path.join(MODEL_DIR, fn))

    manifest_tmp = os.path.join(MODEL_DIR, "manifest.json.tmp")
    with open(manifest_tmp, "w") as f:
        json.dump(manifest, f, indent=4)
    os.replace(manifest_tmp, os.path.join(MODEL_DIR, "manifest.json"))
    print("\n[SUCCESS] All live models trained and manifest.json written.")

if __name__ == '__main__':
    train_models()
