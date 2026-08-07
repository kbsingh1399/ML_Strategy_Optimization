import importlib.machinery
import importlib.util
import os
import sys
import pandas as pd
import numpy as np

# Get the path to the compiled .pyc file in the same directory
_dir = os.path.dirname(os.path.abspath(__file__))
pyc_path = os.path.join(_dir, "unified_backtest.pyc")

# Load the compiled module under an internal name to prevent recursion
loader = importlib.machinery.SourcelessFileLoader("unified_backtest_compiled", pyc_path)
spec = importlib.util.spec_from_loader("unified_backtest_compiled", loader)
compiled_module = importlib.util.module_from_spec(spec)
sys.modules["unified_backtest_compiled"] = compiled_module
spec.loader.exec_module(compiled_module)

# Expose all attributes to this module's namespace
globals().update(compiled_module.__dict__)

# Apply directory path override to point to the correct Google Drive folder
_GDRIVE_PARQUET   = r"G:\My Drive\_Trading_Data\15m\parquet"
_LOCAL_PARQUET    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backtesting_data"))
PQ_DIR = _GDRIVE_PARQUET if os.path.exists(_GDRIVE_PARQUET) else _LOCAL_PARQUET
compiled_module.PQ_DIR = PQ_DIR

# --- CUSTOM REPLACEMENT LOADER & FEATURE ENGINEERING OVERRIDES ---

def zscore(s, w):
    return (s - s.rolling(w, min_periods=1).mean()) / s.rolling(w, min_periods=1).std().replace(0, 1e-10)

def custom_load_asset(symbol):
    # Try the default G Drive pattern
    summary_path = os.path.join(PQ_DIR, f"Master_{symbol}_15m_Final_Summary.parquet")
    footprint_path = os.path.join(PQ_DIR, f"Master_{symbol}_15m_Final_Footprint.parquet")
    
    # Fallback pattern if named differently
    if not os.path.exists(summary_path):
        summary_path = os.path.join(PQ_DIR, f"{symbol}_15m_summary.parquet")
        
    if not os.path.exists(summary_path):
        # Fallback to compiled load_asset if summary file cannot be found
        return compiled_module.load_asset(symbol)
        
    df_s = pd.read_parquet(summary_path)
    
    # Clean and parse TimeStamp column to ts_key
    ts_col_s = 'TimeStamp' if 'TimeStamp' in df_s.columns else 'Timestamp'
    df_s['ts_key'] = pd.to_datetime(df_s[ts_col_s], utc=True, errors='coerce').dt.tz_localize(None)
    
    if os.path.exists(footprint_path):
        df_f = pd.read_parquet(footprint_path)
        
        # Clean and parse Timestamp column in footprint to ts_key
        ts_col_f = 'TimeStamp' if 'TimeStamp' in df_f.columns else 'Timestamp'
        df_f['ts_key'] = pd.to_datetime(df_f[ts_col_f], utc=True, errors='coerce').dt.tz_localize(None)
        
        # Drop overlapping columns to prevent suffix duplication
        cols_to_drop = [c for c in ['Symbol', 'POC Price', 'Candle #', 'Timestamp', 'TimeStamp', 'time'] if c in df_f.columns]
        df_f_clean = df_f.drop(columns=cols_to_drop, errors='ignore')
        
        # Perform 1-to-1 inner merge on ts_key
        df = pd.merge(df_s, df_f_clean, on='ts_key', how='inner')
    else:
        df = df_s
        
    df = df.rename(columns={'ts_key': 'ts'})
    df = df.sort_values('ts').reset_index(drop=True)
    
    # Drop time column because it is mostly NaN
    if 'time' in df.columns:
        df = df.drop(columns=['time'])
        
    # Convert columns (excluding Symbol and ts) to numeric
    for c in df.columns:
        if c not in ['Symbol', 'ts']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
    df = df.set_index('ts')
    return df

def custom_prep(df, btc_ref):
    if btc_ref is not None:
        df = df.join(btc_ref, how='left')
        
    df['atr'] = (df['High'] - df['Low']).rolling(14, min_periods=1).mean()
    df['cvd_delta'] = df['CVD'].diff(5)
    
    if 'btc_CVD' in df.columns:
        df['btc_cvd_mom'] = df['btc_CVD'].diff(2)
    else:
        df['btc_cvd_mom'] = 0.0
        
    df['ema_fast'] = df['Close'].ewm(span=200, min_periods=200).mean()
    df['ema_slow'] = df['Close'].ewm(span=800, min_periods=800).mean()
    df['macro_score'] = (df['ema_fast'] - df['ema_slow']) / df['atr'].replace(0, 1e-10)
    df['macro'] = np.where(df['macro_score'] > 0.5, 1, np.where(df['macro_score'] < -0.5, -1, 0))
    
    feats = ['macro']
    
    for k in [4, 10, 20]:
        df[f'z_cvd_{k}'] = zscore(df['CVD'], k)
        if 'btc_CVD' in df.columns:
            df[f'z_btc_{k}'] = zscore(df['btc_CVD'], k)
        else:
            df[f'z_btc_{k}'] = 0.0
        feats.extend([f'z_cvd_{k}', f'z_btc_{k}'])
        
    df['vol_regime'] = zscore(df['atr'], 100)
    feats.extend(['cvd_delta', 'btc_cvd_mom', 'vol_regime'])
    
    if 'Agg. OI' in df.columns:
        df['z_oi'] = zscore(pd.to_numeric(df['Agg. OI'], errors='coerce').ffill(), 100)
    else:
        df['z_oi'] = 0.0
    feats.append('z_oi')
    
    if 'Long/Short Ratio (Account)' in df.columns:
        df['z_ls'] = zscore(pd.to_numeric(df['Long/Short Ratio (Account)'], errors='coerce').ffill(), 100)
    else:
        df['z_ls'] = 0.0
    feats.append('z_ls')
    
    if 'Agg. Funding Rate' in df.columns:
        df['funding'] = pd.to_numeric(df['Agg. Funding Rate'], errors='coerce').fillna(0)
    else:
        df['funding'] = 0.0
    feats.append('funding')
    
    if 'Agg. Liq Long' in df.columns:
        df['liq_long_5'] = pd.to_numeric(df['Agg. Liq Long'], errors='coerce').fillna(0).rolling(5, min_periods=1).sum()
    else:
        df['liq_long_5'] = 0.0
    feats.append('liq_long_5')
    
    if 'Agg. Liq Short' in df.columns:
        df['liq_short_5'] = pd.to_numeric(df['Agg. Liq Short'], errors='coerce').fillna(0).rolling(5, min_periods=1).sum()
    else:
        df['liq_short_5'] = 0.0
    feats.append('liq_short_5')
    
    df[feats] = df[feats].fillna(0)
    
    # --- ENHANCED FEATURES FROM ROADMAP ---
    df['ATR'] = df['atr'] # copy to match case in enhancements file
    
    # Import the drop-in enhancement functions
    # (Since we are local to alpha_squeezer_v17, we add _base directly)
    _base = os.path.dirname(os.path.abspath(__file__))
    if _base not in sys.path:
        sys.path.insert(0, _base)
        
    from alpha_squeezer_v17_enhancements import (
        add_absorption_features,
        add_depth_imbalance_features,
        add_velocity_features,
        add_regime_features
    )
    
    df = add_absorption_features(df)
    df = add_depth_imbalance_features(df)
    df = add_velocity_features(df)
    df = add_regime_features(df)
    
    # Numeric conversions/replacements for categories to keep it safe for LGBM
    if 'regime_state' in df.columns:
        df['regime_state'] = np.where(df['regime_score'] < -0.5, 0, 
                             np.where(df['regime_score'] > 0.5, 2, 1))
                             
    new_feats = [
        'abs_ratio_long', 'abs_ratio_short', 'absorption_flag_long', 'absorption_flag_short', 
        'whale_z', 'oi_diff_std_3', 'poc_stability',
        'depth_imbalance_z', 'avg_trade_size_bid_z', 'avg_trade_size_ask_z', 
        'poc_offset_atr', 'poc_location', 'cvd_divergence',
        'trade_velocity', 'trade_accel', 'delta_intensity', 'delta_intensity_roc', 
        'funding_accel', 'oi_velocity', 'liq_velocity_long', 'liq_velocity_short',
        'regime_score'
    ]
    
    # Check if footprint columns exist in the DataFrame before adding new features
    # If footprint file is not present, we will skip these to avoid KeyErrors
    existing_new_feats = [f for f in new_feats if f in df.columns]
    
    if existing_new_feats:
        df[existing_new_feats] = df[existing_new_feats].fillna(0)
        
    ga_selected = ['z_cvd_4', 'z_btc_4', 'z_btc_10', 'z_cvd_20', 'z_btc_20', 'btc_cvd_mom', 'z_oi', 'z_ls', 'liq_long_5']
    feats = [f for f in ga_selected if f in feats] + existing_new_feats
    
    # Safely replace infs with NaN to prevent XGBoost crashes
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Clip extremely large values to prevent float32 overflow during XGBoost training
    if existing_new_feats:
        for f in feats:
            df[f] = df[f].clip(lower=-1e8, upper=1e8)
    
    close_1h = df['Close'].rolling(4, min_periods=1).mean()
    ema_1h_50 = close_1h.ewm(span=50, min_periods=1).mean()
    ema_1h_200 = close_1h.ewm(span=200, min_periods=1).mean()
    atr_1h = (df['High'].rolling(4).max() - df['Low'].rolling(4).min()).rolling(14, min_periods=1).mean()
    df['macro_1h_score'] = (ema_1h_50 - ema_1h_200) / atr_1h.replace(0, 1e-10)
    df['macro_1h'] = np.where(df['macro_1h_score'] > 0.3, 1, np.where(df['macro_1h_score'] < -0.3, -1, 0))
    
    df['vol_regime_gate'] = np.where(df['vol_regime'] < -1.0, 0, 1)
    
    return df, feats

# Override the compiled functions
load_asset = custom_load_asset
compiled_module.load_asset = custom_load_asset

prep = custom_prep
compiled_module.prep = custom_prep
