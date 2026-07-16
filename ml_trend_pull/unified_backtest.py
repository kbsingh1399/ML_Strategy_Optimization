import importlib.machinery
import importlib.util
import os
import sys
import numpy as np
import pandas as pd
import numba

# Load the parent compiled module under a unique name to prevent recursion
_base = os.path.dirname(os.path.abspath(__file__))
_proj = os.path.dirname(os.path.dirname(_base))
_strat_dir = os.path.join(_proj, "reversal_engine_3", "Strategies")
pyc_path = os.path.join(_strat_dir, "unified_backtest.pyc")

loader = importlib.machinery.SourcelessFileLoader("unified_backtest_compiled", pyc_path)
spec = importlib.util.spec_from_loader("unified_backtest_compiled", loader)
compiled_module = importlib.util.module_from_spec(spec)
sys.modules["unified_backtest_compiled"] = compiled_module
spec.loader.exec_module(compiled_module)

# Expose all attributes to this module's namespace
globals().update(compiled_module.__dict__)

# Apply directory path override to point to the correct Google Drive folder
local_backtest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backtesting_data"))
PQ_DIR = local_backtest_dir if os.path.exists(local_backtest_dir) else r"G:\My Drive\_Trading_Data\15m\parquet"
compiled_module.PQ_DIR = PQ_DIR

# Define custom indicators for ML_Trend_Pull
def zscore(s, w):
    return (s - s.rolling(w, min_periods=1).mean()) / s.rolling(w, min_periods=1).std().replace(0, 1e-10)

def custom_prep(df, btc_ref):
    if btc_ref is not None:
        cols_to_use = [c for c in btc_ref.columns if c not in df.columns]
        if cols_to_use:
            df = df.join(btc_ref[cols_to_use], how='left')
        
    df['atr'] = (df['High'] - df['Low']).rolling(14, min_periods=1).mean()
    df['cvd_delta'] = df['CVD'].diff(5)
    
    if 'btc_CVD' in df.columns:
        df['btc_cvd_mom'] = df['btc_CVD'].diff(2)
    else:
        df['btc_cvd_mom'] = 0.0
        
    # Trend indicators: EMA 200 and EMA 800
    df['ema_fast'] = df['Close'].ewm(span=200, min_periods=200).mean()
    df['ema_slow'] = df['Close'].ewm(span=800, min_periods=800).mean()
    
    # Trend state: 1 for uptrend, -1 for downtrend, 0 for neutral
    df['macro_score'] = (df['ema_fast'] - df['ema_slow']) / df['atr'].replace(0, 1e-10)
    df['macro'] = np.where(df['macro_score'] > 0.5, 1, np.where(df['macro_score'] < -0.5, -1, 0))
    
    # Pullback features:
    # 1. Price distance to short-term EMAs (8, 21, 50)
    df['ema_8'] = df['Close'].ewm(span=8, min_periods=1).mean()
    df['ema_21'] = df['Close'].ewm(span=21, min_periods=1).mean()
    df['ema_50'] = df['Close'].ewm(span=50, min_periods=1).mean()
    
    # Normalized pullback depths
    df['pull_ema8'] = (df['Close'] - df['ema_8']) / df['atr'].replace(0, 1e-10)
    df['pull_ema21'] = (df['Close'] - df['ema_21']) / df['atr'].replace(0, 1e-10)
    df['pull_ema50'] = (df['Close'] - df['ema_50']) / df['atr'].replace(0, 1e-10)
    
    # 2. Oscillators for oversold/overbought pullback levels
    # RSI(14)
    delta = df['Close'].diff()
    gain = (delta.clip(lower=0)).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Stochastic %K (14, 3)
    low_14 = df['Low'].rolling(14, min_periods=1).min()
    high_14 = df['High'].rolling(14, min_periods=1).max()
    df['stoch_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14).replace(0, 1e-10)
    
    # CVD z-scores
    for k in [4, 10, 20]:
        df[f'z_cvd_{k}'] = zscore(df['CVD'], k)
        if 'btc_CVD' in df.columns:
            df[f'z_btc_{k}'] = zscore(df['btc_CVD'], k)
        else:
            df[f'z_btc_{k}'] = 0.0
            
    df['vol_regime'] = zscore(df['atr'], 100)
    
    # Compile features list
    feats = [
        'macro', 'pull_ema8', 'pull_ema21', 'pull_ema50', 'rsi', 'stoch_k',
        'z_cvd_4', 'z_btc_4', 'z_btc_10', 'z_cvd_20', 'z_btc_20',
        'cvd_delta', 'btc_cvd_mom', 'vol_regime'
    ]
    
    # Optional orderbook features if present
    if 'Agg. OI' in df.columns:
        df['z_oi'] = zscore(pd.to_numeric(df['Agg. OI'], errors='coerce').ffill(), 100)
        feats.append('z_oi')
    if 'Long/Short Ratio (Account)' in df.columns:
        df['z_ls'] = zscore(pd.to_numeric(df['Long/Short Ratio (Account)'], errors='coerce').ffill(), 100)
        feats.append('z_ls')
    if 'Agg. Funding Rate' in df.columns:
        df['funding'] = pd.to_numeric(df['Agg. Funding Rate'], errors='coerce').fillna(0)
        feats.append('funding')
        
    df[feats] = df[feats].fillna(0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    # Clip values for numerical stability in XGBoost
    for f in feats:
        df[f] = df[f].clip(lower=-1e8, upper=1e8)
        
    return df, feats

# Fast label building using Numba
@numba.njit
def _build_labels_numba(c, h, l, a, direction, tp_mult, sl_mult, max_bars):
    n = len(c)
    labels = np.zeros(n)
    for i in range(n - max_bars - 1):
        atr_val = a[i]
        if np.isnan(atr_val) or atr_val == 0:
            continue
        entry = c[i]
        sl = entry - sl_mult * atr_val if direction == 1 else entry + sl_mult * atr_val
        tp = entry + tp_mult * atr_val if direction == 1 else entry - tp_mult * atr_val
        
        limit = min(i + max_bars + 1, n)
        for j in range(i + 1, limit):
            if direction == 1:
                if l[j] <= sl:
                    break
                if h[j] >= tp:
                    labels[i] = 1
                    break
            else:
                if h[j] >= sl:
                    break
                if l[j] <= tp:
                    labels[i] = 1
                    break
    return labels

def build_labels_fast(df, direction, tp_mult, sl_mult):
    c = df['Close'].values
    h = df['High'].values
    l = df['Low'].values
    a = df['atr'].values
    max_bars = globals().get('MAX_BARS', 96)
    return _build_labels_numba(c, h, l, a, direction, tp_mult, sl_mult, max_bars)

# Override functions
prep = custom_prep
compiled_module.prep = custom_prep
load_asset = compiled_module.load_asset
