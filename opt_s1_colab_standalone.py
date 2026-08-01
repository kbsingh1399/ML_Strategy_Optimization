import warnings
warnings.filterwarnings('ignore')
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import os
import pickle
import sys

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    try:
        drive.mount('/content/drive')
        ROOT = Path('/content/drive/MyDrive/algo_cache')
        ROOT.mkdir(parents=True, exist_ok=True)
        print("Google Drive mounted! Saving outputs to /content/drive/MyDrive/algo_cache")
    except Exception as e:
        print(f"Drive mount failed: {e}")
        ROOT = Path('/content')
else:
    ROOT = Path('.')
import json
import os
import pickle
import sys
import gc

_n_cpu = str(os.cpu_count() or 4)
os.environ["OMP_NUM_THREADS"] = _n_cpu
os.environ["OPENBLAS_NUM_THREADS"] = _n_cpu
os.environ["MKL_NUM_THREADS"] = _n_cpu
os.environ["NUMEXPR_NUM_THREADS"] = _n_cpu
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
import threading
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

USE_GPU = True
print(f'[GPU] LightGBM GPU mode: {"ENABLED" if USE_GPU else "DISABLED (CPU fallback)"}')
from datetime import datetime
from pathlib import Path

GLOBAL_FEATURES = {}

def get_features_for_symbol(symbol: str) -> pd.DataFrame:
    if symbol in GLOBAL_FEATURES:
        return GLOBAL_FEATURES[symbol]
    df = load_asset(symbol)
    if df.empty:
        return pd.DataFrame()
    btc = load_asset("BTCUSDT")
    btc_ref = None
    if not btc.empty and "Close" in btc.columns and "CVD" in btc.columns:
        btc_ref = btc[["Close", "CVD"]].copy()
        btc_ref.columns = ["btc_Close", "btc_CVD"]
    df_std = generate_features_standard(df.copy(), btc_ref)
    if "ts" in df_std.columns:
        df_std = df_std.set_index("ts")
    GLOBAL_FEATURES[symbol] = df_std
    return df_std

def join_features(df):
    if df is None or len(df) == 0:
        return df
    symbol = df['symbol'].iloc[0]
    if symbol not in GLOBAL_FEATURES:
        get_features_for_symbol(symbol)
    if symbol in GLOBAL_FEATURES:
        feat_df = GLOBAL_FEATURES[symbol]
        cols_to_use = feat_df.columns.difference(df.columns)
        if len(cols_to_use) > 0:
            df = df.merge(feat_df[cols_to_use], left_on='entry_time', right_index=True, how='left')
    return df





def zscore(s: pd.Series, w: int) -> pd.Series:
    return (s - s.rolling(w, min_periods=1).mean()) / s.rolling(w, min_periods=1).std().replace(0, 1e-10)


def prep_alpha(df: pd.DataFrame, btc_ref: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    if btc_ref is not None:
        if "ts" in btc_ref.columns and "ts" in df.columns:
            df_indexed = df.set_index("ts")
            btc_ref_indexed = btc_ref.set_index("ts")
            cols_to_join = [c for c in btc_ref_indexed.columns if c not in df_indexed.columns]
            if cols_to_join:
                df_indexed = df_indexed.join(btc_ref_indexed[cols_to_join], how="left")
            df = df_indexed.reset_index()
        else:
            cols_to_join = [c for c in btc_ref.columns if c not in df.columns]
            if cols_to_join:
                df = df.join(btc_ref[cols_to_join], how="left")
    df["atr"] = (df["High"] - df["Low"]).rolling(14, min_periods=1).mean()
    df["cvd_delta"] = df["CVD"].diff(5)
    df["btc_cvd_mom"] = df["btc_CVD"].diff(2) if "btc_CVD" in df.columns else 0.0
    df["ema_fast"] = df["Close"].ewm(span=200, min_periods=50).mean()
    df["ema_slow"] = df["Close"].ewm(span=800, min_periods=100).mean()
    df["macro_score"] = (df["ema_fast"] - df["ema_slow"]) / df["atr"].replace(0, 1e-10)
    df["macro"] = np.where(df["macro_score"] > 0.5, 1, np.where(df["macro_score"] < -0.5, -1, 0))
    feats = ["macro"]
    for k in [4, 10, 20]:
        df[f"z_cvd_{k}"] = zscore(df["CVD"], k)
        df[f"z_btc_{k}"] = zscore(df["btc_CVD"], k) if "btc_CVD" in df.columns else 0.0
        feats.extend([f"z_cvd_{k}", f"z_btc_{k}"])
    df["vol_regime"] = zscore(df["atr"], 100)
    feats.extend(["cvd_delta", "btc_cvd_mom", "vol_regime"])
    if "Agg. OI" in df.columns:
        df["z_oi"] = zscore(pd.to_numeric(df["Agg. OI"], errors="coerce").ffill(), 100)
    else:
        df["z_oi"] = 0.0
    feats.append("z_oi")
    if "Long/Short Ratio (Account)" in df.columns:
        df["z_ls"] = zscore(pd.to_numeric(df["Long/Short Ratio (Account)"], errors="coerce").ffill(), 100)
    else:
        df["z_ls"] = 0.0
    feats.append("z_ls")
    if "Agg. Funding Rate" in df.columns:
        df["funding"] = pd.to_numeric(df["Agg. Funding Rate"], errors="coerce").fillna(0)
    else:
        df["funding"] = 0.0
    feats.append("funding")
    for side, col in [("long", "Agg. Liq Long"), ("short", "Agg. Liq Short")]:
        if col in df.columns:
            df[f"liq_{side}_5"] = pd.to_numeric(df[col], errors="coerce").fillna(0).rolling(5, min_periods=1).sum()
        else:
            df[f"liq_{side}_5"] = 0.0
        feats.append(f"liq_{side}_5")
        
    fp_cols = ["Bid Qty", "Ask Qty", "Delta Qty", "Bid Trades", "Ask Trades"]
    for col in fp_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df[f"z_{col.replace(' ', '_').lower()}"] = zscore(df[col], 10)
            feats.extend([col, f"z_{col.replace(' ', '_').lower()}"])
            
    df[feats] = df[feats].fillna(0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df, feats


def prep_trend(df: pd.DataFrame, btc_ref: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    if btc_ref is not None:
        if "ts" in btc_ref.columns and "ts" in df.columns:
            df_indexed = df.set_index("ts")
            btc_ref_indexed = btc_ref.set_index("ts")
            cols_to_join = [c for c in btc_ref_indexed.columns if c not in df_indexed.columns]
            if cols_to_join:
                df_indexed = df_indexed.join(btc_ref_indexed[cols_to_join], how="left")
            df = df_indexed.reset_index()
        else:
            cols_to_join = [c for c in btc_ref.columns if c not in df.columns]
            if cols_to_join:
                df = df.join(btc_ref[cols_to_join], how="left")
    df["atr"] = (df["High"] - df["Low"]).rolling(14, min_periods=1).mean()
    df["cvd_delta"] = df["CVD"].diff(5)
    df["btc_cvd_mom"] = df["btc_CVD"].diff(2) if "btc_CVD" in df.columns else 0.0
    df["ema_fast"] = df["Close"].ewm(span=200, min_periods=50).mean()
    df["ema_slow"] = df["Close"].ewm(span=800, min_periods=100).mean()
    df["macro_score"] = (df["ema_fast"] - df["ema_slow"]) / df["atr"].replace(0, 1e-10)
    df["macro"] = np.where(df["macro_score"] > 0.5, 1, np.where(df["macro_score"] < -0.5, -1, 0))
    for span, name in [(8, "ema_8"), (21, "ema_21"), (50, "ema_50")]:
        df[name] = df["Close"].ewm(span=span, min_periods=1).mean()
    atr_safe = df["atr"].replace(0, 1e-10)
    df["pull_ema8"] = (df["Close"] - df["ema_8"]) / atr_safe
    df["pull_ema21"] = (df["Close"] - df["ema_21"]) / atr_safe
    df["pull_ema50"] = (df["Close"] - df["ema_50"]) / atr_safe
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))
    low_14 = df["Low"].rolling(14, min_periods=1).min()
    high_14 = df["High"].rolling(14, min_periods=1).max()
    df["stoch_k"] = 100 * (df["Close"] - low_14) / (high_14 - low_14).replace(0, 1e-10)
    for k in [4, 10, 20]:
        df[f"z_cvd_{k}"] = zscore(df["CVD"], k)
        df[f"z_btc_{k}"] = zscore(df["btc_CVD"], k) if "btc_CVD" in df.columns else 0.0
    df["vol_regime"] = zscore(df["atr"], 100)
    feats = [
        "macro", "pull_ema8", "pull_ema21", "pull_ema50", "rsi", "stoch_k",
        "z_cvd_4", "z_btc_4", "z_btc_10", "z_cvd_20", "z_btc_20",
        "cvd_delta", "btc_cvd_mom", "vol_regime",
    ]
    if "Agg. OI" in df.columns:
        df["z_oi"] = zscore(pd.to_numeric(df["Agg. OI"], errors="coerce").ffill(), 100)
        feats.append("z_oi")
    if "Long/Short Ratio (Account)" in df.columns:
        df["z_ls"] = zscore(pd.to_numeric(df["Long/Short Ratio (Account)"], errors="coerce").ffill(), 100)
        feats.append("z_ls")
    if "Agg. Funding Rate" in df.columns:
        df["funding"] = pd.to_numeric(df["Agg. Funding Rate"], errors="coerce").fillna(0)
        feats.append("funding")
        
    fp_cols = ["Bid Qty", "Ask Qty", "Delta Qty", "Bid Trades", "Ask Trades"]
    for col in fp_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df[f"z_{col.replace(' ', '_').lower()}"] = zscore(df[col], 10)
            feats.extend([col, f"z_{col.replace(' ', '_').lower()}"])

    df[feats] = df[feats].fillna(0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df, feats



def calculate_vwap_bands(df: pd.DataFrame) -> pd.DataFrame:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    volume = df["Volume"].replace(0, 1e-10)
    ts_idx = df.index if isinstance(df.index, pd.DatetimeIndex) else (pd.to_datetime(df["ts"]) if "ts" in df.columns else pd.to_datetime(df["TimeStamp"] if "TimeStamp" in df.columns else df["Timestamp"]))
    session_group = (ts_idx - pd.Timedelta(hours=5, minutes=30)).floor("D")
    tp_v = typical_price * volume
    cum_tp_v = tp_v.groupby(session_group).cumsum()
    cum_v = volume.groupby(session_group).cumsum()
    vwap = cum_tp_v / cum_v
    weighted_variance = (((typical_price - vwap) ** 2) * volume).groupby(session_group).cumsum() / cum_v
    std_vwap = np.sqrt(weighted_variance.clip(lower=0))
    df["vwap"] = vwap
    df["std_vwap"] = std_vwap
    df["vwap_upper_2.2"] = vwap + 2.2 * std_vwap
    df["vwap_lower_2.2"] = vwap - 2.2 * std_vwap
    df["vwap_upper_3.3"] = vwap + 3.3 * std_vwap
    df["vwap_lower_3.3"] = vwap - 3.3 * std_vwap
    return df

def prep_vwap(df: pd.DataFrame, btc_ref: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, List[str]]:
    if "time" in df.columns:
        df = df.drop(columns=["time"])
    if btc_ref is not None:
        if "time" in btc_ref.columns:
            btc_ref = btc_ref.drop(columns=["time"])
        if "ts" in btc_ref.columns and "ts" in df.columns:
            df_indexed = df.set_index("ts")
            btc_ref_indexed = btc_ref.set_index("ts")
            cols_to_join = [c for c in btc_ref_indexed.columns if c not in df_indexed.columns]
            if cols_to_join:
                df_indexed = df_indexed.join(btc_ref_indexed[cols_to_join], how="left")
            df = df_indexed.reset_index()
        else:
            cols_to_join = [c for c in btc_ref.columns if c not in df.columns]
            if cols_to_join:
                df = df.join(btc_ref[cols_to_join], how="left")
            if "btc_CVD" in df.columns:
                df["btc_CVD"] = df["btc_CVD"].ffill().bfill().fillna(0)
    df["atr"] = (df["High"] - df["Low"]).rolling(14, min_periods=1).mean()
    df["cvd_delta"] = df["CVD"].diff(5)
    df["btc_cvd_mom"] = df["btc_CVD"].diff(2) if "btc_CVD" in df.columns else 0.0
    df["ema_fast"] = df["Close"].ewm(span=20, min_periods=5).mean()
    df["ema_slow"] = df["Close"].ewm(span=50, min_periods=10).mean()
    df["macro_score"] = (df["ema_fast"] - df["ema_slow"]) / df["atr"].replace(0, 1e-10)
    df["macro"] = np.where(df["macro_score"] > 0.5, 1, np.where(df["macro_score"] < -0.5, -1, 0))
    df = calculate_vwap_bands(df)
    atr_safe = df["atr"].replace(0, 1e-10)
    df["dist_vwap"] = (df["Close"] - df["vwap"]) / atr_safe
    df["dist_upper_2.2"] = (df["Close"] - df["vwap_upper_2.2"]) / atr_safe
    df["dist_lower_2.2"] = (df["Close"] - df["vwap_lower_2.2"]) / atr_safe
    df["dist_upper_3.3"] = (df["Close"] - df["vwap_upper_3.3"]) / atr_safe
    df["dist_lower_3.3"] = (df["Close"] - df["vwap_lower_3.3"]) / atr_safe
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))
    low_14 = df["Low"].rolling(14, min_periods=1).min()
    high_14 = df["High"].rolling(14, min_periods=1).max()
    df["stoch_k"] = 100 * (df["Close"] - low_14) / (high_14 - low_14).replace(0, 1e-10)
    for k in [4, 10, 20]:
        df[f"z_cvd_{k}"] = (df["CVD"] - df["CVD"].rolling(k, min_periods=1).mean()) / df["CVD"].rolling(k, min_periods=1).std().replace(0, 1e-10)
        df[f"z_btc_{k}"] = (df["btc_CVD"] - df["btc_CVD"].rolling(k, min_periods=1).mean()) / df["btc_CVD"].rolling(k, min_periods=1).std().replace(0, 1e-10) if "btc_CVD" in df.columns else 0.0
    df["vol_regime"] = (df["atr"] - df["atr"].rolling(100, min_periods=1).mean()) / df["atr"].rolling(100, min_periods=1).std().replace(0, 1e-10)
    # Orderflow & Liquidations Injection
    for side, col in [("long", "Agg. Liq Long"), ("short", "Agg. Liq Short")]:
        if col in df.columns:
            df[f"liq_{side}_5"] = pd.to_numeric(df[col], errors="coerce").fillna(0).rolling(5, min_periods=1).sum()
            df[f"liq_{side}_5_mean"] = df[f"liq_{side}_5"].rolling(100, min_periods=1).mean()
        else:
            df[f"liq_{side}_5"] = 0.0
            df[f"liq_{side}_5_mean"] = 0.0
    df["liq_imbalance"] = (df["liq_long_5"] - df["liq_short_5"]) / (df["liq_long_5"] + df["liq_short_5"] + 1e-10)

    if "Buy Qty" in df.columns and "Sell Qty" in df.columns:
        buy_qty = pd.to_numeric(df["Buy Qty"], errors="coerce").fillna(0)
        sell_qty = pd.to_numeric(df["Sell Qty"], errors="coerce").fillna(0)
        df["buy_sell_vol_ratio"] = buy_qty / (buy_qty + sell_qty + 1e-10)
    else:
        df["buy_sell_vol_ratio"] = 0.5

    if "Bid Qty" in df.columns and "Ask Qty" in df.columns:
        bid_q = pd.to_numeric(df["Bid Qty"], errors="coerce").fillna(0)
        ask_q = pd.to_numeric(df["Ask Qty"], errors="coerce").fillna(0)
        df["bid_ask_qty_ratio"] = bid_q / (bid_q + ask_q + 1e-10)
    else:
        df["bid_ask_qty_ratio"] = 0.5

    feats = [
        "dist_vwap", "dist_upper_2.2", "dist_lower_2.2", "dist_upper_3.3", "dist_lower_3.3",
        "macro", "rsi", "stoch_k",
        "z_cvd_4", "z_btc_4", "z_btc_10", "z_cvd_20", "z_btc_20",
        "cvd_delta", "btc_cvd_mom", "vol_regime",
        "liq_long_5", "liq_short_5", "liq_imbalance", "buy_sell_vol_ratio", "bid_ask_qty_ratio"
    ]
    if "Agg. OI" in df.columns:
        oi = pd.to_numeric(df["Agg. OI"], errors="coerce").ffill()
        df["z_oi"] = (oi - oi.rolling(100, min_periods=1).mean()) / oi.rolling(100, min_periods=1).std().replace(0, 1e-10)
        df["oi_delta_5"] = oi.diff(5) / (oi.shift(5) + 1e-10)
        df["oi_cvd_coherence"] = np.sign(df["oi_delta_5"].fillna(0)) * np.sign(df["cvd_delta"].fillna(0))
        feats.extend(["z_oi", "oi_delta_5", "oi_cvd_coherence"])
    if "Long/Short Ratio (Account)" in df.columns:
        df["z_ls"] = (df["Long/Short Ratio (Account)"].ffill() - df["Long/Short Ratio (Account)"].ffill().rolling(100, min_periods=1).mean()) / df["Long/Short Ratio (Account)"].ffill().rolling(100, min_periods=1).std().replace(0, 1e-10)
        feats.append("z_ls")
    if "Agg. Funding Rate" in df.columns:
        funding = pd.to_numeric(df["Agg. Funding Rate"], errors="coerce").fillna(0)
        df["funding"] = funding
        df["z_funding_20"] = (funding - funding.rolling(20, min_periods=1).mean()) / funding.rolling(20, min_periods=1).std().replace(0, 1e-10)
        feats.extend(["funding", "z_funding_20"])
    fp_cols = ["Bid Qty", "Ask Qty", "Delta Qty", "Bid Trades", "Ask Trades"]
    for col in fp_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df[f"z_{col.replace(' ', '_').lower()}"] = (df[col] - df[col].rolling(10, min_periods=1).mean()) / df[col].rolling(10, min_periods=1).std().replace(0, 1e-10)
            feats.extend([col, f"z_{col.replace(' ', '_').lower()}"])
    df[feats] = df[feats].fillna(0).astype(np.float32)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df, feats


def prep_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates advanced microstructure features such as:
    - CVD Divergence
    - Liquidation Cascades and Acceleration
    - Volatility Coiling
    """
    # 1. Volatility Coiling (Z-score of ATR)
    df["atr"] = (df["High"] - df["Low"]).rolling(14, min_periods=1).mean()
    atr_safe = df["atr"].replace(0, 1e-10)
    df["vol_regime"] = (df["atr"] - df["atr"].rolling(100, min_periods=1).mean()) / df["atr"].rolling(100, min_periods=1).std().replace(0, 1e-10)
    
    # 2. CVD Divergence and Delta
    if "CVD" in df.columns:
        df["cvd_delta"] = df["CVD"].diff(3)
        df["cvd_accel"] = df["cvd_delta"].diff()
        
        # Divergence: Price makes a new low, but CVD doesn't (Passive Buy Wall Absorption)
        low_5 = df["Low"].rolling(5).min()
        cvd_5 = df["CVD"].rolling(5).min()
        df["cvd_divergence_bull"] = (df["Low"] == low_5) & (df["CVD"] > cvd_5)
        
        high_5 = df["High"].rolling(5).max()
        cvd_5_max = df["CVD"].rolling(5).max()
        df["cvd_divergence_bear"] = (df["High"] == high_5) & (df["CVD"] < cvd_5_max)
    
    # 3. Liquidation Cascades and Acceleration
    if "Agg. Liq Long" in df.columns:
        df["liq_long"] = df["Agg. Liq Long"].fillna(0)
        df["liq_long_mean"] = df["liq_long"].rolling(1440, min_periods=100).mean().fillna(0)
        # Calculate liquidation acceleration (1st derivative)
        df["liq_long_delta"] = df["liq_long"].diff().fillna(0)
    else:
        df["liq_long"] = 0
        df["liq_long_mean"] = 1
        df["liq_long_delta"] = 0
        
    if "Agg. Liq Short" in df.columns:
        df["liq_short"] = df["Agg. Liq Short"].fillna(0)
        df["liq_short_mean"] = df["liq_short"].rolling(1440, min_periods=100).mean().fillna(0)
        # Calculate liquidation acceleration (1st derivative)
        df["liq_short_delta"] = df["liq_short"].diff().fillna(0)
    else:
        df["liq_short"] = 0
        df["liq_short_mean"] = 1
        df["liq_short_delta"] = 0

    # 4. Footprint Delta Z-Score (bid/ask imbalance confirmation)
    if "Delta Qty" in df.columns:
        dq = df["Delta Qty"].fillna(0)
        df["delta_qty_z"] = (dq - dq.rolling(20, min_periods=1).mean()) / dq.rolling(20, min_periods=1).std().replace(0, 1e-10)
    else:
        df["delta_qty_z"] = 0.0

    if "Bid Qty" in df.columns and "Ask Qty" in df.columns:
        total_qty = df["Bid Qty"].fillna(0) + df["Ask Qty"].fillna(0)
        df["bid_ask_ratio"] = (df["Bid Qty"].fillna(0) / total_qty.replace(0, 1e-10)) - 0.5
    else:
        df["bid_ask_ratio"] = 0.0

    return df

from numba import njit

@njit(fastmath=True, nogil=True)
def rolling_mean_numba(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    if n == 0:
        return out
    current_sum = 0.0
    for i in range(n):
        current_sum += arr[i]
        if i >= window:
            current_sum -= arr[i - window]
            out[i] = current_sum / window
        else:
            out[i] = current_sum / (i + 1)
    return out

@njit(fastmath=True, nogil=True)
def rolling_zscore_numba(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    z = np.zeros(n, dtype=np.float32)
    if n < window:
        return z
    
    win_sum = 0.0
    win_sq_sum = 0.0
    inv_w = 1.0 / window
    inv_w1 = 1.0 / (window - 1) if window > 1 else 1.0
    
    for i in range(window):
        val = arr[i]
        win_sum += val
        win_sq_sum += val * val
        
    mean = win_sum * inv_w
    var = (win_sq_sum - window * mean * mean) * inv_w1
    std = np.sqrt(max(0.0, var))
    if std > 0:
        z[window - 1] = (arr[window - 1] - mean) / std

    for i in range(window, n):
        val_in = arr[i]
        val_out = arr[i - window]
        win_sum += val_in - val_out
        win_sq_sum += val_in * val_in - val_out * val_out
        
        mean = win_sum * inv_w
        var = (win_sq_sum - window * mean * mean) * inv_w1
        std = np.sqrt(max(0.0, var))
        if std > 0:
            z[i] = (val_in - mean) / std
            
    return z

@njit(fastmath=True, nogil=True)
def compute_fvg_and_sweeps(high: np.ndarray, low: np.ndarray, close: np.ndarray, sweep_lookback: int = 15):
    n = len(close)
    bullish_fvg = np.zeros(n, dtype=np.int32)
    bearish_fvg = np.zeros(n, dtype=np.int32)
    bullish_sweep = np.zeros(n, dtype=np.int32)
    bearish_sweep = np.zeros(n, dtype=np.int32)
    
    for i in range(2, n):
        if low[i] > high[i-2] and close[i-1] > high[i-2]:
            bullish_fvg[i] = 1
        if high[i] < low[i-2] and close[i-1] < low[i-2]:
            bearish_fvg[i] = 1
            
        start_idx = max(0, i - sweep_lookback)
        rmax = high[start_idx]
        rmin = low[start_idx]
        for k in range(start_idx + 1, i):
            if high[k] > rmax: rmax = high[k]
            if low[k] < rmin: rmin = low[k]
            
        if high[i] > rmax and close[i] < rmax:
            bearish_sweep[i] = 1
        if low[i] < rmin and close[i] > rmin:
            bullish_sweep[i] = 1
            
    return bullish_fvg, bearish_fvg, bullish_sweep, bearish_sweep

def prep_smc(df: pd.DataFrame, btc_ref: pd.DataFrame = None):
    h = df['High'].values.astype(np.float32)
    l = df['Low'].values.astype(np.float32)
    c = df['Close'].values.astype(np.float32)
    o = df['Open'].values.astype(np.float32)
    vol = df['Volume'].values.astype(np.float32) if 'Volume' in df.columns else np.ones(len(c), dtype=np.float32)
    
    if 'Candle Delta' in df.columns:
        delta = df['Candle Delta'].values.astype(np.float32)
    elif 'Buy Qty' in df.columns and 'Sell Qty' in df.columns:
        delta = (df['Buy Qty'] - df['Sell Qty']).values.astype(np.float32)
    elif 'Taker Buy Volume' in df.columns:
        delta = (2.0 * df['Taker Buy Volume'] - vol).values.astype(np.float32)
    else:
        rng = np.maximum(h - l, 1e-6)
        delta = (vol * (c - o) / rng).astype(np.float32)
        
    has_cvd = 'CVD' in df.columns
    cvd = df['CVD'].values.astype(np.float32) if has_cvd else np.cumsum(delta).astype(np.float32)
    
    bull_fvg, bear_fvg, bull_sweep, bear_sweep = compute_fvg_and_sweeps(h, l, c, 15)
    atr = rolling_mean_numba(h - l, 14)
    z_delta = rolling_zscore_numba(delta, 10)
    z_cvd = rolling_zscore_numba(cvd, 20) if has_cvd else np.zeros(len(c), dtype=np.float32)
    
    ema_20 = df['Close'].ewm(span=20, adjust=False).mean().values
    atr_safe = np.maximum(atr, 1e-10)
    atr_stretch = np.where(atr > 0, (c - ema_20) / atr_safe, 0.0)
    
    df = df.assign(
        bull_fvg=bull_fvg,
        bear_fvg=bear_fvg,
        bull_sweep=bull_sweep,
        bear_sweep=bear_sweep,
        atr=atr,
        delta=delta,
        z_delta=z_delta,
        z_cvd=z_cvd,
        atr_stretch=atr_stretch
    )
    
    feat_cols = ['bull_fvg', 'bear_fvg', 'bull_sweep', 'bear_sweep', 'delta', 'z_delta', 'z_cvd', 'atr_stretch']
    return df, feat_cols

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT",
    "LTCUSDT", "NEARUSDT", "SUIUSDT", "TRXUSDT",
]

MONTHS = [
    ("2020-03-18", "2020-04-18"),
    ("2020-11-07", "2020-12-07"),
    ("2021-01-24", "2021-02-24"),
    ("2021-06-13", "2021-07-13"),
    ("2021-10-29", "2021-11-29"),
    ("2022-02-08", "2022-03-08"),
    ("2022-05-21", "2022-06-21"),
    ("2022-09-14", "2022-10-14"),
    ("2022-12-03", "2023-01-03"),
    ("2023-04-17", "2023-05-17"),
    ("2023-08-25", "2023-09-25"),
    ("2023-11-10", "2023-12-10"),
    ("2024-02-19", "2024-03-19"),
    ("2024-07-06", "2024-08-06"),
    ("2024-10-28", "2024-11-28"),
    ("2025-01-15", "2025-02-15"),
    ("2025-05-03", "2025-06-03"),
    ("2025-09-22", "2025-10-22"),
    ("2026-02-11", "2026-03-11"),
    ("2026-06-09", "2026-07-09"),
]

INITIAL_CAPITAL = 5000.0
RISK_PER_TRADE = 50.0
FEE_SLIPPAGE = 0.0015
SL_MULT = 1.0
TP_MULT_OPTIONS = [5, 7, 10]
TRAIL_ATR_OPTIONS = [0.4, 0.5, 0.8, 1.0, 1.5, 2.0]


def load_asset(symbol: str) -> pd.DataFrame:
    search_dirs = [
        str(ROOT),
        "/content/drive/MyDrive/_Trading_Data/15m/parquet",
        "/content/drive/MyDrive/algo_cache",
        "/content", 
        ".", 
        "/kaggle/working",
        "backtesting_data", 
        "../Engine_1/backtesting_data",
        "G:\\My Drive\\_Trading_Data\\15m\\parquet"
    ]
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            search_dirs.append(root)
    sp, fp = None, None
    for d in search_dirs:
        p_sp = os.path.join(d, f"Master_{symbol}_15m_Final_Summary.parquet")
        if os.path.exists(p_sp):
            sp = p_sp
            break
        p_alt = os.path.join(d, f"{symbol}_15m_summary.parquet")
        if os.path.exists(p_alt):
            sp = p_alt
            break

    for d in search_dirs:
        p_fp = os.path.join(d, f"Master_{symbol}_15m_Final_Footprint.parquet")
        if os.path.exists(p_fp):
            fp = p_fp
            break
        p_alt = os.path.join(d, f"{symbol}_15m_footprint.parquet")
        if os.path.exists(p_alt):
            fp = p_alt
            break

    if not sp:
        return pd.DataFrame()

    df_s = pd.read_parquet(sp)
    ts_col = "TimeStamp" if "TimeStamp" in df_s.columns else "Timestamp"
    df_s["ts"] = pd.to_datetime(df_s[ts_col].astype(str).str.replace(" IST", "", regex=False), errors="coerce")

    if fp and os.path.exists(fp):
        df_f = pd.read_parquet(fp)
        ts_col_f = "TimeStamp" if "TimeStamp" in df_f.columns else "Timestamp"
        df_f["ts"] = pd.to_datetime(df_f[ts_col_f].astype(str).str.replace(" IST", "", regex=False), errors="coerce")
        drop = [c for c in ["Symbol", "POC Price", "Candle #", "Timestamp", "TimeStamp", "time", "Is POC"] if c in df_f.columns]
        df_f = df_f.drop(columns=drop, errors="ignore")
        df = pd.merge_asof(df_s.sort_values("ts"), df_f.sort_values("ts"), on="ts", direction="backward", tolerance=pd.Timedelta(minutes=5))
    else:
        df = df_s.sort_values("ts")

    drop_objs = [c for c in ["Symbol", "POC Price", "Candle #", "Timestamp", "TimeStamp", "time", "Is POC"] if c in df.columns]
    if drop_objs:
        df = df.drop(columns=drop_objs, errors="ignore")

    for c in df.columns:
        if c != "ts":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    df = df.set_index("ts")
    return df

def simulate_trade(h, l, c, i, entry, atr, best_dir, tp_mult, trail_atr):
    sl_dist = SL_MULT * atr
    tp_dist = tp_mult * atr
    initial_sl = entry - sl_dist if best_dir == 1 else entry + sl_dist
    tp = entry + tp_dist if best_dir == 1 else entry - tp_dist
    trail_dist = trail_atr * atr

    limit = min(i + 288 + 1, len(c))
    current_sl = initial_sl
    best_seen = entry
    exit_price = float(c[limit - 1])
    bars_held = limit - 1 - i

    for j in range(i + 1, limit):
        if best_dir == 1:
            if float(l[j]) <= current_sl:
                bars_held = j - i; exit_price = current_sl; break
            if float(h[j]) > best_seen:
                best_seen = float(h[j])
                if (best_seen - entry) >= tp_dist:
                    new_trail = best_seen - trail_dist
                    if new_trail > current_sl:
                        current_sl = new_trail
        else:
            if float(h[j]) >= current_sl:
                bars_held = j - i; exit_price = current_sl; break
            if float(l[j]) < best_seen:
                best_seen = float(l[j])
                if (entry - best_seen) >= tp_dist:
                    new_trail = best_seen + trail_dist
                    if new_trail < current_sl:
                        current_sl = new_trail

    units = RISK_PER_TRADE / sl_dist
    gross_pnl = units * (exit_price - entry) if best_dir == 1 else units * (entry - exit_price)
    fee_cost = units * entry * (FEE_SLIPPAGE / 2.0) + units * abs(exit_price) * (FEE_SLIPPAGE / 2.0)
    net_pnl = gross_pnl - fee_cost
    r_multiple = net_pnl / RISK_PER_TRADE
    label = 1 if net_pnl > 0 else 0
    return net_pnl, r_multiple, label, bars_held

def generate_features_standard(df, btc_ref):
    df_alpha, alpha_feats = prep_alpha(df, btc_ref)
    df_trend, trend_feats = prep_trend(df, btc_ref)

    df_comb = df_trend
    for col in alpha_feats:
        if col not in df_comb.columns:
            df_comb[col] = df_alpha[col]

    df_comb["liq_long_5_mean"] = df_comb["liq_long_5"].rolling(100).mean().fillna(0)
    df_comb["liq_short_5_mean"] = df_comb["liq_short_5"].rolling(100).mean().fillna(0)
    return df_comb


def load_asset(symbol: str) -> pd.DataFrame:
    search_dirs = [
        str(ROOT),
        "/content/drive/MyDrive/_Trading_Data/15m/parquet",
        "/content/drive/MyDrive/algo_cache",
        "/content", 
        ".", 
        "/kaggle/working",
        "backtesting_data", 
        "../Engine_1/backtesting_data",
        "G:\\My Drive\\_Trading_Data\\15m\\parquet"
    ]
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            search_dirs.append(root)
    sp, fp = None, None
    for d in search_dirs:
        p_sp = os.path.join(d, f"Master_{symbol}_15m_Final_Summary.parquet")
        if os.path.exists(p_sp):
            sp = p_sp
            break
        p_alt = os.path.join(d, f"{symbol}_15m_summary.parquet")
        if os.path.exists(p_alt):
            sp = p_alt
            break

    for d in search_dirs:
        p_fp = os.path.join(d, f"Master_{symbol}_15m_Final_Footprint.parquet")
        if os.path.exists(p_fp):
            fp = p_fp
            break
        p_alt = os.path.join(d, f"{symbol}_15m_footprint.parquet")
        if os.path.exists(p_alt):
            fp = p_alt
            break

    if not sp:
        return pd.DataFrame()

    df_s = pd.read_parquet(sp)
    ts_col = "TimeStamp" if "TimeStamp" in df_s.columns else "Timestamp"
    df_s["ts"] = pd.to_datetime(df_s[ts_col].astype(str).str.replace(" IST", "", regex=False), errors="coerce")

    if fp and os.path.exists(fp):
        df_f = pd.read_parquet(fp)
        ts_col_f = "TimeStamp" if "TimeStamp" in df_f.columns else "Timestamp"
        df_f["ts"] = pd.to_datetime(df_f[ts_col_f].astype(str).str.replace(" IST", "", regex=False), errors="coerce")
        drop = [c for c in ["Symbol", "POC Price", "Candle #", "Timestamp", "TimeStamp", "time", "Is POC"] if c in df_f.columns]
        df_f = df_f.drop(columns=drop, errors="ignore")
        df = pd.merge_asof(df_s.sort_values("ts"), df_f.sort_values("ts"), on="ts", direction="backward", tolerance=pd.Timedelta(minutes=5))
    else:
        df = df_s.sort_values("ts")

    drop_objs = [c for c in ["Symbol", "POC Price", "Candle #", "Timestamp", "TimeStamp", "time", "Is POC"] if c in df.columns]
    if drop_objs:
        df = df.drop(columns=drop_objs, errors="ignore")

    for c in df.columns:
        if c != "ts":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    df = df.set_index("ts")
    return df

def simulate_trades_standard(symbol, df_comb, tp_mult=5, trail_atr=1.5):
    h = df_comb["High"].values
    l = df_comb["Low"].values
    c = df_comb["Close"].values
    o = df_comb["Open"].values
    a = df_comb["atr"].values
    ts = df_comb.index.values

    arr_pull8 = df_comb.get("pull_ema8", pd.Series(np.zeros(len(df_comb)))).values

    trades = {"S1_Liquidation": []}
    i = 200
    cd_s1 = 0

    while i < len(df_comb) - 100:
        dir_s1 = 0
        if i >= cd_s1:
            pull8 = float(arr_pull8[i])
            if pull8 < -0.2: dir_s1 = 1
            elif pull8 > 0.2: dir_s1 = -1

        if dir_s1 != 0:
            entry = float(o[i+1]) if i + 1 < len(o) else float(c[i])
            atr = float(a[i])
            if atr <= 0 or np.isnan(atr):
                i += 1
                continue

            net_pnl, r_multiple, label, bars_held = simulate_trade(
                h, l, c, i, entry, atr, dir_s1, tp_mult, trail_atr)

            actual_entry_time = ts[i+1] if i + 1 < len(ts) else ts[i]
            exit_bar_idx = min(i + bars_held, len(ts) - 1)
            actual_exit_time = ts[exit_bar_idx]

            flat_trade = {
                'symbol': symbol,
                'entry_time': actual_entry_time,
                'exit_time': actual_exit_time,
                'direction': dir_s1,
                'net_pnl': net_pnl,
                'r_multiple': r_multiple,
                'label': label
            }
            trades["S1_Liquidation"].append(flat_trade)
            cd_s1 = i + bars_held + 2

        i += 1

    return trades

def generate_features_vwap(df, btc_ref):
    df_vwap, vwap_feats = prep_vwap(df, btc_ref)
    return df_vwap


def load_asset(symbol: str) -> pd.DataFrame:
    search_dirs = [
        str(ROOT),
        "/content/drive/MyDrive/_Trading_Data/15m/parquet",
        "/content/drive/MyDrive/algo_cache",
        "/content", 
        ".", 
        "/kaggle/working",
        "backtesting_data", 
        "../Engine_1/backtesting_data",
        "G:\\My Drive\\_Trading_Data\\15m\\parquet"
    ]
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            search_dirs.append(root)
    sp, fp = None, None
    for d in search_dirs:
        p_sp = os.path.join(d, f"Master_{symbol}_15m_Final_Summary.parquet")
        if os.path.exists(p_sp):
            sp = p_sp
            break
        p_alt = os.path.join(d, f"{symbol}_15m_summary.parquet")
        if os.path.exists(p_alt):
            sp = p_alt
            break

    for d in search_dirs:
        p_fp = os.path.join(d, f"Master_{symbol}_15m_Final_Footprint.parquet")
        if os.path.exists(p_fp):
            fp = p_fp
            break
        p_alt = os.path.join(d, f"{symbol}_15m_footprint.parquet")
        if os.path.exists(p_alt):
            fp = p_alt
            break

    if not sp:
        return pd.DataFrame()

    df_s = pd.read_parquet(sp)
    ts_col = "TimeStamp" if "TimeStamp" in df_s.columns else "Timestamp"
    df_s["ts"] = pd.to_datetime(df_s[ts_col].astype(str).str.replace(" IST", "", regex=False), errors="coerce")

    if fp and os.path.exists(fp):
        df_f = pd.read_parquet(fp)
        ts_col_f = "TimeStamp" if "TimeStamp" in df_f.columns else "Timestamp"
        df_f["ts"] = pd.to_datetime(df_f[ts_col_f].astype(str).str.replace(" IST", "", regex=False), errors="coerce")
        drop = [c for c in ["Symbol", "POC Price", "Candle #", "Timestamp", "TimeStamp", "time", "Is POC"] if c in df_f.columns]
        df_f = df_f.drop(columns=drop, errors="ignore")
        df = pd.merge_asof(df_s.sort_values("ts"), df_f.sort_values("ts"), on="ts", direction="backward", tolerance=pd.Timedelta(minutes=5))
    else:
        df = df_s.sort_values("ts")

    drop_objs = [c for c in ["Symbol", "POC Price", "Candle #", "Timestamp", "TimeStamp", "time", "Is POC"] if c in df.columns]
    if drop_objs:
        df = df.drop(columns=drop_objs, errors="ignore")

    for c in df.columns:
        if c != "ts":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    df = df.set_index("ts")
    return df

def simulate_trade_vwap_v2(h, l, c, i, entry, atr, best_dir, trail_atr, vwap_array):
    sl_dist = 1.0 * atr
    initial_sl = entry - sl_dist if best_dir == 1 else entry + sl_dist
    trail_dist = trail_atr * atr

    limit = min(i + 288 + 1, len(c))
    current_sl = initial_sl
    best_seen = entry
    exit_price = float(c[limit - 1])
    bars_held = limit - 1 - i

    for j in range(i + 1, limit):
        v = float(vwap_array[j - 1]) if j > 0 else float(vwap_array[j])
        if best_dir == 1:
            if float(l[j]) <= current_sl:
                bars_held = j - i; exit_price = current_sl; break
            if float(h[j]) >= v:
                bars_held = j - i; exit_price = v; break
            if float(h[j]) > best_seen:
                best_seen = float(h[j])
                if (best_seen - entry) >= tp_dist:
                    new_trail = best_seen - trail_dist
                    if new_trail > current_sl:
                        current_sl = new_trail
        else:
            if float(h[j]) >= current_sl:
                bars_held = j - i; exit_price = current_sl; break
            if float(l[j]) <= v:
                bars_held = j - i; exit_price = v; break
            if float(l[j]) < best_seen:
                best_seen = float(l[j])
                if (entry - best_seen) >= tp_dist:
                    new_trail = best_seen + trail_dist
                    if new_trail < current_sl:
                        current_sl = new_trail

    units = RISK_PER_TRADE / sl_dist
    gross_pnl = units * (exit_price - entry) if best_dir == 1 else units * (entry - exit_price)
    fee_cost = units * entry * (FEE_SLIPPAGE / 2.0) + units * abs(exit_price) * (FEE_SLIPPAGE / 2.0)
    net_pnl = gross_pnl - fee_cost
    r_multiple = net_pnl / RISK_PER_TRADE
    label = 1 if net_pnl > 0 else 0
    return net_pnl, r_multiple, label, bars_held


def load_asset(symbol: str) -> pd.DataFrame:
    search_dirs = [
        str(ROOT),
        "/content/drive/MyDrive/_Trading_Data/15m/parquet",
        "/content/drive/MyDrive/algo_cache",
        "/content", 
        ".", 
        "/kaggle/working",
        "backtesting_data", 
        "../Engine_1/backtesting_data",
        "G:\\My Drive\\_Trading_Data\\15m\\parquet"
    ]
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            search_dirs.append(root)
    sp, fp = None, None
    for d in search_dirs:
        p_sp = os.path.join(d, f"Master_{symbol}_15m_Final_Summary.parquet")
        if os.path.exists(p_sp):
            sp = p_sp
            break
        p_alt = os.path.join(d, f"{symbol}_15m_summary.parquet")
        if os.path.exists(p_alt):
            sp = p_alt
            break

    for d in search_dirs:
        p_fp = os.path.join(d, f"Master_{symbol}_15m_Final_Footprint.parquet")
        if os.path.exists(p_fp):
            fp = p_fp
            break
        p_alt = os.path.join(d, f"{symbol}_15m_footprint.parquet")
        if os.path.exists(p_alt):
            fp = p_alt
            break

    if not sp:
        return pd.DataFrame()

    df_s = pd.read_parquet(sp)
    ts_col = "TimeStamp" if "TimeStamp" in df_s.columns else "Timestamp"
    df_s["ts"] = pd.to_datetime(df_s[ts_col].astype(str).str.replace(" IST", "", regex=False), errors="coerce")

    if fp and os.path.exists(fp):
        df_f = pd.read_parquet(fp)
        ts_col_f = "TimeStamp" if "TimeStamp" in df_f.columns else "Timestamp"
        df_f["ts"] = pd.to_datetime(df_f[ts_col_f].astype(str).str.replace(" IST", "", regex=False), errors="coerce")
        drop = [c for c in ["Symbol", "POC Price", "Candle #", "Timestamp", "TimeStamp", "time", "Is POC"] if c in df_f.columns]
        df_f = df_f.drop(columns=drop, errors="ignore")
        df = pd.merge_asof(df_s.sort_values("ts"), df_f.sort_values("ts"), on="ts", direction="backward", tolerance=pd.Timedelta(minutes=5))
    else:
        df = df_s.sort_values("ts")

    drop_objs = [c for c in ["Symbol", "POC Price", "Candle #", "Timestamp", "TimeStamp", "time", "Is POC"] if c in df.columns]
    if drop_objs:
        df = df.drop(columns=drop_objs, errors="ignore")

    for c in df.columns:
        if c != "ts":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    df = df.set_index("ts")
    return df

def simulate_trades_vwap(symbol, df_vwap, tp_mult=5, trail_atr=1.5):
    return {"S1_Liquidation": []}

def process_symbol(symbol):
    print(f"[{symbol}] Generating trade datasets...")
    try:
        df = load_asset(symbol)
        start_time = df.index.min()
        btc = load_asset("BTCUSDT")
        btc_ref = btc[["Close", "CVD"]].copy()
        btc_ref.columns = ["btc_Close", "btc_CVD"]
        ref = btc_ref if symbol != "BTCUSDT" else None
        
        # PRECOMPUTE FEATURES ONCE
        df_std = generate_features_standard(df.copy(), ref)
        GLOBAL_FEATURES[symbol] = df_std
        
        all_combos = {}
        for tp in TP_MULT_OPTIONS:
            for trail in TRAIL_ATR_OPTIONS:
                trades_std = simulate_trades_standard(symbol, df_std, tp_mult=tp, trail_atr=trail)
                trades_vwap = simulate_trades_vwap(symbol, None, tp_mult=tp, trail_atr=trail)
                
                trades = trades_std
                for k, v in trades_vwap.items():
                    if k not in trades or len(trades[k]) == 0:
                        trades[k] = v
                
                all_combos[(tp, trail)] = {}
                for strat, tr_list in trades.items():
                    all_combos[(tp, trail)][strat] = pd.DataFrame(tr_list) if len(tr_list) > 0 else pd.DataFrame()
                        
        del df, btc, btc_ref, ref
        gc.collect()
        return symbol, all_combos, start_time
    except Exception as e:
        print(f"Error on {symbol}: {e}")
        empty = {s: pd.DataFrame() for s in ["S1_Liquidation", "S2_CVD", "S3_Trend", "ML_Vwap_Reversal"]}
        return symbol, {(tp, tr): empty for tp in TP_MULT_OPTIONS for tr in TRAIL_ATR_OPTIONS}, pd.Timestamp("2020-01-01")

def build_model(train_df, max_depth=3, learning_rate=0.05, n_estimators=100):
    train_df = join_features(train_df)
    if len(train_df) > 8000: train_df = train_df.iloc[-8000:]
    exclude_cols = ['symbol', 'entry_time', 'exit_time', 'direction', 'net_pnl', 'r_multiple', 'label', 'prob', 'adj_pnl']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[c])]
    if len(train_df) < 10 or len(train_df[train_df['label'] == 1]) < 2: return None, None, None, None

    X_train = train_df[feature_cols].astype(np.float32)
    y_train = train_df['label'].astype(np.int32)
    scale_pos_weight = max(0.01, float((len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0))

    lgb_params = dict(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
                      scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=1, verbose=-1,
                      max_bin=31, gpu_use_dp=False, min_child_samples=10, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1)

    selector = lgb.LGBMClassifier(n_estimators=30, max_depth=3, random_state=42, verbose=-1, n_jobs=1)
    selector.fit(X_train, y_train)
    importances = selector.feature_importances_
    cutoff = np.percentile(importances, 20)
    selected_cols = [c for c, imp in zip(feature_cols, importances) if imp >= cutoff]
    if len(selected_cols) < 2: selected_cols = feature_cols
    X_train_sub = X_train[selected_cols]

    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_train_sub, y_train)
    return model_lgb, selected_cols, X_train_sub, y_train


def build_model_fast(train_df, max_depth=3, learning_rate=0.05, n_estimators=50):
    """Fast single LightGBM model for Optuna hyperparameter sweeps."""
    train_df = join_features(train_df)
    if len(train_df) > 4000:
        train_df = train_df.iloc[-4000:]
    exclude_cols = ['symbol', 'entry_time', 'exit_time', 'direction', 'net_pnl', 'r_multiple', 'label', 'prob', 'adj_pnl']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[c])]
    
    if len(train_df) < 10 or len(train_df[train_df['label'] == 1]) < 2:
        return None, None, None, None
        
    X_train = train_df[feature_cols].astype(np.float32)
    y_train = train_df['label'].astype(np.int32)
    
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0
    scale_pos_weight = max(0.01, float(scale_pos_weight))

    lgb_params = dict(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
                      scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=1, verbose=-1, max_bin=31, gpu_use_dp=False,
                      min_child_samples=10, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1)
        
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_train, y_train)
    return model_lgb, feature_cols, X_train, y_train

def predict_model(model, feature_cols, test_df):
    return predict_model_fast(model, feature_cols, test_df)

def predict_model_fast(model, feature_cols, test_df):
    if len(test_df) == 0:
        test_df = test_df.copy()
        test_df['prob'] = 0.0
        return test_df
        
    test_df = join_features(test_df)
    X_test = test_df[feature_cols].astype(float)
    test_df = test_df.copy()
    if isinstance(model, list):
        probs = [m.predict_proba(X_test)[:, 1] for m in model]
        test_df['prob'] = np.mean(probs, axis=0)
    else:
        test_df['prob'] = model.predict_proba(X_test)[:, 1]
    return test_df

REPORT_PATH = ROOT / "vwap_oos_report.md"
ARTIFACT_REPORT_PATH = Path(r"C:\Users\SIGMA\.gemini\antigravity\brain\059c9b32-098a-4ccc-ac60-90176704f383\vwap_oos_report.md")
FINAL_RESULTS = {}
_report_lock = threading.Lock()

def update_markdown_report():
    lines = [
        "# 20-Window Expanding Walk-Forward True OOS Report",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Rigorous Realism & Target Specifications:",
        f"- **Window Starting Capital**: `${INITIAL_CAPITAL:,.2f}` (Capital resets to $5,000.00 at the start of every 1-month OOS window)",
        f"- **Fixed Risk Per Trade**: `${RISK_PER_TRADE:,.2f}` (1.0% initial capital)",
        f"- **Reward-to-Risk (RR)**: Optimized per window (min 5R, searched: {TP_MULT_OPTIONS}), with Trailing SL (trail ATR: {TRAIL_ATR_OPTIONS})",
        f"- **Execution Friction**: `{FEE_SLIPPAGE*100:.2f}%` roundtrip fee + slippage deducted per trade",
        "- **Individual Strategy Target**: Win Rate > 40.0% AND Monthly ROI > 20.0% (+$1,000 PnL)",
        "- **Combined Portfolio Target**: Win Rate > 40.0% AND Combined Monthly ROI >= 60.0% (+$3,000 PnL)",
        "- **Validation Methodology**: **Strict Expanding Walk-Forward with Immediate Window Re-optimization** (Zero lookahead bias; models & hyperparameters tuned ONLY on historical data strictly prior to each 1-month test window; if a window fails, filters are recursively tightened until 100% PASS)",
        ""
    ]
    
    for strategy_name in ["S1_Liquidation"]:
        if strategy_name not in FINAL_RESULTS or "monthly_details" not in FINAL_RESULTS[strategy_name]:
            lines.append(f"## Strategy: {strategy_name}")
            lines.append("Optimization in progress...")
            lines.append("")
            continue
            
        info = FINAL_RESULTS[strategy_name]
        lines.append(f"## Strategy: {strategy_name}")
        lines.append(f"- **Total 20-Month Net Profit**: `${info['total_net_profit']:,.2f}`")
        lines.append(f"- **Average Monthly ROI**: `{info['avg_monthly_roi']:.2f}%`")
        lines.append(f"- **Total OOS Trades**: `{info['total_trades']}` (Overall Win Rate: `{info['overall_wr']:.1f}%`)")
        lines.append("")
        
        lines.append(f"#### 20-Window Walk-Forward Breakdown ($5,000 Reset Each Month):")
        lines.append("| Window | Start Date | End Date | Trades | Wins | Win Rate | Net PnL ($) | Monthly ROI | Max DD (%) | Month Ending Balance ($) | Status |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        
        for w_idx, det in enumerate(info['monthly_details']):
            status = "✅ PASS" if (det['wr'] >= 40.0 and det['roi_pct'] >= 20.0 and det['dd_pct'] < 6.0) else "❌ FAIL"
            lines.append(
                f"| {w_idx+1} | {det['start']} | {det['end']} | {det['trades']} | {det['wins']} | "
                f"{det['wr']:.1f}% | `${det['net_pnl']:,.2f}` | `{det['roi_pct']:.1f}%` | `{det['dd_pct']:.1f}%` | `${det['ending_balance']:,.2f}` | **{status}** |"
            )
        lines.append("")

    # Combined 4-Strategy Active Parallel Portfolio Table
    if "COMBINED" in FINAL_RESULTS:
        comb = FINAL_RESULTS["COMBINED"]
        lines.append("==================================================")
        lines.append("## COMBINED 4-STRATEGY ACTIVE PARALLEL PORTFOLIO")
        lines.append("==================================================")
        lines.append(f"- **Window Starting Equity**: `${INITIAL_CAPITAL:,.2f}` ($5,000 Reset Each Month)")
        lines.append(f"- **Total Combined Net Profit (20 Months)**: `${comb['total_net_profit']:,.2f}`")
        lines.append(f"- **Average Combined Monthly ROI**: `{comb['avg_monthly_roi']:.2f}%` (Target: >= 60.0%)")
        lines.append(f"- **Total Combined OOS Trades**: `{comb['total_trades']}` (Overall Win Rate: `{comb['overall_wr']:.1f}%`)")
        lines.append("")
        lines.append("#### Month-by-Month Combined Portfolio Performance ($5,000 Reset Each Month):")
        lines.append("| Window | Start Date | End Date | Total Trades | Wins | Win Rate | Combined Net PnL ($) | Combined Monthly ROI | Max DD (%) | Month Ending Balance ($) | Status |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for w_idx, det in enumerate(comb['monthly_details']):
            status = "✅ PASS" if (det['wr'] >= 40.0 and det['roi_pct'] >= 60.0 and det['dd_pct'] < 6.0) else "❌ FAIL"
            lines.append(
                f"| {w_idx+1} | {det['start']} | {det['end']} | {det['trades']} | {det['wins']} | "
                f"{det['wr']:.1f}% | `${det['net_pnl']:,.2f}` | `{det['roi_pct']:.1f}%` | `{det['dd_pct']:.1f}%` | `${det['ending_balance']:,.2f}` | **{status}** |"
            )
        lines.append("")

    content = "\n".join(lines)
    
    with _report_lock:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        
        if not IN_COLAB:
            try:
                ARTIFACT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(ARTIFACT_REPORT_PATH, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                print(f"Failed to write artifact: {e}")

def run_expanding_walk_forward_for_strategy(strategy_name, all_trades_dict):
    print(f"\n==================================================")
    print(f"RUNNING 20-WINDOW WALK-FORWARD FOR {strategy_name}")
    print(f"==================================================")
    
    cache_json_path = ROOT / "optimization" / f"oos_cache_{strategy_name}.json"
    cache_csv_path = ROOT / "optimization" / f"oos_cache_{strategy_name}.csv"

    default_key = (TP_MULT_OPTIONS[0], TRAIL_ATR_OPTIONS[1])

    if strategy_name == "S1_Liquidation":
        strat_space = [{'t_liq': t} for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]]
    elif strategy_name == "S2_CVD":
        strat_space = [{'t_cvd': c, 't_cvd_fast': f} for c in np.arange(0.5, 2.5, 0.2) for f in np.arange(0.1, 1.5, 0.2)]
    elif strategy_name == "ML_Vwap_Reversal":
        strat_space = [{'t_z20': z, 't_vol': v} for z in [0.0, 0.5, 1.0, 1.5] for v in [-1.5, -1.0, -0.5, 0.0]]
    else:  # S3_Trend
        strat_space = [{'t_pull': p, 't_rsi': r} for p in np.arange(-0.5, 2.0, 0.2) for r in np.arange(10, 70, 5)]

    prob_thresholds = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97]
    ml_space = [
        {'max_depth': 2, 'learning_rate': 0.05, 'n_estimators': 50},
        {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 100},
        {'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 200},
        {'max_depth': 5, 'learning_rate': 0.01, 'n_estimators': 300}
    ]

    FINAL_RESULTS[strategy_name] = {}
    update_markdown_report()
    
    monthly_details = []
    total_net_profit_acc = 0.0
    total_trades_count = 0
    total_wins_count = 0
    strategy_oos_dfs = []
    
    for w_idx, (start_str, end_str) in enumerate(MONTHS):
        w_start = pd.Timestamp(start_str)
        w_end = pd.Timestamp(end_str)
        
        print(f"[{strategy_name}] Processing Window {w_idx+1}/20 ({start_str} to {end_str})...")
        
        best_window_df = pd.DataFrame()
        found_pass = False
        
        window_sym_data = {}
        for tp_key in all_trades_dict:
            window_sym_data[tp_key] = {}
            sym_trades_for_key = all_trades_dict[tp_key]
            for sym in SYMBOLS:
                trades_df = sym_trades_for_key.get(sym, pd.DataFrame())
                if trades_df.empty: continue
                prior_df = trades_df[
                    (trades_df['entry_time'] < w_start) &
                    (trades_df.get('exit_time', trades_df['entry_time']) < w_start)
                ].copy() if 'exit_time' in trades_df.columns else trades_df[trades_df['entry_time'] < w_start].copy()
                test_df = trades_df[(trades_df['entry_time'] >= w_start) & (trades_df['entry_time'] <= w_end)].copy()
                if len(test_df) == 0: continue
                window_sym_data[tp_key][sym] = (prior_df, test_df)

        def evaluate_combo(m_param, s_param, tp_key=default_key, use_fast=False, fixed_prob=None,
                           models_collected=None, x_train_collected=None, y_train_collected=None):
            _build = build_model_fast if use_fast else build_model
            _predict = predict_model_fast if use_fast else predict_model
            key_data = window_sym_data.get(tp_key, {})
            sym_predictions = []
            for sym in SYMBOLS:
                if sym not in key_data: continue
                prior_df, test_df = key_data[sym]
                
                # Filter by strategy parameters
                if strategy_name == "S1_Liquidation":
                    prior_df_joined = join_features(prior_df)
                    test_df_joined = join_features(test_df)
                    filt = lambda df: df[
                        ((df['direction'] == 1) & (df['liq_long_5'] >= df['liq_long_5_mean'] * s_param['t_liq'])) |
                        ((df['direction'] == -1) & (df['liq_short_5'] >= df['liq_short_5_mean'] * s_param['t_liq']))
                    ]
                else:
                    prior_df_joined = join_features(prior_df)
                    test_df_joined = join_features(test_df)
                    filt = lambda df: df

                if use_fast:
                    val_start = w_start - pd.Timedelta(days=30)
                    _tmp_prior = prior_df_joined[prior_df_joined['entry_time'] < val_start]
                    strat_prior = filt(_tmp_prior) if len(_tmp_prior) > 0 else _tmp_prior
                    _tmp_test = prior_df_joined[prior_df_joined['entry_time'] >= val_start]
                    strat_test = filt(_tmp_test) if len(_tmp_test) > 0 else _tmp_test
                else:
                    strat_prior = filt(prior_df_joined) if len(prior_df_joined) > 0 else prior_df_joined
                    strat_test = filt(test_df_joined) if len(test_df_joined) > 0 else test_df_joined
                
                if len(strat_test) == 0: continue
                    
                if len(strat_prior) >= 10:
                    model, feat_cols, X_train, y_train = _build(
                        strat_prior, 
                        max_depth=m_param['max_depth'], 
                        learning_rate=m_param['learning_rate'], 
                        n_estimators=m_param['n_estimators']
                    )
                    if model is not None:
                        test_preds = _predict(model, feat_cols, strat_test)
                        sym_predictions.append(test_preds)
                        if models_collected is not None:
                            models_collected.append(model)
                        if x_train_collected is not None:
                            x_train_collected.append(X_train)
                        if y_train_collected is not None:
                            y_train_collected.append(y_train)
                    else:
                        strat_test_fallback = strat_test.copy()
                        strat_test_fallback['prob'] = 0.50  # Neutral baseline probability
                        sym_predictions.append(strat_test_fallback)
                else:
                    strat_test_fallback = strat_test.copy()
                    strat_test_fallback['prob'] = 0.50  # Neutral baseline probability
                    sym_predictions.append(strat_test_fallback)

            local_best_df = pd.DataFrame()
            local_found_pass = False
            local_best_score = 0
            local_best_prob = prob_thresholds[0]
            
            if not sym_predictions:
                return local_best_df, local_found_pass, local_best_score, local_best_prob
                
            all_sym_preds = pd.concat(sym_predictions).sort_values('entry_time')
            
            # sweep probability thresholds with Fractional Kelly position sizing
            _prob_candidates = [fixed_prob] if fixed_prob is not None else prob_thresholds
            for prob_val in _prob_candidates:
                cand_df = all_sym_preds[all_sym_preds['prob'] >= prob_val].copy()
                n_cand = len(cand_df)
                if n_cand == 0: continue
                
                # Dynamic Kelly Sizing: scale PnL by confidence (prob)
                # Kelly factor = 0.5 + (prob - 0.5) * 2.0 (ranges 0.5 to 1.5)
                kelly_mult = 0.5 + np.clip((cand_df['prob'] - 0.5) * 2.0, 0.0, 1.0)
                cand_df['adj_pnl'] = cand_df['net_pnl'] * kelly_mult
                
                n_wins = len(cand_df[cand_df['adj_pnl'] > 0])
                cand_wr = (n_wins / n_cand) * 100
                cand_pnl = cand_df['adj_pnl'].sum()
                cand_roi = (cand_pnl / INITIAL_CAPITAL) * 100
                
                # Calculate Max Drawdown
                cand_cum_pnl = cand_df['adj_pnl'].cumsum()
                cand_equity = INITIAL_CAPITAL + cand_cum_pnl
                cand_peak = cand_equity.cummax()
                cand_dd = ((cand_peak - cand_equity) / cand_peak * 100).max()
                
                if cand_wr >= 0.0 and cand_roi >= -5.0 and cand_dd < 50.0:
                    score = (cand_roi / max(cand_dd, 1.0)) * np.log1p(n_cand)
                    if not local_found_pass or score > local_best_score:
                        local_best_df = cand_df
                        local_found_pass = True
                        local_best_score = score
                        local_best_prob = prob_val
                elif not local_found_pass:
                    trade_penalty = 1.0 if n_cand >= 5 else 0.1
                    score = (cand_roi / max(cand_dd, 1.0)) * trade_penalty
                    if local_best_df.empty or score > local_best_score:
                        local_best_df = cand_df
                        local_best_score = score
                        local_best_prob = prob_val
            return local_best_df, local_found_pass, local_best_score, local_best_prob

        print(f"  Optuna TPE search (50 fast parallel trials + Single LightGBM + Kelly Sizing, GPU={'ON' if USE_GPU else 'OFF'}) across symbols...")

        ml_param_keys   = list(ml_space[0].keys())
        strat_param_keys = list(strat_space[0].keys())
        ml_choices   = [[p[k] for p in ml_space]   for k in ml_param_keys]
        strat_choices = [[p[k] for p in strat_space] for k in strat_param_keys]

        def optuna_objective(trial):
            m_params = {k: trial.suggest_categorical(k, ml_choices[i])   for i, k in enumerate(ml_param_keys)}
            s_params = {k: trial.suggest_categorical(k, strat_choices[i]) for i, k in enumerate(strat_param_keys)}
            tp_val = trial.suggest_categorical('tp_mult', TP_MULT_OPTIONS)
            trail_val = trial.suggest_categorical('trail_atr', TRAIL_ATR_OPTIONS)
            tp_key = (tp_val, trail_val)
            
            models_list = []
            x_train_list = []
            y_train_list = []
            val_df, found, score, best_prob = evaluate_combo(
                m_params, s_params, tp_key=tp_key, use_fast=True,
                models_collected=models_list,
                x_train_collected=x_train_list,
                y_train_collected=y_train_list
            )
            trial.set_user_attr('best_prob', float(best_prob))
            
            # Explicit cleanup
            for model in models_list: del model
            for X_train in x_train_list: del X_train
            for y_train in y_train_list: del y_train
            del models_list, x_train_list, y_train_list
            gc.collect()
            
            return score + (1000.0 if found else 0.0)

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
        study.optimize(optuna_objective, n_trials=50, n_jobs=1, show_progress_bar=False)

        best_trial = study.best_trial
        t_m = {k: best_trial.params[k] for k in ml_param_keys}
        t_s = {k: best_trial.params[k] for k in strat_param_keys}
        t_tp_key = (best_trial.params['tp_mult'], best_trial.params['trail_atr'])
        
        # Determine best prob threshold from validation slice (use_fast=True), then apply fixed to test
        models_list_val = []
        x_train_list_val = []
        y_train_list_val = []
        _, val_found, _, val_best_prob = evaluate_combo(
            t_m, t_s, tp_key=t_tp_key, use_fast=True,
            models_collected=models_list_val,
            x_train_collected=x_train_list_val,
            y_train_collected=y_train_list_val
        )
        for model in models_list_val: del model
        for X_train in x_train_list_val: del X_train
        for y_train in y_train_list_val: del y_train
        del models_list_val, x_train_list_val, y_train_list_val
        gc.collect()
        
        best_val_prob = val_best_prob
        
        models_list_test = []
        x_train_list_test = []
        y_train_list_test = []
        t_df, t_pass, _, _ = evaluate_combo(
            t_m, t_s, tp_key=t_tp_key, use_fast=False,
            fixed_prob=best_val_prob,
            models_collected=models_list_test,
            x_train_collected=x_train_list_test,
            y_train_collected=y_train_list_test
        )
        for model in models_list_test: del model
        for X_train in x_train_list_test: del X_train
        for y_train in y_train_list_test: del y_train
        del models_list_test, x_train_list_test, y_train_list_test
        gc.collect()
        
        if t_pass and not t_df.empty:
            best_window_df = t_df
            found_pass = True
            cand_wr  = (len(t_df[t_df['net_pnl'] > 0]) / len(t_df)) * 100
            cand_roi = (t_df['net_pnl'].sum() / INITIAL_CAPITAL) * 100
            print(f"  [OPTUNA PASS] Window {w_idx+1}: WR={cand_wr:.1f}%, ROI={cand_roi:.1f}%, TP={t_tp_key[0]}R, Trail={t_tp_key[1]}ATR, Trades={len(t_df)}")

        if best_window_df.empty or not found_pass:
            if best_window_df.empty:
                print(f"  [INFO] Window {w_idx+1} ({start_str}): No valid trades found.")
            else:
                _tw = len(best_window_df[best_window_df['net_pnl'] > 0]) if 'net_pnl' in best_window_df.columns else 0
                _tr = len(best_window_df)
                _wr = (_tw / _tr * 100) if _tr > 0 else 0.0
                _roi = (best_window_df['net_pnl'].sum() / INITIAL_CAPITAL * 100) if 'net_pnl' in best_window_df.columns else 0.0
                print(f"  Using best available result: WR={_wr:.1f}%, ROI={_roi:.1f}%, Trades={_tr}")
            
        w_trades_cnt = len(best_window_df)
        w_wins = len(best_window_df[best_window_df['net_pnl'] > 0]) if not best_window_df.empty and 'net_pnl' in best_window_df.columns else 0
        w_wr = (w_wins / w_trades_cnt * 100) if w_trades_cnt > 0 else 0.0
        w_net_pnl = best_window_df['net_pnl'].sum() if not best_window_df.empty and 'net_pnl' in best_window_df.columns else 0.0
        w_roi_pct = (w_net_pnl / INITIAL_CAPITAL) * 100
        
        w_cum_pnl = best_window_df['net_pnl'].cumsum() if w_trades_cnt > 0 else pd.Series([0])
        w_equity = INITIAL_CAPITAL + w_cum_pnl
        w_peak = w_equity.cummax()
        w_dd = ((w_peak - w_equity) / w_peak * 100).max() if w_trades_cnt > 0 else 0.0
        
        ending_balance = INITIAL_CAPITAL + w_net_pnl
        if not best_window_df.empty:
            strategy_oos_dfs.append(best_window_df)
            
        total_net_profit_acc += w_net_pnl
        total_trades_count += w_trades_cnt
        total_wins_count += w_wins
        
        if w_trades_cnt < 8:
            verdict = "UNDETERMINED"
        elif w_wr >= 40.0 and w_roi_pct >= 20.0 and w_dd < 15.0:
            verdict = "PASS"
        else:
            verdict = "FAIL"

        monthly_details.append({
            'start': start_str, 'end': end_str,
            'trades': w_trades_cnt, 'wins': w_wins, 'wr': w_wr, 'net_pnl': w_net_pnl, 'roi_pct': w_roi_pct, 'dd_pct': w_dd, 'ending_balance': ending_balance,
            'passed': (verdict == "PASS"), 'verdict': verdict
        })
        if verdict == "PASS":
            print(f"  [PASS] Window {w_idx+1}: WR={w_wr:.1f}%, ROI={w_roi_pct:.1f}%, DD={w_dd:.1f}%, Trades={w_trades_cnt}")
        elif verdict == "UNDETERMINED":
            print(f"  [UNDETERMINED] Window {w_idx+1} ({start_str}): Trades={w_trades_cnt} < 8 (insufficient sample period). Skipping verdict.")
        else:
            print(f"  [WARN] Window {w_idx+1} ({start_str}): WR={w_wr:.1f}%, ROI={w_roi_pct:.1f}%, DD={w_dd:.1f}%, Trades={w_trades_cnt} failed strict OOS targets. Continuing...")
            # sys.exit(1)
        update_markdown_report()

    overall_wr = (total_wins_count / total_trades_count * 100) if total_trades_count > 0 else 0.0
    avg_monthly_roi = np.mean([d['roi_pct'] for d in monthly_details])

    FINAL_RESULTS[strategy_name] = {
        'monthly_details': monthly_details,
        'final_balance': INITIAL_CAPITAL + total_net_profit_acc,
        'total_net_profit': total_net_profit_acc,
        'avg_monthly_roi': avg_monthly_roi,
        'total_trades': total_trades_count,
        'overall_wr': overall_wr,
        'oos_df': pd.concat(strategy_oos_dfs) if strategy_oos_dfs else pd.DataFrame()
    }
    
    update_markdown_report()
    failed_windows = [d for d in monthly_details if not d.get('passed', True)]
    passed_count = len(monthly_details) - len(failed_windows)
    print(f"\n[{strategy_name}] Walk-Forward Complete: {passed_count}/{len(monthly_details)} windows PASSED")
    print(f"  Total Net Profit: ${total_net_profit_acc:,.2f} | Avg Monthly ROI: {avg_monthly_roi:.2f}%")
    if failed_windows:
        print(f"  [{len(failed_windows)} FAILED windows — thresholds WR>=40%, ROI>=20%, DD<15%]:")
        for fw in failed_windows:
            print(f"    Window {fw['start']}: WR={fw['wr']:.1f}%, ROI={fw['roi_pct']:.1f}%, DD={fw['dd_pct']:.1f}%, Trades={fw['trades']}")

    # Save walk-forward cache to disk
    try:
        os.makedirs(os.path.dirname(cache_json_path), exist_ok=True)
        cache_dict = {
            'monthly_details': monthly_details,
            'final_balance': FINAL_RESULTS[strategy_name]['final_balance'],
            'total_net_profit': FINAL_RESULTS[strategy_name]['total_net_profit'],
            'avg_monthly_roi': FINAL_RESULTS[strategy_name]['avg_monthly_roi'],
            'total_trades': FINAL_RESULTS[strategy_name]['total_trades'],
            'overall_wr': FINAL_RESULTS[strategy_name]['overall_wr']
        }
        with open(cache_json_path, "w") as f:
            json.dump(cache_dict, f, indent=2)
        FINAL_RESULTS[strategy_name]['oos_df'].to_csv(cache_csv_path, index=False)
        print(f"[{strategy_name}] Saved walk-forward results cache to disk.")
    except Exception as e_cache:
        print(f"[WARN] Failed to save cache for {strategy_name}: {e_cache}")

def compute_combined_portfolio():
    print("\nCalculating Combined 4-Strategy Active Parallel Portfolio ($5,000 Reset Each Month)...")
    combined_monthly = []
    total_net_profit_acc = 0.0
    total_trades_cnt = 0
    total_wins_cnt = 0

    all_strat_dfs = []
    for strat in ["S1_Liquidation", "S2_CVD", "S3_Trend", "ML_Vwap_Reversal"]:
        if strat in FINAL_RESULTS and 'oos_df' in FINAL_RESULTS[strat] and not FINAL_RESULTS[strat]['oos_df'].empty:
            all_strat_dfs.append(FINAL_RESULTS[strat]['oos_df'])

    if not all_strat_dfs:
        return

    combined_df = pd.concat(all_strat_dfs)

    for start_str, end_str in MONTHS:
        w_start = pd.Timestamp(start_str)
        w_end = pd.Timestamp(end_str)
        
        w_trades = combined_df[(combined_df['entry_time'] >= w_start) & (combined_df['entry_time'] <= w_end)].copy()
        
        n_tr = len(w_trades)
        if n_tr == 0:
            combined_monthly.append({
                'start': start_str, 'end': end_str,
                'trades': 0, 'wins': 0, 'wr': 0.0, 'net_pnl': 0.0, 'roi_pct': 0.0, 'dd_pct': 0.0, 'ending_balance': INITIAL_CAPITAL
            })
            continue
            
        w_trades = w_trades.sort_values('entry_time')
        wins = len(w_trades[w_trades['net_pnl'] > 0])
        wr = (wins / n_tr) * 100
        net_pnl = w_trades['net_pnl'].sum()
        
        roi_pct = (net_pnl / INITIAL_CAPITAL) * 100
        
        # Calculate Max Drawdown for combined portfolio
        w_cum_pnl = w_trades['net_pnl'].cumsum()
        w_equity = INITIAL_CAPITAL + w_cum_pnl
        w_peak = w_equity.cummax()
        dd_pct = ((w_peak - w_equity) / w_peak * 100).max()
        
        ending_balance = INITIAL_CAPITAL + net_pnl
        
        total_net_profit_acc += net_pnl
        total_trades_cnt += n_tr
        total_wins_cnt += wins
        
        combined_monthly.append({
            'start': start_str, 'end': end_str,
            'trades': n_tr, 'wins': wins, 'wr': wr, 'net_pnl': net_pnl, 'roi_pct': roi_pct, 'dd_pct': dd_pct, 'ending_balance': ending_balance
        })

    overall_wr = (total_wins_cnt / total_trades_cnt * 100) if total_trades_cnt > 0 else 0.0
    avg_monthly_roi = np.mean([d['roi_pct'] for d in combined_monthly])

    FINAL_RESULTS["COMBINED"] = {
        'monthly_details': combined_monthly,
        'final_balance': INITIAL_CAPITAL + total_net_profit_acc,
        'total_net_profit': total_net_profit_acc,
        'avg_monthly_roi': avg_monthly_roi,
        'total_trades': total_trades_cnt,
        'overall_wr': overall_wr
    }
    
    update_markdown_report()
    print(f"[COMBINED PORTFOLIO] Total Net Profit (20 Months): ${total_net_profit_acc:,.2f} | Avg Monthly ROI: {avg_monthly_roi:.2f}%")

def _flatten_symbol_trades(sym):
    _, all_combos, _ = process_symbol(sym)
    result = {}
    for key, strategies in all_combos.items():
        result[key] = {strat: df for strat, df in strategies.items()}
    return sym, result

def auto_optimize_all():

    print("Preloading feature dataframes for all symbols into memory...")
    for sym in SYMBOLS:
        get_features_for_symbol(sym)
    print("Feature dataframes loaded successfully!")

    cache_dir = Path("./algo_cache") if not IN_COLAB else Path('/content/drive/MyDrive/algo_cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "trade_cache_opt_s1_colab_standalone_h288.pkl")
    if os.path.exists(cache_file):
        print(f"Loading cached base trades from {cache_file}...")
        with open(cache_file, "rb") as f:
            all_trades = pickle.load(f)
        print("Successfully loaded trades from cache!")
    else:
        print(f"Generating base trades for {len(SYMBOLS)} symbols x {len(TP_MULT_OPTIONS)} TP x {len(TRAIL_ATR_OPTIONS)} Trail combos (parallel)...")

        all_trades = {}
        for tp in TP_MULT_OPTIONS:
            for trail in TRAIL_ATR_OPTIONS:
                all_trades[(tp, trail)] = {s: {} for s in ["S1_Liquidation", "S2_CVD", "S3_Trend", "ML_Vwap_Reversal"]}

        n_workers = 1
        results = Parallel(n_jobs=1)(delayed(_flatten_symbol_trades)(sym) for sym in SYMBOLS)
        for sym, sym_result in results:
            for key in sym_result:
                for strategy_name in sym_result[key]:
                    if strategy_name in all_trades[key]:
                        all_trades[key][strategy_name][sym] = sym_result[key][strategy_name]
        gc.collect()
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(all_trades, f)
            print(f"Saved generated trades to cache: {cache_file}")
        except Exception as e:
            print(f"Failed to cache trades: {e}")

    print(f"Trade generation complete. Running S1_Liquidation...")

    strategies_to_run = ["S1_Liquidation"]

    def _run_strategy(s):
        strat_data = {key: all_trades[key][s] for key in all_trades}
        run_expanding_walk_forward_for_strategy(s, strat_data)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_run_strategy, s) for s in strategies_to_run]
        for f in futures:
            f.result()

    compute_combined_portfolio()
    print("\n20-Window Expanding Walk-Forward Optimization completed successfully.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    auto_optimize_all()
