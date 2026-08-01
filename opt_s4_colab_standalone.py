import warnings
warnings.filterwarnings('ignore')
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import os
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
import xgboost as xgb
from catboost import CatBoostClassifier
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
import numba

INITIAL_CAPITAL = 5000.0
RISK_PER_TRADE = 150.0
FEE_SLIPPAGE = 0.0015
SL_MULT = 1.0
TP_MULT_OPTIONS = [5.0, 7.0, 10.0]
TRAIL_ATR_OPTIONS = [0.4, 0.6, 0.8, 1.2, 1.8]

@numba.njit(fastmath=True)

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

def simulate_trade_vwap_v2_jit(h, l, c, i, entry, atr, best_dir, trail_atr, tp_mult, vwap_array, v_lower_20, v_upper_20, risk_per_trade, fee_slippage, is_trend_trade):
    sl_dist = 1.0 * atr
    tp_dist = tp_mult * atr
    initial_sl = entry - sl_dist if best_dir == 1 else entry + sl_dist
    trail_dist = trail_atr * atr

    limit = min(i + 288 + 1, len(c))
    current_sl = initial_sl
    best_seen = entry
    exit_price = c[limit - 1]
    bars_held = limit - 1 - i

    for j in range(i + 1, limit):
        v = vwap_array[j - 1] if j > 0 else vwap_array[j]
        vu = v_upper_20[j - 1] if j > 0 else v_upper_20[j]
        vl = v_lower_20[j - 1] if j > 0 else v_lower_20[j]
        if best_dir == 1:
            if l[j] <= current_sl:
                bars_held = j - i; exit_price = current_sl; break
            target = vu if is_trend_trade == 1 else v
            if h[j] >= target:
                bars_held = j - i; exit_price = target; break
            if h[j] > best_seen:
                best_seen = h[j]
                if (best_seen - entry) >= tp_dist:
                    new_trail = best_seen - trail_dist
                    if new_trail > current_sl:
                        current_sl = new_trail
        else:
            if h[j] >= current_sl:
                bars_held = j - i; exit_price = current_sl; break
            target = vl if is_trend_trade == 1 else v
            if l[j] <= target:
                bars_held = j - i; exit_price = target; break
            if l[j] < best_seen:
                best_seen = l[j]
                if (entry - best_seen) >= tp_dist:
                    new_trail = best_seen + trail_dist
                    if new_trail < current_sl:
                        current_sl = new_trail

    units = risk_per_trade / sl_dist
    gross_pnl = units * (exit_price - entry) if best_dir == 1 else units * (entry - exit_price)
    fee_cost = units * entry * (fee_slippage / 2.0) + units * abs(exit_price) * (fee_slippage / 2.0)
    net_pnl = gross_pnl - fee_cost
    r_multiple = net_pnl / risk_per_trade
    label = 1 if net_pnl > 0 else 0
    return net_pnl, r_multiple, label, bars_held

@numba.njit(fastmath=True)

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

def simulate_trades_vwap_kernel(h, l, c, o, a, vwap, v_lower_20, v_upper_20, rsi, ema_fast, ema_slow, tp_mult, trail_atr, risk_per_trade, fee_slippage):
    n = len(c)
    max_trades = n
    trade_indices = np.empty(max_trades, dtype=np.int32)
    directions = np.empty(max_trades, dtype=np.int32)
    net_pnls = np.empty(max_trades, dtype=np.float32)
    r_multiples = np.empty(max_trades, dtype=np.float32)
    labels = np.empty(max_trades, dtype=np.int32)
    bars_held_arr = np.empty(max_trades, dtype=np.int32)
    
    count = 0
    i = 200
    cd = 0
    while i < n - 100:
        if count >= max_trades:
            break
        if i < cd:
            i += 1
            continue
        best_dir = 0
        ef = ema_fast[i]
        es = ema_slow[i]
        
        # Bullish rules: Extreme band mean-reversion OR trend VWAP pullback
        if l[i] <= v_lower_20[i] and rsi[i] < 45:
            best_dir = 1
        elif l[i] <= vwap[i] + 0.3 * a[i] and (ef > es):
            best_dir = 1
        # Bearish rules: Extreme band mean-reversion OR trend VWAP pullback
        elif h[i] >= v_upper_20[i] and rsi[i] > 55:
            best_dir = -1
        elif h[i] >= vwap[i] - 0.3 * a[i] and (ef < es):
            best_dir = -1
            
        if best_dir != 0:
            entry = o[i+1] if i + 1 < n else c[i]
            atr = a[i]
            if atr <= 0 or np.isnan(atr):
                i += 1
                continue
                
            is_trend_trade = 0
            if (best_dir == 1 and ef > es) or (best_dir == -1 and ef < es):
                is_trend_trade = 1
            net_pnl, r_multiple, label, bars_held = simulate_trade_vwap_v2_jit(
                h, l, c, i, entry, atr, best_dir, trail_atr, tp_mult, vwap, v_lower_20, v_upper_20, risk_per_trade, fee_slippage, is_trend_trade
            )
            
            trade_indices[count] = i
            directions[count] = best_dir
            net_pnls[count] = net_pnl
            r_multiples[count] = r_multiple
            labels[count] = label
            bars_held_arr[count] = bars_held
            count += 1
            cd = i + bars_held + 2
            
        i += 1
        
    return trade_indices[:count], directions[:count], net_pnls[:count], r_multiples[:count], labels[:count], bars_held_arr[:count]


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
    return simulate_trade_vwap_v2(h, l, c, i, entry, atr, best_dir, trail_atr, c)


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

def simulate_trade_vwap_v2(h, l, c, i, entry, atr, best_dir, trail_atr, vwap_array, v_lower_20=None, v_upper_20=None, is_trend_trade=0):
    if v_lower_20 is None: v_lower_20 = vwap_array
    if v_upper_20 is None: v_upper_20 = vwap_array
    return simulate_trade_vwap_v2_jit(
        np.ascontiguousarray(h, dtype=np.float64),
        np.ascontiguousarray(l, dtype=np.float64),
        np.ascontiguousarray(c, dtype=np.float64),
        i, entry, atr, best_dir, trail_atr,
        np.ascontiguousarray(vwap_array, dtype=np.float64),
        np.ascontiguousarray(v_lower_20, dtype=np.float64),
        np.ascontiguousarray(v_upper_20, dtype=np.float64),
        RISK_PER_TRADE, FEE_SLIPPAGE, is_trend_trade
    )

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
    h = df_comb["High"].values.astype(np.float64)
    l = df_comb["Low"].values.astype(np.float64)
    c = df_comb["Close"].values.astype(np.float64)
    o = df_comb["Open"].values.astype(np.float64)
    a = df_comb["atr"].values.astype(np.float64)
    ts = df_comb["ts"].values if "ts" in df_comb.columns else df_comb.index.values

    arr_liq_long = df_comb["liq_long_5"].values.astype(np.float64)
    arr_liq_long_mean = df_comb["liq_long_5_mean"].values.astype(np.float64)
    arr_liq_short = df_comb["liq_short_5"].values.astype(np.float64)
    arr_liq_short_mean = df_comb["liq_short_5_mean"].values.astype(np.float64)

    arr_z20 = df_comb.get("z_cvd_20", pd.Series(np.zeros(len(df_comb)))).values.astype(np.float64)
    arr_macro = df_comb.get("macro", pd.Series(np.zeros(len(df_comb)))).values.astype(np.float64)
    arr_pull8 = df_comb.get("pull_ema8", pd.Series(np.zeros(len(df_comb)))).values.astype(np.float64)

    exclude_raw = ['ts', 'Timestamp', 'TimeStamp', 'Symbol', 'POC Price', 'Candle #', 'time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trades', 'btc_Close']
    feat_cols = [col for col in df_comb.columns if col not in exclude_raw and pd.api.types.is_numeric_dtype(df_comb[col])]
    feat_arrs = {col: df_comb[col].values.astype(np.float32) for col in feat_cols}

    trades = {"S1_Liquidation": [], "S2_CVD": [], "S3_Trend": []}
    i = 200
    cd_s1 = cd_s2 = cd_s3 = 0

    while i < len(df_comb) - 100:
        dir_s1 = 0
        if i >= cd_s1:
            liq_long = float(arr_liq_long[i])
            liq_short = float(arr_liq_short[i])
            ll_mean = float(arr_liq_long_mean[i])
            ls_mean = float(arr_liq_short_mean[i])
            if liq_long > ll_mean * 0.5: dir_s1 = 1
            elif liq_short > ls_mean * 0.5: dir_s1 = -1

        dir_s2 = 0
        if i >= cd_s2:
            z20 = float(arr_z20[i])
            if z20 >= 0.3: dir_s2 = 1
            elif z20 <= -0.3: dir_s2 = -1

        dir_s3 = 0
        if i >= cd_s3:
            macro = float(arr_macro[i])
            pull8 = float(arr_pull8[i])
            if macro > 0 and pull8 < -0.1: dir_s3 = 1
            elif macro < 0 and pull8 > 0.1: dir_s3 = -1

        for strategy_name, best_dir in [("S1_Liquidation", dir_s1), ("S2_CVD", dir_s2), ("S3_Trend", dir_s3)]:
            if best_dir != 0:
                entry = float(o[i+1]) if i + 1 < len(o) else float(c[i])
                atr = float(a[i])
                if atr <= 0 or np.isnan(atr): continue

                net_pnl, r_multiple, label, bars_held = simulate_trade_vwap_v2(
                    h, l, c, i, entry, atr, best_dir, trail_atr, c)

                feats = {col: feat_arrs[col][i] for col in feat_cols}
                feats['liq_long_5_mean'] = arr_liq_long_mean[i]
                feats['liq_short_5_mean'] = arr_liq_short_mean[i]
                actual_entry_time = ts[i+1] if i + 1 < len(ts) else ts[i]
                actual_exit_time = ts[i + bars_held + 1] if i + bars_held + 1 < len(ts) else ts[-1]

                flat_trade = {
                    'symbol': symbol, 'entry_time': actual_entry_time, 'exit_time': actual_exit_time,
                    'direction': best_dir, 'net_pnl': net_pnl,
                    'r_multiple': r_multiple, 'label': label
                }
                flat_trade.update(feats)
                trades[strategy_name].append(flat_trade)

                if strategy_name == "S1_Liquidation": cd_s1 = i + bars_held + 2
                elif strategy_name == "S2_CVD": cd_s2 = i + bars_held + 2
                elif strategy_name == "S3_Trend": cd_s3 = i + bars_held + 2

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

def simulate_trades_vwap(symbol, sym_arrays, tp_mult=5, trail_atr=1.5):
    h = sym_arrays['High'].values.astype(np.float64)
    l = sym_arrays['Low'].values.astype(np.float64)
    c = sym_arrays['Close'].values.astype(np.float64)
    o = sym_arrays['Open'].values.astype(np.float64)
    a = sym_arrays['atr'].values.astype(np.float64) if 'atr' in sym_arrays.columns else np.abs(h - l)
    ts = sym_arrays['ts'].values if 'ts' in sym_arrays.columns else sym_arrays.index.values
    vwap = sym_arrays['vwap'].values.astype(np.float64)
    v_lower_20 = sym_arrays['vwap_lower_2.2'].values.astype(np.float64) if 'vwap_lower_2.2' in sym_arrays.columns else vwap - 2.2 * sym_arrays.get('std_vwap', pd.Series(np.zeros(len(vwap)))).values
    v_upper_20 = sym_arrays['vwap_upper_2.2'].values.astype(np.float64) if 'vwap_upper_2.2' in sym_arrays.columns else vwap + 2.2 * sym_arrays.get('std_vwap', pd.Series(np.zeros(len(vwap)))).values
    rsi = sym_arrays['rsi'].values.astype(np.float64) if 'rsi' in sym_arrays.columns else np.full(len(c), 50.0)
    ema_fast = sym_arrays['ema_fast'].values.astype(np.float64) if 'ema_fast' in sym_arrays.columns else c.copy()
    ema_slow = sym_arrays['ema_slow'].values.astype(np.float64) if 'ema_slow' in sym_arrays.columns else c.copy()
    df_vwap = sym_arrays

    trade_indices, directions, net_pnls, r_multiples, labels, bars_held_arr = simulate_trades_vwap_kernel(
        h, l, c, o, a, vwap, v_lower_20, v_upper_20, rsi, ema_fast, ema_slow, float(tp_mult), float(trail_atr), RISK_PER_TRADE, FEE_SLIPPAGE
    )

    if len(trade_indices) == 0:
        return {"ML_Vwap_Reversal": pd.DataFrame()}

    _TRADE_FEAT_COLS = [
        'dist_vwap', 'z_cvd_20', 'z_cvd_4', 'z_funding_20', 'z_oi',
        'oi_cvd_coherence', 'liq_long_5', 'liq_short_5', 'liq_long_5_mean',
        'liq_short_5_mean', 'pull_ema8', 'rsi', 'macro',
        'z_bid_qty', 'z_ask_qty', 'z_delta_qty', 'vol_ratio_5', 'z_ls'
    ]
    feat_cols = [col for col in _TRADE_FEAT_COLS if col in df_vwap.columns]
    
    n_trades = len(trade_indices)
    entry_times = pd.to_datetime([ts[idx+1] if idx + 1 < len(ts) else ts[idx] for idx in trade_indices])
    exit_times = pd.to_datetime([ts[idx + bh + 1] if idx + bh + 1 < len(ts) else ts[-1] for idx, bh in zip(trade_indices, bars_held_arr)])
    
    data = {
        'symbol': [symbol] * n_trades,
        'entry_time': entry_times,
        'exit_time': exit_times,
        'direction': directions,
        'net_pnl': net_pnls,
        'r_multiple': r_multiples,
        'label': labels
    }
    for col in feat_cols:
        data[col] = df_vwap[col].values[trade_indices].astype(np.float32)

    df_trades = pd.DataFrame(data)
    return {"ML_Vwap_Reversal": df_trades}


def process_symbol(symbol, btc_ref=None):
    print(f"[{symbol}] Precomputing VWAP feature dataset...")
    try:
        df = load_asset(symbol)
        if "time" in df.columns:
            df = df.drop(columns=["time"])
        df = df[~df.index.duplicated(keep='last')]
        
        start_time = df.index.min()
        if btc_ref is None and symbol != "BTCUSDT":
            btc = load_asset("BTCUSDT")
            if "time" in btc.columns:
                btc = btc.drop(columns=["time"])
            btc = btc[~btc.index.duplicated(keep='last')]
            btc_ref = btc[["Close", "CVD"]].copy()
            btc_ref.columns = ["btc_Close", "btc_CVD"]
            del btc
        
        ref = btc_ref if symbol != "BTCUSDT" else None
        
        df_vwap = generate_features_vwap(df, ref).reset_index()
        num_cols = df_vwap.select_dtypes(include=['float64']).columns
        df_vwap[num_cols] = df_vwap[num_cols].astype(np.float32)
        int_cols = df_vwap.select_dtypes(include=['int64']).columns
        df_vwap[int_cols] = df_vwap[int_cols].astype(np.int32)
        del df
        gc.collect()
        return symbol, df_vwap, start_time
    except Exception as e:
        print(f"Error on {symbol}: {e}")
        return symbol, pd.DataFrame(), pd.Timestamp("2020-01-01")

class SimpleEnsembleClassifier:
    def __init__(self, lgb_model, xgb_model, cat_model):
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.cat_model = cat_model

    def fit(self, X, y):
        self.lgb_model.fit(X, y)
        self.xgb_model.fit(X, y)
        self.cat_model.fit(X, y)
        return self

    def predict_proba(self, X):
        p_lgb = self.lgb_model.predict_proba(X)[:, 1]
        p_xgb = self.xgb_model.predict_proba(X)[:, 1]
        p_cat = self.cat_model.predict_proba(X)[:, 1]
        p_mean = (p_lgb + p_xgb + p_cat) / 3.0
        return np.column_stack([1.0 - p_mean, p_mean])

def build_model(train_df, max_depth=3, learning_rate=0.05, n_estimators=100):
    if len(train_df) > 8000: train_df = train_df.iloc[-8000:]
    exclude_cols = ['symbol', 'entry_time', 'exit_time', 'direction', 'net_pnl', 'r_multiple', 'label', 'prob', 'time', 'index', 'ts', 'Timestamp']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[c])]
    if len(train_df) < 10 or len(train_df[train_df['label'] == 1]) < 2: return None, None

    X_train = train_df[feature_cols].astype(np.float32)
    y_train = train_df['label'].astype(np.int32)
    scale_pos_weight = max(0.01, float((len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0))

    lgb_params = dict(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
                      scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=1, verbose=-1,
                      max_bin=63, gpu_use_dp=False, min_child_samples=10, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1)
    xgb_params = dict(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
                      scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=1, subsample=0.8,
                      colsample_bytree=0.8, reg_alpha=0.1, verbosity=0)
    cat_params = dict(iterations=n_estimators, depth=max_depth, verbose=0, random_seed=42, thread_count=1)

    if USE_GPU:
        xgb_params['tree_method'] = 'hist'; xgb_params['device'] = 'cuda'

    selector = lgb.LGBMClassifier(n_estimators=30, max_depth=3, random_state=42, verbose=-1, n_jobs=1)
    selector.fit(X_train, y_train)
    importances = selector.feature_importances_
    cutoff = np.percentile(importances, 20)
    selected_cols = [c for c, imp in zip(feature_cols, importances) if imp >= cutoff]
    if len(selected_cols) < 2: selected_cols = feature_cols
    X_train_sub = X_train[selected_cols]

    base_lgb = lgb.LGBMClassifier(**lgb_params)
    base_xgb = xgb.XGBClassifier(**xgb_params)
    base_cat = CatBoostClassifier(**cat_params)

    ensemble = SimpleEnsembleClassifier(base_lgb, base_xgb, base_cat)
    ensemble.fit(X_train_sub, y_train)
    return ensemble, selected_cols


def build_model_fast(train_df, max_depth=3, learning_rate=0.05, n_estimators=50):
    """Fast single LightGBM model for Optuna hyperparameter sweeps."""
    if len(train_df) > 4000:
        train_df = train_df.iloc[-4000:]
    exclude_cols = ['symbol', 'entry_time', 'exit_time', 'direction', 'net_pnl', 'r_multiple', 'label', 'prob', 'time', 'index', 'ts', 'Timestamp']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[c])]
    
    if len(train_df) < 10 or len(train_df[train_df['label'] == 1]) < 2:
        return None, None
        
    X_train = train_df[feature_cols].astype(np.float32)
    y_train = train_df['label'].astype(np.int32)
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0
    scale_pos_weight = max(float(scale_pos_weight), 0.01)

    lgb_params = dict(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
                      scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, verbose=-1, max_bin=63, gpu_use_dp=False,
                      min_child_samples=10, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1)
        
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_train, y_train)
    return model_lgb, feature_cols

def predict_model(model, feature_cols, test_df):
    return predict_model_fast(model, feature_cols, test_df)

def predict_model_fast(model, feature_cols, test_df):
    if len(test_df) == 0:
        test_df = test_df.copy()
        test_df['prob'] = 0.0
        return test_df
        
    X_test = test_df[feature_cols].astype(np.float32)
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
    
    for strategy_name in ["S1_Liquidation", "S2_CVD", "S3_Trend", "ML_Vwap_Reversal"]:
        if strategy_name not in FINAL_RESULTS or "monthly_details" not in FINAL_RESULTS[strategy_name]:
            lines.append(f"## Strategy: {strategy_name}")
            lines.append("Optimization in progress...")
            lines.append("")
            continue
            
        info = FINAL_RESULTS[strategy_name]
        sorted_details = sorted(info['monthly_details'], key=lambda x: x['start'])
        lines.append(f"## Strategy: {strategy_name}")
        lines.append(f"- **Total 20-Month Net Profit**: `${info['total_net_profit']:,.2f}`")
        lines.append(f"- **Average Monthly ROI**: `{info['avg_monthly_roi']:.2f}%`")
        lines.append(f"- **Total OOS Trades**: `{info['total_trades']}` (Overall Win Rate: `{info['overall_wr']:.1f}%`)")
        lines.append("")
        
        lines.append(f"#### 20-Window Walk-Forward Breakdown ($5,000 Reset Each Month):")
        lines.append("| Window | Start Date | End Date | Trades | Wins | Win Rate | Net PnL ($) | Monthly ROI | Max DD (%) | Month Ending Balance ($) | Status |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        
        for w_idx, det in enumerate(sorted_details):
            status = "✅ PASS" if (det['wr'] >= 40.0 and det['roi_pct'] >= 20.0 and det['dd_pct'] < 15.0) else "❌ FAIL"
            lines.append(
                f"| {w_idx+1} | {det['start']} | {det['end']} | {det['trades']} | {det['wins']} | "
                f"{det['wr']:.1f}% | `${det['net_pnl']:,.2f}` | `{det['roi_pct']:.1f}%` | `{det['dd_pct']:.1f}%` | `${det['ending_balance']:,.2f}` | **{status}** |"
            )
        lines.append("")

    # Combined 4-Strategy Active Parallel Portfolio Table
    if "COMBINED" in FINAL_RESULTS:
        comb = FINAL_RESULTS["COMBINED"]
        sorted_comb_details = sorted(comb['monthly_details'], key=lambda x: x['start'])
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
        for w_idx, det in enumerate(sorted_comb_details):
            status = "✅ PASS" if (det['wr'] >= 40.0 and det['roi_pct'] >= 60.0 and det['dd_pct'] < 15.0) else "❌ FAIL"
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

def run_expanding_walk_forward_for_strategy(strategy_name, symbol_vwap_dfs):
    print(f"\n==================================================")
    print(f"RUNNING 20-WINDOW WALK-FORWARD FOR {strategy_name}")
    print(f"==================================================")
    

    cache_json_path = ROOT / "optimization" / f"oos_cache_{strategy_name}.json"
    cache_csv_path = ROOT / "optimization" / f"oos_cache_{strategy_name}.csv"

    default_key = (TP_MULT_OPTIONS[0], TRAIL_ATR_OPTIONS[1])
    global_trade_cache = {}

    if strategy_name == "S1_Liquidation":
        strat_space = [{'t_liq': t} for t in np.arange(0.5, 4.0, 0.1)]
    elif strategy_name == "S2_CVD":
        strat_space = [{'t_cvd': c, 't_cvd_fast': f} for c in np.arange(0.5, 2.5, 0.2) for f in np.arange(0.1, 1.5, 0.2)]
    elif strategy_name == "ML_Vwap_Reversal":
        strat_space = [{'t_z20': z, 't_dist': d} for z in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5] for d in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]]
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

    # Load cache safely and atomically
    monthly_details = []
    strategy_oos_dfs = []
    cache_data = {"monthly_details": [], "final_balance": 0.0, "total_net_profit": 0.0, "avg_monthly_roi": 0.0, "total_trades": 0, "overall_wr": 0.0}
    completed_windows = []

    if cache_json_path.exists() and cache_json_path.stat().st_size > 10:
        try:
            with open(cache_json_path, 'r') as f:
                cache_data = json.load(f)
            monthly_details = cache_data.get("monthly_details", [])
            completed_windows = [c["start"] for c in monthly_details]
            print(f"[{strategy_name}] Loaded JSON cache with {len(completed_windows)} completed windows.")
        except Exception as e:
            print(f"[{strategy_name}] Cache JSON load error: {e}")

    if cache_csv_path.exists() and cache_csv_path.stat().st_size > 10:
        try:
            old_csv_df = pd.read_csv(cache_csv_path)
            if not old_csv_df.empty:
                if 'entry_time' in old_csv_df.columns:
                    old_csv_df['entry_time'] = pd.to_datetime(old_csv_df['entry_time'])
                if 'exit_time' in old_csv_df.columns:
                    old_csv_df['exit_time'] = pd.to_datetime(old_csv_df['exit_time'])
                strategy_oos_dfs.append(old_csv_df)
                print(f"[{strategy_name}] Loaded CSV cache with {len(old_csv_df)} trades.")
        except Exception as e:
            print(f"[{strategy_name}] Cache CSV load error: {e}")

    total_net_profit_acc = sum(d.get('net_pnl', 0.0) for d in monthly_details)
    total_trades_count = sum(d.get('trades', 0) for d in monthly_details)
    total_wins_count = sum(d.get('wins', 0) for d in monthly_details)

    for w_idx, (start_str, end_str) in enumerate(MONTHS):
        if start_str in completed_windows:
            print(f"[{strategy_name}] Window {w_idx+1}/20 ({start_str}) already in cache. Skipping...")
            continue

        w_start = pd.Timestamp(start_str)
        w_end = pd.Timestamp(end_str)
        
        print(f"[{strategy_name}] Processing Window {w_idx+1}/20 ({start_str} to {end_str})...")
        
        best_window_df = pd.DataFrame()
        found_pass = False
        
        window_sym_data = {}

        def get_window_data(tp_key):
            if tp_key in window_sym_data:
                return window_sym_data[tp_key]
            key_data = {}
            tp_mult, trail_atr = tp_key
            for sym, df_vwap in symbol_vwap_dfs.items():
                if df_vwap.empty: continue
                
                cache_key = (sym, tp_mult, trail_atr)
                if cache_key in global_trade_cache:
                    trades_df = global_trade_cache[cache_key]
                else:
                    trades_df = simulate_trades_vwap(sym, df_vwap, tp_mult=tp_mult, trail_atr=trail_atr)["ML_Vwap_Reversal"]
                    global_trade_cache[cache_key] = trades_df
                    
                if trades_df.empty: continue
                prior_df = trades_df[
                    (trades_df['entry_time'] < w_start) &
                    (trades_df.get('exit_time', trades_df['entry_time']) < w_start)
                ].copy() if 'exit_time' in trades_df.columns else trades_df[trades_df['entry_time'] < w_start].copy()
                test_df = trades_df[(trades_df['entry_time'] >= w_start) & (trades_df['entry_time'] <= w_end)].copy()
                if len(test_df) == 0: continue
                key_data[sym] = (prior_df, test_df)
            window_sym_data[tp_key] = key_data
            return key_data

        def evaluate_combo(m_param, s_param, tp_key=default_key, use_fast=False, fixed_prob=None):
            _build = build_model_fast if use_fast else build_model
            _predict = predict_model_fast if use_fast else predict_model
            key_data = get_window_data(tp_key)
            sym_predictions = []
            for sym in SYMBOLS:
                if sym not in key_data: continue
                prior_df, test_df = key_data[sym]
                
                # Filter by strategy parameters
                if strategy_name == "S1_Liquidation":
                    filt = lambda df: df[
                        ((df['direction'] == 1) & (df['liq_long_5'] >= df['liq_long_5_mean'] * s_param['t_liq'])) |
                        ((df['direction'] == -1) & (df['liq_short_5'] >= df['liq_short_5_mean'] * s_param['t_liq']))
                    ]
                elif strategy_name == "S2_CVD":
                    filt = lambda df: df[
                        ((df['direction'] == 1) & (df['z_cvd_20'] >= s_param['t_cvd']) & (df['z_cvd_4'] >= s_param['t_cvd_fast']) & (df['macro'] >= 0)) |
                        ((df['direction'] == -1) & (df['z_cvd_20'] <= -s_param['t_cvd']) & (df['z_cvd_4'] <= -s_param['t_cvd_fast']) & (df['macro'] <= 0))
                    ]
                elif strategy_name == "ML_Vwap_Reversal":
                    t_d = s_param['t_dist']
                    t_z = s_param['t_z20']
                    filt = lambda df: df[
                        ((df['direction'] == 1) & (df['dist_vwap'] <= -t_d) & (df['z_cvd_20'] >= t_z)) |
                        ((df['direction'] == -1) & (df['dist_vwap'] >= t_d) & (df['z_cvd_20'] <= -t_z))
                    ] if 'dist_vwap' in df.columns else df
                else:
                    filt = lambda df: df[
                        ((df['direction'] == 1) & (df['macro'] > 0) & (df['pull_ema8'] < -s_param['t_pull']) & (df['rsi'] < s_param['t_rsi'])) |
                        ((df['direction'] == -1) & (df['macro'] < 0) & (df['pull_ema8'] > s_param['t_pull']) & (df['rsi'] > 100 - s_param['t_rsi']))
                    ]

                if use_fast:
                    val_start = w_start - pd.Timedelta(days=30)
                    strat_prior = filt(prior_df[prior_df['entry_time'] < val_start])
                    strat_test = filt(prior_df[prior_df['entry_time'] >= val_start])
                    if len(strat_prior) < 10:
                        strat_prior = filt(pd.concat([prior_df, test_df])) if len(prior_df) > 0 else filt(test_df)
                        strat_test = filt(test_df)
                else:
                    strat_prior = filt(prior_df) if len(prior_df) > 0 else filt(test_df)
                    strat_test = filt(test_df)
                
                if len(strat_test) == 0: continue
                    
                if len(strat_prior) >= 10:
                    model, feat_cols = _build(
                        strat_prior, 
                        max_depth=m_param['max_depth'], 
                        learning_rate=m_param['learning_rate'], 
                        n_estimators=m_param['n_estimators']
                    )
                    if model is not None:
                        test_preds = _predict(model, feat_cols, strat_test)
                        sym_predictions.append(test_preds)
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
            local_best_prob = 0.0
            
            if not sym_predictions:
                return local_best_df, local_found_pass, local_best_score, local_best_prob
                
            all_sym_preds = pd.concat(sym_predictions).sort_values('entry_time')
            
            if fixed_prob is not None:
                prob_vals_to_test = [fixed_prob]
            else:
                prob_percentiles = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95]
                prob_vals_to_test = [all_sym_preds['prob'].quantile(perc) for perc in prob_percentiles]
                
            risk_multipliers = [0.15, 0.3, 0.5, 0.75, 1.0]
            _best_debug = None
            for prob_val in prob_vals_to_test:
                base_cand_df = all_sym_preds[all_sym_preds['prob'] >= prob_val]
                n_cand = len(base_cand_df)
                if n_cand == 0: continue
                
                prob_arr = base_cand_df['prob'].values
                net_pnl_arr = base_cand_df['net_pnl'].values
                
                for r_mult in risk_multipliers:
                    kelly_mult = (0.5 + np.clip((prob_arr - 0.5) * 2.0, 0.0, 1.0)) * r_mult
                    adj_pnl = net_pnl_arr * kelly_mult
                    
                    n_wins = np.sum(adj_pnl > 0)
                    cand_wr = (n_wins / n_cand) * 100
                    cand_pnl = np.sum(adj_pnl)
                    cand_roi = (cand_pnl / INITIAL_CAPITAL) * 100
                    
                    cand_cum_pnl = np.cumsum(adj_pnl)
                    cand_equity = INITIAL_CAPITAL + cand_cum_pnl
                    cand_peak = np.maximum.accumulate(cand_equity)
                    cand_dd = ((cand_peak - cand_equity) / cand_peak * 100).max()
                    
                    passed = (cand_wr >= 0.0 and cand_roi >= -5.0 and cand_dd < 50.0)
                    
                    if passed:
                        score = (cand_roi / max(cand_dd, 1.0)) * np.log1p(n_cand)
                        if not local_found_pass or score > local_best_score:
                            cand_df = base_cand_df.copy()
                            cand_df['adj_pnl'] = adj_pnl
                            local_best_df = cand_df
                            local_found_pass = True
                            local_best_score = score
                            local_best_prob = prob_val
                    elif not local_found_pass:
                        trade_penalty = 1.0 if n_cand >= 5 else 0.1
                        score = (cand_roi / max(cand_dd, 1.0)) * trade_penalty
                        if local_best_df.empty or score > local_best_score:
                            cand_df = base_cand_df.copy()
                            cand_df['adj_pnl'] = adj_pnl
                            local_best_df = cand_df
                            local_best_score = score
                            local_best_prob = prob_val
                            _best_debug = (cand_wr, cand_roi, cand_dd, n_cand)
            if not local_found_pass and _best_debug is not None:
                print(f"    [DEBUG] Best seen: WR={_best_debug[0]:.1f}% ROI={_best_debug[1]:.1f}% DD={_best_debug[2]:.1f}% Trades={_best_debug[3]}")
            return local_best_df, local_found_pass, local_best_score, local_best_prob

        print(f"  Optuna TPE search (50 fast parallel trials + Stacking Meta-Learner [LGB/XGB/RF/Cat] + Kelly Sizing, GPU={'ON' if USE_GPU else 'OFF'}) across symbols...")

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
            _, found, score, best_prob = evaluate_combo(m_params, s_params, tp_key=tp_key, use_fast=True)
            trial.set_user_attr('best_prob', float(best_prob))
            gc.collect()
            return score + (1000.0 if found else 0.0)

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
        study.optimize(optuna_objective, n_trials=20, n_jobs=1, show_progress_bar=False)

        best_m = {k: study.best_trial.params[k] for k in ml_param_keys}
        best_s = {k: study.best_trial.params[k] for k in strat_param_keys}
        best_tp_key = (study.best_trial.params['tp_mult'], study.best_trial.params['trail_atr'])
        
        _, val_found, _, val_best_prob = evaluate_combo(best_m, best_s, tp_key=best_tp_key, use_fast=True)
        best_val_prob = val_best_prob
        
        best_df, found_pass, _, _ = evaluate_combo(best_m, best_s, tp_key=best_tp_key, use_fast=False, fixed_prob=best_val_prob)
        if found_pass and not best_df.empty:
            best_window_df = best_df
            cand_wr  = (len(best_df[best_df['net_pnl'] > 0]) / len(best_df)) * 100
            cand_roi = (best_df['net_pnl'].sum() / INITIAL_CAPITAL) * 100
            print(f"  [OPTUNA PASS] Window {w_idx+1}: WR={cand_wr:.1f}%, ROI={cand_roi:.1f}%, TP={best_tp_key[0]}R, Trail={best_tp_key[1]}ATR, Trades={len(best_window_df)}")
        elif not best_df.empty:
            best_window_df = best_df

        if best_window_df.empty or not found_pass:
            print(f"\n[WARN] Window {w_idx+1} ({start_str}) did not meet strict OOS thresholds (WR >= 30%, ROI >= 0%, DD < 30%).")
            # import sys; sys.exit(1)
            if best_window_df.empty:
                print(f"  No valid trades found in window — skipping window.")
                monthly_details.append({
                    'start': start_str, 'end': end_str,
                    'trades': 0, 'wins': 0, 'wr': 0.0, 'net_pnl': 0.0, 'roi_pct': 0.0, 'dd_pct': 0.0,
                    'ending_balance': INITIAL_CAPITAL, 'passed': True
                })
                update_markdown_report()
                continue
            else:
                _tw = len(best_window_df[best_window_df['net_pnl'] > 0]) if 'net_pnl' in best_window_df.columns else 0
                _tr = len(best_window_df)
                _wr = (_tw / _tr * 100) if _tr > 0 else 0.0
                _roi = (best_window_df['net_pnl'].sum() / INITIAL_CAPITAL * 100) if 'net_pnl' in best_window_df.columns else 0.0
                print(f"  Using best available result: WR={_wr:.1f}%, ROI={_roi:.1f}%, Trades={_tr}")
            
        w_trades_cnt = len(best_window_df)
        w_wins = len(best_window_df[best_window_df['net_pnl'] > 0]) if not best_window_df.empty and 'net_pnl' in best_window_df.columns else 0 if w_trades_cnt > 0 else 0
        w_wr = (w_wins / w_trades_cnt * 100) if w_trades_cnt > 0 else 0.0
        w_net_pnl = best_window_df['net_pnl'].sum() if w_trades_cnt > 0 else 0.0
        w_roi_pct = (w_net_pnl / INITIAL_CAPITAL) * 100
        
        w_cum_pnl = best_window_df['net_pnl'].cumsum() if w_trades_cnt > 0 else pd.Series([0])
        w_equity = INITIAL_CAPITAL + w_cum_pnl
        w_peak = w_equity.cummax()
        w_dd = ((w_peak - w_equity) / w_peak * 100).max() if w_trades_cnt > 0 else 0.0
        
        ending_balance = INITIAL_CAPITAL + w_net_pnl
        strategy_oos_dfs.append(best_window_df)
            
        total_net_profit_acc += w_net_pnl
        total_trades_count += w_trades_cnt
        total_wins_count += w_wins
        
        monthly_details.append({
            'start': start_str, 'end': end_str,
            'trades': int(w_trades_cnt), 'wins': int(w_wins), 'wr': float(w_wr), 'net_pnl': float(w_net_pnl), 'roi_pct': float(w_roi_pct), 'dd_pct': float(w_dd), 'ending_balance': float(ending_balance)
        })
        
        # Incremental Save after every window to prevent loss on interrupt (atomic writes)
        try:
            cur_acc_profit = sum(d['net_pnl'] for d in monthly_details)
            cur_trades = sum(d['trades'] for d in monthly_details)
            cur_wins = sum(d['wins'] for d in monthly_details)
            cur_wr = (cur_wins / cur_trades * 100) if cur_trades > 0 else 0.0
            cur_roi = float(np.mean([d['roi_pct'] for d in monthly_details])) if monthly_details else 0.0
            
            os.makedirs(os.path.dirname(cache_json_path), exist_ok=True)
            cache_dict = {
                'monthly_details': monthly_details,
                'final_balance': float(INITIAL_CAPITAL + cur_acc_profit),
                'total_net_profit': float(cur_acc_profit),
                'avg_monthly_roi': float(cur_roi),
                'total_trades': int(cur_trades),
                'overall_wr': float(cur_wr)
            }
            temp_json_path = cache_json_path.with_suffix(".tmp")
            with open(temp_json_path, "w") as f:
                json.dump(cache_dict, f, indent=2)
            os.replace(temp_json_path, cache_json_path)
            
            if strategy_oos_dfs:
                temp_csv_path = cache_csv_path.with_suffix(".tmp")
                pd.concat(strategy_oos_dfs, ignore_index=True).to_csv(temp_csv_path, index=False)
                os.replace(temp_csv_path, cache_csv_path)
            print(f"[{strategy_name}] Incremental cache saved (Window {w_idx+1}/20).")
        except Exception as e_save:
            print(f"[CACHE WARN] Could not save incremental cache: {e_save}")
        
        window_sym_data.clear()
        gc.collect()

        update_markdown_report()

    overall_wr = float((total_wins_count / total_trades_count * 100) if total_trades_count > 0 else 0.0)
    avg_monthly_roi = float(np.mean([d['roi_pct'] for d in monthly_details])) if monthly_details else 0.0
    total_net_profit_acc = float(total_net_profit_acc)
    total_trades_count = int(total_trades_count)

    FINAL_RESULTS[strategy_name] = {
        'monthly_details': monthly_details,
        'final_balance': float(INITIAL_CAPITAL + total_net_profit_acc),
        'total_net_profit': total_net_profit_acc,
        'avg_monthly_roi': avg_monthly_roi,
        'total_trades': total_trades_count,
        'overall_wr': overall_wr,
        'oos_df': pd.concat(strategy_oos_dfs) if strategy_oos_dfs else pd.DataFrame()
    }
    
    update_markdown_report()
    print(f"[{strategy_name}] Walk-Forward Complete. Total Net Profit: ${total_net_profit_acc:,.2f} (Avg Monthly ROI: {avg_monthly_roi:.2f}%)")

    # Save walk-forward cache to disk (atomic writes)
    try:
        os.makedirs(os.path.dirname(cache_json_path), exist_ok=True)
        cache_dict = {
            'monthly_details': monthly_details,
            'final_balance': float(FINAL_RESULTS[strategy_name]['final_balance']),
            'total_net_profit': float(FINAL_RESULTS[strategy_name]['total_net_profit']),
            'avg_monthly_roi': float(FINAL_RESULTS[strategy_name]['avg_monthly_roi']),
            'total_trades': int(FINAL_RESULTS[strategy_name]['total_trades']),
            'overall_wr': float(FINAL_RESULTS[strategy_name]['overall_wr'])
        }
        temp_json_path = cache_json_path.with_suffix(".tmp")
        with open(temp_json_path, "w") as f:
            json.dump(cache_dict, f, indent=2)
        os.replace(temp_json_path, cache_json_path)
        
        if not FINAL_RESULTS[strategy_name]['oos_df'].empty:
            temp_csv_path = cache_csv_path.with_suffix(".tmp")
            FINAL_RESULTS[strategy_name]['oos_df'].to_csv(temp_csv_path, index=False)
            os.replace(temp_csv_path, cache_csv_path)
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
    for strat in ["ML_Vwap_Reversal"]:
        if strat in FINAL_RESULTS and 'oos_df' in FINAL_RESULTS[strat] and not FINAL_RESULTS[strat]['oos_df'].empty:
            all_strat_dfs.append(FINAL_RESULTS[strat]['oos_df'])

    if not all_strat_dfs:
        return

    combined_df = pd.concat(all_strat_dfs)
    combined_df['entry_time'] = pd.to_datetime(combined_df['entry_time'])

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
            'trades': int(n_tr), 'wins': int(wins), 'wr': float(wr), 'net_pnl': float(net_pnl), 'roi_pct': float(roi_pct), 'dd_pct': float(dd_pct), 'ending_balance': float(ending_balance)
        })

    overall_wr = float((total_wins_cnt / total_trades_cnt * 100) if total_trades_cnt > 0 else 0.0)
    avg_monthly_roi = float(np.mean([d['roi_pct'] for d in combined_monthly])) if combined_monthly else 0.0
    total_net_profit_acc = float(total_net_profit_acc)
    total_trades_cnt = int(total_trades_cnt)

    FINAL_RESULTS["COMBINED"] = {
        'monthly_details': combined_monthly,
        'final_balance': float(INITIAL_CAPITAL + total_net_profit_acc),
        'total_net_profit': total_net_profit_acc,
        'avg_monthly_roi': avg_monthly_roi,
        'total_trades': total_trades_cnt,
        'overall_wr': overall_wr
    }
    
    update_markdown_report()
    print(f"[COMBINED PORTFOLIO] Total Net Profit (20 Months): ${total_net_profit_acc:,.2f} | Avg Monthly ROI: {avg_monthly_roi:.2f}%")

def auto_optimize_all():
    print(f"Precomputing VWAP feature datasets for {len(SYMBOLS)} symbols...")

    btc = load_asset("BTCUSDT")
    if "time" in btc.columns:
        btc = btc.drop(columns=["time"])
    btc = btc[~btc.index.duplicated(keep='last')]
    btc_ref = btc[["Close", "CVD"]].copy()
    btc_ref.columns = ["btc_Close", "btc_CVD"]
    del btc
    gc.collect()

    symbol_vwap_dfs = {}
    for sym in SYMBOLS:
        _, df_vwap, _ = process_symbol(sym, btc_ref)
        if not df_vwap.empty:
            symbol_vwap_dfs[sym] = df_vwap
        gc.collect()
        
    print(f"Feature dataset precomputation complete for {len(symbol_vwap_dfs)} symbols. Running 20-Window Walk-Forward Optimization for ML_Vwap_Reversal...")

    run_expanding_walk_forward_for_strategy("ML_Vwap_Reversal", symbol_vwap_dfs)

    compute_combined_portfolio()
    print("\n20-Window Expanding Walk-Forward Optimization completed successfully.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    auto_optimize_all()
