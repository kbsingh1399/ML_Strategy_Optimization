"""
Live Model Trainer for Engine_1
===================================
Replicates the exact training methodology from standalone strategies (opt_s1 through opt_s6):
- Expanding window (trains on all available historical data prior to now)
- Path-dependent trade simulation with trailing stops (0.8 ATR) and fee deduction (0.15%)
- Rich feature engineering (VWAP bands, SMC Fair Value Gaps, Liquidations, CVD z-scores)
- Feature selection (top 80% importance) + 3-model ensemble (LGBM + XGBoost + CatBoost)
- Atomic model saving with manifest for hot-swapping
"""

import os
import sys
import gc
import json
import logging
from collections import deque
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from numba import njit
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
log = logging.getLogger("live_model_trainer")

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CACHE_DIR = os.path.join(MODEL_DIR, "trade_cache")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT", "LINKUSDT",
    "AVAXUSDT", "SUIUSDT", "NEARUSDT", "DOTUSDT", "LTCUSDT"
]

RISK_PER_TRADE = 50.0
FEE_SLIPPAGE = 0.0015
SL_MULT = 1.0

TP_MULT_OPTIONS = [7.0]
TRAIL_ATR_OPTIONS = [0.8]

# -------------------------------------------------------------------------
# NUMBA HELPER FUNCTIONS
# -------------------------------------------------------------------------
def zscore(s: pd.Series, w: int) -> pd.Series:
    return (s - s.rolling(w, min_periods=1).mean()) / s.rolling(w, min_periods=1).std().replace(0, 1e-10)

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

# -------------------------------------------------------------------------
# FEATURE ENGINEERING PREPROCESSORS
# -------------------------------------------------------------------------
def _add_advanced_features(df: pd.DataFrame, feature_list: list,
                            price_col: str = "Close",
                            cvd_col: str = "CVD") -> None:
    """Add log-returns, rolling higher moments, and temporal lags
    to the DataFrame and extend feature_list in-place."""
    new_feats = []

    # ── Log returns (1-bar, 3-bar, 5-bar) ────────────────────────
    if price_col in df.columns:
        for lag in [1, 3, 5]:
            col = f"log_ret_{lag}"
            df[col] = np.log(df[price_col] / df[price_col].shift(lag).replace(0, np.nan))
            df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)
            new_feats.append(col)

    # ── Rolling higher moments of log_ret_1 (20-bar window) ──────
    if "log_ret_1" in df.columns:
        r = df["log_ret_1"]
        for w in [10, 20]:
            df[f"ret_skew_{w}"] = r.rolling(w, min_periods=5).skew().fillna(0)
            df[f"ret_kurt_{w}"] = r.rolling(w, min_periods=5).kurt().fillna(0)
            new_feats.extend([f"ret_skew_{w}", f"ret_kurt_{w}"])

    # ── ATR ratio (current ATR / 100-bar mean) ───────────────────
    if "atr" in df.columns:
        atr_ma100 = df["atr"].rolling(100, min_periods=10).mean()
        df["atr_ratio"] = (df["atr"] / (atr_ma100 + 1e-10)).clip(0.3, 3.0)
        df["atr_ratio"] = df["atr_ratio"].fillna(1.0)
        new_feats.append("atr_ratio")

    # ── CVD acceleration (2nd derivative) ────────────────────────
    if cvd_col in df.columns:
        cvd_vals = df[cvd_col].ffill()
        df["cvd_accel"] = cvd_vals.diff().diff().fillna(0)
        new_feats.append("cvd_accel")

    # ── Temporal lags: p8_t-1, cvd_d_t-1, atr_ratio_t-1 ─────────
    for src_col, lag_name in [("atr_ratio", "atr_ratio_lag1"),
                               ("log_ret_1", "ret_lag1")]:
        if src_col in df.columns:
            df[lag_name] = df[src_col].shift(1).fillna(0)
            new_feats.append(lag_name)

    # ── Price / ATR ratio (normalized price) ──────────────────────
    if price_col in df.columns and "atr" in df.columns:
        atr_safe = df["atr"].replace(0, 1e-10)
        df["price_atr"] = df[price_col] / atr_safe
        new_feats.append("price_atr")

    feature_list.extend(new_feats)
    df[new_feats] = df[new_feats].fillna(0).replace([np.inf, -np.inf], 0)

def prep_alpha(df: pd.DataFrame, btc_ref: pd.DataFrame = None):
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
    df["cvd_delta"] = df["CVD"].diff(5) if "CVD" in df.columns else 0.0
    df["btc_cvd_mom"] = df["btc_CVD"].diff(2) if "btc_CVD" in df.columns else 0.0
    df["ema_fast"] = df["Close"].ewm(span=200, min_periods=50).mean()
    df["ema_slow"] = df["Close"].ewm(span=800, min_periods=100).mean()
    df["macro_score"] = (df["ema_fast"] - df["ema_slow"]) / df["atr"].replace(0, 1e-10)
    df["macro"] = np.where(df["macro_score"] > 0.5, 1, np.where(df["macro_score"] < -0.5, -1, 0))
    feats = ["macro"]
    for k in [4, 10, 20]:
        df[f"z_cvd_{k}"] = zscore(df["CVD"], k) if "CVD" in df.columns else 0.0
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
    _add_advanced_features(df, feats)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df, feats

def prep_trend(df: pd.DataFrame, btc_ref: pd.DataFrame = None):
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
    df["cvd_delta"] = df["CVD"].diff(5) if "CVD" in df.columns else 0.0
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
        df[f"z_cvd_{k}"] = zscore(df["CVD"], k) if "CVD" in df.columns else 0.0
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
    _add_advanced_features(df, feats)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df, feats

def calculate_vwap_bands(df: pd.DataFrame) -> pd.DataFrame:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    volume = df["Volume"].replace(0, 1e-10) if "Volume" in df.columns else pd.Series(1.0, index=df.index)
    raw_ts = df.index if isinstance(df.index, pd.DatetimeIndex) else (df["ts"] if "ts" in df.columns else (df["TimeStamp"] if "TimeStamp" in df.columns else df["Timestamp"]))
    ts_idx = pd.DatetimeIndex(pd.to_datetime(raw_ts))
    session_group = (ts_idx - pd.Timedelta(hours=5, minutes=30)).floor("D")
    tp_v = typical_price * volume
    tp2_v = (typical_price ** 2) * volume
    cum_tp_v = tp_v.groupby(session_group).cumsum()
    cum_tp2_v = tp2_v.groupby(session_group).cumsum()
    cum_v = volume.groupby(session_group).cumsum()
    vwap = cum_tp_v / cum_v
    variance = (cum_tp2_v / cum_v) - (vwap ** 2)
    std_vwap = np.sqrt(variance.clip(lower=0))
    df["vwap"] = vwap
    df["std_vwap"] = std_vwap
    df["vwap_upper_2.2"] = vwap + 2.2 * std_vwap
    df["vwap_lower_2.2"] = vwap - 2.2 * std_vwap
    df["vwap_upper_3.3"] = vwap + 3.3 * std_vwap
    df["vwap_lower_3.3"] = vwap - 3.3 * std_vwap
    return df

def prep_vwap(df: pd.DataFrame, btc_ref: pd.DataFrame = None):
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
    df["cvd_delta"] = df["CVD"].diff(5) if "CVD" in df.columns else 0.0
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
        df[f"z_cvd_{k}"] = zscore(df["CVD"], k) if "CVD" in df.columns else 0.0
        df[f"z_btc_{k}"] = zscore(df["btc_CVD"], k) if "btc_CVD" in df.columns else 0.0
    df["vol_regime"] = zscore(df["atr"], 100)
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
        df["z_oi"] = zscore(oi, 100)
        df["oi_delta_5"] = oi.diff(5) / (oi.shift(5) + 1e-10)
        df["oi_cvd_coherence"] = np.sign(df["oi_delta_5"].fillna(0)) * np.sign(df["cvd_delta"].fillna(0))
        feats.extend(["z_oi", "oi_delta_5", "oi_cvd_coherence"])
    if "Long/Short Ratio (Account)" in df.columns:
        df["z_ls"] = zscore(pd.to_numeric(df["Long/Short Ratio (Account)"], errors="coerce").ffill(), 100)
        feats.append("z_ls")
    if "Agg. Funding Rate" in df.columns:
        funding = pd.to_numeric(df["Agg. Funding Rate"], errors="coerce").fillna(0)
        df["funding"] = funding
        df["z_funding_20"] = zscore(funding, 20)
        feats.extend(["funding", "z_funding_20"])
    fp_cols = ["Bid Qty", "Ask Qty", "Delta Qty", "Bid Trades", "Ask Trades"]
    for col in fp_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df[f"z_{col.replace(' ', '_').lower()}"] = zscore(df[col], 10)
            feats.extend([col, f"z_{col.replace(' ', '_').lower()}"])
    df[feats] = df[feats].fillna(0).astype(np.float32)
    _add_advanced_features(df, feats)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df, feats

# -------------------------------------------------------------------------
# DATA LOADING FUNCTION
# -------------------------------------------------------------------------
_GDRIVE_PARQUET = r"G:\My Drive\_Trading_Data\15m\parquet"
_LOCAL_PARQUET   = os.path.join(BASE_DIR, "backtesting_data")

def load_asset(symbol: str) -> pd.DataFrame:
    search_dirs = [
        _GDRIVE_PARQUET,
        _LOCAL_PARQUET,
        os.path.join(BASE_DIR, "Seeding"),
        BASE_DIR
    ]
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
        # Fallback to combined_seed_history.xlsx if present
        seed_excel = os.path.join(BASE_DIR, "Seeding", "combined_seed_history.xlsx")
        if os.path.exists(seed_excel):
            try:
                import openpyxl
                wb = openpyxl.load_workbook(seed_excel, read_only=True)
                sheet_names = wb.sheetnames
                wb.close()
                matching_sheet = None
                for s in sheet_names:
                    if s.upper() == symbol.upper() or symbol.upper().startswith(s.upper()):
                        matching_sheet = s
                        break
                if matching_sheet:
                    df_s = pd.read_excel(seed_excel, sheet_name=matching_sheet)
                    ts_col = "TimeStamp" if "TimeStamp" in df_s.columns else ("Timestamp" if "Timestamp" in df_s.columns else "ts")
                    df_s["ts"] = pd.to_datetime(df_s[ts_col].astype(str).str.replace(" IST", "", regex=False), errors="coerce")
                    df_s = df_s.sort_values("ts")
                    for c in df_s.columns:
                        if c != "ts":
                            df_s[c] = pd.to_numeric(df_s[c], errors="coerce").astype(np.float32)
                    df_s = df_s.set_index("ts")
                    return df_s
            except Exception as exc:
                print(f"[LoadAsset] Failed to load {symbol} from Excel seed: {exc}")
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
        df = pd.merge_asof(df_s.sort_values("ts"), df_f.sort_values("ts"), on="ts", direction="nearest", tolerance=pd.Timedelta(minutes=5))
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

# -------------------------------------------------------------------------
# TRADE SIMULATION (LABEL GENERATION)
# -------------------------------------------------------------------------
def simulate_trade(h, l, c, i, entry, atr, best_dir, tp_mult=7.0, trail_atr=0.8):
    sl_dist = SL_MULT * atr
    tp_dist = tp_mult * atr
    initial_sl = entry - sl_dist if best_dir == 1 else entry + sl_dist
    tp = entry + tp_dist if best_dir == 1 else entry - tp_dist
    trail_dist = trail_atr * atr

    limit = min(i + 96 + 1, len(c))
    current_sl = initial_sl
    best_seen = entry
    exit_price = float(c[limit - 1])
    bars_held = limit - 1 - i

    for j in range(i + 1, limit):
        if best_dir == 1:
            if float(h[j]) > best_seen:
                best_seen = float(h[j])
                if (best_seen - entry) >= 5.0 * sl_dist:
                    new_trail = best_seen - trail_dist
                    if new_trail > current_sl:
                        current_sl = new_trail
            if float(l[j]) <= current_sl:
                bars_held = j - i; exit_price = current_sl; break
            if float(h[j]) >= tp:
                bars_held = j - i; exit_price = tp; break
        else:
            if float(l[j]) < best_seen:
                best_seen = float(l[j])
                if (entry - best_seen) >= 5.0 * sl_dist:
                    new_trail = best_seen + trail_dist
                    if new_trail < current_sl:
                        current_sl = new_trail
            if float(h[j]) >= current_sl:
                bars_held = j - i; exit_price = current_sl; break
            if float(l[j]) <= tp:
                bars_held = j - i; exit_price = tp; break

    units = RISK_PER_TRADE / sl_dist
    gross_pnl = units * (exit_price - entry) if best_dir == 1 else units * (entry - exit_price)
    fee_cost = units * entry * (FEE_SLIPPAGE / 2.0) + units * abs(exit_price) * (FEE_SLIPPAGE / 2.0)
    net_pnl = gross_pnl - fee_cost
    r_multiple = net_pnl / RISK_PER_TRADE
    label = 1 if net_pnl > 0 else 0
    return net_pnl, r_multiple, label, bars_held

def generate_features_standard(df, btc_ref):
    if 'Bid Qty' in df.columns and 'Ask Qty' in df.columns:
        df['Delta Qty'] = df['Bid Qty'] - df['Ask Qty']
    df_alpha, alpha_feats = prep_alpha(df, btc_ref)
    df_trend, trend_feats = prep_trend(df, btc_ref)
    df_comb = df_trend
    for col in alpha_feats:
        if col not in df_comb.columns:
            df_comb[col] = df_alpha[col]
    df_comb["liq_long_5_mean"] = df_comb["liq_long_5"].rolling(100).mean().fillna(0)
    df_comb["liq_short_5_mean"] = df_comb["liq_short_5"].rolling(100).mean().fillna(0)
    return df_comb

def simulate_trades_standard(symbol, df_comb, tp_mult=7.0, trail_atr=0.8):
    h = df_comb["High"].values
    l = df_comb["Low"].values
    c = df_comb["Close"].values
    o = df_comb["Open"].values
    a = df_comb["atr"].values
    ts = df_comb.index.values

    arr_liq_long_mean = df_comb["liq_long_5_mean"].values
    arr_liq_short_mean = df_comb["liq_short_5_mean"].values
    arr_z20 = df_comb.get("z_cvd_20", pd.Series(np.zeros(len(df_comb)))).values
    arr_macro = df_comb.get("macro", pd.Series(np.zeros(len(df_comb)))).values
    arr_pull8 = df_comb.get("pull_ema8", pd.Series(np.zeros(len(df_comb)))).values
    arr_z_delta_qty = df_comb.get("z_delta_qty", pd.Series(np.zeros(len(df_comb)))).values
    arr_z_bid_qty = df_comb.get("z_bid_qty", pd.Series(np.zeros(len(df_comb)))).values
    arr_z_ask_qty = df_comb.get("z_ask_qty", pd.Series(np.zeros(len(df_comb)))).values
    arr_stoch_k = df_comb.get("stoch_k", pd.Series(np.zeros(len(df_comb)))).values
    arr_pull50 = df_comb.get("pull_ema50", pd.Series(np.zeros(len(df_comb)))).values
    arr_cvd_delta = df_comb.get("cvd_delta", pd.Series(np.zeros(len(df_comb)))).values

    exclude_raw = ['ts', 'Timestamp', 'TimeStamp', 'Symbol', 'POC Price', 'Candle #', 'time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trades', 'btc_Close']
    feat_cols = [col for col in df_comb.columns if col not in exclude_raw and pd.api.types.is_numeric_dtype(df_comb[col])]
    feat_arrs = {col: df_comb[col].values for col in feat_cols}

    trades = {"S1_Liquidation": [], "S2_CVD": [], "S3_Trend": [], "S5_Microstructure": [], "S6_SMC_Orderflow": []}
    i = 200
    cd_s1 = cd_s2 = cd_s3 = cd_s5 = cd_s6 = 0

    while i < len(df_comb) - 100:
        dir_s1 = 0
        if i >= cd_s1:
            pull8 = float(arr_pull8[i])
            if pull8 < -0.2: dir_s1 = 1
            elif pull8 > 0.2: dir_s1 = -1

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

        dir_s5 = 0
        if i >= cd_s5:
            z_delta = float(arr_z_delta_qty[i])
            z_bid = float(arr_z_bid_qty[i])
            z_ask = float(arr_z_ask_qty[i])
            if z_delta > 0.5 and z_bid > z_ask: dir_s5 = 1
            elif z_delta < -0.5 and z_ask > z_bid: dir_s5 = -1

        dir_s6 = 0
        if i >= cd_s6:
            stk = float(arr_stoch_k[i])
            p50 = float(arr_pull50[i])
            cvd_d = float(arr_cvd_delta[i])
            if stk < 20 and p50 < 0 and cvd_d > 0: dir_s6 = 1
            elif stk > 80 and p50 > 0 and cvd_d < 0: dir_s6 = -1

        for strategy_name, best_dir in [("S1_Liquidation", dir_s1), ("S2_CVD", dir_s2), ("S3_Trend", dir_s3), ("S5_Microstructure", dir_s5), ("S6_SMC_Orderflow", dir_s6)]:
            if best_dir != 0:
                entry = float(o[i+1]) if i + 1 < len(o) else float(c[i])
                atr = float(a[i])
                if atr <= 0 or np.isnan(atr): continue

                net_pnl, r_multiple, label, bars_held = simulate_trade(
                    h, l, c, i, entry, atr, best_dir, tp_mult, trail_atr)

                feats = {col: feat_arrs[col][i] for col in feat_cols}
                feats['liq_long_5_mean'] = arr_liq_long_mean[i]
                feats['liq_short_5_mean'] = arr_liq_short_mean[i]
                actual_entry_time = ts[i+1] if i + 1 < len(ts) else ts[i]

                flat_trade = {
                    'symbol': symbol, 'entry_time': actual_entry_time,
                    'direction': best_dir, 'net_pnl': net_pnl,
                    'r_multiple': r_multiple, 'label': label
                }
                flat_trade.update(feats)
                trades[strategy_name].append(flat_trade)

                if strategy_name == "S1_Liquidation": cd_s1 = i + bars_held + 2
                elif strategy_name == "S2_CVD": cd_s2 = i + bars_held + 2
                elif strategy_name == "S3_Trend": cd_s3 = i + bars_held + 2
                elif strategy_name == "S5_Microstructure": cd_s5 = i + bars_held + 2
                elif strategy_name == "S6_SMC_Orderflow": cd_s6 = i + bars_held + 2

        i += 1

    return trades


# -------------------------------------------------------------------------
# ENSEMBLE MODEL CLASSIFIER & BUILDER
# -------------------------------------------------------------------------
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

def build_model(train_df, max_depth=4, learning_rate=0.03, n_estimators=200):
    if len(train_df) > 10000:
        train_df = train_df.iloc[-10000:]
    exclude_cols = ['symbol', 'entry_time', 'direction', 'net_pnl', 'r_multiple', 'label']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    if len(train_df) < 10 or len(train_df[train_df['label'] == 1]) < 2:
        return None, None

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

    # Feature selection: prune bottom 20%
    selector = lgb.LGBMClassifier(n_estimators=30, max_depth=3, random_state=42, verbose=-1, n_jobs=1)
    selector.fit(X_train, y_train)
    importances = selector.feature_importances_
    cutoff = np.percentile(importances, 20)
    selected_cols = [c for c, imp in zip(feature_cols, importances) if imp >= cutoff]
    if len(selected_cols) < 2:
        selected_cols = feature_cols
    X_train_sub = X_train[selected_cols]

    base_lgb = lgb.LGBMClassifier(**lgb_params)
    base_xgb = xgb.XGBClassifier(**xgb_params)
    base_cat = CatBoostClassifier(**cat_params)

    ensemble = SimpleEnsembleClassifier(base_lgb, base_xgb, base_cat)
    ensemble.fit(X_train_sub, y_train)
    return ensemble, selected_cols

# -------------------------------------------------------------------------
# MAIN EXPANDING-WINDOW TRAINER FOR LIVE DEPLOYMENT
# -------------------------------------------------------------------------
def build_model_fast(train_df, max_depth=3, learning_rate=0.05, n_estimators=50):
    if len(train_df) > 4000:
        train_df = train_df.iloc[-4000:]
    exclude_cols = ['symbol', 'entry_time', 'direction', 'net_pnl', 'r_multiple', 'label']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    if len(train_df) < 10 or len(train_df[train_df['label'] == 1]) < 2:
        return None, None
    X_train = train_df[feature_cols].astype(np.float32)
    y_train = train_df['label'].astype(np.int32)
    scale_pos_weight = max(0.01, float((len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0))

    selector = lgb.LGBMClassifier(n_estimators=30, max_depth=3, random_state=42, verbose=-1, n_jobs=4)
    selector.fit(X_train, y_train)
    importances = selector.feature_importances_
    cutoff = np.percentile(importances, 20)
    selected_cols = [c for c, imp in zip(feature_cols, importances) if imp >= cutoff]
    if len(selected_cols) < 2:
        selected_cols = feature_cols
    X_train_sub = X_train[selected_cols]

    lgb_params = dict(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
                      scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=4, verbose=-1, max_bin=63)
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_train_sub, y_train)
    return model_lgb, selected_cols

def predict_model_fast(model, feature_cols, test_df):
    if len(test_df) == 0:
        test_df = test_df.copy()
        test_df['prob'] = 0.0
        return test_df
    X_test = test_df[feature_cols].astype(float)
    test_df = test_df.copy()
    test_df['prob'] = model.predict_proba(X_test)[:, 1]
    return test_df


@njit(fastmath=True, nogil=True)
def compute_fvg_and_sweeps(high: np.ndarray, low: np.ndarray, close: np.ndarray, sweep_lookback: int=15):
    n = len(close)
    bullish_fvg = np.zeros(n, dtype=np.int32)
    bearish_fvg = np.zeros(n, dtype=np.int32)
    bullish_sweep = np.zeros(n, dtype=np.int32)
    bearish_sweep = np.zeros(n, dtype=np.int32)
    for i in range(2, n):
        if low[i] > high[i - 2] and close[i - 1] > high[i - 2]:
            bullish_fvg[i] = 1
        if high[i] < low[i - 2] and close[i - 1] < low[i - 2]:
            bearish_fvg[i] = 1
        start_idx = max(0, i - sweep_lookback)
        rmax = high[start_idx]
        rmin = low[start_idx]
        for k in range(start_idx + 1, i):
            if high[k] > rmax:
                rmax = high[k]
            if low[k] < rmin:
                rmin = low[k]
        if high[i] > rmax and close[i] < rmax:
            bearish_sweep[i] = 1
        if low[i] < rmin and close[i] > rmin:
            bullish_sweep[i] = 1
    return (bullish_fvg, bearish_fvg, bullish_sweep, bearish_sweep)

def prep_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates advanced microstructure features such as:
    - CVD Divergence
    - Liquidation Cascades and Acceleration
    - Volatility Coiling
    """
    df['atr'] = (df['High'] - df['Low']).rolling(14, min_periods=1).mean()
    atr_safe = df['atr'].replace(0, 1e-10)
    df['vol_regime'] = (df['atr'] - df['atr'].rolling(100, min_periods=1).mean()) / df['atr'].rolling(100, min_periods=1).std().replace(0, 1e-10)
    if 'CVD' in df.columns:
        df['cvd_delta'] = df['CVD'].diff(3)
        df['cvd_accel'] = df['cvd_delta'].diff()
        low_5 = df['Low'].rolling(5).min()
        cvd_5 = df['CVD'].rolling(5).min()
        df['cvd_divergence_bull'] = (df['Low'] == low_5) & (df['CVD'] > cvd_5)
        high_5 = df['High'].rolling(5).max()
        cvd_5_max = df['CVD'].rolling(5).max()
        df['cvd_divergence_bear'] = (df['High'] == high_5) & (df['CVD'] < cvd_5_max)
    if 'Agg. Liq Long' in df.columns:
        df['liq_long'] = df['Agg. Liq Long'].fillna(0)
        df['liq_long_mean'] = df['liq_long'].rolling(1440, min_periods=100).mean().fillna(0)
        df['liq_long_delta'] = df['liq_long'].diff().fillna(0)
    else:
        df['liq_long'] = 0
        df['liq_long_mean'] = 1
        df['liq_long_delta'] = 0
    if 'Agg. Liq Short' in df.columns:
        df['liq_short'] = df['Agg. Liq Short'].fillna(0)
        df['liq_short_mean'] = df['liq_short'].rolling(1440, min_periods=100).mean().fillna(0)
        df['liq_short_delta'] = df['liq_short'].diff().fillna(0)
    else:
        df['liq_short'] = 0
        df['liq_short_mean'] = 1
        df['liq_short_delta'] = 0
    if 'Delta Qty' in df.columns:
        dq = df['Delta Qty'].fillna(0)
        df['delta_qty_z'] = (dq - dq.rolling(20, min_periods=1).mean()) / dq.rolling(20, min_periods=1).std().replace(0, 1e-10)
    else:
        df['delta_qty_z'] = 0.0
    if 'Bid Qty' in df.columns and 'Ask Qty' in df.columns:
        total_qty = df['Bid Qty'].fillna(0) + df['Ask Qty'].fillna(0)
        df['bid_ask_ratio'] = df['Bid Qty'].fillna(0) / total_qty.replace(0, 1e-10) - 0.5
    else:
        df['bid_ask_ratio'] = 0.0
    return df

def prep_smc(df: pd.DataFrame, btc_ref: pd.DataFrame=None):
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
        rng = np.maximum(h - l, 1e-06)
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
    df = df.assign(bull_fvg=bull_fvg, bear_fvg=bear_fvg, bull_sweep=bull_sweep, bear_sweep=bear_sweep, atr=atr, delta=delta, z_delta=z_delta, z_cvd=z_cvd, atr_stretch=atr_stretch)
    feat_cols = ['bull_fvg', 'bear_fvg', 'bull_sweep', 'bear_sweep', 'delta', 'z_delta', 'z_cvd', 'atr_stretch']
    _add_advanced_features(df, feat_cols)
    return (df, feat_cols)

@njit(fastmath=True)
def simulate_trade_vwap_v2_jit(h, l, c, i, entry, atr, best_dir, trail_atr, vwap_array, v_lower_20, v_upper_20, risk_per_trade, fee_slippage, is_trend_trade):
    sl_dist = 1.0 * atr
    initial_sl = entry - sl_dist if best_dir == 1 else entry + sl_dist
    trail_dist = trail_atr * atr

    limit = min(i + 96 + 1, len(c))
    current_sl = initial_sl
    best_seen = entry
    exit_price = c[limit - 1]
    bars_held = limit - 1 - i

    for j in range(i + 1, limit):
        v = vwap_array[j]
        vu = v_upper_20[j]
        vl = v_lower_20[j]
        if best_dir == 1:
            if h[j] > best_seen:
                best_seen = h[j]
                if (best_seen - entry) >= 5.0 * sl_dist:
                    new_trail = best_seen - trail_dist
                    if new_trail > current_sl:
                        current_sl = new_trail
            if l[j] <= current_sl:
                bars_held = j - i; exit_price = current_sl; break
            
            target = vu if is_trend_trade == 1 else v
            if h[j] >= target:
                bars_held = j - i; exit_price = target; break
        else:
            if l[j] < best_seen:
                best_seen = l[j]
                if (entry - best_seen) >= 5.0 * sl_dist:
                    new_trail = best_seen + trail_dist
                    if new_trail < current_sl:
                        current_sl = new_trail
            if h[j] >= current_sl:
                bars_held = j - i; exit_price = current_sl; break
            
            target = vl if is_trend_trade == 1 else v
            if l[j] <= target:
                bars_held = j - i; exit_price = target; break

    units = risk_per_trade / sl_dist
    gross_pnl = units * (exit_price - entry) if best_dir == 1 else units * (entry - exit_price)
    fee_cost = units * entry * (fee_slippage / 2.0) + units * abs(exit_price) * (fee_slippage / 2.0)
    net_pnl = gross_pnl - fee_cost
    r_multiple = net_pnl / risk_per_trade
    label = 1 if net_pnl > 0 else 0
    return net_pnl, r_multiple, label, bars_held

@njit(fastmath=True)
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
                h, l, c, i, entry, atr, best_dir, trail_atr, vwap, v_lower_20, v_upper_20, risk_per_trade, fee_slippage, is_trend_trade
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

def simulate_trade_vwap_v2(h, l, c, i, entry, atr, best_dir, trail_atr, vwap_array, v_lower_20=None, v_upper_20=None, is_trend_trade=0):
    if v_lower_20 is None:
        v_lower_20 = vwap_array
    if v_upper_20 is None:
        v_upper_20 = vwap_array
    return simulate_trade_vwap_v2_jit(np.ascontiguousarray(h, dtype=np.float64), np.ascontiguousarray(l, dtype=np.float64), np.ascontiguousarray(c, dtype=np.float64), i, entry, atr, best_dir, trail_atr, np.ascontiguousarray(vwap_array, dtype=np.float64), np.ascontiguousarray(v_lower_20, dtype=np.float64), np.ascontiguousarray(v_upper_20, dtype=np.float64), RISK_PER_TRADE, FEE_SLIPPAGE, is_trend_trade)

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
    trade_indices, directions, net_pnls, r_multiples, labels, bars_held_arr = simulate_trades_vwap_kernel(h, l, c, o, a, vwap, v_lower_20, v_upper_20, rsi, ema_fast, ema_slow, float(tp_mult), float(trail_atr), RISK_PER_TRADE, FEE_SLIPPAGE)
    if len(trade_indices) == 0:
        return {'ML_Vwap_Reversal': pd.DataFrame()}
    _TRADE_FEAT_COLS = ['dist_vwap', 'z_cvd_20', 'z_cvd_4', 'z_funding_20', 'z_oi', 'oi_cvd_coherence', 'liq_long_5', 'liq_short_5', 'liq_long_5_mean', 'liq_short_5_mean', 'pull_ema8', 'rsi', 'macro', 'z_bid_qty', 'z_ask_qty', 'z_delta_qty', 'vol_ratio_5', 'z_ls']
    feat_cols = [col for col in _TRADE_FEAT_COLS if col in df_vwap.columns]
    n_trades = len(trade_indices)
    entry_times = pd.to_datetime([ts[idx + 1] if idx + 1 < len(ts) else ts[idx] for idx in trade_indices])
    exit_times = pd.to_datetime([ts[idx + bh + 1] if idx + bh + 1 < len(ts) else ts[-1] for idx, bh in zip(trade_indices, bars_held_arr)])
    data = {'symbol': [symbol] * n_trades, 'entry_time': entry_times, 'exit_time': exit_times, 'direction': directions, 'net_pnl': net_pnls, 'r_multiple': r_multiples, 'label': labels}
    for col in feat_cols:
        data[col] = df_vwap[col].values[trade_indices].astype(np.float32)
    df_trades = pd.DataFrame(data)
    return {'ML_Vwap_Reversal': df_trades}

def simulate_trades_microstructure(symbol, df_vwap, tp_mult=5, trail_atr=1.5):
    h = df_vwap['High'].values
    l = df_vwap['Low'].values
    c = df_vwap['Close'].values
    o = df_vwap['Open'].values
    a = df_vwap['atr'].values
    ts = df_vwap.index.values
    vwap = df_vwap['vwap'].values if 'vwap' in df_vwap.columns else c
    v_lower_20 = df_vwap['vwap_lower_2.2'].values if 'vwap_lower_2.2' in df_vwap.columns else vwap - 2.2 * a
    v_upper_20 = df_vwap['vwap_upper_2.2'].values if 'vwap_upper_2.2' in df_vwap.columns else vwap + 2.2 * a
    ema_fast = df_vwap['ema_fast'].values if 'ema_fast' in df_vwap.columns else c
    ema_slow = df_vwap['ema_slow'].values if 'ema_slow' in df_vwap.columns else c
    rsi = df_vwap['rsi'].values if 'rsi' in df_vwap.columns else np.full(len(c), 50.0)
    exclude_raw = ['ts', 'Timestamp', 'TimeStamp', 'Symbol', 'POC Price', 'Candle #', 'time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trades', 'btc_Close']
    feat_cols = [col for col in df_vwap.columns if col not in exclude_raw and pd.api.types.is_numeric_dtype(df_vwap[col])]
    feat_arrs = {col: df_vwap[col].values.astype(np.float32) for col in feat_cols}
    trades = {'S5_Microstructure': []}
    i = 200
    cd = 0
    while i < len(df_vwap) - 100:
        if i < cd:
            i += 1
            continue
        best_dir = 0
        ef = ema_fast[i]
        es = ema_slow[i]
        if l[i] <= v_lower_20[i] and rsi[i] < 45:
            best_dir = 1
        elif l[i] <= vwap[i] + 0.3 * a[i] and ef > es:
            best_dir = 1
        elif h[i] >= v_upper_20[i] and rsi[i] > 55:
            best_dir = -1
        elif h[i] >= vwap[i] - 0.3 * a[i] and ef < es:
            best_dir = -1
        if best_dir != 0:
            entry = float(o[i + 1]) if i + 1 < len(o) else float(c[i])
            atr = float(a[i])
            if atr <= 0 or np.isnan(atr):
                i += 1
                continue
            net_pnl, r_multiple, label, bars_held = simulate_trade_vwap_v2(h, l, c, i, entry, atr, best_dir, trail_atr, vwap)
            feats = {col: feat_arrs[col][i] for col in feat_cols}
            actual_entry_time = ts[i + 1] if i + 1 < len(ts) else ts[i]
            flat_trade = {'symbol': symbol, 'entry_time': actual_entry_time, 'direction': best_dir, 'net_pnl': net_pnl, 'r_multiple': r_multiple, 'label': label}
            flat_trade.update(feats)
            trades['S5_Microstructure'].append(flat_trade)
            cd = i + bars_held + 2
        i += 1
    return trades

def simulate_trades_smc(symbol, df_smc, tp_mult=5, trail_atr=1.5):
    h = df_smc['High'].values
    l = df_smc['Low'].values
    c = df_smc['Close'].values
    o = df_smc['Open'].values
    a = df_smc['atr'].values
    ts = df_smc.index.values
    vwap = df_smc['vwap'].values if 'vwap' in df_smc.columns else c
    v_lower_20 = df_smc['vwap_lower_2.2'].values if 'vwap_lower_2.2' in df_smc.columns else vwap - 2.2 * a
    v_upper_20 = df_smc['vwap_upper_2.2'].values if 'vwap_upper_2.2' in df_smc.columns else vwap + 2.2 * a
    ema_fast = df_smc['ema_fast'].values if 'ema_fast' in df_smc.columns else c
    ema_slow = df_smc['ema_slow'].values if 'ema_slow' in df_smc.columns else c
    rsi = df_smc['rsi'].values if 'rsi' in df_smc.columns else np.full(len(c), 50.0)
    exclude_raw = ['ts', 'Timestamp', 'TimeStamp', 'Symbol', 'POC Price', 'Candle #', 'time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trades', 'btc_Close']
    feat_cols = [col for col in df_smc.columns if col not in exclude_raw and pd.api.types.is_numeric_dtype(df_smc[col])]
    feat_arrs = {col: df_smc[col].values.astype(np.float32) for col in feat_cols}
    trades = {'S6_SMC_Orderflow': []}
    i = 200
    cd = 0
    while i < len(df_smc) - 100:
        if i < cd:
            i += 1
            continue
        best_dir = 0
        ef = ema_fast[i]
        es = ema_slow[i]
        if l[i] <= v_lower_20[i] and rsi[i] < 45:
            best_dir = 1
        elif l[i] <= vwap[i] + 0.3 * a[i] and ef > es:
            best_dir = 1
        elif h[i] >= v_upper_20[i] and rsi[i] > 55:
            best_dir = -1
        elif h[i] >= vwap[i] - 0.3 * a[i] and ef < es:
            best_dir = -1
        if best_dir != 0:
            entry = float(o[i + 1]) if i + 1 < len(o) else float(c[i])
            atr = float(a[i])
            if atr <= 0 or np.isnan(atr):
                i += 1
                continue
            net_pnl, r_multiple, label, bars_held = simulate_trade_vwap_v2(h, l, c, i, entry, atr, best_dir, tp_mult, trail_atr, vwap)
            feats = {col: feat_arrs[col][i] for col in feat_cols}
            actual_entry_time = ts[i + 1] if i + 1 < len(ts) else ts[i]
            flat_trade = {'symbol': symbol, 'entry_time': actual_entry_time, 'direction': best_dir, 'net_pnl': net_pnl, 'r_multiple': r_multiple, 'label': label}
            flat_trade.update(feats)
            trades['S6_SMC_Orderflow'].append(flat_trade)
            cd = i + bars_held + 2
        i += 1
    return trades

def train_all_strategies():
    print("=" * 60)
    print("LIVE MODEL TRAINER - UNIFIED STANDALONE REPLICATION (MULTI-STAGE)")
    print("=" * 60)
    
    print("\n[1/3] Loading BTC reference data...")
    btc = load_asset("BTCUSDT")
    btc_ref = None
    if not btc.empty:
        btc_ref = btc[["Close", "CVD"]].copy() if "CVD" in btc.columns else btc[["Close"]].copy()
        btc_ref.columns = ["btc_Close", "btc_CVD"] if "CVD" in btc.columns else ["btc_Close"]

    print("\n[2/3] Generating simulated trade datasets (Expanding Window)...")
    all_combos = {}
    
    for tp in TP_MULT_OPTIONS:
        for trail in TRAIL_ATR_OPTIONS:
            all_combos[(tp, trail)] = {"S1_Liquidation": [], "S2_CVD": [], "S3_Trend": [], "ML_Vwap_Reversal": [], "S5_Microstructure": [], "S6_SMC_Orderflow": []}

    for sym in SYMBOLS:
        print(f"  Processing {sym}...")
        df = load_asset(sym)
        if df.empty: continue
        ref = btc_ref if sym != "BTCUSDT" else None
        
        df_std = generate_features_standard(df.copy(), ref)
        df_vwap, _ = prep_vwap(df.copy(), ref)
        df_micro = prep_microstructure(df_vwap.copy())
        df_smc, _ = prep_smc(df.copy(), ref)
        
        for tp in TP_MULT_OPTIONS:
            for trail in TRAIL_ATR_OPTIONS:
                trades_std = simulate_trades_standard(sym, df_std, tp_mult=tp, trail_atr=trail)
                trades_vwap = simulate_trades_vwap(sym, df_vwap, tp_mult=tp, trail_atr=trail)
                trades_micro = simulate_trades_microstructure(sym, df_micro, tp_mult=tp, trail_atr=trail)
                trades_smc = simulate_trades_smc(sym, df_smc, tp_mult=tp, trail_atr=trail)
                
                all_combos[(tp, trail)]["S1_Liquidation"].extend(trades_std["S1_Liquidation"])
                all_combos[(tp, trail)]["S2_CVD"].extend(trades_std["S2_CVD"])
                all_combos[(tp, trail)]["S3_Trend"].extend(trades_std["S3_Trend"])
                all_combos[(tp, trail)]["ML_Vwap_Reversal"].extend(trades_vwap["ML_Vwap_Reversal"])
                all_combos[(tp, trail)]["S5_Microstructure"].extend(trades_micro["S5_Microstructure"])
                all_combos[(tp, trail)]["S6_SMC_Orderflow"].extend(trades_smc["S6_SMC_Orderflow"])
                
        del df_std, df_vwap, df_micro, df_smc, df
        gc.collect()

    print("\n[3/3] Training Standalone Ensemble Models with Optuna TPE Hyperparameter Tuning...")
    manifest = {}
    m_path_main = os.path.join(MODEL_DIR, "manifest.json")
    if os.path.exists(m_path_main):
        try:
            with open(m_path_main, 'r') as f: manifest.update(json.load(f))
        except: pass
    
    tp_model_dir = os.path.join(BASE_DIR, "ml_trend_pull", "models")
    liq_model_dir = os.path.join(BASE_DIR, "Liquidation", "models")
    os.makedirs(tp_model_dir, exist_ok=True)
    os.makedirs(liq_model_dir, exist_ok=True)

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        HAS_OPTUNA = True
    except ImportError:
        HAS_OPTUNA = False
        print("  [INFO] Optuna not found, cannot optimize properly.")

    strategy_spaces = {
        "S1_Liquidation": [{'t_liq': t} for t in [1.5, 2.0, 2.5, 3.0]],
        "S2_CVD": [{'t_cvd': c, 't_cvd_fast': f} for c in [1.0, 1.5, 2.0] for f in [0.5, 1.0]],
        "S3_Trend": [{'t_pull': p, 't_rsi': r} for p in [0.2, 0.5, 1.0] for r in [40, 45, 50]],
        "ML_Vwap_Reversal": [{'t_z20': z, 't_vol': v} for z in [1.0, 1.5, 2.0] for v in [0.0, 0.5, 1.0]],
        "S5_Microstructure": [{"t_vol": v, "t_delta": d} for v in [-1.0, 0.0, 1.0, 3.0, 99.0] for d in [0.0, 0.3, 0.6, 1.0]],
        "S6_SMC_Orderflow": [{'t_stoch': s, 't_cvd_d': c} for s in [20, 25, 30] for c in [0, 50, 100]]
    }
    
    ml_space = [
        {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 100},
        {'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 200},
        {'max_depth': 5, 'learning_rate': 0.01, 'n_estimators': 300}
    ]
    prob_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

    precomputed_dfs = {}
    for (tp, trail), strats in all_combos.items():
        precomputed_dfs[(tp, trail)] = {}
        for s_name, raw_trades in strats.items():
            precomputed_dfs[(tp, trail)][s_name] = {}
            if raw_trades:
                df = pd.DataFrame(raw_trades)
                if 'symbol' in df.columns:
                    for sym_key in SYMBOLS:
                        precomputed_dfs[(tp, trail)][s_name][sym_key] = df[df['symbol'] == sym_key]
                else:
                    for sym_key in SYMBOLS:
                        precomputed_dfs[(tp, trail)][s_name][sym_key] = pd.DataFrame()
            else:
                for sym_key in SYMBOLS:
                    precomputed_dfs[(tp, trail)][s_name][sym_key] = pd.DataFrame()


    for strat_name, strat_space in strategy_spaces.items():
        print(f"\n--- Strategy: {strat_name} ---")
        
        for sym in SYMBOLS:
            best_score = -999.0
            best_m_params = ml_space[0]
            best_s_params = strat_space[0]
            best_tp = TP_MULT_OPTIONS[0]
            best_trail = TRAIL_ATR_OPTIONS[0]
            best_prob = 0.6
            
            if HAS_OPTUNA:
                def objective(trial):
                    m_idx = trial.suggest_categorical('m_idx', range(len(ml_space)))
                    s_idx = trial.suggest_categorical('s_idx', range(len(strat_space)))
                    tp_val = trial.suggest_categorical('tp_mult', TP_MULT_OPTIONS)
                    trail_val = trial.suggest_categorical('trail_atr', TRAIL_ATR_OPTIONS)
                    
                    m_param = ml_space[m_idx]
                    s_param = strat_space[s_idx]
                    
                    sym_df = precomputed_dfs[(tp_val, trail_val)][strat_name][sym]
                    if sym_df.empty or len(sym_df) < 20: return -999.0
                    
                    if strat_name == "S1_Liquidation":
                        filt_df = sym_df[
                            ((sym_df['direction'] == 1) & (sym_df['liq_long_5'] >= sym_df['liq_long_5_mean'] * s_param['t_liq'])) |
                            ((sym_df['direction'] == -1) & (sym_df['liq_short_5'] >= sym_df['liq_short_5_mean'] * s_param['t_liq']))
                        ]
                    elif strat_name == "S2_CVD":
                        filt_df = sym_df[
                            ((sym_df['direction'] == 1) & (sym_df['z_cvd_20'] >= s_param['t_cvd']) & (sym_df['z_cvd_4'] >= s_param['t_cvd_fast']) & (sym_df['macro'] >= 0)) |
                            ((sym_df['direction'] == -1) & (sym_df['z_cvd_20'] <= -s_param['t_cvd']) & (sym_df['z_cvd_4'] <= -s_param['t_cvd_fast']) & (sym_df['macro'] <= 0))
                        ]
                    elif strat_name == "ML_Vwap_Reversal":
                        filt_df = sym_df[
                            ((sym_df['direction'] == 1) & (sym_df['z_cvd_20'] >= -s_param['t_z20']) & (sym_df.get('vol_regime', 0) >= s_param['t_vol'])) |
                            ((sym_df['direction'] == -1) & (sym_df['z_cvd_20'] <= s_param['t_z20']) & (sym_df.get('vol_regime', 0) >= s_param['t_vol']))
                        ]
                    elif strat_name == "S3_Trend":
                        filt_df = sym_df[
                            ((sym_df['direction'] == 1) & (sym_df['macro'] > 0) & (sym_df['pull_ema8'] < -s_param['t_pull']) & (sym_df.get('rsi', 50) < s_param['t_rsi'])) |
                            ((sym_df['direction'] == -1) & (sym_df['macro'] < 0) & (sym_df['pull_ema8'] > s_param['t_pull']) & (sym_df.get('rsi', 50) > 100 - s_param['t_rsi']))
                        ]
                    elif strat_name == "S5_Microstructure":
                        filt_df = sym_df[
                            (sym_df.get('vol_regime', 0) <= s_param['t_vol']) & (
                                ((sym_df['direction'] == 1) & (sym_df.get('z_delta_qty', 0) >= s_param['t_delta'])) |
                                ((sym_df['direction'] == -1) & (sym_df.get('z_delta_qty', 0) <= -s_param['t_delta']))
                            )
                        ]
                    elif strat_name == "S6_SMC_Orderflow":
                        filt_df = sym_df[
                            ((sym_df['direction'] == 1) & (sym_df.get('stoch_k', 50) < s_param['t_stoch']) & (sym_df.get('cvd_delta', 0) > s_param['t_cvd_d'])) |
                            ((sym_df['direction'] == -1) & (sym_df.get('stoch_k', 50) > 100 - s_param['t_stoch']) & (sym_df.get('cvd_delta', 0) < -s_param['t_cvd_d']))
                        ]
                    else:
                        filt_df = pd.DataFrame()
                        
                    if len(filt_df) < 10: return -999.0
                    
                    split_idx = int(len(filt_df) * 0.7)
                    train_part = filt_df.iloc[:split_idx]
                    test_part = filt_df.iloc[split_idx:]
                    if len(train_part) < 5 or len(test_part) < 2: return -999.0
                    
                    m, cols = build_model_fast(train_part, **m_param)
                    if m is None: return -999.0
                    
                    preds = predict_model_fast(m, cols, test_part)
                    
                    local_best = -999.0
                    for prob in prob_thresholds:
                        high_conf = preds[preds['prob'] >= prob]
                        if len(high_conf) < 2: continue
                        km = 0.5 + np.clip((high_conf['prob'] - 0.5) * 2.0, 0.0, 1.0)
                        high_conf = high_conf.copy()
                        high_conf['adj_pnl'] = high_conf['net_pnl'] * km
                        wr = (sum(high_conf['adj_pnl'] > 0) / len(high_conf)) * 100
                        pnl = high_conf['adj_pnl'].sum()
                        score = (pnl * wr) / 100.0
                        if score > local_best:
                            local_best = score
                            
                    return local_best

                study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(objective, n_trials=15, n_jobs=4, show_progress_bar=False)
                
                if study.best_trial and study.best_value > -990.0:
                    best_m_params = ml_space[study.best_trial.params['m_idx']]
                    best_s_params = strat_space[study.best_trial.params['s_idx']]
                    best_tp = study.best_trial.params['tp_mult']
                    best_trail = study.best_trial.params['trail_atr']
                    
                    raw_trades = all_combos[(best_tp, best_trail)][strat_name]
                    df_t = pd.DataFrame(raw_trades)
                    sym_df = df_t[df_t['symbol'] == sym]
                    
                    if strat_name == "S1_Liquidation":
                        filt_df = sym_df[
                            ((sym_df['direction'] == 1) & (sym_df['liq_long_5'] >= sym_df['liq_long_5_mean'] * best_s_params['t_liq'])) |
                            ((sym_df['direction'] == -1) & (sym_df['liq_short_5'] >= sym_df['liq_short_5_mean'] * best_s_params['t_liq']))
                        ]
                    elif strat_name == "S2_CVD":
                        filt_df = sym_df[
                            ((sym_df['direction'] == 1) & (sym_df['z_cvd_20'] >= best_s_params['t_cvd']) & (sym_df['z_cvd_4'] >= best_s_params['t_cvd_fast']) & (sym_df['macro'] >= 0)) |
                            ((sym_df['direction'] == -1) & (sym_df['z_cvd_20'] <= -best_s_params['t_cvd']) & (sym_df['z_cvd_4'] <= -best_s_params['t_cvd_fast']) & (sym_df['macro'] <= 0))
                        ]
                    elif strat_name == "ML_Vwap_Reversal":
                        filt_df = sym_df[
                            ((sym_df['direction'] == 1) & (sym_df['z_cvd_20'] >= -best_s_params['t_z20']) & (sym_df.get('vol_regime', 0) >= best_s_params['t_vol'])) |
                            ((sym_df['direction'] == -1) & (sym_df['z_cvd_20'] <= best_s_params['t_z20']) & (sym_df.get('vol_regime', 0) >= best_s_params['t_vol']))
                        ]
                    elif strat_name == "S3_Trend":
                        filt_df = sym_df[
                            ((sym_df['direction'] == 1) & (sym_df['macro'] > 0) & (sym_df['pull_ema8'] < -best_s_params['t_pull']) & (sym_df.get('rsi', 50) < best_s_params['t_rsi'])) |
                            ((sym_df['direction'] == -1) & (sym_df['macro'] < 0) & (sym_df['pull_ema8'] > best_s_params['t_pull']) & (sym_df.get('rsi', 50) > 100 - best_s_params['t_rsi']))
                        ]
                    elif strat_name == "S5_Microstructure":
                        filt_df = sym_df[
                            ((sym_df['direction'] == 1) & (sym_df.get('z_delta_qty', 0) >= best_s_params['t_zdelta']) & (sym_df.get('z_bid_qty', 0) > sym_df.get('z_ask_qty', 0))) |
                            ((sym_df['direction'] == -1) & (sym_df.get('z_delta_qty', 0) <= -best_s_params['t_zdelta']) & (sym_df.get('z_ask_qty', 0) > sym_df.get('z_bid_qty', 0)))
                        ]
                    elif strat_name == "S6_SMC_Orderflow":
                        filt_df = sym_df[
                            ((sym_df['direction'] == 1) & (sym_df.get('stoch_k', 50) < best_s_params['t_stoch']) & (sym_df.get('pull_ema50', 0) < 0) & (sym_df.get('cvd_delta', 0) > best_s_params['t_cvd_d'])) |
                            ((sym_df['direction'] == -1) & (sym_df.get('stoch_k', 50) > 100 - best_s_params['t_stoch']) & (sym_df.get('pull_ema50', 0) > 0) & (sym_df.get('cvd_delta', 0) < -best_s_params['t_cvd_d']))
                        ]
                    else:
                        filt_df = pd.DataFrame()
                    
                    split_idx = int(len(filt_df) * 0.7)
                    train_part = filt_df.iloc[:split_idx]
                    test_part = filt_df.iloc[split_idx:]
                    m, cols = build_model_fast(train_part, **best_m_params)
                    if m is not None:
                        preds = predict_model_fast(m, cols, test_part)
                        best_local = -999.0
                        for prob in prob_thresholds:
                            hc = preds[preds['prob'] >= prob]
                            if len(hc) < 2: continue
                            km = 0.5 + np.clip((hc['prob'] - 0.5) * 2.0, 0.0, 1.0)
                            hc = hc.copy()
                            hc['adj_pnl'] = hc['net_pnl'] * km
                            score = hc['adj_pnl'].sum() * (sum(hc['adj_pnl'] > 0) / len(hc))
                            if score > best_local:
                                best_local = score
                                best_prob = prob
                    
                    print(f"  [OPTUNA OK] {strat_name} / {sym}: tp={best_tp}, trail={best_trail}, prob>={best_prob}")
            
            raw_trades = all_combos[(best_tp, best_trail)][strat_name]
            if not raw_trades: continue
            df_t = pd.DataFrame(raw_trades)
            if df_t.empty or 'symbol' not in df_t.columns: continue
            sym_df = df_t[df_t['symbol'] == sym]
            if len(sym_df) < 10: continue
            
            if strat_name == "S1_Liquidation":
                filt_df = sym_df[
                    ((sym_df['direction'] == 1) & (sym_df['liq_long_5'] >= sym_df['liq_long_5_mean'] * best_s_params['t_liq'])) |
                    ((sym_df['direction'] == -1) & (sym_df['liq_short_5'] >= sym_df['liq_short_5_mean'] * best_s_params['t_liq']))
                ]
            elif strat_name == "S2_CVD":
                filt_df = sym_df[
                    ((sym_df['direction'] == 1) & (sym_df['z_cvd_20'] >= best_s_params['t_cvd']) & (sym_df['z_cvd_4'] >= best_s_params['t_cvd_fast']) & (sym_df['macro'] >= 0)) |
                    ((sym_df['direction'] == -1) & (sym_df['z_cvd_20'] <= -best_s_params['t_cvd']) & (sym_df['z_cvd_4'] <= -best_s_params['t_cvd_fast']) & (sym_df['macro'] <= 0))
                ]
            elif strat_name == "ML_Vwap_Reversal":
                filt_df = sym_df[
                    ((sym_df['direction'] == 1) & (sym_df['z_cvd_20'] >= -best_s_params['t_z20']) & (sym_df.get('vol_regime', 0) >= best_s_params['t_vol'])) |
                    ((sym_df['direction'] == -1) & (sym_df['z_cvd_20'] <= best_s_params['t_z20']) & (sym_df.get('vol_regime', 0) >= best_s_params['t_vol']))
                ]
            elif strat_name == "S3_Trend":
                filt_df = sym_df[
                    ((sym_df['direction'] == 1) & (sym_df['macro'] > 0) & (sym_df['pull_ema8'] < -best_s_params['t_pull']) & (sym_df.get('rsi', 50) < best_s_params['t_rsi'])) |
                    ((sym_df['direction'] == -1) & (sym_df['macro'] < 0) & (sym_df['pull_ema8'] > best_s_params['t_pull']) & (sym_df.get('rsi', 50) > 100 - best_s_params['t_rsi']))
                ]
            elif strat_name == "S5_Microstructure":
                filt_df = sym_df[
                    ((sym_df['direction'] == 1) & (sym_df.get('z_delta_qty', 0) >= best_s_params['t_delta']) & (sym_df.get('z_bid_qty', 0) > sym_df.get('z_ask_qty', 0))) |
                    ((sym_df['direction'] == -1) & (sym_df.get('z_delta_qty', 0) <= -best_s_params['t_delta']) & (sym_df.get('z_ask_qty', 0) > sym_df.get('z_bid_qty', 0)))
                ]
            elif strat_name == "S6_SMC_Orderflow":
                filt_df = sym_df[
                    ((sym_df['direction'] == 1) & (sym_df.get('stoch_k', 50) < best_s_params['t_stoch']) & (sym_df.get('pull_ema50', 0) < 0) & (sym_df.get('cvd_delta', 0) > best_s_params['t_cvd_d'])) |
                    ((sym_df['direction'] == -1) & (sym_df.get('stoch_k', 50) > 100 - best_s_params['t_stoch']) & (sym_df.get('pull_ema50', 0) > 0) & (sym_df.get('cvd_delta', 0) < -best_s_params['t_cvd_d']))
                ]
            else:
                filt_df = pd.DataFrame()
                
            ensemble, selected_cols = build_model(filt_df, **best_m_params)
            if ensemble is None:
                print(f"  [WARN] {strat_name} / {sym}: Insufficient labels for full model.")
                continue

            lgb_tmp = os.path.join(MODEL_DIR, f"tmp_{strat_name}_{sym}_lgb.txt")
            xgb_tmp = os.path.join(MODEL_DIR, f"tmp_{strat_name}_{sym}_xgb.json")
            cb_tmp = os.path.join(MODEL_DIR, f"tmp_{strat_name}_{sym}_cb.cbm")
            cols_tmp = os.path.join(MODEL_DIR, f"tmp_{strat_name}_{sym}_cols.json")

            ensemble.lgb_model.booster_.save_model(lgb_tmp)
            ensemble.xgb_model.save_model(xgb_tmp)
            ensemble.cat_model.save_model(cb_tmp)
            with open(cols_tmp, 'w') as f: json.dump(selected_cols, f)

            os.replace(lgb_tmp, os.path.join(MODEL_DIR, f"{strat_name}_{sym}_lgb.txt"))
            os.replace(xgb_tmp, os.path.join(MODEL_DIR, f"{strat_name}_{sym}_xgb.json"))
            os.replace(cb_tmp, os.path.join(MODEL_DIR, f"{strat_name}_{sym}_cb.cbm"))
            os.replace(cols_tmp, os.path.join(MODEL_DIR, f"{strat_name}_{sym}_cols.json"))

            if strat_name in ["S2_CVD", "S3_Trend"]:
                for side in ["long", "short"]:
                    tp_lgb = os.path.join(tp_model_dir, f"tmp_{strat_name}_{sym}_{side}_lgb.txt")
                    tp_xgb = os.path.join(tp_model_dir, f"tmp_{strat_name}_{sym}_{side}_xgb.json")
                    tp_cb = os.path.join(tp_model_dir, f"tmp_{strat_name}_{sym}_{side}_cb.cbm")
                    ensemble.lgb_model.booster_.save_model(tp_lgb)
                    ensemble.xgb_model.save_model(tp_xgb)
                    ensemble.cat_model.save_model(tp_cb)
                    os.replace(tp_lgb, os.path.join(tp_model_dir, f"{strat_name}_{sym}_{side}_lgb.txt"))
                    os.replace(tp_xgb, os.path.join(tp_model_dir, f"{strat_name}_{sym}_{side}_xgb.json"))
                    os.replace(tp_cb, os.path.join(tp_model_dir, f"{strat_name}_{sym}_{side}_cb.cbm"))

            if strat_name == "S1_Liquidation":
                lq_lgb = os.path.join(liq_model_dir, f"tmp_{strat_name}_{sym}_lgb.txt")
                lq_xgb = os.path.join(liq_model_dir, f"tmp_{strat_name}_{sym}_xgb.json")
                lq_cb = os.path.join(liq_model_dir, f"tmp_{strat_name}_{sym}_cb.cbm")
                lq_cols = os.path.join(liq_model_dir, f"tmp_{strat_name}_{sym}_cols.json")
                ensemble.lgb_model.booster_.save_model(lq_lgb)
                ensemble.xgb_model.save_model(lq_xgb)
                ensemble.cat_model.save_model(lq_cb)
                with open(lq_cols, 'w') as f: json.dump(selected_cols, f)
                os.replace(lq_lgb, os.path.join(liq_model_dir, f"{strat_name}_{sym}_lgb.txt"))
                os.replace(lq_xgb, os.path.join(liq_model_dir, f"{strat_name}_{sym}_xgb.json"))
                os.replace(lq_cb, os.path.join(liq_model_dir, f"{strat_name}_{sym}_cb.cbm"))
                os.replace(lq_cols, os.path.join(liq_model_dir, f"{strat_name}_{sym}_cols.json"))

            manifest[f"{strat_name}_{sym}"] = {
                "trained_at": datetime.utcnow().isoformat(),
                "num_trades": len(filt_df),
                "selected_cols": selected_cols,
                "s_params": best_s_params,
                "tp_mult": best_tp,
                "trail_atr": best_trail,
                "prob_threshold": best_prob
            }
            print(f"  [OK] Saved {strat_name} / {sym} full model with optimized params.")

    for m_dir in [MODEL_DIR, tp_model_dir, liq_model_dir]:
        m_path = os.path.join(m_dir, "manifest.json")
        m_tmp = os.path.join(m_dir, "tmp_manifest.json")
        with open(m_tmp, 'w') as f:
            json.dump(manifest, f, indent=4)
        os.replace(m_tmp, m_path)
        
        param_path = os.path.join(m_dir, "optimized_params.json")
        p_tmp = os.path.join(m_dir, "tmp_optimized_params.json")
        with open(p_tmp, 'w') as f:
            json.dump(manifest, f, indent=4)
        os.replace(p_tmp, param_path)

    print("\n[SUCCESS] Live Model Training complete (Multi-Stage Optimized).")

class OnlineModelUpdater:
    """Incremental model updater using LightGBM refit().

    Every `update_every_bars` (default 96 = ~24h of 15m candles),
    the model is refit on the most recent `window_bars` candles.
    This adapts to regime shifts without full retraining.

    Usage in EnsembleStrategyPredictor._run_inference():
        self.online_updater[symbol].on_new_candle(close, high, low, atr, features)
    """
    def __init__(self, symbol: str, model_dir: str = None,
                 update_every_bars: int = 96,
                 window_bars: int = 500,
                 strategy_name: str = "Ensemble_6Strategy",
                 label_horizon: int = 96):
        self.symbol = symbol
        self.strategy_name = strategy_name
        self.update_every_bars = update_every_bars
        self.window_bars = window_bars
        self.label_horizon = label_horizon
        self.model_dir = model_dir or os.path.join(
            os.path.dirname(__file__), "models")
        self.bars_since_update: int = 0
        self._feature_buffer: deque = deque(maxlen=window_bars)
        self._label_buffer: deque = deque(maxlen=window_bars)
        self._model: Optional[lgb.Booster] = None
        self._feature_cols: List[str] = []
        
        # Pending queue to compute labels after label_horizon bars
        # List of dicts: {"close": float, "atr": float, "features": dict, "highs": List[float], "lows": List[float]}
        self._pending_queue: List[dict] = []
        
        self._load_or_init_model()

    def _load_or_init_model(self):
        """Load existing model from disk or train from buffer."""
        try:
            path = os.path.join(
                self.model_dir,
                f"{self.strategy_name}_{self.symbol}_online_lgb.txt")
            if os.path.exists(path):
                self._model = lgb.Booster(model_file=path)
                cols_path = path.replace("_lgb.txt", "_cols.json")
                if os.path.exists(cols_path):
                    with open(cols_path) as f:
                        self._feature_cols = json.load(f)
                log.info(f"[OnlineUpdater] Loaded existing model for "
                         f"{self.symbol}: {len(self._feature_cols)} features")
        except Exception as e:
            log.warning(f"[OnlineUpdater] Could not load model: {e}")

    def on_new_candle(self, close: float, high: float, low: float, atr: float, features: dict):
        """Feed a new closed candle and resolve labels for older candles."""
        # 1. Update all pending candles with the high/low of this new candle
        for item in self._pending_queue:
            item["highs"].append(high)
            item["lows"].append(low)
            
        # 2. Add the new candle to the pending queue
        self._pending_queue.append({
            "close": close,
            "atr": atr,
            "features": features,
            "highs": [],
            "lows": []
        })
        
        # 3. Check if the oldest candle in queue is now resolved (has label_horizon future bars)
        while self._pending_queue and len(self._pending_queue[0]["highs"]) >= self.label_horizon:
            oldest = self._pending_queue.pop(0)
            label = self._calculate_triple_barrier_label(oldest)
            
            # Append features and label together to maintain alignment
            self._feature_buffer.append(oldest["features"])
            self._label_buffer.append(label)
            
            self.bars_since_update += 1
            if self.bars_since_update >= self.update_every_bars:
                self._refit()
                self.bars_since_update = 0

    def _calculate_triple_barrier_label(self, item: dict) -> int:
        """Compute triple barrier label: 1 if TP hit first, else 0."""
        close = item["close"]
        atr = item["atr"]
        if atr <= 0:
            return 0
            
        tp_barrier = close + 2.0 * atr
        sl_barrier = close - 1.0 * atr
        
        # Check highs and lows in future path
        for h_val, l_val in zip(item["highs"], item["lows"]):
            if h_val >= tp_barrier:
                return 1
            if l_val <= sl_barrier:
                return 0
        return 0  # Time exit / no barrier hit

    def _refit(self):
        """Refit the model on the in-memory buffer window."""
        if len(self._feature_buffer) < 50:
            return
        if len(self._label_buffer) < 10:
            return

        X = pd.DataFrame(list(self._feature_buffer))
        y = pd.Series(list(self._label_buffer))

        # Align to same length
        min_len = min(len(X), len(y))
        X = X.iloc[-min_len:].reset_index(drop=True)
        y = y.iloc[-min_len:].reset_index(drop=True)

        # Keep only numeric features
        self._feature_cols = [c for c in X.columns
                              if pd.api.types.is_numeric_dtype(X[c])]
        if len(self._feature_cols) < 2:
            return

        X_sub = X[self._feature_cols].astype(np.float32)
        pos_weight = max(1, int((len(y) - y.sum()) / max(y.sum(), 1)))

        try:
            if self._model is not None:
                # Incremental refit (keeps tree structure, updates leaves)
                train_data = lgb.Dataset(
                    X_sub, label=y,
                    feature_name=self._feature_cols)
                self._model = lgb.train(
                    {'objective': 'binary', 'verbose': -1,
                     'scale_pos_weight': pos_weight,
                     'num_leaves': 31, 'learning_rate': 0.02,
                     'max_depth': 4, 'min_child_samples': 10,
                     'subsample': 0.8, 'colsample_bytree': 0.8,
                     'reg_alpha': 0.1, 'n_jobs': 1},
                    train_data,
                    num_boost_round=20,
                    init_model=self._model,
                    keep_training_booster=True)
            else:
                # First train
                train_data = lgb.Dataset(
                    X_sub, label=y,
                    feature_name=self._feature_cols)
                self._model = lgb.train(
                    {'objective': 'binary', 'verbose': -1,
                     'scale_pos_weight': pos_weight,
                     'num_leaves': 31, 'learning_rate': 0.03,
                     'max_depth': 4, 'min_child_samples': 10,
                     'n_jobs': 1},
                    train_data,
                    num_boost_round=50)

            # Save to disk
            path = os.path.join(
                self.model_dir,
                f"{self.strategy_name}_{self.symbol}_online_lgb.txt")
            cols_path = path.replace("_lgb.txt", "_cols.json")
            self._model.save_model(path)
            with open(cols_path, 'w') as f:
                json.dump(self._feature_cols, f)

            log.info(f"[OnlineUpdater] {self.symbol}: refit on "
                     f"{min_len} samples, {len(self._feature_cols)} features")
        except Exception as e:
            log.warning(f"[OnlineUpdater] {self.symbol}: refit failed — {e}")

    def predict_proba(self, features: dict) -> Optional[float]:
        """Get probability for a single feature dict."""
        if self._model is None or not self._feature_cols:
            return None
        try:
            X = pd.DataFrame([features])
            for c in self._feature_cols:
                if c not in X.columns:
                    X[c] = 0.0
            X_sub = X[self._feature_cols].astype(np.float32)
            return float(self._model.predict(X_sub)[0])
        except Exception:
            return None

if __name__ == "__main__":
    train_all_strategies()
