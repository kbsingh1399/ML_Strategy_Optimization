# C:\Users\SIGMA\Documents\Project - Coinglass Trading\Engine_1\Engine_1.py
# Production-Grade Coinglass + Binance Footprint Scraper Terminal
# Built from scratch - fully modular, clean, and robust.

from __future__ import annotations
import os
import sys
import time
import json
import asyncio
import signal
import collections
import dataclasses
import threading
import math
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

import aiohttp
import websockets
import socket
from playwright.async_api import async_playwright, Page, BrowserContext

base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(base_dir, "..", ".env"))
from rich.console import Console
from rich.live import Live
from rich.table import Table
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import pandas as pd
import numpy as np

# Reconfigure sys.stdout and sys.stderr to use UTF-8 encoding to prevent UnicodeEncodeError on Windows
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import ctypes

def get_process_memory_usage() -> int:
    try:
        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]
        psapi = ctypes.windll.psapi
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        
        pmc = PROCESS_MEMORY_COUNTERS_EX()
        pmc.cb = ctypes.sizeof(pmc)
        handle = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
            return pmc.PrivateUsage
    except Exception:
        pass
    return 0

base_dir = os.path.dirname(os.path.abspath(__file__))
EXECUTION_MODE = os.environ.get("EXECUTION_MODE", "LIVE")
ENGINE_RISK_PCT = float(os.environ.get("ENGINE_RISK_PCT", "0.005"))
MT5_LIVE = os.environ.get("MT5_LIVE", "0") == "1"

# Global Setup for unified_backtest import (dynamically switched via ACTIVE_STRATEGY)
ACTIVE_STRATEGY = os.environ.get("ACTIVE_STRATEGY", "alpha_squeezer_v17")
STRATEGY_DISPLAY_NAME = ACTIVE_STRATEGY.replace("_", " ").title().replace(" ", "_")
try:
    sys.modules.pop('unified_backtest', None)
    sys.modules.pop('unified_backtest_compiled', None)
    as_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ACTIVE_STRATEGY)
    if as_path not in sys.path:
        sys.path.insert(0, as_path)
    from unified_backtest import prep
except ImportError:
    as_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alpha_squeezer_v17')
    if as_path not in sys.path:
        sys.path.insert(0, as_path)
    from unified_backtest import prep

# ML_Trend_Pull prep import — isolated namespace to avoid module cache collisions
try:
    import importlib.machinery as _tp_machinery
    import importlib.util as _tp_util
    _tp_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_trend_pull')
    _tp_ub_path = os.path.join(_tp_base, 'unified_backtest.py')
    _tp_loader = _tp_machinery.SourceFileLoader('trend_pull_backtest', _tp_ub_path)
    _tp_spec = _tp_util.spec_from_loader('trend_pull_backtest', _tp_loader)
    _tp_mod = _tp_util.module_from_spec(_tp_spec)
    _tp_spec.loader.exec_module(_tp_mod)
    trend_pull_prep = _tp_mod.custom_prep
except Exception as _tp_err:
    print(f"[Setup] [WARN] Could not load ML_Trend_Pull prep: {_tp_err}")
    trend_pull_prep = None

def _parse_suffix_float(val: Any) -> float | None:
    if val is None:
        return None
    s = str(val).strip().replace(",", "")
    # Replace all unicode minus variants with ASCII minus
    for ch in ("\u2212", "\u2012", "\u2013", "\u2014"):
        s = s.replace(ch, "-")
    if not s or s.lower() == "n/a" or s == "--" or s.lower() == "nan" or s == "∅":
        return None
    try:
        mul = 1.0
        if s.endswith("%"):
            s = s[:-1]
        if s.lower().endswith("k"):
            mul = 1000.0
            s = s[:-1]
        elif s.lower().endswith("m"):
            mul = 1000000.0
            s = s[:-1]
        elif s.lower().endswith("b"):
            mul = 1000000000.0
            s = s[:-1]
        elif s.lower().endswith("t"):
            mul = 1000000000000.0
            s = s[:-1]
        f = float(s)
        return f * mul if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None

def parse_float(val: Any) -> float:
    res = _parse_suffix_float(val)
    return res if res is not None else 0.0

def finite_float_or_none(val: Any) -> float | None:
    return _parse_suffix_float(val)

def get_historical_timestamps(symbol: str, start_time_ts: int, steps: int) -> List[int]:
    is_crypto = symbol not in ["XAUUSDT", "XAGUSDT", "CLUSDT", "NATGASUSDT"]
    if is_crypto:
        return [int(start_time_ts - i * 900) for i in range(steps)]

    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    ny_tz = ZoneInfo("America/New_York")

    def is_active(dt):
        day = dt.weekday()
        hour = dt.hour
        if day == 4:  # Friday
            if hour >= 17: return False
        elif day == 5:  # Saturday
            return False
        elif day == 6:  # Sunday
            if hour < 18: return False
        if day in (0, 1, 2, 3):  # Mon-Thu
            if hour == 17: return False
        return True

    dt = datetime.fromtimestamp(start_time_ts, tz=ny_tz)
    dt = dt.replace(minute=(dt.minute // 15) * 15, second=0, microsecond=0)

    while not is_active(dt):
        dt -= timedelta(minutes=15)

    timestamps = []
    for _ in range(steps):
        timestamps.append(int(dt.timestamp()))
        dt -= timedelta(minutes=15)
        while not is_active(dt):
            dt -= timedelta(minutes=15)

    return timestamps

def calculate_commodity_gap(symbol: str, latest_time: int, current_time: int) -> int:
    is_crypto = symbol not in ["XAUUSDT", "XAGUSDT", "CLUSDT", "NATGASUSDT"]
    if is_crypto:
        return max(0, int((current_time - latest_time) / 900))
    timestamps = get_historical_timestamps(symbol, current_time, 2000)
    for idx, ts in enumerate(timestamps):
        if ts <= latest_time:
            return idx
    return 1000

# --- GLOBAL CONFIGURATION ---
URL = "https://www.coinglass.com/tv/layout/s9"
TAB1_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT", "LINKUSDT"]
TAB2_SYMBOLS = ["AVAXUSDT", "SUIUSDT", "NEARUSDT", "DOTUSDT", "LTCUSDT", "XAUUSDT", "XAGUSDT", "CLUSDT", "NATGASUSDT"]
ALL_SYMBOLS = TAB1_SYMBOLS + TAB2_SYMBOLS
REFRESH_HZ = 2.0  # 2 Hz rendering = 0.5s interval
STALE_NS = 5_000_000_000  # 5 seconds staleness threshold

# --- INDICATOR CONFIGURATION ---
_INDICATORS_TO_INJECT = [
    # (search_term, check_str, modal_type)
    ("Aggregated Futures Cumulative Volume Delta (CVD)", "futures cumulative", "coinglass"),
    ("Aggregated Spot Cumulative Volume Delta (CVD)",    "spot cumulative",    "coinglass"),
    ("Relative Strength Index",                          "rsi",                "tv"),
    ("Funding Rates",                                    "funding rate",       "coinglass"),
    ("Aggregated Liquidations",                          "liquidations",       "coinglass"),
    ("Long/Short Ratio (Accounts)",                      "long/short",         "coinglass"),
    ("Aggregated Open Interest(STABLECOIN-margined,Candles)", "open interest",  "coinglass"),
    ("Aggregated Futures Bid & Ask",                     "bid & ask",          "coinglass"),
    ("Whale Index",                                      "whale index",        "coinglass"),
    ("Taker Buy/Sell Count",                             "taker buy/sell",     "coinglass"),
]

# --- STATE MANAGEMENT ---
_FLOAT_FIELDS = {
    'price', 'volume', 'rsi', 'fut_cvd', 'spot_cvd', 'liq_long', 'liq_short',
    'funding', 'ls_ratio', 'oi', 'fp_delta', 'fp_poc', 'coins_bid', 'coins_ask',
    'dollars_bid', 'dollars_ask', 'whale_idx', 'tk_buy_cnt', 'tk_sell_cnt'
}

@dataclasses.dataclass
class AssetSnapshot:
    symbol: str
    price: float = 0.0
    volume: float = 0.0
    rsi: float = 0.0
    fut_cvd: float = 0.0
    spot_cvd: float = 0.0
    liq_long: float = 0.0
    liq_short: float = 0.0
    funding: float = 0.0
    ls_ratio: float = 0.0
    oi: float = 0.0
    fp_delta: float = 0.0
    fp_poc: float = 0.0
    coins_bid: float = 0.0
    coins_ask: float = 0.0
    dollars_bid: float = 0.0
    dollars_ask: float = 0.0
    whale_idx: float = 0.0
    tk_buy_cnt: float = 0.0
    tk_sell_cnt: float = 0.0
    strategy_armed: str = ""
    ts_ns: int = 0
    seq: int = 0

    def __post_init__(self):
        for f in _FLOAT_FIELDS:
            try:
                setattr(self, f, float(getattr(self, f)))
            except (ValueError, TypeError):
                setattr(self, f, 0.0)

class LiveStrategyPredictor:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.models_long = {}
        self.models_short = {}
        self.models_long_xgb = {}
        self.models_short_xgb = {}
        self.models_long_cb = {}
        self.models_short_cb = {}
        self.configs = {}
        self.candles_history = {} # symbol -> collections.deque
        self.current_candle = {}  # symbol -> dict
        self._cached_signal = {}  # symbol -> armed_str (cached at candle close)
        self._last_predict_bar = {}  # symbol -> open_time of last predicted bar
        self._lock = threading.RLock()
        self.has_lgb = False
        self.last_model_mtime = 0
        
        try:
            import lightgbm as lgb
            self.has_lgb = True
        except ImportError:
            print("[Strategy] Warning: lightgbm is not installed. Predictions will be skipped.")
            
        self.load_models()

    def check_model_updates(self) -> None:
        """Checks if manifest.json was updated by the training process and hot-swaps them."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        manifest_path = os.path.join(base_dir, ACTIVE_STRATEGY, "models", "manifest.json")
        if os.path.exists(manifest_path):
            mtime = os.path.getmtime(manifest_path)
            if mtime > self.last_model_mtime:
                time.sleep(0.1)  # small buffer for disk sync
                print(f"[Strategy] Detected new WFO model manifest (mtime: {mtime}). Initiating Hot-Swap...")
                self.load_models()
                self.last_model_mtime = mtime

    def load_models(self) -> None:
        if not self.has_lgb:
            return
        import lightgbm as lgb
        import xgboost as xgb
        from catboost import CatBoostClassifier
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        configs_dir = os.path.join(base_dir, ACTIVE_STRATEGY, "agent5_configs")
        models_dir = os.path.join(base_dir, ACTIVE_STRATEGY, "models")
        
        for sym in self.symbols:
            cfg_path = os.path.join(configs_dir, f"{sym}.json")
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, 'r') as f:
                        self.configs[sym] = json.load(f)
                except Exception as e:
                    print(f"[Strategy] Error loading config for {sym}: {e}")
            
            cfg = self.configs.get(sym, {})
            if cfg.get('score', -99999) <= 0:
                continue
                
            lgb_long_path = os.path.join(models_dir, f"{sym}_long_lgb.txt")
            lgb_short_path = os.path.join(models_dir, f"{sym}_short_lgb.txt")
            xgb_long_path = os.path.join(models_dir, f"{sym}_long_xgb.json")
            xgb_short_path = os.path.join(models_dir, f"{sym}_short_xgb.json")
            cb_long_path = os.path.join(models_dir, f"{sym}_long_cb.cbm")
            cb_short_path = os.path.join(models_dir, f"{sym}_short_cb.cbm")
            
            try:
                # Load LightGBM
                if os.path.exists(lgb_long_path):
                    self.models_long[sym] = lgb.Booster(model_file=lgb_long_path)
                if os.path.exists(lgb_short_path):
                    self.models_short[sym] = lgb.Booster(model_file=lgb_short_path)
                
                # Load XGBoost
                if os.path.exists(xgb_long_path):
                    xgb_model = xgb.XGBClassifier()
                    xgb_model.load_model(xgb_long_path)
                    self.models_long_xgb[sym] = xgb_model
                if os.path.exists(xgb_short_path):
                    xgb_model = xgb.XGBClassifier()
                    xgb_model.load_model(xgb_short_path)
                    self.models_short_xgb[sym] = xgb_model
                
                # Load CatBoost
                if os.path.exists(cb_long_path):
                    cb_model = CatBoostClassifier()
                    cb_model.load_model(cb_long_path)
                    self.models_long_cb[sym] = cb_model
                if os.path.exists(cb_short_path):
                    cb_model = CatBoostClassifier()
                    cb_model.load_model(cb_short_path)
                    self.models_short_cb[sym] = cb_model
            except Exception as e:
                print(f"[Strategy] Error loading models for {sym}: {e}")

        print(f"[Strategy] Loaded {len(self.models_long)} LONG and {len(self.models_short)} SHORT ensemble models.")

        # Staleness warning: flag models older than 48 hours
        for sym in list(self.models_long) + list(self.models_short):
            side = 'long' if sym in self.models_long else 'short'
            model_path = os.path.join(models_dir, f"{sym}_{side}_lgb.txt")
            if os.path.exists(model_path):
                age_hours = (time.time() - os.path.getmtime(model_path)) / 3600
                if age_hours > 48:
                    print(f"[Strategy] WARNING: {sym}_{side} model is {age_hours:.0f}h old — retrain may have failed.")

        # Update manifest modification time to prevent double-load
        manifest_path = os.path.join(base_dir, ACTIVE_STRATEGY, "models", "manifest.json")
        if os.path.exists(manifest_path):
            self.last_model_mtime = os.path.getmtime(manifest_path)

    def set_history(self, symbol: str, candles: collections.deque | list) -> None:
        now_open = int(time.time() // 900) * 900
        cleaned = []
        for c in candles:
            try:
                ot = int(c.get("open_time", 0))
            except Exception:
                continue
            # Keep only closed bars
            if ot > 0 and ot < now_open:
                row = dict(c)
                row["open_time"] = ot
                cleaned.append(row)
                
        cleaned.sort(key=lambda r: r["open_time"])
        cleaned = cleaned[-1200:]
        
        self.candles_history[symbol] = collections.deque(cleaned, maxlen=1200)
        # Prevent immediate stale/mid-candle prediction after seed/restart
        if cleaned:
            self._last_predict_bar[symbol] = cleaned[-1]["open_time"]

    def load_history_from_disk(self) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        combined_path = os.path.join(base_dir, "Seeding", "combined_seed_history.xlsx")
        if not os.path.exists(combined_path):
            print(f"[Strategy] No combined seeding file found at {combined_path} to pre-load history.")
            return
            
        import openpyxl
        import pandas as pd
        from datetime import datetime, timezone
        try:
            print("[Strategy] Loading historical candles from combined workbook...")
            wb = openpyxl.load_workbook(combined_path, read_only=True)
            for sheetname in wb.sheetnames:
                sym = sheetname
                if sym not in self.symbols:
                    continue
                ws = wb[sym]
                rows = list(ws.iter_rows(values_only=True))
                if len(rows) < 2:
                    continue
                headers = rows[0]
                data_rows = rows[1:][-1200:]
                
                candle_list = []
                for row in data_rows:
                    d = dict(zip(headers, row))
                    val = d.get("open_time")
                    if isinstance(val, datetime):
                        d["open_time"] = int(val.replace(tzinfo=timezone.utc).timestamp())
                    elif isinstance(val, (int, float)):
                        d["open_time"] = int(val)
                    elif isinstance(val, str):
                        try:
                            val_clean = val.replace(" IST", "").strip()
                            dt = pd.to_datetime(val_clean)
                            from datetime import timedelta
                            dt_utc = dt - timedelta(hours=5, minutes=30)
                            d["open_time"] = int(dt_utc.timestamp())
                        except Exception:
                            try:
                                d["open_time"] = int(float(val))
                            except Exception:
                                pass
                    candle_list.append(d)
                self.set_history(sym, candle_list)
            print(f"[Strategy] Loaded history for {len(self.candles_history)} symbols from Excel disk cache.")
        except Exception as e:
            print(f"[Strategy] Error reading combined seed history workbook: {e}")

    def compute_features(self, df: pd.DataFrame, btc_df: pd.DataFrame):
        df = df.sort_values('open_time').reset_index(drop=True)
        btc_df = btc_df.sort_values('open_time').reset_index(drop=True)
        
        # Safely convert open_time to DatetimeIndex
        def safe_to_datetime(s):
            s_clean = s.astype(str).str.replace(' IST', '', regex=False)
            try:
                return pd.to_datetime(pd.to_numeric(s_clean), unit='s')
            except (ValueError, TypeError):
                return pd.to_datetime(s_clean, errors='coerce')

        # Convert cumulative columns to 15m deltas
        cum_cols = [
            'liq_long', 'liq_short', 'tk_buy_cnt', 'tk_sell_cnt',
            'dollars_bid', 'dollars_ask', 'coins_bid', 'coins_ask'
        ]
        
        delta_df = pd.DataFrame()
        for col in cum_cols:
            raw_col = pd.to_numeric(df.get(col, 0.0), errors='coerce').fillna(0.0)
            diff_col = raw_col.diff().fillna(0.0)
            delta_df[col] = np.where(diff_col < 0.0, 0.0, diff_col)

        # Construct summary/footprint dataframe matching Parquet columns
        mapped_df = pd.DataFrame()
        mapped_df['ts'] = safe_to_datetime(df['open_time'])
        mapped_df['Open'] = df['open'].astype(float)
        mapped_df['High'] = df['high'].astype(float)
        mapped_df['Price High'] = df['high'].astype(float)
        mapped_df['Low'] = df['low'].astype(float)
        mapped_df['Price Low'] = df['low'].astype(float)
        mapped_df['Close'] = df['close'].astype(float)
        mapped_df['Volume'] = df['volume'].astype(float)
        mapped_df['RSI'] = df['rsi'].astype(float)
        mapped_df['CVD'] = df['fut_cvd'].astype(float)
        mapped_df['Agg. OI'] = df['oi'].astype(float)
        mapped_df['Long/Short Ratio (Account)'] = df['ls_ratio'].astype(float)
        mapped_df['Agg. Funding Rate'] = df['funding'].astype(float)
        mapped_df['Agg. Liq Long'] = delta_df['liq_long'].astype(float)
        mapped_df['Agg. Liq Short'] = delta_df['liq_short'].astype(float)
        
        coins_bid = delta_df['coins_bid'].astype(float)
        coins_ask = delta_df['coins_ask'].astype(float)
        dollars_bid = delta_df['dollars_bid'].astype(float)
        dollars_ask = delta_df['dollars_ask'].astype(float)
        
        mapped_df['Bid Qty'] = coins_bid
        mapped_df['Ask Qty'] = coins_ask
        mapped_df['total_qty'] = coins_bid + coins_ask
        mapped_df['Delta Qty'] = coins_bid - coins_ask
        mapped_df['Candle Delta'] = coins_bid - coins_ask
        
        mapped_df['Bid USD'] = dollars_bid
        mapped_df['Ask USD'] = dollars_ask
        mapped_df['Delta USD'] = dollars_bid - dollars_ask
        
        def parse_col(col_name):
            raw = df.get(col_name)
            if raw is None:
                return pd.Series(0.0, index=df.index)
            return pd.to_numeric(raw, errors='coerce').fillna(0.0).astype(float)
            
        mapped_df['Whale Ind'] = parse_col('whale_idx')
        mapped_df['Bid Trades'] = delta_df['tk_buy_cnt'].astype(float)
        mapped_df['Ask Trades'] = delta_df['tk_sell_cnt'].astype(float)
        
        # Fall back to Close if fp_poc is 0 or missing (e.g. for non-Binance commodity assets)
        fp_poc = df.get('fp_poc', df['close']).astype(float)
        fp_poc = np.where((fp_poc == 0.0) | pd.isna(fp_poc), df['close'].astype(float), fp_poc)
        mapped_df['POC Price'] = fp_poc
        
        mapped_df = mapped_df.set_index('ts')
        
        # Construct BTC reference
        btc_ref = pd.DataFrame()
        btc_ref['ts'] = safe_to_datetime(btc_df['open_time'])
        btc_ref['btc_Close'] = btc_df['close'].astype(float)
        btc_ref['btc_CVD'] = btc_df['fut_cvd'].astype(float)
        btc_ref = btc_ref.set_index('ts')
        
        df_feat, feats = prep(mapped_df, btc_ref)
        df_feat = df_feat.reset_index()
        
        return df_feat, feats

    def on_tick_update(self, symbol: str, snap: AssetSnapshot, trade_tracker: Any = None) -> AssetSnapshot:
        with self._lock:
            self.check_model_updates()
            return self._on_tick_update_locked(symbol, snap, trade_tracker)

    def _on_tick_update_locked(self, symbol: str, snap: AssetSnapshot, trade_tracker: Any = None) -> AssetSnapshot:
        if not self.has_lgb or snap.price <= 0.0:
            return snap
            
        now = time.time()
        open_time = int(now // 900) * 900
        
        if symbol not in self.candles_history:
            self.candles_history[symbol] = collections.deque(maxlen=1200)
            
        history = self.candles_history[symbol]
        
        if symbol not in self.current_candle or self.current_candle[symbol].get('open_time') != open_time:
            prev = self.current_candle.get(symbol)
            if prev and int(prev.get("open_time", 0)) < open_time:
                prev_ot = int(prev["open_time"])
                if not history or int(history[-1].get("open_time", 0)) != prev_ot:
                    history.append(dict(prev))
            self.current_candle[symbol] = {
                "open_time": open_time,
                "open": snap.price,
                "high": snap.price,
                "low": snap.price,
                "close": snap.price,
                "volume": snap.volume,
                "rsi": snap.rsi,
                "fut_cvd": snap.fut_cvd,
                "spot_cvd": snap.spot_cvd,
                "funding": snap.funding,
                "liq_long": snap.liq_long,
                "liq_short": snap.liq_short,
                "ls_ratio": snap.ls_ratio,
                "oi": snap.oi,
                "coins_bid": snap.coins_bid,
                "coins_ask": snap.coins_ask,
                "dollars_bid": snap.dollars_bid,
                "dollars_ask": snap.dollars_ask,
                "whale_idx": snap.whale_idx,
                "tk_buy_cnt": snap.tk_buy_cnt,
                "tk_sell_cnt": snap.tk_sell_cnt,
                "fp_poc": snap.fp_poc
            }
        else:
            candle = self.current_candle[symbol]
            candle["close"] = snap.price
            if snap.price > candle["high"]:
                candle["high"] = snap.price
            if snap.price < candle["low"] or candle["low"] == 0.0:
                candle["low"] = snap.price
            candle["volume"] = snap.volume
            candle["rsi"] = snap.rsi
            candle["fut_cvd"] = snap.fut_cvd
            candle["spot_cvd"] = snap.spot_cvd
            candle["funding"] = snap.funding
            candle["liq_long"] = snap.liq_long
            candle["liq_short"] = snap.liq_short
            candle["ls_ratio"] = snap.ls_ratio
            candle["oi"] = snap.oi
            candle["coins_bid"] = snap.coins_bid
            candle["coins_ask"] = snap.coins_ask
            candle["dollars_bid"] = snap.dollars_bid
            candle["dollars_ask"] = snap.dollars_ask
            candle["whale_idx"] = snap.whale_idx
            candle["tk_buy_cnt"] = snap.tk_buy_cnt
            candle["tk_sell_cnt"] = snap.tk_sell_cnt
            candle["fp_poc"] = snap.fp_poc
            
        if len(history) < 850:
            return snap

        # Determine the last closed bar's timestamp
        last_bar_time = history[-1].get('open_time', 0) if history else 0
        need_predict = (last_bar_time != self._last_predict_bar.get(symbol, 0))

        if need_predict:
            combined = list(history)
            df = pd.DataFrame(combined)

            btc_hist = self.candles_history.get('BTCUSDT', collections.deque())
            btc_combined = list(btc_hist)

            if len(btc_combined) < 20:
                return snap

            btc_df = pd.DataFrame(btc_combined)

            try:
                df_feat, feats = self.compute_features(df, btc_df)
                last_row = df_feat.iloc[-1]

                p_long = 0.0
                p_short = 0.0
                votes_long = 0
                votes_short = 0
                
                cfg = self.configs.get(symbol, {})
                base_votes = cfg.get("ensemble_min_votes", 2)
                base_conf = cfg.get("confidence", 0.5261)
                vol_limit = cfg.get("vol_limit", 1.0)
                sl_mult = cfg.get("sl_mult", 1.0)
                tp_mult = cfg.get("tp_mult", 3.5)
                
                macro = int(df_feat.iloc[-1].get('macro', 0))
                macro_1h = int(df_feat.iloc[-1].get('macro_1h', 0))
                regime_val = int(df_feat.iloc[-1].get('regime_state', 2))
                vol_regime_gate = float(df_feat.iloc[-1].get('vol_regime_gate', 1.0))
                atr_val = float(df_feat.iloc[-1].get('atr', 0.0))
                atr_pct = atr_val / snap.price if snap.price > 0 else 0.0
                
                # Regime-adaptive confidence threshold — matches agent6_exact.get_regime_confidence()
                CONF_TRENDING = 0.56
                CONF_TRANSITIONAL = 0.62
                CONF_RANGE = 999.0
                if regime_val == 2:
                    conf_threshold = CONF_RANGE       # range-bound: no entries
                elif regime_val == 1:
                    conf_threshold = max(base_conf, CONF_TRANSITIONAL)
                else:
                    conf_threshold = max(base_conf, CONF_TRENDING)
                
                min_votes = 3 if regime_val == 1 else base_votes
                
                if symbol in self.models_long:
                    expected_feats = self.models_long[symbol].feature_name()
                    X_df = pd.DataFrame([last_row[expected_feats]])
                    
                    p_lgb_l = float(self.models_long[symbol].predict(X_df)[0])
                    n_models_l = 1
                    p_xgb_l = 0.0
                    if symbol in self.models_long_xgb:
                        p_xgb_l = float(self.models_long_xgb[symbol].predict_proba(X_df)[0, 1])
                        n_models_l += 1
                    p_cb_l = 0.0
                    if symbol in self.models_long_cb:
                        p_cb_l = float(self.models_long_cb[symbol].predict_proba(X_df)[0, 1])
                        n_models_l += 1
                        
                    p_long = (p_lgb_l + p_xgb_l + p_cb_l) / max(n_models_l, 1)
                    votes_long = int(p_lgb_l > 0.5) + int(p_xgb_l > 0.5) + int(p_cb_l > 0.5)
                    
                if symbol in self.models_short:
                    expected_feats = self.models_short[symbol].feature_name()
                    X_df = pd.DataFrame([last_row[expected_feats]])
                    
                    p_lgb_s = float(self.models_short[symbol].predict(X_df)[0])
                    n_models_s = 1
                    p_xgb_s = 0.0
                    if symbol in self.models_short_xgb:
                        p_xgb_s = float(self.models_short_xgb[symbol].predict_proba(X_df)[0, 1])
                        n_models_s += 1
                    p_cb_s = 0.0
                    if symbol in self.models_short_cb:
                        p_cb_s = float(self.models_short_cb[symbol].predict_proba(X_df)[0, 1])
                        n_models_s += 1
                        
                    p_short = (p_lgb_s + p_xgb_s + p_cb_s) / max(n_models_s, 1)
                    votes_short = int(p_lgb_s > 0.5) + int(p_xgb_s > 0.5) + int(p_cb_s > 0.5)
                    
                armed_str = ""
                if vol_regime_gate == 0:
                    armed_str = ""
                elif p_long > conf_threshold and p_long > p_short and macro == 1 and macro_1h == 1 and votes_long >= min_votes:
                    armed_str = f"LONG ({p_long:.2f})"
                elif p_short > conf_threshold and p_short > p_long and macro == -1 and macro_1h == -1 and votes_short >= min_votes:
                    armed_str = f"SHORT ({p_short:.2f})"
                    
                # Cache the signal and metadata for mid-candle replay
                self._cached_signal[symbol] = {
                    'armed_str': armed_str,
                    'atr_val': atr_val,
                    'macro': macro,
                    'sl_mult': sl_mult,
                    'tp_mult': tp_mult,
                    'vol_regime': float(df_feat.iloc[-1].get('vol_regime', 0.0)),
                    'last_closed_time': last_bar_time
                }
                self._last_predict_bar[symbol] = last_bar_time
                
                # Entry logic fires only at candle close
                has_active = False
                if trade_tracker:
                    with trade_tracker.lock:
                        has_active = any(t['symbol'] == symbol and t['strategy'] == STRATEGY_DISPLAY_NAME for t in trade_tracker.active_trades.values())
                
                if armed_str and trade_tracker and not has_active and atr_val > 0:
                    if last_bar_time > trade_tracker.last_entry_bar.get(symbol, 0):
                        # FIX 3: Respect re-entry cooldown set on TP/SL close
                        cooldown_until = trade_tracker.reentry_cooldown_until.get(f"{STRATEGY_DISPLAY_NAME}:{symbol}", 0)
                        if time.time() < cooldown_until:
                            pass  # blocked — wait out the cooldown
                        else:
                            direction = 1 if armed_str.startswith("LONG") else -1
                            # FIX 4: Enforce 0.3% minimum SL distance — widen proportionally
                            # rather than dropping the trade so R:R ratio is preserved.
                            min_sl_pct = 0.003
                            raw_sl_dist = sl_mult * atr_val
                            effective_sl_dist = max(raw_sl_dist, snap.price * min_sl_pct)
                            rr_ratio = tp_mult / sl_mult if sl_mult > 0 else tp_mult
                            effective_tp_dist = effective_sl_dist * rr_ratio
                            sl = snap.price - effective_sl_dist if direction == 1 else snap.price + effective_sl_dist
                            tp = snap.price + effective_tp_dist if direction == 1 else snap.price - effective_tp_dist
                            risk_mult = 0.75 if regime_val == 1 else 1.0
                            trail_act = min(cfg.get("trail_act", 0.5), 1.0)
                            trade_tracker.trigger_entry(
                                symbol, STRATEGY_DISPLAY_NAME, direction, snap.price, sl, tp, atr_val, macro,
                                float(df_feat.iloc[-1].get('vol_regime', 0.0)),
                                risk_mult=risk_mult, trail_act=trail_act, regime_val=regime_val
                            )
                            trade_tracker.last_entry_bar[symbol] = last_bar_time

            except Exception as e:
                import traceback
                print(f"[Strategy] {symbol} prediction error: {e}\n{traceback.format_exc()}")

        # Replay cached signal on every tick for display
        cached = self._cached_signal.get(symbol, {})
        armed_str = cached.get('armed_str', '')

        if trade_tracker:
            with trade_tracker.lock:
                trades = [t for t in trade_tracker.active_trades.values() if t['symbol'] == symbol and t['strategy'] == STRATEGY_DISPLAY_NAME]
            if trades:
                trade = trades[0]
                dir_str = "LONG" if trade['direction'] == 1 else "SHORT"
                pnl = trade.get('live_pnl_pct', 0.0)
                armed_str = f"HOLD {dir_str} ({pnl:+.2f}%)"

        snap = dataclasses.replace(snap, strategy_armed=armed_str)
        return snap

class LiveLiquidationPredictor:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.models = {}
        self.candles_history = {}
        self.current_candle = {}
        self._lock = threading.RLock()
        # Equity curve tracking — rolling window of recent capital levels (OOS Proposal 3)
        self.recent_capitals = []
        self._compute_rolling_features = None  # lazy-loaded once
        self.latest_atr = {}
        self.last_model_mtime = 0

        self.load_models()
        
    def check_model_updates(self):
        """Checks if manifest.json was updated by the training process and hot-swaps them."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        manifest_path = os.path.join(base_dir, "Liquidation", "models", "manifest.json")
        if os.path.exists(manifest_path):
            mtime = os.path.getmtime(manifest_path)
            if mtime > self.last_model_mtime:
                time.sleep(0.1)  # small buffer for disk sync
                print(f"[Liquidation] Detected new WFO model manifest (mtime: {mtime}). Initiating Hot-Swap...")
                self.load_models()
                self.last_model_mtime = mtime

    def load_models(self):
        import pickle
        import sys
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "Liquidation", "models")
        liq_dir = os.path.join(base_dir, "Liquidation")
        if liq_dir not in sys.path:
            sys.path.insert(0, liq_dir)

        try:
            from features import compute_rolling_features  # noqa: F401 — validate importable
            self._compute_rolling_features = compute_rolling_features
            with open(os.path.join(models_dir, "lgb_model.pkl"), "rb") as f:
                self.models['lgb'] = pickle.load(f)
            with open(os.path.join(models_dir, "xgb_model.pkl"), "rb") as f:
                self.models['xgb'] = pickle.load(f)
            with open(os.path.join(models_dir, "cb_model.pkl"), "rb") as f:
                self.models['cb'] = pickle.load(f)
            
            # Assert class order to prevent silent class/label inversion
            for name, model in self.models.items():
                classes = list(model.classes_)
                assert classes == [0, 1, 2], f"Model {name} classes mismatch: {classes} vs [0, 1, 2]"
                
            print("[Liquidation] Successfully loaded all ML Liquidation models and verified class mappings.")
        except Exception as e:
            print(f"[Liquidation] Error loading ML Liquidation models: {e}")

        # Update manifest modification time to prevent double-load
        manifest_path = os.path.join(base_dir, "Liquidation", "models", "manifest.json")
        if os.path.exists(manifest_path):
            self.last_model_mtime = os.path.getmtime(manifest_path)

    def on_tick_update(self, symbol: str, snap: AssetSnapshot, trade_tracker: Any = None) -> AssetSnapshot:
        with self._lock:
            if snap.price <= 0.0: return snap
            
            # Periodically check for hot-swaps
            self.check_model_updates()
            
            now = time.time()
            open_time = int(now // 900) * 900
            
            if symbol not in self.candles_history:
                self.candles_history[symbol] = collections.deque(maxlen=1200)
                
            history = self.candles_history[symbol]
            
            if symbol not in self.current_candle or self.current_candle[symbol].get('open_time') != open_time:
                prev = self.current_candle.get(symbol)
                if prev and int(prev.get("open_time", 0)) < open_time:
                    prev_ot = int(prev["open_time"])
                    if not history or int(history[-1].get("open_time", 0)) != prev_ot:
                        history.append(dict(prev))
                
                self.current_candle[symbol] = {
                    "open_time": open_time, "open": snap.price, "high": snap.price,
                    "low": snap.price, "close": snap.price, "volume": snap.volume,
                    "fut_cvd": snap.fut_cvd, "liq_long": snap.liq_long, "liq_short": snap.liq_short,
                    "coins_bid": snap.coins_bid, "coins_ask": snap.coins_ask,
                    "dollars_bid": snap.dollars_bid, "dollars_ask": snap.dollars_ask,
                    "tk_buy_cnt": snap.tk_buy_cnt, "tk_sell_cnt": snap.tk_sell_cnt,
                    "fp_poc": snap.fp_poc
                }
                
                # Inference only on candle close
                if len(history) > 250:
                    self._run_inference(symbol, snap.price, trade_tracker)
            else:
                candle = self.current_candle[symbol]
                candle["close"] = snap.price
                if snap.price > candle["high"]: candle["high"] = snap.price
                if snap.price < candle["low"] or candle["low"] == 0.0: candle["low"] = snap.price
                candle["volume"] = snap.volume
                candle["fut_cvd"] = snap.fut_cvd
                candle["liq_long"] = snap.liq_long
                candle["liq_short"] = snap.liq_short
                candle["coins_bid"] = snap.coins_bid
                candle["coins_ask"] = snap.coins_ask
                candle["dollars_bid"] = snap.dollars_bid
                candle["dollars_ask"] = snap.dollars_ask
                candle["tk_buy_cnt"] = snap.tk_buy_cnt
                candle["tk_sell_cnt"] = snap.tk_sell_cnt
                candle["fp_poc"] = snap.fp_poc
                
            return snap
            
    # Minimum history rows before ML Liquidation may open any trade.
    # Prevents stale/placeholder prices during the startup warm-up window.
    MIN_WARMUP_BARS = 50

    def _run_inference(self, symbol, current_price, trade_tracker):
        try:
            import pandas as pd
            import polars as pl
            import numpy as np

            compute_rolling_features = self._compute_rolling_features
            if compute_rolling_features is None:
                return  # models failed to load at init, abort silently

            # FIX 1: Gate on warm-up — refuse to trade until we have real prices
            if current_price <= 0:
                return
            history = list(self.candles_history[symbol])
            if len(history) < self.MIN_WARMUP_BARS:
                return
            df = pd.DataFrame(history)

            # Convert cumulative columns to 15m deltas
            cum_cols = [
                'liq_long', 'liq_short', 'tk_buy_cnt', 'tk_sell_cnt',
                'dollars_bid', 'dollars_ask', 'coins_bid', 'coins_ask'
            ]
            
            delta_df = pd.DataFrame()
            for col in cum_cols:
                raw_col = pd.to_numeric(df.get(col, 0.0), errors='coerce').fillna(0.0)
                diff_col = raw_col.diff().fillna(0.0)
                delta_df[col] = np.where(diff_col < 0.0, 0.0, diff_col)

            # Build mapped DataFrame matching training column names
            mapped_df = pd.DataFrame()
            mapped_df['datetime'] = pd.to_datetime(df['open_time'], unit='s')
            mapped_df['Open'] = df['open'].astype(float)
            mapped_df['High'] = df['high'].astype(float)
            mapped_df['Low'] = df['low'].astype(float)
            mapped_df['Close'] = df['close'].astype(float)
            mapped_df['Volume'] = df['volume'].astype(float)
            mapped_df['CVD'] = df['fut_cvd'].astype(float)
            mapped_df['Agg. Liq Long'] = delta_df['liq_long'].astype(float)
            mapped_df['Agg. Liq Short'] = delta_df['liq_short'].astype(float)
            mapped_df['Bid Trades'] = delta_df['tk_buy_cnt'].astype(float)
            mapped_df['Ask Trades'] = delta_df['tk_sell_cnt'].astype(float)
            mapped_df['Bid USD'] = delta_df['dollars_bid'].astype(float)
            mapped_df['Ask USD'] = delta_df['dollars_ask'].astype(float)
            mapped_df['Candle Delta'] = delta_df['coins_bid'].astype(float) - delta_df['coins_ask'].astype(float)
            mapped_df['Delta USD'] = delta_df['dollars_bid'].astype(float) - delta_df['dollars_ask'].astype(float)

            fp_poc = df.get('fp_poc', df['close']).astype(float)
            mapped_df['POC Price'] = np.where((fp_poc == 0.0) | pd.isna(fp_poc), df['close'].astype(float), fp_poc)

            pldf = pl.from_pandas(mapped_df)
            pldf = compute_rolling_features(pldf)

            # --- 4H Trend Filter: EMA crossover on resampled 4H bars ---
            df_4h = pldf.sort("datetime").group_by_dynamic(
                "datetime",
                every="4h",
                closed="left"
            ).agg([
                pl.col("Close").last().alias("Close")
            ]).with_columns([
                pl.col("Close").ewm_mean(span=9).alias("ema_9"),
                pl.col("Close").ewm_mean(span=21).alias("ema_21")
            ]).with_columns([
                pl.when(pl.col("ema_9") > pl.col("ema_21")).then(pl.lit(1))
                  .when(pl.col("ema_9") < pl.col("ema_21")).then(pl.lit(-1))
                  .otherwise(pl.lit(0)).alias("trend_4h")
            ])
            # Shift by 4 hours to align with completed bars only, preventing lookahead leakage
            df_4h = df_4h.with_columns((pl.col("datetime") + pl.duration(hours=4)).cast(pldf.schema["datetime"]))
            
            pldf = pldf.sort("datetime").join_asof(
                df_4h.select(["datetime", "trend_4h"]),
                on="datetime",
                strategy="backward"
            )
            pldf = pldf.with_columns(pl.col("trend_4h").fill_null(0))

            last_row = pldf[-1]
            trend_4h = last_row["trend_4h"].item()

            # Update latest ATR for trailing stops
            atr_val = last_row["atr"].item()
            if not np.isnan(atr_val) and atr_val > 0:
                self.latest_atr[symbol] = atr_val

            FEATURE_COLS = [
                "trigger_type", "poc_pos", "delta_usd_ratio", "size_ratio",
                "trade_ratio", "liq_long_z_50", "liq_short_z_50",
                "liq_long_z_200", "liq_short_z_200", "cvd_z_10",
                "cvd_z_50", "cvd_z_200", "atr_ratio", "close_to_ema_200"
            ]

            # --- FIX 3: Use z_200 >= 3.0 — matches generate_labels() training threshold ---
            liq_long_z_200 = last_row["liq_long_z_200"].item()
            liq_short_z_200 = last_row["liq_short_z_200"].item()
            trigger_type = 0
            if liq_long_z_200 >= 3.0 and liq_short_z_200 >= 3.0:
                # Dual-event: take the larger spike, matching generate_labels() tie-break
                trigger_type = 1 if liq_long_z_200 >= liq_short_z_200 else -1
            elif liq_long_z_200 >= 3.0:
                trigger_type = 1
            elif liq_short_z_200 >= 3.0:
                trigger_type = -1

            if trigger_type == 0:
                return

            X = last_row.with_columns(pl.lit(trigger_type).cast(pl.Int8).alias("trigger_type")).select(FEATURE_COLS).to_pandas()
            if 'lgb' not in self.models:
                return

            lgb_probs = self.models['lgb'].predict_proba(X)
            xgb_probs = self.models['xgb'].predict_proba(X)
            cb_probs = self.models['cb'].predict_proba(X)

            ens_probs = (lgb_probs + xgb_probs + cb_probs) / 3.0
            # Class mapping in train.py:
            # For long liq (+1): Class 0 = Breakout (-1), Class 2 = Reversal (+1)
            # For short liq (-1): Class 0 = Reversal (-1), Class 2 = Breakout (+1)
            if trigger_type == 1:
                prob_breakout = float(ens_probs[0, 0])
                prob_reversal = float(ens_probs[0, 2])
            else:
                prob_breakout = float(ens_probs[0, 2])
                prob_reversal = float(ens_probs[0, 0])
                
            prob_hold = float(ens_probs[0, 1])
            
            # Confidence gates — prefer OOS-optimized values when available
            _lp = getattr(self, "_opt_params", None)
            if _lp is None:
                try:
                    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Liquidation", "optimized_params.json")
                    if os.path.exists(_p):
                        with open(_p, "r") as _f:
                            self._opt_params = json.load(_f)
                        _lp = self._opt_params
                except Exception:
                    _lp = {}
                    self._opt_params = _lp
            MIN_REVERSAL_CONF = float((_lp or {}).get("min_reversal_conf", 0.58))
            MIN_BREAKOUT_CONF = float((_lp or {}).get("min_breakout_conf", 0.80))
            MIN_EDGE_VS_HOLD = float((_lp or {}).get("min_edge_vs_hold", 0.06))
            MIN_EDGE_VS_OTHER_DIRECTION = 0.04
            
            pattern = None
            confidence = 0.0
            
            if (prob_reversal >= MIN_REVERSAL_CONF 
                and prob_reversal - prob_hold >= MIN_EDGE_VS_HOLD 
                and prob_reversal - prob_breakout >= MIN_EDGE_VS_OTHER_DIRECTION):
                pattern = "reversal"
                confidence = prob_reversal
            elif (prob_breakout >= MIN_BREAKOUT_CONF 
                  and prob_breakout - prob_hold >= MIN_EDGE_VS_HOLD 
                  and prob_breakout - prob_reversal >= MIN_EDGE_VS_OTHER_DIRECTION):
                pattern = "breakout"
                confidence = prob_breakout
                
            if not pattern:
                return

            if trigger_type == 1:
                direction = 1 if pattern == "reversal" else -1
            else:
                direction = -1 if pattern == "reversal" else 1
                
            max_liq_z = max(float(liq_long_z_200), float(liq_short_z_200))
            opposing_4h_trend = (direction == 1 and trend_4h == -1) or (direction == -1 and trend_4h == 1)
            
            if pattern == "breakout" and opposing_4h_trend:
                return
                
            if pattern == "reversal" and opposing_4h_trend:
                if confidence < 0.58 and max_liq_z < 4.0:
                    return

            atr_val = self.latest_atr.get(symbol, 0.0)
            if atr_val <= 0:
                return

            # --- FIX 2: Equity curve throttle matching OOS Proposal 3 ---
            # OOS: skip entirely at >2.5% equity deviation, half-size at >1.5%
            EQUITY_MA_WINDOW = 5
            equity_ma = sum(self.recent_capitals[-EQUITY_MA_WINDOW:]) / min(len(self.recent_capitals), EQUITY_MA_WINDOW)
            current_capital = trade_tracker.current_capital if trade_tracker else equity_ma
            equity_deviation = (equity_ma - current_capital) / equity_ma * 100.0 if equity_ma > 0 else 0.0

            if equity_deviation > 2.5:
                return  # Hard skip — protect against funded account daily DD breach

            risk_mult = 0.5 if equity_deviation > 1.5 else 1.0

            # OOS-optimized SL/TP/trail — load once from Liquidation/optimized_params.json
            liq_params = getattr(self, "_opt_params", None)
            if liq_params is None:
                liq_params = {
                    "sl_mult": 0.9,
                    "tp_mult": 8.55,  # default >5R profile
                    "min_reversal_conf": 0.58,
                    "min_breakout_conf": 0.80,
                    "trail_act_reversal": 4.0,
                    "trail_act_breakout": 5.0,
                }
                try:
                    _lp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Liquidation", "optimized_params.json")
                    if os.path.exists(_lp):
                        with open(_lp, "r") as _f:
                            liq_params.update(json.load(_f))
                        self._opt_params = liq_params
                except Exception:
                    self._opt_params = liq_params
            sl_mult = float(liq_params.get("sl_mult", 0.9))
            tp_mult = float(liq_params.get("tp_mult", 8.55))
            # Enforce minimum 5R target even if stale config present
            if sl_mult > 0 and (tp_mult / sl_mult) < 5.0:
                tp_mult = sl_mult * 5.5
            trail_act = float(
                liq_params.get("trail_act_reversal", 4.0) if pattern == "reversal"
                else liq_params.get("trail_act_breakout", 5.0)
            )
            sl = current_price - sl_mult * atr_val if direction == 1 else current_price + sl_mult * atr_val
            tp = current_price + tp_mult * atr_val if direction == 1 else current_price - tp_mult * atr_val

            if trade_tracker:
                trade_tracker.trigger_entry(
                    symbol, "ML_Liquidation_Runner", direction, current_price, sl, tp, atr_val, macro=0,
                    vol_regime=0, risk_mult=risk_mult, trail_act=trail_act, regime_val=0
                )
        except Exception as e:
            import traceback
            print(f"[Liquidation] {symbol} prediction error: {e}\n{traceback.format_exc()}")

    def record_closed_capital(self, capital: float) -> None:
        """Called externally when a Liquidation trade closes to update equity curve tracker."""
        self.recent_capitals.append(capital)
        if len(self.recent_capitals) > 50:
            self.recent_capitals = self.recent_capitals[-50:]


class LiveTrendPullPredictor:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.models_long = {}
        self.models_short = {}
        self.models_long_xgb = {}
        self.models_short_xgb = {}
        self.models_long_cb = {}
        self.models_short_cb = {}
        self.configs = {}
        self.candles_history = {}
        self.current_candle = {}
        self._cached_signal = {}
        self._last_predict_bar = {}
        self._lock = threading.RLock()
        self.has_lgb = False
        self.latest_atr = {}
        self.last_model_mtime = 0

        try:
            import lightgbm as lgb
            self.has_lgb = True
        except ImportError:
            print("[ML_Trend_Pull] Warning: lightgbm not installed. Predictions skipped.")

        self.load_models()

    def check_model_updates(self) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        manifest_path = os.path.join(base_dir, "ml_trend_pull", "models", "manifest.json")
        if os.path.exists(manifest_path):
            mtime = os.path.getmtime(manifest_path)
            if mtime > self.last_model_mtime:
                time.sleep(0.1)
                print(f"[ML_Trend_Pull] Detected new model manifest (mtime: {mtime}). Hot-Swap...")
                self.load_models()
                self.last_model_mtime = mtime

    def load_models(self) -> None:
        if not self.has_lgb:
            return
        import lightgbm as lgb
        import xgboost as xgb
        from catboost import CatBoostClassifier

        base_dir = os.path.dirname(os.path.abspath(__file__))
        configs_dir = os.path.join(base_dir, "ml_trend_pull", "agent5_configs")
        models_dir = os.path.join(base_dir, "ml_trend_pull", "models")

        for sym in self.symbols:
            cfg_path = os.path.join(configs_dir, f"{sym}.json")
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, 'r') as f:
                        self.configs[sym] = json.load(f)
                except Exception as e:
                    print(f"[ML_Trend_Pull] Error loading config for {sym}: {e}")

            cfg = self.configs.get(sym, {})
            if cfg.get('score', -99999) <= 0:
                continue

            lgb_long_path = os.path.join(models_dir, f"{sym}_long_lgb.txt")
            lgb_short_path = os.path.join(models_dir, f"{sym}_short_lgb.txt")
            xgb_long_path = os.path.join(models_dir, f"{sym}_long_xgb.json")
            xgb_short_path = os.path.join(models_dir, f"{sym}_short_xgb.json")
            cb_long_path = os.path.join(models_dir, f"{sym}_long_cb.cbm")
            cb_short_path = os.path.join(models_dir, f"{sym}_short_cb.cbm")

            try:
                if os.path.exists(lgb_long_path):
                    self.models_long[sym] = lgb.Booster(model_file=lgb_long_path)
                if os.path.exists(lgb_short_path):
                    self.models_short[sym] = lgb.Booster(model_file=lgb_short_path)
                if os.path.exists(xgb_long_path):
                    m = xgb.XGBClassifier(); m.load_model(xgb_long_path)
                    self.models_long_xgb[sym] = m
                if os.path.exists(xgb_short_path):
                    m = xgb.XGBClassifier(); m.load_model(xgb_short_path)
                    self.models_short_xgb[sym] = m
                if os.path.exists(cb_long_path):
                    m = CatBoostClassifier(); m.load_model(cb_long_path)
                    self.models_long_cb[sym] = m
                if os.path.exists(cb_short_path):
                    m = CatBoostClassifier(); m.load_model(cb_short_path)
                    self.models_short_cb[sym] = m
            except Exception as e:
                print(f"[ML_Trend_Pull] Error loading models for {sym}: {e}")

        print(f"[ML_Trend_Pull] Loaded {len(self.models_long)} LONG and {len(self.models_short)} SHORT ensemble models.")

        for sym in list(self.models_long) + list(self.models_short):
            side = 'long' if sym in self.models_long else 'short'
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_trend_pull", "models", f"{sym}_{side}_lgb.txt")
            if os.path.exists(model_path):
                age_hours = (time.time() - os.path.getmtime(model_path)) / 3600
                if age_hours > 48:
                    print(f"[ML_Trend_Pull] WARNING: {sym}_{side} model is {age_hours:.0f}h old.")

        # Update manifest modification time to prevent double-load
        manifest_path = os.path.join(base_dir, "ml_trend_pull", "models", "manifest.json")
        if os.path.exists(manifest_path):
            self.last_model_mtime = os.path.getmtime(manifest_path)

    def on_tick_update(self, symbol: str, snap: AssetSnapshot, trade_tracker: Any = None) -> AssetSnapshot:
        with self._lock:
            self.check_model_updates()
            return self._on_tick_update_locked(symbol, snap, trade_tracker)

    def _on_tick_update_locked(self, symbol: str, snap: AssetSnapshot, trade_tracker: Any = None) -> AssetSnapshot:
        if not self.has_lgb or snap.price <= 0.0:
            return snap
        if trend_pull_prep is None:
            return snap

        now = time.time()
        open_time = int(now // 900) * 900

        if symbol not in self.candles_history:
            self.candles_history[symbol] = collections.deque(maxlen=1200)

        history = self.candles_history[symbol]

        if symbol not in self.current_candle or self.current_candle[symbol].get('open_time') != open_time:
            prev = self.current_candle.get(symbol)
            if prev and int(prev.get("open_time", 0)) < open_time:
                prev_ot = int(prev["open_time"])
                if not history or int(history[-1].get("open_time", 0)) != prev_ot:
                    history.append(dict(prev))

            self.current_candle[symbol] = {
                "open_time": open_time, "open": snap.price, "high": snap.price,
                "low": snap.price, "close": snap.price, "volume": snap.volume,
                "fut_cvd": snap.fut_cvd, "oi": snap.oi, "ls_ratio": snap.ls_ratio,
                "funding": snap.funding
            }

            if len(history) > 250:
                self._run_inference(symbol, snap.price, trade_tracker)
        else:
            candle = self.current_candle[symbol]
            candle["close"] = snap.price
            if snap.price > candle["high"]: candle["high"] = snap.price
            if snap.price < candle["low"] or candle["low"] == 0.0: candle["low"] = snap.price
            candle["volume"] = snap.volume
            candle["fut_cvd"] = snap.fut_cvd
            candle["oi"] = snap.oi
            candle["ls_ratio"] = snap.ls_ratio
            candle["funding"] = snap.funding

        cached = self._cached_signal.get(symbol, {})
        armed_str = cached.get('armed_str', '')
        if trade_tracker:
            with trade_tracker.lock:
                trades = [t for t in trade_tracker.active_trades.values() if t['symbol'] == symbol and t['strategy'] == "ML_Trend_Pull"]
            if trades:
                trade = trades[0]
                dir_str = "LONG" if trade['direction'] == 1 else "SHORT"
                pnl = trade.get('live_pnl_pct', 0.0)
                armed_str = f"TP_{dir_str} ({pnl:+.2f}%)"

        return snap

    def _run_inference(self, symbol, current_price, trade_tracker):
        try:
            import pandas as pd
            import numpy as np

            history = list(self.candles_history[symbol])
            if len(history) < 250:
                return

            df = pd.DataFrame(history)
            mapped_df = pd.DataFrame()
            mapped_df['datetime'] = pd.to_datetime(df['open_time'], unit='s')
            mapped_df['Open'] = df['open'].astype(float)
            mapped_df['High'] = df['high'].astype(float)
            mapped_df['Low'] = df['low'].astype(float)
            mapped_df['Close'] = df['close'].astype(float)
            mapped_df['Volume'] = df.get('volume', 0).astype(float)
            mapped_df['CVD'] = df.get('fut_cvd', 0).astype(float)
            mapped_df['Agg. OI'] = df.get('oi', 0.0).astype(float)
            mapped_df['Long/Short Ratio (Account)'] = df.get('ls_ratio', 0.0).astype(float)
            mapped_df['Agg. Funding Rate'] = df.get('funding', 0.0).astype(float)
            mapped_df = mapped_df.set_index('datetime')

            btc_hist = self.candles_history.get('BTCUSDT')
            btc_ref = None
            if btc_hist and len(btc_hist) > 50 and symbol != 'BTCUSDT':
                btc_df = pd.DataFrame(list(btc_hist))
                btc_ref = pd.DataFrame()
                btc_ref.index = pd.to_datetime(btc_df['open_time'], unit='s')
                btc_ref['btc_Close'] = btc_df['close'].astype(float)
                btc_ref['btc_CVD'] = btc_df.get('fut_cvd', 0).astype(float)

            df_feat, feats = trend_pull_prep(mapped_df, btc_ref)
            df_feat = df_feat.reset_index()

            if len(df_feat) < 10:
                return

            last_row = df_feat.iloc[-1]
            atr_val = last_row.get('atr', 0.0)
            if np.isnan(atr_val) or atr_val <= 0:
                return
            self.latest_atr[symbol] = atr_val

            cfg = self.configs.get(symbol, {})
            score = cfg.get('score', -99999)
            if score <= 0:
                return

            model_type = cfg.get('model', 'lightgbm')
            confidence = cfg.get('confidence', 0.53)
            sl_mult = cfg.get('sl_mult', 1.0)
            tp_mult = cfg.get('tp_mult', 5.0)
            trail_act = cfg.get('trail_act', 0.5)

            X_df = pd.DataFrame([last_row[feats]])

            for direction in [1, -1]:
                dir_label = 'long' if direction == 1 else 'short'
                models_dict = self.models_long if direction == 1 else self.models_short
                models_xgb = self.models_long_xgb if direction == 1 else self.models_short_xgb
                models_cb = self.models_long_cb if direction == 1 else self.models_short_cb

                lgb_model = models_dict.get(symbol)
                if lgb_model is None:
                    continue

                lgb_prob = lgb_model.predict(X_df)[0]
                xgb_model = models_xgb.get(symbol)
                cb_model = models_cb.get(symbol)

                probs = [lgb_prob]
                if xgb_model:
                    xgb_prob = xgb_model.predict_proba(X_df)[0, 1]
                    probs.append(xgb_prob)
                if cb_model:
                    cb_prob = cb_model.predict_proba(X_df)[0, 1]
                    probs.append(cb_prob)

                avg_prob = sum(probs) / len(probs)
                if avg_prob < confidence:
                    continue

                macro = last_row.get('macro', 0)
                if direction == 1 and macro < 0:
                    continue
                if direction == -1 and macro > 0:
                    continue

                if trade_tracker:
                    with trade_tracker.lock:
                        has_active = any(
                            t['symbol'] == symbol and t['strategy'] == "ML_Trend_Pull"
                            for t in trade_tracker.active_trades.values()
                        )
                    if has_active:
                        continue

                    cooldown_key = f"ML_Trend_Pull:{symbol}"
                    cooldown_until = trade_tracker.reentry_cooldown_until.get(cooldown_key, 0)
                    if time.time() < cooldown_until:
                        continue

                    min_sl_pct = 0.003
                    raw_sl_dist = sl_mult * atr_val
                    effective_sl_dist = max(raw_sl_dist, current_price * min_sl_pct)
                    rr_ratio = tp_mult / sl_mult if sl_mult > 0 else tp_mult
                    effective_tp_dist = effective_sl_dist * rr_ratio
                    sl = current_price - effective_sl_dist if direction == 1 else current_price + effective_sl_dist
                    tp = current_price + effective_tp_dist if direction == 1 else current_price - effective_tp_dist

                    trade_tracker.trigger_entry(
                        symbol, "ML_Trend_Pull", direction, current_price, sl, tp, atr_val, macro,
                        float(last_row.get('vol_regime', 0.0)),
                        risk_mult=1.0, trail_act=trail_act, regime_val=0
                    )
                    self._cached_signal[symbol] = {'armed_str': f"{'LONG' if direction == 1 else 'SHORT'} ({avg_prob:.2f})"}
                    break

        except Exception as e:
            import traceback
            print(f"[ML_Trend_Pull] {symbol} prediction error: {e}\n{traceback.format_exc()}")


class Engine1TradeTracker:
    # FIX 3: Re-entry cooldown constants.
    # After a TP exit, block same-symbol re-entry for this many seconds (4 × 15m bars).
    # After an SL exit, block for 2 × 15m bars.
    REENTRY_COOLDOWN_TP_SECS = 3600   # 1 hour
    REENTRY_COOLDOWN_SL_SECS = 1800   # 30 minutes

    def _cooldown_key(self, strategy: str, symbol: str) -> str:
        return f"{strategy}:{symbol}"

    def _cooldown_secs_after_close(self, strategy: str, reason: str) -> int:
        if strategy == "ML_Liquidation_Runner":
            if reason == "TP": return 0
            if reason in ("SL", "BE", "TRAIL"): return 15 * 60
        if strategy in (STRATEGY_DISPLAY_NAME, "AlphaSqueezer_V17", "AlphaSqueezer_V11"):
            if reason == "TP": return self.REENTRY_COOLDOWN_TP_SECS
            return self.REENTRY_COOLDOWN_SL_SECS
        if strategy == "ML_Trend_Pull":
            if reason == "TP": return self.REENTRY_COOLDOWN_TP_SECS
            return self.REENTRY_COOLDOWN_SL_SECS
        return 15 * 60


    def __init__(self, initial_capital=4907.37):
        self.active_trades = {}
        self.last_entry_bar = {}
        self.reentry_cooldown_until = {}  # FIX 3: {symbol: unix_timestamp}
        self.history = []
        self.on_close_callbacks = []  # callables(strategy, capital) notified on trade close
        # Always anchored to Engine_1 folder — immune to cwd differences
        self.log_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "Engine_1_trade_logs.json"
        )
        self.lock = threading.RLock()
        
        # --- MT5 Broker Initialization ---
        from concurrent.futures import ThreadPoolExecutor
        self.broker_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="MT5Broker")

        from mt5_broker import MT5Broker
        import MetaTrader5 as mt5
        self.mt5_broker = MT5Broker(
            dry_run=not MT5_LIVE, 
            account_size=initial_capital, 
            risk_pct=ENGINE_RISK_PCT
        )
        if self.mt5_broker.connect():
            info = mt5.account_info()
            if info:
                initial_capital = info.balance
        self.mt5_broker.account_size = initial_capital
        # ---------------------------------
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        import zoneinfo
        from datetime import datetime
        broker_tz = zoneinfo.ZoneInfo("Europe/Athens")
        self.last_rollover_day = datetime.now(broker_tz).strftime("%Y-%m-%d")
        self.daily_start_capital = self.current_capital
        self.emergency_halt = False
        self.load_history()


    def _translate_to_mt5_price(self, trade: dict, engine_price: float) -> float:
        if not trade.get("mt5_entry") or not trade.get("entry_price"): return engine_price
        return float(trade["mt5_entry"]) * (float(engine_price) / float(trade["entry_price"]))

    def _broker_submit(self, fn, *args, **kwargs) -> None:
        if not hasattr(self, "broker_executor") or self.broker_executor is None:
            print(f"[MT5][FATAL] broker_executor missing — cannot dispatch {fn.__name__}")
            return
        try:
            fut = self.broker_executor.submit(fn, *args, **kwargs)
            def _log_result(f):
                try:
                    res = f.result()
                    if res is False:
                        print(f"[MT5][WARN] {fn.__name__} returned False for args: {args}")
                except Exception as e:
                    print(f"[MT5][ERROR] {fn.__name__} raised: {e}")
            fut.add_done_callback(_log_result)
        except Exception as exc:
            print(f"[MT5] Failed to submit broker action: {exc}")

    def load_history(self):
        """Load only from this engine's own log — never from Engine 3 whose
        pnl_usd uses an incompatible unit-based sizing model."""
        with self.lock:
            if not os.path.exists(self.log_file):
                return
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Support envelope format with __meta__
                meta = {}
                if isinstance(data, dict) and '__meta__' in data:
                    meta = data['__meta__']
                    data = data.get('trades', [])
                if not isinstance(data, list):
                    return
                self.last_entry_bar = meta.get('last_entry_bar', {})
                # FIX 2: Strip test/dry-run artifacts injected by boot scripts.
                # These have synthetic trade_ids and corrupt the P&L baseline.
                raw_history = [t for t in data if t.get('exit_price')]
                self.history = [
                    t for t in raw_history
                    if 'test' not in t.get('trade_id', '').lower()
                    and 'emergency_test' not in t.get('trade_id', '').lower()
                ]
                self.current_capital = self.initial_capital + sum(
                    t.get('pnl_usd', 0.0) for t in self.history
                )
                self.daily_start_capital = meta.get('daily_start_capital', self.current_capital)
                self.last_rollover_day = meta.get('last_rollover_day', self.last_rollover_day)
                for t in data:
                    if not t.get('exit_price') and t.get('trade_id'):
                        self.active_trades[t['trade_id']] = t
            except Exception as e:
                print(f"[TradeTracker] [ERROR] Failed to load trade history: {e}")

    def save_history(self):
        with self.lock:
            try:
                all_trades = list(self.history) + list(self.active_trades.values())
                envelope = {
                    '__meta__': {
                        'last_entry_bar': dict(self.last_entry_bar),
                        'daily_start_capital': self.daily_start_capital,
                        'last_rollover_day': self.last_rollover_day
                    },
                    'trades': all_trades
                }
                tmp = self.log_file + ".tmp"
                with open(tmp, 'w', encoding='utf-8') as f:
                    json.dump(envelope, f, indent=4)
                os.replace(tmp, self.log_file)
            except Exception:
                pass

    def update_day(self) -> None:
        with self.lock:
            import zoneinfo
            from datetime import datetime
            broker_tz = zoneinfo.ZoneInfo("Europe/Athens")
            now_day = datetime.now(broker_tz).strftime("%Y-%m-%d")
            
            if self.last_rollover_day != now_day:
                active_list = list(self.active_trades.values())
                unrealized_pnl = sum(t.get('live_pnl_usd', 0.0) for t in active_list)
                current_equity = self.current_capital + unrealized_pnl
                
                self.daily_start_capital = current_equity
                self.last_rollover_day = now_day
                print(f"[RiskGovernor] Daily starting capital rolled over to ${self.daily_start_capital:.2f} at Athens server day {now_day}")

    def trigger_entry(self, symbol: str, strategy: str, direction: int, entry_price: float, sl: float, tp: float, atr: float, macro: int, vol_regime: float, risk_mult: float = 1.0, trail_act: float = 0.5, regime_val: int = 0) -> None:
        with self.lock:
            if getattr(self, 'emergency_halt', False):
                print(f"[RiskGovernor] Entry blocked. Symbol={symbol} Strategy={strategy}. Emergency halt active.")
                return

            # --- GLOBAL RISK GOVERNOR (Blueberry Funded Account Limits) ---
            active_list = list(self.active_trades.values())
            unrealized_pnl = sum(t.get('live_pnl_usd', 0.0) for t in active_list)
            current_equity = self.current_capital + unrealized_pnl

            # 1. Daily Drawdown Check (Hard limit 5%, Guardrail 4.0%)
            daily_dd = (self.daily_start_capital - current_equity) / self.daily_start_capital * 100.0 if self.daily_start_capital > 0 else 0.0
            if daily_dd >= 4.0:
                print(f"[RiskGovernor] Entry blocked. Symbol={symbol} Strategy={strategy}. Daily drawdown ({daily_dd:.2f}%) exceeds 4% guardrail.")
                return

            # 2. Total Drawdown Check (Hard limit 10%, Guardrail 8.0% of $5,000 initial capital)
            total_dd = (self.initial_capital - current_equity) / self.initial_capital * 100.0
            if total_dd >= 8.0:
                print(f"[RiskGovernor] Entry blocked. Symbol={symbol} Strategy={strategy}. Total drawdown ({total_dd:.2f}%) exceeds 8% guardrail.")
                return

            cool_key = self._cooldown_key(strategy, symbol)
            cooldown_until = self.reentry_cooldown_until.get(cool_key, 0.0)
            if time.time() < cooldown_until:
                print(f"[RiskGovernor] Entry blocked by cooldown. Symbol={symbol} Strategy={strategy} Remaining={(cooldown_until - time.time()):.0f}s")
                return

            strategy_trades = [t for t in self.active_trades.values() if t['strategy'] == strategy]
            if any(t['symbol'] == symbol for t in strategy_trades):
                return
            
            max_concurrent = 3 if regime_val == 1 else 5
            if len(strategy_trades) >= max_concurrent:
                return
                
            if direction not in (1, -1):
                return
            
            if direction == 1 and not (sl < entry_price < tp):
                return
            if direction == -1 and not (tp < entry_price < sl):
                return
                
            stop_dist = abs(entry_price - sl)
            tick_size = globals().get("TICK_SIZES", {}).get(symbol, 0.0001)
            
            # min_stop with TP factor scaling — matches agent6_exact lines 286-290
            min_stop = 5.0 * tick_size
            tp_dist = abs(tp - entry_price)
            if stop_dist < min_stop:
                factor = min_stop / stop_dist
                stop_dist = min_stop
                tp_dist = tp_dist * factor
                if direction == 1:
                    sl = entry_price - stop_dist
                    tp = entry_price + tp_dist
                else:
                    sl = entry_price + stop_dist
                    tp = entry_price - tp_dist
            
            # Guard: reject if stop > 5x ATR — matches agent6_exact line 292
            if atr > 0 and stop_dist > 5.0 * atr:
                return
            
            # Guard: reject if stop_pct < 0.3% — live tick noise kills tighter stops
            stop_pct = stop_dist / entry_price
            if stop_pct < 0.0030:
                return
            
            risk_capital = max(0.0, self.current_capital) * ENGINE_RISK_PCT * risk_mult
            
            if risk_capital <= 0.0 or stop_dist <= 0:
                return
                
            units = risk_capital / stop_dist

            # 3. Overall Portfolio Open Stop Risk Check (Max 4% of current equity)
            open_stop_risk = 0.0
            for t in active_list:
                t_units = t.get('units', 0.0)
                t_dir = t.get('direction', 1)
                t_ep = t.get('entry_price', 0.0)
                t_sl = t.get('sl', 0.0)
                if t_dir == 1:
                    risk_pts = max(0.0, t_ep - t_sl)
                else:
                    risk_pts = max(0.0, t_sl - t_ep)
                open_stop_risk += t_units * risk_pts
            new_trade_risk = units * stop_dist
            total_portfolio_risk = open_stop_risk + new_trade_risk
            if total_portfolio_risk > current_equity * 0.04:
                print(f"[RiskGovernor] Entry blocked. Symbol={symbol} Strategy={strategy}. Total portfolio stop risk (${total_portfolio_risk:.2f}) exceeds 4% of equity (${current_equity * 0.04:.2f}).")
                return

            trade_id = f"{strategy}_{symbol}_{'LONG' if direction == 1 else 'SHORT'}_{int(time.time_ns())}"
            self.active_trades[trade_id] = {
                "trade_id": trade_id,
                "symbol": symbol,
                "strategy": strategy,
                "direction": direction,
                "entry_price": entry_price,
                "entry_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "entry_timestamp": time.time(),
                "sl": sl,
                "tp": tp,
                "units": units,
                "live_pnl_pct": 0.0,
                "live_pnl_usd": 0.0,
                "atr": atr,
                "macro": macro,
                "vol_regime": vol_regime,
                "sl_dist": stop_dist,
                "trail_act": trail_act,
                "trail_buf": 0.5
            }
            
            # --- MT5 Execution Dispatch ---
            mt5_res = self.mt5_broker.execute_trade(symbol, direction, entry_price, sl, tp, strategy)
            if mt5_res:
                self.active_trades[trade_id]["mt5_symbol"] = mt5_res.get("mt5_symbol")
                self.active_trades[trade_id]["mt5_ticket"] = mt5_res.get("mt5_ticket")
                self.active_trades[trade_id]["mt5_order"] = mt5_res.get("mt5_order")
                self.active_trades[trade_id]["mt5_deal"] = mt5_res.get("mt5_deal")
                self.active_trades[trade_id]["mt5_entry"] = mt5_res.get("mt5_entry")
                self.active_trades[trade_id]["mt5_sl"] = mt5_res.get("mt5_sl")
                self.active_trades[trade_id]["mt5_tp"] = mt5_res.get("mt5_tp")
                self.active_trades[trade_id]["mt5_lot"] = mt5_res.get("lot")
                self.active_trades[trade_id]["is_pending"] = mt5_res.get("is_pending", False)
            else:
                print(f"[TradeTracker] MT5 rejected {symbol} ({strategy}) - removing phantom trade.")
                self.active_trades.pop(trade_id, None)
                return
            # ------------------------------
            
            self.save_history()

    def update_live_pnl(self, symbol: str, current_price: float, store: Optional[Any] = None) -> None:
        with self.lock:
            # 1. Update individual trade PnL
            trades_for_symbol = [t for t in self.active_trades.values() if t['symbol'] == symbol]
            for trade in trades_for_symbol:
                if trade.get("is_pending"):
                    mt5_order = trade.get("mt5_order")
                    if mt5_order and not self.mt5_broker.dry_run:
                        if not self.mt5_broker.is_order_pending(mt5_order):
                            # Resolve real position ticket (order ticket != position ticket)
                            pos_ticket = None
                            if hasattr(self.mt5_broker, "resolve_position_from_order"):
                                pos_ticket = self.mt5_broker.resolve_position_from_order(
                                    mt5_order, trade.get("mt5_symbol")
                                )
                            if pos_ticket is None and self.mt5_broker.has_position(mt5_order):
                                pos_ticket = mt5_order  # fallback
                            if pos_ticket:
                                print(f"[MT5] Pending limit order {mt5_order} for {symbol} filled -> pos={pos_ticket}. Activating trade.")
                                trade["is_pending"] = False
                                trade["mt5_ticket"] = pos_ticket
                            else:
                                print(f"[MT5] Pending limit order {mt5_order} for {symbol} was cancelled/expired. Removing phantom trade.")
                                del self.active_trades[trade["trade_id"]]
                                continue
                    elif self.mt5_broker.dry_run:
                        trade["is_pending"] = False
                    
                    if trade.get("is_pending"):
                        continue
                        
                direction = trade['direction']
                entry_price = trade['entry_price']
                pnl_pct = (current_price - entry_price) / entry_price * 100.0 if direction == 1 else (entry_price - current_price) / entry_price * 100.0
                pnl_usd = trade['units'] * (current_price - entry_price) * direction
                trade['live_pnl_pct'] = pnl_pct
                trade['live_pnl_usd'] = pnl_usd

            # 2. Run global equity drawdown breach checks (Hard Kill-Switch)
            active_list = list(self.active_trades.values())
            unrealized_pnl = sum(t.get('live_pnl_usd', 0.0) for t in active_list)
            current_equity = self.current_capital + unrealized_pnl

            daily_dd = (self.daily_start_capital - current_equity) / self.daily_start_capital * 100.0 if self.daily_start_capital > 0 else 0.0
            total_dd = (self.initial_capital - current_equity) / self.initial_capital * 100.0

            if daily_dd >= 4.5 or total_dd >= 9.0:
                if not getattr(self, 'emergency_halt', False):
                    self.emergency_halt = True
                    print(f"[RiskGovernor] [CRITICAL] EMERGENCY HALT TRIGGERED! Daily DD={daily_dd:.2f}%, Total DD={total_dd:.2f}%. Closing all active trades.")
                
                any_closed = False
                for trade in list(self.active_trades.values()):
                    trade_sym = trade['symbol']
                    if trade_sym == symbol:
                        exit_price = current_price
                    elif store is not None:
                        snap_obj = store._data.get(trade_sym)
                        exit_price = snap_obj.price if (snap_obj and snap_obj.price > 0.0) else trade['entry_price']
                    else:
                        exit_price = trade['entry_price']

                    trade['exit_price'] = exit_price
                    trade['exit_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
                    trade['exit_reason'] = "EMERGENCY_HALT"
                    
                    entry_price = trade['entry_price']
                    direction = trade['direction']
                    live_pnl_pct = (exit_price - entry_price) / entry_price * 100.0 if direction == 1 else (entry_price - exit_price) / entry_price * 100.0
                    live_pnl_usd = trade['units'] * (exit_price - entry_price) * direction

                    pnl_pct = live_pnl_pct - 0.06
                    pnl_usd = live_pnl_usd - (trade['units'] * entry_price * 0.0006)
                    
                    trade['pnl_pct'] = pnl_pct
                    trade['pnl_usd'] = pnl_usd
                    
                    self.history.append(trade)
                    self.current_capital += pnl_usd
                    
                    if trade.get("mt5_ticket") and not self.mt5_broker.dry_run:
                        self._broker_submit(self.mt5_broker.close_position, trade["mt5_ticket"], "EMERGENCY_HALT")
                        
                    del self.active_trades[trade['trade_id']]
                    any_closed = True
                    
                    closed_strategy = trade.get('strategy', '')
                    for cb in self.on_close_callbacks:
                        try:
                            cb(closed_strategy, self.current_capital)
                        except Exception:
                            pass
                if any_closed:
                    self.save_history()

    def check_exits(self, symbol: str, current_price: float, current_atr_or_dict: Any = 0.0) -> None:
        with self.lock:
            trades_for_symbol = [t for t in self.active_trades.values() if t['symbol'] == symbol]
            any_closed = False
            for trade in trades_for_symbol:
                if trade.get("is_pending"):
                    continue
                direction = trade['direction']
                sl = trade['sl']
                tp = trade['tp']
                entry_price = trade['entry_price']
                
                # Resolve current ATR based on strategy if a dict is passed
                if isinstance(current_atr_or_dict, dict):
                    strategy = trade.get('strategy', STRATEGY_DISPLAY_NAME)
                    current_atr = float(current_atr_or_dict.get(strategy, 0.0) or 0.0)
                else:
                    current_atr = float(current_atr_or_dict or 0.0)
                
                # --- Trailing Stop Logic (Dynamic — follows price) ---
                sl_dist = trade.get('sl_dist')
                trail_act = trade.get('trail_act')
                trail_buf = trade.get('trail_buf', 0.5)
                
                activation_r = float(trade.get("trail_act", 1.0) or 0.0)
                if sl_dist and activation_r > 0.0 and current_atr > 0:
                    if direction == 1:
                        cur_r = (current_price - entry_price) / sl_dist
                        if cur_r >= activation_r:
                            ns = entry_price + (cur_r - trail_buf) * sl_dist
                            if ns > sl:
                                if abs(ns - sl) < sl_dist * 0.05:
                                    pass  # Throttle modification spam
                                else:
                                    trade["sl"] = ns
                                    sl = ns
                                    if trade.get("mt5_ticket") and not self.mt5_broker.dry_run:
                                        mt5_sl = self._translate_to_mt5_price(trade, sl)
                                        mt5_tp = self._translate_to_mt5_price(trade, trade["tp"])
                                        self._broker_submit(self.mt5_broker.modify_sltp, trade["mt5_symbol"], trade["mt5_ticket"], mt5_sl, mt5_tp)
                    else:
                        cur_r = (entry_price - current_price) / sl_dist
                        if cur_r >= activation_r:
                            ns = entry_price - (cur_r - trail_buf) * sl_dist
                            if ns < sl:
                                if abs(ns - sl) < sl_dist * 0.05:
                                    pass  # Throttle modification spam
                                else:
                                    trade["sl"] = ns
                                    sl = ns
                                    if trade.get("mt5_ticket") and not self.mt5_broker.dry_run:
                                        mt5_sl = self._translate_to_mt5_price(trade, sl)
                                        mt5_tp = self._translate_to_mt5_price(trade, trade["tp"])
                                        self._broker_submit(self.mt5_broker.modify_sltp, trade["mt5_symbol"], trade["mt5_ticket"], mt5_sl, mt5_tp)
                
                should_close = False
                reason = ""
                
                # --- MAX_BARS Timeout Exit (Parity with agent6_exact.py) ---
                # 96 bars of 15m = 24 hours (86400 seconds)
                elapsed_time = time.time() - trade.get('entry_timestamp', time.time())
                if elapsed_time >= 86400:
                    should_close = True
                    reason = "TIMEOUT"
                
                if not should_close:
                    if direction == 1:
                        if current_price <= sl:
                            should_close = True
                            reason = "SL"
                        elif current_price >= tp:
                            should_close = True
                            reason = "TP"
                    else:
                        if current_price >= sl:
                            should_close = True
                            reason = "SL"
                        elif current_price <= tp:
                            should_close = True
                            reason = "TP"
                        
                if should_close:
                    if trade.get("closing_dispatched"):
                        continue
                        
                    exit_price = current_price
                    trade['exit_price'] = exit_price
                    trade['exit_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
                    trade['exit_reason'] = reason
                    
                    entry_price = trade['entry_price']
                    pnl_pct = (exit_price - entry_price) / entry_price * 100.0 if direction == 1 else (entry_price - exit_price) / entry_price * 100.0
                    pnl_pct -= 0.06
                    
                    pnl_usd = (trade['units'] * (exit_price - entry_price) * direction) - (trade['units'] * entry_price * 0.0006)
                    
                    trade['pnl_pct'] = pnl_pct
                    trade['pnl_usd'] = pnl_usd
                    
                    if trade.get("mt5_ticket") and not self.mt5_broker.dry_run:
                        trade["closing_dispatched"] = True
                        
                        def make_close_cb(t_id, t_dict):
                            def _cb(f):
                                try:
                                    res = f.result()
                                    with self.lock:
                                        if res and t_id in self.active_trades:
                                            self.history.append(t_dict)
                                            self.current_capital += t_dict.get('pnl_usd', 0)
                                            del self.active_trades[t_id]
                                            self.save_history()
                                            
                                            try:
                                                import MetaTrader5 as mt5
                                                acc = mt5.account_info()
                                                if acc and acc.balance > 0:
                                                    self.current_capital = acc.balance
                                            except Exception:
                                                pass
                                        elif not res and t_id in self.active_trades:
                                            print(f"[MT5] Close rejected/failed for {t_id}. Re-arming local state.")
                                            self.active_trades[t_id]["closing_dispatched"] = False
                                except Exception as e:
                                    print(f"[MT5] Exception during async close for {t_id}: {e}")
                                    with self.lock:
                                        if t_id in self.active_trades:
                                            self.active_trades[t_id]["closing_dispatched"] = False
                            return _cb
                            
                        if hasattr(self, "broker_executor") and self.broker_executor:
                            fut = self.broker_executor.submit(self.mt5_broker.close_position, trade["mt5_ticket"], reason)
                            fut.add_done_callback(make_close_cb(trade["trade_id"], trade.copy()))
                    else:
                        self.history.append(trade)
                        self.current_capital += pnl_usd
                        del self.active_trades[trade['trade_id']]
                        any_closed = True
                        
                    closed_strategy = trade.get("strategy", "")
                    cooldown_secs = self._cooldown_secs_after_close(closed_strategy, reason)
                    if cooldown_secs > 0:
                        self.reentry_cooldown_until[self._cooldown_key(closed_strategy, symbol)] = time.time() + cooldown_secs

                    for cb in self.on_close_callbacks:
                        try:
                            cb(closed_strategy, self.current_capital)
                        except Exception:
                            pass
                    
            if any_closed:
                self.save_history()

    def get_stats(self) -> dict:
        with self.lock:
            total = len(self.history)
            if total == 0:
                return {"total": 0, "winrate": 0.0, "total_pnl_usd": 0.0, "current_capital": self.current_capital}
            wins = sum(1 for t in self.history if t.get('pnl_usd', 0.0) > 0)
            total_pnl_usd = sum(t.get('pnl_usd', 0.0) for t in self.history)
            return {
                "total": total,
                "winrate": (wins / total) * 100.0,
                "total_pnl_usd": total_pnl_usd,
                "current_capital": self.current_capital
            }

    def reconcile_with_mt5(self) -> None:
        """
        Keep active_trades in absolute sync with MT5 terminal positions.
        - Drop local trades whose MT5 position is gone (broker SL/TP hit).
        - Promote filled pending orders to live tickets.
        Called periodically from the rollover watchdog (non-blocking path).
        """
        if getattr(self.mt5_broker, "dry_run", True):
            return
        with self.lock:
            try:
                broker_positions = {}
                if hasattr(self.mt5_broker, "list_engine_positions"):
                    for p in self.mt5_broker.list_engine_positions():
                        broker_positions[int(p.ticket)] = p

                stale_ids = []
                for tid, trade in list(self.active_trades.items()):
                    if trade.get("is_pending"):
                        mt5_order = trade.get("mt5_order")
                        if mt5_order and not self.mt5_broker.is_order_pending(mt5_order):
                            pos_ticket = None
                            if hasattr(self.mt5_broker, "resolve_position_from_order"):
                                pos_ticket = self.mt5_broker.resolve_position_from_order(
                                    mt5_order, trade.get("mt5_symbol")
                                )
                            if pos_ticket:
                                trade["is_pending"] = False
                                trade["mt5_ticket"] = pos_ticket
                            else:
                                stale_ids.append(tid)
                        continue

                    ticket = trade.get("mt5_ticket")
                    if not ticket:
                        continue
                    if ticket not in broker_positions and not self.mt5_broker.has_position(ticket):
                        # Broker already closed (SL/TP) — archive locally at last known price
                        trade["exit_price"] = trade.get("entry_price")
                        trade["exit_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                        trade["exit_reason"] = "BROKER_SYNC"
                        trade["pnl_pct"] = trade.get("live_pnl_pct", 0.0) - 0.06
                        trade["pnl_usd"] = trade.get("live_pnl_usd", 0.0)
                        self.history.append(trade)
                        self.current_capital += trade.get("pnl_usd", 0.0)
                        stale_ids.append(tid)
                        print(f"[MT5 SYNC] Removed orphaned local trade {tid} (ticket={ticket})")

                for tid in stale_ids:
                    self.active_trades.pop(tid, None)
                if stale_ids:
                    self.save_history()
            except Exception as e:
                print(f"[MT5 SYNC] reconcile error: {e}")

class SnapshotStore:
    def __init__(self, symbols: List[str], predictor: LiveStrategyPredictor = None, liquidation_predictor: Any = None, trade_tracker: Any = None, trend_pull_predictor: Any = None):
        self._data: Dict[str, AssetSnapshot] = {s: AssetSnapshot(symbol=s) for s in symbols}
        self._locks = {s: asyncio.Lock() for s in symbols}
        self._seq = 0
        self.predictor = predictor
        self.liquidation_predictor = liquidation_predictor
        self.trend_pull_predictor = trend_pull_predictor
        self.trade_tracker = trade_tracker

    async def update(self, symbol: str, source: str = "binance", **patch: Any) -> None:
        if symbol not in self._data:
            return
        async with self._locks[symbol]:
            cur = self._data[symbol]
            clean_patch = {}
            is_binance_symbol = symbol not in ("XAUUSDT", "XAGUSDT", "CLUSDT", "NATGASUSDT")
            for k, v in patch.items():
                if not hasattr(cur, k):
                    continue
                if k in ("price", "open", "high", "low", "close"):
                    # allow coinglass to provide price if Binance feeds are blocked
                    pass
                    fv = finite_float_or_none(v)
                    if fv is None:
                        continue
                    if k == "price" and fv <= 0.0:
                        continue
                    clean_patch[k] = fv
                else:
                    clean_patch[k] = v

            if not clean_patch:
                return

            self._seq += 1
            new_snap = dataclasses.replace(cur, seq=self._seq, ts_ns=time.time_ns(), **clean_patch)
            
            if self.trade_tracker:
                self.trade_tracker.update_day()

            price_updated = "price" in clean_patch
            
            if self.trade_tracker and price_updated:
                atr_dict = {}
                if self.predictor:
                    cached = self.predictor._cached_signal.get(symbol, {})
                    atr_dict[STRATEGY_DISPLAY_NAME] = cached.get('atr_val', 0.0)
                if getattr(self, 'liquidation_predictor', None):
                    atr_dict["ML_Liquidation_Runner"] = getattr(self.liquidation_predictor, 'latest_atr', {}).get(symbol, 0.0)
                if getattr(self, 'trend_pull_predictor', None):
                    atr_dict["ML_Trend_Pull"] = getattr(self.trend_pull_predictor, 'latest_atr', {}).get(symbol, 0.0)
                self.trade_tracker.check_exits(symbol, new_snap.price, atr_dict)
                self.trade_tracker.update_live_pnl(symbol, new_snap.price, self)
            price_fresh = price_updated and new_snap.price > 0.0
            self._data[symbol] = new_snap

            if price_fresh:
                def _run_ml_predictors(sym: str, snap_obj, tracker):
                    if self.predictor:
                        self.predictor.on_tick_update(sym, snap_obj, tracker)
                    if getattr(self, 'liquidation_predictor', None):
                        self.liquidation_predictor.on_tick_update(sym, snap_obj, tracker)
                    if getattr(self, 'trend_pull_predictor', None):
                        self.trend_pull_predictor.on_tick_update(sym, snap_obj, tracker)
                        
                # Fire and forget ML predictions so they don't block the WebSocket price stream
                asyncio.create_task(asyncio.to_thread(_run_ml_predictors, symbol, new_snap, self.trade_tracker))

    def snapshot(self) -> Dict[str, AssetSnapshot]:
        # GIL-atomic shallow copy of dict references; safe for lock-free reads
        return dict(self._data)

# --- BINANCE FOOTPRINT ENGINE ---
TICK_SIZES = {
    "BTCUSDT": 10.0,       # ~0.016% - 12-31 rows
    "ETHUSDT": 0.25,       # ~0.015% - 12-30 rows (was 0.5, too blocky)
    "XRPUSDT": 0.0002,     # ~0.018% - 11-27 rows
    "BNBUSDT": 0.05,       # ~0.009% - 23-58 rows (was 0.1)
    "SOLUSDT": 0.01,       # ~0.016% - 12-32 rows (was 0.02)
    "DOGEUSDT": 0.00002,   # ~0.024% - 8-21 rows
    "ADAUSDT": 0.00003,    # ~0.019% - 10-26 rows (was 0.00005)
    "TRXUSDT": 0.00005,    # ~0.016% - 13-32 rows
    "LINKUSDT": 0.001,     # ~0.013% - 15-38 rows (was 0.002)
    "AVAXUSDT": 0.001,     # ~0.016% - 12-32 rows (was 0.002)
    "XLMUSDT": 0.00005,    # ~0.028% - 7-18 rows
    "HBARUSDT": 0.00002,   # ~0.026% - 8-20 rows
    "LTCUSDT": 0.01,       # ~0.024% - 8-21 rows (was 0.02, critical fix)
    "DOTUSDT": 0.0002,     # ~0.021% - 9-23 rows (was 0.0005, critical fix)
    "XMRUSDT": 0.05,       # ~0.016% - 12-31 rows
    "NEARUSDT": 0.0005,    # ~0.024% - 8-21 rows
    "UNIUSDT": 0.0005,     # ~0.021% - 10-24 rows
    "FILUSDT": 0.0002,     # ~0.027% - 7-19 rows
    "SUIUSDT": 0.0005,     # Tick size for SUI
    "XAUUSDT": 0.25,       # Tick size for Gold
    "XAGUSDT": 0.002,      # Tick size for Silver
    "CLUSDT": 0.005,       # Tick size for Crude Oil
    "NATGASUSDT": 0.0002,  # Tick size for Natural Gas
}

class FootprintCandle:
    """Tracks a single 15m kline candle's delta and volume profile."""
    def __init__(self, tick_size: float):
        self.tick_size = tick_size
        self.candle_open_ms: int = 0
        self.delta: float = 0.0
        self.volume_profile: Dict[float, float] = collections.defaultdict(float)

    def _bucket(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size

    def update(self, candle_open_ms: int, buy_vol: float, sell_vol: float, close_price: float) -> None:
        """Called with the latest kline data. Resets automatically on new candle."""
        if candle_open_ms != self.candle_open_ms:
            # New 15m candle opened — reset everything
            self.candle_open_ms = candle_open_ms
            self.delta = 0.0
            self.volume_profile.clear()
        self.delta = buy_vol - sell_vol
        # Volume profile: accumulate on close price bucket (klines don't provide tick-level data)
        bucket = self._bucket(close_price)
        self.volume_profile[bucket] = buy_vol + sell_vol

    @property
    def poc(self) -> float:
        if not self.volume_profile:
            return 0.0
        return max(self.volume_profile.items(), key=lambda kv: kv[1])[0]


class BinanceTradePriceWebSocketFeed:
    def __init__(self, symbols: List[str], store: SnapshotStore):
        self.symbols = [
            s for s in symbols
            if s not in ("XAUUSDT", "XAGUSDT", "CLUSDT", "NATGASUSDT")
        ]
        self.store = store
        self.running = True
        self.last_heartbeat_ns = time.time_ns()
        self.last_emit_ns: Dict[str, int] = {}
        self.tab_id = "binance_ws"
        self.skip_watchdog = True  # WS may be network-blocked (ISP); REST feed covers all price needs
        
    async def run(self) -> None:
        if not self.symbols:
            return
            
        streams = "/".join(f"{s.lower()}@aggTrade" for s in self.symbols)
        url = f"wss://fstream.binance.com/stream?streams={streams}"
        print(f"[Binance WS] Starting with URL: {url}")
        
        while self.running:
            try:
                # Wrap connect with timeout to prevent watchdog hangs
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    open_timeout=15,  # Prevents hanging during connection
                    max_queue=4096,
                ) as ws:
                    print("[Binance WS] Connected trade price stream.")
                    async for raw in ws:
                        if not self.running:
                            break
                            
                        self.last_heartbeat_ns = time.time_ns()
                        
                        try:
                            msg = json.loads(raw)
                            data = msg.get("data", {})
                            sym = data.get("s")
                            p_str = data.get("p")
                            
                            def finite_float_or_none(v):
                                try:
                                    val = float(v)
                                    import math
                                    if math.isfinite(val): return val
                                    return None
                                except:
                                    return None

                            price = finite_float_or_none(p_str)
                            
                            if sym not in self.symbols or price is None or price <= 0:
                                continue
                                
                            # Track WebSocket message queue & processing lag
                            event_time_ms = data.get("E")
                            if event_time_ms:
                                lag_sec = (time.time() * 1000 - event_time_ms) / 1000.0
                                if lag_sec > 2.0:
                                    print(f"\n[ALERT] [LAG] WebSocket message processing lag for {sym} is {lag_sec:.2f}s!")
                                
                            now_ns = time.time_ns()
                            last_ns = self.last_emit_ns.get(sym, 0)
                            if now_ns - last_ns < 150_000_000:  # 150 ms
                                continue
                            self.last_emit_ns[sym] = now_ns
                            
                            await self.store.update(sym, source="binance_ws", price=price)
                        except Exception as inner_e:
                            continue
                            
            except Exception as e:
                print(f"[Binance WS] Disconnected/error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5.0)


class BinanceFootprintFeed:
    """Polls Binance Futures klines REST API every 5s to derive 15m candle delta and POC."""
    def __init__(self, symbols: List[str], store: SnapshotStore):
        self.symbols = symbols
        self.store = store
        # Exclude commodities not on Binance Futures
        self.valid_symbols = [s for s in symbols if s not in ["XAUUSDT", "XAGUSDT", "CLUSDT", "NATGASUSDT"]]
        self.candles = {}
        for s in self.valid_symbols:
            tick = TICK_SIZES.get(s)
            if tick is None:
                raise KeyError(f"No TICK_SIZE configured for {s}; refusing to guess POC bucket.")
            self.candles[s] = FootprintCandle(tick)
        self.last_heartbeat_ns = time.time_ns()
        self.running = True
        self.consecutive_failures = 0
        self.was_failing = False

    async def run(self) -> None:
        url = "https://fapi.binance.com/fapi/v1/klines"

        async def _fetch_one(session: aiohttp.ClientSession, idx: int, sym: str) -> None:
            try:
                params = {"symbol": sym, "interval": "15m", "limit": 1}
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data:
                            item = data[-1]
                            candle_open_ms = int(item[0])
                            tot_vol = float(item[5])
                            buy_vol = float(item[9])
                            sell_vol = tot_vol - buy_vol
                            close_price = float(item[4])
                            candle = self.candles[sym]
                            candle.update(candle_open_ms, buy_vol, sell_vol, close_price)
                            await self.store.update(
                                sym,
                                source="binance",
                                price=close_price,
                                open=float(item[1]),
                                high=float(item[2]),
                                low=float(item[3]),
                                close=close_price,
                                fp_delta=candle.delta,
                                fp_poc=candle.poc
                            )
                            successes[idx] = True
            except Exception:
                pass  # individual symbol errors are covered by the outer consolidated warning

        while self.running:
            try:
                # Use ThreadedResolver to bypass shielded asyncio resolver warnings
                connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver(), family=socket.AF_INET)
                async with aiohttp.ClientSession(connector=connector) as session:
                    while self.running:
                        self.last_heartbeat_ns = time.time_ns()
                        successes = [False] * len(self.valid_symbols)
                        await asyncio.gather(*[_fetch_one(session, idx, s) for idx, s in enumerate(self.valid_symbols)])
                        
                        if any(successes):
                            if self.was_failing:
                                print("[Binance Feed] [INFO] Connection restored.")
                                self.was_failing = False
                            self.consecutive_failures = 0
                        else:
                            self.consecutive_failures += 1
                            if self.consecutive_failures == 1:
                                print("[Binance Feed] [WARN] Connection issues detected (all queries failed).")
                            elif self.consecutive_failures % 30 == 0:  # ~Every 1 minute
                                print(f"[Binance Feed] [WARN] Connection is still down (consecutive failures: {self.consecutive_failures})")
                            self.was_failing = True
                            
                        await asyncio.sleep(2.0)
            except Exception as e:
                if self.consecutive_failures <= 3:
                    print(f"[Binance Feed] [WARN] Session error: {e}. Retrying in 10s...")
                await asyncio.sleep(10.0)

# --- COINGLASS JS SHIMS ---
INIT_SCRIPT = ""

# JS run inside the TradingView iframe — extracts OHLCV and all indicator legend
SINGLE_FRAME_EXTRACTION_JS = r'''() => {
    let data = {
        volume: 'N/A',
        open_interest: 'N/A',
        funding_rate: '0.0',
        ls_ratio: 'N/A',
        futures_cvd: 'N/A',
        spot_cvd: 'N/A',
        rsi: 'N/A',
        liquidations_long: '0.0',
        liquidations_short: '0.0',
        coins_bid: 'N/A',
        coins_ask: 'N/A',
        dollars_bid: 'N/A',
        dollars_ask: 'N/A',
        whale_index: 'N/A',
        taker_buy_count: 'N/A',
        taker_sell_count: 'N/A'
    };
    let minusRe = /[\u2212\u2012\u2013\u2014]/g;
    let numRe = /^[+-]?\d+(\.\d+)?([KkMmBbTt])?%?$|^[+-]?0\.0+\d*%?$/;

    let extractNumericVals = (fullText) => {
        let cleanText = fullText.replace(minusRe, '-');
        let matches = cleanText.match(/[+-]?\d+(?:\.\d+)?(?:[KkMmBbTt])?%?/g) || [];
        return matches;
    };

    // 1. Extract Symbol & OHLC
    let seriesEl = document.querySelector('[data-name="legend-series-item"], [class*="series-"]');
    if (seriesEl) {
        let txt = (seriesEl.innerText || '').replace(minusRe, '-');
        let lines = txt.split('\n').map(s => s.trim()).filter(s => s);
        if (lines.length > 0) {
            data.symbol = lines[0].split(' ')[0].trim();
            let valTitles = Array.from(seriesEl.querySelectorAll('[class*="valueTitle-"]'));
            let valValues = Array.from(seriesEl.querySelectorAll('[class*="valueValue-"]'));
            for (let k = 0; k < valTitles.length; k++) {
                let t = valTitles[k].innerText.trim();
                let v = valValues[k] ? valValues[k].innerText.trim().replace(minusRe, '-') : '';
                if (t === 'O') data.open = v;
                else if (t === 'H') data.high = v;
                else if (t === 'L') data.low = v;
                else if (t === 'C') data.close = v;
            }
        }
    }

    // 2. Extract Indicators
    let legends = Array.from(document.querySelectorAll('[data-name="legend-source-item"], [class*="study-"], .legend-TG1_J52N, [class*="legend-"]'));
    for (let el of legends) {
        let fullText = el.innerText || '';
        if (!fullText.trim()) continue;
        let lines = fullText.split('\n').map(s => s.trim()).filter(s => s);
        if (lines.length === 0) continue;
        let title = lines[0].toLowerCase();
        let lowerText = fullText.toLowerCase();

        if (el === seriesEl || el.className.includes('series-')) continue;

        let numVals = extractNumericVals(fullText);

        if (title.includes('volume') && !title.includes('cumulative') && !title.includes('bid & ask') && !title.includes('taker')) {
            if (numVals.length > 0) data.volume = numVals[numVals.length - 1];
        } else if (title.includes('open interest') && !title.includes('funding')) {
            if (numVals.length > 0) data.open_interest = numVals[numVals.length - 1];
        } else if (title.includes('funding')) {
            for (let j = lines.length - 1; j >= 1; j--) {
                let t = lines[j].trim();
                let clean = t.replace(/,/g, '').replace(minusRe, '-').replace(/%/g, '');
                if (numRe.test(clean)) {
                    let raw = parseFloat(clean);
                    data.funding_rate = !isNaN(raw) ? (raw * 100).toFixed(4) + '%' : t;
                    break;
                }
            }
        } else if (title.includes('long/short')) {
            if (numVals.length > 0) data.ls_ratio = numVals[numVals.length - 1];
        } else if (title.includes('futures cumulative')) {
            if (numVals.length > 0) data.futures_cvd = numVals[numVals.length - 1];
        } else if (title.includes('spot cumulative')) {
            if (numVals.length > 0) data.spot_cvd = numVals[numVals.length - 1];
        } else if (title.includes('rsi')) {
            if (numVals.length > 0) {
                let rsiVal = numVals[numVals.length - 1];
                if (rsiVal !== '100.00' && rsiVal !== '0.00') data.rsi = rsiVal;
            }
        } else if (title.includes('liquidation')) {
            for (let j = 1; j < lines.length; j++) {
                let t = lines[j].trim();
                let clean = t.replace(/,/g, '').replace(minusRe, '-');
                if (numRe.test(clean) || clean === '∅') {
                    if (clean === '∅') {
                        // ignore
                    } else if (clean.includes('-')) {
                        data.liquidations_short = t;
                    } else {
                        data.liquidations_long = t;
                    }
                }
            }
        } else if (lowerText.includes('bid & ask') || lowerText.includes('taker buy/sell volume') || lowerText.includes('taker buy/sell value')) {
            if (lowerText.includes('coins') || lowerText.includes('volume')) {
                if (numVals.length >= 2) {
                    data.coins_bid = numVals[numVals.length - 2];
                    data.coins_ask = numVals[numVals.length - 1];
                }
            }
            if (lowerText.includes('dollars') || lowerText.includes('value')) {
                if (numVals.length >= 2) {
                    data.dollars_bid = numVals[numVals.length - 2];
                    data.dollars_ask = numVals[numVals.length - 1];
                }
            }
        } else if (lowerText.includes('whale index')) {
            if (numVals.length > 0) data.whale_index = numVals[numVals.length - 1];
        } else if (lowerText.includes('taker buy/sell count')) {
            if (numVals.length >= 3) {
                data.taker_buy_count = numVals[numVals.length - 3];
                data.taker_sell_count = numVals[numVals.length - 2];
            } else if (numVals.length === 2) {
                data.taker_buy_count = numVals[numVals.length - 2];
                data.taker_sell_count = numVals[numVals.length - 1];
            }
        }
    }
    
    let rawLegends = legends.map(el => el.innerText || '');
    return { success: true, data: data, rawLegends: rawLegends };
}
'''

# --- BROWSER SCRAPER PAGE CLASS ---
class CoinglassTab:
    def __init__(self, context: BrowserContext, symbols: List[str], store: SnapshotStore, tab_id: str):
        self.context = context
        self.symbols = symbols
        self.store = store
        self.tab_id = tab_id
        self.is_seeding = False
        self.page: Optional[Page] = None
        self.last_heartbeat_ns = time.time_ns()
        self.running = True
        self._response_tasks: set[asyncio.Task] = set()
        self.poll_failures = 0

    async def start(self) -> None:
        self.page = await self.context.new_page()
        
        # Suppress noisy TradingView internal console spam; only print errors and CoinGlass messages
        def _on_console(msg):
            text = msg.text
            typ = msg.type
            skip_patterns = (
                "Recurring script engine stop",
                "76 custom indicators loaded",
                "Content Security Policy",
                "WebSocket connection to",
                "ERR_NAME_NOT_RESOLVED",
                "502",
                "wss.coinglass.com",
                "net::ERR_",
                "Failed to fetch",
            )
            if any(p in text for p in skip_patterns):
                return
            if typ in ("error", "warning") or "coinglass" in text.lower():
                print(f"[{self.tab_id} CONSOLE] {typ} {text}")

        def _on_page_error(exc):
            msg = str(exc)
            # Filter generic browser resource errors that are not actionable
            if any(p in msg for p in ("unknown compression", "net::", "ERR_", "Failed to fetch", "ResizeObserver", "reading 'symbol'")):
                return
            print(f"[{self.tab_id} PAGE ERROR] {msg}")

        self.page.on("console", _on_console)
        self.page.on("pageerror", _on_page_error)
        
        # Intercept HTTP API responses natively to capture Open Interest and Funding Rates securely
        # without introducing compression encoding errors on the page.
        async def handle_response(response):
            try:
                url = response.url
                if any(k in url for k in ("open-interest", "funding-rate", "liquidation", "long-short", "rsi", "cumulative-volume")):
                    body = await response.text()
                    await self._route_payload({"url": url, "body": body})
            except Exception:
                pass

        def _spawn_response_task(response):
            task = asyncio.create_task(handle_response(response))
            self._response_tasks.add(task)
            task.add_done_callback(self._response_tasks.discard)

        self.page.on("response", _spawn_response_task)
        
        print(f"[{self.tab_id}] Opening layout: {URL}...")
        await self.page.goto(URL, wait_until="load", timeout=45000)
        print(f"[{self.tab_id}] Waiting 15 seconds for layout to load...")
        await asyncio.sleep(15)  # Wait 15 seconds for S9 chart renders

    async def reconnect(self, focus_lock: asyncio.Lock) -> None:
        print(f"[{self.tab_id}] [RECOVERY] Attempting to reconnect/restart the tab...")
        self.is_seeding = True
        try:
            self.running = False
            if self.page:
                try:
                    await self.page.close()
                except Exception:
                    pass
            self.running = True
            await self.start()
            await self.inject_and_configure_all(focus_lock)
            print(f"[{self.tab_id}] [RECOVERY] Tab successfully restarted and re-configured.")
            self.last_heartbeat_ns = time.time_ns()
        except Exception as e:
            print(f"[{self.tab_id}] [RECOVERY ERROR] Failed to restart tab: {e}")
        finally:
            self.is_seeding = False

    async def inject_and_configure_all(self, focus_lock: asyncio.Lock):
        """Programmatic JS-based S9 indicator & symbol configuration"""
        print(f"[{self.tab_id}] Bringing tab to front...")
        await self.page.bring_to_front()
        await asyncio.sleep(0.5)
        
        # Wait for layout containers to render fully
        try:
            print(f"[{self.tab_id}] Waiting for layout containers to render...")
            await self.page.wait_for_selector("#tv_chart_container_win1, #tv_chart_container_main", state="attached", timeout=30000)
            await self.page.wait_for_selector("#tv_chart_container_win9", state="attached", timeout=30000)
            await asyncio.sleep(2.0)
        except Exception as e:
            print(f"[{self.tab_id}] [WARN] Timeout waiting for layout containers: {e}")

        print(f"[{self.tab_id}] Configuring symbols and indicators on grid layout via JS API...")
        for win_idx, symbol in enumerate(self.symbols, start=1):
            print(f"[{self.tab_id}] [Config] Configuring window {win_idx}/9 for {symbol}")
            container_id = f"tv_chart_container_win{win_idx}"
            selector = f"#{container_id}" if win_idx != 1 else f"#{container_id}, #tv_chart_container_main"
            container = self.page.locator(selector).first
            
            if await container.count() > 0:
                try:
                    iframe = container.locator("iframe").first
                    await iframe.wait_for(state="attached", timeout=15000)
                    iframe_handle = await iframe.element_handle(timeout=15000)
                    if iframe_handle:
                        frame = await iframe_handle.content_frame()
                        if frame:
                            # Evaluate programmatic setup script inside the iframe
                            res = await frame.evaluate(f'''async () => {{
                                try {{
                                    // 1. Wait for TradingView API and CoinGlass custom studies metadata cache to load
                                    let apiReady = false;
                                    for (let i = 0; i < 40; i++) {{
                                        if (typeof tradingViewApi !== 'undefined' && 
                                            tradingViewApi.activeChart && 
                                            tradingViewApi._chartApiInstance && 
                                            tradingViewApi._chartApiInstance._studyEngine && 
                                            tradingViewApi._chartApiInstance._studyEngine._metainfoCache) {{
                                            
                                            let cache = tradingViewApi._chartApiInstance._studyEngine._metainfoCache;
                                            let keys = Object.keys(cache);
                                            if (keys.some(k => cache[k].description && cache[k].description.includes('CoinGlass'))) {{
                                                apiReady = true;
                                                break;
                                            }}
                                        }}
                                        await new Promise(r => setTimeout(r, 500));
                                    }}
                                    
                                    if (!apiReady) {{
                                        return {{ success: false, error: 'CoinGlass indicator metadata cache not ready' }};
                                    }}
                                    
                                    // 2. Set Symbol
                                    tradingViewApi.changeSymbol("Binance_{symbol}");
                                    
                                    // 3. Set Timeframe to 15m
                                    if (typeof chartWidgetCollection !== 'undefined') {{
                                        chartWidgetCollection.setResolution('15');
                                    }}
                                    
                                    // 4. Scan and inject missing indicators
                                    let ac = tradingViewApi.activeChart();
                                    if (ac) {{
                                         let existing = [];
                                         try {{
                                             existing = ac.getAllStudies() || [];
                                         }} catch (err) {{}}
                                         
                                         let hasVolume = existing.some(s => s.name && s.name.includes('Volume') && !s.name.includes('Delta'));
                                         let hasFutCVD = existing.some(s => s.name && s.name.includes('Futures Cumulative Volume Delta'));
                                         let hasSpotCVD = existing.some(s => s.name && s.name.includes('Spot Cumulative Volume Delta'));
                                         let hasRSI = existing.some(s => s.name && s.name.includes('Relative Strength Index'));
                                         let hasFunding = existing.some(s => s.name && s.name.includes('Funding Rates'));
                                         let hasLiq = existing.some(s => s.name && s.name.includes('Aggregated Liquidations'));
                                         let hasLS = existing.some(s => s.name && s.name.includes('Long/Short Ratio'));
                                         let hasOI = existing.some(s => s.name && s.name.includes('Open Interest'));
                                         let hasWhale = existing.some(s => s.name && s.name.includes('Whale Index'));
                                         let hasTaker = existing.some(s => s.name && s.name.includes('Taker Buy/Sell Count'));
                                         
                                         let bidAsks = existing.filter(s => s.name && s.name.includes('Bid & Ask'));
                                         
                                         let injectedAny = false;
                                         if (!hasVolume) {{ ac.createStudy('Volume', false, false); injectedAny = true; }}
                                         if (!hasFutCVD) {{ ac.createStudy('<CoinGlass> Aggregated Futures Cumulative Volume Delta (CVD)', false, false); injectedAny = true; }}
                                         if (!hasSpotCVD) {{ ac.createStudy('<CoinGlass> Aggregated Spot Cumulative Volume Delta (CVD)', false, false); injectedAny = true; }}
                                         if (!hasRSI) {{ ac.createStudy('Relative Strength Index', false, false); injectedAny = true; }}
                                         if (!hasFunding) {{ ac.createStudy('<CoinGlass> Funding Rates(Open Interest Weighted,Candles)', false, false); injectedAny = true; }}
                                         if (!hasLiq) {{ ac.createStudy('<CoinGlass> Aggregated Liquidations ', false, false); injectedAny = true; }}
                                         if (!hasLS) {{ ac.createStudy('<CoinGlass> Long/Short Ratio (Accounts)', false, false); injectedAny = true; }}
                                         if (!hasOI) {{ ac.createStudy('<CoinGlass> Aggregated Open Interest(STABLECOIN-margined,Candles)', false, false); injectedAny = true; }}
                                         if (!hasWhale) {{ ac.createStudy('<CoinGlass> Whale Index', false, false); injectedAny = true; }}
                                         if (!hasTaker) {{ ac.createStudy('<CoinGlass> Taker Buy/Sell Count', false, false); injectedAny = true; }}
                                         
                                         if (bidAsks.length < 2) {{
                                             for (let b of bidAsks) {{
                                                 try {{ ac.removeStudy(b.id); }} catch(e) {{}}
                                             }}
                                             ac.createStudy('<CoinGlass> Aggregated Futures Bid & Ask ', false, false, {{ "Depth": 1, "symbol": "Main chart symbol", "Measure": "Coins" }});
                                             ac.createStudy('<CoinGlass> Aggregated Futures Bid & Ask ', false, false, {{ "Depth": 1, "symbol": "Main chart symbol", "Measure": "Dollars" }});
                                             injectedAny = true;
                                         }}
                                         
                                         // Force save layout if anything was injected
                                         if (injectedAny && tradingViewApi._saveChartService && typeof tradingViewApi._saveChartService.saveChart === 'function') {{
                                             try {{
                                                 tradingViewApi._saveChartService.saveChart();
                                             }} catch(se) {{}}
                                         }}
                                         
                                         let dump = ac.getAllStudies().map(s => ({{id: s.id, name: s.name}}));
                                         return {{ success: true, dump: dump }};
                                    }}
                                    return {{ success: false, error: 'Active chart not found' }};
                                }} catch (e) {{
                                    return {{ success: false, error: e.message }};
                                }}
                            }}''')
                            if res and "dump" in res:
                                try:
                                    with open(os.path.join(base_dir, "Seeding", f"studies_{self.tab_id}_{symbol}.json"), "w") as f:
                                        json.dump(res["dump"], f, indent=2)
                                except Exception: pass
                                
                            if not res or not res.get("success"):
                                print(f"[{self.tab_id}] [WARN] Programmatic setup failed for {symbol}: {res.get('error') if res else 'Unknown'}")
                            else:
                                print(f"[{self.tab_id}] [Config] Symbol & Indicators verified/configured for {symbol}")
                except Exception as e:
                    print(f"[{self.tab_id}] [WARN] Error configuring window {win_idx} for {symbol}: {e}")
            await asyncio.sleep(0.1)

        # Wait for studies to load data from network
        print(f"[{self.tab_id}] Waiting 15 seconds for TradingView studies to load historical data...")
        await asyncio.sleep(15.0)

        try:
            await self.page.screenshot(path=os.path.join(base_dir, "Seeding", f"{self.tab_id}_layout.png"))
        except Exception as e:
            print(f"[{self.tab_id}] [WARN] Screenshot failed: {e}")
        print(f"[{self.tab_id}] Setup & Indicator injection complete.")

    async def poll_loop(self) -> None:
        """Background data poller extracting DOM legend values & JS shims"""
        async def _fetch_frame(win_idx: int) -> bool:
            sym = self.symbols[win_idx - 1]
            container_id = f"tv_chart_container_win{win_idx}"
            selector = f"#{container_id}" if win_idx != 1 else f"#{container_id}, #tv_chart_container_main"
            container = self.page.locator(selector).first
            
            if await container.count() > 0:
                iframe = container.locator("iframe").first
                if await iframe.count() > 0:
                    iframe_handle = await iframe.element_handle(timeout=10000)
                    frame = await iframe_handle.content_frame() if iframe_handle else None
                    if frame:
                        try:
                            res = await frame.evaluate(SINGLE_FRAME_EXTRACTION_JS)
                        except Exception as eval_exc:
                            print(f"[{self.tab_id}] [POLL ERROR] {sym} frame eval: {eval_exc}")
                            return False
                        if res and res.get("success"):
                            d = res["data"]
                            sym_actual = (d.get("symbol") or "").strip().upper()
                            # Check actual extracted symbol matches what we expect
                            if sym_actual and sym_actual in self.symbols:
                                target_sym = sym_actual
                            else:
                                target_sym = sym
                                
                            price_val = parse_float(d.get("close") or d.get("price") or 0.0)
                            rsi_val = parse_float(d.get("rsi", 0.0))
                            if rsi_val in (100.0, 0.0):
                                rsi_val = self.store._data.get(target_sym, AssetSnapshot(symbol=target_sym)).rsi
                            await self.store.update(
                                target_sym,
                                source="coinglass",
                                price=price_val,
                                volume=parse_float(d.get("volume", 0.0)),
                                rsi=rsi_val,
                                fut_cvd=parse_float(d.get("futures_cvd", 0.0)),
                                spot_cvd=parse_float(d.get("spot_cvd", 0.0)),
                                funding=parse_float(d.get("funding_rate", 0.0)),
                                liq_long=parse_float(d.get("liquidations_long", 0.0)),
                                liq_short=parse_float(d.get("liquidations_short", 0.0)),
                                ls_ratio=parse_float(d.get("ls_ratio", 0.0)),
                                oi=parse_float(d.get("open_interest", 0.0)),
                                coins_bid=parse_float(d.get("coins_bid", 0.0)),
                                coins_ask=parse_float(d.get("coins_ask", 0.0)),
                                dollars_bid=parse_float(d.get("dollars_bid", 0.0)),
                                dollars_ask=parse_float(d.get("dollars_ask", 0.0)),
                                whale_idx=parse_float(d.get("whale_index", 0.0)),
                                tk_buy_cnt=parse_float(d.get("taker_buy_count", 0.0)),
                                tk_sell_cnt=parse_float(d.get("taker_sell_count", 0.0))
                            )
                            return True
            return False

        while self.running:
            try:
                results = await asyncio.gather(*[_fetch_frame(i) for i in range(1, 10)], return_exceptions=True)
                has_success = False
                for r in results:
                    if isinstance(r, Exception):
                        print(f"[{self.tab_id}] [POLL ERROR] Subtask failed: {r}")
                        self.poll_failures += 1
                    elif r is True:
                        has_success = True

                if has_success:
                    self.last_heartbeat_ns = time.time_ns()
                    self.poll_failures = 0
                else:
                    self.poll_failures += 1
            except Exception as e:
                print(f"[{self.tab_id}] [POLL ERROR] Outer: {e}")
                self.poll_failures += 1
            
            if self.poll_failures > 5:
                print(f"[{self.tab_id}] [WATCHDOG] Max failures exceeded ({self.poll_failures}). Auto-healing by reloading page...")
                try:
                    await self.page.reload(wait_until="load", timeout=30000)
                    self.poll_failures = 0
                except Exception as ex:
                    print(f"[{self.tab_id}] [WATCHDOG] Failed to reload page: {ex}")
                    self.poll_failures = 0

            await asyncio.sleep(0.5)

    async def _route_payload(self, entry: dict) -> None:
        url = entry.get("url", "")
        body = entry.get("body", "")
        try:
            payload = json.loads(body)
        except Exception:
            return
        
        # Route to appropriate update target
        if "open-interest" in url:
            await self._apply(payload, "oi")
        elif "funding-rate" in url:
            await self._apply(payload, "funding")
        elif "liquidation" in url:
            await self._apply_liq(payload)
        elif "long-short" in url:
            await self._apply(payload, "ls_ratio")
        elif "cumulative-volume" in url:
            if "futures" in url:
                await self._apply(payload, "fut_cvd")
            else:
                await self._apply(payload, "spot_cvd")
        elif "rsi" in url:
            await self._apply(payload, "rsi")

    async def _apply(self, payload: Any, field_name: str) -> None:
        data = payload.get("data", [])
        if isinstance(data, list):
            for row in data:
                sym = row.get("symbol")
                if sym in self.symbols:
                    val = parse_float(row.get("value", 0.0))
                    await self.store.update(sym, source="coinglass", **{field_name: val})
        elif isinstance(data, dict):
            for sym, val in data.items():
                if sym in self.symbols:
                    await self.store.update(sym, source="coinglass", **{field_name: parse_float(val)})

    async def _apply_liq(self, payload: Any) -> None:
        data = payload.get("data", [])
        if isinstance(data, list):
            for row in data:
                sym = row.get("symbol")
                if sym in self.symbols:
                    long_liq = parse_float(row.get("longLiq", 0.0))
                    short_liq = parse_float(row.get("shortLiq", 0.0))
                    await self.store.update(sym, source="coinglass", liq_long=long_liq, liq_short=short_liq)

    async def seed_symbol(self, symbol: str, excel_executor, focus_lock: asyncio.Lock) -> None:
        """Performs visual backward walk to collect 50 candles and export to Excel"""
        self.is_seeding = True
        win_idx = self.symbols.index(symbol) + 1
        container_id = f"tv_chart_container_win{win_idx}"
        selector = f"#{container_id}" if win_idx != 1 else f"#{container_id}, #tv_chart_container_main"
        container = self.page.locator(selector).first
        
        async with focus_lock:
            print(f"[{self.tab_id}] Seeding {symbol} in Window {win_idx}. Acquired focus lock. Bringing tab to front...")
            await self.page.bring_to_front()
            await asyncio.sleep(0.5)
            
            iframe = container.locator("iframe").first
            try:
                await iframe.wait_for(state="attached", timeout=15000)
            except Exception:
                pass
            iframe_handle = await iframe.element_handle(timeout=10000)
            if not iframe_handle:
                print(f"[{self.tab_id}] [ERROR] No iframe handle for seeding {symbol}")
                return
            frame = await iframe_handle.content_frame()
            if not frame:
                print(f"[{self.tab_id}] [ERROR] Content frame missing for seeding {symbol}")
                return
                
            # Resolve the first canvas inside the frame
            canvas = frame.locator("canvas").first
            try:
                await canvas.wait_for(state="visible", timeout=5000)
            except Exception:
                print(f"[{self.tab_id}] [ERROR] Canvas element not visible for {symbol}")
                return
                
            # Click canvas center to focus TradingView inner context
            await canvas.click(force=True, timeout=5000)
            await asyncio.sleep(0.3)
            
            # Explicitly focus the window/document body
            await frame.evaluate("() => { window.focus(); if (document.body) document.body.focus(); }")
            await asyncio.sleep(0.2)
            
            # Press Escape to close any potential dialogs
            await self.page.keyboard.press("Escape")
            await asyncio.sleep(0.2)
            
            # Reset visual using Alt+r
            await self.page.keyboard.press("Alt+r")
            await asyncio.sleep(1.0)
            
            # Wait for canvas to become visible/attached again after chart reset
            try:
                await canvas.wait_for(state="visible", timeout=5000)
            except Exception:
                pass
            
            # Right-click canvas to open context menu (forces browser focus delegation)
            await canvas.click(button="right", force=True, timeout=5000)
            await asyncio.sleep(0.5)
            await self.page.keyboard.press("Escape")
            await asyncio.sleep(0.2)
            
            # Wait up to 10 seconds for indicators to load historical data from network
            print(f"[{self.tab_id}] waiting for indicators to populate historical data for {symbol}...")
            for attempt in range(20):
                res = await frame.evaluate(SINGLE_FRAME_EXTRACTION_JS)
                if res and res.get("success"):
                    d = res["data"]
                    if (d.get("volume") not in ("N/A", "0", None) and
                        d.get("rsi") not in ("N/A", "100.00", None) and
                        d.get("futures_cvd") != "N/A" and
                        d.get("spot_cvd") != "N/A" and
                        d.get("open_interest") != "N/A"):
                        print(f"[{self.tab_id}] Indicators populated in {attempt * 0.5:.1f}s")
                        break
                await asyncio.sleep(0.5)

            rect = await canvas.bounding_box()
            if not rect:
                print(f"[{self.tab_id}] [ERROR] Cannot get canvas bounding box for {symbol}")
                return

            x_pos = rect["x"] + rect["width"] - 60
            y_pos = rect["y"] + rect["height"] * 0.5

            # Hover and click to focus on the rightmost section of the canvas
            await self.page.mouse.move(x_pos, y_pos)
            await self.page.mouse.click(x_pos, y_pos)
            await asyncio.sleep(0.2)

            # ArrowLeft snaps crosshair to the latest candle
            await self.page.keyboard.press("ArrowLeft")
            await asyncio.sleep(0.3)

            # --- Dynamic Gap Calculation ---
            target_steps = 850
            existing_rows = []
            base_dir = os.path.dirname(os.path.abspath(__file__))
            combined_path = os.path.join(base_dir, "Seeding", "combined_seed_history.xlsx")
            if os.path.exists(combined_path):
                import pandas as pd
                try:
                    df = pd.read_excel(combined_path, sheet_name=symbol)
                    if not df.empty and "open_time" in df.columns:
                        existing_rows = df.to_dict('records')
                        
                        # Handle potential datetime vs int vs str timestamp differences
                        for r in existing_rows:
                            val = r.get("open_time")
                            if hasattr(val, "timestamp"):
                                from datetime import timezone
                                r["open_time"] = int(val.replace(tzinfo=timezone.utc).timestamp())
                            elif isinstance(val, (int, float)):
                                r["open_time"] = int(val)
                            elif isinstance(val, str):
                                try:
                                    val_clean = val.replace(" IST", "").strip()
                                    dt = pd.to_datetime(val_clean)
                                    from datetime import timedelta
                                    dt_utc = dt - timedelta(hours=5, minutes=30)
                                    r["open_time"] = int(dt_utc.timestamp())
                                except Exception:
                                    try:
                                        r["open_time"] = int(float(val))
                                    except Exception:
                                        pass
                        
                        # Filter out invalid open_times for max calc
                        valid_times = [r["open_time"] for r in existing_rows if isinstance(r.get("open_time"), int)]
                        if valid_times:
                            latest_time = max(valid_times)
                            current_time = int((time.time() // 900) * 900)
                            gap_candles = calculate_commodity_gap(symbol, latest_time, current_time)
                            existing_count = len(existing_rows)
                            
                            if existing_count + gap_candles >= 850:
                                target_steps = min(gap_candles + 2, 850)
                            else:
                                target_steps = 850
                                
                            print(f"\n==================================================")
                            print(f"[{self.tab_id}] {symbol} SEEDING DATABASE CHECK:")
                            print(f"[{self.tab_id}] Database has {existing_count} candles.")
                            print(f"[{self.tab_id}] Gap from offline time: {gap_candles} missing candles (calendar adjusted).")
                            print(f"[{self.tab_id}] -> WILL SCRAPE {target_steps} CANDLES NOW TO WARM UP.")
                            print(f"==================================================\n")
                        else:
                            print(f"\n==================================================")
                            print(f"[{self.tab_id}] {symbol} Found Excel sheet but no valid `open_time` ints parsed.")
                            print(f"[{self.tab_id}] -> WILL SCRAPE {target_steps} CANDLES NOW TO WARM UP.")
                            print(f"==================================================\n")
                except Exception as e:
                    print(f"[{self.tab_id}] [WARN] Could not read existing seed for {symbol}: {e}")
            else:
                print(f"\n==================================================")
                print(f"[{self.tab_id}] {symbol} No existing seed history found in Excel.")
                print(f"[{self.tab_id}] -> WILL SCRAPE {target_steps} CANDLES NOW TO WARM UP.")
                print(f"==================================================\n")
            # -------------------------------

            candles = collections.deque(maxlen=1000)
            stalls = 0
            debug_dicts = []
            
            last_key = None
            is_crypto = symbol not in ["XAUUSDT", "XAGUSDT", "CLUSDT", "NATGASUSDT"]
            
            computed_timestamps = get_historical_timestamps(symbol, int((time.time() // 900) * 900), target_steps)

            for step in range(target_steps * 2):
                if len(candles) >= target_steps:
                    print(f"[{self.tab_id}] {symbol} Reached target {target_steps} candles. Stopping walk.")
                    break
                    
                if step % 20 == 0:
                    print(f"[{self.tab_id}] Seeding {symbol}: candle {len(candles)}/{target_steps}...")
                
                # 1. Wait for DOM to update after moving crosshair (avoid duplicate read stalls)
                d = None
                for attempt in range(6):
                    res = await frame.evaluate(SINGLE_FRAME_EXTRACTION_JS)
                    if res and res.get("success"):
                        temp_d = res["data"]
                        close = parse_float(temp_d.get("close", temp_d.get("price", 0.0)))
                        volume = parse_float(temp_d.get("volume", 0.0))
                        rsi = parse_float(temp_d.get("rsi", 50.0))
                        val_key = (close, volume, rsi)
                        
                        if val_key != last_key:
                            d = temp_d
                            last_key = val_key
                            break
                    await asyncio.sleep(0.04)

                if d is None:
                    stalls += 1
                    if stalls > 4:
                        print(f"[{self.tab_id}] [WARN] Seeding stalled for {symbol} at step {step}. Ending early.")
                        break
                    # Recover visual focus delegation
                    await canvas.focus()
                    await asyncio.sleep(0.05)
                    await self.page.keyboard.press("ArrowLeft")
                    await asyncio.sleep(0.1)
                    continue
                
                stalls = 0
                
                # 2. Wait up to 600ms for lazy-loaded indicators (CVD, OI) to populate if they are currently N/A
                if is_crypto:
                    for load_attempt in range(4):
                        if (d.get("futures_cvd") != "N/A" and 
                            d.get("spot_cvd") != "N/A" and 
                            d.get("open_interest") != "N/A"):
                            break
                        await asyncio.sleep(0.15)
                        res = await frame.evaluate(SINGLE_FRAME_EXTRACTION_JS)
                        if res and res.get("success"):
                            d = res["data"]
                            
                if symbol == "BTCUSDT":
                    debug_dicts.append({
                        "step": step,
                        "data": d,
                        "rawLegends": res.get("rawLegends", []) if res else []
                    })
                    
                candle_data = {
                    "open_time": computed_timestamps[len(candles)],
                    "open":       parse_float(d.get("open",   0.0)),
                    "high":       parse_float(d.get("high",   0.0)),
                    "low":        parse_float(d.get("low",    0.0)),
                    "close":      parse_float(d.get("close",  d.get("price", 0.0))),
                    "volume":     parse_float(d.get("volume", 0.0)),
                    "rsi":        parse_float(d.get("rsi",    50.0)),
                    "fut_cvd":    parse_float(d.get("futures_cvd",      0.0)),
                    "spot_cvd":   parse_float(d.get("spot_cvd",         0.0)),
                    "funding":    parse_float(d.get("funding_rate",      0.0)),
                    "liq_long":   abs(parse_float(d.get("liquidations_long",  0.0))),
                    "liq_short":  abs(parse_float(d.get("liquidations_short", 0.0))),
                    "ls_ratio":   parse_float(d.get("ls_ratio",           1.0)),
                    "oi":         parse_float(d.get("open_interest",      0.0)),
                    "coins_bid":  parse_float(d.get("coins_bid", 0.0)),
                    "coins_ask":  parse_float(d.get("coins_ask", 0.0)),
                    "dollars_bid": parse_float(d.get("dollars_bid", 0.0)),
                    "dollars_ask": parse_float(d.get("dollars_ask", 0.0)),
                    "whale_idx":  parse_float(d.get("whale_index", 0.0)),
                    "tk_buy_cnt": parse_float(d.get("taker_buy_count", 0.0)),
                    "tk_sell_cnt": parse_float(d.get("taker_sell_count", 0.0)),
                }
                
                candles.appendleft(candle_data)
                
                # Step left — move crosshair one candle back
                await self.page.keyboard.press("ArrowLeft")
                await asyncio.sleep(0.08)

            # Restore view
            await self.page.keyboard.press("Alt+r")
            await asyncio.sleep(0.5)

            scraped_list = list(candles)
            final_list = scraped_list
            if existing_rows:
                all_data = existing_rows + scraped_list
                # Deduplicate by open_time, keeping newest (scraped over existing due to order)
                dedup = {r["open_time"]: r for r in all_data if isinstance(r.get("open_time"), int)}
                sorted_vals = sorted(dedup.values(), key=lambda x: x["open_time"])
                final_list = sorted_vals

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(excel_executor, _dump_xlsx, symbol, final_list)
            
            if self.store.predictor:
                self.store.predictor.set_history(symbol, final_list)
            
            if candles:
                last = list(candles)[-1]
                missing = [k for k, v in last.items() if v == 0.0 and k not in ("liq_long", "liq_short")]
                if missing:
                    print(f"[{self.tab_id}] [WARN] {symbol}: zero fields = {missing}")
                else:
                    print(f"[{self.tab_id}] [OK]   {symbol}: all fields populated (close={last['close']}, vol={last['volume']}, funding={last['funding']})")
                    
            if symbol == "BTCUSDT":
                try:
                    with open(os.path.join(base_dir, "Seeding", "seeding_debug_BTCUSDT.json"), "w", encoding="utf-8") as f:
                        json.dump(debug_dicts, f, indent=2)
                    await self.page.screenshot(path=os.path.join(base_dir, "Seeding", f"diag_{self.tab_id}_{symbol}.png"), clip={"x": 0, "y": 0, "width": 600, "height": 400})
                except Exception:
                    pass
            print(f"[{self.tab_id}] [Success] Seeded {symbol} with {len(candles)} candles.")

def fetch_binance_funding_rates(symbol: str) -> List[Dict[str, Any]]:
    import urllib.request
    import json
    url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=100"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"[Binance API] Failed to fetch funding rate for {symbol}: {e}")
        return []

def fetch_binance_open_interest(symbol: str) -> List[Dict[str, Any]]:
    import urllib.request
    import json
    url = f"https://fapi.binance.com/fapi/v1/openInterestHist?symbol={symbol}&period=15m&limit=120"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"[Binance API] Failed to fetch open interest for {symbol}: {e}")
        return []

def fetch_binance_ls_ratio(symbol: str) -> List[Dict[str, Any]]:
    import urllib.request
    import json
    url = f"https://fapi.binance.com/data/globalLongShortAccountRatio?symbol={symbol}&period=15m&limit=120"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"[Binance API] Failed to fetch long/short ratio for {symbol}: {e}")
        return []

def _dump_xlsx(symbol: str, rows: List[Dict[str, Any]]) -> None:
    crypto_symbols = {
        "BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", 
        "DOGEUSDT", "ADAUSDT", "TRXUSDT", "LINKUSDT", "AVAXUSDT", 
        "SUIUSDT", "NEARUSDT", "DOTUSDT", "LTCUSDT"
    }
    
    if rows and symbol in crypto_symbols:
        # 1. Backfill funding rate if all zeros
        if all(r.get("funding", 0.0) == 0.0 for r in rows):
            api_rates = fetch_binance_funding_rates(symbol)
            if api_rates:
                api_rates.sort(key=lambda x: x["fundingTime"])
                for r in rows:
                    row_time_ms = r.get("open_time", 0) * 1000
                    matching_rate = 0.0
                    for item in api_rates:
                        if item["fundingTime"] <= row_time_ms:
                            matching_rate = float(item["fundingRate"])
                        else:
                            break
                    r["funding"] = matching_rate

        # 2. Backfill open interest if all zeros
        if all(r.get("oi", 0.0) == 0.0 for r in rows):
            api_oi = fetch_binance_open_interest(symbol)
            if api_oi:
                api_oi.sort(key=lambda x: x["timestamp"])
                for r in rows:
                    row_time_ms = r.get("open_time", 0) * 1000
                    matching_oi = 0.0
                    for item in api_oi:
                        if item["timestamp"] <= row_time_ms:
                            matching_oi = float(item["sumOpenInterest"])
                        else:
                            break
                    r["oi"] = matching_oi

        # 3. Backfill long/short ratio if all zeros/default
        if all(r.get("ls_ratio", 1.0) == 1.0 or r.get("ls_ratio", 1.0) == 0.0 for r in rows):
            api_ls = fetch_binance_ls_ratio(symbol)
            if api_ls:
                api_ls.sort(key=lambda x: x["timestamp"])
                for r in rows:
                    row_time_ms = r.get("open_time", 0) * 1000
                    matching_ls = 1.0
                    for item in api_ls:
                        if item["timestamp"] <= row_time_ms:
                            matching_ls = float(item["longShortRatio"])
                        else:
                            break
                    r["ls_ratio"] = matching_ls

    # 4. Apply general forward-fill and backward-fill for all numeric columns to handle scattered zeros
    if rows:
        fill_fields = [
            "open", "high", "low", "close", "volume", "rsi", "fut_cvd", "spot_cvd", "funding", "ls_ratio", "oi",
            "coins_bid", "coins_ask", "dollars_bid", "dollars_ask", "whale_idx", "tk_buy_cnt", "tk_sell_cnt"
        ]
        for field in fill_fields:
            non_zero_vals = [r.get(field, 0.0) for r in rows if r.get(field, 0.0) != 0.0]
            if non_zero_vals:
                # Forward fill
                last_val = non_zero_vals[0]
                for r in rows:
                    val = r.get(field, 0.0)
                    if val != 0.0:
                        last_val = val
                    else:
                        r[field] = last_val
                # Backward fill
                last_val = non_zero_vals[-1]
                for r in reversed(rows):
                    val = r.get(field, 0.0)
                    if val != 0.0:
                        last_val = val
                    else:
                        r[field] = last_val

    wb = Workbook()
    ws = wb.active
    ws.title = symbol[:31]
    
    headers = [
        "open_time", "open", "high", "low", "close", "volume", 
        "rsi", "fut_cvd", "spot_cvd", "funding", "liq_long", "liq_short", "ls_ratio", "oi",
        "coins_bid", "coins_ask", "dollars_bid", "dollars_ask", "whale_idx", "tk_buy_cnt", "tk_sell_cnt"
    ]
    
    HDR_FILL = PatternFill("solid", fgColor="1F3864")
    HDR_FONT = Font(bold=True, color="FFFFFF", name="Calibri", size=11)
    CENTER = Alignment(horizontal="center", vertical="center")
    thin = Side(style="thin", color="D9D9D9")
    BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)
    
    ws.append(headers)
    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = HDR_FILL
        cell.font = HDR_FONT
        cell.alignment = CENTER
        cell.border = BORDER
        
    for row_idx, r in enumerate(rows, start=2):
        row_vals = []
        for h in headers:
            val = r.get(h, "")
            if h == "open_time" and isinstance(val, (int, float)):
                from datetime import datetime, timezone, timedelta
                tz_ist = timezone(timedelta(hours=5, minutes=30))
                val = datetime.fromtimestamp(val, tz=tz_ist).strftime("%Y-%m-%d %H:%M:%S IST")
            row_vals.append(val)
        ws.append(row_vals)
        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.alignment = CENTER
            cell.border = BORDER
            
    for col in ws.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max(max_len + 3, 12)
        
    os.makedirs(os.path.join(base_dir, "Seeding"), exist_ok=True)
    filename = os.path.join(base_dir, "Seeding", f"{symbol}_seed_history.xlsx")
    try:
        wb.save(filename)
    except PermissionError:
        import random
        alt_filename = os.path.join(base_dir, "Seeding", f"{symbol}_seed_history_{random.randint(1000, 9999)}.xlsx")
        print(f"[WARN] Permission denied on {filename} (probably open in Excel). Saving to {alt_filename} instead.")
        try:
            wb.save(alt_filename)
        except Exception as e:
            print(f"[ERROR] Failed to save fallback Excel for {symbol}: {e}")

# --- DASHBOARD RENDERER ---
def render_table(snap: Dict[str, AssetSnapshot], trade_tracker: Any = None) -> Any:
    t = Table(title="Coinglass + Binance Footprint Scraper Terminal", expand=True)
    cols = (
        "Symbol", "Price", "RSI", "FutCVD", "SpotCVD", "LiqL", "LiqS", "Fund", "LSR", "OI", 
        "CoinsB", "CoinsA", "USDB", "USDA", "Whale", "BuyC", "SellC", "FP_D", "FP_P", "ARM"
    )
    for col in cols:
        t.add_column(col, justify="center", no_wrap=True)
        
    now = time.time_ns()
    
    def fmt(v: float, fresh: bool, is_pct: bool = False) -> str:
        if v == 0.0 or v is None:
            return "[dim]--[/dim]"
        s = f"{v:.2f}%" if is_pct else f"{v:,.2f}"
        if abs(v) > 1e6:
            s = f"{v:,.0f}"
        return s if fresh else f"[red]{s}[/red]"

    for sym in ALL_SYMBOLS:
        a = snap.get(sym, AssetSnapshot(symbol=sym))
        fresh = (now - a.ts_ns) < STALE_NS
        
        t.add_row(
            sym,
            fmt(a.price, fresh),
            fmt(a.rsi, fresh),
            fmt(a.fut_cvd, fresh),
            fmt(a.spot_cvd, fresh),
            fmt(a.liq_long, fresh),
            fmt(a.liq_short, fresh),
            fmt(a.funding, fresh, is_pct=True),
            fmt(a.ls_ratio, fresh),
            fmt(a.oi, fresh),
            fmt(a.coins_bid, fresh),
            fmt(a.coins_ask, fresh),
            fmt(a.dollars_bid, fresh),
            fmt(a.dollars_ask, fresh),
            fmt(a.whale_idx, fresh),
            fmt(a.tk_buy_cnt, fresh),
            fmt(a.tk_sell_cnt, fresh),
            fmt(a.fp_delta, fresh),
            fmt(a.fp_poc, fresh),
            f"[green]{a.strategy_armed}[/green]" if a.strategy_armed else "[dim]--[/dim]"
        )

    if trade_tracker is None:
        return t

    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    stats = trade_tracker.get_stats()
    
    # Active trades section
    active_lines = []
    with trade_tracker.lock:
        active_snap = list(trade_tracker.active_trades.values())
        history_snap = list(trade_tracker.history[-3:])

    for tr in active_snap:
        dir_str = "[bold green]LONG[/]" if tr['direction'] == 1 else "[bold red]SHORT[/]"
        pnl_usd = tr.get('live_pnl_usd', 0.0)
        pnl_pct = tr.get('live_pnl_pct', 0.0)
        pnl_str = f"[bold green]+${pnl_usd:.2f} (+{pnl_pct:+.2f}%)[/]" if pnl_usd >= 0 else f"[bold red]-${abs(pnl_usd):.2f} ({pnl_pct:+.2f}%)[/]"
        mt5_info = f" | MT5 Entry: {tr['mt5_entry']:.4f} (Lot: {tr['mt5_lot']:.2f})" if 'mt5_entry' in tr else ""
        active_lines.append(f"{tr['symbol']} | {dir_str} | Entry: {tr['entry_price']:.4f} | SL: {tr['sl']:.4f} | TP: {tr['tp']:.4f} | Live PnL: {pnl_str}{mt5_info}")

    active_text = "\n".join(active_lines) if active_lines else "[dim]No active trades[/dim]"

    # History trades section (last 3)
    history_lines = []
    for tr in history_snap:
        dir_str = "[bold green]LONG[/]" if tr['direction'] == 1 else "[bold red]SHORT[/]"
        pnl_usd = tr.get('pnl_usd', 0.0)
        pnl_pct = tr.get('pnl_pct', 0.0)
        pnl_str = f"[bold green]+${pnl_usd:.2f} (+{pnl_pct:+.2f}%)[/]" if pnl_usd >= 0 else f"[bold red]-${abs(pnl_usd):.2f} ({pnl_pct:+.2f}%)[/]"
        reason = tr.get('exit_reason', 'EXIT')
        history_lines.append(f"{tr['symbol']} | {dir_str} | Exit: {tr['exit_price']:.4f} | Reason: {reason} | Final: {pnl_str}")

    history_text = "\n".join(history_lines) if history_lines else "[dim]No trade history[/dim]"

    # Stats string
    winrate = stats['winrate']
    total_pnl = stats['total_pnl_usd']
    pnl_pct = total_pnl / trade_tracker.initial_capital * 100.0 if trade_tracker.initial_capital > 0 else 0.0
    pnl_clr = "green" if total_pnl >= 0 else "red"
    pnl_sign = "+" if total_pnl >= 0 else ""

    stats_text = (
        f"Initial Capital: [bold]${trade_tracker.initial_capital:,.2f}[/]  |  Current Capital: [bold]${stats['current_capital']:.2f}[/]  |  "
        f"Total PnL: [bold {pnl_clr}]{pnl_sign}${total_pnl:.2f} ({pnl_pct:+.2f}%)[/]  |  "
        f"Trades: [bold]{stats['total']}[/]  |  Winrate: [bold]{winrate:.1f}%[/]"
    )

    trade_table = Table(show_header=True, header_style="bold bright_magenta", border_style="magenta", expand=True)
    trade_table.add_column("Active Trades", justify="left", ratio=1)
    trade_table.add_column(stats_text, justify="left", ratio=1)
    trade_table.add_row(active_text, history_text)

    return Group(t, trade_table)

async def renderer_loop(store: SnapshotStore, stop: asyncio.Event) -> None:
    console = Console()
    loop_cnt = 0
    with Live(render_table(store.snapshot(), store.trade_tracker), console=console, refresh_per_second=REFRESH_HZ, screen=True) as live:
        while not stop.is_set():
            snap = store.snapshot()
            live.update(render_table(snap, store.trade_tracker))
            
            loop_cnt += 1
            if loop_cnt % 20 == 0:  # Every 10 seconds at 2Hz REFRESH_HZ
                try:
                    serializable_snap = {}
                    for sym, a in snap.items():
                        serializable_snap[sym] = {
                            "price": a.price, "volume": a.volume, "rsi": a.rsi, "fut_cvd": a.fut_cvd, "spot_cvd": a.spot_cvd,
                            "liq_long": a.liq_long, "liq_short": a.liq_short, "funding": a.funding,
                            "ls_ratio": a.ls_ratio, "oi": a.oi,
                            "coins_bid": a.coins_bid, "coins_ask": a.coins_ask,
                            "dollars_bid": a.dollars_bid, "dollars_ask": a.dollars_ask,
                            "whale_idx": a.whale_idx, "tk_buy_cnt": a.tk_buy_cnt, "tk_sell_cnt": a.tk_sell_cnt,
                            "fp_delta": a.fp_delta, "fp_poc": a.fp_poc,
                            "strategy_armed": a.strategy_armed, "ts_ns": a.ts_ns
                        }
                    def _write_debug():
                        try:
                            tmp_path = os.path.join(base_dir, "Seeding", "snapshot_debug.json.tmp")
                            with open(tmp_path, "w", encoding="utf-8") as f:
                                json.dump(serializable_snap, f, indent=4)
                            os.replace(tmp_path, os.path.join(base_dir, "Seeding", "snapshot_debug.json"))
                        except Exception:
                            pass
                    await asyncio.to_thread(_write_debug)
                except Exception:
                    pass
            await asyncio.sleep(1.0 / REFRESH_HZ)

# --- WATCHDOG ---
async def watchdog(components: List[Any], focus_lock: asyncio.Lock, stop: asyncio.Event) -> None:
    # Initialize/reset heartbeats for all components on startup to ignore the configuration time
    now_start = time.time_ns()
    for c in components:
        if hasattr(c, 'last_heartbeat_ns'):
            c.last_heartbeat_ns = now_start

    tab_tasks = {}
    for c in components:
        if isinstance(c, CoinglassTab):
            tab_tasks[c] = asyncio.create_task(c.poll_loop())
            
    try:
        while not stop.is_set():
            now = time.time_ns()
            for c in components:
                if hasattr(c, 'last_heartbeat_ns') and now - c.last_heartbeat_ns > 90_000_000_000:
                    if getattr(c, 'skip_watchdog', False):
                        continue
                    print(f"[Watchdog] [WARN] Subsystem '{c.__class__.__name__}' ({getattr(c, 'tab_id', 'Unknown')}) hung. Heartbeat silent >90s.")
                    if isinstance(c, CoinglassTab):
                        print(f"[Watchdog] [RECOVERY] Attempting recovery for '{c.tab_id}'...")
                        if c in tab_tasks and not tab_tasks[c].done():
                            tab_tasks[c].cancel()
                            try:
                                await tab_tasks[c]
                            except asyncio.CancelledError:
                                pass
                        try:
                            await c.reconnect(focus_lock)
                            tab_tasks[c] = asyncio.create_task(c.poll_loop())
                            # Reset heartbeats for all components to prevent false positives from the blocking recovery
                            now_after = time.time_ns()
                            for comp in components:
                                if hasattr(comp, 'last_heartbeat_ns'):
                                    comp.last_heartbeat_ns = now_after
                        except Exception as rec_err:
                            print(f"[Watchdog] [ERROR] Recovery failed for '{c.tab_id}': {rec_err}")
            # Check Python process memory usage to catch memory leaks
            mem_mb = get_process_memory_usage() / (1024 * 1024)
            if mem_mb > 3584.0:  # 3.5 GB limit to allow initial retraining/seeding spikes
                print(f"\n[Watchdog] [ALERT] [MEMORY] Python memory usage is extremely high ({mem_mb:.1f} MB)!")
            await asyncio.sleep(5.0)
    finally:
        for task in tab_tasks.values():
            if not task.done():
                task.cancel()
        if tab_tasks:
            await asyncio.gather(*tab_tasks.values(), return_exceptions=True)

def combine_seeding_files() -> None:
    import glob
    import copy
    from openpyxl import load_workbook, Workbook
    from openpyxl.utils import get_column_letter

    files = glob.glob(os.path.join(base_dir, "Seeding", "*_seed_history.xlsx"))
    files = [f for f in files if "combined_seed" not in os.path.basename(f).lower()]
    if not files:
        print("[Setup] No seeding files found to combine.")
        return

    print(f"[Setup] Combining {len(files)} seeding files into a single workbook...")
    combined_wb = Workbook()
    default_sheet = combined_wb.active
    combined_wb.remove(default_sheet)

    for f in sorted(files):
        symbol = os.path.basename(f).replace("_seed_history.xlsx", "")
        try:
            wb = load_workbook(f)
            source_ws = wb.active
            target_ws = combined_wb.create_sheet(title=symbol[:31])

            for row in source_ws.iter_rows():
                for cell in row:
                    new_cell = target_ws.cell(row=cell.row, column=cell.column, value=cell.value)
                    if cell.has_style:
                        new_cell.font = copy.copy(cell.font)
                        new_cell.fill = copy.copy(cell.fill)
                        new_cell.border = copy.copy(cell.border)
                        new_cell.alignment = copy.copy(cell.alignment)
                        new_cell.number_format = cell.number_format

            for col in source_ws.columns:
                col_letter = get_column_letter(col[0].column)
                target_ws.column_dimensions[col_letter].width = source_ws.column_dimensions[col_letter].width
            wb.close()
        except Exception as copy_exc:
            print(f"[Setup] [WARN] Failed to copy {symbol} sheet: {copy_exc}")

    combined_filename = os.path.join(base_dir, "Seeding", "combined_seed_history.xlsx")
    tmp_filename = combined_filename + ".tmp"
    try:
        combined_wb.save(tmp_filename)
        os.replace(tmp_filename, combined_filename)
        print(f"[Setup] Combined workbook saved successfully: {combined_filename}")
        
        # Clean up individual seed files
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass
        print("[Setup] Cleaned up individual symbol seeding files.")
    except Exception as e:
        print(f"[Setup] [WARN] Failed to save combined workbook: {e}")

# --- MAIN CONTROLLER ---
async def main(skip_seed: bool = False) -> None:
    print("=" * 60)
    print(f"  SYSTEM STARTUP - MODE: {EXECUTION_MODE}")
    print("  WARNING: NO REAL METATRADER 5 TRADE ORDERS WILL BE SENT")
    print("  TRADES ARE SIMULATED LOCALLY IN THE TRACKER FILE")
    print("=" * 60)

    # 0. Clear existing ML models to prevent conflicts before retraining
    print("[Setup] Clearing existing ML model files before retraining...")
    for sub in (ACTIVE_STRATEGY, 'Liquidation', 'ml_trend_pull'):
        m_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), sub, 'models')
        if os.path.exists(m_dir):
            for file in os.listdir(m_dir):
                if file.endswith('.pkl'):
                    try:
                        os.remove(os.path.join(m_dir, file))
                    except Exception as clear_err:
                        print(f"[Setup] [WARN] Could not remove old model file {file}: {clear_err}")

    # 0. Live Model Retraining on latest Parquet data
    print(f"[Setup] Running Live Model Retraining on latest Parquet data for {ACTIVE_STRATEGY}...")
    try:
        as_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ACTIVE_STRATEGY)
        if as_path not in sys.path:
            sys.path.insert(0, as_path)
        import importlib
        model_trainer_mod = importlib.import_module("model_trainer")
        # Ensure we reload the correct strategy module if it was previously loaded
        importlib.reload(model_trainer_mod)
        model_trainer_mod.train_models()
    except Exception as retrain_err:
        print(f"[Setup] [WARN] Failed to retrain {ACTIVE_STRATEGY} models: {retrain_err}")

    try:
        liq_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Liquidation')
        if liq_path not in sys.path:
            sys.path.append(liq_path)
        from train import train_ensemble
        print("[Setup] Retraining ML Liquidation models on latest data...")
        train_ensemble()
    except Exception as retrain_err:
        print(f"[Setup] [WARN] Failed to retrain ML Liquidation models: {retrain_err}")

    # Retrain ML_Trend_Pull models
    try:
        tp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_trend_pull')
        if tp_path not in sys.path:
            sys.path.insert(0, tp_path)
        import importlib
        sys.modules.pop('model_trainer', None)
        tp_trainer = importlib.import_module('model_trainer')
        importlib.reload(tp_trainer)
        print("[Setup] Retraining ML_Trend_Pull models on latest data...")
        tp_trainer.train_models()
    except Exception as retrain_err:
        print(f"[Setup] [WARN] Failed to retrain ML_Trend_Pull models: {retrain_err}")

    # Initialize LiveStrategyPredictor & load cached history
    predictor = LiveStrategyPredictor(ALL_SYMBOLS)
    predictor.load_history_from_disk()
    
    liquidation_predictor = LiveLiquidationPredictor(ALL_SYMBOLS)
    trend_pull_predictor = LiveTrendPullPredictor(ALL_SYMBOLS)

    # Warm up history from AlphaSqueezer's seeded disk data safely (Deepcopy)
    import copy
    for sym in ALL_SYMBOLS:
        if sym in predictor.candles_history:
            liquidation_predictor.candles_history[sym] = collections.deque(
                [copy.deepcopy(c) for c in predictor.candles_history[sym]], maxlen=1200
            )
            trend_pull_predictor.candles_history[sym] = collections.deque(
                [copy.deepcopy(c) for c in predictor.candles_history[sym]], maxlen=1200
            )
    print(f"[Setup] Warmed up ML Liquidation history deque with {len(liquidation_predictor.candles_history.get(ALL_SYMBOLS[0], []))} rows.")
    print(f"[Setup] Warmed up ML_Trend_Pull history deque with {len(trend_pull_predictor.candles_history.get(ALL_SYMBOLS[0], []))} rows.")

    trade_tracker = Engine1TradeTracker()
    liquidation_predictor.recent_capitals = [trade_tracker.current_capital]
    trade_tracker.on_close_callbacks.append(
        lambda strategy, capital: liquidation_predictor.record_closed_capital(capital)
        if strategy == "ML_Liquidation_Runner" else None
    )
    def run_retrain_proc():
        import sys
        import os
        import importlib
        base_dir = os.path.dirname(os.path.abspath(__file__))
        as_path = os.path.join(base_dir, ACTIVE_STRATEGY)
        liq_path = os.path.join(base_dir, 'Liquidation')
        if as_path not in sys.path:
            sys.path.insert(0, as_path)
        if liq_path not in sys.path:
            sys.path.append(liq_path)
            
        print(f"[Background Process] Starting Live Retraining for {ACTIVE_STRATEGY}...")
        try:
            model_trainer_mod = importlib.import_module("model_trainer")
            importlib.reload(model_trainer_mod)
            model_trainer_mod.train_models()
        except Exception as e:
            print(f"[Background Process] {ACTIVE_STRATEGY} retrain failed: {e}")
        try:
            from train import train_ensemble
            train_ensemble()
        except Exception as e:
            print(f"[Background Process] Liquidation retrain failed: {e}")
        try:
            tp_path = os.path.join(base_dir, 'ml_trend_pull')
            if tp_path not in sys.path:
                sys.path.insert(0, tp_path)
            sys.modules.pop('model_trainer', None)
            tp_trainer = importlib.import_module('model_trainer')
            importlib.reload(tp_trainer)
            tp_trainer.train_models()
        except Exception as e:
            print(f"[Background Process] ML_Trend_Pull retrain failed: {e}")
        print("[Background Process] Live Retraining finished.")

    def background_retrain_loop():
        import time
        import multiprocessing
        while True:
            # Sleep for 24 hours (86400 seconds)
            time.sleep(86400)
            print("[Background Thread] Launching 24hr Live Retraining Subprocess...")
            try:
                p = multiprocessing.Process(target=run_retrain_proc)
                p.start()
                p.join()
            except Exception as ex:
                print(f"[Background Thread] Subprocess retraining manager crashed: {ex}")

    import threading
    retrain_thread = threading.Thread(target=background_retrain_loop, daemon=True)
    retrain_thread.start()
    print("[Setup] Launched 24hr Background Retraining Manager Thread (Process-isolated).")

    store = SnapshotStore(ALL_SYMBOLS, predictor, liquidation_predictor, trade_tracker, trend_pull_predictor)
    stop = asyncio.Event()
    
    print("[Setup] Launching Chromium instance with persistent profile...")
    async with async_playwright() as pw:
        user_data_dir = os.path.join(os.getcwd(), "chrome_profile")
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir,
            headless=False,
            viewport={"width": 1920, "height": 1080},
            args=[
                "--disable-features=CalculateNativeWinOcclusion",
                "--disable-background-timer-throttling",
                "--start-maximized",
                "--remote-debugging-port=9222"
            ]
        )
        
        # 1. Performing Session Login first
        print("[Setup] Navigating to Coinglass Login...")
        login_page = await ctx.new_page()
        
        for attempt in range(3):
            try:
                await login_page.goto("https://www.coinglass.com/login", wait_until="load", timeout=45000)
                break
            except Exception as exc:
                print(f"[Setup] [WARN] Login navigation attempt {attempt+1} failed: {exc}")
                if attempt == 2:
                    raise exc
                await asyncio.sleep(5.0)
        await asyncio.sleep(5)
        
        os.makedirs(os.path.join(base_dir, "Seeding"), exist_ok=True)
        await login_page.screenshot(path=os.path.join(base_dir, "Seeding", "login_init.png"))
        
        email_input = login_page.locator("input[placeholder='Email']").first
        if await email_input.count() > 0:
            email = os.environ.get("COINGLASS_EMAIL")
            password = os.environ.get("COINGLASS_PASSWORD")
            if not email or not password:
                raise ValueError("Missing COINGLASS_EMAIL or COINGLASS_PASSWORD environment variables.")
            
            await email_input.click()
            await email_input.fill(email)
            await asyncio.sleep(0.3)

            pass_input = login_page.locator("input[placeholder='Password']").first
            await pass_input.click()
            await pass_input.fill(password)
            await asyncio.sleep(0.3)

            await login_page.screenshot(path=os.path.join(base_dir, "Seeding", "login_filled.png"))
            print("[Setup] Submitting login form...")

            # Try explicit button click first, fallback to JS click, last resort Enter key
            try:
                btn = login_page.locator("button:has-text('Login')").first
                if await btn.count() > 0:
                    await btn.wait_for(state="visible", timeout=5000)
                    await btn.click()
                else:
                    raise Exception("button not found via locator")
            except Exception:
                try:
                    await login_page.evaluate('''() => {
                        const b = Array.from(document.querySelectorAll('button'))
                            .find(el => el.textContent.trim() === 'Login');
                        if (b) b.click();
                    }''')
                except Exception:
                    # Most reliable: press Enter on password field
                    await pass_input.press("Enter")

            print("[Setup] Waiting for post-login redirect...")
            try:
                await login_page.wait_for_url(lambda url: "/login" not in url, timeout=20000)
                print("[Setup] Login successful — redirected away from /login.")
            except Exception:
                print("[Setup] [WARN] No redirect detected — may already be logged in or login failed.")
            await login_page.screenshot(path=os.path.join(base_dir, "Seeding", "login_after_submit.png"))
            print("[Setup] Waiting 5 seconds to ensure session cookies are fully persisted...")
            await asyncio.sleep(5.0)
        else:
            print("[Setup] Form inputs not detected, assuming session already active.")
        
        # 2. Open Scraping Tabs (while login page is open to hold session cache)
        tab1 = CoinglassTab(ctx, TAB1_SYMBOLS, store, "TAB_1")
        tab2 = CoinglassTab(ctx, TAB2_SYMBOLS, store, "TAB_2")
        binance = BinanceFootprintFeed(ALL_SYMBOLS, store)
        binance_ws = BinanceTradePriceWebSocketFeed(ALL_SYMBOLS, store)
        
        await asyncio.gather(tab1.start(), tab2.start())
        
        # Close login page now that layout tabs have initialized
        try:
            await login_page.close()
        except Exception:
            pass
        
        # 3. Configure grid symbols & indicators
        focus_lock = asyncio.Lock()
        await asyncio.gather(
            tab1.inject_and_configure_all(focus_lock),
            tab2.inject_and_configure_all(focus_lock)
        )

        # 4. Historical Seeding
        from concurrent.futures import ThreadPoolExecutor
        excel_pool = ThreadPoolExecutor(max_workers=4)

        if skip_seed:
            print("[Setup] --skip-seed flag active. Skipping historical seeding.")
        else:
            async def seed_wrapper(tab: CoinglassTab, sym: str):
                try:
                    for attempt in range(3):
                        try:
                            if not tab.page or tab.page.is_closed():
                                print(f"[{tab.tab_id}] [RECOVERY] Page closed on seeding attempt {attempt+1}. Reconnecting...")
                                await tab.reconnect(focus_lock)
                            await tab.seed_symbol(sym, excel_pool, focus_lock)
                            break
                        except Exception as e:
                            print(f"[Setup] [WARN] Seeding failed for {sym} (attempt {attempt+1}/3): {e}")
                            if "closed" in str(e).lower() or "navigation" in str(e).lower() or "locator" in str(e).lower() or "timeout" in str(e).lower():
                                try:
                                    await tab.reconnect(focus_lock)
                                except Exception as rec_err:
                                    print(f"[Setup] [ERROR] Failed to reconnect tab during seeding retry: {rec_err}")
                            if attempt == 2:
                                raise
                            await asyncio.sleep(3.0)
                finally:
                    tab.is_seeding = False

            seeding_tasks = [seed_wrapper(tab1, sym) for sym in TAB1_SYMBOLS] + \
                            [seed_wrapper(tab2, sym) for sym in TAB2_SYMBOLS]

            print("[Setup] Launching historical seeding...")
            await asyncio.gather(*seeding_tasks)
            print("[Setup] Seeding phase complete! Starting real-time feeds...")
            combine_seeding_files()
        
        # 5. Run Live feeds & Terminal display
        async def tab_switcher():
            active_tab = tab1
            while not stop.is_set():
                await asyncio.sleep(60.0)
                if stop.is_set():
                    break
                if tab1.is_seeding or tab2.is_seeding:
                    continue
                try:
                    try:
                        async with asyncio.timeout(3.0):
                            async with focus_lock:
                                if active_tab.page and not active_tab.page.is_closed():
                                    await active_tab.page.bring_to_front()
                    except asyncio.TimeoutError:
                        print(f"[Switcher] Warning: focus_lock timeout. Bypassing lock to force {active_tab.name} to front.")
                        if active_tab.page and not active_tab.page.is_closed():
                            await active_tab.page.bring_to_front()
                            
                    active_tab = tab2 if active_tab is tab1 else tab1
                except Exception as e:
                    print(f"[Switcher] Failed to switch to {active_tab.name}: {e}")

        async def rollover_watchdog(tracker, stop_event):
            while not stop_event.is_set():
                try:
                    tracker.update_day()
                    # Non-blocking MT5 position sync (prevents order-tracking drift)
                    if hasattr(tracker, "reconcile_with_mt5"):
                        await asyncio.to_thread(tracker.reconcile_with_mt5)
                except Exception as ex:
                    print(f"[Watchdog] [ERROR] Rollover watchdog failed: {ex}")
                await asyncio.sleep(30.0)  # tighter sync cadence for exit safety

        async def event_loop_monitor(stop_event: asyncio.Event, threshold_sec: float = 0.5) -> None:
            consecutive_blocks = 0
            while not stop_event.is_set():
                start = time.time()
                await asyncio.sleep(0.1)
                elapsed = time.time() - start - 0.1
                if elapsed > threshold_sec:
                    consecutive_blocks += 1
                    print(f"\n[ALERT] [LATENCY] Event loop blocked for {elapsed:.2f}s! Potential CPU-bound task in event loop. Consecutive count: {consecutive_blocks}")
                    if consecutive_blocks >= 5:
                        print("\n[Watchdog] [ALERT] [LATENCY_CRITICAL] Event loop blocked consecutively 5 times. Process is hung.")
                else:
                    consecutive_blocks = 0

        tasks = [
            asyncio.create_task(event_loop_monitor(stop)),
            asyncio.create_task(binance.run()),
            asyncio.create_task(binance_ws.run()),
            asyncio.create_task(renderer_loop(store, stop)),
            asyncio.create_task(watchdog([tab1, tab2, binance, binance_ws], focus_lock, stop)),
            asyncio.create_task(tab_switcher()),
            asyncio.create_task(rollover_watchdog(trade_tracker, stop))
        ]
        
        # Handle graceful exit triggers
        loop = asyncio.get_running_loop()
        def sig_handler():
            print("\n[Exit] Termination signal received. Stopping...")
            stop.set()
            tab1.running = False
            tab2.running = False
            binance.running = False
            
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, sig_handler)
            except NotImplementedError:
                pass
                
        try:
            while not stop.is_set():
                await asyncio.sleep(1.0)
        except (KeyboardInterrupt, SystemExit):
            sig_handler()
        finally:
            print("[Setup] Cleaning up tasks and closing browser...")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            excel_pool.shutdown(wait=True)
            await ctx.close()
        
    print("[Exit] Shutdown complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Coinglass + Binance Footprint Scraper")
    parser.add_argument("--skip-seed", action="store_true", help="Skip historical Excel seeding and go straight to live feeds")
    args = parser.parse_args()
    asyncio.run(main(skip_seed=args.skip_seed))
