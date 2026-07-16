"""
ML Trend Pull — Live Strategy for Engine 3
================================================
Walk-Forward LightGBM/XGBoost/CatBoost Meta-Labeled strategy with Zeno Risk Sizing
and Staged Trailing Stop.

Integrates with Engine 3 via BaseTrendStrategy.evaluate_trigger().
"""

import os
import sys
import time
import threading
import numpy as np
import pandas as pd

# Ensure Core is importable
_CORE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Core"))
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

from base_trend_strategy import BaseTrendStrategy
from config import get_param
from meta_labeler import get_meta_label

# Try importing LightGBM
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ─── DEFAULTS (overridable via Core/config.py) ───────────────────────
DEFAULT_CONFIDENCE     = 0.72  
DEFAULT_TP_MULT        = 8.0    # ATR multiples - OOS optimized >5R
DEFAULT_SL_MULT        = 1.0
DEFAULT_TRAIL_ACT_R    = 4.0    # Activate trailing after +4.0R
DEFAULT_TRAIL_BUFFER   = 0.5    # Trail distance in ATR multiples
DEFAULT_MAX_BARS       = 96     # Max hold time (15m bars = 24 hours)
DEFAULT_COOLDOWN       = 900    # 15 min cooldown
DEFAULT_KLINE_LIMIT    = 1000   

# Zeno Risk Sizing
MAX_DD_LIMIT           = 300.0
ZENO_DENOM             = 4.0
RISK_CAP               = 120.0

_base = os.path.dirname(__file__)
if os.path.exists(os.path.join(_base, "models")):
    DEFAULT_MODEL_DIR = os.path.join(_base, "models")
else:
    DEFAULT_MODEL_DIR = os.path.join(_base, "Strategies", "ml_trend_pull", "models")


class MLTrendPull(BaseTrendStrategy):
    """
    Walk-Forward Ensemble Meta-Labeled strategy.
    Multiple instances are created in Engine_3.py for multi-asset coverage.
    """

    def __init__(self, feed, symbol: str, model_dir: str = None):
        super().__init__(feed, f"ML_Trend_Pull_{symbol[:3]}")
        self.symbol = symbol
        self.tf_min = 15          
        self.tf = "15m"
        self.indicators = {}      

        # Model storage
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.model_long = None
        self.model_short = None
        self._model_lock = threading.Lock()
        self._last_model_load = 0
        self._load_models()

        self._feature_cache = {}
        self._feature_cache_ts = 0

        self._btc_cache = None
        self._btc_cache_ts = 0

        self.wfo_params = {}
        _base = os.path.dirname(__file__)
        if os.path.exists(os.path.join(_base, "final_wfo_params.json")):
            wfo_path = os.path.join(_base, "final_wfo_params.json")
        else:
            wfo_path = os.path.join(_base, "agent5_configs", f"{self.symbol}.json")
            
        if os.path.exists(wfo_path):
            import json
            try:
                with open(wfo_path, 'r') as f:
                    all_params = json.load(f)
                    if self.symbol in all_params:
                        self.wfo_params = all_params.get(self.symbol, {})
                    else:
                        self.wfo_params = all_params # direct symbol config
            except Exception as e:
                print(f"[{self.strategy_name}] Error loading configs: {e}")

    def _load_models(self):
        """Load pre-trained LightGBM boosters from disk."""
        if not HAS_LGB:
            print(f"[{self.strategy_name}] WARNING: lightgbm not installed.")
            return

        long_path = os.path.join(self.model_dir, f"{self.symbol}_long_lgb.txt")
        short_path = os.path.join(self.model_dir, f"{self.symbol}_short_lgb.txt")

        with self._model_lock:
            if os.path.exists(long_path):
                self.model_long = lgb.Booster(model_file=long_path)
                print(f"[{self.strategy_name}] Loaded LONG model from {long_path}")
            if os.path.exists(short_path):
                self.model_short = lgb.Booster(model_file=short_path)
                print(f"[{self.strategy_name}] Loaded SHORT model from {short_path}")

        self._last_model_load = time.time()

    def _maybe_reload_models(self):
        if time.time() - self._last_model_load > 21600:  # 6 hours
            self._load_models()

    @staticmethod
    def _zscore(series, window=100):
        mean = series.rolling(window, min_periods=1).mean()
        std = series.rolling(window, min_periods=1).std().replace(0, 1e-10)
        return (series - mean) / std

    def _fetch_btc_reference(self):
        now = time.time()
        if self._btc_cache is not None and (now - self._btc_cache_ts) < 60:
            return self._btc_cache

        klines = self.get_raw_klines("BTCUSDT", self.tf_min, limit=DEFAULT_KLINE_LIMIT)
        if not klines or len(klines) < 30:
            return None

        btc_df = pd.DataFrame(klines, columns=[
            'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'close_time', 'quote_vol', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        for c in ['Open', 'High', 'Low', 'Close', 'Volume', 'taker_buy_base']:
            btc_df[c] = pd.to_numeric(btc_df[c], errors='coerce')
        btc_df['open_time'] = pd.to_numeric(btc_df['open_time'], errors='coerce')

        btc_df['btc_CVD'] = (2 * btc_df['taker_buy_base'] - btc_df['Volume']).cumsum()
        btc_df['btc_Close'] = btc_df['Close']

        self._btc_cache = btc_df[['open_time', 'btc_Close', 'btc_CVD']].copy()
        self._btc_cache_ts = now
        return self._btc_cache

    def _compute_features(self, klines):
        if not klines or len(klines) < 60:
            return None, None

        df = pd.DataFrame(klines, columns=[
            'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'close_time', 'quote_vol', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        for c in ['Open', 'High', 'Low', 'Close', 'Volume', 'taker_buy_base']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')

        # CVD from taker buy/sell
        df['CVD'] = (2 * df['taker_buy_base'] - df['Volume']).cumsum()
        df['atr'] = (df['High'] - df['Low']).rolling(14, min_periods=1).mean()
        df['cvd_delta'] = df['CVD'].diff(5)

        # BTC reference
        btc_ref = self._fetch_btc_reference()
        if btc_ref is not None and len(btc_ref) > 0:
            df['btc_CVD'] = df['open_time'].map(
                btc_ref.set_index('open_time')['btc_CVD']
            ).ffill().bfill().fillna(0.0)
            df['btc_Close'] = df['open_time'].map(
                btc_ref.set_index('open_time')['btc_Close']
            ).ffill().bfill().fillna(0.0)
        else:
            df['btc_CVD'] = 0.0
            df['btc_Close'] = df['Close']

        if 'btc_CVD' in df.columns:
            df['btc_cvd_mom'] = df['btc_CVD'].diff(2)
        else:
            df['btc_cvd_mom'] = 0.0

        # Trend indicators: EMA 200 and EMA 800
        df['ema_fast'] = df['Close'].ewm(span=200, min_periods=200).mean()
        df['ema_slow'] = df['Close'].ewm(span=800, min_periods=800).mean()
        
        # Trend state: 1 for uptrend, -1 for downtrend, 0 for neutral
        macro_score = (df['ema_fast'] - df['ema_slow']) / df['atr'].replace(0, 1e-10)
        df['macro'] = np.where(macro_score > 0.5, 1, np.where(macro_score < -0.5, -1, 0))
        
        # Pullback features:
        df['ema_8'] = df['Close'].ewm(span=8, min_periods=1).mean()
        df['ema_21'] = df['Close'].ewm(span=21, min_periods=1).mean()
        df['ema_50'] = df['Close'].ewm(span=50, min_periods=1).mean()
        
        df['pull_ema8'] = (df['Close'] - df['ema_8']) / df['atr'].replace(0, 1e-10)
        df['pull_ema21'] = (df['Close'] - df['ema_21']) / df['atr'].replace(0, 1e-10)
        df['pull_ema50'] = (df['Close'] - df['ema_50']) / df['atr'].replace(0, 1e-10)
        
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
            df[f'z_cvd_{k}'] = self._zscore(df['CVD'], k)
            df[f'z_btc_{k}'] = self._zscore(df['btc_CVD'], k)
            
        df['vol_regime'] = self._zscore(df['atr'], 100)

        feat_cols = [
            'macro', 'pull_ema8', 'pull_ema21', 'pull_ema50', 'rsi', 'stoch_k',
            'z_cvd_4', 'z_btc_4', 'z_btc_10', 'z_cvd_20', 'z_btc_20',
            'cvd_delta', 'btc_cvd_mom', 'vol_regime'
        ]

        # Orderbook metrics if present from feed
        hist_data = self.feed.history.get(self.symbol, {})
        def _get_cg_series(key):
            raw_tuples = hist_data.get(key, [])
            if not raw_tuples:
                return pd.Series()
            times = [pd.to_datetime(t, unit='s') for t, v in raw_tuples]
            vals = [v for t, v in raw_tuples]
            s = pd.Series(vals, index=times).replace('N/A', np.nan).replace('', np.nan)
            s_15m = s.astype(float).resample('15min').last().ffill().fillna(0.0)
            return s_15m

        oi_s = _get_cg_series('oi')
        if len(oi_s) > 0:
            df['z_oi'] = self._zscore(oi_s, 100).iloc[-1]
            feat_cols.append('z_oi')
        else:
            df['z_oi'] = 0.0

        ls_s = _get_cg_series('ls_ratio')
        if len(ls_s) > 0:
            df['z_ls'] = self._zscore(ls_s, 100).iloc[-1]
            feat_cols.append('z_ls')
        else:
            df['z_ls'] = 0.0

        fund_s = _get_cg_series('funding')
        if len(fund_s) > 0:
            df['funding'] = fund_s.iloc[-1]
            feat_cols.append('funding')
        else:
            df['funding'] = 0.0

        df[feat_cols] = df[feat_cols].fillna(0)
        return df, feat_cols

    def evaluate_trigger(self, symbol: str):
        if symbol != self.symbol:
            return None, "Symbol mismatch"

        fd = self.feed.get_latest().get(symbol, {})
        live_price = fd.get('price', 0.0)
        if live_price <= 0:
            return None, "No live price"

        self._maybe_reload_models()

        klines = self.get_raw_klines(symbol, self.tf_min, limit=DEFAULT_KLINE_LIMIT)
        if not klines or len(klines) < 60:
            return None, f"Warming up ({len(klines) if klines else 0}/{DEFAULT_KLINE_LIMIT} bars)"

        df, feat_cols = self._compute_features(klines)
        if df is None:
            return None, "Feature computation failed"

        latest = df.iloc[-1]
        atr = latest['atr']
        macro = int(latest['macro'])

        if pd.isna(atr) or atr <= 0:
            return None, "ATR invalid"

        # Indicators for dashboard
        self.indicators[symbol] = {
            'atr': float(atr),
            'macro': macro,
            'ema_slow': float(latest['ema_slow']),
            'cvd_delta': float(latest['cvd_delta']),
            'vol_regime': float(latest['vol_regime']),
        }

        # Config overrides
        confidence = self.wfo_params.get('confidence', get_param(symbol, 'ML_Trend_Pull', 'confidence_threshold', DEFAULT_CONFIDENCE))
        tp_mult = self.wfo_params.get('tp_mult', get_param(symbol, 'ML_Trend_Pull', 'tp_mult', DEFAULT_TP_MULT))
        sl_mult = self.wfo_params.get('sl_mult', get_param(symbol, 'ML_Trend_Pull', 'sl_mult', DEFAULT_SL_MULT))
        trail_act = self.wfo_params.get('trail_act', get_param(symbol, 'ML_Trend_Pull', 'trail_activation_r', DEFAULT_TRAIL_ACT_R))
        trail_buf = get_param(symbol, 'ML_Trend_Pull', 'trail_buffer_atr', DEFAULT_TRAIL_BUFFER)

        prob_long = 0.0
        prob_short = 0.0
        features_row = df[feat_cols].iloc[[-1]]

        with self._model_lock:
            if self.model_long is not None:
                prob_long = float(self.model_long.predict(features_row)[0])
            if self.model_short is not None:
                prob_short = float(self.model_short.predict(features_row)[0])

        self.indicators[symbol]['prob_long'] = prob_long
        self.indicators[symbol]['prob_short'] = prob_short

        direction = 0
        prob = 0.0

        if prob_long > confidence and prob_long > prob_short and macro == 1:
            direction = 1
            prob = prob_long
        elif prob_short > confidence and prob_short > prob_long and macro == -1:
            direction = -1
            prob = prob_short
        else:
            return None, f"Searching (L={prob_long:.3f} S={prob_short:.3f} M={macro})"

        # Meta Label Veto
        dir_str = "LONG" if direction == 1 else "SHORT"
        recent_liqs = float(latest.get('liq_short_5', 0.0)) if direction == 1 else float(latest.get('liq_long_5', 0.0))
        vol_z = float(latest.get('vol_regime', 0.0))
        regime_str = "Expansion" if vol_z > 1.0 else "Contraction" if vol_z < -1.0 else "Neutral"
        
        context_str = f"Market Regime: {regime_str}\nAsset: {self.symbol}\nVolatility Z-Score: {vol_z:.2f}\nDirection: {dir_str}\nRecent Liquidations: {recent_liqs:.2f}"
        meta_verdict = get_meta_label(context_str)
        if meta_verdict == "UNSAFE":
            return None, f"VETO: Meta-Labeler flagged {dir_str} signal as UNSAFE."

        # Compute SL/TP
        entry_price = live_price
        if direction == 1:
            sl = entry_price - sl_mult * atr
            tp = entry_price + tp_mult * atr
        else:
            sl = entry_price + sl_mult * atr
            tp = entry_price - tp_mult * atr

        zeno_risk_pct = 0.005  # 0.5% default

        signal = {
            "direction": direction,
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "risk_pct": zeno_risk_pct,
            "use_lock_and_trail": False,
            "trail_activation_atr": trail_act,
            "trail_dist_atr": trail_buf,
            "entry_atr": float(atr),
            "ml_confidence": prob,
            "meta": {
                "model": "ML_Trend_Pull",
                "prob_long": prob_long,
                "prob_short": prob_short,
                "atr": float(atr),
                "macro": macro,
                "vol_regime": float(latest['vol_regime']),
            }
        }

        return signal, f"{dir_str}: ML pullback signal (p={prob:.3f}, ATR={atr:.6f})"
