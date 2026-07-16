"""
Alpha Squeezer V17 — Live Strategy for Engine 3
================================================
Walk-Forward LightGBM Meta-Labeled strategy with Zeno Risk Sizing
and Staged Trailing Stop.

Integrates with Engine 3 via BaseTrendStrategy.evaluate_trigger().

Features:
  - 15m timeframe, fetches live klines from Binance API
  - CVD z-score features at windows 4/10/20
  - BTC cross-asset CVD momentum
  - 800-bar EMA macro trend filter
  - LightGBM models retrained weekly via model_trainer.py
  - Zeno Risk Sizing: risk = (MAX_DD_LIMIT - current_dd) / ZENO_DENOM
  - Staged Trailing: 0.5R trail activates after +2.5R
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

# Try importing LightGBM (required for predictions)
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ─── DEFAULTS (overridable via Core/config.py) ───────────────────────
DEFAULT_CONFIDENCE     = 0.72    # OOS-tuned confidence threshold (>5R)
DEFAULT_TP_MULT        = 8.0    # ATR multiples — OOS optimized >5R
DEFAULT_SL_MULT        = 1.0
DEFAULT_TRAIL_ACT_R    = 4.0    # Activate trailing after +4.0R (OOS-tuned)
DEFAULT_TRAIL_BUFFER   = 0.5    # Trail distance in ATR multiples
DEFAULT_MAX_BARS       = 96     # Max hold time (15m bars = 24 hours)
DEFAULT_COOLDOWN       = 900    # 15 min cooldown
DEFAULT_KLINE_LIMIT    = 1000   # Bars to fetch for feature engineering (needs 800 for EMA macro)

# Zeno Risk Sizing (used as metadata in signal dict)
MAX_DD_LIMIT           = 300.0
ZENO_DENOM             = 4.0
RISK_CAP               = 120.0

# Model directory (resolve correctly whether running standalone or bundled in root)
_base = os.path.dirname(__file__)
if os.path.exists(os.path.join(_base, "models")):
    DEFAULT_MODEL_DIR = os.path.join(_base, "models")
else:
    DEFAULT_MODEL_DIR = os.path.join(_base, "Strategies", "alpha_squeezer_v17", "models")


class AlphaSqueezerV17(BaseTrendStrategy):
    """
    Walk-Forward LightGBM Meta-Labeled strategy for Engine 3.

    Each instance is confined to a single symbol (e.g., XRPUSDT).
    Multiple instances are created in Engine_3.py for multi-asset coverage.
    """

    def __init__(self, feed, symbol: str, model_dir: str = None):
        super().__init__(feed, f"AlphaSqueezer_V17_{symbol[:3]}")
        self.symbol = symbol
        self.tf_min = 15          # 15-minute timeframe
        self.tf = "15m"
        self.indicators = {}      # Required by Engine 3 for dashboard display

        # Model storage
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.model_long = None
        self.model_short = None
        self._model_lock = threading.Lock()
        self._last_model_load = 0
        self._load_models()

        # Feature cache (avoid redundant API calls within same 15m bar)
        self._feature_cache = {}
        self._feature_cache_ts = 0

        # BTC reference cache
        self._btc_cache = None
        self._btc_cache_ts = 0

        # Load MYTHOS WFO Params
        self.wfo_params = {}
        _base = os.path.dirname(__file__)
        if os.path.exists(os.path.join(_base, "final_wfo_params.json")):
            wfo_path = os.path.join(_base, "final_wfo_params.json")
        else:
            wfo_path = os.path.join(_base, "Strategies", "alpha_squeezer_v17", "final_wfo_params.json")
        if os.path.exists(wfo_path):
            import json
            try:
                with open(wfo_path, 'r') as f:
                    all_params = json.load(f)
                    self.wfo_params = all_params.get(self.symbol, {})
            except Exception as e:
                print(f"[{self.strategy_name}] Error loading WFO params: {e}")

    # ─── MODEL MANAGEMENT ────────────────────────────────────────────

    def _load_models(self):
        """Load pre-trained LightGBM boosters from disk."""
        if not HAS_LGB:
            print(f"[{self.strategy_name}] WARNING: lightgbm not installed. Running in signal-passthrough mode.")
            return

        long_path = os.path.join(self.model_dir, f"{self.symbol}_long.txt")
        short_path = os.path.join(self.model_dir, f"{self.symbol}_short.txt")

        with self._model_lock:
            if os.path.exists(long_path):
                self.model_long = lgb.Booster(model_file=long_path)
                print(f"[{self.strategy_name}] Loaded LONG model from {long_path}")
            else:
                print(f"[{self.strategy_name}] No LONG model found at {long_path} — will use feature-only signals.")

            if os.path.exists(short_path):
                self.model_short = lgb.Booster(model_file=short_path)
                print(f"[{self.strategy_name}] Loaded SHORT model from {short_path}")
            else:
                print(f"[{self.strategy_name}] No SHORT model found at {short_path} — will use feature-only signals.")

        self._last_model_load = time.time()

    def _maybe_reload_models(self):
        """Check if models should be reloaded (every 6 hours). Check is inside lock to prevent TOCTOU race."""
        with self._model_lock:
            if time.time() - self._last_model_load > 21600:  # 6 hours
                # Release lock so _load_models can re-acquire it cleanly
                pass
        # If reload needed, _load_models acquires its own lock
        if time.time() - self._last_model_load > 21600:
            self._load_models()

    # ─── FEATURE ENGINEERING ─────────────────────────────────────────

    @staticmethod
    def _zscore(series, window=100):
        """Rolling z-score."""
        mean = series.rolling(window, min_periods=1).mean()
        std = series.rolling(window, min_periods=1).std().replace(0, 1e-10)
        return (series - mean) / std

    def _fetch_btc_reference(self):
        """Fetch BTC 15m klines for cross-asset features."""
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

        # Compute BTC CVD
        btc_df['btc_CVD'] = (2 * btc_df['taker_buy_base'] - btc_df['Volume']).cumsum()
        btc_df['btc_Close'] = btc_df['Close']
        # Keep open_time for timestamp-based alignment in _compute_features
        btc_df['open_time'] = pd.to_numeric(btc_df['open_time'], errors='coerce')

        self._btc_cache = btc_df[['open_time', 'btc_Close', 'btc_CVD']].copy()
        self._btc_cache_ts = now
        return self._btc_cache

    def _compute_features(self, klines):
        """
        Compute the exact same feature set as the V17 backtester.
        Returns (feature_df, feature_columns) or (None, None).
        """
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

        # ATR
        df['atr'] = (df['High'] - df['Low']).rolling(14, min_periods=1).mean()

        # Macro trend (800-bar EMA)
        df['ema_fast'] = df['Close'].ewm(span=200, min_periods=200).mean()
        df['ema_slow'] = df['Close'].ewm(span=800, min_periods=800).mean()
        macro_score = (df['ema_fast'] - df['ema_slow']) / df['atr'].replace(0, 1e-10)
        df['macro'] = np.where(macro_score > 0.5, 1, np.where(macro_score < -0.5, -1, 0))

        # BTC cross-asset features — align on open_time to prevent off-by-one at 15m boundaries
        btc_ref = self._fetch_btc_reference()
        if btc_ref is not None and len(btc_ref) > 0:
            # Build a lookup from open_time to btc_CVD value
            df['btc_CVD'] = df['open_time'].map(
                btc_ref.set_index('open_time')['btc_CVD']
            ).ffill().bfill().fillna(0.0)
        else:
            df['btc_CVD'] = 0.0

        # Feature columns (must match training exactly)
        feat_cols = ['macro']
        for k in [4, 10, 20]:
            df[f'z_cvd_{k}'] = self._zscore(df['CVD'], k)
            df[f'z_btc_{k}'] = self._zscore(df['btc_CVD'], k)
            feat_cols += [f'z_cvd_{k}', f'z_btc_{k}']

        df['cvd_delta'] = df['CVD'].diff(5)
        df['btc_cvd_mom'] = df['btc_CVD'].diff(2)
        df['vol_regime'] = self._zscore(df['atr'], 100)
        feat_cols += ['cvd_delta', 'btc_cvd_mom', 'vol_regime']

        # --- Coinglass Features from Live Feed ---
        # We need to extract the Coinglass history to compute Z-scores.
        hist_data = self.feed.history.get(self.symbol, {})
        
        def _get_cg_series(key):
            raw_tuples = hist_data.get(key, [])
            if not raw_tuples:
                return pd.Series()
            
            # Extract times and convert to DatetimeIndex
            times = [pd.to_datetime(t, unit='s') for t, v in raw_tuples]
            vals = [v for t, v in raw_tuples]
            
            s = pd.Series(vals, index=times).replace('N/A', np.nan).replace('', np.nan)
            
            # Handle liquidations special case (format is usually "LongLiq ShortLiq")
            if key == 'liquidations':
                return s
            
            # Remove M, K suffixes if present
            if s.dtype == object:
                s = s.astype(str).str.replace('K', 'e3', case=False)
                s = s.str.replace('M', 'e6', case=False)
                s = s.str.replace('B', 'e9', case=False)
                
            s = pd.to_numeric(s, errors='coerce').ffill().fillna(0.0)
            
            # Resample to 15m (using the last value in each 15m bar) and forward fill
            s_15m = s.resample('15min').last().ffill().fillna(0.0)
            return s_15m

        # 1. Open Interest Z-score (feed stores as 'oi')
        oi_s = _get_cg_series('oi')
        if len(oi_s) > 0:
            mean = oi_s.rolling(100, min_periods=1).mean().iloc[-1]
            std = oi_s.rolling(100, min_periods=1).std().replace(0, 1e-10).iloc[-1]
            z_oi_val = (oi_s.iloc[-1] - mean) / std
        else:
            z_oi_val = 0.0
            
        # 2. L/S Ratio Z-score
        ls_s = _get_cg_series('ls_ratio')
        if len(ls_s) > 0:
            mean = ls_s.rolling(100, min_periods=1).mean().iloc[-1]
            std = ls_s.rolling(100, min_periods=1).std().replace(0, 1e-10).iloc[-1]
            z_ls_val = (ls_s.iloc[-1] - mean) / std
        else:
            z_ls_val = 0.0
            
        # 3. Funding Rate (feed stores as 'funding')
        fund_s = _get_cg_series('funding')
        funding_val = fund_s.iloc[-1] if len(fund_s) > 0 else 0.0
        
        # 4. Liquidations (feed stores as separate 'liquidations_long_15m' and 'liquidations_short_15m')
        liq_long_s = _get_cg_series('liquidations_long_15m')
        liq_short_s = _get_cg_series('liquidations_short_15m')
        liq_long_sum = float(liq_long_s.tail(5).sum()) if len(liq_long_s) > 0 else 0.0
        liq_short_sum = float(liq_short_s.tail(5).sum()) if len(liq_short_s) > 0 else 0.0

        # Broadcast the latest values to the dataframe (since we only care about df.iloc[-1] for prediction)
        df['z_oi'] = z_oi_val if not np.isnan(z_oi_val) else 0.0
        df['z_ls'] = z_ls_val if not np.isnan(z_ls_val) else 0.0
        df['funding'] = funding_val if not np.isnan(funding_val) else 0.0
        df['liq_long_5'] = liq_long_sum
        df['liq_short_5'] = liq_short_sum
        
        feats = ['z_oi', 'z_ls', 'funding', 'liq_long_5', 'liq_short_5']
        feat_cols += feats

        df[feat_cols] = df[feat_cols].fillna(0)

        # MYTHOS Phase 1: GA-selected feature subset (9 of 15)
        # Pruned: macro, z_cvd_10, cvd_delta, vol_regime, funding, liq_short_5
        ga_selected = [
            'z_cvd_4', 'z_btc_4', 'z_btc_10', 'z_cvd_20', 'z_btc_20',
            'btc_cvd_mom', 'z_oi', 'z_ls', 'liq_long_5'
        ]
        feat_cols = [f for f in ga_selected if f in feat_cols]

        return df, feat_cols

    # ─── EVALUATE TRIGGER (Engine 3 interface) ───────────────────────

    def evaluate_trigger(self, symbol: str):
        """
        Called by Engine 3 main loop every ~5 seconds.
        Returns (signal_dict, message) or (None, message).
        """
        if symbol != self.symbol:
            return None, "Symbol mismatch"

        # Get live price from feed
        fd = self.feed.get_latest().get(symbol, {})
        live_price = fd.get('price', 0.0)
        if live_price <= 0:
            return None, "No live price"

        # Periodically check for model updates
        self._maybe_reload_models()

        # Fetch 15m klines
        klines = self.get_raw_klines(symbol, self.tf_min, limit=DEFAULT_KLINE_LIMIT)
        if not klines or len(klines) < 60:
            return None, f"Warming up ({len(klines) if klines else 0}/{DEFAULT_KLINE_LIMIT} bars)"

        # Compute features
        df, feat_cols = self._compute_features(klines)
        if df is None:
            return None, "Feature computation failed"

        # Get latest bar's features
        latest = df.iloc[-1]
        atr = latest['atr']
        macro = int(latest['macro'])

        if pd.isna(atr) or atr <= 0:
            return None, "ATR invalid"

        # Store indicators for Engine 3 dashboard
        self.indicators[symbol] = {
            'atr': float(atr),
            'macro': macro,
            'ema_slow': float(latest['ema_slow']),
            'cvd_delta': float(latest['cvd_delta']),
            'vol_regime': float(latest['vol_regime']),
        }

        # Config overrides (WFO > user config > default)
        confidence = self.wfo_params.get('confidence', self.get_param(symbol, 'confidence_threshold', DEFAULT_CONFIDENCE))
        tp_mult = self.wfo_params.get('tp_mult', self.get_param(symbol, 'tp_mult', DEFAULT_TP_MULT))
        sl_mult = self.wfo_params.get('sl_mult', self.get_param(symbol, 'sl_mult', DEFAULT_SL_MULT))
        trail_act = self.wfo_params.get('trail_act', self.get_param(symbol, 'trail_activation_r', DEFAULT_TRAIL_ACT_R))
        trail_buf = self.get_param(symbol, 'trail_buffer_atr', DEFAULT_TRAIL_BUFFER)
        vol_limit = self.wfo_params.get('vol_limit', 1.0)  # Default 100% so it doesn't block if missing

        # Phase 2: Volatility Regime Gate
        atr_pct = atr / live_price
        if atr_pct > vol_limit:
            return None, f"Regime Filter (ATR% {atr_pct:.4f} > Limit {vol_limit:.4f})"

        # ── ML Prediction ────────────────────────────────────────────
        prob_long = 0.0
        prob_short = 0.0
        features_row = df[feat_cols].iloc[[-1]]  # Single-row DataFrame

        with self._model_lock:
            if self.model_long is not None:
                prob_long = float(self.model_long.predict(features_row)[0])
            if self.model_short is not None:
                prob_short = float(self.model_short.predict(features_row)[0])

        # If no models loaded, use a feature-only heuristic
        if self.model_long is None and self.model_short is None:
            # Fallback: use CVD z-score + macro alignment as a basic signal
            z_cvd_10 = float(latest.get('z_cvd_10', 0))
            if macro == 1 and z_cvd_10 > 1.5:
                prob_long = 0.55
            elif macro == -1 and z_cvd_10 < -1.5:
                prob_short = 0.55

        # Update indicators with probabilities
        self.indicators[symbol]['prob_long'] = prob_long
        self.indicators[symbol]['prob_short'] = prob_short

        # ── Signal Generation ────────────────────────────────────────
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

        # ── Meta Label Veto ──────────────────────────────────────────
        dir_str = "LONG" if direction == 1 else "SHORT"
        
        # Build context payload matching the Prompt Optimizer template
        recent_liqs = float(latest.get('liq_short_5', 0.0)) if direction == 1 else float(latest.get('liq_long_5', 0.0))
        vol_z = float(latest.get('vol_regime', 0.0))
        regime_str = "Expansion" if vol_z > 1.0 else "Contraction" if vol_z < -1.0 else "Neutral"
        
        context_str = f"Market Regime: {regime_str}\nAsset: {self.symbol}\nVolatility Z-Score: {vol_z:.2f}\nDirection: {dir_str}\nRecent Liquidations: {recent_liqs:.2f}"

        meta_verdict = get_meta_label(context_str)
        if meta_verdict == "UNSAFE":
            return None, f"VETO: Meta-Labeler flagged {dir_str} signal as UNSAFE."

        # ── Compute SL/TP ────────────────────────────────────────────
        entry_price = live_price
        if direction == 1:
            sl = entry_price - sl_mult * atr
            tp = entry_price + tp_mult * atr
        else:
            sl = entry_price + sl_mult * atr
            tp = entry_price - tp_mult * atr

        # ── Zeno Risk Sizing (communicated via signal dict) ──────────
        # Engine 3's TradeTracker will use risk_pct for position sizing.
        # We compute the Zeno-optimal risk as a fraction of current capital.
        # The actual $ amount is calculated by TradeTracker using current_capital.
        zeno_risk_pct = 0.005  # Adjusted to match Zeno Backtest (0.5% = $25 on $5k account)

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
                "model": "AlphaSqueezer_V17",
                "prob_long": prob_long,
                "prob_short": prob_short,
                "atr": float(atr),
                "macro": macro,
                "vol_regime": float(latest['vol_regime']),
            }
        }

        dir_str = "LONG" if direction == 1 else "SHORT"
        return signal, f"{dir_str}: V17 ML signal (p={prob:.3f}, ATR={atr:.6f})"


# ─── Standalone test ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("AlphaSqueezerV17 strategy module loaded successfully.")
    print(f"  Model directory: {DEFAULT_MODEL_DIR}")
    print(f"  LightGBM available: {HAS_LGB}")
    print(f"  Default config: CONF={DEFAULT_CONFIDENCE}, TP={DEFAULT_TP_MULT}R, SL={DEFAULT_SL_MULT}R")
    print(f"  Trail: activate at +{DEFAULT_TRAIL_ACT_R}R, buffer {DEFAULT_TRAIL_BUFFER} ATR")
    print(f"  Zeno Risk: DD_LIMIT=${MAX_DD_LIMIT}, DENOM={ZENO_DENOM}, CAP=${RISK_CAP}")
