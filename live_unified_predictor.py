import os
import time
import json
import threading
import collections
import dataclasses
from typing import List, Any, Dict
import pandas as pd
import numpy as np
import importlib

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from pipeline_diag import DIAG
except ImportError:
    DIAG = None

from live_model_trainer import (
    generate_features_standard,
    prep_vwap,
    rolling_mean_numba,
    rolling_zscore_numba,
    predict_model_fast,
    TP_MULT_OPTIONS,
    TRAIL_ATR_OPTIONS
)
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from numba import njit

class SimpleEnsembleClassifier:
    def __init__(self, lgb_model, xgb_model, cat_model):
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.cat_model = cat_model

    def predict_proba(self, X):
        p_lgb = self.lgb_model.predict(X) if hasattr(self.lgb_model, 'predict') and 'Booster' in str(type(self.lgb_model)) else self.lgb_model.predict_proba(X)[:, 1]
        p_xgb = self.xgb_model.predict_proba(X)[:, 1]
        p_cat = self.cat_model.predict_proba(X)[:, 1]
        p_mean = (p_lgb + p_xgb + p_cat) / 3.0
        return np.column_stack([1.0 - p_mean, p_mean])



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
    return (df, feat_cols)

class UnifiedLivePredictor:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.models = {}
        self.features_cols = {}
        self.candles_history = {s: collections.deque(maxlen=1200) for s in symbols}
        self.current_candle = {}
        self._lock = threading.RLock()
        self.recent_capitals = []
        self.latest_atr = {}
        self.last_model_mtime = 0
        self.strategies = ['S1_Liquidation', 'S2_CVD', 'S3_Trend', 'ML_Vwap_Reversal', 'S5_Microstructure', 'S6_SMC_Orderflow']
        self._last_predict_bar = {}
        self._cached_signal = {}
        self.manifest_data = {}
        self.load_models()

    def check_model_updates(self):
        now = time.time()
        if now - getattr(self, '_last_model_check_time', 0.0) < 10.0:
            return
        self._last_model_check_time = now
        
        manifest_path = os.path.join(BASE_DIR, 'ml_trend_pull', 'models', 'manifest.json')
        if os.path.exists(manifest_path):
            mtime = os.path.getmtime(manifest_path)
            if mtime > self.last_model_mtime:
                with self._lock:
                    if mtime <= self.last_model_mtime:
                        return
                    self.last_model_mtime = mtime
                time.sleep(0.1)
                print(f'[UnifiedPredictor] Detected new model manifest (mtime: {mtime}). Hot-Swap...')
                self.load_models()

    def load_models(self):
        print('[UnifiedPredictor] Loading unified ensemble models across all 6 strategies...')
        search_dirs = [
            os.path.join(BASE_DIR, 'models'),
            os.path.join(BASE_DIR, 'Liquidation', 'models'),
            os.path.join(BASE_DIR, 'ml_trend_pull', 'models')
        ]
        
        temp_models = {}
        temp_features_cols = {}
        temp_manifest_data = {}
        
        for strat in self.strategies:
            temp_models[strat] = {}
            temp_features_cols[strat] = {}
            for sym in self.symbols:
                for m_dir in search_dirs:
                    lgb_path = os.path.join(m_dir, f'{strat}_{sym}_lgb.txt')
                    xgb_path = os.path.join(m_dir, f'{strat}_{sym}_xgb.json')
                    cb_path = os.path.join(m_dir, f'{strat}_{sym}_cb.cbm')
                    cols_path = os.path.join(m_dir, f'{strat}_{sym}_cols.json')
                    
                    if os.path.exists(lgb_path) and os.path.exists(xgb_path) and os.path.exists(cb_path):
                        try:
                            lgb_model = lgb.Booster(model_file=lgb_path)
                            xgb_model = xgb.XGBClassifier()
                            xgb_model.load_model(xgb_path)
                            cat_model = CatBoostClassifier()
                            cat_model.load_model(cb_path)
                            
                            temp_models[strat][sym] = SimpleEnsembleClassifier(lgb_model, xgb_model, cat_model)
                            with open(cols_path, 'r') as f:
                                temp_features_cols[strat][sym] = json.load(f)
                            break
                        except Exception as e:
                            print(f'[UnifiedPredictor] Error loading {strat} for {sym}: {e}')
                            
        for m_dir in search_dirs:
            manifest_path = os.path.join(m_dir, 'manifest.json')
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, 'r') as f:
                        temp_manifest_data.update(json.load(f))
                except Exception as e:
                    print(f"[UnifiedPredictor] Error loading manifest from {m_dir}: {e}")
                    
        # Atomic swap under lock to prevent race conditions during hot-swapping
        with self._lock:
            self.models = temp_models
            self.features_cols = temp_features_cols
            self.manifest_data.update(temp_manifest_data)
            for m_dir in search_dirs:
                manifest_path = os.path.join(m_dir, 'manifest.json')
                if os.path.exists(manifest_path):
                    self.last_model_mtime = max(self.last_model_mtime, os.path.getmtime(manifest_path))

        for strat in self.strategies:
            cnt = len(self.models.get(strat, {}))
            print(f"[UnifiedPredictor] Strategy '{strat}': {cnt}/{len(self.symbols)} active ensemble models loaded.")
        print('[UnifiedPredictor] Finished loading all strategy models.')

    def on_tick_update(self, symbol, snap, trade_tracker=None):
        if snap.price <= 0.0: return snap
        self.check_model_updates()
        
        history_copy = None
        current_candle_copy = None
        new_bar = False

        with self._lock:
            now = time.time()
            open_time = int(now // 900) * 900
            history = self.candles_history[symbol]
            
            if symbol not in self.current_candle or self.current_candle[symbol].get('open_time') != open_time:
                new_bar = True
                prev = self.current_candle.get(symbol)
                if prev and int(prev.get('open_time', 0)) < open_time:
                    prev_ot = int(prev['open_time'])
                    if not history or int(history[-1].get('open_time', 0)) != prev_ot:
                        history.append(dict(prev))
                
                self.current_candle[symbol] = {
                    'open_time': open_time, 'open': snap.price, 'high': snap.price,
                    'low': snap.price, 'close': snap.price, 'volume': snap.volume,
                    'fut_cvd': snap.fut_cvd, 'liq_long': abs(snap.liq_long), 'liq_short': abs(snap.liq_short),
                    'coins_bid': abs(snap.coins_bid), 'coins_ask': abs(snap.coins_ask),
                    'dollars_bid': abs(snap.dollars_bid), 'dollars_ask': abs(snap.dollars_ask),
                    'tk_buy_cnt': abs(snap.tk_buy_cnt), 'tk_sell_cnt': abs(snap.tk_sell_cnt),
                    'fp_poc': snap.fp_poc,
                    'oi': snap.oi, 'funding': snap.funding, 'ls_ratio': snap.ls_ratio,
                    'rsi': snap.rsi, 'whale_idx': snap.whale_idx
                }
            else:
                candle = self.current_candle[symbol]
                candle['close'] = snap.price
                if snap.price > candle['high']: candle['high'] = snap.price
                if snap.price < candle['low'] or candle['low'] == 0.0: candle['low'] = snap.price
                candle['volume'] = snap.volume
                candle['fut_cvd'] = snap.fut_cvd
                candle['liq_long'] = abs(snap.liq_long)
                candle['liq_short'] = abs(snap.liq_short)
                candle['coins_bid'] = abs(snap.coins_bid)
                candle['coins_ask'] = abs(snap.coins_ask)
                candle['dollars_bid'] = abs(snap.dollars_bid)
                candle['dollars_ask'] = abs(snap.dollars_ask)
                candle['tk_buy_cnt'] = abs(snap.tk_buy_cnt)
                candle['tk_sell_cnt'] = abs(snap.tk_sell_cnt)
                candle['fp_poc'] = snap.fp_poc
                candle['oi'] = snap.oi
                candle['funding'] = snap.funding
                candle['ls_ratio'] = snap.ls_ratio
                candle['rsi'] = snap.rsi
                candle['whale_idx'] = snap.whale_idx
                
            if len(history) >= 249:
                history_copy = list(history)
                if symbol in self.current_candle:
                    current_candle_copy = dict(self.current_candle[symbol])
                
        if history_copy:
            self._run_inference_with_copied_history(symbol, snap.price, trade_tracker, history_copy, current_candle_copy, trigger_trade=new_bar)

        return snap

    def _run_inference_with_copied_history(self, symbol, current_price, trade_tracker, history, current_candle, trigger_trade=True):
        if current_price <= 0 or not history: return
        
        # Throttle ML prediction for live updates to max 1 per second per symbol to prevent GIL starvation
        if not trigger_trade:
            now = time.time()
            if not hasattr(self, '_last_ml_inference'):
                self._last_ml_inference = {}
            if now - self._last_ml_inference.get(symbol, 0) < 1.0:
                return
            self._last_ml_inference[symbol] = now
            
        history = list(history)
        if not trigger_trade and current_candle:
            history.append(dict(current_candle))
        
        # Gate 0: Pre-Warmup Guard
        if len(history) < 250:
            print(f"[Pipeline Gate 0] [WARMUP_INCOMPLETE] Blocking ML inference for {symbol}. Candles: {len(history)}/250")
            if 'DIAG' in globals() and DIAG:
                DIAG.record(
                    symbol=symbol, strategy="ALL",
                    bar_ts=history[-1].get('open_time', 0) if history else 0,
                    strat_triggered=False,
                    p_long=0.0, p_short=0.0, threshold=0.0,
                    skipped_backlog=False, skipped_duplicate=False, skipped_cooldown=False,
                    trade_sent=False
                )
            return
        
        df = pd.DataFrame(history)
        df['ts'] = pd.to_datetime(df['open_time'], unit='s')
        rename_map = {
            'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume',
            'fut_cvd':'CVD', 'oi':'Agg. OI', 'funding':'Agg. Funding Rate', 'ls_ratio':'Long/Short Ratio (Account)',
            'liq_long':'Agg. Liq Long', 'liq_short':'Agg. Liq Short',
            'coins_bid':'Bid Qty', 'coins_ask':'Ask Qty',
            'tk_buy_cnt':'Bid Trades', 'tk_sell_cnt':'Ask Trades'
        }
        df.rename(columns=rename_map, inplace=True)
        if 'Bid Qty' in df.columns and 'Ask Qty' in df.columns:
            df['Bid Qty'] = pd.to_numeric(df['Bid Qty'], errors='coerce').fillna(0.0).abs()
            df['Ask Qty'] = pd.to_numeric(df['Ask Qty'], errors='coerce').fillna(0.0).abs()
            df['Delta Qty'] = df['Bid Qty'] - df['Ask Qty']
        if 'Bid Trades' in df.columns and 'Ask Trades' in df.columns:
            df['Bid Trades'] = pd.to_numeric(df['Bid Trades'], errors='coerce').fillna(0.0).abs()
            df['Ask Trades'] = pd.to_numeric(df['Ask Trades'], errors='coerce').fillna(0.0).abs()
        if 'Agg. Liq Long' in df.columns and 'Agg. Liq Short' in df.columns:
            df['Agg. Liq Long'] = pd.to_numeric(df['Agg. Liq Long'], errors='coerce').fillna(0.0).abs()
            df['Agg. Liq Short'] = pd.to_numeric(df['Agg. Liq Short'], errors='coerce').fillna(0.0).abs()

        btc_ref = pd.DataFrame()
        if symbol != "BTCUSDT" and "BTCUSDT" in self.candles_history and len(self.candles_history["BTCUSDT"]) > 0:
            with self._lock:
                btc_history = list(self.candles_history["BTCUSDT"])
            if btc_history:
                df_btc = pd.DataFrame(btc_history)
                df_btc.rename(columns=rename_map, inplace=True)
                if 'open_time' in df_btc.columns:
                    df_btc['ts'] = pd.to_datetime(df_btc['open_time'], unit='s')
                    if 'ts' in df_btc.columns:
                        btc_ref['ts'] = df_btc['ts']
                if 'Close' in df_btc.columns:
                    btc_ref['btc_Close'] = df_btc['Close']
                if 'CVD' in df_btc.columns:
                    btc_ref['btc_CVD'] = df_btc['CVD']
        if btc_ref.empty:
            if 'ts' in df.columns:
                btc_ref['ts'] = df['ts']
            btc_ref['btc_Close'] = df['Close']
            btc_ref['btc_CVD'] = df['CVD']

        df_std = generate_features_standard(df.copy(), btc_ref)
        df_vwap, _ = prep_vwap(df.copy(), btc_ref)
        df_micro = prep_microstructure(df.copy())
        df_smc, _ = prep_smc(df.copy(), btc_ref)
        
        if df_std.empty or df_vwap.empty or df_micro.empty or df_smc.empty: return

        # Map missing summary/footprint columns to satisfy trained model features.
        # Use .reset_index(drop=True).values when pulling from df to avoid index-alignment NaNs
        # (df may have a different RangeIndex than target after generate_features_standard).
        _n = len(df)
        _whale = pd.to_numeric(df['whale_idx'], errors='coerce').fillna(0.0).values if 'whale_idx' in df.columns else np.zeros(_n)
        _rsi   = pd.to_numeric(df['rsi'],      errors='coerce').fillna(50.0).values if 'rsi'      in df.columns else np.full(_n, 50.0)
        _busd  = pd.to_numeric(df['dollars_bid'], errors='coerce').fillna(0.0).values if 'dollars_bid' in df.columns else np.zeros(_n)
        _ausd  = pd.to_numeric(df['dollars_ask'], errors='coerce').fillna(0.0).values if 'dollars_ask' in df.columns else np.zeros(_n)

        for target in [df_std, df_vwap, df_micro, df_smc]:
            nt = len(target)
            target['Candle'] = np.arange(nt)
            target['Buy Qty'] = target['Bid Qty'].values if 'Bid Qty' in target.columns else 0.0
            target['Sell Qty'] = target['Ask Qty'].values if 'Ask Qty' in target.columns else 0.0
            target['Candle Delta'] = target['Buy Qty'] - target['Sell Qty']
            target['Whale Ind'] = _whale[-nt:]
            target['RSI']       = _rsi[-nt:]
            target['Net Shorts'] = 0.0
            target['Net Longs']  = 0.0
            target['Price Low']  = target['Low'].values  if 'Low'  in target.columns else 0.0
            target['Price High'] = target['High'].values if 'High' in target.columns else 0.0
            target['Mid Price']  = (target['Price High'] + target['Price Low']) / 2.0
            target['Bid USD']    = _busd[-nt:]
            target['Ask USD']    = _ausd[-nt:]
            target['Delta USD']  = target['Bid USD'] - target['Ask USD']
            target['total_qty']  = target['Buy Qty'] + target['Sell Qty']
        
        last_std = df_std.iloc[-1:]
        last_vwap = df_vwap.iloc[-1:]
        
        atr_val = float(last_std['atr'].iloc[0])
        if atr_val > 0 and not np.isnan(atr_val):
            self.latest_atr[symbol] = atr_val

        if not last_vwap.empty and 'vwap' in last_vwap.columns:
            vwap_val = float(last_vwap['vwap'].iloc[0])
            vwap_u = float(last_vwap['v_upper_20'].iloc[0]) if 'v_upper_20' in last_vwap.columns else vwap_val
            vwap_l = float(last_vwap['v_lower_20'].iloc[0]) if 'v_lower_20' in last_vwap.columns else vwap_val
            
            if not hasattr(self, 'latest_vwap'): self.latest_vwap = {}
            if not hasattr(self, 'latest_vwap_upper'): self.latest_vwap_upper = {}
            if not hasattr(self, 'latest_vwap_lower'): self.latest_vwap_lower = {}
            
            self.latest_vwap[symbol] = vwap_val
            self.latest_vwap_upper[symbol] = vwap_u
            self.latest_vwap_lower[symbol] = vwap_l
            
        EQUITY_MA_WINDOW = 5
        equity_ma = sum(self.recent_capitals[-EQUITY_MA_WINDOW:]) / max(1, min(len(self.recent_capitals), EQUITY_MA_WINDOW))
        current_capital = trade_tracker.current_capital if trade_tracker else equity_ma
        equity_deviation = (equity_ma - current_capital) / equity_ma * 100.0 if equity_ma > 0 else 0.0

        if equity_deviation > 2.5:
            return

        risk_mult = 0.5 if equity_deviation > 1.5 else 1.0
        
        with self._lock:
            if symbol not in self._cached_signal:
                self._cached_signal[symbol] = {'ml_signals': {}}
            elif 'ml_signals' not in self._cached_signal[symbol]:
                self._cached_signal[symbol]['ml_signals'] = {}
            
        for strat in self.strategies:
            if symbol not in self.models[strat]: continue
            model = self.models[strat][symbol]
            cols = self.features_cols[strat][symbol]
            
            target_df = last_vwap if strat == 'ML_Vwap_Reversal' else last_std
            
            # Compute probabilities for visualization
            p_long = 0.5
            p_short = 0.5
            
            target_df_long = target_df.copy()
            target_df_long['direction'] = 1
            missing_long = [c for c in cols if c not in target_df_long.columns]
            if not missing_long:
                try:
                    X_long = target_df_long[cols].astype(np.float32)
                    p_long = float(model.predict_proba(X_long)[0, 1])
                except Exception:
                    pass
                    
            target_df_short = target_df.copy()
            target_df_short['direction'] = -1
            missing_short = [c for c in cols if c not in target_df_short.columns]
            if not missing_short:
                try:
                    X_short = target_df_short[cols].astype(np.float32)
                    p_short = float(model.predict_proba(X_short)[0, 1])
                except Exception:
                    pass
                    
            max_prob = max(p_long, p_short)
            direction_name = 'Long' if p_long > p_short else 'Short'
            
            strat_sym_key = f"{strat}_{symbol}"
            m_data = self.manifest_data.get(strat_sym_key, {})
            prob_threshold = m_data.get("prob_threshold", 0.6)
            
            # Cache the signal
            with self._lock:
                self._cached_signal[symbol]['ml_signals'][strat] = {
                    'prob_score': max_prob,
                    'trigger_threshold': prob_threshold,
                    'key_feature': f"Direction",
                    'key_feature_val': 1.0 if direction_name == 'Long' else -1.0
                }
            
            # Entry logic (Only trigger on bar bounds)
            if trigger_trade:
                for direction in [1, -1]:
                    target_df_dir = target_df.copy()
                    target_df_dir['direction'] = direction
                    missing = [c for c in cols if c not in target_df_dir.columns]
                    if missing: continue
                    X = target_df_dir[cols].astype(np.float32)
                    
                    triggered = False
                    row = target_df_dir.iloc[0]
                    s_params = m_data.get('s_params', {})
                    if strat == 'S1_Liquidation':
                        pull8 = row.get('pull_ema8', 0)
                        ll = row.get('liq_long_5', 0)
                        ls = row.get('liq_short_5', 0)
                        llm = row.get('liq_long_5_mean', 1)
                        lsm = row.get('liq_short_5_mean', 1)
                        t_liq = s_params.get('t_liq', 2.0)
                        if direction == 1 and pull8 < -0.2 and ll > 0 and llm > 0 and ll >= llm * t_liq: triggered = True
                        if direction == -1 and pull8 > 0.2 and ls > 0 and lsm > 0 and ls >= lsm * t_liq: triggered = True
                    elif strat == 'S2_CVD':
                        z20 = row.get('z_cvd_20', 0)
                        z4 = row.get('z_cvd_4', 0)
                        mac = row.get('macro', 0)
                        t_cvd = s_params.get('t_cvd', 1.5)
                        t_cvd_fast = s_params.get('t_cvd_fast', 0.5)
                        if direction == 1 and z20 >= 0.3 and z20 >= t_cvd and z4 >= t_cvd_fast and mac >= 0: triggered = True
                        if direction == -1 and z20 <= -0.3 and z20 <= -t_cvd and z4 <= -t_cvd_fast and mac <= 0: triggered = True
                    elif strat == 'S3_Trend':
                        pull = row.get('pull_ema8', 0)
                        mac = row.get('macro', 0)
                        rsi = row.get('rsi', 50)
                        t_pull = s_params.get('t_pull', 0.5)
                        t_rsi = s_params.get('t_rsi', 45)
                        if direction == 1 and mac > 0 and pull < -0.1 and pull < -t_pull and rsi < t_rsi: triggered = True
                        if direction == -1 and mac < 0 and pull > 0.1 and pull > t_pull and rsi > 100 - t_rsi: triggered = True
                    elif strat == 'ML_Vwap_Reversal':
                        z20 = row.get('z_cvd_20', 0)
                        vol = row.get('vol_regime', 0)
                        t_z20 = s_params.get('t_z20', 1.5)
                        t_vol = s_params.get('t_vol', 0.5)
                        
                        low = row.get('Low', 0)
                        high = row.get('High', 0)
                        c = row.get('Close', 0)
                        v_u = row.get('v_upper_20', c)
                        v_l = row.get('v_lower_20', c)
                        vwap = row.get('vwap', c)
                        rsi = row.get('rsi', 50)
                        ef = row.get('ema_fast', c)
                        es = row.get('ema_slow', c)
                        a = row.get('atr', 0)
                        
                        base_long = (low <= v_l and rsi < 45) or (low <= vwap + 0.3 * a and ef > es)
                        base_short = (high >= v_u and rsi > 55) or (high >= vwap - 0.3 * a and ef < es)
                        
                        if direction == 1 and base_long and z20 <= -t_z20 and vol >= t_vol: triggered = True
                        if direction == -1 and base_short and z20 >= t_z20 and vol >= t_vol: triggered = True
                    elif strat == 'S5_Microstructure':
                        vol_reg = row.get('vol_regime', 0)
                        delta_z = row.get('delta_qty_z', 0)
                        z_bid = row.get('z_bid_qty', 0)
                        z_ask = row.get('z_ask_qty', 0)
                        t_vol = s_params.get('t_vol', 1.0)
                        t_delta = s_params.get('t_delta', 1.0)
                        if vol_reg <= t_vol:
                            if direction == 1 and delta_z > 0.5 and z_bid > z_ask and delta_z >= t_delta: triggered = True
                            if direction == -1 and delta_z < -0.5 and z_ask > z_bid and delta_z <= -t_delta: triggered = True
                    elif strat == 'S6_SMC_Orderflow':
                        z_delta = row.get('z_delta', 0)
                        stk = row.get('stoch_k', 50)
                        p50 = row.get('pull_ema50', 0)
                        cvd_d = row.get('cvd_delta', 0)
                        t_delta = s_params.get('t_delta', 1.0)
                        if direction == 1 and stk < 20 and p50 < 0 and cvd_d > 0 and z_delta >= t_delta: triggered = True
                        if direction == -1 and stk > 80 and p50 > 0 and cvd_d < 0 and z_delta <= -t_delta: triggered = True
                    
                    if not triggered:
                        if 'DIAG' in globals() and DIAG:
                            DIAG.record(
                                symbol=symbol, strategy=strat,
                                bar_ts=row.get('ts').timestamp() if hasattr(row.get('ts'), 'timestamp') else time.time(),
                                strat_triggered=False,
                                p_long=round(float(p_long), 4) if direction == 1 else 0.0, 
                                p_short=round(float(p_short), 4) if direction == -1 else 0.0,
                                threshold=float(prob_threshold),
                                skipped_backlog=False, skipped_duplicate=False, skipped_cooldown=False,
                                trade_sent=False
                            )
                        continue
                    
                    try:
                        prob = p_long if direction == 1 else p_short
                        tp_mult = m_data.get("tp_mult", TP_MULT_OPTIONS[0])
                        trail_act = m_data.get("trail_atr", TRAIL_ATR_OPTIONS[0])
                        sl_mult = 1.0
                        
                        if prob >= prob_threshold:
                            sl = current_price - sl_mult * atr_val if direction == 1 else current_price + sl_mult * atr_val
                            tp = current_price + tp_mult * atr_val if direction == 1 else current_price - tp_mult * atr_val
                            
                            if trade_tracker:
                                # Gate 5.5 - State Freshness Check
                                if trade_tracker and hasattr(trade_tracker, 'last_reconcile_ts'):
                                    if time.time() - trade_tracker.last_reconcile_ts > 90:
                                        print(f"[Pipeline Gate 5.5] Forcing sync reconcile_with_mt5 for {symbol} due to stale state (>90s).")
                                        trade_tracker.reconcile_with_mt5()

                                strategy_trades = [t for t in trade_tracker.active_trades.values() if t.get('strategy') == strat]
                                duplicate_exists = any(t.get('symbol') == symbol for t in strategy_trades)
                                cool_key = trade_tracker._cooldown_key(strat, symbol) if hasattr(trade_tracker, '_cooldown_key') else f"{strat}:{symbol}"
                                cooldown_until = getattr(trade_tracker, 'reentry_cooldown_until', {}).get(cool_key, 0.0)
                                in_cooldown = time.time() < cooldown_until

                                if 'DIAG' in globals() and DIAG:
                                    DIAG.record(
                                        symbol=symbol, strategy=strat,
                                        bar_ts=row.get('ts').timestamp() if hasattr(row.get('ts'), 'timestamp') else time.time(),
                                        strat_triggered=True,
                                        p_long=round(float(p_long), 4) if direction == 1 else 0.0,
                                        p_short=round(float(p_short), 4) if direction == -1 else 0.0,
                                        threshold=float(prob_threshold),
                                        skipped_backlog=False,
                                        skipped_duplicate=duplicate_exists,
                                        skipped_cooldown=in_cooldown,
                                        trade_sent=(not duplicate_exists) and (not in_cooldown)
                                    )

                                trade_tracker.trigger_entry(
                                    symbol, strat, direction, current_price, sl, tp, atr_val, macro=0,
                                    vol_regime=0, risk_mult=risk_mult, trail_act=trail_act, regime_val=0
                                )
                        else:
                            if 'DIAG' in globals() and DIAG:
                                DIAG.record(
                                    symbol=symbol, strategy=strat,
                                    bar_ts=row.get('ts').timestamp() if hasattr(row.get('ts'), 'timestamp') else time.time(),
                                    strat_triggered=True,
                                    p_long=round(float(p_long), 4) if direction == 1 else 0.0,
                                    p_short=round(float(p_short), 4) if direction == -1 else 0.0,
                                    threshold=float(prob_threshold),
                                    skipped_backlog=False,
                                    skipped_duplicate=False,
                                    skipped_cooldown=False,
                                    trade_sent=False
                                )
                    except Exception as e:
                        print(f'Error in unified inference {strat} {symbol}: {e}')

    def set_history(self, symbol: str, candle_list: List[Dict[str, Any]]) -> None:
        if not candle_list:
            return
        now_open = int(time.time() // 900) * 900
        clean_list = []
        for c in candle_list:
            row = dict(c)
            ot = row.get("open_time", 0)
            if isinstance(ot, (int, float)):
                ot_sec = int(ot // 1000 if ot > 1e11 else ot)
            else:
                ot_sec = 0
            # Keep only closed bars
            if ot_sec > 0 and ot_sec < now_open:
                row["open_time"] = ot_sec
                clean_list.append(row)
        clean_list.sort(key=lambda x: x["open_time"])
        clean_list = clean_list[-1200:]
        with self._lock:
            self.candles_history[symbol] = collections.deque(clean_list, maxlen=1200)
            if clean_list:
                self._last_predict_bar[symbol] = clean_list[-1]["open_time"]

    def record_closed_capital(self, capital: float) -> None:
        with self._lock:
            self.recent_capitals.append(capital)
            if len(self.recent_capitals) > 50:
                self.recent_capitals = self.recent_capitals[-50:]
