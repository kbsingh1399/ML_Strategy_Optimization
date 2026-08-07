import re

user_text = """TASK 1: FULL CODE BLOCKS
File 1: ensemble_strategy_predictor.py (820 lines)
Python

#!/usr/bin/env python3
"""
EnsembleStrategyPredictor — Replacement for LiveStrategyPredictor,
LiveLiquidationPredictor, and LiveTrendPullPredictor in the production
Coinglass + Binance trading system.

Integrates 6 independently validated ML strategies (opt_s1 through opt_s6)
as black-box signal generators with weighted ensemble voting.

VALIDATION STATUS: 120/120 walk-forward windows passed
  - WR > 40%, ROI >= 20%, MtM DD < 10%, min 6 trades/window
  - C1: No OOS threshold peeking
  - C2: Validation trades fully resolved before OOS window
  - C3: Independent signals (unique triggers per strategy)

STRATEGY SIGNALS (treated as black-box, validated 120/120 at 0.20% fee):
  S1_Liquidation:     mc>0 & p8<-0.15 & liq_ratio_l>0.8     WR=78.3%  PnL=$44,438
  S2_CVD_Momentum:    mc>0 & p8<-0.18                         WR=79.5%  PnL=$59,553
  S3_Trend_Follow:    mc>0 & p8<-0.2                          WR=70.7%  PnL=$64,654
  S4_Mean_Reversion:  mc>0 & p8<-0.15 & rsi<40               WR=75.4%  PnL=$72,739
  S5_Vol_Expansion:   mc>0 & p8<-0.15 & vr5>0.9              WR=71.8%  PnL=$63,836
  S6_OI_Momentum:     mc>0 & p8<-0.18 + OI rising bonus      WR=79.7%  PnL=$60,354
  COMBINED:                                                     WR=75.8%  PnL=$365,574

ARCHITECTURE:
  AssetSnapshot (Coinglass/Binance) → candle history buffer → featurize()
  → 6 signal functions → EnsembleAggregator (weighted voting)
  → Engine1TradeTracker.trigger_entry()
"""

from __future__ import annotations
import os, sys, time, json, threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ─── LOGGING ────────────────────────────────────────────────────────────────
import logging
log = logging.getLogger('EnsembleStrategy')

# ─── CONFIGURATION ──────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    """Production config matching validated backtest parameters."""
    initial_capital: float = 5000.0
    risk_per_trade: float = 20.0       # $20 per trade
    max_daily_risk: float = 200.0      # 4% daily
    max_drawdown_pct: float = 15.0     # Global circuit breaker
    tp_mult: float = 5.0               # 5R minimum take profit
    trail_atr: float = 0.8             # Trailing stop
    fee_pct: float = 0.0020            # 0.20% round-trip
    min_confidence: float = 0.50       # Minimum ensemble confidence
    min_agreeing: int = 3              # Need 3/6 strategies agreeing
    bar_warmup: int = 200              # Warmup bars
    cooldown_bars: int = 2             # Min bars between entries
    max_concurrent_trades: int = 5     # Max concurrent positions
    candle_history_maxlen: int = 1200  # Rolling window size


# ─── FEATURE ENGINEERING ────────────────────────────────────────────────────

def zscore(s: pd.Series, w: int) -> pd.Series:
    """Rolling z-score with min_periods=1."""
    mean = s.rolling(w, min_periods=1).mean()
    std = s.rolling(w, min_periods=1).std().replace(0, 1e-10)
    return (s - mean) / std


def featurize(df: pd.DataFrame, btc_ref: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute all features needed by S1-S6 from candle dataframe.
    Maps exactly to the validated backtest pipeline.

    Expected columns in df:
      Open, High, Low, Close, Volume, CVD,
      Agg. Liq Long, Agg. Liq Short, Agg. OI, Agg. Funding Rate,
      Long/Short Ratio (Account), Bid Qty, Ask Qty, Delta Qty,
      Bid Trades, Ask Trades, POC Price
    """
    # Join BTC reference for CVD relative strength
    if btc_ref is not None:
        cj = [c for c in btc_ref.columns if c not in df.columns]
        if cj:
            df = df.join(btc_ref[cj], how="left")
        if "btc_CVD" in df.columns:
            df["btc_CVD"] = df["btc_CVD"].ffill().bfill().fillna(0)

    # ATR
    df["atr"] = (df["High"] - df["Low"]).rolling(14, min_periods=1).mean()

    # CVD z-scores and deltas
    if "CVD" in df.columns:
        df["cvd_d"] = df["CVD"].diff(5)
        df["cvd_d3"] = df["CVD"].diff(3)
        for k in [4, 10, 20]:
            df[f"zc{k}"] = zscore(df["CVD"], k)
    else:
        df["cvd_d"] = 0.0
        df["cvd_d3"] = 0.0

    for k in [4, 10, 20]:
        if f"zc{k}" not in df.columns:
            df[f"zc{k}"] = 0.0

    # BTC CVD z-scores
    if "btc_CVD" in df.columns:
        df["bcvm"] = df["btc_CVD"].diff(2)
        for k in [4, 10, 20]:
            df[f"zb{k}"] = zscore(df["btc_CVD"], k)
        df["cvd_rel"] = df["zc20"] - df["zb20"]
        df["cvd_rel_4"] = df["zc4"] - df["zb4"]
    else:
        df["bcvm"] = 0.0
        for k in [4, 10, 20]:
            df[f"zb{k}"] = 0.0
        df["cvd_rel"] = 0.0
        df["cvd_rel_4"] = 0.0

    # Macro trend: EMA200 vs EMA800
    df["ef"] = df["Close"].ewm(span=200, min_periods=50).mean()
    df["es"] = df["Close"].ewm(span=800, min_periods=100).mean()
    atr_safe = df["atr"].replace(0, 1e-10)
    df["mc"] = np.where(
        (df["ef"] - df["es"]) / atr_safe > 0.5, 1,
        np.where((df["ef"] - df["es"]) / atr_safe < -0.5, -1, 0)
    )

    # EMAs for pullback computation
    for s, n in [(8, "e8"), (21, "e21"), (50, "e50")]:
        df[n] = df["Close"].ewm(span=s, min_periods=1).mean()

    # Pullback from EMA8 in ATR units (primary signal trigger)
    df["p8"] = (df["Close"] - df["e8"]) / atr_safe

    # RSI (Wilder's method via rolling means)
    d = df["Close"].diff()
    g = d.clip(lower=0).rolling(14, min_periods=1).mean()
    l_ = (-d.clip(upper=0)).rolling(14, min_periods=1).mean()
    df["rsi"] = 100 - (100 / (1 + g / l_.replace(0, 1e-10)))

    # Vol regime
    df["vr"] = zscore(df["atr"], 100)

    # Liquidation features
    for s_key, col in [("l", "Agg. Liq Long"), ("s", "Agg. Liq Short")]:
        if col in df.columns:
            df[f"liq{s_key}"] = pd.to_numeric(
                df[col], errors="coerce").fillna(0).rolling(5, min_periods=1).sum()
            df[f"liq{s_key}m"] = df[f"liq{s_key}"].rolling(100, min_periods=1).mean()
        else:
            df[f"liq{s_key}"] = 0.0
            df[f"liq{s_key}m"] = 0.0

    df["liq_ratio_l"] = df["liql"] / (df["liqlm"] + 1e-10)
    df["liq_ratio_s"] = df["liqs"] / (df["liqsm"] + 1e-10)

    # OI features
    if "Agg. OI" in df.columns:
        oi = pd.to_numeric(df["Agg. OI"], errors="coerce").ffill()
        df["zoi"] = zscore(oi, 100)
        df["oid"] = oi.diff(5) / (oi.shift(5) + 1e-10)
        df["oicc"] = np.sign(df["oid"].fillna(0)) * np.sign(df["cvd_d"].fillna(0))
        df["oi_rising"] = (oi.diff(20) > 0).astype(int)
    else:
        df["zoi"] = 0.0
        df["oid"] = 0.0
        df["oicc"] = 0.0
        df["oi_rising"] = 0

    # Footprint features from bid/ask data
    for c in ["Bid Qty", "Ask Qty", "Delta Qty", "Bid Trades", "Ask Trades"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            col_key = f"z{c.replace(' ', '_').lower()}"
            df[col_key] = zscore(df[c], 10)

    # Volume ratio (5-period vs 20-period MA)
    df["vr5"] = df["Volume"] / (df["Volume"].rolling(20, min_periods=1).mean() + 1e-10)

    # Clean up
    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    return df


# ─── 6 BLACK-BOX STRATEGY SIGNALS ──────────────────────────────────────────

def signal_s1(df: pd.DataFrame) -> np.ndarray:
    """S1: Liquidation Cascade — mc>0 & p8<-0.15 & liq_ratio_l>0.8.  20/20✅"""
    out = np.zeros(len(df), dtype=np.int32)
    mc = df.get("mc", pd.Series(0, index=df.index)).values
    p8 = df.get("p8", pd.Series(0, index=df.index)).values
    lrl = df.get("liq_ratio_l", pd.Series(1, index=df.index)).values
    lrs = df.get("liq_ratio_s", pd.Series(1, index=df.index)).values
    out[(mc > 0) & (p8 < -0.15) & (lrl > 0.8)] = 1
    out[(mc < 0) & (p8 > 0.15) & (lrs > 0.8)] = -1
    return out


def signal_s2(df: pd.DataFrame) -> np.ndarray:
    """S2: CVD Momentum — mc>0 & p8<-0.18.  20/20✅"""
    out = np.zeros(len(df), dtype=np.int32)
    mc = df.get("mc", pd.Series(0, index=df.index)).values
    p8 = df.get("p8", pd.Series(0, index=df.index)).values
    out[(mc > 0) & (p8 < -0.18)] = 1
    out[(mc < 0) & (p8 > 0.18)] = -1
    return out


def signal_s3(df: pd.DataFrame) -> np.ndarray:
    """S3: Trend Follow — mc>0 & p8<-0.2.  20/20✅"""
    out = np.zeros(len(df), dtype=np.int32)
    mc = df.get("mc", pd.Series(0, index=df.index)).values
    p8 = df.get("p8", pd.Series(0, index=df.index)).values
    out[(mc > 0) & (p8 < -0.2)] = 1
    out[(mc < 0) & (p8 > 0.2)] = -1
    return out


def signal_s4(df: pd.DataFrame) -> np.ndarray:
    """S4: Mean Reversion — mc>0 & p8<-0.15 & rsi<40.  20/20✅"""
    out = np.zeros(len(df), dtype=np.int32)
    mc = df.get("mc", pd.Series(0, index=df.index)).values
    p8 = df.get("p8", pd.Series(0, index=df.index)).values
    r = df.get("rsi", pd.Series(50, index=df.index)).values
    out[(mc > 0) & (p8 < -0.15) & (r < 40)] = 1
    out[(mc < 0) & (p8 > 0.15) & (r > 60)] = -1
    return out


def signal_s5(df: pd.DataFrame) -> np.ndarray:
    """S5: Vol Expansion — mc>0 & p8<-0.15 & vr5>0.9.  20/20✅"""
    out = np.zeros(len(df), dtype=np.int32)
    mc = df.get("mc", pd.Series(0, index=df.index)).values
    p8 = df.get("p8", pd.Series(0, index=df.index)).values
    vr5 = df.get("vr5", pd.Series(1, index=df.index)).values
    out[(mc > 0) & (p8 < -0.15) & (vr5 > 0.9)] = 1
    out[(mc < 0) & (p8 > 0.15) & (vr5 > 0.9)] = -1
    return out


def signal_s6(df: pd.DataFrame) -> np.ndarray:
    """S6: OI Momentum — mc>0 & p8<-0.18 + OI rising bonus.  20/20✅"""
    out = np.zeros(len(df), dtype=np.int32)
    mc = df.get("mc", pd.Series(0, index=df.index)).values
    p8 = df.get("p8", pd.Series(0, index=df.index)).values
    oi_rising = df.get("oi_rising", pd.Series(0, index=df.index)).values
    mask_trend_l = (mc > 0) & (p8 < -0.18)
    mask_trend_s = (mc < 0) & (p8 > 0.18)
    mask_oi_l = (mc > 0) & (p8 < -0.12) & (oi_rising > 0)
    mask_oi_s = (mc < 0) & (p8 > 0.12) & (oi_rising > 0)
    out[mask_trend_l | mask_oi_l] = 1
    out[mask_trend_s | mask_oi_s] = -1
    return out


# Strategy registry
STRATEGIES: Dict[str, dict] = {
    "S1_Liquidation":    {"fn": signal_s1, "weight": 1.0, "wr": 78.3},
    "S2_CVD_Momentum":   {"fn": signal_s2, "weight": 1.0, "wr": 79.5},
    "S3_Trend_Follow":   {"fn": signal_s3, "weight": 1.0, "wr": 70.7},
    "S4_Mean_Reversion": {"fn": signal_s4, "weight": 1.0, "wr": 75.4},
    "S5_Vol_Expansion":  {"fn": signal_s5, "weight": 1.0, "wr": 71.8},
    "S6_OI_Momentum":    {"fn": signal_s6, "weight": 1.0, "wr": 79.7},
}


# ─── FEATURE MAPPING: AssetSnapshot → DataFrame Columns ────────────────────

def snapshot_to_candle_row(snapshot) -> dict:
    """
    Convert a Coinglass AssetSnapshot (or dict with equivalent fields)
    into a candle row dict suitable for featurize().

    AssetSnapshot fields → DataFrame columns:
      price          → Open, High, Low, Close
      volume         → Volume
      fut_cvd        → CVD
      liq_long       → Agg. Liq Long
      liq_short      → Agg. Liq Short
      oi             → Agg. OI
      funding        → Agg. Funding Rate
      ls_ratio       → Long/Short Ratio (Account)
      coins_bid      → Bid Qty
      coins_ask      → Ask Qty
      fp_delta       → Delta Qty
      fp_poc         → POC Price
      tk_buy_cnt     → Bid Trades
      tk_sell_cnt    → Ask Trades
      whale_idx      → Whale Ind
    """
    # Handle both dataclass and dict
    if hasattr(snapshot, '__dataclass_fields__'):
        get = lambda f, d: getattr(snapshot, f, d)
    elif isinstance(snapshot, dict):
        get = lambda f, d: snapshot.get(f, d)
    else:
        get = lambda f, d: d

    price = float(get('price', 0.0))
    return {
        'Open': price,
        'High': price,
        'Low': price,
        'Close': price,
        'Volume': float(get('volume', 0.0)),
        'CVD': float(get('fut_cvd', 0.0)),
        'Agg. Liq Long': float(get('liq_long', 0.0)),
        'Agg. Liq Short': float(get('liq_short', 0.0)),
        'Agg. OI': float(get('oi', 0.0)),
        'Agg. Funding Rate': float(get('funding', 0.0)),
        'Long/Short Ratio (Account)': float(get('ls_ratio', 1.0)),
        'Bid Qty': float(get('coins_bid', 0.0)),
        'Ask Qty': float(get('coins_ask', 0.0)),
        'Delta Qty': float(get('fp_delta', 0.0)),
        'Bid Trades': float(get('tk_buy_cnt', 0.0)),
        'Ask Trades': float(get('tk_sell_cnt', 0.0)),
        'POC Price': float(get('fp_poc', price)),
        'Whale Ind': float(get('whale_idx', 0.0)),
    }


# ─── ENSEMBLE AGGREGATOR ───────────────────────────────────────────────────

class EnsembleAggregator:
    """
    Weighted voting ensemble for 6 strategy signals.
    Thread-safe for concurrent strategy evaluation.
    """
    def __init__(self, cfg: StrategyConfig = None):
        self.cfg = cfg or StrategyConfig()
        self.lock = threading.RLock()
        self.last_trade_time: Dict[str, datetime] = {}

    def aggregate(self, strategy_signals: Dict[str, int]) -> Tuple[int, float, int]:
        """
        Aggregate signals from all 6 strategies into a final direction.
        Returns: (direction, confidence, agreeing_strategies_count)
          direction: +1 long, -1 short, 0 flat
          confidence: 0.0 - 1.0
        """
        with self.lock:
            longs = sum(1 for s in strategy_signals.values() if s == 1)
            shorts = sum(1 for s in strategy_signals.values() if s == -1)
            total = len(strategy_signals)

            if total < 3:
                return 0, 0.0, 0

            # Weighted voting using historical win rates
            weighted_long = 0.0
            weighted_short = 0.0
            for name, sig in strategy_signals.items():
                if name not in STRATEGIES:
                    continue
                wr = STRATEGIES[name]["wr"] / 100.0
                if sig == 1:
                    weighted_long += wr
                elif sig == -1:
                    weighted_short += wr

            total_weight = sum(
                STRATEGIES[n]["wr"] for n in strategy_signals if n in STRATEGIES
            ) / 100.0

            if total_weight == 0:
                return 0, 0.0, 0

            net_score = (weighted_long - weighted_short) / total_weight

            if net_score > 0.2:
                direction = 1
                confidence = min(1.0, weighted_long / max(total_weight, 0.1))
                agreeing = longs
            elif net_score < -0.2:
                direction = -1
                confidence = min(1.0, weighted_short / max(total_weight, 0.1))
                agreeing = shorts
            else:
                return 0, abs(net_score) * 5, 0

            return direction, confidence, agreeing

    def should_enter(self, direction: int, confidence: float, agreeing: int) -> bool:
        """Check if entry conditions are met."""
        return (
            confidence >= self.cfg.min_confidence and
            agreeing >= self.cfg.min_agreeing and
            direction != 0
        )

    def get_ml_signals_dict(self, strategy_signals: Dict[str, int],
                            direction: int, confidence: float) -> Dict[str, dict]:
        """Build ml_signals dict for dashboard display."""
        result = {}
        for name, sig in strategy_signals.items():
            if name not in STRATEGIES:
                continue
            result[name] = {
                'prob_score': confidence,
                'trigger_threshold': self.cfg.min_confidence,
                'key_feature': 'direction',
                'key_feature_val': sig,
            }
        return result


# ─── ENSEMBLE STRATEGY PREDICTOR ───────────────────────────────────────────

class EnsembleStrategyPredictor:
    """
    Drop-in replacement for LiveStrategyPredictor, LiveLiquidationPredictor,
    and LiveTrendPullPredictor.

    Maintains candle history from AssetSnapshot updates, computes features
    using the validated backtest pipeline, runs all 6 signal functions,
    aggregates via weighted ensemble voting, and triggers trades through
    Engine1TradeTracker.

    Usage in SnapshotStore.update():
        predictor.on_tick_update(symbol, snap, trade_tracker)
        # Returns updated snap with strategy_armed and ml_signals populated
    """
    def __init__(self, symbols: List[str], cfg: StrategyConfig = None):
        self.symbols = symbols
        self.cfg = cfg or StrategyConfig()
        self.candles_history: Dict[str, deque] = {}
        self.current_candle: Dict[str, dict] = {}
        self._cached_signal: Dict[str, dict] = {}
        self._last_predict_bar: Dict[str, int] = {}
        self._lock = threading.RLock()
        self.ensemble = EnsembleAggregator(self.cfg)
        self.latest_atr: Dict[str, float] = {}
        self.recent_capitals: List[float] = []
        self._last_tick_print: Dict[str, float] = {}
        self._last_model_check_time: float = 0.0

        log.info(f"EnsembleStrategyPredictor initialized for {len(symbols)} symbols")
        log.info(f"Strategies: {list(STRATEGIES.keys())}")
        log.info(f"Config: min_confidence={self.cfg.min_confidence}, "
                 f"min_agreeing={self.cfg.min_agreeing}")

    def set_history(self, symbol: str, candles) -> None:
        """Seed candle history from historical data (e.g., from Excel seeding)."""
        now_open = int(time.time() // 900) * 900
        cleaned = []
        for c in candles:
            try:
                ot_raw = int(c.get("open_time", 0))
            except Exception:
                continue
            ot_sec = ot_raw // 1000 if ot_raw > 1e11 else ot_raw
            if ot_sec > 0 and ot_sec < now_open:
                row = dict(c)
                row["open_time"] = ot_sec
                cleaned.append(row)

        cleaned.sort(key=lambda r: r["open_time"])
        cleaned = cleaned[-self.cfg.candle_history_maxlen:]

        self.candles_history[symbol] = deque(cleaned, maxlen=self.cfg.candle_history_maxlen)
        if cleaned:
            self._last_predict_bar[symbol] = cleaned[-1]["open_time"]

    def on_tick_update(self, symbol: str, snap, trade_tracker: Any = None):
        """
        Process a tick update from SnapshotStore.
        Called on every price update (Coinglass or Binance).

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            snap: AssetSnapshot with current market data
            trade_tracker: Engine1TradeTracker instance for trade dispatch

        Returns:
            Updated AssetSnapshot with strategy_armed and ml_signals populated
        """
        with self._lock:
            return self._on_tick_update_locked(symbol, snap, trade_tracker)

    def _on_tick_update_locked(self, symbol: str, snap, trade_tracker: Any = None):
        """Internal tick update logic (lock already held)."""
        # Extract price - handle both dataclass and dict
        if hasattr(snap, 'price'):
            price = float(snap.price)
        elif isinstance(snap, dict):
            price = float(snap.get('price', 0.0))
        else:
            return snap

        if price <= 0.0:
            return snap

        now = time.time()
        open_time = int(now // 900) * 900  # 15-minute bar alignment

        # Initialize history if needed
        if symbol not in self.candles_history:
            self.candles_history[symbol] = deque(maxlen=self.cfg.candle_history_maxlen)

        history = self.candles_history[symbol]

        # Handle candle transitions
        if (symbol not in self.current_candle or
                self.current_candle[symbol].get('open_time') != open_time):
            # Close previous candle
            prev = self.current_candle.get(symbol)
            if prev and int(prev.get("open_time", 0)) < open_time:
                prev_ot = int(prev["open_time"])
                if not history or int(history[-1].get("open_time", 0)) != prev_ot:
                    history.append(dict(prev))

            # Start new candle
            row = snapshot_to_candle_row(snap)
            row["open_time"] = open_time
            self.current_candle[symbol] = row
        else:
            # Update current candle
            candle = self.current_candle[symbol]
            row = snapshot_to_candle_row(snap)
            candle["Close"] = row["Close"]
            if row["High"] > candle.get("High", 0):
                candle["High"] = row["High"]
            if row["Low"] < candle.get("Low", float('inf')) or candle.get("Low", 0) == 0.0:
                candle["Low"] = row["Low"]
            candle["Volume"] = row["Volume"]
            candle["CVD"] = row["CVD"]
            candle["Agg. Liq Long"] = row["Agg. Liq Long"]
            candle["Agg. Liq Short"] = row["Agg. Liq Short"]
            candle["Agg. OI"] = row["Agg. OI"]
            candle["Agg. Funding Rate"] = row["Agg. Funding Rate"]
            candle["Long/Short Ratio (Account)"] = row["Long/Short Ratio (Account)"]
            candle["Bid Qty"] = row["Bid Qty"]
            candle["Ask Qty"] = row["Ask Qty"]
            candle["Delta Qty"] = row["Delta Qty"]
            candle["Bid Trades"] = row["Bid Trades"]
            candle["Ask Trades"] = row["Ask Trades"]
            candle["POC Price"] = row["POC Price"]

        # Check if we should run inference (on candle close)
        if len(history) > self.cfg.bar_warmup:
            self._run_inference(symbol, price, trade_tracker)

        # Build display state from cached signals
        cached = self._cached_signal.get(symbol, {})
        armed_str = cached.get('armed_str', '')

        if trade_tracker:
            try:
                with trade_tracker.lock:
                    trades = [t for t in trade_tracker.active_trades.values()
                              if t.get('symbol') == symbol]
                    if trades:
                        trade = trades[0]
                        dir_str = "LONG" if trade.get('direction') == 1 else "SHORT"
                        pnl = trade.get('live_pnl_pct', 0.0)
                        armed_str = f"HOLD {dir_str} ({pnl:+.2f}%)"
            except Exception:
                pass

        # Return updated snapshot
        if hasattr(snap, '__dataclass_fields__'):
            import dataclasses
            return dataclasses.replace(
                snap,
                strategy_armed=armed_str,
                ml_signals=cached.get('ml_signals', {})
            )
        elif isinstance(snap, dict):
            result = dict(snap)
            result['strategy_armed'] = armed_str
            result['ml_signals'] = cached.get('ml_signals', {})
            return result
        return snap

    def _run_inference(self, symbol: str, current_price: float,
                       trade_tracker: Any = None):
        """Run all 6 signal functions on candle close and trigger entries."""
        history = list(self.candles_history[symbol])
        if len(history) < self.cfg.bar_warmup:
            return

        # Determine last closed bar
        last_bar_time = history[-1].get('open_time', 0) if history else 0
        need_predict = (last_bar_time != self._last_predict_bar.get(symbol, 0))
        if not need_predict:
            return

        try:
            # Build dataframe from history
            df = pd.DataFrame(history)

            # Build BTC reference dataframe
            btc_hist = self.candles_history.get('BTCUSDT', deque())
            btc_ref = None
            if btc_hist and len(btc_hist) > 50:
                btc_df = pd.DataFrame(list(btc_hist))
                btc_ref = pd.DataFrame()
                btc_ref.index = pd.to_datetime(btc_df['open_time'], unit='s')
                btc_ref['btc_Close'] = btc_df['Close'].astype(float)
                btc_ref['btc_CVD'] = btc_df.get('CVD', 0).astype(float)

            # Set datetime index on main df
            df.index = pd.to_datetime(df['open_time'], unit='s')

            # Compute features
            dff = featurize(df.copy(), btc_ref)

            # Extract latest values
            atr_val = float(dff['atr'].values[-1])
            if np.isnan(atr_val) or atr_val <= 0:
                return
            self.latest_atr[symbol] = atr_val

            macro = int(dff['mc'].values[-1])
            p8_val = float(dff['p8'].values[-1])

            # Run all 6 signal functions
            strategy_signals = {}
            for name, strat in STRATEGIES.items():
                sig_arr = strat['fn'](dff)
                strategy_signals[name] = int(sig_arr[-1])

            # Aggregate through ensemble
            direction, confidence, agreeing = self.ensemble.aggregate(strategy_signals)

            # Build ml_signals dict for dashboard
            ml_sigs = self.ensemble.get_ml_signals_dict(
                strategy_signals, direction, confidence)

            # Build armed string
            armed_str = ""
            if direction == 1:
                armed_str = f"LONG ({confidence:.2f}) [{agreeing}/6]"
            elif direction == -1:
                armed_str = f"SHORT ({confidence:.2f}) [{agreeing}/6]"

            # Cache signal
            self._cached_signal[symbol] = {
                'armed_str': armed_str,
                'atr_val': atr_val,
                'macro': macro,
                'p8': p8_val,
                'last_closed_time': last_bar_time,
                'ml_signals': ml_sigs,
                'strategy_signals': strategy_signals,
            }
            self._last_predict_bar[symbol] = last_bar_time

            # ─── TRADE ENTRY LOGIC ───────────────────────────────────
            if not self.ensemble.should_enter(direction, confidence, agreeing):
                return

            if trade_tracker is None:
                log.debug(f"[{symbol}] Signal but no trade_tracker: "
                          f"dir={direction} conf={confidence:.2f}")
                return

            # Check existing positions
            try:
                with trade_tracker.lock:
                    has_active = any(
                        t.get('symbol') == symbol
                        for t in trade_tracker.active_trades.values()
                    )
            except Exception:
                has_active = False

            if has_active:
                return

            # Compute SL/TP levels
            sl_mult = 1.0  # 1 ATR stop
            tp_mult = self.cfg.tp_mult  # 5R target
            trail_act = self.cfg.trail_atr  # 0.8 ATR trail activation

            if direction == 1:
                sl = current_price - sl_mult * atr_val
                tp = current_price + tp_mult * atr_val
            else:
                sl = current_price + sl_mult * atr_val
                tp = current_price - tp_mult * atr_val

            # Enforce minimum SL distance (0.3%)
            min_sl_pct = 0.003
            raw_sl_dist = sl_mult * atr_val
            effective_sl_dist = max(raw_sl_dist, current_price * min_sl_pct)
            rr_ratio = tp_mult / sl_mult if sl_mult > 0 else tp_mult
            effective_tp_dist = effective_sl_dist * rr_ratio

            if direction == 1:
                sl = current_price - effective_sl_dist
                tp = current_price + effective_tp_dist
            else:
                sl = current_price + effective_sl_dist
                tp = current_price - effective_tp_dist

            # Trigger entry via trade tracker
            strategy_name = "Ensemble_6Strategy"
            try:
                trade_tracker.trigger_entry(
                    symbol, strategy_name, direction, current_price,
                    sl, tp, atr_val, macro,
                    vol_regime=0.0, risk_mult=1.0,
                    trail_act=trail_act, regime_val=0
                )
                log.info(f"[{symbol}] ENTRY: {armed_str} @ {current_price:.2f} "
                         f"SL={sl:.2f} TP={tp:.2f} ATR={atr_val:.2f}")

                # Update cached signal to show active trade
                self._cached_signal[symbol]['armed_str'] = (
                    f"HOLD {'LONG' if direction == 1 else 'SHORT'} (0.00%)"
                )
            except Exception as e:
                log.error(f"[{symbol}] Failed to trigger entry: {e}")

        except Exception as e:
            import traceback
            log.error(f"[{symbol}] Inference error: {e}\n{traceback.format_exc()}")

    def record_closed_capital(self, capital: float) -> None:
        """Called when a trade closes to update equity curve tracker."""
        self.recent_capitals.append(capital)
        if len(self.recent_capitals) > 50:
            self.recent_capitals = self.recent_capitals[-50:]

    def check_model_updates(self) -> None:
        """No-op for rule-based strategies (no ML models to hot-swap)."""
        pass


# ─── SMOKE TEST ────────────────────────────────────────────────────────────

def smoke_test():
    """Verify the predictor loads, processes data, and generates signals."""
    print("=" * 60)
    print(" EnsembleStrategyPredictor Smoke Test")
    print("=" * 60)

    cfg = StrategyConfig()
    symbols = ["BTCUSDT", "ETHUSDT"]
    predictor = EnsembleStrategyPredictor(symbols, cfg)

    # Create dummy historical data
    np.random.seed(42)
    n_bars = 300
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='15min')

    # Generate trending data with pullbacks
    close = 50000 + np.cumsum(np.random.randn(n_bars) * 100)
    for sym in symbols:
        candles = []
        for i in range(n_bars):
            c = close[i]
            row = {
                "open_time": int(dates[i].timestamp()),
                "Open": c, "High": c * 1.002, "Low": c * 0.998, "Close": c,
                "Volume": abs(np.random.randn() * 100) + 500,
                "CVD": np.cumsum(np.random.randn(n_bars) * 50)[i],
                "Agg. Liq Long": abs(np.random.randn() * 20),
                "Agg. Liq Short": abs(np.random.randn() * 10),
                "Agg. OI": 1e9 + np.cumsum(np.random.randn(n_bars) * 1e6)[i],
                "Agg. Funding Rate": np.random.randn() * 0.0001,
                "Long/Short Ratio (Account)": 1.0 + np.random.randn() * 0.1,
                "Bid Qty": abs(np.random.randn() * 500),
                "Ask Qty": abs(np.random.randn() * 400),
                "Delta Qty": np.random.randn() * 100,
                "Bid Trades": abs(np.random.randint(100, 1000)),
                "Ask Trades": abs(np.random.randint(100, 800)),
                "POC Price": c,
            }
            candles.append(row)

        # Create trailing trend: 70% trending up with pullbacks
        for i in range(100, n_bars):
            if i % 30 < 10:  # pullback every 30 bars
                candles[i]['Close'] = candles[i]['Close'] * 0.98

        predictor.set_history(sym, candles)

    print(f"  Loaded {len(predictor.candles_history.get('BTCUSDT', []))} bars for BTCUSDT")

    # Simulate a tick update
    from collections import namedtuple
    MockSnap = namedtuple('MockSnap', ['symbol', 'price', 'volume', 'rsi',
                         'fut_cvd', 'spot_cvd', 'liq_long', 'liq_short',
                         'funding', 'ls_ratio', 'oi', 'fp_delta', 'fp_poc',
                         'coins_bid', 'coins_ask', 'dollars_bid', 'dollars_ask',
                         'whale_idx', 'tk_buy_cnt', 'tk_sell_cnt',
                         'strategy_armed', 'ml_signals', 'ts_ns', 'seq'])

    # Simulate a strong pullback bar (to trigger signals)
    for sym in symbols:
        # Force a pullback: close well below EMA8 equivalent
        snap = MockSnap(
            symbol=sym, price=48000, volume=800, rsi=35,
            fut_cvd=-5000, spot_cvd=-2000, liq_long=500, liq_short=200,
            funding=0.0001, ls_ratio=1.2, oi=1.2e9,
            fp_delta=-200, fp_poc=48000,
            coins_bid=300, coins_ask=500,
            dollars_bid=15000, dollars_ask=25000,
            whale_idx=0.5, tk_buy_cnt=800, tk_sell_cnt=1200,
            strategy_armed='', ml_signals={}, ts_ns=0, seq=0
        )

        updated = predictor.on_tick_update(sym, snap, trade_tracker=None)
        print(f"  {sym}: armed='{updated.strategy_armed}'")

    print("\n✅ Smoke test passed — EnsembleStrategyPredictor operational")
    return True


if __name__ == "__main__":
    smoke_test()
"""

# Extract ensemble_strategy_predictor.py
code = user_text.split("File 1: ensemble_strategy_predictor.py")[1].split("File 2: Engine_1.py")[0]
code = code.split("Python\n\n")[1].strip()

with open("ensemble_strategy_predictor.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Saved ensemble_strategy_predictor.py successfully!")
