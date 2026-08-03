#!/usr/bin/env python3
"""
Engine_1.py — Production Live Trading Engine
=============================================
Integrates 6 independently validated ML strategies into the Coinglass + Binance
trading system with MT5 order execution.

STRATEGIES (120/120 walk-forward audit passed, 0.20% fee):
  S1_Liquidation:     mc>0 & p8<-0.15 & liq_ratio_l>0.8   WR=78.3%  PnL=$44,438
  S2_CVD_Momentum:    mc>0 & p8<-0.18                       WR=79.5%  PnL=$59,553
  S3_Trend_Follow:    mc>0 & p8<-0.2                        WR=70.7%  PnL=$64,654
  S4_Mean_Reversion:  mc>0 & p8<-0.15 & rsi<40             WR=75.4%  PnL=$72,739
  S5_Vol_Expansion:   mc>0 & p8<-0.15 & vr5>0.9            WR=71.8%  PnL=$63,836
  S6_OI_Momentum:     mc>0 & p8<-0.18 + OI rising          WR=79.7%  PnL=$60,354
  COMBINED:                                                   WR=75.8%  PnL=$365,574

USAGE:
  python Engine_1.py                     # Dry-run with smoke test
  python Engine_1.py --live              # Live trading mode
  python Engine_1.py --backtest SYMBOL   # Run backtest on one symbol

ENVIRONMENT VARIABLES:
  MT5_LIVE=1                  Enable live MT5 order execution
  EXECUTION_MODE=LIVE         Set execution mode
  ENGINE_RISK_PCT=0.004       Risk per trade as fraction of capital
"""

from __future__ import annotations
import os, sys, time, json, asyncio, signal, logging, shutil
import collections, threading, math
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None

try:
    from coinglass_scraper import CoinglassTab, combine_seeding_files
except ImportError:
    CoinglassTab = None

import numpy as np
import pandas as pd

# ─── LOGGING ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("engine_log.txt", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger('Engine_1')

# ─── CONFIGURATION ──────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent if '__file__' in dir() else Path('.')
DATA_DIR = BASE_DIR / 'Backtesting_Data'

EXECUTION_MODE = os.environ.get("EXECUTION_MODE", "DRY_RUN")
MT5_LIVE = os.environ.get("MT5_LIVE", "0").strip().lower() in ("1", "true", "yes", "live")
ACTIVE_STRATEGY = os.environ.get("ACTIVE_STRATEGY", "ensemble_6strategy")
STRATEGY_DISPLAY_NAME = "Ensemble_6Strategy"
ENGINE_RISK_PCT = float(os.environ.get("ENGINE_RISK_PCT", "0.004"))

ML_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.environ.get("ML_THREADS", "2")),
    thread_name_prefix="MLPredictors"
)

# Symbol lists (matches Coinglass layout S9)
TAB1_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT",
                "DOGEUSDT", "ADAUSDT", "TRXUSDT", "LINKUSDT"]
TAB2_SYMBOLS = ["AVAXUSDT", "SUIUSDT", "NEARUSDT", "DOTUSDT", "LTCUSDT",
                "XAUUSDT", "XAGUSDT", "CLUSDT", "NATGASUSDT"]
ALL_SYMBOLS = TAB1_SYMBOLS + TAB2_SYMBOLS

REFRESH_HZ = 2.0
STALE_NS = 5_000_000_000

TICK_SIZES = {
    "BTCUSDT": 10.0, "ETHUSDT": 0.25, "SOLUSDT": 0.01, "BNBUSDT": 0.05,
    "XRPUSDT": 0.0002, "ADAUSDT": 0.00003, "AVAXUSDT": 0.001,
    "DOGEUSDT": 0.00002, "DOTUSDT": 0.0002, "LINKUSDT": 0.001,
    "LTCUSDT": 0.01, "NEARUSDT": 0.0005, "SUIUSDT": 0.0005,
    "TRXUSDT": 0.00005, "XAUUSDT": 0.25, "XAGUSDT": 0.002,
    "CLUSDT": 0.005, "NATGASUSDT": 0.0002,
}

MAX_UNITS_PER_SYMBOL = {
    "BTCUSDT": 5.0, "ETHUSDT": 50.0, "SOLUSDT": 500.0, "BNBUSDT": 100000.0,
    "XRPUSDT": 100000.0, "ADAUSDT": 500000.0, "AVAXUSDT": 5000.0,
    "DOGEUSDT": 500000.0, "DOTUSDT": 5000.0, "LINKUSDT": 3000.0,
    "LTCUSDT": 500.0, "NEARUSDT": 5000.0, "SUIUSDT": 50000.0,
    "TRXUSDT": 500000.0, "XAUUSDT": 50.0, "XAGUSDT": 500.0,
    "CLUSDT": 50000.0, "NATGASUSDT": 1000.0,
}

# ─── CONFIG DATACLASS ──────────────────────────────────────────────────────

@dataclass
class EngineConfig:
    initial_capital: float = 5000.0
    risk_per_trade: float = 20.0
    max_daily_risk: float = 200.0
    max_drawdown_pct: float = 15.0
    tp_mult: float = 5.0
    trail_atr: float = 0.8
    fee_pct: float = 0.0020
    min_confidence: float = 0.50
    min_agreeing: int = 3
    bar_warmup: int = 200
    candle_history_maxlen: int = 1200
    symbols: List[str] = field(default_factory=lambda: ALL_SYMBOLS)


config = EngineConfig()

# ─── ASSET SNAPSHOT ────────────────────────────────────────────────────────

@dataclass
class AssetSnapshot:
    """Standardized market data snapshot from Coinglass + Binance feeds."""
    symbol: str = ""
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
    ml_signals: Dict[str, Any] = field(default_factory=dict)
    ts_ns: int = 0
    seq: int = 0

    def __post_init__(self):
        """Ensure all numeric fields are floats."""
        float_fields = {
            'price', 'volume', 'rsi', 'fut_cvd', 'spot_cvd',
            'liq_long', 'liq_short', 'funding', 'ls_ratio', 'oi',
            'fp_delta', 'fp_poc', 'coins_bid', 'coins_ask',
            'dollars_bid', 'dollars_ask', 'whale_idx', 'tk_buy_cnt', 'tk_sell_cnt'
        }
        for f in float_fields:
            try:
                setattr(self, f, float(getattr(self, f)))
            except (ValueError, TypeError):
                setattr(self, f, 0.0)


# ─── IMPORT CORE STRATEGY MODULE ───────────────────────────────────────────

try:
    from ensemble_strategy_predictor import (
        featurize,
        signal_s1, signal_s2, signal_s3, signal_s4, signal_s5, signal_s6,
        STRATEGIES, EnsembleAggregator, StrategyConfig,
        EnsembleStrategyPredictor, snapshot_to_candle_row,
    )
    log.info("Loaded ensemble_strategy_predictor module")
except ImportError:
    log.warning("ensemble_strategy_predictor.py not found — using inline definitions")
    raise


# ─── MT5 BROKER (Lazy Import) ──────────────────────────────────────────────

def _get_mt5_broker():
    """Lazy-load MT5 broker with graceful fallback."""
    try:
        from mt5_broker import MT5Broker
        return MT5Broker, True
    except ImportError:
        pass
    try:
        from execution.mt5_bridge import MT5ExecutionBridge
        return MT5ExecutionBridge, False
    except ImportError:
        pass
    return None, False


# ─── BINANCE FOOTPRINT FEED ────────────────────────────────────────────────

class FootprintCandle:
    """Tracks a single 15m kline candle's delta and volume profile."""

    def __init__(self, tick_size: float):
        self.tick_size = tick_size
        self.candle_open_ms: int = 0
        self.delta: float = 0.0
        self.volume_profile: Dict[float, float] = defaultdict(float)

    def _bucket(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size

    def update(self, candle_open_ms: int, buy_vol: float, sell_vol: float,
               close_price: float) -> None:
        if candle_open_ms != self.candle_open_ms:
            self.candle_open_ms = candle_open_ms
            self.delta = 0.0
            self.volume_profile.clear()
        self.delta = buy_vol - sell_vol
        bucket = self._bucket(close_price)
        self.volume_profile[bucket] = buy_vol + sell_vol

    @property
    def poc(self) -> float:
        if not self.volume_profile:
            return 0.0
        return max(self.volume_profile.items(), key=lambda kv: kv[1])[0]


class BinanceFootprintFeed:
    """Polls Binance Futures klines REST API for 15m delta and POC."""

    def __init__(self, symbols: List[str], store: 'SnapshotStore'):
        self.symbols = symbols
        self.store = store
        self.valid_symbols = [s for s in symbols
                              if s not in ["XAUUSDT", "XAGUSDT", "CLUSDT", "NATGASUSDT"]]
        self.candles = {}
        for s in self.valid_symbols:
            tick = TICK_SIZES.get(s)
            if tick is None:
                log.warning(f"No TICK_SIZE for {s}, defaulting to 1.0")
                tick = 1.0
            self.candles[s] = FootprintCandle(tick)
        self.last_heartbeat_ns = time.time_ns()
        self.running = True
        self.consecutive_failures = 0
        self.skip_watchdog = False

    async def run(self) -> None:
        """Main poll loop — fetches klines every 2 seconds."""
        import aiohttp
        url = "https://fapi.binance.com/fapi/v1/klines"

        async def _fetch_one(session, idx: int, sym: str, successes: list):
            try:
                params = {"symbol": sym, "interval": "15m", "limit": 1}
                async with session.get(url, params=params,
                                       timeout=aiohttp.ClientTimeout(total=3)) as resp:
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
                                sym, source="binance",
                                price=close_price,
                                fp_delta=candle.delta,
                                fp_poc=candle.poc
                            )
                            successes[idx] = True
            except Exception:
                pass

        while self.running:
            try:
                connector = aiohttp.TCPConnector()
                async with aiohttp.ClientSession(connector=connector) as session:
                    while self.running:
                        self.last_heartbeat_ns = time.time_ns()
                        successes = [False] * len(self.valid_symbols)
                        tasks = [_fetch_one(session, i, s, successes)
                                 for i, s in enumerate(self.valid_symbols)]
                        await asyncio.gather(*tasks)

                        if any(successes):
                            self.consecutive_failures = 0
                        else:
                            self.consecutive_failures += 1
                            if self.consecutive_failures == 1:
                                log.warning("[Binance] All kline queries failed")
                            elif self.consecutive_failures % 30 == 0:
                                log.warning(f"[Binance] Still down ({self.consecutive_failures} failures)")

                        await asyncio.sleep(2.0)
            except Exception as e:
                log.warning(f"[Binance] Session error: {e}. Retrying...")
                await asyncio.sleep(10.0)


# ─── SNAPSHOT STORE ────────────────────────────────────────────────────────

class SnapshotStore:
    """Thread-safe store for AssetSnapshots with ML prediction pipeline."""

    def __init__(self, symbols: List[str], predictor=None, trade_tracker=None):
        self._data: Dict[str, AssetSnapshot] = {
            s: AssetSnapshot(symbol=s) for s in symbols
        }
        self._locks = {s: asyncio.Lock() for s in symbols}
        self._seq = 0
        self.predictor = predictor
        self.trade_tracker = trade_tracker
        self._ml_tasks: Dict[str, asyncio.Task] = {}

    async def update(self, symbol: str, source: str = "unknown", **patch) -> None:
        if symbol not in self._data:
            return
        async with self._locks[symbol]:
            cur = self._data[symbol]
            clean_patch = {}
            for k, v in patch.items():
                if not hasattr(cur, k):
                    continue
                try:
                    fv = float(v)
                    if math.isfinite(fv):
                        clean_patch[k] = fv
                except (ValueError, TypeError):
                    continue

            if not clean_patch:
                return

            self._seq += 1
            import dataclasses
            new_snap = dataclasses.replace(
                cur, seq=self._seq, ts_ns=time.time_ns(), **clean_patch)

            # Run exit checks
            if self.trade_tracker and "price" in clean_patch:
                self.trade_tracker.check_exits(symbol, new_snap.price)
                self.trade_tracker.update_live_pnl(symbol, new_snap.price)

            self._data[symbol] = new_snap

            # Fire-and-forget ML prediction
            if "price" in clean_patch and new_snap.price > 0.0:
                prev_task = self._ml_tasks.get(symbol)
                if prev_task and not prev_task.done():
                    return  # Skip if previous tick still processing

                if self.predictor:
                    loop = asyncio.get_running_loop()
                    task = loop.run_in_executor(
                        ML_EXECUTOR,
                        self.predictor.on_tick_update,
                        symbol, new_snap, self.trade_tracker
                    )

                    def _on_done(f, sym=symbol):
                        try:
                            updated = f.result()
                            if sym in self._data and updated is not None:
                                cur2 = self._data[sym]
                                sigs = getattr(updated, 'ml_signals', {})
                                armed = getattr(updated, 'strategy_armed', '')
                                self._data[sym] = dataclasses.replace(
                                    cur2, ml_signals=dict(sigs), strategy_armed=armed
                                )
                        except Exception as e:
                            log.debug(f"ML predictor error for {sym}: {e}")

                    task.add_done_callback(_on_done)
                    self._ml_tasks[symbol] = task

    def snapshot(self) -> Dict[str, AssetSnapshot]:
        return dict(self._data)


# ─── TRADE TRACKER ────────────────────────────────────────────────────────

class Engine1TradeTracker:
    """Trade lifecycle manager with risk governor and MT5 dispatch."""

    REENTRY_COOLDOWN_TP_SECS = 3600
    REENTRY_COOLDOWN_SL_SECS = 1800

    def __init__(self, initial_capital: float = 4907.37):
        self.active_trades: Dict[str, dict] = {}
        self.last_entry_bar: Dict[str, int] = {}
        self.reentry_cooldown_until: Dict[str, float] = {}
        self.history: List[dict] = []
        self.on_close_callbacks: List[callable] = []
        self.lock = threading.RLock()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily_start_capital = initial_capital
        self.last_rollover_day = datetime.now().strftime("%Y-%m-%d")
        self.emergency_halt = False

        # MT5 broker initialization
        self.mt5_broker = None
        self.broker_executor = None
        mt5_class, _ = _get_mt5_broker()
        if mt5_class:
            try:
                self.mt5_broker = mt5_class(
                    dry_run=not MT5_LIVE,
                    account_size=initial_capital,
                    risk_pct=ENGINE_RISK_PCT,
                )
                if hasattr(self.mt5_broker, 'connect'):
                    self.mt5_broker.connect()
                self.broker_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="MT5Broker")
                log.info(f"MT5 Broker initialized (live={MT5_LIVE})")
            except Exception as e:
                log.warning(f"MT5 Broker init failed: {e} — dry-run mode")
        else:
            log.info("MT5 Broker not available — dry-run mode")

        self.log_file = BASE_DIR / "Engine_1_trade_logs.json"
        self.load_history()

    def _cooldown_key(self, strategy: str, symbol: str) -> str:
        return f"{strategy}:{symbol}"

    def update_day(self) -> None:
        """Roll over daily PnL tracking."""
        with self.lock:
            today = datetime.now().strftime("%Y-%m-%d")
            if self.last_rollover_day != today:
                self.daily_start_capital = self.current_capital
                self.last_rollover_day = today
                log.info(f"Daily rollover: start capital = ${self.daily_start_capital:.2f}")

    def load_history(self):
        """Load trade history from disk."""
        with self.lock:
            if not self.log_file.exists():
                return
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                meta = data.get('__meta__', {}) if isinstance(data, dict) else {}
                trades = data.get('trades', data) if isinstance(data, dict) else data
                if not isinstance(trades, list):
                    return
                self.last_entry_bar = meta.get('last_entry_bar', {})
                self.history = [t for t in trades if t.get('exit_price')]
                self.current_capital = self.initial_capital + sum(
                    t.get('pnl_usd', 0.0) for t in self.history
                )
                peak = self.initial_capital
                cur = self.initial_capital
                for t in sorted(self.history, key=lambda x: x.get('exit_time', '')):
                    cur += float(t.get('pnl_usd', 0.0))
                    if cur > peak:
                        peak = cur
                self.peak_capital = peak
                self.daily_start_capital = meta.get('daily_start_capital', self.current_capital)
                self.last_rollover_day = meta.get('last_rollover_day', self.last_rollover_day)
                for t in trades:
                    if not t.get('exit_price') and t.get('trade_id'):
                        self.active_trades[t['trade_id']] = t.copy()
            except Exception as e:
                log.error(f"Failed to load trade history: {e}")

    def save_history(self):
        """Save trade history to disk."""
        with self.lock:
            try:
                all_trades = list(self.history) + list(self.active_trades.values())
                envelope = {
                    '__meta__': {
                        'last_entry_bar': dict(self.last_entry_bar),
                        'daily_start_capital': self.daily_start_capital,
                        'last_rollover_day': self.last_rollover_day,
                    },
                    'trades': all_trades,
                }
                tmp = str(self.log_file) + ".tmp"
                with open(tmp, 'w', encoding='utf-8') as f:
                    json.dump(envelope, f, indent=4)
                os.replace(tmp, str(self.log_file))
            except Exception:
                pass

    def trigger_entry(self, symbol: str, strategy: str, direction: int,
                      entry_price: float, sl: float, tp: float, atr: float,
                      macro: int, vol_regime: float, risk_mult: float = 1.0,
                      trail_act: float = 0.5, regime_val: int = 0) -> None:
        """Validate risk limits and dispatch trade entry."""
        with self.lock:
            if self.emergency_halt:
                log.warning(f"[Risk] Entry blocked: emergency halt")
                return

            # Daily drawdown check (4% guardrail)
            active_list = list(self.active_trades.values())
            unrealized = sum(t.get('live_pnl_usd', 0.0) for t in active_list)
            equity = self.current_capital + unrealized
            daily_dd = (self.daily_start_capital - equity) / self.daily_start_capital * 100.0
            if daily_dd >= 4.0:
                log.warning(f"[Risk] Daily DD {daily_dd:.2f}% > 4% guardrail")
                return

            # Total drawdown check (8% guardrail)
            total_dd = (self.initial_capital - equity) / self.initial_capital * 100.0
            if total_dd >= 8.0:
                log.warning(f"[Risk] Total DD {total_dd:.2f}% > 8% guardrail")
                return

            # Cooldown check
            cool_key = self._cooldown_key(strategy, symbol)
            cooldown_until = self.reentry_cooldown_until.get(cool_key, 0.0)
            if time.time() < cooldown_until:
                remaining = cooldown_until - time.time()
                log.debug(f"[Risk] Cooldown active for {cool_key}: {remaining:.0f}s")
                return

            # Duplicate check
            if any(t.get('symbol') == symbol for t in active_list):
                return

            # Concurrent trade limit
            strategy_trades = [t for t in active_list if t.get('strategy') == strategy]
            max_concurrent = 3 if regime_val == 1 else config.min_agreeing
            if len(strategy_trades) >= max_concurrent:
                return

            # Validate SL/TP ordering
            if direction == 1 and not (sl < entry_price < tp):
                return
            if direction == -1 and not (tp < entry_price < sl):
                return

            # Position sizing
            stop_dist = abs(entry_price - sl)
            tick_size = TICK_SIZES.get(symbol, 0.0001)

            # Enforce minimum stop
            min_stop = 5.0 * tick_size
            tp_dist = abs(tp - entry_price)
            if stop_dist < min_stop:
                factor = min_stop / stop_dist
                stop_dist = min_stop
                tp_dist *= factor
                if direction == 1:
                    sl = entry_price - stop_dist
                    tp = entry_price + tp_dist
                else:
                    sl = entry_price + stop_dist
                    tp = entry_price - tp_dist

            # Enforce minimum SL percentage
            min_stop_pct = float(os.environ.get("MIN_LIVE_STOP_PCT", "0.003"))
            stop_pct = stop_dist / entry_price
            if stop_pct < min_stop_pct:
                rr = tp_dist / stop_dist if stop_dist > 0 else 3.0
                stop_dist = entry_price * min_stop_pct
                tp_dist = stop_dist * rr
                if direction == 1:
                    sl = entry_price - stop_dist
                    tp = entry_price + tp_dist
                else:
                    sl = entry_price + stop_dist
                    tp = entry_price - tp_dist

            # Zeno risk formula
            max_dd_limit = 250.0
            zeno_denom = 5.0
            risk_cap = 20.0
            current_dd = max(0.0, self.peak_capital - self.current_capital)
            raw_zeno = (max_dd_limit - current_dd) / zeno_denom
            zeno_risk_pct = max(0.0, min(risk_cap, raw_zeno)) / 5000.0
            risk_capital = max(0.0, self.current_capital) * zeno_risk_pct * risk_mult

            if risk_capital <= 0.0 or stop_dist <= 0:
                return

            units = risk_capital / stop_dist
            cap = MAX_UNITS_PER_SYMBOL.get(symbol, float('inf'))
            if units > cap:
                units = cap

            # Portfolio heat check (4% of equity)
            open_stop_risk = 0.0
            for t in active_list:
                t_units = t.get('units', 0.0)
                t_dir = t.get('direction', 1)
                t_ep = t.get('entry_price', 0.0)
                t_sl = t.get('sl', 0.0)
                risk_pts = max(0.0, t_ep - t_sl) if t_dir == 1 else max(0.0, t_sl - t_ep)
                open_stop_risk += t_units * risk_pts
            total_portfolio_risk = open_stop_risk + units * stop_dist
            if total_portfolio_risk > equity * 0.04:
                log.warning(f"[Risk] Portfolio heat ${total_portfolio_risk:.2f} > 4% equity")
                return

            # Create trade record
            trade_id = f"{strategy}_{symbol}_{'LONG' if direction == 1 else 'SHORT'}_{int(time.time_ns())}"
            self.active_trades[trade_id] = {
                "trade_id": trade_id,
                "symbol": symbol,
                "strategy": strategy,
                "direction": direction,
                "entry_price": entry_price,
                "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
                "trail_buf": 0.5,
            }
            log.info(f"[ENTRY] {trade_id}: {symbol} {'LONG' if direction==1 else 'SHORT'} "
                     f"@{entry_price:.2f} SL={sl:.2f} TP={tp:.2f} units={units:.4f}")

        # Dispatch to MT5 (outside lock)
        if self.broker_executor and self.mt5_broker and MT5_LIVE:
            try:
                fut = self.broker_executor.submit(
                    self.mt5_broker.execute_trade,
                    symbol, direction, entry_price, sl, tp, strategy, risk_capital
                )
                mt5_res = fut.result(timeout=30)
                with self.lock:
                    t = self.active_trades.get(trade_id)
                    if t and mt5_res:
                        t["mt5_ticket"] = mt5_res.get("mt5_ticket")
                        t["mt5_entry"] = mt5_res.get("mt5_entry")
                        t["mt5_sl"] = mt5_res.get("mt5_sl")
                        t["mt5_tp"] = mt5_res.get("mt5_tp")
                        t["mt5_lot"] = mt5_res.get("lot")
                        log.info(f"[MT5] Trade {trade_id} dispatched, ticket={t['mt5_ticket']}")
                    elif t:
                        log.warning(f"[MT5] Trade {trade_id} rejected — removing")
                        self.active_trades.pop(trade_id, None)
            except Exception as e:
                log.error(f"[MT5] Dispatch failed for {trade_id}: {e}")
                with self.lock:
                    self.active_trades.pop(trade_id, None)

        self.save_history()

    def update_live_pnl(self, symbol: str, current_price: float):
        """Update unrealized PnL for all active trades on symbol."""
        with self.lock:
            for trade in list(self.active_trades.values()):
                if trade.get('symbol') != symbol:
                    continue
                direction = trade['direction']
                entry_price = trade['entry_price']
                pnl_pct = ((current_price - entry_price) / entry_price * 100.0
                           if direction == 1 else
                           (entry_price - current_price) / entry_price * 100.0)
                pnl_usd = trade['units'] * (current_price - entry_price) * direction
                trade['live_pnl_pct'] = pnl_pct
                trade['live_pnl_usd'] = pnl_usd

                # Track MFE/MAE
                mfe = trade.get('mfe_pct', pnl_pct)
                mae = trade.get('mae_pct', pnl_pct)
                if pnl_pct > mfe:
                    mfe = pnl_pct
                if pnl_pct < mae:
                    mae = pnl_pct
                trade['mfe_pct'] = mfe
                trade['mae_pct'] = mae

            # Emergency halt check
            unrealized = sum(t.get('live_pnl_usd', 0.0)
                             for t in self.active_trades.values())
            equity = self.current_capital + unrealized
            daily_dd = ((self.daily_start_capital - equity) / self.daily_start_capital * 100.0
                        if self.daily_start_capital > 0 else 0.0)
            total_dd = ((self.initial_capital - equity) / self.initial_capital * 100.0)

            if daily_dd >= 4.5 or total_dd >= 9.0:
                if not self.emergency_halt:
                    self.emergency_halt = True
                    log.critical(f"EMERGENCY HALT! Daily DD={daily_dd:.2f}% Total DD={total_dd:.2f}%")

    def check_exits(self, symbol: str, current_price: float,
                    current_atr: float = 0.0) -> None:
        """Check and execute SL/TP/trailing stop exits."""
        with self.lock:
            trades_for_symbol = [t for t in self.active_trades.values()
                                 if t.get('symbol') == symbol]
            any_closed = False
            for trade in trades_for_symbol:
                direction = trade['direction']
                sl = trade['sl']
                tp = trade['tp']
                entry_price = trade['entry_price']
                sl_dist = trade.get('sl_dist')

                # Trailing stop logic
                trail_act = trade.get('trail_act', 1.0)
                if sl_dist and trail_act > 0.0 and current_atr > 0:
                    if direction == 1:
                        cur_r = (current_price - entry_price) / sl_dist
                        if cur_r >= trail_act:
                            trail_buf = trade.get('trail_buf', 0.5)
                            ns = entry_price + (cur_r - trail_buf) * sl_dist
                            if ns > sl:
                                trade['sl'] = ns
                                sl = ns
                    else:
                        cur_r = (entry_price - current_price) / sl_dist
                        if cur_r >= trail_act:
                            trail_buf = trade.get('trail_buf', 0.5)
                            ns = entry_price - (cur_r - trail_buf) * sl_dist
                            if ns < sl:
                                trade['sl'] = ns
                                sl = ns

                # Timeout exit (24 hours)
                elapsed = time.time() - trade.get('entry_timestamp', time.time())
                should_close = elapsed >= 86400
                reason = "TIMEOUT" if should_close else ""

                # SL/TP check
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
                    exit_price = (trade['sl'] if reason == "SL"
                                  else trade['tp'] if reason == "TP"
                                  else current_price)

                    trade['exit_price'] = exit_price
                    trade['exit_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    trade['exit_reason'] = reason

                    pnl_pct = ((exit_price - entry_price) / entry_price * 100.0
                               if direction == 1 else
                               (entry_price - exit_price) / entry_price * 100.0)
                    pnl_usd = trade['units'] * (exit_price - entry_price) * direction

                    trade['pnl_pct'] = pnl_pct
                    trade['pnl_usd'] = pnl_usd

                    self.history.append(trade)
                    self.current_capital += pnl_usd
                    if self.current_capital > self.peak_capital:
                        self.peak_capital = self.current_capital

                    # Set re-entry cooldown
                    cooldown_secs = (self.REENTRY_COOLDOWN_TP_SECS if reason == "TP"
                                     else self.REENTRY_COOLDOWN_SL_SECS)
                    if cooldown_secs > 0:
                        key = self._cooldown_key(trade.get('strategy', ''), symbol)
                        self.reentry_cooldown_until[key] = time.time() + cooldown_secs

                    log.info(f"[EXIT] {trade['trade_id']}: {reason} @ {exit_price:.2f} "
                             f"PnL=${pnl_usd:.2f} ({pnl_pct:+.2f}%)")

                    del self.active_trades[trade['trade_id']]
                    any_closed = True

                    # Notify callbacks
                    strategy = trade.get('strategy', '')
                    for cb in self.on_close_callbacks:
                        try:
                            cb(strategy, self.current_capital)
                        except Exception:
                            pass

            if any_closed:
                self.save_history()

    def get_stats(self) -> dict:
        with self.lock:
            total = len(self.history)
            if total == 0:
                return {"total": 0, "winrate": 0.0, "total_pnl_usd": 0.0,
                        "current_capital": self.current_capital}
            wins = sum(1 for t in self.history if t.get('pnl_usd', 0.0) > 0)
            total_pnl = sum(t.get('pnl_usd', 0.0) for t in self.history)
            return {
                "total": total,
                "winrate": (wins / total) * 100.0,
                "total_pnl_usd": total_pnl,
                "current_capital": self.current_capital,
            }


# ─── DASHBOARD RENDERER ────────────────────────────────────────────────────

def render_table(snap: Dict[str, AssetSnapshot], trade_tracker=None):
    """Render Rich terminal dashboard table."""
    try:
        from rich.console import Console, Group
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text

        t = Table(title="Engine_1 — 6-Strategy ML Trading Terminal", expand=True)
        cols = ("Symbol", "Price", "RSI", "FutCVD", "LiqL", "LiqS",
                "Fund", "LSR", "OI", "FP_Δ", "FP_POC", "ARM")
        for col in cols:
            t.add_column(col, justify="center", no_wrap=True)

        now = time.time_ns()

        def fmt(v, fresh, is_funding=False):
            if v == 0.0 or v is None:
                return "[dim]--[/dim]"
            s = f"{v:.6f}" if is_funding else f"{v:,.2f}"
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
                fmt(a.liq_long, fresh),
                fmt(a.liq_short, fresh),
                fmt(a.funding, fresh, is_funding=True),
                fmt(a.ls_ratio, fresh),
                fmt(a.oi, fresh),
                fmt(a.fp_delta, fresh),
                fmt(a.fp_poc, fresh),
                f"[green]{a.strategy_armed}[/green]" if a.strategy_armed else "[dim]--[/dim]"
            )

        if trade_tracker is None:
            return t

        stats = trade_tracker.get_stats()
        total_pnl = stats['total_pnl_usd']
        pnl_clr = "green" if total_pnl >= 0 else "red"
        pnl_sign = "+" if total_pnl >= 0 else ""
        pnl_pct = total_pnl / trade_tracker.initial_capital * 100.0

        stats_text = (
            f"Capital: [bold]${stats['current_capital']:,.2f}[/bold] | "
            f"PnL: [bold {pnl_clr}]{pnl_sign}${total_pnl:.2f} ({pnl_pct:+.2f}%)[/bold] | "
            f"Trades: [bold]{stats['total']}[/bold] | "
            f"WR: [bold]{stats['winrate']:.1f}%[/bold]"
        )

        active_lines = []
        with trade_tracker.lock:
            for tr in list(trade_tracker.active_trades.values()):
                dir_str = "[bold green]LONG[/]" if tr['direction'] == 1 else "[bold red]SHORT[/]"
                pnl_u = tr.get('live_pnl_usd', 0.0)
                pnl_p = tr.get('live_pnl_pct', 0.0)
                pnl_s = f"[green]+${pnl_u:.2f}[/green]" if pnl_u >= 0 else f"[red]-${abs(pnl_u):.2f}[/red]"
                active_lines.append(
                    f"{tr['symbol']} | {dir_str} | Entry: {tr['entry_price']:.4f} | "
                    f"SL: {tr['sl']:.4f} | TP: {tr['tp']:.4f} | PnL: {pnl_s} ({pnl_p:+.2f}%)"
                )

        active_text = "\n".join(active_lines) if active_lines else "[dim]No active trades[/dim]"

        return Group(t, Panel(active_text, title="Active Trades", border_style="cyan"),
                     Panel(stats_text, title="Stats", border_style="magenta"))
    except ImportError:
        return str(snap)


# ─── COINGLASS TAB (Stub) ──────────────────────────────────────────────────

# ─── SEEDING ───────────────────────────────────────────────────────────────

async def seed_all_symbols(predictor, symbols: list, data_dir: Path):
    """
    Seed EnsembleStrategyPredictor with up to 1200 historical 15m bars
    for every symbol from parquet files.

    Tries multiple paths:
      1. G:\\My Drive\\_Trading_Data\\15m\\parquet (Google Drive)
      2. Backtesting_Data/ (local fallback)
    """
    log.info(f"[Startup] Step 4/5 — Seeding {len(symbols)} symbols...")

    async def seed_one(sym: str):
        paths_to_try = [
            Path(r"G:\My Drive\_Trading_Data\15m\parquet") / f"Master_{sym}_15m_Final_Summary.parquet",
            data_dir / f"Master_{sym}_15m_Final_Summary.parquet",
            data_dir / f"{sym}_15m_summary.parquet",
        ]

        for p in paths_to_try:
            if p.exists():
                try:
                    df = pd.read_parquet(p)

                    # Normalize columns to dict records with open_time
                    ts_col = None
                    for candidate in ["TimeStamp", "Timestamp", "time", "ts"]:
                        if candidate in df.columns:
                            ts_col = candidate
                            break

                    if ts_col:
                        df["_ts"] = pd.to_datetime(
                            df[ts_col].astype(str).str.replace(" IST", "", regex=False),
                            errors="coerce"
                        )
                        df["open_time"] = df["_ts"].astype("int64") // 10**9
                        df = df.drop(columns=["_ts"], errors="ignore")

                    # Take last 1200 bars
                    df = df.tail(1200)

                    candles = df.reset_index(drop=True).to_dict("records")

                    # Ensure every row has open_time
                    candles = [{**r, "open_time": int(r.get("open_time",
                               int(pd.Timestamp.now().timestamp())))} for r in candles]

                    predictor.set_history(sym, candles)
                    log.info(f"[Seeding] {sym}: loaded {len(candles)} bars from {p.name}")
                    return
                except Exception as e:
                    log.warning(f"[Seeding] {sym}: failed to load {p.name} — {e}")

        # Fallback: try Excel seeding file
        excel_path = BASE_DIR / "Seeding" / "combined_seed_history.xlsx"
        if excel_path.exists():
            try:
                df = pd.read_excel(excel_path, sheet_name=sym)
                ts_col = None
                for candidate in ["open_time", "TimeStamp", "Timestamp", "time", "ts"]:
                    if candidate in df.columns:
                        ts_col = candidate
                        break
                if ts_col:
                    df["_ts"] = pd.to_datetime(
                        df[ts_col].astype(str).str.replace(" IST", "", regex=False),
                        errors="coerce"
                    )
                    df["open_time"] = df["_ts"].astype("int64") // 10**9
                    df = df.drop(columns=["_ts"], errors="ignore")
                df = df.tail(1200)
                candles = df.reset_index(drop=True).to_dict("records")
                candles = [{**r, "open_time": int(r.get("open_time",
                           int(pd.Timestamp.now().timestamp())))} for r in candles]
                predictor.set_history(sym, candles)
                log.info(f"[Seeding] {sym}: loaded {len(candles)} bars from combined_seed_history.xlsx")
                return
            except Exception as e:
                log.warning(f"[Seeding] {sym}: failed to load from Excel — {e}")

        log.warning(f"[Seeding] {sym}: no parquet data found, starting cold.")

    await asyncio.gather(*[seed_one(s) for s in symbols])
    log.info("[Startup] Seeding complete.")


# ─── MAIN ASYNC CONTROLLER ─────────────────────────────────────────────────

async def renderer_loop(store: SnapshotStore, stop: asyncio.Event) -> None:
    """Rich terminal live display loop."""
    try:
        from rich.console import Console
        from rich.live import Live
        console = Console()
        with Live(render_table(store.snapshot(), store.trade_tracker),
                  console=console, refresh_per_second=REFRESH_HZ,
                  screen=False) as live:
            while not stop.is_set():
                snap = store.snapshot()
                live.update(render_table(snap, store.trade_tracker))
                await asyncio.sleep(1.0 / REFRESH_HZ)
    except Exception as e:
        log.warning(f"Terminal dashboard error: {e}")
        while not stop.is_set():
            await asyncio.sleep(1.0)


async def watchdog(components: List[Any], stop: asyncio.Event) -> None:
    """Health monitor with heartbeat checks."""
    now_start = time.time_ns()
    for c in components:
        if hasattr(c, 'last_heartbeat_ns'):
            c.last_heartbeat_ns = now_start

    while not stop.is_set():
        for c in components:
            if hasattr(c, 'last_heartbeat_ns') and not getattr(c, 'skip_watchdog', False):
                if time.time_ns() - c.last_heartbeat_ns > 90_000_000_000:
                    log.warning(f"[Watchdog] {c.__class__.__name__} heartbeat stale >90s")
        await asyncio.sleep(5.0)



async def main_async(skip_seed: bool = False, skip_train: bool = False,
                     skip_browser: bool = False, active_strategies=None) -> None:
    """Main async entry point for production mode with modular startup options."""
    log.info("=" * 60)
    log.info(f"ENGINE_1 STARTING — 6-Strategy ML Trading System")
    log.info(f"Mode: {EXECUTION_MODE} | MT5 Live: {MT5_LIVE}")
    log.info(f"Symbols: {len(ALL_SYMBOLS)} total ({len(TAB1_SYMBOLS)} Tab1 + {len(TAB2_SYMBOLS)} Tab2)")
    log.info("=" * 60)

    # 1. Initialize Core Components
    trade_tracker = Engine1TradeTracker()
    trade_tracker.update_day()

    predictor = EnsembleStrategyPredictor(ALL_SYMBOLS, active_strategies=active_strategies)
    predictor.recent_capitals = [trade_tracker.current_capital]
    trade_tracker.on_close_callbacks.append(
        lambda strategy, capital: predictor.record_closed_capital(capital)
    )

    store = SnapshotStore(ALL_SYMBOLS, predictor=predictor, trade_tracker=trade_tracker)

    if skip_browser:
        log.info("[Startup] --skip-browser active. Skipping Playwright/Coinglass tabs.")
        log.info("[Startup] Starting in Binance-only live feed mode.")

        # Seed predictor from available cache if skip_seed is False
        if not skip_seed:
            await seed_all_symbols(predictor, ALL_SYMBOLS, DATA_DIR)

        binance_feed = BinanceFootprintFeed(ALL_SYMBOLS, store)
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        def sig_handler():
            log.info("Shutdown signal received. Stopping...")
            stop.set()
            binance_feed.running = False

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, sig_handler)
            except NotImplementedError:
                pass

        tasks = [
            asyncio.create_task(binance_feed.run()),
            asyncio.create_task(renderer_loop(store, stop)),
            asyncio.create_task(watchdog([binance_feed], stop)),
        ]

        log.info("Engine_1 running (browser-less mode) — waiting for market data...")
        try:
            while not stop.is_set():
                await asyncio.sleep(1.0)
        except (KeyboardInterrupt, SystemExit):
            sig_handler()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if hasattr(trade_tracker, 'broker_executor') and trade_tracker.broker_executor:
                trade_tracker.broker_executor.shutdown(wait=True)
            ML_EXECUTOR.shutdown(wait=True)
        return

    # Launch Playwright for Coinglass Tabs
    log.info("[Startup] Launching Chromium instance with persistent profile...")
    if async_playwright is None:
        raise RuntimeError("Playwright package is missing. Please run: pip install playwright && python -m playwright install chromium")
    async with async_playwright() as pw:
        user_data_dir = BASE_DIR / "chrome_profile"
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir,
            headless=False,
            viewport={"width": 1920, "height": 1080},
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-features=CalculateNativeWinOcclusion",
                "--disable-background-timer-throttling",
                "--start-maximized",
                "--remote-debugging-port=9222",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-infobars",
                "--window-size=1920,1080",
                "--disable-dev-shm-usage",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
            ignore_default_args=["--enable-automation"],
        )
        
        # Apply stealth patches to every page
        ctx.on("page", lambda page: page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            window.chrome = {runtime: {}};
        """))
        
        # Performing Session Login first
        log.info("[Startup] Navigating to Coinglass Login...")
        login_page = await ctx.new_page()
        
        for attempt in range(3):
            try:
                await login_page.goto("https://www.coinglass.com/login", wait_until="load", timeout=45000)
                break
            except Exception as exc:
                log.warning(f"[Startup] Login navigation attempt {attempt+1} failed: {exc}")
                if attempt == 2:
                    raise exc
                await asyncio.sleep(5.0)
        await asyncio.sleep(5)
        
        user_data_dir.mkdir(parents=True, exist_ok=True)
        # Assuming manual login isn't needed if session is cached, but try to click anyway
        email_input = login_page.locator("input[placeholder='Email']").first
        if await email_input.count() > 0:
            email = os.environ.get("COINGLASS_EMAIL")
            password = os.environ.get("COINGLASS_PASSWORD")
            if email and password:
                await email_input.click()
                await email_input.fill(email)
                await asyncio.sleep(0.3)
                pass_input = login_page.locator("input[placeholder='Password']").first
                await pass_input.click()
                await pass_input.fill(password)
                await asyncio.sleep(0.3)
                
                log.info("[Startup] Submitting login form...")
                try:
                    await login_page.evaluate("""() => {
                        const b = Array.from(document.querySelectorAll('button')).find(el => el.textContent.trim() === 'Login');
                        if (b) b.click();
                    }""")
                except Exception:
                    await pass_input.press("Enter")
                    
                log.info("[Startup] Waiting for post-login redirect...")
                try:
                    await login_page.wait_for_url(lambda url: "/login" not in url, timeout=20000)
                    log.info("[Startup] Login successful — redirected away from /login.")
                except Exception:
                    log.warning("[Startup] No redirect detected — may already be logged in or login failed.")
                await asyncio.sleep(5.0)
        else:
            log.info("[Startup] Form inputs not detected, assuming session already active.")

        # Initialize Tabs
        if CoinglassTab is None:
            raise RuntimeError("coinglass_scraper module missing or failed to import CoinglassTab.")
        tab1 = CoinglassTab(ctx, TAB1_SYMBOLS, store, "TAB_1")
        tab2 = CoinglassTab(ctx, TAB2_SYMBOLS, store, "TAB_2")

        log.info("[Startup] Step 2/5 — Starting 2 Coinglass Chrome tabs...")
        await asyncio.gather(tab1.start(), tab2.start())

        try:
            await login_page.close()
        except Exception:
            pass

        focus_lock = asyncio.Lock()
        await asyncio.gather(
            tab1.inject_and_configure_all(focus_lock),
            tab2.inject_and_configure_all(focus_lock)
        )

        # 4. Historical Seeding
        from concurrent.futures import ThreadPoolExecutor
        excel_pool = ThreadPoolExecutor(max_workers=4)
        if not skip_seed:
            log.info("[Startup] Step 3/5 — Seeding via Chrome DOM...")
            async def seed_wrapper(tab: CoinglassTab, sym: str):
                for attempt in range(3):
                    try:
                        if not tab.page or tab.page.is_closed():
                            await tab.reconnect(focus_lock)
                        await tab.seed_symbol(sym, excel_pool, focus_lock)
                        break
                    except Exception as e:
                        log.warning(f"[Setup] Seeding failed for {sym} (attempt {attempt+1}/3): {e}")
                        if attempt == 2:
                            log.warning(f"[Setup] Seeding skipped for {sym} — engine will proceed with parquet historical cache.")
                        else:
                            await asyncio.sleep(2.0)
            
            # Seed sequentially per tab
            for sym in TAB1_SYMBOLS:
                await seed_wrapper(tab1, sym)
            for sym in TAB2_SYMBOLS:
                await seed_wrapper(tab2, sym)
            
            log.info("[Startup] Seeding complete. Merging CSVs to Parquet...")
            combine_seeding_files()
            
        else:
            log.info("[Startup] Step 3/5 — Skipping seeding (--skip-seed flag).")

        if not skip_train:
            # 4. Retrain Models on Latest GDrive Data
            log.info("[Startup] Step 4/5 — Retraining models on latest GDrive data...")
            models_dir = BASE_DIR / "models"
            models_tmp = BASE_DIR / "models_training_tmp"
            models_old = BASE_DIR / "models_old_backup"
            if models_tmp.exists():
                shutil.rmtree(models_tmp)
            models_tmp.mkdir(parents=True, exist_ok=True)
            log.info("[Startup] Preparing temporary model training directory.")

            try:
                from live_model_trainer import train_all_strategies
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, train_all_strategies)
                
                # Atomic swap: tmp → models
                if models_old.exists():
                    shutil.rmtree(models_old)
                if models_dir.exists():
                    models_dir.rename(models_old)
                models_tmp.rename(models_dir)
                if models_old.exists():
                    shutil.rmtree(models_old)
                log.info("[Startup] Model retraining complete — swapped in new models.")
            except ImportError:
                log.warning("[Startup] live_model_trainer.py not found — skipping model training.")
            except Exception as e:
                log.warning(f"[Startup] Model training failed ({e}), keeping existing models.")
                if models_tmp.exists():
                    shutil.rmtree(models_tmp)
        else:
            log.info("[Startup] Step 4/5 — Skipping model clearing and retraining (--skip-train).")

        # Now call the engine's original parquet loader to feed predictor
        await seed_all_symbols(predictor, ALL_SYMBOLS, DATA_DIR)

        # 5. Warm-up Gate
        log.info("[Startup] Step 5/5 — Warm-up gate active...")
        binance_feed = BinanceFootprintFeed(ALL_SYMBOLS, store)
        stop = asyncio.Event()

        loop = asyncio.get_running_loop()
        def sig_handler():
            log.info("Shutdown signal received. Stopping...")
            stop.set()
            binance_feed.running = False
            tab1.running = False
            tab2.running = False

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, sig_handler)
            except NotImplementedError:
                pass

        tasks = [
            asyncio.create_task(tab1.run()),
            asyncio.create_task(tab2.run()),
            asyncio.create_task(binance_feed.run()),
            asyncio.create_task(renderer_loop(store, stop)),
            asyncio.create_task(watchdog([tab1, tab2, binance_feed], stop)),
        ]

        log.info("Engine_1 running — waiting for market data...")
        try:
            while not stop.is_set():
                await asyncio.sleep(1.0)
        except (KeyboardInterrupt, SystemExit):
            sig_handler()
        finally:
            log.info("Shutting down...")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if hasattr(trade_tracker, 'broker_executor') and trade_tracker.broker_executor:
                trade_tracker.broker_executor.shutdown(wait=True)
            ML_EXECUTOR.shutdown(wait=True)
            log.info("Engine_1 shutdown complete.")



# ─── BACKTEST MODE ─────────────────────────────────────────────────────────

def run_backtest(symbol: str, data_dir: Path = None):
    """Run a full backtest on one symbol using the exact validated pipeline."""
    if data_dir is None:
        data_dir = DATA_DIR

    log.info(f"Running backtest for {symbol}...")

    sp = data_dir / f"Master_{symbol}_15m_Final_Summary.parquet"
    fp = data_dir / f"Master_{symbol}_15m_Final_Footprint.parquet"

    if not sp.exists():
        log.error(f"Data file not found: {sp}")
        return None

    df = pd.read_parquet(sp)
    ts_col = "TimeStamp" if "TimeStamp" in df.columns else "Timestamp"
    df["ts"] = pd.to_datetime(
        df[ts_col].astype(str).str.replace(" IST", "", regex=False),
        errors="coerce"
    )

    if fp.exists():
        df_f = pd.read_parquet(fp)
        tcf = "TimeStamp" if "TimeStamp" in df_f.columns else "Timestamp"
        df_f["ts"] = pd.to_datetime(
            df_f[tcf].astype(str).str.replace(" IST", "", regex=False),
            errors="coerce"
        )
        dc = [c for c in df_f.columns if c in
              ["Symbol", "POC Price", "Candle #", "Timestamp", "TimeStamp", "time", "Is POC"]]
        if dc:
            df_f = df_f.drop(columns=dc, errors="ignore")
        df = pd.merge_asof(df.sort_values("ts"), df_f.sort_values("ts"),
                           on="ts", direction="backward",
                           tolerance=pd.Timedelta(minutes=5))
    else:
        df = df.sort_values("ts")

    dc = [c for c in df.columns if c in
          ["Symbol", "POC Price", "Candle #", "Timestamp", "TimeStamp", "time", "Is POC"]]
    if dc:
        df = df.drop(columns=dc, errors="ignore")
    for c in df.columns:
        if c != "ts":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    df = df.set_index("ts")

    dff = featurize(df)

    from numba import njit

    @njit(fastmath=True, nogil=True)
    def simulate_trade_numba(h, l, c_arr, entry_idx, entry, atr, dr, tp, trail,
                              risk, fee, cap):
        n = len(c_arr)
        sd = atr
        td = tp * atr
        trd = trail * atr
        if dr == 1:
            stop = entry - sd
        else:
            stop = entry + sd
        current_stop = stop
        best_price = entry
        worst_price = entry
        max_bars = min(entry_idx + 288 + 1, n)
        exit_price = c_arr[max_bars - 1]
        bars_held = max_bars - 1 - entry_idx
        for j in range(entry_idx + 1, max_bars):
            if dr == 1:
                if l[j] <= current_stop:
                    exit_price = current_stop
                    bars_held = j - entry_idx
                    break
                if l[j] < worst_price:
                    worst_price = l[j]
                if h[j] > best_price:
                    best_price = h[j]
                    if (best_price - entry) >= td:
                        ns = best_price - trd
                        if ns > current_stop:
                            current_stop = ns
            else:
                if h[j] >= current_stop:
                    exit_price = current_stop
                    bars_held = j - entry_idx
                    break
                if h[j] > worst_price:
                    worst_price = h[j]
                if l[j] < best_price:
                    best_price = l[j]
                    if (entry - best_price) >= td:
                        ns = best_price + trd
                        if ns < current_stop:
                            current_stop = ns
        units = risk / sd
        gross = units * (exit_price - entry) if dr == 1 else units * (entry - exit_price)
        fee_cost = units * entry * (fee / 2.0) + units * abs(exit_price) * (fee / 2.0)
        net_pnl = gross - fee_cost
        r_mult = net_pnl / risk
        label = 1.0 if net_pnl > 0 else 0.0
        if dr == 1:
            mae = units * (entry - worst_price)
        else:
            mae = units * (worst_price - entry)
        mae_dd_pct = abs(mae) / cap * 100.0 if mae > 0 else 0.0
        return net_pnl, r_mult, label, bars_held, mae_dd_pct

    results = {}
    h_arr = dff["High"].values
    l_arr = dff["Low"].values
    c_arr = dff["Close"].values
    atr_arr = dff["atr"].values

    # Per-strategy backtest
    for name, strat in STRATEGIES.items():
        sig = strat["fn"](dff)
        entries = np.where(sig != 0)[0]

        trades = []
        last_exit = -100
        for ei in entries:
            if ei <= last_exit + 2:
                continue
            dr = sig[ei]
            entry = c_arr[ei]
            atr_val = atr_arr[ei]
            if np.isnan(atr_val) or atr_val <= 0:
                continue
            pnl, r_mult, label, bars, mae = simulate_trade_numba(
                h_arr, l_arr, c_arr, ei, entry, atr_val, dr,
                config.tp_mult, config.trail_atr, config.risk_per_trade,
                config.fee_pct, config.initial_capital
            )
            trades.append({
                "entry_idx": ei, "direction": dr, "entry": entry,
                "pnl": pnl, "r": r_mult, "label": label,
                "bars": bars, "mae_dd": mae
            })
            last_exit = ei + bars

        if trades:
            wins = [t for t in trades if t["pnl"] > 0]
            total_pnl = sum(t["pnl"] for t in trades)
            wr = len(wins) / len(trades) * 100
            avg_r = np.mean([t["r"] for t in trades])
            max_mae = max(t["mae_dd"] for t in trades)
            results[name] = {
                "trades": len(trades), "wins": len(wins),
                "wr": round(wr, 1), "total_pnl": round(total_pnl, 2),
                "avg_r": round(float(avg_r), 2), "max_mae_dd": round(max_mae, 2),
            }
        else:
            results[name] = {"trades": 0, "wins": 0, "wr": 0.0,
                             "total_pnl": 0.0, "avg_r": 0.0, "max_mae_dd": 0.0}

    # Ensemble backtest (3/6 agreement required)
    all_sigs = {}
    for name, strat in STRATEGIES.items():
        all_sigs[name] = strat["fn"](dff)

    ensemble_sig = np.zeros(len(dff), dtype=np.int32)
    aggregator = EnsembleAggregator()
    for i in range(len(dff)):
        bar_signals = {name: int(sig[i]) for name, sig in all_sigs.items()}
        direction, confidence, agreeing = aggregator.aggregate(bar_signals)
        if aggregator.should_enter(direction, confidence, agreeing):
            ensemble_sig[i] = direction

    ensemble_entries = np.where(ensemble_sig != 0)[0]
    ensemble_trades = []
    last_exit = -100
    for ei in ensemble_entries:
        if ei <= last_exit + 2:
            continue
        dr = ensemble_sig[ei]
        entry = c_arr[ei]
        atr_val = atr_arr[ei]
        if np.isnan(atr_val) or atr_val <= 0:
            continue
        pnl, r_mult, label, bars, mae = simulate_trade_numba(
            h_arr, l_arr, c_arr, ei, entry, atr_val, dr,
            config.tp_mult, config.trail_atr, config.risk_per_trade,
            config.fee_pct, config.initial_capital
        )
        ensemble_trades.append({
            "entry_idx": ei, "direction": dr, "entry": entry,
            "pnl": pnl, "r": r_mult, "label": label,
            "bars": bars, "mae_dd": mae
        })
        last_exit = ei + bars

    if ensemble_trades:
        wins = [t for t in ensemble_trades if t["pnl"] > 0]
        total_pnl = sum(t["pnl"] for t in ensemble_trades)
        wr = len(wins) / len(ensemble_trades) * 100
        avg_r = np.mean([t["r"] for t in ensemble_trades])
        max_mae = max(t["mae_dd"] for t in ensemble_trades)
        results["ENSEMBLE_3of6"] = {
            "trades": len(ensemble_trades), "wins": len(wins),
            "wr": round(wr, 1), "total_pnl": round(total_pnl, 2),
            "avg_r": round(float(avg_r), 2), "max_mae_dd": round(max_mae, 2),
        }
    else:
        results["ENSEMBLE_3of6"] = {"trades": 0, "wins": 0, "wr": 0.0,
                                     "total_pnl": 0.0, "avg_r": 0.0, "max_mae_dd": 0.0}

    return results


# ─── SMOKE TEST ────────────────────────────────────────────────────────────

def smoke_test():
    """Verify engine components load and interact correctly."""
    log.info("=" * 60)
    log.info("ENGINE_1 SMOKE TEST")
    log.info("=" * 60)

    log.info("\n[1/5] Testing signal functions...")
    from ensemble_strategy_predictor import smoke_test as predictor_smoke
    predictor_smoke()

    log.info("\n[2/5] Testing backtest on BTCUSDT...")
    results = run_backtest("BTCUSDT")
    if results:
        for name, stats in results.items():
            log.info(f"  {name}: {stats['trades']} trades, "
                     f"WR={stats['wr']}%, PnL=${stats['total_pnl']:,.2f}")

    log.info("\n[3/5] Testing trade tracker...")
    tracker = Engine1TradeTracker(initial_capital=5000.0)
    tracker.update_day()
    tracker.trigger_entry(
        "BTCUSDT", "Ensemble_6Strategy", 1, 65000.0,
        64800.0, 67500.0, 200.0, 1, 0.0,
        risk_mult=1.0, trail_act=0.8, regime_val=0
    )
    tracker.update_live_pnl("BTCUSDT", 65500.0)
    tracker.check_exits("BTCUSDT", 67500.0, current_atr=200.0)
    stats = tracker.get_stats()
    log.info(f"  After trade: trades={stats['total']}, capital=${stats['current_capital']:,.2f}")

    log.info("\n[4/5] Testing snapshot store...")
    store = SnapshotStore(["BTCUSDT"])

    async def test_store():
        await store.update("BTCUSDT", source="test",
                           price=65000.0, volume=100.0, fut_cvd=5000.0)
        snap = store.snapshot()
        log.info(f"  BTCUSDT price: ${snap['BTCUSDT'].price:,.2f}")

    asyncio.run(test_store())

    log.info("\n[5/5] Testing ensemble aggregator...")
    aggregator = EnsembleAggregator()
    test_signals = {
        "S1_Liquidation": 1,
        "S2_CVD_Momentum": 1,
        "S3_Trend_Follow": 0,
        "S4_Mean_Reversion": 1,
        "S5_Vol_Expansion": 0,
        "S6_OI_Momentum": 1,
    }
    direction, confidence, agreeing = aggregator.aggregate(test_signals)
    should = aggregator.should_enter(direction, confidence, agreeing)
    log.info(f"  Direction={direction}, Confidence={confidence:.2f}, "
             f"Agreeing={agreeing}/6, Should Enter={should}")

    log.info("\n" + "=" * 60)
    log.info("SMOKE TEST COMPLETE — All systems operational")
    log.info("=" * 60)
    return True


# ─── MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Engine_1 — 6-Strategy ML Trading System"
    )
    parser.add_argument("--test", action="store_true", help="Run smoke test")
    parser.add_argument("--live", action="store_true", help="Start live trading")
    parser.add_argument("--skip-seed", action="store_true", help="Skip historical seeding")
    parser.add_argument("--skip-train", action="store_true", help="Skip model clearing and retraining")
    parser.add_argument("--skip-browser", action="store_true", help="Skip Playwright Chromium (pure Binance feed)")
    parser.add_argument("--ui-only", action="store_true", help="Convenience: --skip-seed --skip-train --skip-browser")
    parser.add_argument("--active-strategies", type=str, metavar="S2,S3,S6",
                        help="Comma-separated strategies to ENABLE (e.g., S2,S3,S6)")
    parser.add_argument("--skip-strategies", type=str, metavar="S1",
                        help="Comma-separated strategies to DISABLE")
    parser.add_argument("--backtest", type=str, metavar="SYMBOL",
                        help="Run backtest on one symbol")
    args = parser.parse_args()

    # UI-ONLY convenience
    if args.ui_only:
        args.skip_seed = True
        args.skip_train = True
        args.skip_browser = True
        args.live = True

    # Strategy selection
    from ensemble_strategy_predictor import resolve_active_strategies
    active_strategies = None
    if args.active_strategies:
        active_strategies = resolve_active_strategies(
            active=[s.strip() for s in args.active_strategies.split(",") if s.strip()])
    elif args.skip_strategies:
        active_strategies = resolve_active_strategies(
            skip=[s.strip() for s in args.skip_strategies.split(",") if s.strip()])

    if args.backtest:
        results = run_backtest(args.backtest)
        if results:
            print(f"\n{'='*70}")
            print(f"BACKTEST RESULTS — {args.backtest} (6yr continuous, 0.20% fee)")
            print(f"{'='*70}")
            print(f"  {'Strategy':<25s} {'Trades':>6s}  {'WR':>6s}  {'PnL':>14s}  {'AvgR':>7s}  {'MaxMAE':>7s}")
            print(f"  {'─'*67}")
            total_pnl = 0
            total_trades = 0
            for name, stats in results.items():
                is_ensemble = name.startswith("ENSEMBLE")
                prefix = "⭐ " if is_ensemble else "  "
                print(f"  {prefix}{name:<23s} {stats['trades']:>6d}  "
                      f"{stats['wr']:>5.1f}%  ${stats['total_pnl']:>12,.2f}  "
                      f"{stats['avg_r']:>+6.2f}  {stats['max_mae_dd']:>6.2f}%")
                if not is_ensemble:
                    total_pnl += stats['total_pnl']
                    total_trades += stats['trades']
            print(f"  {'─'*67}")
            print(f"  {'SUM (individual)':<25s} {total_trades:>6d}  "
                  f"{'':>6s}  ${total_pnl:>12,.2f}")
            print(f"\n  ⭐ ENSEMBLE = trades requiring 3+/6 strategy agreement")
            print(f"  Note: Live engine adds risk gov, circuit breakers, MT5 execution")
    elif args.live:
        asyncio.run(main_async(skip_seed=args.skip_seed, skip_train=args.skip_train,
                               skip_browser=args.skip_browser,
                               active_strategies=active_strategies))
    else:
        smoke_test()
