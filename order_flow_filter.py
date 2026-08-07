# order_flow_filter.py — OrderFlowMicrostructureFilter
# Bid/Ask depth imbalance, spoofing filter, liquidation wall absorption detector

import time
import math
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import threading

import numpy as np

log = logging.getLogger('OrderFlow')

# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BookLevel:
    """Single level of the order book."""
    price: float = 0.0
    quantity: float = 0.0
    notional: float = 0.0          # price × quantity
    spoof_score: float = 0.0       # 0 = stable, 1 = likely spoofed
    weight: float = 1.0            # contribution weight to imbalance

@dataclass
class DepthSnapshot:
    """Full order book depth snapshot at a point in time."""
    timestamp: float = 0.0
    bids: List[BookLevel] = field(default_factory=list)
    asks: List[BookLevel] = field(default_factory=list)
    raw_imbalance: float = 0.0
    dollar_imbalance: float = 0.0
    adjusted_imbalance: float = 0.0
    spoof_count: int = 0

@dataclass
class AbsorptionSignal:
    """Liquidation wall absorption detection result."""
    timestamp: float = 0.0
    detected: bool = False
    liq_spike_z: float = 0.0          # Liquidation z-score
    price_response_atr: float = 0.0    # Price move in ATR units
    cvd_confirmation: float = 0.0      # CVD z-score during cascade
    bid_depth_delta: float = 1.0       # Bid depth ratio vs baseline
    absorption_score: float = 0.0      # 0-1 composite score
    signal_direction: int = 0          # +1 bullish (long liq absorbed), -1 bearish
    confidence: float = 0.0

# ═══════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════

class OrderFlowMicrostructureFilter:
    """
    Real-time order flow imbalance and microstructure anomaly detector.
    
    Features:
    1. Top-5 Bid/Ask depth imbalance with spoofing detection
    2. Liquidation wall absorption detector
    3. Integrated confidence scoring for trade bias
    
    Designed to be instantiated per-symbol in EnsembleStrategyPredictor.
    """
    
    def __init__(self,
                 symbol: str,
                 n_depth_levels: int = 5,
                 spoof_lookback_sec: float = 2.0,
                 spoof_threshold: float = 0.70,
                 exclusion_threshold: float = 0.90,
                 imbalance_smoothing: float = 0.3,     # EWMA alpha for imbalance
                 liq_spike_std: float = 3.0,           # Z-score threshold for liq spike
                 price_nonresponse_atr: float = 0.5,    # Max price move (ATR) for non-response
                 bid_depth_baseline_bars: int = 20,     # Bars for bid depth baseline
                 absorption_confidence_threshold: float = 0.60,
                 signal_decay_sec: float = 900.0):      # 15 minutes
        """
        Parameters:
            symbol: trading symbol (for logging)
            n_depth_levels: number of order book levels to track
            spoof_lookback_sec: window for spoof detection stability check
            spoof_threshold: instability ratio threshold for spoof flagging
            exclusion_threshold: ratio above which level is fully excluded
            imbalance_smoothing: EWMA smoothing factor for imbalance series
            liq_spike_std: min z-score for liquidation spike classification
            price_nonresponse_atr: max price move (in ATR) to qualify as "non-response"
            bid_depth_baseline_bars: bars to compute bid depth rolling baseline
            absorption_confidence_threshold: min score for absorption signal
            signal_decay_sec: how long an absorption signal stays valid
        """
        self.symbol = symbol
        self.n_levels = n_depth_levels
        self.spoof_lookback = spoof_lookback_sec
        self.spoof_threshold = spoof_threshold
        self.exclusion_threshold = exclusion_threshold
        self.imbalance_alpha = imbalance_smoothing
        self.liq_spike_std = liq_spike_std
        self.price_nonresponse_atr = price_nonresponse_atr
        self.bid_baseline_bars = bid_depth_baseline_bars
        self.absorption_threshold = absorption_confidence_threshold
        self.signal_decay = signal_decay_sec
        
        # ── Order book state ──
        self._depth_history: deque = deque(maxlen=50)        # Depth snapshots
        self._level_max_qty: Dict[str, Dict[float, float]] = {}  # level_key → {timestamp → qty}
        self._level_quantities: deque = deque(maxlen=20)      # Rolling window of (bid_qtys, ask_qtys)
        
        # ── Smoothed imbalance series ──
        self._imbalance_ewma: float = 0.0
        self._imbalance_zscore: float = 0.0
        self._imbalance_history: deque = deque(maxlen=100)
        
        # ── Liquidation and CVD tracking ──
        self._liq_long_history: deque = deque(maxlen=100)
        self._liq_short_history: deque = deque(maxlen=100)
        self._cvd_history: deque = deque(maxlen=100)
        self._price_delta_history: deque = deque(maxlen=100)
        self._atr_history: deque = deque(maxlen=100)
        self._bid_depth_history: deque = deque(maxlen=bid_depth_baseline_bars)
        
        # ── Latest signals ──
        self._latest_depth: DepthSnapshot = DepthSnapshot()
        self._latest_absorption: AbsorptionSignal = AbsorptionSignal()
        self._depth_valid: bool = False
        self._absorption_valid: bool = False
        
        # ── Thread safety ──
        self._lock = threading.Lock()
        
        log.info(f"[OrderFlow] {symbol}: initialized (levels={n_depth_levels}, "
                 f"spoof_lookback={spoof_lookback_sec}s)")
    
    # ══════════════════════════════════════════════════════════════════════
    # TOP-5 DEPTH IMBALANCE WITH SPOOFING DETECTION
    # ══════════════════════════════════════════════════════════════════════
    
    def update_depth(self, 
                     bid_prices: List[float], bid_quantities: List[float],
                     ask_prices: List[float], ask_quantities: List[float],
                     timestamp: float = None) -> DepthSnapshot:
        """
        Process a new order book depth snapshot from Binance.
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            n_bids = min(len(bid_prices), len(bid_quantities), self.n_levels)
            n_asks = min(len(ask_prices), len(ask_quantities), self.n_levels)
            
            # ── Build BookLevel objects ──
            bids = []
            for i in range(n_bids):
                qty = float(bid_quantities[i])
                price = float(bid_prices[i])
                bids.append(BookLevel(price=price, quantity=qty, 
                                      notional=price * qty))
            
            asks = []
            for i in range(n_asks):
                qty = float(ask_quantities[i])
                price = float(ask_prices[i])
                asks.append(BookLevel(price=price, quantity=qty,
                                      notional=price * qty))
            
            # ── Spoofing detection per level ──
            spoof_count = 0
            snapshot_bid_qtys = []
            snapshot_ask_qtys = []
            
            for i, level in enumerate(bids):
                key = f"bid_{i}"
                spoof_score = self._compute_spoof_score(key, level.quantity, timestamp)
                level.spoof_score = spoof_score
                level.weight = max(0.0, 1.0 - spoof_score)
                if spoof_score > self.exclusion_threshold:
                    level.weight = 0.0
                    spoof_count += 1
                snapshot_bid_qtys.append((level.quantity, level.weight))
            
            for i, level in enumerate(asks):
                key = f"ask_{i}"
                spoof_score = self._compute_spoof_score(key, level.quantity, timestamp)
                level.spoof_score = spoof_score
                level.weight = max(0.0, 1.0 - spoof_score)
                if spoof_score > self.exclusion_threshold:
                    level.weight = 0.0
                    spoof_count += 1
                snapshot_ask_qtys.append((level.quantity, level.weight))
            
            # ── Compute imbalances ──
            total_bid_notional = sum(l.notional for l in bids)
            total_ask_notional = sum(l.notional for l in asks)
            
            # Raw imbalance (unweighted)
            raw_bid_vol = sum(l.quantity for l in bids)
            raw_ask_vol = sum(l.quantity for l in asks)
            denom_raw = raw_bid_vol + raw_ask_vol
            raw_imb = (raw_bid_vol - raw_ask_vol) / max(denom_raw, 1e-10)
            
            # Dollar-weighted imbalance
            denom_dollar = total_bid_notional + total_ask_notional
            dollar_imb = (total_bid_notional - total_ask_notional) / max(denom_dollar, 1e-10)
            
            # Adjusted imbalance (spoof-filtered)
            adj_bid_notional = sum(l.notional * l.weight for l in bids)
            adj_ask_notional = sum(l.notional * l.weight for l in asks)
            denom_adj = adj_bid_notional + adj_ask_notional
            adj_imb = (adj_bid_notional - adj_ask_notional) / max(denom_adj, 1e-10)
            
            # ── EWMA smoothing ──
            self._imbalance_ewma = (self.imbalance_alpha * adj_imb + 
                                    (1.0 - self.imbalance_alpha) * self._imbalance_ewma)
            self._imbalance_history.append(self._imbalance_ewma)
            
            # ── Z-score of imbalance ──
            if len(self._imbalance_history) >= 10:
                hist_arr = np.array(self._imbalance_history)
                mu = np.mean(hist_arr)
                sigma = np.std(hist_arr) + 1e-10
                self._imbalance_zscore = (self._imbalance_ewma - mu) / sigma
            
            # ── Store snapshot ──
            snap = DepthSnapshot(
                timestamp=timestamp,
                bids=bids, asks=asks,
                raw_imbalance=raw_imb,
                dollar_imbalance=dollar_imb,
                adjusted_imbalance=adj_imb,
                spoof_count=spoof_count,
            )
            self._depth_history.append(snap)
            self._level_quantities.append((snapshot_bid_qtys, snapshot_ask_qtys))
            self._latest_depth = snap
            self._depth_valid = True
            
            return snap
    
    def update_depth_from_coinglass(self,
                                     coins_bid: float, coins_ask: float,
                                     dollars_bid: float, dollars_ask: float,
                                     timestamp: float = None) -> DepthSnapshot:
        """
        Fallback: approximate depth imbalance from Coinglass DOM data.
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            bids = [BookLevel(price=0.0, quantity=coins_bid, 
                               notional=max(dollars_bid, coins_bid * 0))] if coins_bid > 0 else []
            asks = [BookLevel(price=0.0, quantity=coins_ask,
                               notional=max(dollars_ask, coins_ask * 0))] if coins_ask > 0 else []
            
            total_bid = dollars_bid if dollars_bid > 0 else coins_bid
            total_ask = dollars_ask if dollars_ask > 0 else coins_ask
            denom = total_bid + total_ask
            
            imbalance = (total_bid - total_ask) / max(denom, 1e-10) if denom > 0 else 0.0
            
            self._imbalance_ewma = (self.imbalance_alpha * imbalance + 
                                    (1.0 - self.imbalance_alpha) * self._imbalance_ewma)
            self._imbalance_history.append(self._imbalance_ewma)
            
            snap = DepthSnapshot(
                timestamp=timestamp,
                bids=bids, asks=asks,
                raw_imbalance=imbalance,
                dollar_imbalance=imbalance,
                adjusted_imbalance=imbalance,
            )
            self._latest_depth = snap
            self._depth_valid = True
            return snap
    
    def _compute_spoof_score(self, level_key: str, current_qty: float, 
                              timestamp: float) -> float:
        """Compute spoofing likelihood for a single order book level."""
        if level_key not in self._level_max_qty:
            self._level_max_qty[level_key] = {}
        
        max_data = self._level_max_qty[level_key]
        
        cutoff = timestamp - self.spoof_lookback
        stale_keys = [t for t in max_data if t < cutoff]
        for k in stale_keys:
            del max_data[k]
        
        max_data[timestamp] = current_qty
        
        values = list(max_data.values())
        if len(values) < 2:
            return 0.0
        
        max_val = max(values)
        min_val = min(values)
        range_val = max_val - min_val
        
        if max_val <= 0:
            return 0.0
        
        spoof_score = min(1.0, range_val / max(max_val, 1e-10))
        return float(spoof_score)
    
    # ══════════════════════════════════════════════════════════════════════
    # LIQUIDATION WALL ABSORPTION DETECTOR
    # ══════════════════════════════════════════════════════════════════════
    
    def update_liquidation_cascade(self,
                                    liq_long: float, liq_short: float,
                                    cvd: float, price_delta: float,
                                    atr: float, bid_depth: float,
                                    timestamp: float = None) -> AbsorptionSignal:
        """
        Detect if a liquidation cascade is being absorbed by limit buyers.
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            self._liq_long_history.append(liq_long)
            self._liq_short_history.append(liq_short)
            self._cvd_history.append(cvd)
            self._price_delta_history.append(price_delta)
            self._atr_history.append(atr)
            self._bid_depth_history.append(bid_depth)
            
            if len(self._liq_long_history) < 30:
                return AbsorptionSignal()
            
            liq_arr = np.array(self._liq_long_history)
            liq_mu = np.mean(liq_arr[:-1])
            liq_sigma = np.std(liq_arr[:-1]) + 1e-10
            liq_long_z = (liq_long - liq_mu) / liq_sigma
            
            liq_s_arr = np.array(self._liq_short_history)
            liq_s_mu = np.mean(liq_s_arr[:-1])
            liq_s_sigma = np.std(liq_s_arr[:-1]) + 1e-10
            liq_short_z = (liq_short - liq_s_mu) / liq_s_sigma
            
            liq_spike_z = max(liq_long_z, liq_short_z)
            is_long_liq_spike = liq_long_z > liq_short_z
            
            atr_val = max(atr, 1e-10)
            price_response_atr = abs(price_delta) / atr_val
            
            cvd_arr = np.array(self._cvd_history)
            if len(cvd_arr) >= 3:
                cvd_diffs = np.diff(cvd_arr)
                diff_mu = np.mean(cvd_diffs[:-1])
                diff_sigma = np.std(cvd_diffs[:-1]) + 1e-10
                cvd_delta = cvd_diffs[-1]
                cvd_delta_z = (cvd_delta - diff_mu) / diff_sigma
            else:
                cvd_delta_z = 0.0
            
            bid_depth_arr = np.array(self._bid_depth_history)
            if len(bid_depth_arr) >= self.bid_baseline_bars:
                bid_baseline = np.mean(bid_depth_arr[-self.bid_baseline_bars:-1])
                bid_depth_delta = bid_depth / max(bid_baseline, 1e-10)
            else:
                bid_depth_delta = 1.0
            
            absorption = False
            signal_direction = 0
            absorption_score = 0.0
            
            if liq_spike_z > self.liq_spike_std:
                if is_long_liq_spike:
                    price_pass = price_response_atr < self.price_nonresponse_atr
                    cvd_pass = cvd_delta_z > -1.0
                    depth_pass = bid_depth_delta > 1.2
                    
                    if price_pass and (cvd_pass or depth_pass):
                        absorption = True
                        signal_direction = 1  # Bullish
                        
                        price_score = max(0.0, 1.0 - price_response_atr / self.price_nonresponse_atr)
                        liq_score = min(1.0, liq_spike_z / 5.0)
                        cvd_score = max(0.0, min(1.0, 1.0 + cvd_delta_z / 3.0))
                        depth_score = min(1.0, bid_depth_delta / 3.0)
                        
                        absorption_score = (liq_score * 0.35 + price_score * 0.25 + 
                                           cvd_score * 0.20 + depth_score * 0.20)
                else:
                    price_pass = price_response_atr < self.price_nonresponse_atr
                    cvd_pass = cvd_delta_z < 1.0
                    depth_pass = bid_depth_delta < 0.8
                    
                    if price_pass and (cvd_pass or depth_pass):
                        absorption = True
                        signal_direction = -1  # Bearish
                        
                        price_score = max(0.0, 1.0 - price_response_atr / self.price_nonresponse_atr)
                        liq_score = min(1.0, liq_spike_z / 5.0)
                        cvd_score = max(0.0, min(1.0, 1.0 - cvd_delta_z / 3.0))
                        depth_score = min(1.0, (1.0 - bid_depth_delta) / 0.5)
                        
                        absorption_score = (liq_score * 0.35 + price_score * 0.25 + 
                                           cvd_score * 0.20 + depth_score * 0.20)
        
            signal = AbsorptionSignal(
                timestamp=timestamp,
                detected=absorption and absorption_score > self.absorption_threshold,
                liq_spike_z=liq_spike_z,
                price_response_atr=price_response_atr,
                cvd_confirmation=cvd_delta_z,
                bid_depth_delta=bid_depth_delta,
                absorption_score=absorption_score,
                signal_direction=signal_direction,
                confidence=absorption_score,
            )
            
            self._latest_absorption = signal
            self._absorption_valid = signal.detected
            return signal
    
    def get_imbalance(self) -> float:
        return self._imbalance_ewma
    
    def get_imbalance_zscore(self) -> float:
        return self._imbalance_zscore
    
    def is_imbalance_extreme(self, threshold: float = 2.0) -> bool:
        return abs(self._imbalance_zscore) > threshold
    
    def get_imbalance_direction(self) -> int:
        if not self.is_imbalance_extreme():
            return 0
        return 1 if self._imbalance_zscore > 0 else -1
    
    def is_absorption_active(self) -> bool:
        if not self._absorption_valid:
            return False
        age = time.time() - self._latest_absorption.timestamp
        return age < self.signal_decay
    
    def get_absorption_signal(self) -> AbsorptionSignal:
        if not self.is_absorption_active():
            return AbsorptionSignal()
        return self._latest_absorption
    
    def get_trade_bias(self) -> Tuple[int, float]:
        bias = 0
        confidence = 0.0
        
        if self.is_absorption_active():
            abs_sig = self._latest_absorption
            if abs_sig.absorption_score > 0.60:
                bias = abs_sig.signal_direction
                confidence = max(confidence, abs_sig.absorption_score)
        
        if self.is_imbalance_extreme(threshold=2.5):
            imb_dir = self.get_imbalance_direction()
            imb_conf = min(0.5, abs(self._imbalance_zscore) / 5.0)
            
            if bias == 0:
                bias = imb_dir
                confidence = imb_conf
            elif bias == imb_dir:
                confidence = max(confidence, min(0.9, confidence + imb_conf * 0.5))
        
        if self._depth_valid and self._latest_depth.spoof_count > 0:
            total_levels = len(self._latest_depth.bids) + len(self._latest_depth.asks)
            if total_levels > 0:
                spoof_ratio = self._latest_depth.spoof_count / total_levels
                if spoof_ratio > 0.3:
                    confidence *= max(0.3, 1.0 - spoof_ratio)
        
        return (bias, confidence)
