Python# TARGET: ensemble_strategy_predictor.py
# ═══════════════════════════════════════════════════════════════════
# PATCH 2 — Dynamic EWMA live-accuracy weights in EnsembleAggregator
# FIND:  class EnsembleAggregator:
# REPLACE the aggregate() method AND add _compute_live_weights():
# ═══════════════════════════════════════════════════════════════════

class EnsembleAggregator:
    def __init__(self, cfg=None, active_strategies=None):
        self.cfg = cfg or StrategyConfig()
        self.lock = threading.RLock()
        self.last_trade_time: Dict[str, datetime] = {}
        self.active_strategies = active_strategies if active_strategies is not None else ALL_STRATEGY_KEYS
        self._eff_min_agree = min(self.cfg.min_agreeing, len(self.active_strategies))
        self._strategy_r_history: Dict[str, List[float]] = {s: [] for s in ALL_STRATEGY_KEYS}
        # ── Live accuracy tracking ──────────────────────────────
        self._strategy_wins: Dict[str, int] = {s: 0 for s in ALL_STRATEGY_KEYS}
        self._strategy_total: Dict[str, int] = {s: 0 for s in ALL_STRATEGY_KEYS}
        self._live_wr: Dict[str, float] = {s: STATIC_WR.get(s, 0.70) for s in ALL_STRATEGY_KEYS}
        self._min_samples_for_live_wr: int = 10  # need 10 trades before trusting live WR

    def record_strategy_outcome(self, strategy_name: str, r_mult: float):
        with self.lock:
            if strategy_name in self._strategy_r_history:
                self._strategy_r_history[strategy_name].append(r_mult)
                if len(self._strategy_r_history[strategy_name]) > 50:
                    self._strategy_r_history[strategy_name].pop(0)
                # Track win/loss for EWMA accuracy
                self._strategy_total[strategy_name] += 1
                if r_mult > 0:
                    self._strategy_wins[strategy_name] += 1
            elif strategy_name == "Ensemble_6Strategy":
                for s in self.active_strategies:
                    if s not in self._strategy_r_history:
                        self._strategy_r_history[s] = []
                    self._strategy_r_history[s].append(r_mult)
                    if len(self._strategy_r_history[s]) > 50:
                        self._strategy_r_history[s].pop(0)
                    self._strategy_total[s] += 1
                    if r_mult > 0:
                        self._strategy_wins[s] += 1

    def _compute_live_weights(self) -> Dict[str, float]:
        """Compute EWMA live-accuracy weights.
        Blends static backtest WR (70%) with live WR (30%) after
        minimum samples, transitioning to 50/50 after 30 trades.
        """
        weights = {}
        for name in self.active_strategies:
            if name not in STATIC_WR:
                weights[name] = 0.75
                continue
            static_wr = STATIC_WR[name]
            n = self._strategy_total.get(name, 0)
            if n < self._min_samples_for_live_wr:
                weights[name] = static_wr
                continue
            live_wr = (self._strategy_wins.get(name, 0) / max(n, 1))
            # Blend ratio: 70% static / 30% live at 10 trades,
            #            50% / 50% at 30+ trades
            blend = min(0.50, 0.30 + 0.20 * ((n - 10) / 20.0))
            blended_wr = static_wr * (1.0 - blend) + live_wr * blend
            # Floor at 0.40 — never completely zero a strategy
            weights[name] = max(0.40, blended_wr)
        return weights

    def aggregate(self, strategy_signals: Dict[str, int]) -> Tuple[int, float, int]:
        with self.lock:
            filtered = {k: v for k, v in strategy_signals.items() if k in self.active_strategies}
            longs = sum(1 for s in filtered.values() if s == 1)
            shorts = sum(1 for s in filtered.values() if s == -1)
            total = len(filtered)
            if total < 1:
                return 0, 0.0, 0

            # ── Dynamic live-accuracy weights ─────────────────────
            live_weights = self._compute_live_weights()

            weighted_long = 0.0
            weighted_short = 0.0
            for name, sig in filtered.items():
                w = live_weights.get(name, STATIC_WR.get(name, 0.70))
                if sig == 1:
                    weighted_long += w
                elif sig == -1:
                    weighted_short += w

            total_weight = sum(live_weights.get(n, 0.70)
                               for n in filtered)
            if total_weight == 0:
                return 0, 0.0, 0

            net_score = (weighted_long - weighted_short) / total_weight

            if net_score > 0.2:
                return 1, min(1.0, weighted_long / max(total_weight, 0.1)), longs
            elif net_score < -0.2:
                return -1, min(1.0, weighted_short / max(total_weight, 0.1)), shorts
            return 0, abs(net_score) * 5, 0

    def should_enter(self, direction, confidence, agreeing):
        return (confidence >= self.cfg.min_confidence and
                agreeing >= self._eff_min_agree and direction != 0)

    def get_ml_signals_dict(self, strategy_signals, direction, confidence):
        return {n: {'prob_score': confidence,
                     'trigger_threshold': self.cfg.min_confidence,
                     'key_feature': 'direction',
                     'key_feature_val': s}
                for n, s in strategy_signals.items() if n in STRATEGIES}


# ── Static backtest win rates (fallback before live data) ──────────
STATIC_WR = {
    "S1_Liquidation": 0.783,
    "S2_CVD_Momentum": 0.795,
    "S3_Trend_Follow": 0.707,
    "S4_Mean_Reversion": 0.754,
    "S5_Vol_Expansion": 0.718,
    "S6_OI_Momentum": 0.797,
    "S7_CVD_Divergence": 0.720,
}