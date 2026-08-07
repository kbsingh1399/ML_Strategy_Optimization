Python# ensemble_strategy_predictor.py, chunk 4
STATIC_WR = {"S1...": 0.783, "S2...": 0.795, ... "S7...": 0.720}

class EnsembleAggregator:
    def _compute_live_weights(self):
        # Blends static WR with live WR:
        # 70/30 at 10 trades → 50/50 at 30+ trades
        # Floors at 0.40 — never zero out a strategy
        blend = min(0.50, 0.30 + 0.20 * ((n - 10) / 20.0))
        blended_wr = static_wr * (1.0 - blend) + live_wr * blend
        return max(0.40, blended_wr)