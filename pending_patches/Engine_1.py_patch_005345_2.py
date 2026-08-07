Python# TARGET: Engine_1.py
# ═══════════════════════════════════════════════════════════════════
# PATCH 2 — Wire liq_cascade short-block into _run_inference 
# FIND in _run_inference() (in ensemble_strategy_predictor.py, not
# Engine_1.py — the predictor's inference):
#     if not self.ensemble.should_enter(direction, confidence, agreeing): return
#     if trade_tracker is None: return
# ADD between them:
# ═══════════════════════════════════════════════════════════════════

            # ── Cascade guard: block SHORTS into long-liq spikes ──
            liq_cascade = 0
            if 'liq_cascade' in dff.columns:
                liq_cascade = int(dff['liq_cascade'].values[-1])
            if direction == -1 and liq_cascade:
                log.warning(
                    f"[CASCADE] SHORT blocked for {symbol}: "
                    f"long-liq cascade active"
                )
                return