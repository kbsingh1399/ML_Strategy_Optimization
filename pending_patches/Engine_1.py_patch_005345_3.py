Python# TARGET: Engine_1.py
# ═══════════════════════════════════════════════════════════════════
# PATCH 3 — Ensure SnapshotStore propagates ATR from predictor cache
# FIND in SnapshotStore.update():
#     cur_atr = getattr(new_snap, 'atr', 0.0)
# REPLACE with:
# ═══════════════════════════════════════════════════════════════════

            cur_atr = 0.0
            if self.predictor and hasattr(self.predictor, 'latest_atr'):
                cur_atr = self.predictor.latest_atr.get(symbol, 0.0)
            self.trade_tracker.check_exits(
                symbol, new_snap.price, current_atr=cur_atr)