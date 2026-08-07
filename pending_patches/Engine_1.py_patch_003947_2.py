Python# TARGET: Engine_1.py
# ═══════════════════════════════════════════════════════════════════
# PATCH 2 — FIX SnapshotStore to propagate ATR from predictor
# FIND in SnapshotStore.update():
#     cur_atr = getattr(new_snap, 'atr', 0.0)
# REPLACE with ATR lookup from predictor cache:
# ═══════════════════════════════════════════════════════════════════

            # Run exit checks — propagate latest ATR from predictor
            if self.trade_tracker and "price" in clean_patch:
                cur_atr = 0.0
                if self.predictor and hasattr(self.predictor, 'latest_atr'):
                    cur_atr = self.predictor.latest_atr.get(symbol, 0.0)
                self.trade_tracker.check_exits(
                    symbol, new_snap.price, current_atr=cur_atr)
                self.trade_tracker.update_live_pnl(symbol, new_snap.price)