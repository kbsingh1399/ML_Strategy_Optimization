Python# TARGET: Engine_1.py
# ═══════════════════════════════════════════════════════════════════
# PATCH 3 — POSITION RECONCILIATION restart count carry-forward
# FIND in Engine1TradeTracker.load_history(), near the end where
# it recovers active trades:
#     for t in trades:
#         if not t.get('exit_price') and t.get('trade_id'):
#             self.active_trades[t['trade_id']] = t.copy()
# ADD after that block, before the `except`:
# ═══════════════════════════════════════════════════════════════════

                # Restore anti-martingale counter from recent losses
                recent_losses = 0
                for t in reversed(self.history):
                    if t.get('pnl_usd', 0.0) < 0:
                        recent_losses += 1
                    else:
                        break
                if recent_losses > 0:
                    self.consecutive_losses = recent_losses
                    log.info(
                        f"[History] Restored consecutive-loss counter: "
                        f"{self.consecutive_losses} from saved trades"
                    )