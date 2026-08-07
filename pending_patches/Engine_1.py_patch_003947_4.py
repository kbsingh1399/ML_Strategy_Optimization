Python# TARGET: Engine_1.py
# ═══════════════════════════════════════════════════════════════════
# PATCH 4 — ADD bi-directional ATR widening when volatility compresses
# FIND in check_exits() the ATR tightening block:
#     if (entry_atr > 0 and current_atr > 0 and
#             current_atr > entry_atr * 1.30):
# AFTER the tightening block, ADD a widening block for vol compression:
# ═══════════════════════════════════════════════════════════════════

                # ── ATR widening when volatility compresses ──────
                # If vol drops below 60% of entry ATR, price noise is
                # reduced → the original stop may be too loose, but we
                # leave it alone.  However, if the SL was PREVIOUSLY
                # tightened and vol now compresses, restore 85% of
                # original distance to avoid premature stop-outs.
                if (entry_atr > 0 and current_atr > 0 and
                        current_atr < entry_atr * 0.60):
                    orig_sl_dist = trade.get('sl_dist', abs(entry_price - sl))
                    cur_sl_dist = abs(entry_price - trade['sl'])
                    # Only widen if stop was previously tightened
                    if cur_sl_dist < orig_sl_dist * 0.95:
                        restore_dist = max(orig_sl_dist * 0.85,
                                           entry_price * 0.003)
                        if direction == 1:
                            trade['sl'] = entry_price - restore_dist
                        else:
                            trade['sl'] = entry_price + restore_dist
                        sl = trade['sl']
                        log.debug(
                            f"[ATR-Widen] {trade['trade_id']}: "
                            f"vol compressed below 60% → SL widened "
                            f"to {sl:.4f}"
                        )