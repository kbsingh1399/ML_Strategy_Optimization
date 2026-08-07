Python# TARGET: Engine_1.py — ALREADY PRESENT at 92f33909
# Tighten when vol > 1.30x entry ATR (15% reduction, 0.3% floor)
entry_atr = trade.get('atr', 0.0)
if (entry_atr > 0 and current_atr > 0 and
        current_atr > entry_atr * 1.30):
    old_sl = trade['sl']
    old_sl_dist = abs(entry_price - old_sl)
    min_sl_dist = entry_price * 0.003
    new_sl_dist = max(old_sl_dist * 0.85, min_sl_dist)
    if direction == 1:
        trade['sl'] = entry_price - new_sl_dist
    else:
        trade['sl'] = entry_price + new_sl_dist
    sl = trade['sl']
    log.debug(f"[ATR-Tighten] {trade['trade_id']}: "
              f"entry_ATR={entry_atr:.4f} cur_ATR={current_atr:.4f} "
              f"(ratio={current_atr/entry_atr:.2f}) → "
              f"SL tightened from {old_sl:.4f} to {sl:.4f}")

# Widen when vol < 0.60x entry ATR (restore 85% of original)
if (entry_atr > 0 and current_atr > 0 and
        current_atr < entry_atr * 0.60):
    orig_sl_dist = trade.get('sl_dist', abs(entry_price - sl))
    cur_sl_dist = abs(entry_price - trade['sl'])
    if cur_sl_dist < orig_sl_dist * 0.95:
        restore_dist = max(orig_sl_dist * 0.85, entry_price * 0.003)
        if direction == 1:
            trade['sl'] = entry_price - restore_dist
        else:
            trade['sl'] = entry_price + restore_dist
        sl = trade['sl']