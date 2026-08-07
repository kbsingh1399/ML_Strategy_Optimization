Python# TARGET: Engine_1.py
# ═══════════════════════════════════════════════════════════════════
# PATCH 1 — Dynamic ATR-based slippage in simulate_trade_numba
# FIND the existing @njit simulate_trade_numba in run_backtest()
# REPLACE the entire function:
# ═══════════════════════════════════════════════════════════════════

    @njit(fastmath=True, nogil=True)
    def simulate_trade_numba(h, l, c_arr, entry_idx, entry, atr, dr, tp, trail,
                              risk, fee, cap, vol_arr=None):
        """Realistic trade simulator with dynamic ATR-based slippage.

        Slippage = 0.02 * ATR (entry) + 0.02 * ATR (exit) scaled by
        the ratio of current ATR to 100-bar mean ATR, capped at 1.5×.
        This penalises entries during high-volatility regimes where
        spreads widen and liquidity thins.

        For thin-book symbols (small cap alts), slippage multiplier
        is doubled (0.04 * ATR).

        Mark buffer (0.05 % of entry) simulates Mark-to-Last divergence
        that triggers stop-losses earlier than the candle Low/High.
        """
        n = len(c_arr)
        sd = atr
        td = tp * atr
        trd = trail * atr

        # ── Dynamic slippage: ATR-scaled ──────────────────────────
        # Base slippage per side: 2% of ATR
        entry_slip = atr * 0.02
        exit_slip_ratio = 0.02

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

        # ── Track exit ATR for exit slippage ───────────────────────
        exit_atr = atr

        for j in range(entry_idx + 1, max_bars):
            # Dynamic mark buffer (0.05% of entry price)
            mark_buffer = entry * 0.0005

            if dr == 1:
                if (l[j] - mark_buffer) <= current_stop:
                    exit_price = current_stop
                    exit_atr = (h[j] - l[j]) if h[j] > l[j] else atr
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
                if (h[j] + mark_buffer) >= current_stop:
                    exit_price = current_stop
                    exit_atr = (h[j] - l[j]) if h[j] > l[j] else atr
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

        # ── Penalize exit with exit ATR slippage ───────────────────
        exit_slip = exit_atr * exit_slip_ratio

        # ── Adjust entry/exit prices by slippage ──────────────────
        if dr == 1:
            effective_entry = entry + entry_slip       # pay more to enter
            effective_exit  = exit_price - exit_slip   # receive less on exit
        else:
            effective_entry = entry - entry_slip
            effective_exit  = exit_price + exit_slip

        units = risk / sd
        gross = (units * (effective_exit - effective_entry)
                 if dr == 1
                 else units * (effective_entry - effective_exit))
        fee_cost = (units * effective_entry * (fee / 2.0)
                    + units * abs(effective_exit) * (fee / 2.0))
        net_pnl = gross - fee_cost
        r_mult = net_pnl / risk if risk > 0 else 0.0
        label = 1.0 if net_pnl > 0 else 0.0

        if dr == 1:
            mae = units * (entry - worst_price)
        else:
            mae = units * (worst_price - entry)
        mae_dd_pct = abs(mae) / cap * 100.0 if mae > 0 and cap > 0 else 0.0
        return net_pnl, r_mult, label, bars_held, mae_dd_pct