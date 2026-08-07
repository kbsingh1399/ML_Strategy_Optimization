Python# TARGET: Engine_1.py
# ═══════════════════════════════════════════════════════════════════
# PATCH 3 — Fast-path stop-loss check using symbol-indexed lookup.
# REPLACE check_exits() in Engine1TradeTracker with this version.
# Also ADD a _symbol_to_trade_ids index that is kept in sync.
# ═══════════════════════════════════════════════════════════════════

# ── ADD inside Engine1TradeTracker.__init__(), after self.active_trades:
        # Fast path: symbol → [trade_id, ...] index
        self._sym_to_ids: Dict[str, List[str]] = defaultdict(list)

# ── ADD inside trigger_entry(), right after self.active_trades[trade_id] = {...}:
        self._sym_to_ids.setdefault(symbol, []).append(trade_id)

# ── ADD this helper method (after load_history):

    def _remove_trade(self, trade_id: str, symbol: str) -> None:
        """Remove trade from both dict and symbol index."""
        self.active_trades.pop(trade_id, None)
        ids = self._sym_to_ids.get(symbol, [])
        if trade_id in ids:
            ids.remove(trade_id)

# ── REPLACE the entire check_exits method with this fast-path version:

    def check_exits(self, symbol: str, current_price: float,
                    current_atr: float = 0.0) -> None:
        """Fast-path stop-loss check using symbol-indexed lookup.

        Instead of iterating ALL trades for every symbol update, we
        directly index into only the trades for `symbol`.  Bail early
        if no trades are active for this symbol or if the price hasn't
        crossed any threshold.

        Broker modify/close jobs are submitted outside the lock.
        """
        # ── Fast bail: no trades for this symbol ──────────────────
        trade_ids = self._sym_to_ids.get(symbol, [])
        if not trade_ids:
            return

        broker_modify_jobs: List[Tuple] = []
        broker_close_jobs: List[Tuple] = []
        any_closed = False

        with self.lock:
            # Re-filter: remove stale entries
            trades = []
            for tid in list(trade_ids):
                t = self.active_trades.get(tid)
                if t and not t.get('is_pending', False):
                    trades.append((tid, t))
                elif not t:
                    trade_ids.remove(tid)

            if not trades:
                return

            for tid, trade in trades:
                direction = trade['direction']
                sl  = trade['sl']
                tp  = trade['tp']
                ep  = trade['entry_price']
                sd  = trade.get('sl_dist')
                atr = current_atr if current_atr > 0 else trade.get('atr', 0.0)

                # ── Early skip: price hasn't moved meaningfully ──
                # Only check SL if price is within 10% of the stop level
                if sd and atr > 0:
                    sl_range = sd * 0.10
                    if direction == 1:
                        if current_price > sl + sl_range:
                            pass  # price well above SL — skip heavy checks
                    else:
                        if current_price < sl - sl_range:
                            pass

                # ── Trailing stop (simplified fast path) ──────────
                trail_act = trade.get('trail_act', 1.0)
                if sd and atr > 0:
                    if direction == 1:
                        cur_r = (current_price - ep) / sd
                        if cur_r >= trail_act:
                            trail_buf = trade.get('trail_buf', 0.5)
                            ns = ep + (cur_r - trail_buf) * sd
                            if ns > sl:
                                trade['sl'] = ns
                                sl = ns
                                broker_modify_jobs.append((symbol, ns, tp))
                    else:
                        cur_r = (ep - current_price) / sd
                        if cur_r >= trail_act:
                            trail_buf = trade.get('trail_buf', 0.5)
                            ns = ep - (cur_r - trail_buf) * sd
                            if ns < sl:
                                trade['sl'] = ns
                                sl = ns
                                broker_modify_jobs.append((symbol, ns, tp))

                # ── ATR dynamic tightening ────────────────────────
                entry_atr = trade.get('atr', 0.0)
                if (entry_atr > 0 and current_atr > 0 and
                        current_atr > entry_atr * 1.30):
                    old_sl_dist = abs(ep - sl)
                    min_dist = ep * 0.003
                    new_dist = max(old_sl_dist * 0.85, min_dist)
                    if direction == 1:
                        trade['sl'] = ep - new_dist
                    else:
                        trade['sl'] = ep + new_dist
                    sl = trade['sl']

                # ── Hard SL/TP check ──────────────────────────────
                hit = False
                if direction == 1:
                    if current_price <= sl:
                        hit, reason = True, "SL"
                    elif current_price >= tp:
                        hit, reason = True, "TP"
                else:
                    if current_price >= sl:
                        hit, reason = True, "SL"
                    elif current_price <= tp:
                        hit, reason = True, "TP"

                # ── Timeout (24h) ─────────────────────────────────
                if not hit:
                    elapsed = time.time() - trade.get('entry_timestamp', time.time())
                    if elapsed >= 86400:
                        hit, reason = True, "TIMEOUT"

                if not hit:
                    continue

                # ── Exit: book the fill ───────────────────────────
                exit_px = trade['sl'] if reason == "SL" \
                     else trade['tp'] if reason == "TP" \
                     else current_price

                trade['exit_price'] = exit_px
                trade['exit_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                trade['exit_reason'] = reason

                pnl_pct = ((exit_px - ep) / ep * 100.0
                           if direction == 1 else
                           (ep - exit_px) / ep * 100.0)
                pnl_usd = trade.get('units', 0.0) * (exit_px - ep) * direction

                trade['pnl_pct'] = pnl_pct
                trade['pnl_usd'] = pnl_usd

                self.history.append(trade)
                self.current_capital += pnl_usd
                if self.current_capital > self.peak_capital:
                    self.peak_capital = self.current_capital

                # Anti-martingale
                if pnl_usd < 0:
                    self.consecutive_losses += 1
                    scale = max(self.anti_martingale_floor,
                                self.anti_martingale_factor ** self.consecutive_losses)
                    log.info(f"[Risk] Consecutive losses: "
                             f"{self.consecutive_losses} → scale={scale:.0%}")
                else:
                    if self.consecutive_losses > 0:
                        log.info(f"[Risk] Win resets loss counter "
                                 f"(was {self.consecutive_losses})")
                    self.consecutive_losses = 0

                # Cooldown
                cooldown = (self.REENTRY_COOLDOWN_TP_SECS if reason == "TP"
                            else self.REENTRY_COOLDOWN_SL_SECS)
                if cooldown > 0:
                    key = self._cooldown_key(trade.get('strategy', ''), symbol)
                    self.reentry_cooldown_until[key] = time.time() + cooldown

                log.info(f"[EXIT] {trade['trade_id']}: {reason} @ {exit_px:.2f} "
                         f"PnL=${pnl_usd:.2f} ({pnl_pct:+.2f}%)")

                self._remove_trade(tid, symbol)
                broker_close_jobs.append((symbol, reason))
                any_closed = True

                # Callbacks
                strategy = trade.get('strategy', '')
                for cb in self.on_close_callbacks:
                    try:
                        cb(strategy, self.current_capital)
                    except Exception:
                        pass

            if any_closed:
                self.save_history()

        # ── Dispatch broker jobs outside lock ──────────────────────
        if self.broker_executor and self.broker and LIVE_TRADING:
            for sym, new_sl, tp in broker_modify_jobs:
                self.broker_executor.submit(
                    self.broker.modify_sltp, sym, 0, new_sl, tp)
            for sym, reason in broker_close_jobs:
                self.broker_executor.submit(
                    self.broker.close_position, sym, reason)