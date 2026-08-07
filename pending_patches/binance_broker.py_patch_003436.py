Python# TARGET: binance_broker.py
# REPLACE the entire execute_trade method with this optimized version:

    def execute_trade(self, symbol: str, direction: int, entry_price: float,
                      sl: float, tp: float, strategy: str,
                      risk_capital: float) -> Optional[dict]:
        """Complete trade lifecycle with fee optimization:

        1. [GATE] Minimum profit threshold (EV > 2× fees + slippage)
        2. [SPLIT] Slice order if notional ≥ $5K
        3. [MAKER] Post-only LIMIT+GTX (earn −0.02% rebate)
           → fallback to MARKET on timeout
        4. [SL/TP] Algo orders with priceProtect + MARK_PRICE
        """
        with self.lock:
            pos_side = "LONG" if direction == 1 else "SHORT"
            stop_dist = abs(entry_price - sl)
            if stop_dist <= 0:
                log.error(f"[Binance] Invalid stop distance for {symbol}")
                return None

            quantity = _round_qty(symbol, risk_capital / stop_dist)

            if symbol not in self.symbol_info_cache:
                self.symbol_info_cache[symbol] = _get_symbol_info(symbol)
            symbol_info = self.symbol_info_cache[symbol]

            notional = quantity * entry_price
            min_notional = _min_qty(symbol, symbol_info)
            if notional < min_notional:
                quantity = _round_qty(symbol, min_notional / entry_price, symbol_info)

            if quantity <= 0:
                return None

            # ── GATE 1: Profit threshold ────────────────────────────
            passes, reason = self._validate_profit_threshold(
                symbol, entry_price, tp, sl, quantity, direction)
            if not passes:
                log.warning(f"[Binance] Trade REJECTED — {reason}")
                return None

            # ── GATE 2: Slice ───────────────────────────────────────
            slices = self._slice_quantity(symbol, quantity, entry_price)
            n_slices = len(slices)

            # ── GATE 3: LIMIT+GTX → MARKET ──────────────────────────
            tick_size = 0.01
            for f in symbol_info.get('filters', []):
                if f.get('filterType') == 'PRICE_FILTER':
                    tick_size = float(f.get('tickSize', 0.01))
                    break

            limit_price = (entry_price - tick_size if direction == 1
                           else entry_price + tick_size)

            entry_result = None
            total_filled_qty = 0.0
            all_order_ids = []

            for slice_idx, slice_qty in enumerate(slices):
                if slice_idx > 0:
                    time.sleep(self.inter_slice_delay_secs)

                if n_slices == 1 or slice_idx == 0:
                    limit_result = self.place_entry_limit_post_only(
                        symbol, pos_side, slice_qty, limit_price)
                    if limit_result and not limit_result.get('error'):
                        order_id = limit_result.get('orderId')
                        if order_id:
                            t0 = time.time()
                            filled = False
                            while time.time() - t0 < self.post_only_timeout_secs:
                                time.sleep(0.3)
                                if self._check_order_filled(symbol, order_id):
                                    filled = True
                                    break
                            if filled:
                                entry_result = limit_result
                                total_filled_qty += slice_qty
                                all_order_ids.append(order_id)
                                log.info(f"[Binance] LIMIT+GTX filled "
                                         f"slice {slice_idx+1}/{n_slices} "
                                         f"(maker rebate: {MAKER_FEE*100:+.3f}%)")
                                continue
                            else:
                                self._cancel_limit_order(symbol, order_id)

                    mkt_result = self.place_entry_market(symbol, pos_side, slice_qty)
                    if not mkt_result:
                        return None
                    entry_result = mkt_result
                else:
                    mkt_result = self.place_entry_market(symbol, pos_side, slice_qty)
                    if not mkt_result:
                        return None
                    if mkt_result.get('orderId'):
                        all_order_ids.append(mkt_result.get('orderId'))

                total_filled_qty += slice_qty

            if total_filled_qty <= 0:
                return None

            # ── SL/TP algos ─────────────────────────────────────────
            trade_id = f"{strategy}_{symbol}_{pos_side}_{int(time.time()*1000)}"
            actual_entry = float(
                entry_result.get('avgPrice', entry_result.get('price', entry_price))
                if entry_result else entry_price)
            if actual_entry == 0.0:
                actual_entry = entry_price

            sl_result, tp_result = self.place_sltp_algo(
                symbol, pos_side, actual_entry, sl, tp, trade_id)

            log.info(f"[Binance] TRADE: {trade_id} {pos_side} {symbol} "
                     f"qty={total_filled_qty:.4f} entry≈{actual_entry:.4f} "
                     f"notional≈${total_filled_qty*actual_entry:,.0f} slices={n_slices}")

            return {
                "mt5_symbol": symbol,
                "mt5_ticket": all_order_ids[0] if all_order_ids else None,
                "mt5_entry": actual_entry, "mt5_sl": sl, "mt5_tp": tp,
                "lot": total_filled_qty, "trade_id": trade_id,
                "sl_algo_id": sl_result.get('algoId') if sl_result else None,
                "tp_algo_id": tp_result.get('algoId') if tp_result else None,
                "is_pending": False, "order_ids": all_order_ids, "n_slices": n_slices,
            }