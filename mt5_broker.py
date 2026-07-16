import MetaTrader5 as mt5
import math
import threading
import time


class MT5Broker:
    """
    Low-level MT5 execution wrapper with:
    - reconnect + exponential backoff
    - order_send retries on transient retcodes
    - position/order ticket reconciliation helpers
    - tight slippage (deviation) clamps
    """

    # Transient retcodes worth retrying (requote / price off / connection)
    _RETRYABLE_RETCODES = {
        10004,  # REQUOTE
        10006,  # REJECT
        10012,  # TIMEOUT
        10020,  # PRICE_OFF
        10021,  # PRICE_CHANGED
        10024,  # TOO_MANY_REQUESTS
        10028,  # TRADE_TIMEOUT
        10031,  # CONNECTION
    }

    def __init__(self, dry_run=True, account_size=5000.0, risk_pct=0.005, symbol_map=None, max_abs_basis_pct=0.005):
        self.dry_run = dry_run
        self.account_size = account_size
        self.risk_pct = risk_pct
        self.risk_usd = self.account_size * self.risk_pct
        self.connected = False
        self._lock = threading.RLock()
        self._symbol_cache = {}
        self.symbol_map = symbol_map or {}
        self.max_abs_basis_pct = max_abs_basis_pct
        self._connect_failures = 0
        self._last_connect_attempt = 0.0
        self.magic = 234000

    def connect(self):
        with self._lock:
            now = time.time()
            # Backoff: 1s, 2s, 4s ... cap 30s between re-init attempts
            backoff = min(30.0, 2 ** min(self._connect_failures, 5))
            if self._connect_failures and (now - self._last_connect_attempt) < backoff:
                return False
            self._last_connect_attempt = now
            try:
                mt5.shutdown()
            except Exception:
                pass
            if not mt5.initialize():
                self._connect_failures += 1
                print(f"[MT5] initialize() failed (attempt {self._connect_failures}), error={mt5.last_error()}")
                self.connected = False
                return False
            self.connected = True
            self._connect_failures = 0
            print("[MT5] Connected to MetaTrader 5 successfully!")
            return True

    def ensure_connected(self) -> bool:
        with self._lock:
            if not self.connected:
                return self.connect()
            info = mt5.terminal_info()
            if info is None or not getattr(info, "connected", False):
                print("[MT5] Connection lost. Re-initializing connection...")
                self.connected = False
                return self.connect()
            return True

    def _order_send_with_retry(self, request: dict, max_retries: int = 3):
        """Send order with retry on transient broker errors; refresh price each attempt."""
        last_result = None
        for attempt in range(max_retries):
            if not self.ensure_connected():
                time.sleep(0.2 * (attempt + 1))
                continue
            # Refresh market price for market orders to cut slippage drift
            if request.get("action") == mt5.TRADE_ACTION_DEAL:
                tick = mt5.symbol_info_tick(request["symbol"])
                if tick is not None:
                    if request["type"] in (mt5.ORDER_TYPE_BUY,):
                        request["price"] = float(tick.ask)
                    elif request["type"] in (mt5.ORDER_TYPE_SELL,):
                        request["price"] = float(tick.bid)
            result = mt5.order_send(request)
            last_result = result
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                return result
            code = result.retcode if result else None
            if code not in self._RETRYABLE_RETCODES:
                return result
            time.sleep(0.15 * (attempt + 1))
        return last_result

    def _map_symbol(self, binance_symbol):
        if binance_symbol in self.symbol_map:
            return self.symbol_map[binance_symbol]

        if binance_symbol in self._symbol_cache:
            return self._symbol_cache[binance_symbol]

        base_sym = binance_symbol.replace("USDT", "USD")

        if not self.connected:
            return base_sym

        syms = mt5.symbols_get(group=f"*{base_sym}*")
        if not syms:
            self._symbol_cache[binance_symbol] = base_sym
            return base_sym

        # Prefer exact, visible, trade-enabled symbol
        candidates = list(syms)

        def score(s):
            exact = 1 if s.name == base_sym else 0
            visible = 1 if getattr(s, "visible", False) else 0
            trade_allowed = 1 if getattr(s, "trade_mode", 0) != mt5.SYMBOL_TRADE_MODE_DISABLED else 0
            return (exact, visible, trade_allowed)

        chosen = sorted(candidates, key=score, reverse=True)[0].name
        self._symbol_cache[binance_symbol] = chosen
        return chosen

    def _normalize_price(self, sym_info, price: float) -> float:
        return round(float(price), int(sym_info.digits))

    def _normalize_lot(self, sym_info, raw_lot: float) -> float | None:
        min_lot = float(sym_info.volume_min)
        max_lot = float(sym_info.volume_max)
        step = float(sym_info.volume_step)

        if raw_lot < min_lot:
            return None  # Do not force min lot and over-risk

        lot = math.floor(raw_lot / step) * step
        lot = min(lot, max_lot)
        return round(lot, 8)

    def _loss_per_lot(self, mt5_sym: str, order_type: int, entry: float, sl: float, sym_info) -> float:
        calc = mt5.order_calc_profit(order_type, mt5_sym, 1.0, entry, sl)
        if calc is not None and calc < 0:
            return abs(float(calc))

        # Fallback
        tick_size = float(sym_info.trade_tick_size)
        tick_value = float(sym_info.trade_tick_value)
        if tick_size <= 0 or tick_value <= 0:
            return 0.0
        return abs(entry - sl) / tick_size * tick_value

    def get_mt5_price(self, symbol, direction):
        if not self.ensure_connected():
            return 0.0
        with self._lock:
            mt5_sym = self._map_symbol(symbol)
            if not mt5.symbol_select(mt5_sym, True):
                return 0.0
            info = mt5.symbol_info_tick(mt5_sym)
            if info is None:
                return 0.0
            return info.ask if direction == 1 else info.bid

    def execute_trade(self, binance_symbol, direction, bin_entry, bin_sl, bin_tp, strategy="Engine1"):
        if not self.ensure_connected():
            return None

        with self._lock:
            mt5_sym = self._map_symbol(binance_symbol)
            if not mt5.symbol_select(mt5_sym, True):
                print(f"[MT5 SKIP] {binance_symbol} -> {mt5_sym} not listed.")
                return None

            sym_info = mt5.symbol_info(mt5_sym)
            if sym_info is None:
                print(f"[MT5 SKIP] {mt5_sym}: symbol_info unavailable.")
                return None

            tick = mt5.symbol_info_tick(mt5_sym)
            if tick is None or (tick.bid == 0.0 and tick.ask == 0.0):
                print(f"[MT5 SKIP] {mt5_sym}: no live tick.")
                return None

            mt5_entry = float(tick.ask if direction == 1 else tick.bid)

            if bin_entry <= 0:
                return None

            # Determine if we are placing a limit order for a pullback (Double-Barrel)
            is_limit = False
            target_price = bin_entry

            # For Limit orders, the bin_entry must be strictly "better" by at least 0.05%
            # otherwise it might get rejected as too close to market or just executed as market.
            min_limit_distance = 0.0005

            if direction == 1 and bin_entry < mt5_entry * (1.0 - min_limit_distance):
                order_type = mt5.ORDER_TYPE_BUY_LIMIT
                is_limit = True
            elif direction == -1 and bin_entry > mt5_entry * (1.0 + min_limit_distance):
                order_type = mt5.ORDER_TYPE_SELL_LIMIT
                is_limit = True
            else:
                order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL

            exec_price = self._normalize_price(sym_info, bin_entry) if is_limit else mt5_entry

            sl_pct_dist = abs(bin_entry - bin_sl) / bin_entry
            tp_pct_dist = abs(bin_entry - bin_tp) / bin_entry
            basis_pct = abs(mt5_entry - bin_entry) / bin_entry

            # Reject if broker quote is too far from signal market, UNLESS it is a Limit order
            max_basis_allowed = min(
                self.max_abs_basis_pct,
                max(0.0015, 0.50 * sl_pct_dist),
            )

            if not is_limit and basis_pct > max_basis_allowed:
                print(
                    f"[MT5 SKIP] {binance_symbol}->{mt5_sym}: basis too large. "
                    f"Engine={bin_entry:.8f}, MT5={mt5_entry:.8f}, "
                    f"basis={basis_pct*100:.3f}%, allowed={max_basis_allowed*100:.3f}%"
                )
                return None

            if direction == 1:
                mt5_sl = exec_price * (1.0 - sl_pct_dist)
                mt5_tp = exec_price * (1.0 + tp_pct_dist)
            else:
                mt5_sl = exec_price * (1.0 + sl_pct_dist)
                mt5_tp = exec_price * (1.0 - tp_pct_dist)

            mt5_sl = self._normalize_price(sym_info, mt5_sl)
            mt5_tp = self._normalize_price(sym_info, mt5_tp)
            mt5_entry = self._normalize_price(sym_info, mt5_entry)
            exec_price = self._normalize_price(sym_info, exec_price)

            # Broker minimum stop/freeze-level guard
            point = float(sym_info.point or sym_info.trade_tick_size or 0.0)
            min_stop_dist = max(
                float(getattr(sym_info, "trade_stops_level", 0)) * point,
                float(getattr(sym_info, "trade_freeze_level", 0)) * point,
            )

            if min_stop_dist > 0:
                if abs(exec_price - mt5_sl) < min_stop_dist or abs(exec_price - mt5_tp) < min_stop_dist:
                    print(
                        f"[MT5 SKIP] {mt5_sym}: SL/TP inside broker minimum stop distance. "
                        f"min={min_stop_dist}, entry={exec_price}, sl={mt5_sl}, tp={mt5_tp}"
                    )
                    return None

            acc_info = mt5.account_info()
            current_balance = acc_info.balance if acc_info is not None else self.account_size
            risk_usd = current_balance * self.risk_pct

            loss_per_lot = self._loss_per_lot(mt5_sym, order_type, exec_price, mt5_sl, sym_info)
            if loss_per_lot <= 0:
                print(f"[MT5 SKIP] {mt5_sym}: cannot compute loss_per_lot.")
                return None

            raw_lot = risk_usd / loss_per_lot
            lot = self._normalize_lot(sym_info, raw_lot)

            if lot is None:
                print(
                    f"[MT5 SKIP] {mt5_sym}: min lot would exceed risk. "
                    f"raw_lot={raw_lot:.6f}, min_lot={sym_info.volume_min}"
                )
                return None

            # Deviation is in broker points — clamp hard to cut slippage leaks.
            # Cap at 0.03% of price or 5% of SL distance, whichever is smaller.
            max_slip_pct = min(0.0003, max(0.00005, 0.05 * sl_pct_dist))
            deviation_points = max(10, int((exec_price * max_slip_pct) / point)) if point > 0 else 10
            # Hard ceiling: never allow > 40 points of slippage on crypto CFDs
            deviation_points = min(deviation_points, 40)

            if self.dry_run:
                print(f"[MT5 DRY RUN] {mt5_sym} | {'LONG' if direction == 1 else 'SHORT'}")
                print(f"   Engine Entry: {bin_entry:.8f} | MT5 Entry/Exec: {exec_price:.8f}")
                print(f"   Basis: {basis_pct*100:.3f}% | Allowed: {max_basis_allowed*100:.3f}%")
                print(f"   MT5 SL: {mt5_sl:.8f} | MT5 TP: {mt5_tp:.8f}")
                print(f"   Lot: {lot:.4f} | Risk: ${risk_usd:.2f} | dev={deviation_points}pts")
                return {
                    "mt5_symbol": mt5_sym,
                    "mt5_ticket": None,
                    "mt5_entry": exec_price,
                    "mt5_sl": mt5_sl,
                    "mt5_tp": mt5_tp,
                    "lot": lot,
                    "basis_pct": basis_pct,
                    "is_pending": is_limit,
                }

            action = mt5.TRADE_ACTION_PENDING if is_limit else mt5.TRADE_ACTION_DEAL
            exec_price = self._normalize_price(sym_info, bin_entry) if is_limit else mt5_entry

            request = {
                "action": action,
                "symbol": mt5_sym,
                "volume": float(lot),
                "type": order_type,
                "price": exec_price,
                "sl": mt5_sl,
                "tp": mt5_tp,
                "deviation": deviation_points,
                "magic": self.magic,
                "comment": f"{strategy}" + ("_Limit" if is_limit else ""),
                "type_time": mt5.ORDER_TIME_GTC,
            }
            if not is_limit:
                request["type_filling"] = mt5.ORDER_FILLING_IOC

            check = mt5.order_check(request)
            if check is not None and check.retcode not in (0, mt5.TRADE_RETCODE_DONE):
                print(f"[MT5 SKIP] order_check failed: retcode={check.retcode}, comment={check.comment}")
                return None

            result = self._order_send_with_retry(request, max_retries=3)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                code = result.retcode if result else "None"
                comment = result.comment if result else "None"
                print(f"[MT5] Order failed after retries. Code={code}, Comment={comment}")
                return None

            # Resolve position ticket: prefer deal->position, then latest magic match.
            position_ticket = None
            order_ticket = int(getattr(result, "order", 0) or 0)
            deal_ticket = int(getattr(result, "deal", 0) or 0)

            if not is_limit:
                # Market fill: position should exist immediately
                if deal_ticket:
                    try:
                        deals = mt5.history_deals_get(ticket=deal_ticket)
                        if deals:
                            position_ticket = int(getattr(deals[0], "position_id", 0) or 0) or None
                    except Exception:
                        pass
                if position_ticket is None:
                    positions = mt5.positions_get(symbol=mt5_sym)
                    if positions:
                        mine = [p for p in positions if getattr(p, "magic", None) == self.magic]
                        candidates = mine or list(positions)
                        latest = max(candidates, key=lambda p: getattr(p, "time_msc", getattr(p, "time", 0)))
                        position_ticket = int(latest.ticket)
            else:
                # Pending: ticket is the order id until fill
                position_ticket = None

            print(
                f"[MT5 LIVE] [{strategy}] Order sent: {mt5_sym}, "
                f"pos={position_ticket}, order={order_ticket}, pending={is_limit}"
            )

            return {
                "mt5_symbol": mt5_sym,
                "mt5_ticket": position_ticket,
                "mt5_order": order_ticket,
                "mt5_deal": deal_ticket,
                "mt5_entry": exec_price,
                "mt5_sl": mt5_sl,
                "mt5_tp": mt5_tp,
                "lot": lot,
                "basis_pct": basis_pct,
                "is_pending": is_limit,
            }

    def modify_sltp(self, mt5_sym: str, position_ticket: int, sl: float, tp: float) -> bool:
        if self.dry_run:
            print(f"[MT5 DRY RUN] Modify SLTP {mt5_sym} ticket={position_ticket} SL={sl} TP={tp}")
            return True

        if not self.ensure_connected() or not position_ticket:
            return False

        with self._lock:
            sym_info = mt5.symbol_info(mt5_sym)
            if sym_info is None:
                return False

            sl = self._normalize_price(sym_info, sl)
            tp = self._normalize_price(sym_info, tp)

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": mt5_sym,
                "position": int(position_ticket),
                "sl": sl,
                "tp": tp,
                "magic": self.magic,
                "comment": "Engine1 SLTP modify",
            }

            result = self._order_send_with_retry(request, max_retries=2)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                code = result.retcode if result else "None"
                comment = result.comment if result else "None"
                print(f"[MT5] SLTP modify failed. Code={code}, Comment={comment}")
                return False

            print(f"[MT5] SLTP modified: {mt5_sym} ticket={position_ticket} SL={sl} TP={tp}")
            return True

    def close_position(self, position_ticket: int, reason: str = "ENGINE_EXIT") -> bool:
        if self.dry_run:
            print(f"[MT5 DRY RUN] Close position ticket={position_ticket}, reason={reason}")
            return True

        if not self.ensure_connected() or not position_ticket:
            return False

        with self._lock:
            positions = mt5.positions_get(ticket=position_ticket)
            if not positions:
                # Already closed by broker SL/TP.
                return True

            pos = positions[0]
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                return False

            if pos.type == mt5.POSITION_TYPE_BUY:
                close_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                close_type = mt5.ORDER_TYPE_BUY
                price = tick.ask

            # Tight close deviation — exits must not bleed R on slippage
            sym_info = mt5.symbol_info(pos.symbol)
            point = float(getattr(sym_info, "point", 0) or 0.01) if sym_info else 0.01
            deviation_points = min(30, max(10, int(0.0002 * price / point) if point > 0 else 20))

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "position": int(position_ticket),
                "volume": float(pos.volume),
                "type": close_type,
                "price": price,
                "deviation": deviation_points,
                "magic": self.magic,
                "comment": f"Engine1 close {reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = self._order_send_with_retry(request, max_retries=3)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                code = result.retcode if result else "None"
                comment = result.comment if result else "None"
                print(f"[MT5] Close failed. ticket={position_ticket}, Code={code}, Comment={comment}")
                return False

            print(f"[MT5] Closed position ticket={position_ticket}, reason={reason}")
            return True

    def is_order_pending(self, order_ticket: int) -> bool:
        if self.dry_run:
            return False
        if not self.ensure_connected() or not order_ticket:
            return False
        with self._lock:
            orders = mt5.orders_get(ticket=order_ticket)
            if orders and len(orders) > 0:
                return True
            return False

    def has_position(self, ticket: int) -> bool:
        """Return True if a live position exists for ticket OR position_id."""
        if self.dry_run:
            return True
        if not self.ensure_connected() or not ticket:
            return False
        with self._lock:
            pos = mt5.positions_get(ticket=ticket)
            if pos and len(pos) > 0:
                return True
            all_pos = mt5.positions_get()
            if all_pos:
                for p in all_pos:
                    if getattr(p, "identifier", None) == ticket or getattr(p, "ticket", None) == ticket:
                        return True
            return False

    def resolve_position_from_order(self, order_ticket: int, mt5_sym: str | None = None) -> int | None:
        """
        After a pending limit fills, map order ticket -> position ticket.
        Critical for active_trades sync (was a major tracking drift leak).
        """
        if self.dry_run or not order_ticket:
            return None
        if not self.ensure_connected():
            return None
        with self._lock:
            # Still pending?
            orders = mt5.orders_get(ticket=order_ticket)
            if orders and len(orders) > 0:
                return None
            # Search open positions for matching magic, prefer symbol match
            positions = mt5.positions_get(symbol=mt5_sym) if mt5_sym else mt5.positions_get()
            if not positions:
                return None
            mine = [p for p in positions if getattr(p, "magic", None) == self.magic]
            if not mine:
                return None
            latest = max(mine, key=lambda p: getattr(p, "time_msc", getattr(p, "time", 0)))
            return int(latest.ticket)

    def list_engine_positions(self) -> list:
        """Return all open MT5 positions owned by this engine magic."""
        if self.dry_run or not self.ensure_connected():
            return []
        with self._lock:
            positions = mt5.positions_get()
            if not positions:
                return []
            return [p for p in positions if getattr(p, "magic", None) == self.magic]

    def cancel_pending_order(self, order_ticket: int) -> bool:
        if self.dry_run:
            return True
        if not self.ensure_connected() or not order_ticket:
            return False
        with self._lock:
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": int(order_ticket),
                "magic": self.magic,
                "comment": "Engine1 cancel pending",
            }
            result = self._order_send_with_retry(request, max_retries=2)
            return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
