import MetaTrader5 as mt5
import math
import threading
import time


import logging

log = logging.getLogger("Engine_1")

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

    _SUCCESS_RETCODES = {
        getattr(mt5, "TRADE_RETCODE_DONE", 10009),
        getattr(mt5, "TRADE_RETCODE_PLACED", 10008),
        getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010),
    }

    def __init__(self, dry_run=True, account_size=5000.0, risk_pct=0.004, symbol_map=None, max_abs_basis_pct=0.005, magic=234000):
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
        self.magic = magic
        self._mt5_launched = False

    def connect(self):
        if self.dry_run:
            self.connected = True
            return True

        # Auto-open MT5 GUI once (not on every reconnect)
        if not self._mt5_launched:
            try:
                import os as _os, subprocess as _sp
                mt5_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
                if _os.path.exists(mt5_path):
                    _sp.Popen([mt5_path])
                    time.sleep(3.0)  # wait for terminal to become ready
                    self._mt5_launched = True
            except Exception:
                pass

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
                log.info(f"[MT5] initialize() failed (attempt {self._connect_failures}), error={mt5.last_error()}")
                self.connected = False
                return False
            self.connected = True
            self._connect_failures = 0
            log.info("[MT5] Connected to MetaTrader 5 successfully!")
            return True

    def ensure_connected(self) -> bool:
        if getattr(self, "dry_run", False):
            return True
        with self._lock:
            if not self.connected:
                return self.connect()
            info = mt5.terminal_info()
            if info is None or not getattr(info, "connected", False):
                log.info("[MT5] Connection lost. Re-initializing connection...")
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
                    old_price = request.get("price", 0.0)
                    new_price = float(tick.ask) if request["type"] in (mt5.ORDER_TYPE_BUY,) else float(tick.bid)
                    if old_price > 0 and new_price != old_price:
                        delta = new_price - old_price
                        request["price"] = new_price
                        if "sl" in request and request["sl"] > 0:
                            request["sl"] = round(request["sl"] + delta, 8)
                        if "tp" in request and request["tp"] > 0:
                            request["tp"] = round(request["tp"] + delta, 8)
                    else:
                        request["price"] = new_price
            result = mt5.order_send(request)
            last_result = result
            if result is not None and result.retcode in self._SUCCESS_RETCODES:
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

    def _normalize_lot(self, sym_info, raw_lot: float, allow_min_lot: bool = False) -> float | None:
        min_lot = float(sym_info.volume_min)
        max_lot = float(sym_info.volume_max)
        step = float(sym_info.volume_step)

        if raw_lot < min_lot:
            if allow_min_lot:
                return min_lot
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

    def execute_trade(self, binance_symbol, direction, bin_entry, bin_sl, bin_tp, strategy="Engine1", risk_usd=None):
        if getattr(self, "dry_run", False):
            print(f"[MT5 SIMULATION] Executing paper trade for {binance_symbol} direction={direction} at {bin_entry:.4f}")
            return {
                "mt5_symbol": binance_symbol.replace("USDT", "USD"),
                "mt5_ticket": int(time.time()),
                "mt5_order": int(time.time()),
                "mt5_deal": int(time.time()),
                "mt5_entry": bin_entry,
                "mt5_sl": bin_sl,
                "mt5_tp": bin_tp,
                "lot": 0.1,
                "is_pending": False
            }

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
            if time.time() - tick.time > 300:
                print(f"[MT5 SKIP] {mt5_sym}: stale tick data ({int(time.time() - tick.time)}s old, market likely closed).")
                return None

            mt5_entry = float(tick.ask if direction == 1 else tick.bid)

            if bin_entry <= 0:
                return None

            # Dynamic Asset-Aware Execution Strategy
            crypto_majors = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"}
            commodities = {"XAUUSDT", "XAGUSDT", "CLUSDT", "NATGASUSDT"}

            sl_pct_dist = abs(bin_entry - bin_sl) / bin_entry
            tp_pct_dist = abs(bin_entry - bin_tp) / bin_entry
            
            # Directional slippage: Positive means worse price for us on MT5
            if direction == 1:
                slippage_pct = (mt5_entry - bin_entry) / bin_entry
            else:
                slippage_pct = (bin_entry - mt5_entry) / bin_entry

            is_limit = False

            if binance_symbol in crypto_majors:
                # Category 1: Crypto Majors -> Always Market Order (Instant Fill)
                order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
                is_limit = False
            elif binance_symbol in commodities:
                max_comm_slip = 0.0025
                if slippage_pct > max_comm_slip:
                    print(
                        f"[MT5 REJECT] {binance_symbol}->{mt5_sym}: slippage {slippage_pct*100:.3f}% > {max_comm_slip*100:.2f}%. "
                        f"Trade rejected to prevent unfilled limit order."
                    )
                    return None
                else:
                    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
            else:
                max_alt_slip = min(self.max_abs_basis_pct, max(0.0030, 0.50 * sl_pct_dist))
                if slippage_pct > max_alt_slip:
                    print(
                        f"[MT5 REJECT] {binance_symbol}->{mt5_sym}: slippage {slippage_pct*100:.3f}% > {max_alt_slip*100:.2f}%. "
                        f"Trade rejected to prevent unfilled limit order."
                    )
                    return None
                else:
                    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL

            exec_price = mt5_entry

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

            if risk_usd is None:
                acc_info = mt5.account_info()
                if acc_info is not None:
                    effective_capital = min(float(acc_info.balance), float(acc_info.equity))
                else:
                    effective_capital = self.account_size
                risk_usd = effective_capital * self.risk_pct

            loss_per_lot = self._loss_per_lot(mt5_sym, order_type, exec_price, mt5_sl, sym_info)
            if loss_per_lot <= 0:
                print(f"[MT5 SKIP] {mt5_sym}: cannot compute loss_per_lot.")
                return None

            raw_lot = risk_usd / loss_per_lot
            min_lot = float(sym_info.volume_min)

            # Allow min_lot override if min_lot dollar risk is within $45 or <= 2.25x target risk
            allow_min = False
            if raw_lot < min_lot and loss_per_lot > 0:
                min_lot_risk = min_lot * loss_per_lot
                if min_lot_risk <= max(45.0, 2.25 * risk_usd):
                    allow_min = True

            lot = self._normalize_lot(sym_info, raw_lot, allow_min_lot=allow_min)

            if lot is None:
                print(
                    f"[MT5 SKIP] {mt5_sym}: min lot would exceed risk. "
                    f"raw_lot={raw_lot:.6f}, min_lot={sym_info.volume_min}"
                )
                return None

            # Deviation is in broker points — clamp hard to cut slippage leaks.
            # Cap at 0.20% of price or 20% of SL distance, whichever is smaller.
            max_slip_pct = min(0.0020, max(0.00050, 0.20 * sl_pct_dist))
            deviation_points = max(20, int((exec_price * max_slip_pct) / point)) if point > 0 else 20
            # Hard ceiling: dynamically scale for crypto CFDs, cap at 1000 points
            deviation_points = min(deviation_points, 1000)

            if self.dry_run:
                print(f"[MT5 DRY RUN] {mt5_sym} | {'LONG' if direction == 1 else 'SHORT'}")
                print(f"   Engine Entry: {bin_entry:.8f} | MT5 Entry/Exec: {exec_price:.8f}")
                print(f"   Slippage: {slippage_pct*100:.3f}%")
                print(f"   MT5 SL: {mt5_sl:.8f} | MT5 TP: {mt5_tp:.8f}")
                print(f"   Lot: {lot:.4f} | Risk: ${risk_usd:.2f} | dev={deviation_points}pts")
                return {
                    "mt5_symbol": mt5_sym,
                    "mt5_ticket": None,
                    "mt5_entry": exec_price,
                    "mt5_sl": mt5_sl,
                    "mt5_tp": mt5_tp,
                    "lot": lot,
                    "basis_pct": slippage_pct,
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
                "type_time": mt5.ORDER_TIME_SPECIFIED if is_limit else mt5.ORDER_TIME_GTC,
            }
            if is_limit:
                # Expire limit orders after 4 hours if not filled
                request["expiration"] = int(time.time() + 4 * 3600)
            else:
                request["type_filling"] = mt5.ORDER_FILLING_IOC

            valid_check_codes = {0}.union(self._SUCCESS_RETCODES)
            check = mt5.order_check(request)
            if check is not None and check.retcode not in valid_check_codes:
                print(f"[MT5 SKIP] order_check failed: retcode={check.retcode}, comment={check.comment}")
                return None

            result = self._order_send_with_retry(request, max_retries=3)
            if result is None or result.retcode not in self._SUCCESS_RETCODES:
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
                        candidates = mine
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
                "basis_pct": slippage_pct,
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

            # Wide exit deviation — prioritize execution over slippage for SL/TP exits
            sym_info = mt5.symbol_info(pos.symbol)
            point = float(getattr(sym_info, "point", 0) or 0.01) if sym_info else 0.01
            deviation_points = max(50, int(0.005 * price / point) if point > 0 else 100)

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
            mine = [p for p in positions
                    if getattr(p, "magic", None) == self.magic
                    and getattr(p, "identifier", None) == order_ticket]
            if mine:
                return int(mine[0].ticket)
            fallback = [p for p in positions if getattr(p, "magic", None) == self.magic]
            if not fallback:
                return None
            latest = max(fallback, key=lambda p: getattr(p, "time_msc", getattr(p, "time", 0)))
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

    def get_position_history_profit(self, position_ticket: int) -> tuple[float, float]:
        """Returns (profit, price) of a closed position based on history deals. Fixes LEAK-17."""
        if self.dry_run or not self.ensure_connected() or not position_ticket:
            return 0.0, 0.0
        with self._lock:
            deals = mt5.history_deals_get(position=position_ticket)
            if not deals:
                return 0.0, 0.0
            
            pos_deals = [d for d in deals if getattr(d, "position_id", None) == position_ticket]
            if not pos_deals:
                return 0.0, 0.0
                
            total_profit = sum(getattr(d, "profit", 0.0) + getattr(d, "swap", 0.0) + getattr(d, "commission", 0.0) for d in pos_deals)
            # Find the final deal (usually the out deal) to get exit price
            out_deals = [d for d in pos_deals if getattr(d, "entry", mt5.DEAL_ENTRY_IN) in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY)]
            exit_price = getattr(out_deals[-1], "price", 0.0) if out_deals else 0.0
            
            return total_profit, exit_price

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

    def get_account_balance_and_equity(self) -> tuple[float, float]:
        """Return (balance, equity) from live MT5 account."""
        if getattr(self, "dry_run", False) or not self.ensure_connected():
            return 0.0, 0.0
        with self._lock:
            acc = mt5.account_info()
            if acc is not None:
                return float(acc.balance), float(acc.equity)
            return 0.0, 0.0

