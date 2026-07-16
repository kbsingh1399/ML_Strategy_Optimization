import MetaTrader5 as mt5
import math
import threading

class MT5Broker:
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

    def connect(self):
        with self._lock:
            if not mt5.initialize():
                print(f"[MT5] initialize() failed, error code = {mt5.last_error()}")
                self.connected = False
                return False
            self.connected = True
            print("[MT5] Connected to MetaTrader 5 successfully!")
            return True

    def ensure_connected(self) -> bool:
        with self._lock:
            if not self.connected:
                return self.connect()
            info = mt5.terminal_info()
            if info is None or not getattr(info, "connected", False):
                print("[MT5] Connection lost. Re-initializing connection...")
                return self.connect()
            return True

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
            return None # Do not force min lot and over-risk
            
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

            # Deviation is in broker points, not percent.
            max_slip_pct = min(0.0003, max(0.00005, 0.05 * sl_pct_dist))
            deviation_points = max(20, int((exec_price * max_slip_pct) / point)) if point > 0 else 20

            if self.dry_run:
                print(f"[MT5 DRY RUN] {mt5_sym} | {'LONG' if direction == 1 else 'SHORT'}")
                print(f"   Engine Entry: {bin_entry:.8f} | MT5 Entry/Exec: {exec_price:.8f}")
                print(f"   Basis: {basis_pct*100:.3f}% | Allowed: {max_basis_allowed*100:.3f}%")
                print(f"   MT5 SL: {mt5_sl:.8f} | MT5 TP: {mt5_tp:.8f}")
                print(f"   Lot: {lot:.4f} | Risk: ${risk_usd:.2f}")
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
                "magic": 234000,
                "comment": f"{strategy}" + ("_Limit" if is_limit else ""),
                "type_time": mt5.ORDER_TIME_GTC,
            }
            if not is_limit:
                request["type_filling"] = mt5.ORDER_FILLING_IOC

            check = mt5.order_check(request)
            if check is not None and check.retcode not in (0, mt5.TRADE_RETCODE_DONE):
                print(f"[MT5 SKIP] order_check failed: retcode={check.retcode}, comment={check.comment}")
                return None

            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                code = result.retcode if result else "None"
                comment = result.comment if result else "None"
                print(f"[MT5] Order failed. Code={code}, Comment={comment}")
                return None
                
            # Find latest position ticket for this symbol/magic.
            position_ticket = None
            positions = mt5.positions_get(symbol=mt5_sym)
            if positions:
                mine = [p for p in positions if getattr(p, "magic", None) == 234000]
                candidates = mine or list(positions)
                latest = max(candidates, key=lambda p: getattr(p, "time_msc", getattr(p, "time", 0)))
                position_ticket = int(latest.ticket)
            print(f"[MT5 LIVE] [{strategy}] Order sent: {mt5_sym}, ticket={position_ticket}")
            
            return {
                "mt5_symbol": mt5_sym,
                "mt5_ticket": position_ticket,
                "mt5_order": int(result.order),
                "mt5_deal": int(result.deal),
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
                "magic": 234000,
                "comment": "Engine1 SLTP modify",
            }
            
            result = mt5.order_send(request)
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
                
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "position": int(position_ticket),
                "volume": float(pos.volume),
                "type": close_type,
                "price": price,
                "deviation": 50,
                "magic": 234000,
                "comment": f"Engine1 close {reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
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

