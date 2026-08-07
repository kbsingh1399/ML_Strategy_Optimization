"""
Binance Futures Execution Broker for Engine_1.
Pure Binance Futures perpetual swap execution. No MT5 dependencies.
Supports Dry-Run (paper trading) and Live Futures trading via REST API.
"""

import os
import time
import math
import hmac
import hashlib
import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

log = logging.getLogger("BinanceBroker")

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"


def _load_env():
    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()


_load_env()


class BinanceBroker:
    """Binance Futures perpetual swap execution engine."""

    MAX_RETRIES = 3
    RETRY_BACKOFF = [1.0, 3.0, 5.0]

    def __init__(
        self,
        dry_run: bool = True,
        account_size: float = 5000.0,
        risk_pct: float = 0.005,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        use_testnet: bool = False,
    ):
        self.dry_run = dry_run
        self.account_size = account_size
        self.risk_pct = risk_pct
        self.api_key = api_key or os.environ.get("BINANCE_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("BINANCE_SECRET_KEY", "")
        self.use_testnet = use_testnet or os.environ.get("BINANCE_USE_TESTNET", "").lower() == "true"

        if self.use_testnet:
            self.base_url = "https://testnet.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"

        self.symbol_rules: Dict[str, dict] = {}
        self.valid_perpetuals: set = set()
        self.active_orders: Dict[str, dict] = {}
        self.time_offset = 0

        log.info(
            f"BinanceBroker initialized (dry_run={self.dry_run}, "
            f"testnet={self.use_testnet}, base_url={self.base_url})"
        )

    def _sign_params(self, params: dict) -> dict:
        params["timestamp"] = int((time.time() * 1000) + self.time_offset)
        params["recvWindow"] = 60000
        query_str = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.secret_key.encode("utf-8"),
            query_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    def _backoff_sleep(self, seconds: float):
        """Non-blocking wait loop to keep event loops responsive without blocking thread pools."""
        end = time.time() + seconds
        while time.time() < end:
            time.sleep(0.01)

    def _request(
        self, method: str, endpoint: str,
        params: Optional[dict] = None, signed: bool = True,
        max_retries: int = 3,
    ) -> Optional[dict]:
        """Make REST request to Binance Futures API with retry logic."""
        params = params or {}
        headers = {}

        for attempt in range(max_retries):
            req_params = dict(params)
            if signed:
                if not self.api_key or not self.secret_key:
                    log.error("[Binance] Missing API key or secret key for signed request.")
                    return None
                req_params = self._sign_params(req_params)
                headers = {"X-MBX-APIKEY": self.api_key}

            query_str = urllib.parse.urlencode(req_params)
            url = f"{self.base_url}{endpoint}"
            data = None

            if method in ("GET", "DELETE"):
                if query_str:
                    url = f"{url}?{query_str}"
            elif method in ("POST", "PUT"):
                headers["Content-Type"] = "application/x-www-form-urlencoded"
                data = query_str.encode("utf-8")

            try:
                req = urllib.request.Request(url, data=data, headers=headers, method=method)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    res_bytes = resp.read()
                    return json.loads(res_bytes.decode("utf-8"))

            except urllib.error.HTTPError as e:
                err_msg = e.read().decode("utf-8") if hasattr(e, "read") else str(e)

                if e.code in (429, 418):
                    wait = self.RETRY_BACKOFF[min(attempt, len(self.RETRY_BACKOFF) - 1)]
                    log.warning(f"[Binance] Rate limited ({e.code}). Retry {attempt+1}/{max_retries} in {wait}s...")
                    self._backoff_sleep(wait)
                    continue

                if e.code >= 500:
                    wait = self.RETRY_BACKOFF[min(attempt, len(self.RETRY_BACKOFF) - 1)]
                    log.warning(f"[Binance] Server error {e.code}. Retry {attempt+1}/{max_retries} in {wait}s...")
                    self._backoff_sleep(wait)
                    continue

                # Timestamp drift: re-sync and retry once
                if "-1021" in err_msg and attempt == 0:
                    log.warning("[Binance] Timestamp drift detected, re-syncing server time...")
                    self._sync_server_time()
                    continue

                log.error(f"[Binance API Error] {method} {endpoint}: {e.code} — {err_msg}")
                return None

            except Exception as e:
                if attempt < max_retries - 1:
                    wait = self.RETRY_BACKOFF[min(attempt, len(self.RETRY_BACKOFF) - 1)]
                    log.warning(f"[Binance] Network error: {e}. Retry {attempt+1}/{max_retries} in {wait}s...")
                    self._backoff_sleep(wait)
                    continue
                log.error(f"[Binance Request Failed] {method} {endpoint}: {e}")
                return None

        log.error(f"[Binance] All {max_retries} retries exhausted for {method} {endpoint}")
        return None

    def _sync_server_time(self):
        try:
            res = self._request("GET", "/fapi/v1/time", signed=False, max_retries=1)
            if res and "serverTime" in res:
                self.time_offset = res["serverTime"] - int(time.time() * 1000)
        except Exception:
            pass

    def connect(self) -> bool:
        """Sync server time and fetch exchange info precision rules."""
        try:
            self._sync_server_time()
            log.info(f"[Binance] Connected. Server time offset: {self.time_offset}ms")

            info = self._request("GET", "/fapi/v1/exchangeInfo", signed=False)
            if info and "symbols" in info:
                for s in info["symbols"]:
                    sym = s["symbol"]
                    price_prec = s.get("pricePrecision", 2)
                    qty_prec = s.get("quantityPrecision", 3)
                    min_qty = 0.001
                    step_size = 0.001
                    tick_size = 0.01

                    for f in s.get("filters", []):
                        if f.get("filterType") == "LOT_SIZE":
                            min_qty = float(f.get("minQty", 0.001))
                            step_size = float(f.get("stepSize", 0.001))
                        elif f.get("filterType") == "PRICE_FILTER":
                            tick_size = float(f.get("tickSize", 0.01))

                    self.symbol_rules[sym] = {
                        "price_prec": price_prec,
                        "qty_prec": qty_prec,
                        "min_qty": min_qty,
                        "step_size": step_size,
                        "tick_size": tick_size,
                    }

                    if s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING":
                        self.valid_perpetuals.add(sym)

                log.info(f"[Binance] Loaded rules for {len(self.symbol_rules)} contracts, "
                         f"{len(self.valid_perpetuals)} active perpetuals.")

            if not self.dry_run:
                bal, eq = self.get_account_balance_and_equity()
                log.info(f"[Binance] Authenticated! USDT Wallet: ${bal:,.2f}, Equity: ${eq:,.2f}")
            return True
        except Exception as e:
            log.error(f"[Binance Connect Failed] {e}")
            return False

    def ensure_connected(self) -> bool:
        return True

    def is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol is a valid, actively trading Binance Futures perpetual."""
        if not self.valid_perpetuals:
            return symbol in self.symbol_rules
        return symbol in self.valid_perpetuals

    def get_account_balance_and_equity(self) -> Tuple[float, float]:
        details = self.get_account_details()
        return details["balance"], details["equity"]

    def get_account_details(self) -> Dict[str, float]:
        """Fetch USDT-specific balance, equity, and unrealized PnL."""
        if self.dry_run:
            return {"balance": self.account_size, "equity": self.account_size, "unrealized_pnl": 0.0}

        res = self._request("GET", "/fapi/v2/account", signed=True)
        if res:
            usdt_bal = 0.0
            usdt_eq = 0.0
            usdt_upnl = 0.0
            for asset in res.get("assets", []):
                if asset.get("asset") == "USDT":
                    usdt_bal = float(asset.get("walletBalance", 0.0))
                    usdt_eq = float(asset.get("marginBalance", 0.0))
                    usdt_upnl = float(asset.get("unrealizedProfit", 0.0))
                    break
            if usdt_bal == 0.0:
                usdt_bal = float(res.get("totalWalletBalance", 0.0))
                usdt_eq = float(res.get("totalMarginBalance", 0.0))
                usdt_upnl = float(res.get("totalUnrealizedProfit", 0.0))
            return {"balance": usdt_bal, "equity": usdt_eq, "unrealized_pnl": usdt_upnl}
        return {"balance": 0.0, "equity": 0.0, "unrealized_pnl": 0.0}

    def _round_step(self, val: float, step: float, direction: str = "nearest") -> float:
        if step <= 0:
            return val
        precision = int(round(-math.log10(step))) if step < 1 else 0
        factor = 10 ** precision
        if direction == "down":
            return math.floor(val * factor) / factor
        elif direction == "up":
            return math.ceil(val * factor) / factor
        return round(val * factor) / factor

    def _format_price(self, symbol: str, price: float, direction: str = "nearest") -> float:
        """Round price to exchange tick size (PRICE_FILTER), not just decimal precision."""
        rules = self.symbol_rules.get(symbol)
        if rules and "tick_size" in rules:
            return self._round_step(price, rules["tick_size"], direction)
        prec = rules["price_prec"] if rules else 2
        return round(price, prec)

    def _format_qty(self, symbol: str, qty: float) -> float:
        rules = self.symbol_rules.get(symbol, {"qty_prec": 3, "step_size": 0.001, "min_qty": 0.001})
        step = rules["step_size"]
        min_q = rules["min_qty"]
        formatted = self._round_step(qty, step)
        return max(formatted, min_q)

    def _place_algo_conditional(
        self, symbol: str, side: str, order_type: str, trigger_price: float, label: str
    ) -> Optional[dict]:
        """Place a conditional algo order (SL or TP) on Binance Futures."""
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "stopPrice": str(trigger_price),
            "closePosition": "true",
            "workingType": "MARK_PRICE",
            "priceProtect": "true",
            "algoType": "CONDITIONAL",
        }
        res = self._request("POST", "/fapi/v1/algoOrder", params=params, signed=True)
        if res and ("algoId" in res or "clientAlgoId" in res):
            log.info(f"[BINANCE LIVE] Attached {label}: {trigger_price} (algoId={res.get('algoId')})")
        else:
            log.warning(f"[Binance] {label} response: {res} — Engine_1 check_exits will manage fallback.")
        return res

    def execute_trade(
        self,
        binance_symbol: str,
        direction: int,
        bin_entry: float,
        bin_sl: float,
        bin_tp: float,
        strategy: str,
        risk_capital: float,
    ) -> Optional[dict]:
        """Execute market order on Binance Futures with attached SL and TP."""
        stop_dist = abs(bin_entry - bin_sl)
        if stop_dist <= 0 or bin_entry <= 0:
            return None

        if not self.is_valid_symbol(binance_symbol):
            log.error(f"[Binance] {binance_symbol} is not a valid active perpetual. Rejecting trade.")
            return None

        raw_qty = risk_capital / stop_dist
        qty = self._format_qty(binance_symbol, raw_qty)
        entry_price = self._format_price(binance_symbol, bin_entry)
        sl_price = self._format_price(binance_symbol, bin_sl)
        tp_price = self._format_price(binance_symbol, bin_tp)

        side = "BUY" if direction == 1 else "SELL"
        opposite_side = "SELL" if direction == 1 else "BUY"

        MIN_NOTIONAL = 5.0
        if qty * bin_entry < MIN_NOTIONAL:
            log.warning(
                f"[Binance] Skipping {side} {binance_symbol}: notional "
                f"${qty * bin_entry:.2f} < ${MIN_NOTIONAL} minimum."
            )
            return None

        client_order_id = f"E1_{strategy}_{int(time.time_ns() % 1_000_000_000)}"

        if self.dry_run:
            pseudo_ticket = int(time.time_ns() % 1_000_000_000)
            log.info(
                f"[BINANCE DRY RUN] [{strategy}] {side} {qty} {binance_symbol} @ {entry_price:.4f} "
                f"SL={sl_price:.4f} TP={tp_price:.4f} (ticket={pseudo_ticket})"
            )
            return {
                "symbol": binance_symbol,
                "order_id": pseudo_ticket,
                "entry_price": entry_price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "lot": qty,
                "basis_pct": 0.0,
                "is_pending": False,
            }

        log.info(f"[BINANCE LIVE] Dispatching {side} {qty} {binance_symbol} @ Market...")

        # GAP 4: Latency Guard Pre-Check
        try:
            ticker = self._request("GET", "/fapi/v1/ticker/bookTicker", params={"symbol": binance_symbol})
            if ticker and "askPrice" in ticker and "bidPrice" in ticker:
                current_price = float(ticker["askPrice"]) if direction == 1 else float(ticker["bidPrice"])
                if current_price > 0:
                    drift = abs(current_price - bin_entry) / bin_entry
                    if drift > 0.0015:  # 0.15% max drift tolerance
                        log.error(f"[BINANCE DRIFT REJECT] {binance_symbol} drift {drift:.4%} > 0.15% limit. Aborting.")
                        return None
        except Exception as e:
            log.warning(f"[Binance] Latency guard check failed, proceeding anyway: {e}")

        order_params = {
            "symbol": binance_symbol,
            "side": side,
            "type": "MARKET",
            "quantity": qty,
            "newClientOrderId": client_order_id,
        }
        res = self._request("POST", "/fapi/v1/order", params=order_params, signed=True)
        if not res or "orderId" not in res:
            log.error(f"[BINANCE LIVE REJECT] Failed to place market entry for {binance_symbol}")
            return None

        # GAP 7 FIX: Fill Verification
        status = res.get("status", "")
        executed_qty = float(res.get("executedQty", 0.0))
        if status in ("REJECTED", "EXPIRED", "CANCELED") or (status == "NEW" and executed_qty == 0.0):
            # Double check fill status by querying order endpoint if NEW
            query_res = self._request("GET", "/fapi/v1/order", params={"symbol": binance_symbol, "orderId": res["orderId"]}, signed=True)
            if query_res:
                status = query_res.get("status", status)
                executed_qty = float(query_res.get("executedQty", executed_qty))
                res["avgPrice"] = query_res.get("avgPrice", res.get("avgPrice", 0.0))
                res["cumQuote"] = query_res.get("cumQuote", res.get("cumQuote", 0.0))

        if status not in ("FILLED", "PARTIALLY_FILLED") and executed_qty == 0.0:
            log.error(f"[BINANCE UNVERIFIED FILL] Order {res.get('orderId')} status={status}, executedQty={executed_qty}. Aborting trade.")
            return None

        order_id = int(res["orderId"])
        cum_quote = float(res.get("cumQuote", 0.0))
        avg_price = (cum_quote / executed_qty) if executed_qty > 0 and cum_quote > 0 else float(res.get("avgPrice", entry_price))
        if avg_price == 0.0:
            avg_price = entry_price

        # GAP 2: Dollar-distance SL/TP locking
        sl_dist = abs(entry_price - sl_price)
        tp_dist = abs(tp_price - entry_price)

        if direction == 1:
            final_sl = self._format_price(binance_symbol, avg_price - sl_dist, "down")
            final_tp = self._format_price(binance_symbol, avg_price + tp_dist, "nearest")
        else:
            final_sl = self._format_price(binance_symbol, avg_price + sl_dist, "up")
            final_tp = self._format_price(binance_symbol, avg_price - tp_dist, "nearest")

        sl_res = None
        try:
            sl_res = self._place_algo_conditional(binance_symbol, opposite_side, "STOP_MARKET", final_sl, "SL")
        except Exception as e:
            log.warning(f"[Binance] SL algo order exception: {e}")

        if not sl_res or ("algoId" not in sl_res and "clientAlgoId" not in sl_res):
            log.error(f"[BINANCE NAKED GUARD] SL placement failed! Closing market entry for {binance_symbol}")
            self.close_position(binance_symbol, "NAKED_GUARD_SL_FAILED")
            return None

        try:
            self._place_algo_conditional(binance_symbol, opposite_side, "TAKE_PROFIT_MARKET", final_tp, "TP")
        except Exception as e:
            log.warning(f"[Binance] TP algo order exception: {e}")

        log.info(f"[BINANCE LIVE SUCCESS] Fill: {binance_symbol} {side} {executed_qty} @ ${avg_price:,.2f} (orderId={order_id})")

        return {
            "symbol": binance_symbol,
            "order_id": order_id,
            "entry_price": avg_price,
            "sl_price": final_sl,
            "tp_price": final_tp,
            "lot": executed_qty,
            "basis_pct": 0.0,
            "is_pending": False,
        }

    def _cancel_all_orders(self, binance_symbol: str):
        """Cancel all open orders and algo orders for a symbol."""
        self._request("DELETE", "/fapi/v1/allOpenOrders", params={"symbol": binance_symbol}, signed=True)
        try:
            open_algos = self._request("GET", "/fapi/v1/openAlgoOrders", params={"symbol": binance_symbol}, signed=True)
            if open_algos:
                for algo in open_algos:
                    self._request("DELETE", "/fapi/v1/algoOrder", params={"symbol": binance_symbol, "algoId": algo["algoId"]}, signed=True)
        except Exception as e:
            log.warning(f"[BINANCE LIVE] Failed to cancel algo orders for {binance_symbol}: {e}")

    def modify_sltp(self, binance_symbol: str, position_ticket: int, sl: float, tp: float) -> bool:
        """Modify open SL/TP orders for symbol on Binance safely without naked window."""
        if self.dry_run:
            log.info(f"[BINANCE DRY RUN] Modify SLTP {binance_symbol} SL={sl} TP={tp}")
            return True

        positions = self._request("GET", "/fapi/v2/account", signed=True)
        if not positions or "positions" not in positions:
            return False

        pos_amt = 0.0
        for p in positions["positions"]:
            if p["symbol"] == binance_symbol:
                pos_amt = float(p["positionAmt"])
                break

        if pos_amt == 0.0:
            log.warning(f"[Binance] Cannot modify SL/TP: No open position for {binance_symbol}")
            return False

        opposite_side = "SELL" if pos_amt > 0 else "BUY"
        formatted_sl = self._format_price(binance_symbol, sl)
        formatted_tp = self._format_price(binance_symbol, tp)

        # GAP 8 FIX: Place new SL order FIRST before canceling old ones to avoid naked position window
        new_sl_res = self._place_algo_conditional(binance_symbol, opposite_side, "STOP_MARKET", formatted_sl, "NEW_SL")
        if not new_sl_res or ("algoId" not in new_sl_res and "clientAlgoId" not in new_sl_res):
            log.critical(f"[BINANCE NAKED GUARD] SL modification FAILED for {binance_symbol} — emergency closing position!")
            self.close_position(binance_symbol, "SL_MOD_FAILED")
            return False

        # Place new TP order
        new_tp_res = self._place_algo_conditional(binance_symbol, opposite_side, "TAKE_PROFIT_MARKET", formatted_tp, "NEW_TP")

        # Now clean up old algo orders except the newly created ones
        new_sl_id = new_sl_res.get("algoId")
        new_tp_id = new_tp_res.get("algoId") if new_tp_res else None
        try:
            open_algos = self._request("GET", "/fapi/v1/openAlgoOrders", params={"symbol": binance_symbol}, signed=True)
            if open_algos:
                for algo in open_algos:
                    algo_id = algo.get("algoId")
                    if algo_id not in (new_sl_id, new_tp_id):
                        self._request("DELETE", "/fapi/v1/algoOrder", params={"symbol": binance_symbol, "algoId": algo_id}, signed=True)
        except Exception as e:
            log.warning(f"[Binance] Exception cleaning old algo orders during modify_sltp: {e}")

        log.info(f"[BINANCE LIVE] SLTP Modified for {binance_symbol}: SL={formatted_sl} TP={formatted_tp}")
        return True

    def close_position(self, symbol: str, reason: str = "ENGINE_EXIT") -> bool:
        """Close open position on Binance Futures with Market order."""
        if self.dry_run:
            log.info(f"[BINANCE DRY RUN] Close position symbol={symbol}, reason={reason}")
            return True

        positions = self._request("GET", "/fapi/v2/positionRisk", params={"symbol": symbol}, signed=True)
        if not positions:
            return True

        for p in positions:
            if p["symbol"] != symbol:
                continue
            amt = float(p.get("positionAmt", 0.0))
            if amt != 0.0:
                self._cancel_all_orders(symbol)

                side = "SELL" if amt > 0 else "BUY"
                close_qty = abs(amt)
                res = self._request("POST", "/fapi/v1/order", params={
                    "symbol": symbol,
                    "side": side,
                    "type": "MARKET",
                    "quantity": close_qty,
                    "reduceOnly": "true",
                }, signed=True)

                if res and "orderId" in res:
                    log.info(f"[BINANCE LIVE] Closed position for {symbol} ({reason}) @ Market")
                    return True
                else:
                    log.error(f"[BINANCE LIVE] Failed to close position for {symbol}")
                    return False
        return True

    def get_position_history_profit(self, position_ticket: int) -> Tuple[float, float]:
        """Fetch realized profit and exit price from user trades."""
        if self.dry_run:
            return 0.0, 0.0
        return 0.0, 0.0
