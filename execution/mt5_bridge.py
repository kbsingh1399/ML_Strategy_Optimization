"""
MT5 Execution Bridge & Risk Guard for Blueberry Markets / Funded Accounts.
Handles dynamic lot sizing, order placement, stop-loss/take-profit calculation,
and daily drawdown kill-switch (3% max daily loss for 5K prop accounts).
"""

import os
import sys
import time
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MT5Bridge")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 module not installed. Running in simulation / dry-run mode.")

class MT5ExecutionBridge:
    def __init__(self, account_id: Optional[int] = None, password: Optional[str] = None, 
                 server: Optional[str] = None, max_daily_loss_pct: float = 3.0, 
                 initial_balance: float = 5000.0, risk_per_trade_usd: float = 10.0):
        self.account_id = account_id
        self.password = password
        self.server = server
        self.max_daily_loss_pct = max_daily_loss_pct
        self.initial_balance = initial_balance
        self.risk_per_trade_usd = risk_per_trade_usd
        self.daily_start_balance = initial_balance
        self.connected = False

    def initialize(self) -> bool:
        if not MT5_AVAILABLE:
            logger.info("[SIMULATION] MT5 initialized in dry-run mode.")
            self.connected = True
            return True

        if not mt5.initialize():
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return False

        if self.account_id and self.password and self.server:
            authorized = mt5.login(self.account_id, password=self.password, server=self.server)
            if not authorized:
                logger.error(f"MT5 login failed for account {self.account_id}: {mt5.last_error()}")
                return False

        account_info = mt5.account_info()
        if account_info:
            logger.info(f"Connected to MT5 Account {account_info.login} on {account_info.server}")
            logger.info(f"Balance: ${account_info.balance:.2f}, Equity: ${account_info.equity:.2f}")
            self.daily_start_balance = account_info.balance
            self.daily_start_date = datetime.now(timezone.utc).date()
            self.connected = True
            return True
        return False

    def check_risk_guard(self) -> bool:
        """Enforces hard 3% daily drawdown kill-switch to protect funded account."""
        if not MT5_AVAILABLE or not self.connected:
            return True

        acc = mt5.account_info()
        if not acc:
            return False

        current_equity = acc.equity
        
        # Check for daily rollover at UTC midnight
        current_date = datetime.now(timezone.utc).date()
        if getattr(self, 'daily_start_date', None) != current_date:
            logger.info(f"New UTC day detected. Rolling over daily start balance from ${self.daily_start_balance:.2f} to ${acc.balance:.2f}")
            self.daily_start_balance = acc.balance
            self.daily_start_date = current_date
            
        daily_loss = self.daily_start_balance - current_equity
        daily_loss_pct = (daily_loss / self.daily_start_balance) * 100.0 if self.daily_start_balance > 0 else 0.0

        if daily_loss_pct >= self.max_daily_loss_pct:
            logger.critical(f"RISK GUARD TRIGGERED! Daily loss is {daily_loss_pct:.2f}% (Limit: {self.max_daily_loss_pct}%). Halting all trades.")
            return False
        return True

    def resolve_symbol(self, symbol: str) -> str:
        """Resolves broker-specific symbol suffixes (e.g. BTCUSDT -> BTCUSD.pi or BTCUSD.p)."""
        if not MT5_AVAILABLE or not self.connected:
            return symbol

        base = symbol.replace("USDT", "USD")
        candidates = [symbol, base, base + ".pi", base + ".p", symbol + ".pi", symbol + ".p", symbol + ".a"]
        for cand in candidates:
            info = mt5.symbol_info(cand)
            if info is not None:
                return cand
        return symbol

    def calculate_lot_size(self, symbol: str, entry_price: float, sl_price: float) -> float:
        """Calculates precise lot size so potential loss equals risk_per_trade_usd ($50)."""
        symbol = self.resolve_symbol(symbol)
        sl_dist = abs(entry_price - sl_price)
        if sl_dist <= 0:
            return 0.01

        if not MT5_AVAILABLE or not self.connected:
            # Simulation fallback calculation
            raw_lots = self.risk_per_trade_usd / (sl_dist * 1.0)
            return max(0.01, round(raw_lots, 2))

        sym_info = mt5.symbol_info(symbol)
        if not sym_info:
            logger.error(f"Symbol info not found for {symbol}")
            return 0.01

        tick_size = sym_info.trade_tick_size or 0.01
        tick_value = sym_info.trade_tick_value or 1.0
        contract_size = sym_info.trade_contract_size or 1.0

        sl_ticks = sl_dist / tick_size
        risk_per_lot = sl_ticks * tick_value

        if risk_per_lot <= 0:
            return sym_info.volume_min

        raw_lots = self.risk_per_trade_usd / risk_per_lot
        step = sym_info.volume_step or 0.01
        lots = round(raw_lots / step) * step
        lots = max(sym_info.volume_min, min(sym_info.volume_max, lots))
        return round(lots, 2)

    def execute_order(self, symbol: str, direction: int, sl_price: float, tp_price: float) -> Dict[str, Any]:
        """Places a market buy/sell order with exact SL/TP and risk-managed lot sizing."""
        symbol = self.resolve_symbol(symbol)
        if not self.check_risk_guard():
            return {"status": "REJECTED", "reason": "Daily Risk Guard Triggered"}

        if not MT5_AVAILABLE or not self.connected:
            logger.info(f"[SIMULATION] Executing {('BUY' if direction == 1 else 'SELL')} on {symbol} | SL: {sl_price:.4f} | TP: {tp_price:.4f}")
            return {"status": "SIMULATED_SUCCESS", "symbol": symbol, "direction": direction}

        sym_info = mt5.symbol_info(symbol)
        if not sym_info:
            return {"status": "ERROR", "reason": f"Symbol {symbol} not found"}

        if not sym_info.visible:
            mt5.symbol_select(symbol, True)

        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Tick data unavailable for {symbol}")
            return {"status": "ERROR", "reason": f"Tick data unavailable for {symbol}"}
        price = tick.ask if direction == 1 else tick.bid
        lots = self.calculate_lot_size(symbol, price, sl_price)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 20,
            "magic": 1001,
            "comment": "Engine_1 ML Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            logger.error(f"Order failed for {symbol}: mt5.order_send returned None (check MT5 connection)")
            return {"status": "FAILED", "reason": "MT5 returned None"}
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed for {symbol}: {result.comment} (code {result.retcode})")
            return {"status": "FAILED", "code": result.retcode, "comment": result.comment}

        logger.info(f"SUCCESS: Executed {symbol} {'BUY' if direction == 1 else 'SELL'} | Volume: {lots} lots | Ticket: {result.order}")
        return {"status": "SUCCESS", "ticket": result.order, "lots": lots, "price": result.price}

if __name__ == "__main__":
    dry_run = "--live" not in sys.argv
    print(f"MT5 Bridge Test (Dry-Run: {dry_run})")
    if dry_run:
        print("Running dry-run calculation check...")
        bridge = MT5ExecutionBridge(max_daily_loss_pct=3.0, initial_balance=5000.0, risk_per_trade_usd=10.0)
        lot = bridge.calculate_lot_size("BTCUSD.pi", 64000.0, 63000.0)
        print(f"Calculated Lot Size for $10 Risk ($1,000 SL Dist): {lot} Lots")
    else:
        bridge = MT5ExecutionBridge(max_daily_loss_pct=3.0, initial_balance=5000.0, risk_per_trade_usd=10.0)
        bridge.initialize()
        sim_res = bridge.execute_order("BTCUSD", direction=1, sl_price=64000.0, tp_price=68000.0)
        print("Test Execution Result:", sim_res)
