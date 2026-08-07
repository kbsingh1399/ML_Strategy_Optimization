Python# TARGET: Engine_1.py
# ═══════════════════════════════════════════════════════════════════
# PATCH 1 — FIX ATR field missing from AssetSnapshot
# FIND the AssetSnapshot dataclass definition and ADD 'atr' field:
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AssetSnapshot:
    """Standardized market data snapshot from Coinglass + Binance feeds."""
    symbol: str = ""
    price: float = 0.0
    volume: float = 0.0
    rsi: float = 0.0
    atr: float = 0.0               # ← ADD THIS LINE
    fut_cvd: float = 0.0
    # ... rest unchanged ...