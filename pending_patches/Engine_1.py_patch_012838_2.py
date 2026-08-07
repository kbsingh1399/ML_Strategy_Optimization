Python# TARGET: Engine_1.py
# ═══════════════════════════════════════════════════════════════════
# PATCH 2 — NumPy structured circular buffer for 1200-bar candle
# history. ADD this class before SnapshotStore.  Use it in
# EnsembleStrategyPredictor instead of deque[dict].
# ═══════════════════════════════════════════════════════════════════

import numpy as np

CANDLE_DTYPE = np.dtype([
    ("open_time", np.int64),
    ("Open",       np.float64),
    ("High",       np.float64),
    ("Low",        np.float64),
    ("Close",      np.float64),
    ("Volume",     np.float64),
    ("CVD",        np.float64),
    ("liq_long",   np.float64),
    ("liq_short",  np.float64),
    ("oi",         np.float64),
    ("funding",    np.float64),
    ("ls_ratio",   np.float64),
    ("bid_qty",    np.float64),
    ("ask_qty",    np.float64),
    ("delta_qty",  np.float64),
    ("bid_trades", np.float64),
    ("ask_trades", np.float64),
    ("poc_price",  np.float64),
    ("atr",        np.float64),
])

CANDLE_COLUMNS = [
    "open_time", "Open", "High", "Low", "Close", "Volume",
    "CVD", "Agg. Liq Long", "Agg. Liq Short", "Agg. OI",
    "Agg. Funding Rate", "Long/Short Ratio (Account)",
    "Bid Qty", "Ask Qty", "Delta Qty",
    "Bid Trades", "Ask Trades", "POC Price", "atr",
]

CANDLE_BUFFER_FIELD_MAP = {
    "Open": "Open", "High": "High", "Low": "Low",
    "Close": "Close", "Volume": "Volume", "CVD": "CVD",
    "Agg. Liq Long": "liq_long", "Agg. Liq Short": "liq_short",
    "Agg. OI": "oi", "Agg. Funding Rate": "funding",
    "Long/Short Ratio (Account)": "ls_ratio",
    "Bid Qty": "bid_qty", "Ask Qty": "ask_qty",
    "Delta Qty": "delta_qty", "Bid Trades": "bid_trades",
    "Ask Trades": "ask_trades", "POC Price": "poc_price",
    "atr": "atr",
}


class CandleBuffer:
    """Pre-allocated NumPy circular buffer for 15m candle history.

    Replaces deque[dict] with a flat structured array.  Appends are
    O(1) with wrap-around index.  to_dataframe() produces a pd.DataFrame
    directly from the structured array — no dict unpacking.

    Memory: 1200 bars × 19 fields × 8 bytes = ~182 KB per symbol.
    vs ~280 KB for dict-based deque (2× savings).
    """

    def __init__(self, maxlen: int = 1200):
        self.maxlen = maxlen
        self._buf = np.zeros(maxlen, dtype=CANDLE_DTYPE)
        self._write_idx: int = 0
        self._count: int = 0

    def append(self, row: dict) -> None:
        """Append a candle row dict.  Wraps when full."""
        idx = self._write_idx
        rec = self._buf[idx]
        rec["open_time"] = int(row.get("open_time", 0))
        for py_col, np_field in CANDLE_BUFFER_FIELD_MAP.items():
            rec[np_field] = float(row.get(py_col, 0.0))
        self._write_idx = (idx + 1) % self.maxlen
        self._count = min(self._count + 1, self.maxlen)

    def __len__(self) -> int:
        return self._count

    def get_slice(self, n: int = None) -> np.ndarray:
        """Return the last `n` rows as a structured array in time order."""
        n = n or self._count
        n = min(n, self._count)
        if n <= 0:
            return self._buf[:0]
        start = (self._write_idx - n) % self.maxlen
        if start + n <= self.maxlen:
            return self._buf[start:start + n].copy()
        # Wrap-around
        return np.concatenate([
            self._buf[start:],
            self._buf[:start + n - self.maxlen],
        ])

    def to_dataframe(self, n: int = None) -> "pd.DataFrame":
        """Convert to DataFrame for featurize()."""
        arr = self.get_slice(n)
        df = pd.DataFrame(arr)
        # Map structured field names back to expected column names
        reverse_map = {v: k for k, v in CANDLE_BUFFER_FIELD_MAP.items()}
        reverse_map["open_time"] = "open_time"
        df = df.rename(columns=reverse_map)
        return df

    def update_latest(self, row: dict) -> None:
        """Update the latest (current, unclosed) candle in-place."""
        if self._count == 0:
            self.append(row)
            return
        idx = (self._write_idx - 1) % self.maxlen
        rec = self._buf[idx]
        for py_col, np_field in CANDLE_BUFFER_FIELD_MAP.items():
            val = float(row.get(py_col, 0.0))
            if val != 0.0:
                rec[np_field] = val
        rec["atr"] = float(row.get("atr", rec["atr"]))