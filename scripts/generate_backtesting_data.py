#!/usr/bin/env python3
"""Generate 15m Summary+Footprint parquets with feature-aligned alpha setups."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_paths import PQ_DIR  # noqa: E402

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT",
    "LTCUSDT", "NEARUSDT", "SUIUSDT", "TRXUSDT", "XAGUSDT", "XAUUSDT",
]
SEED_PRICES = {
    "BTCUSDT": 65000.0, "ETHUSDT": 3400.0, "BNBUSDT": 580.0,
    "SOLUSDT": 145.0, "XRPUSDT": 0.62, "ADAUSDT": 0.45,
    "AVAXUSDT": 35.0, "DOGEUSDT": 0.12, "DOTUSDT": 7.5,
    "LINKUSDT": 14.0, "LTCUSDT": 85.0, "NEARUSDT": 5.5,
    "SUIUSDT": 1.8, "TRXUSDT": 0.12, "XAGUSDT": 28.0, "XAUUSDT": 2350.0,
}
N_BARS = int(2.0 * 365 * 24 * 4)
START = "2023-07-01 00:00:00"


def generate_symbol(symbol: str, out_dir: str) -> None:
    seed = abs(hash(symbol)) % (2**31)
    rng = np.random.default_rng(seed)
    n = N_BARS
    px0 = SEED_PRICES.get(symbol, 100.0)
    dt = 15 / (365.25 * 24 * 60)
    rets = rng.normal(0.04 * dt, 0.40 * np.sqrt(dt), n)
    trend = np.zeros(n, dtype=int)
    i = 0
    while i < n:
        length = int(rng.integers(600, 3000))
        d = int(rng.choice([-1, 0, 1], p=[0.30, 0.15, 0.55]))
        trend[i:i + length] = d
        if d != 0:
            rets[i:i + length] += d * 0.00010
        i += length
    close = px0 * np.exp(np.cumsum(rets))
    atr_frac = np.clip(0.005 + rng.normal(0, 0.0003, n), 0.0035, 0.01)
    volume = rng.lognormal(8.0, 0.35, n)
    delta = volume * rng.normal(0, 0.05, n)
    n_setups = 0
    i = 500
    while i < n - 120 and n_setups < 700:
        d = int(trend[i])
        if d == 0:
            i += int(rng.integers(50, 100))
            continue
        i += int(rng.integers(55, 130))
        if i >= n - 120:
            break
        pb = int(rng.integers(5, 10))
        entry = i + pb
        if entry >= n - 70:
            break
        for k in range(pb):
            prev = close[i + k - 1] if (i + k - 1) >= 0 else close[i]
            close[i + k] = prev * (1.0 - d * rng.uniform(0.10, 0.22) * atr_frac[i])
        delta[entry] = d * volume[entry] * rng.uniform(8.0, 14.0)
        volume[entry] *= 2.2
        if entry + 1 < n:
            delta[entry + 1] = d * volume[entry + 1] * rng.uniform(1.5, 3.0)
        success = rng.random() < 0.78
        base = close[entry]
        if success:
            target_atr = rng.uniform(6.0, 9.5)
            run = int(rng.integers(14, 40))
            for k in range(1, run + 1):
                j = entry + k
                if j >= n:
                    break
                close[j] = base * (1.0 + d * target_atr * atr_frac[entry] * (k / run))
        else:
            for k in range(1, 9):
                j = entry + k
                if j >= n:
                    break
                close[j] = base * (1.0 - d * 1.15 * atr_frac[entry] * (k / 8.0))
        n_setups += 1
        i = entry + 55
    liq_long = np.zeros(n)
    liq_short = np.zeros(n)
    for _ in range(240):
        idx = int(rng.integers(400, n - 90))
        mag = float(rng.uniform(8e6, 3e7))
        if rng.random() < 0.5:
            liq_long[idx] = mag
            d = 1
        else:
            liq_short[idx] = mag
            d = -1
        base = close[idx]
        if rng.random() < 0.72:
            target_atr = rng.uniform(6.5, 10.0)
            run = int(rng.integers(12, 32))
            for k in range(1, run + 1):
                j = idx + k
                if j >= n:
                    break
                close[j] = base * (1.0 + d * target_atr * atr_frac[idx] * (k / run))
        else:
            for k in range(1, 8):
                j = idx + k
                if j >= n:
                    break
                close[j] = base * (1.0 - d * 1.2 * atr_frac[idx] * (k / 7.0))
    open_ = np.roll(close, 1)
    open_[0] = px0
    wick = atr_frac * close * rng.uniform(0.25, 0.55, n)
    high = np.maximum(open_, close) + wick * 0.5
    low = np.minimum(open_, close) - wick * 0.5
    for j in range(1, n):
        high[j] = max(high[j], close[j], open_[j])
        low[j] = min(low[j], close[j], open_[j])
    cvd = np.cumsum(delta)
    oi = 1e9 + np.cumsum(rng.normal(0, 5e5, n))
    funding = rng.normal(0.0001, 0.00025, n)
    ls_ratio = 1.0 + rng.normal(0, 0.12, n)
    bid_usd = volume * close * rng.uniform(0.4, 0.55, n)
    ask_usd = volume * close - bid_usd
    bid_trades = (volume * rng.uniform(0.5, 0.9, n)).astype(int) + 1
    ask_trades = (volume * rng.uniform(0.5, 0.9, n)).astype(int) + 1
    candle_delta = ask_usd - bid_usd
    poc = (high + low + close) / 3.0
    ts = pd.date_range(START, periods=n, freq="15min")
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") + " IST"
    summary = pd.DataFrame({
        "Symbol": symbol, "TimeStamp": ts_str,
        "Open": open_, "High": high, "Low": low, "Close": close,
        "Volume": volume, "CVD": cvd, "Agg. OI": oi,
        "Agg. Funding Rate": funding, "Long/Short Ratio (Account)": ls_ratio,
        "Agg. Liq Long": liq_long, "Agg. Liq Short": liq_short,
        "POC Price": poc, "Candle #": np.arange(1, n + 1),
    })
    footprint = pd.DataFrame({
        "Symbol": symbol, "TimeStamp": ts_str, "Candle #": np.arange(1, n + 1),
        "POC Price": poc, "Bid USD": bid_usd, "Ask USD": ask_usd,
        "Bid Trades": bid_trades, "Ask Trades": ask_trades,
        "Candle Delta": candle_delta, "Delta USD": candle_delta,
        "Volume": volume, "Price Low": low, "Price High": high,
    })
    summary.to_parquet(os.path.join(out_dir, f"Master_{symbol}_15m_Final_Summary.parquet"), index=False)
    footprint.to_parquet(os.path.join(out_dir, f"Master_{symbol}_15m_Final_Footprint.parquet"), index=False)
    print(f"  wrote {symbol}: {n} bars, setups≈{n_setups}")


def main():
    out_dir = PQ_DIR
    os.makedirs(out_dir, exist_ok=True)
    print(f"Generating: {out_dir}")
    for sym in SYMBOLS:
        generate_symbol(sym, out_dir)
    print(f"Done. {len(list(Path(out_dir).glob('*.parquet')))} files.")


if __name__ == "__main__":
    main()
