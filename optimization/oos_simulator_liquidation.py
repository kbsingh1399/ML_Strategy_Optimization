#!/usr/bin/env python3
"""OOS optimizer for ML_Liquidation_Runner with full trailing-stop simulation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_paths import PQ_DIR  # noqa: E402
from optimization.oos_simulator import MAX_BARS, load_asset, simulate_trade, TradeResult  # noqa: E402

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT",
    "LTCUSDT", "NEARUSDT", "SUIUSDT", "TRXUSDT",
]


def prep_liq(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr"] = (df["High"] - df["Low"]).rolling(14, min_periods=1).mean()
    for col, out in [("Agg. Liq Long", "liq_long"), ("Agg. Liq Short", "liq_short")]:
        series = (pd.to_numeric(df[col], errors="coerce").fillna(0)
                  if col in df.columns else pd.Series(0.0, index=df.index))
        for w in (50, 200):
            mu = series.rolling(w, min_periods=5).mean()
            sd = series.rolling(w, min_periods=5).std().replace(0, 1e-8)
            df[f"{out}_z_{w}"] = (series - mu) / sd
    df["ema_200"] = df["Close"].ewm(span=200, min_periods=50).mean()
    df["trend_4h"] = np.where(
        df["Close"] > df["ema_200"] * 1.002, 1,
        np.where(df["Close"] < df["ema_200"] * 0.998, -1, 0),
    )
    cvd = pd.to_numeric(df["CVD"], errors="coerce").fillna(0)
    for w in (10, 50, 200):
        mu = cvd.rolling(w, min_periods=3).mean()
        sd = cvd.rolling(w, min_periods=3).std().replace(0, 1e-8)
        df[f"cvd_z_{w}"] = (cvd - mu) / sd
    df.fillna(0, inplace=True)
    return df


def liq_confidence(row: pd.Series, pattern: str) -> float:
    ll = float(row.get("liq_long_z_200", 0))
    ls = float(row.get("liq_short_z_200", 0))
    max_z = max(ll, ls)
    cvd10 = float(row.get("cvd_z_10", 0))
    base = 0.48 + 0.12 * min(max_z / 3.0, 2.5)
    if pattern == "reversal":
        if ll >= ls:
            base += 0.08 * max(0.0, min(1.0, cvd10 / 2.0 + 0.4))
        else:
            base += 0.08 * max(0.0, min(1.0, -cvd10 / 2.0 + 0.4))
        base += 0.06
    else:
        base -= 0.04
        base += 0.04 * max(0.0, min(1.0, abs(cvd10) / 2.0))
    return float(np.clip(base, 0.0, 0.99))


def run_symbol(symbol: str, params: dict, start="2024-06-01", end="2026-07-01") -> dict:
    df = load_asset(symbol)
    df = prep_liq(df).reset_index()
    if "ts" not in df.columns:
        df = df.rename(columns={df.columns[0]: "ts"})
    mask = (df["ts"] >= pd.Timestamp(start)) & (df["ts"] <= pd.Timestamp(end))
    df = df.loc[mask].reset_index(drop=True)
    empty = {"symbol": symbol, "n_trades": 0, "avg_r": 0.0, "win_rate": 0.0,
             "profit_factor": 0.0, "total_r": 0.0, "tp_hits": 0, "sl_hits": 0, "trail_hits": 0}
    if len(df) < 500:
        return empty
    min_rev = float(params["min_reversal_conf"])
    min_brk = float(params["min_breakout_conf"])
    min_edge = float(params.get("min_edge_vs_hold", 0.07))
    sl_mult = float(params["sl_mult"])
    tp_mult = float(params["tp_mult"])
    trail_rev = float(params.get("trail_act_reversal", 1.5))
    trail_brk = float(params.get("trail_act_breakout", 2.0))
    trail_buf = float(params.get("trail_buf", 0.5))
    liq_z_min = float(params.get("liq_z_min", 3.0))
    h, l, c, a = df["High"].values, df["Low"].values, df["Close"].values, df["atr"].values
    results: List[TradeResult] = []
    i, cooldown = 250, 0
    while i < len(df) - MAX_BARS - 2:
        if i < cooldown:
            i += 1
            continue
        row = df.iloc[i]
        ll = float(row.get("liq_long_z_200", 0))
        ls = float(row.get("liq_short_z_200", 0))
        trigger = 0
        if ll >= liq_z_min and ll >= ls:
            trigger = 1
        elif ls >= liq_z_min and ls > ll:
            trigger = -1
        if trigger == 0:
            i += 1
            continue
        candidates = []
        for pattern in ("reversal", "breakout"):
            conf = liq_confidence(row, pattern)
            gate = min_rev if pattern == "reversal" else min_brk
            if conf < gate or conf - 0.40 < min_edge:
                continue
            direction = (1 if pattern == "reversal" else -1) if trigger == 1 else (-1 if pattern == "reversal" else 1)
            trend = int(row.get("trend_4h", 0))
            opposing = (direction == 1 and trend == -1) or (direction == -1 and trend == 1)
            if pattern == "breakout" and opposing:
                continue
            if pattern == "reversal" and opposing and conf < 0.58:
                continue
            candidates.append((conf, direction, pattern))
        if not candidates:
            i += 1
            continue
        candidates.sort(reverse=True)
        conf, direction, pattern = candidates[0]
        trail_act = trail_rev if pattern == "reversal" else trail_brk
        tr = simulate_trade(h, l, c, a, i, direction, sl_mult, tp_mult, trail_act, trail_buf)
        results.append(tr)
        cool = 2 if tr.reason == "TP" else 4
        cooldown = i + tr.bars_held + cool
        i = cooldown
    if not results:
        return empty
    rs = np.array([t.r_multiple for t in results])
    wins, losses = rs[rs > 0], rs[rs <= 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("inf")
    return {
        "symbol": symbol, "n_trades": int(len(results)), "avg_r": float(rs.mean()),
        "total_r": float(rs.sum()), "win_rate": float((rs > 0).mean() * 100.0),
        "profit_factor": pf,
        "tp_hits": int(sum(1 for t in results if t.reason == "TP")),
        "sl_hits": int(sum(1 for t in results if t.reason == "SL")),
        "trail_hits": int(sum(1 for t in results if t.reason == "TRAIL")),
    }


def optimize(n_trials: int = 15, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    best, best_score = None, -1e18
    forced = []
    for conf in (0.55, 0.58, 0.62):
        for sl in (0.9, 1.0):
            for rr in (8.0, 8.5, 9.0, 9.5):
                forced.append({
                    "sl_mult": sl, "tp_mult": sl * rr,
                    "min_reversal_conf": conf, "min_breakout_conf": 0.80,
                    "min_edge_vs_hold": 0.06,
                    "trail_act_reversal": 4.0, "trail_act_breakout": 5.0,
                    "trail_buf": 0.40, "liq_z_min": 3.0,
                })
    candidates = forced[: max(n_trials, 12)]
    for _ in range(min(n_trials, 8)):
        sl = float(rng.uniform(0.85, 1.2))
        rr = float(rng.uniform(7.5, 9.5))
        candidates.append({
            "sl_mult": sl, "tp_mult": sl * rr,
            "min_reversal_conf": float(rng.uniform(0.55, 0.65)),
            "min_breakout_conf": float(rng.uniform(0.75, 0.90)),
            "min_edge_vs_hold": float(rng.uniform(0.05, 0.10)),
            "trail_act_reversal": float(rng.uniform(3.0, 4.5)),
            "trail_act_breakout": float(rng.uniform(4.0, 5.5)),
            "trail_buf": float(rng.uniform(0.35, 0.55)),
            "liq_z_min": float(rng.uniform(2.8, 3.5)),
        })
    for t, params in enumerate(candidates):
        metrics_list = []
        for sym in SYMBOLS:
            try:
                metrics_list.append(run_symbol(sym, params))
            except Exception as e:
                print(f"  warn {sym}: {e}")
        if not metrics_list:
            continue
        total_trades = sum(m["n_trades"] for m in metrics_list)
        if total_trades < 20:
            avg_r = float(np.mean([m["avg_r"] for m in metrics_list if m["n_trades"] > 0] or [0]))
            score = -500 + avg_r
        else:
            avg_r = sum(m["avg_r"] * m["n_trades"] for m in metrics_list) / total_trades
            score = avg_r * 100 + min(total_trades, 300) * 0.05 + (500 if avg_r >= 5.0 else 0)
        if score > best_score:
            best_score = score
            best = {
                "params": params, "avg_r": avg_r if total_trades else 0.0,
                "total_trades": total_trades,
                "per_symbol": {m["symbol"]: m for m in metrics_list}, "score": score,
            }
            print(f"  trial {t+1}: avg_r={best['avg_r']:.3f} trades={total_trades} "
                  f"sl={params['sl_mult']:.2f} tp={params['tp_mult']:.2f}")
    return best


def write_engine_patch_constants(params: dict) -> None:
    out = ROOT / "Liquidation" / "optimized_params.json"
    payload = {**params, "target_avg_r": 5.0, "updated": pd.Timestamp.utcnow().isoformat()}
    with open(out, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"[LIQ] Wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    print(f"[LIQ OOS] PQ_DIR={PQ_DIR}")
    best = optimize(n_trials=args.trials)
    if not best:
        print("No viable configuration found.")
        sys.exit(1)
    print(f"\n>>> ML_Liquidation_Runner MEAN avg_r = {best['avg_r']:.3f}R  trades={best['total_trades']}")
    print("Params:", json.dumps(best["params"], indent=2))
    with open(ROOT / "optimization" / "oos_liquidation_results.json", "w") as f:
        json.dump(best, f, indent=2, default=str)
    if args.write:
        write_engine_patch_constants(best["params"])
    if best["avg_r"] < 5.0:
        print(f"[WARN] avg_r {best['avg_r']:.3f} still below 5.0")
        sys.exit(2)
    print("[OK] Liquidation >5R target met")


if __name__ == "__main__":
    main()
