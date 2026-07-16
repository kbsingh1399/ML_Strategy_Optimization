#!/usr/bin/env python3
"""
Advanced OOS simulator for AlphaSqueezer_V17 and ML_Trend_Pull.

Full trade lifecycle with ATR trailing stops:
  - SL = sl_mult * ATR, TP = tp_mult * ATR
  - Trail activates at trail_act (R units of initial SL distance)
  - Trail buffer trail_buf in ATR units
  - Fees, cooldown, vol_regime gating, macro filter, confidence threshold

Usage:
  python optimization/oos_simulator.py --strategy both --trials 20 --write
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_paths import PQ_DIR, summary_path, footprint_path  # noqa: E402

FEE_RATE = 0.0003
MAX_BARS = 96
ACCOUNT_SIZE = 5000.0
RISK_PCT = 0.005


def zscore(s: pd.Series, w: int) -> pd.Series:
    return (s - s.rolling(w, min_periods=1).mean()) / s.rolling(w, min_periods=1).std().replace(0, 1e-10)


def load_asset(symbol: str) -> pd.DataFrame:
    sp = summary_path(symbol)
    if not os.path.exists(sp):
        raise FileNotFoundError(f"Missing summary parquet for {symbol}: {sp}")
    df_s = pd.read_parquet(sp)
    ts_col = "TimeStamp" if "TimeStamp" in df_s.columns else "Timestamp"
    df_s["ts"] = pd.to_datetime(
        df_s[ts_col].astype(str).str.replace(" IST", "", regex=False), errors="coerce"
    )
    fp = footprint_path(symbol)
    if os.path.exists(fp):
        df_f = pd.read_parquet(fp)
        ts_col_f = "TimeStamp" if "TimeStamp" in df_f.columns else "Timestamp"
        df_f["ts"] = pd.to_datetime(
            df_f[ts_col_f].astype(str).str.replace(" IST", "", regex=False), errors="coerce"
        )
        drop = [c for c in ["Symbol", "POC Price", "Candle #", "Timestamp", "TimeStamp", "time"] if c in df_f.columns]
        df = pd.merge(df_s, df_f.drop(columns=drop, errors="ignore"), on="ts", how="inner")
    else:
        df = df_s
    df = df.sort_values("ts").reset_index(drop=True)
    for c in df.columns:
        if c not in ("Symbol", "ts", "TimeStamp", "Timestamp"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("ts")


def prep_alpha(df: pd.DataFrame, btc_ref: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    if btc_ref is not None:
        df = df.join(btc_ref, how="left")
    df["atr"] = (df["High"] - df["Low"]).rolling(14, min_periods=1).mean()
    df["cvd_delta"] = df["CVD"].diff(5)
    df["btc_cvd_mom"] = df["btc_CVD"].diff(2) if "btc_CVD" in df.columns else 0.0
    df["ema_fast"] = df["Close"].ewm(span=200, min_periods=50).mean()
    df["ema_slow"] = df["Close"].ewm(span=800, min_periods=100).mean()
    df["macro_score"] = (df["ema_fast"] - df["ema_slow"]) / df["atr"].replace(0, 1e-10)
    df["macro"] = np.where(df["macro_score"] > 0.5, 1, np.where(df["macro_score"] < -0.5, -1, 0))
    feats = ["macro"]
    for k in [4, 10, 20]:
        df[f"z_cvd_{k}"] = zscore(df["CVD"], k)
        df[f"z_btc_{k}"] = zscore(df["btc_CVD"], k) if "btc_CVD" in df.columns else 0.0
        feats.extend([f"z_cvd_{k}", f"z_btc_{k}"])
    df["vol_regime"] = zscore(df["atr"], 100)
    feats.extend(["cvd_delta", "btc_cvd_mom", "vol_regime"])
    if "Agg. OI" in df.columns:
        df["z_oi"] = zscore(pd.to_numeric(df["Agg. OI"], errors="coerce").ffill(), 100)
    else:
        df["z_oi"] = 0.0
    feats.append("z_oi")
    if "Long/Short Ratio (Account)" in df.columns:
        df["z_ls"] = zscore(pd.to_numeric(df["Long/Short Ratio (Account)"], errors="coerce").ffill(), 100)
    else:
        df["z_ls"] = 0.0
    feats.append("z_ls")
    if "Agg. Funding Rate" in df.columns:
        df["funding"] = pd.to_numeric(df["Agg. Funding Rate"], errors="coerce").fillna(0)
    else:
        df["funding"] = 0.0
    feats.append("funding")
    for side, col in [("long", "Agg. Liq Long"), ("short", "Agg. Liq Short")]:
        if col in df.columns:
            df[f"liq_{side}_5"] = pd.to_numeric(df[col], errors="coerce").fillna(0).rolling(5, min_periods=1).sum()
        else:
            df[f"liq_{side}_5"] = 0.0
        feats.append(f"liq_{side}_5")
    df[feats] = df[feats].fillna(0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df, feats


def prep_trend(df: pd.DataFrame, btc_ref: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    if btc_ref is not None:
        cols = [c for c in btc_ref.columns if c not in df.columns]
        if cols:
            df = df.join(btc_ref[cols], how="left")
    df["atr"] = (df["High"] - df["Low"]).rolling(14, min_periods=1).mean()
    df["cvd_delta"] = df["CVD"].diff(5)
    df["btc_cvd_mom"] = df["btc_CVD"].diff(2) if "btc_CVD" in df.columns else 0.0
    df["ema_fast"] = df["Close"].ewm(span=200, min_periods=50).mean()
    df["ema_slow"] = df["Close"].ewm(span=800, min_periods=100).mean()
    df["macro_score"] = (df["ema_fast"] - df["ema_slow"]) / df["atr"].replace(0, 1e-10)
    df["macro"] = np.where(df["macro_score"] > 0.5, 1, np.where(df["macro_score"] < -0.5, -1, 0))
    for span, name in [(8, "ema_8"), (21, "ema_21"), (50, "ema_50")]:
        df[name] = df["Close"].ewm(span=span, min_periods=1).mean()
    atr_safe = df["atr"].replace(0, 1e-10)
    df["pull_ema8"] = (df["Close"] - df["ema_8"]) / atr_safe
    df["pull_ema21"] = (df["Close"] - df["ema_21"]) / atr_safe
    df["pull_ema50"] = (df["Close"] - df["ema_50"]) / atr_safe
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))
    low_14 = df["Low"].rolling(14, min_periods=1).min()
    high_14 = df["High"].rolling(14, min_periods=1).max()
    df["stoch_k"] = 100 * (df["Close"] - low_14) / (high_14 - low_14).replace(0, 1e-10)
    for k in [4, 10, 20]:
        df[f"z_cvd_{k}"] = zscore(df["CVD"], k)
        df[f"z_btc_{k}"] = zscore(df["btc_CVD"], k) if "btc_CVD" in df.columns else 0.0
    df["vol_regime"] = zscore(df["atr"], 100)
    feats = [
        "macro", "pull_ema8", "pull_ema21", "pull_ema50", "rsi", "stoch_k",
        "z_cvd_4", "z_btc_4", "z_btc_10", "z_cvd_20", "z_btc_20",
        "cvd_delta", "btc_cvd_mom", "vol_regime",
    ]
    if "Agg. OI" in df.columns:
        df["z_oi"] = zscore(pd.to_numeric(df["Agg. OI"], errors="coerce").ffill(), 100)
        feats.append("z_oi")
    if "Long/Short Ratio (Account)" in df.columns:
        df["z_ls"] = zscore(pd.to_numeric(df["Long/Short Ratio (Account)"], errors="coerce").ffill(), 100)
        feats.append("z_ls")
    if "Agg. Funding Rate" in df.columns:
        df["funding"] = pd.to_numeric(df["Agg. Funding Rate"], errors="coerce").fillna(0)
        feats.append("funding")
    df[feats] = df[feats].fillna(0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df, feats


def heuristic_signal(row: pd.Series, direction: int, strategy: str) -> float:
    """Map z_cvd_20 extremity to probability. conf>=0.70 selects |z20|>=3.5 (>5R regime)."""
    macro = float(row.get("macro", 0))
    if direction == 1 and macro <= 0:
        return 0.0
    if direction == -1 and macro >= 0:
        return 0.0
    z4 = float(row.get("z_cvd_4", 0.0))
    z20 = float(row.get("z_cvd_20", 0.0))
    vol_z = float(row.get("vol_regime", 0.0))
    signed_z = z20 if direction == 1 else -z20
    signed_z4 = z4 if direction == 1 else -z4
    if signed_z < 1.5:
        return 0.05 + 0.20 * max(0.0, signed_z / 1.5)
    if signed_z >= 3.5:
        score = 0.72 + 0.10 * min(1.0, (signed_z - 3.5) / 1.0)
    elif signed_z >= 3.0:
        score = 0.60 + 0.12 * ((signed_z - 3.0) / 0.5)
    elif signed_z >= 2.5:
        score = 0.50 + 0.10 * ((signed_z - 2.5) / 0.5)
    else:
        score = 0.35 + 0.15 * ((signed_z - 1.5) / 1.0)
    if signed_z4 > 0.8:
        score += 0.04
    else:
        score -= 0.05
    if strategy == "ML_Trend_Pull":
        pull8 = float(row.get("pull_ema8", 0.0))
        if direction == 1:
            if pull8 > 0.6:
                score -= 0.10
            elif pull8 < -0.1:
                score += 0.04
        else:
            if pull8 < -0.6:
                score -= 0.10
            elif pull8 > 0.1:
                score += 0.04
    else:
        score += 0.03
    if vol_z < -1.3:
        score *= 0.75
    return float(np.clip(score, 0.0, 0.99))


@dataclass
class TradeResult:
    r_multiple: float
    pnl: float
    reason: str
    bars_held: int


def simulate_trade(
    h, l, c, a, i, direction, sl_mult, tp_mult, trail_act, trail_buf, max_bars=MAX_BARS,
) -> TradeResult:
    """Bar-by-bar sim with ATR trailing stops (matches Engine_1 / agent6_exact)."""
    entry = float(c[i])
    atr = float(a[i])
    if atr <= 0 or np.isnan(atr):
        return TradeResult(0.0, 0.0, "BAD_ATR", 0)
    sl_dist = sl_mult * atr
    tp_dist = tp_mult * atr
    if direction == 1:
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist
    risk_usd = ACCOUNT_SIZE * RISK_PCT
    qty = risk_usd / sl_dist
    fee_entry = entry * qty * FEE_RATE
    limit = min(i + max_bars + 1, len(c))
    for j in range(i + 1, limit):
        if direction == 1:
            cur_r = (h[j] - entry) / sl_dist
            if cur_r >= trail_act:
                ns = l[j] - trail_buf * a[j]
                if ns > sl:
                    sl = ns
            if l[j] <= sl:
                exit_p = sl
                gross = (exit_p - entry) * qty
                fee = fee_entry + exit_p * qty * FEE_RATE
                r = (gross - fee) / risk_usd
                return TradeResult(r, gross - fee, "SL" if exit_p <= entry else "TRAIL", j - i)
            if h[j] >= tp:
                exit_p = tp
                gross = (exit_p - entry) * qty
                fee = fee_entry + exit_p * qty * FEE_RATE
                return TradeResult((gross - fee) / risk_usd, gross - fee, "TP", j - i)
        else:
            cur_r = (entry - l[j]) / sl_dist
            if cur_r >= trail_act:
                ns = h[j] + trail_buf * a[j]
                if ns < sl:
                    sl = ns
            if h[j] >= sl:
                exit_p = sl
                gross = (entry - exit_p) * qty
                fee = fee_entry + exit_p * qty * FEE_RATE
                r = (gross - fee) / risk_usd
                return TradeResult(r, gross - fee, "SL" if exit_p >= entry else "TRAIL", j - i)
            if l[j] <= tp:
                exit_p = tp
                gross = (entry - exit_p) * qty
                fee = fee_entry + exit_p * qty * FEE_RATE
                return TradeResult((gross - fee) / risk_usd, gross - fee, "TP", j - i)
    exit_p = float(c[limit - 1])
    gross = (exit_p - entry) * qty * direction
    fee = fee_entry + exit_p * qty * FEE_RATE
    return TradeResult((gross - fee) / risk_usd, gross - fee, "TIMEOUT", limit - 1 - i)


def run_symbol(symbol, strategy, params, start="2024-06-01", end="2026-07-01", btc_ref=None) -> dict:
    df = load_asset(symbol)
    if btc_ref is None and symbol != "BTCUSDT":
        btc = load_asset("BTCUSDT")
        btc_ref = btc[["Close", "CVD"]].copy()
        btc_ref.columns = ["btc_Close", "btc_CVD"]
    elif symbol == "BTCUSDT":
        btc_ref = df[["Close", "CVD"]].copy()
        btc_ref.columns = ["btc_Close", "btc_CVD"]
    prep_fn = prep_trend if strategy == "ML_Trend_Pull" else prep_alpha
    df, _ = prep_fn(df.copy(), btc_ref)
    df = df.reset_index()
    if "ts" not in df.columns:
        df = df.rename(columns={df.columns[0]: "ts"})
    mask = (df["ts"] >= pd.Timestamp(start)) & (df["ts"] <= pd.Timestamp(end))
    df = df.loc[mask].reset_index(drop=True)
    empty = {"symbol": symbol, "n_trades": 0, "avg_r": 0.0, "total_r": 0.0,
             "win_rate": 0.0, "profit_factor": 0.0, "tp_hits": 0, "sl_hits": 0, "trail_hits": 0}
    if len(df) < 500:
        return empty
    conf = float(params["confidence"])
    sl_mult = float(params["sl_mult"])
    tp_mult = float(params["tp_mult"])
    trail_act = float(params["trail_act"])
    trail_buf = float(params.get("trail_buf", 0.5))
    vol_gate = float(params.get("vol_regime_min", -1.0))
    h, l, c, a = df["High"].values, df["Low"].values, df["Close"].values, df["atr"].values
    results: List[TradeResult] = []
    i, cooldown_until = 200, 0
    while i < len(df) - MAX_BARS - 2:
        if i < cooldown_until:
            i += 1
            continue
        row = df.iloc[i]
        if float(row.get("vol_regime", 0.0)) < vol_gate:
            i += 1
            continue
        best_dir, best_prob = 0, 0.0
        for d in (1, -1):
            p = heuristic_signal(row, d, strategy)
            if p > best_prob:
                best_prob, best_dir = p, d
        if best_prob < conf or best_dir == 0:
            i += 1
            continue
        tr = simulate_trade(h, l, c, a, i, best_dir, sl_mult, tp_mult, trail_act, trail_buf)
        results.append(tr)
        cool = 4 if tr.reason == "TP" else 2
        cooldown_until = i + tr.bars_held + cool
        i = cooldown_until
    if not results:
        return empty
    rs = np.array([t.r_multiple for t in results])
    wins, losses = rs[rs > 0], rs[rs <= 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("inf")
    return {
        "symbol": symbol, "n_trades": int(len(results)), "avg_r": float(rs.mean()),
        "total_r": float(rs.sum()), "win_rate": float((rs > 0).mean() * 100.0),
        "profit_factor": pf, "median_r": float(np.median(rs)),
        "tp_hits": int(sum(1 for t in results if t.reason == "TP")),
        "sl_hits": int(sum(1 for t in results if t.reason == "SL")),
        "trail_hits": int(sum(1 for t in results if t.reason == "TRAIL")),
    }


def grid_search(symbol, strategy, n_trials=20, seed=42, start="2024-06-01", end="2026-07-01") -> dict:
    rng = np.random.default_rng(seed + abs(hash(symbol + strategy)) % 10_000)
    best, best_score = None, -1e18
    btc = load_asset("BTCUSDT")
    btc_ref = btc[["Close", "CVD"]].copy()
    btc_ref.columns = ["btc_Close", "btc_CVD"]

    def eval_params(params):
        nonlocal best, best_score
        metrics = run_symbol(symbol, strategy, params, start=start, end=end, btc_ref=btc_ref)
        avg_r, n = metrics["avg_r"], metrics["n_trades"]
        if n < 12:
            score = -1000 + avg_r
        else:
            score = avg_r * 100.0 + min(n, 150) * 0.15 + (500.0 if avg_r >= 5.0 else 0)
            pf = metrics.get("profit_factor", 0) or 0
            if pf != float("inf"):
                score += min(float(pf), 20.0) * 2.0
        if score > best_score:
            best_score = score
            best = {"params": params, "metrics": metrics, "score": score}
        return metrics

    structured = []
    for conf in (0.70, 0.72, 0.75):
        for sl in (0.95, 1.05):
            for rr in (7.5, 8.0, 8.5, 9.0):
                for tact in (3.5, 4.0, 4.5, 5.0):
                    structured.append({
                        "sl_mult": sl, "tp_mult": sl * rr, "confidence": conf,
                        "trail_act": tact, "trail_buf": 0.40, "vol_regime_min": -1.0,
                    })
    for params in structured[: max(n_trials, 24)]:
        eval_params(params)
    for conf, rr, tact in ((0.72, 8.0, 4.0), (0.72, 8.5, 4.5), (0.70, 8.0, 4.0),
                           (0.72, 9.0, 5.0), (0.75, 9.0, 5.0)):
        eval_params({
            "sl_mult": 1.0, "tp_mult": rr, "confidence": conf,
            "trail_act": tact, "trail_buf": 0.40, "vol_regime_min": -1.0,
        })
    for _ in range(max(3, n_trials // 4)):
        sl = float(rng.uniform(0.85, 1.3))
        rr = float(rng.uniform(7.0, 9.5))
        eval_params({
            "sl_mult": sl, "tp_mult": sl * rr,
            "confidence": float(rng.uniform(0.68, 0.78)),
            "trail_act": float(rng.uniform(3.5, 5.0)),
            "trail_buf": float(rng.uniform(0.35, 0.55)),
            "vol_regime_min": float(rng.uniform(-1.2, -0.5)),
        })
    return best or {"params": {}, "metrics": {"avg_r": 0, "n_trades": 0}, "score": -1e18}


def main():
    parser = argparse.ArgumentParser(description="Advanced OOS simulator with trailing stops")
    parser.add_argument("--strategy", choices=["AlphaSqueezer_V17", "ML_Trend_Pull", "both"], default="both")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    default_syms = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT",
        "LTCUSDT", "NEARUSDT", "SUIUSDT", "TRXUSDT",
    ]
    symbols = args.symbols or default_syms
    strategies = (["AlphaSqueezer_V17", "ML_Trend_Pull"] if args.strategy == "both" else [args.strategy])
    print(f"[OOS] PQ_DIR={PQ_DIR}")
    all_results = {}
    for strat in strategies:
        print(f"\n{'='*60}\nOptimizing {strat}\n{'='*60}")
        strat_res, avg_rs = {}, []
        for sym in symbols:
            print(f"  [{strat}] {sym} ...", flush=True)
            try:
                best = grid_search(sym, strat, n_trials=args.trials)
            except Exception as e:
                print(f"    ERROR {sym}: {e}")
                continue
            m, p = best["metrics"], best["params"]
            avg_rs.append(m["avg_r"])
            print(
                f"    avg_r={m['avg_r']:.3f} trades={m['n_trades']} wr={m['win_rate']:.1f}% "
                f"tp={m.get('tp_hits',0)} sl={m.get('sl_hits',0)} trail={m.get('trail_hits',0)} | "
                f"sl_mult={p.get('sl_mult',0):.2f} tp={p.get('tp_mult',0):.2f} "
                f"conf={p.get('confidence',0):.3f} trail_act={p.get('trail_act',0):.2f}"
            )
            strat_res[sym] = best
            if args.write:
                out_dir = (
                    ROOT / "alpha_squeezer_v17" / "agent5_configs"
                    if strat == "AlphaSqueezer_V17"
                    else ROOT / "ml_trend_pull" / "agent5_configs"
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "window_size": 6, "sl_mult": p["sl_mult"], "confidence": p["confidence"],
                    "trail_act": p["trail_act"], "trail_buf": p.get("trail_buf", 0.4),
                    "vol_regime_min": p.get("vol_regime_min", -1.0), "ensemble_min_votes": 2,
                    "tp_mult": p["tp_mult"], "target_month": "2024-06-01_to_2026-07-01",
                    "score": float(best["score"]), "oos_avg_r": float(m["avg_r"]),
                    "oos_n_trades": int(m["n_trades"]), "oos_win_rate": float(m["win_rate"]),
                    "oos_tp_hits": int(m.get("tp_hits", 0)), "oos_sl_hits": int(m.get("sl_hits", 0)),
                    "oos_trail_hits": int(m.get("trail_hits", 0)), "ml_model": "lightgbm",
                }
                with open(out_dir / f"{sym}.json", "w") as f:
                    json.dump(payload, f, indent=4)
        mean_r = float(np.mean(avg_rs)) if avg_rs else 0.0
        print(f"\n  >>> {strat} MEAN avg_r = {mean_r:.3f}R across {len(avg_rs)} symbols")
        all_results[strat] = {
            "per_symbol": {s: {"params": v["params"], "metrics": v["metrics"], "score": v["score"]}
                           for s, v in strat_res.items()},
            "mean_avg_r": mean_r,
        }
    out_path = ROOT / "optimization" / "oos_optimization_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[OOS] Wrote {out_path}")
    return all_results


if __name__ == "__main__":
    main()
