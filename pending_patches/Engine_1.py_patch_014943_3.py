Python# TARGET: Engine_1.py
# ═══════════════════════════════════════════════════════════════════
# PATCH 3 — Walk-forward metric report in run_backtest()
# FIND the per-strategy backtest results dict that currently has:
#   results[name] = {"trades": ..., "wins": ..., "wr": ..., ...}
# ADD these keys after the existing dict entries.  Also ADD a
# summary section at the end of run_backtest() that prints the
# enhanced metrics.
# ═══════════════════════════════════════════════════════════════════

        if trades:
            wins = [t for t in trades if t["pnl"] > 0]
            total_pnl = sum(t["pnl"] for t in trades)
            wr = len(wins) / len(trades) * 100
            avg_r = np.mean([t["r"] for t in trades])
            max_mae = max(t["mae_dd"] for t in trades)

            # ── Enhanced metrics: Sharpe, Calmar, Sortino ─────────
            r_vals = np.array([t["r"] for t in trades])
            r_std = np.std(r_vals) if len(r_vals) > 1 else 1.0
            cum_r = np.cumsum(r_vals)
            peak = np.maximum.accumulate(cum_r)
            drawdowns = peak - cum_r
            max_dd_r = np.max(drawdowns) if len(drawdowns) > 0 else 1.0
            # Annualised: 96 bars/day × 365 days
            ann_factor = np.sqrt(96 * 365 / len(trades))
            sharpe = (avg_r / r_std) * ann_factor if r_std > 0 else 0.0
            calmar = cum_r[-1] / max_dd_r if max_dd_r > 0 else 0.0
            down = r_vals[r_vals < 0]
            down_std = np.std(down) if len(down) > 1 else r_std
            sortino = (avg_r / down_std) * ann_factor if down_std > 0 else sharpe

            # Max consecutive losses
            max_cons_loss = 0
            cons = 0
            for t in trades:
                if t["pnl"] <= 0:
                    cons += 1
                    max_cons_loss = max(max_cons_loss, cons)
                else:
                    cons = 0

            results[name] = {
                "trades": len(trades), "wins": len(wins),
                "wr": round(wr, 1), "total_pnl": round(total_pnl, 2),
                "avg_r": round(float(avg_r), 3),
                "max_mae_dd": round(max_mae, 2),
                "sharpe": round(float(sharpe), 3),
                "calmar": round(float(calmar), 3),
                "sortino": round(float(sortino), 3),
                "max_cons_losses": max_cons_loss,
                "profit_factor": (
                    round(sum(t["pnl"] for t in wins) /
                          max(abs(sum(t["pnl"] for t in trades if t["pnl"] < 0)), 1.0), 2)
                    if wins and any(t["pnl"] < 0 for t in trades) else
                    (round(total_pnl, 2) if total_pnl > 0 else 0.0)
                ),
            }