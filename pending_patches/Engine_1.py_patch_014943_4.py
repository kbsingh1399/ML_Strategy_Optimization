Python# TARGET: Engine_1.py
# FIND the print section in run_backtest that shows the results table
# UPDATE the header and format to include the new columns:
            print(f"\n{'='*90}")
            print(f"BACKTEST RESULTS — {args.backtest} (realistic fills, tiered fees)")
            print(f"{'='*90}")
            print(f"  {'Strategy':<22s} {'Trades':>6s} {'WR':>6s} {'PnL':>12s} "
                  f"{'Sharpe':>7s} {'Calmar':>7s} {'Sortino':>7s} {'MaxCL':>5s}")
            print(f"  {'─'*86}")
            for name, stats in results.items():
                is_ensemble = name.startswith("ENSEMBLE")
                prefix = "⭐ " if is_ensemble else "  "
                print(f"  {prefix}{name:<20s} {stats['trades']:>6d} "
                      f"{stats['wr']:>5.1f}% ${stats['total_pnl']:>11,.2f} "
                      f"{stats.get('sharpe', 0):>+6.2f} "
                      f"{stats.get('calmar', 0):>+6.2f} "
                      f"{stats.get('sortino', 0):>+6.2f} "
                      f"{stats.get('max_cons_losses', 0):>5d}")