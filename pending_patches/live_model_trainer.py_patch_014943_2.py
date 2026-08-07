Python# TARGET: live_model_trainer.py
# ═══════════════════════════════════════════════════════════════════
# PATCH 2 — Walk-forward with purge/embargo and OOS metric logging
# FIND the walk_forward_train_and_evaluate function (or ADD it
# right before train_all_strategies).  Then replace the Optuna
# evaluation loop to call this instead of the inline 70/30 split.
# ═══════════════════════════════════════════════════════════════════

def walk_forward_evaluate(df_trades: pd.DataFrame,
                          n_windows: int = 5,
                          embargo_bars: int = 96,
                          min_train_trades: int = 20,
                          ml_params: dict = None,
                          prob_threshold: float = 0.6) -> dict:
    """Walk-forward validation with embargo and metric reporting.

    Partition: [train_0 | embargo | test_0] [train_1 | embargo | test_1] ...
    Each window advances by ~1/n of the data after purging the embargo.
    Returns aggregated OOS metrics and per-window detail.

    Returns:
        dict with keys:
          - 'oos_sharpe': annualized Sharpe of concatenated OOS returns
          - 'oos_calmar': annualized Calmar (return / max_drawdown)
          - 'oos_sortino': annualized Sortino (downside-only Sharpe)
          - 'oos_wr': OOS win rate
          - 'oos_avg_r': OOS average R-multiple
          - 'windows': list of per-window dicts {sharpe, calmar, wr, n_trades, ...}
    """
    if ml_params is None:
        ml_params = {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 100}

    df = df_trades.sort_values('entry_time').reset_index(drop=True)
    n_total = len(df)
    if n_total < min_train_trades * 2:
        return None

    window_size = n_total // n_windows
    per_window = []
    all_oos_r = []

    for w in range(n_windows):
        # Train = everything before current window
        train_end = w * window_size
        # OOS = current window, after embargo gap
        oos_start = train_end + embargo_bars
        oos_end = min(oos_start + window_size, n_total)

        if oos_start >= n_total or oos_end - oos_start < 5:
            break

        # Purge train data: only trades whose exit_time < oos_start
        # to prevent overlapping-trade leakage
        if 'exit_time' in df.columns:
            train_mask = (df.index < train_end) & (
                pd.to_datetime(df['exit_time']) < pd.to_datetime(
                    df['entry_time'].iloc[oos_start]
                    if oos_start < n_total
                    else df['entry_time'].iloc[-1]
                )
            )
            train_df = df[train_mask]
        else:
            train_df = df.iloc[:train_end]

        oos_df = df.iloc[oos_start:oos_end]

        if len(train_df) < min_train_trades or len(oos_df) < 5:
            continue

        # ── Train model ──────────────────────────────────────────
        m, cols = build_model_fast(train_df, **ml_params)
        if m is None:
            continue

        # ── Predict on OOS ───────────────────────────────────────
        preds = predict_model_fast(m, cols, oos_df)
        high_conf = preds[preds['prob'] >= prob_threshold]
        if len(high_conf) < 2:
            continue

        # ── Compute metrics ──────────────────────────────────────
        r_series = high_conf['r_multiple'].values
        win_r = r_series[r_series > 0]
        loss_r = np.abs(r_series[r_series < 0])

        oos_wr = len(win_r) / len(r_series) * 100
        oos_avg_r = np.mean(r_series)
        std_r = np.std(r_series) if len(r_series) > 1 else 1.0

        # Sharpe (annualized, assuming 96 bars/day)
        bars_per_year = 96 * 365
        if std_r > 0:
            oos_sharpe = (oos_avg_r / std_r) * np.sqrt(bars_per_year / len(r_series))
        else:
            oos_sharpe = 0.0

        # Calmar = return / max drawdown
        cum_r = np.cumsum(r_series)
        peak = np.maximum.accumulate(cum_r)
        dd = peak - cum_r
        max_dd = np.max(dd) if len(dd) > 0 else 1.0
        oos_calmar = cum_r[-1] / max_dd if max_dd > 0 else cum_r[-1]

        # Sortino = return / downside deviation
        downside = r_series[r_series < 0]
        down_std = np.std(downside) if len(downside) > 1 else std_r
        oos_sortino = (oos_avg_r / down_std) * np.sqrt(
            bars_per_year / len(r_series)) if down_std > 0 else oos_sharpe

        all_oos_r.extend(r_series.tolist())
        per_window.append({
            'window': w,
            'n_train': len(train_df),
            'n_oos': len(oos_df),
            'n_trades': len(high_conf),
            'wr': round(oos_wr, 1),
            'avg_r': round(float(oos_avg_r), 3),
            'sharpe': round(float(oos_sharpe), 3),
            'calmar': round(float(oos_calmar), 3),
            'sortino': round(float(oos_sortino), 3),
            'total_r': round(float(np.sum(r_series)), 3),
        })

    if not all_oos_r:
        return None

    all_r = np.array(all_oos_r)
    bars_per_year = 96 * 365
    agg_std = np.std(all_r) if len(all_r) > 1 else 1.0
    agg_sharpe = (np.mean(all_r) / agg_std) * np.sqrt(
        bars_per_year / len(all_r)) if agg_std > 0 else 0.0
    cum_all = np.cumsum(all_r)
    peak_all = np.maximum.accumulate(cum_all)
    dd_all = peak_all - cum_all
    max_dd_all = np.max(dd_all) if len(dd_all) > 0 else 1.0
    agg_calmar = cum_all[-1] / max_dd_all if max_dd_all > 0 else cum_all[-1]
    agg_wr = sum(1 for r in all_r if r > 0) / len(all_r) * 100

    down_r = all_r[all_r < 0]
    down_std = np.std(down_r) if len(down_r) > 1 else agg_std
    agg_sortino = (np.mean(all_r) / down_std) * np.sqrt(
        bars_per_year / len(all_r)) if down_std > 0 else agg_sharpe

    return {
        'oos_sharpe': round(float(agg_sharpe), 3),
        'oos_calmar': round(float(agg_calmar), 3),
        'oos_sortino': round(float(agg_sortino), 3),
        'oos_wr': round(float(agg_wr), 1),
        'oos_avg_r': round(float(np.mean(all_r)), 3),
        'oos_trades': len(all_r),
        'windows': per_window,
    }