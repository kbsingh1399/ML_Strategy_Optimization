Python# TARGET: ensemble_strategy_predictor.py
# FIND:    df["vr5"] = df["Volume"] / (...)
#          df = df.fillna(0).replace(...)
# INSERT between them:

    # ── CVD Divergence: price-CVD micro-divergence ──────────────────
    if "CVD" in df.columns:
        ph  = df["High"].rolling(20, min_periods=5).max()
        pl  = df["Low"].rolling(20, min_periods=5).min()
        cvh = df["CVD"].rolling(20, min_periods=5).max()
        cvl = df["CVD"].rolling(20, min_periods=5).min()
        df["cvd_div_bear"] = (
            (df["High"] >= ph.shift(1) * 0.995) &
            (df["CVD"] < cvh.shift(1) * 0.95)
        ).astype(int)
        df["cvd_div_bull"] = (
            (df["Low"] <= pl.shift(1) * 1.005) &
            (df["CVD"] > cvl.shift(1) * 1.05)
        ).astype(int)
        df["cvd_accel"] = df.get("cvd_d", df["Close"] * 0).diff(3)
        df["cvd_absorb"] = (
            (df["cvd_d"].fillna(0) > 0) &
            (df["Close"].diff(3).fillna(0) < 0)
        ).astype(int)
    else:
        for c in ["cvd_div_bear","cvd_div_bull","cvd_accel","cvd_absorb"]:
            df[c] = 0

    # ── CVD Imbalance Ratio (order-book conviction) ─────────────────
    if "Bid Qty" in df.columns and "Ask Qty" in df.columns:
        bq = df["Bid Qty"].fillna(0)
        aq = df["Ask Qty"].fillna(0)
        denom = bq + aq
        df["cvd_imbalance"] = np.where(denom > 0, bq / denom, 0.50)
        # Extreme conviction: > 0.65 buy, < 0.35 sell
        df["cvd_heavy_buy"]  = (df["cvd_imbalance"] > 0.65).astype(int)
        df["cvd_heavy_sell"] = (df["cvd_imbalance"] < 0.35).astype(int)
        # Imbalance z-score (20-bar rolling)
        imb_ma  = df["cvd_imbalance"].rolling(20, min_periods=5).mean()
        imb_std = df["cvd_imbalance"].rolling(20, min_periods=5).std() + 1e-10
        df["cvd_imb_z"] = (df["cvd_imbalance"] - imb_ma) / imb_std
        # Low-imbalance flag: |z| < 0.5 → no conviction
        df["cvd_imb_flat"] = (df["cvd_imb_z"].abs() < 0.5).astype(int)
    else:
        for c in ["cvd_imbalance","cvd_heavy_buy","cvd_heavy_sell",
                  "cvd_imb_z","cvd_imb_flat"]:
            df[c] = 0 if "flat" in c else (0.50 if "imbalance" in c and "z" not in c else 0)

    # ── Liquidation cascade flag ────────────────────────────────────
    if "Agg. Liq Long" in df.columns:
        liq_l   = pd.to_numeric(df["Agg. Liq Long"], errors="coerce").fillna(0)
        liq_ma  = liq_l.rolling(100, min_periods=20).mean()
        liq_std = liq_l.rolling(100, min_periods=20).std() + 1e-10
        df["liq_cascade"] = (
            (liq_l > liq_ma + 2.5 * liq_std) & (liq_ma > 0)
        ).astype(int)
    else:
        df["liq_cascade"] = 0