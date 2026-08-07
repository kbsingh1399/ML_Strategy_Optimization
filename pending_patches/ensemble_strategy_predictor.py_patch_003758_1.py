Python# TARGET: ensemble_strategy_predictor.py
# ═══════════════════════════════════════════════════════════════════
# PATCH 1 — ADD CVD divergence features to featurize()
# FIND the line:  df["vr5"] = df["Volume"] / (...
# and the line:  df = df.fillna(0).replace([np.inf, -np.inf], 0)
# INSERT between them:
# ═══════════════════════════════════════════════════════════════════

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