Python# TARGET: live_model_trainer.py
# ═══════════════════════════════════════════════════════════════════
# PATCH 1 — Enhanced feature engineering: log-returns, rolling stats,
# temporal lags. ADD these after the existing `prep_alpha` function
# returns, and call from within each prep_* function.
# ═══════════════════════════════════════════════════════════════════

def _add_advanced_features(df: pd.DataFrame, feature_list: list,
                            price_col: str = "Close",
                            cvd_col: str = "CVD") -> None:
    """Add log-returns, rolling higher moments, and temporal lags
    to the DataFrame and extend feature_list in-place."""
    new_feats = []

    # ── Log returns (1-bar, 3-bar, 5-bar) ────────────────────────
    if price_col in df.columns:
        for lag in [1, 3, 5]:
            col = f"log_ret_{lag}"
            df[col] = np.log(df[price_col] / df[price_col].shift(lag).replace(0, np.nan))
            df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)
            new_feats.append(col)

    # ── Rolling higher moments of log_ret_1 (20-bar window) ──────
    if "log_ret_1" in df.columns:
        r = df["log_ret_1"]
        for w in [10, 20]:
            df[f"ret_skew_{w}"] = r.rolling(w, min_periods=5).skew().fillna(0)
            df[f"ret_kurt_{w}"] = r.rolling(w, min_periods=5).kurt().fillna(0)
            new_feats.extend([f"ret_skew_{w}", f"ret_kurt_{w}"])

    # ── ATR ratio (current ATR / 100-bar mean) ───────────────────
    if "atr" in df.columns:
        atr_ma100 = df["atr"].rolling(100, min_periods=10).mean()
        df["atr_ratio"] = (df["atr"] / (atr_ma100 + 1e-10)).clip(0.3, 3.0)
        df["atr_ratio"] = df["atr_ratio"].fillna(1.0)
        new_feats.append("atr_ratio")

    # ── CVD acceleration (2nd derivative) ────────────────────────
    if cvd_col in df.columns and cvd_col in df.columns:
        cvd_vals = df[cvd_col].fillna(method='ffill')
        df["cvd_accel"] = cvd_vals.diff().diff().fillna(0)
        new_feats.append("cvd_accel")

    # ── Temporal lags: p8_t-1, cvd_d_t-1, atr_ratio_t-1 ─────────
    for src_col, lag_name in [("atr_ratio", "atr_ratio_lag1"),
                               ("log_ret_1", "ret_lag1")]:
        if src_col in df.columns:
            df[lag_name] = df[src_col].shift(1).fillna(0)
            new_feats.append(lag_name)

    # ── Price / ATR ratio (normalized price) ──────────────────────
    if price_col in df.columns and "atr" in df.columns:
        atr_safe = df["atr"].replace(0, 1e-10)
        df["price_atr"] = df[price_col] / atr_safe
        new_feats.append("price_atr")

    feature_list.extend(new_feats)
    df[new_feats] = df[new_feats].fillna(0).replace([np.inf, -np.inf], 0)