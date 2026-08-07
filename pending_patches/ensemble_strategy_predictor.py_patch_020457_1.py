Python# ensemble_strategy_predictor.py, featurize(), lines in chunk 1
# ── CVD Imbalance Ratio (order-book conviction) ─────────────────
bid_col = "Bid Qty" if "Bid Qty" in df.columns else ("Bid USD" if "Bid USD" in df.columns else "")
ask_col = "Ask Qty" if "Ask Qty" in df.columns else ("Ask USD" if "Ask USD" in df.columns else "")
if bid_col and ask_col:
    bq = df[bid_col].fillna(0)
    aq = df[ask_col].fillna(0)
    denom = bq + aq
    df["cvd_imbalance"] = np.where(denom > 0, bq / denom, 0.50)
    df["cvd_heavy_buy"]  = (df["cvd_imbalance"] > 0.65).astype(int)
    df["cvd_heavy_sell"] = (df["cvd_imbalance"] < 0.35).astype(int)
    df["cvd_imb_z"] = (df["cvd_imbalance"] - imb_ma) / imb_std
    df["cvd_imb_flat"] = (df["cvd_imb_z"].abs() < 0.5).astype(int)