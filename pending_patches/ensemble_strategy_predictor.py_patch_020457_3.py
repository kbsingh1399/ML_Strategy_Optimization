Python# ensemble_strategy_predictor.py, featurize(), end of chunk 1
if "Agg. Liq Long" in df.columns:
    liq_l  = pd.to_numeric(df["Agg. Liq Long"], errors="coerce").fillna(0)
    liq_ma = liq_l.rolling(100, min_periods=20).mean()
    liq_std = liq_l.rolling(100, min_periods=20).std() + 1e-10
    df["liq_cascade"] = ((liq_l > liq_ma + 2.5 * liq_std) & (liq_ma > 0)).astype(int)