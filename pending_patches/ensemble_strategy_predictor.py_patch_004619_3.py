Python# TARGET: ensemble_strategy_predictor.py
# REPLACE the entire signal_s6 function:

def signal_s6(df: pd.DataFrame) -> np.ndarray:
    """S6: OI Momentum — ATR + chop + CVD div + OI-CVD + imbalance"""
    out = np.zeros(len(df), dtype=np.int32)
    mc = df.get("mc", pd.Series(0, index=df.index)).values
    p8 = df.get("p8", pd.Series(0, index=df.index)).values
    oi_rising = df.get("oi_rising", pd.Series(0, index=df.index)).values
    cvd_d = df.get("cvd_d", pd.Series(0, index=df.index)).values

    atr_s = _atr_scale(df)
    chop  = _is_chop(df).astype(np.int32)
    cvdb  = df.get("cvd_div_bear", pd.Series(0, index=df.index)).values
    cvdu  = df.get("cvd_div_bull", pd.Series(0, index=df.index)).values
    heavy_buy  = df.get("cvd_heavy_buy", pd.Series(0, index=df.index)).values
    heavy_sell = df.get("cvd_heavy_sell", pd.Series(0, index=df.index)).values
    imb_flat   = df.get("cvd_imb_flat", pd.Series(0, index=df.index)).values
    # CVD acceleration for trend leg quality check
    cvd_acc = np.zeros(len(df))
    if "CVD" in df.columns:
        cvd_acc = df["CVD"].diff().diff().fillna(0).values

    regime_ok   = _regime_pass(chop, cvdb, cvdu)
    oi_cvd_ok   = _oi_cvd_confluence(oi_rising, cvd_d)
    imb_ok      = _cvd_imbalance_pass(heavy_buy, heavy_sell, imb_flat, mc)

    th_trend = _atr_scale_threshold(0.18, atr_s)
    th_oi    = _atr_scale_threshold(0.12, atr_s)

    # Trend leg: CVD accelerating in direction of trade
    trend_l = (mc > 0) & (p8 < -th_trend) & (cvd_acc > 0)
    trend_s = (mc < 0) & (p8 >  th_trend) & (cvd_acc < 0)

    # OI leg: looser threshold + OI rising + OI-CVD confluence
    oi_l = (mc > 0) & (p8 < -th_oi) & (oi_rising > 0) & oi_cvd_ok
    oi_s = (mc < 0) & (p8 >  th_oi) & (oi_rising > 0) & oi_cvd_ok

    mask_l = (trend_l | oi_l) & regime_ok & imb_ok
    mask_s = (trend_s | oi_s) & regime_ok & imb_ok

    out[mask_l] = 1
    out[mask_s] = -1
    return out