Python# TARGET: ensemble_strategy_predictor.py
# REPLACE the entire signal_s7 function:

def signal_s7(df: pd.DataFrame) -> np.ndarray:
    """S7: CVD-Price Divergence — vectorized directional alpha."""
    out = np.zeros(len(df), dtype=np.int32)
    mc   = df.get("mc", pd.Series(0, index=df.index)).values
    cvdb = df.get("cvd_div_bear", pd.Series(0, index=df.index)).values
    cvdu = df.get("cvd_div_bull", pd.Series(0, index=df.index)).values
    cv_acc = df.get("cvd_accel", pd.Series(0, index=df.index)).values
    chop = _is_chop(df)
    # Bullish: uptrend + bullish CVD divergence + CVD accelerating + not choppy
    mask_l = (mc > 0) & (cvdu > 0) & (cv_acc > 0) & (~chop) & (cvdb == 0)
    # Bearish: downtrend + bearish CVD divergence + CVD accelerating + not choppy
    mask_s = (mc < 0) & (cvdb > 0) & (cv_acc < 0) & (~chop) & (cvdu == 0)
    out[mask_l] = 1
    out[mask_s] = -1
    return out