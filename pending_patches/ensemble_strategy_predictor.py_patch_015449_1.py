Python# ensemble_strategy_predictor.py, ~line 270
def _is_chop(df):
    """3-condition: ATR compressed ≥30% | narrow range < 1.5×ATR | |mc| < 0.3"""
    atr_compress = (atr < atr_ma20 * 0.70)           # condition 1
    range_narrow = (range_mean / atr) < 1.5           # condition 2
    weak_macro   = (np.abs(mc) < 0.3)                # condition 3
    return (atr_compress + range_narrow + weak_macro) >= 2