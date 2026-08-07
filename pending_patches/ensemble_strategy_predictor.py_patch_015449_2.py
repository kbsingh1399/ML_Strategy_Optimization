Python# ensemble_strategy_predictor.py, ~line 260
def _atr_scale(df):
    """atr_ratio = current ATR / 100-period mean ATR | clamped [0.85, 1.15]"""
    bias = 1.0 + 0.50 * (atr_ratio - 1.0)
    return np.clip(bias, 0.85, 1.15)