Python# TARGET: ensemble_strategy_predictor.py
# ═══════════════════════════════════════════════════════════════════
# PATCH 2 — IMPROVE _is_chop() to 3-condition (ATR compression + 
# narrow range + weak macro)
# FIND:  def _is_chop(df: pd.DataFrame) -> np.ndarray:
# REPLACE the entire function:
# ═══════════════════════════════════════════════════════════════════

def _is_chop(df: pd.DataFrame) -> np.ndarray:
    """Chop detection: 2+ of 3 conditions must be met:
    1. ATR compressed ≥30% vs 20-bar-ago ATR mean
    2. Average bar range narrower than 1.5× current ATR
    3. Macro trend is weak (|mc| < 0.3)
    """
    atr = df.get("atr", pd.Series(np.ones(len(df)), index=df.index)).values
    high = df.get("High", pd.Series(np.ones(len(df)), index=df.index)).values
    low  = df.get("Low", pd.Series(np.ones(len(df)), index=df.index)).values
    mc   = df.get("mc", pd.Series(np.zeros(len(df)), index=df.index)).values

    # Condition 1: ATR compressed vs 20-bar-ago rolling mean
    atr_ma20 = pd.Series(atr).rolling(20, min_periods=5).mean().shift(20).fillna(
        pd.Series(atr).rolling(20, min_periods=5).mean()).values
    atr_compress = (atr < atr_ma20 * 0.70)

    # Condition 2: narrow range
    range_mean = pd.Series(high - low).rolling(10, min_periods=3).mean().values
    range_narrow = (range_mean / (atr + 1e-10)) < 1.5

    # Condition 3: weak macro
    weak_macro = (np.abs(mc) < 0.3)

    return (atr_compress.astype(int) + range_narrow.astype(int) +
            weak_macro.astype(int)) >= 2