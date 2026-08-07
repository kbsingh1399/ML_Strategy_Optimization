Python# TARGET: ensemble_strategy_predictor.py
# ═══════════════════════════════════════════════════════════════════
# PATCH 3 — IMPROVE _cvd_ok() to use divergence flags from featurize()
# FIND:  def _cvd_ok(df: pd.DataFrame, direction: int) -> np.ndarray:
# REPLACE the entire function:
# ═══════════════════════════════════════════════════════════════════

def _cvd_ok(df: pd.DataFrame, direction: int) -> np.ndarray:
    """CVD confluence: 5-bar delta must agree with direction,
    AND no bearish divergence for longs / no bullish for shorts.
    """
    if "CVD" not in df.columns:
        return np.ones(len(df), dtype=bool)

    cvd_d5 = df["CVD"].diff(5).fillna(0).values
    cvdb = df.get("cvd_div_bear", pd.Series(0, index=df.index)).values
    cvdu = df.get("cvd_div_bull", pd.Series(0, index=df.index)).values

    if direction == 1:
        return (cvd_d5 > 0) & (cvdb == 0)
    else:
        return (cvd_d5 < 0) & (cvdu == 0)