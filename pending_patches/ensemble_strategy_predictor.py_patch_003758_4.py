Python# TARGET: ensemble_strategy_predictor.py
# ═══════════════════════════════════════════════════════════════════
# PATCH 4 — REPLACE _atr_scale() with symmetric clamped version
# FIND:  def _atr_scale(df: pd.DataFrame) -> np.ndarray:
# REPLACE the entire function:
# ═══════════════════════════════════════════════════════════════════

def _atr_scale(df: pd.DataFrame) -> np.ndarray:
    """Symmetric +/-15% ATR threshold scaling, clamped to [0.85, 1.15].

    Uses atr_ratio: current ATR / 100-period mean ATR.
    atr_ratio=0.70 → scale=0.85 (tighten 15%)
    atr_ratio=1.00 → scale=1.00 (no change)
    atr_ratio=2.00 → scale=1.15 (loosen 15%)

    This replaces the old multiplicative scale which went 0.70–1.40×
    and was asymmetric (30% tighter vs 40% looser).
    """
    if "atr" not in df.columns:
        return np.ones(len(df), dtype=np.float64)
    atr_ma = df["atr"].rolling(100, min_periods=10).mean().values + 1e-10
    atr_ratio = df["atr"].values / atr_ma
    bias = 1.0 + 0.50 * (atr_ratio - 1.0)
    return np.clip(bias, 0.85, 1.15)