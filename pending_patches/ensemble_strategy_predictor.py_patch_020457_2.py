Python# ensemble_strategy_predictor.py, ~line in chunk 2
def _oi_cvd_confluence(oi_rising, cvd_d):
    """True when OI and CVD momentum agree.
    Blocks: OI rising + CVD falling = passive positioning, not real demand."""
    return ~((oi_rising > 0) & (cvd_d < 0))