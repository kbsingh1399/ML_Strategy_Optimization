Python# TARGET: ensemble_strategy_predictor.py
# ADD after _oi_cvd_confluence(), before signal_s1:

def _cvd_imbalance_pass(heavy_buy, heavy_sell, imb_flat, mc):
    """CVD imbalance directional check.
    True = order-book conviction agrees with macro direction.
    Blocks: flat imbalance (no conviction), or heavy buys into shorts,
            heavy sells into longs.
    """
    ok = np.ones(len(mc), dtype=bool)
    ok = ok & (imb_flat == 0)                         # need conviction
    ok = ok & ~((mc > 0) & (heavy_sell > 0))          # don't long into sells
    ok = ok & ~((mc < 0) & (heavy_buy > 0))            # don't short into buys
    return ok