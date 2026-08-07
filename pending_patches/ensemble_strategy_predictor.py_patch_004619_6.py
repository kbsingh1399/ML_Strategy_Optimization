Python# TARGET: ensemble_strategy_predictor.py
# In each of S1 through S5, ADD after the chop filter + _cvd_ok line:
# 
#   imb_flat = df.get("cvd_imb_flat", pd.Series(0, index=df.index)).values
#   heavy_buy  = df.get("cvd_heavy_buy", pd.Series(0, index=df.index)).values
#   heavy_sell = df.get("cvd_heavy_sell", pd.Series(0, index=df.index)).values
#   imb_ok = _cvd_imbalance_pass(heavy_buy, heavy_sell, imb_flat, mc)
#
# And APPEND  & imb_ok  to both mask_l and mask_s.
# Example for S2 (mask_l line):
# BEFORE: mask_l = (mc > 0) & (p8 < -0.18 * ar) & (cvd_accel > 0) & (~chop) & _cvd_ok(df, 1)
# AFTER:  mask_l = (mc > 0) & (p8 < -0.18 * ar) & (cvd_accel > 0) & (~chop) & _cvd_ok(df, 1) & imb_ok