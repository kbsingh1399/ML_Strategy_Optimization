Python# live_model_trainer.py, line ~130, _add_advanced_features()
# Log returns (1/3/5 bar)
for lag in [1, 3, 5]:
    df[f"log_ret_{lag}"] = np.log(df[price_col] / df[price_col].shift(lag)...)
# Rolling skew/kurtosis (10/20 bar windows)
for w in [10, 20]:
    df[f"ret_skew_{w}"] = r.rolling(w).skew()
    df[f"ret_kurt_{w}"] = r.rolling(w).kurt()
# ATR ratio, CVD acceleration (2nd derivative), temporal lags, price/ATR ratio