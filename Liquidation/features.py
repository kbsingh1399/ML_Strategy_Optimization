import polars as pl
import numpy as np
import os

_GDRIVE_PARQUET = r"G:\My Drive\_Trading_Data\15m\parquet"
_LOCAL_PARQUET  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backtesting_data"))
DEFAULT_DIR = _GDRIVE_PARQUET if os.path.exists(_GDRIVE_PARQUET) else _LOCAL_PARQUET

def load_and_merge_data(symbol: str, data_dir: str = DEFAULT_DIR) -> pl.DataFrame:
    """Loads and merges Summary and Footprint Parquet files for a given symbol."""
    sum_path = os.path.join(data_dir, f"Master_{symbol}_15m_Final_Summary.parquet")
    fp_path = os.path.join(data_dir, f"Master_{symbol}_15m_Final_Footprint.parquet")
    
    if not os.path.exists(sum_path) or not os.path.exists(fp_path):
        raise FileNotFoundError(f"Missing files for symbol: {symbol}")
        
    df_sum = pl.read_parquet(sum_path)
    df_fp = pl.read_parquet(fp_path)
    
    # Drop unused columns that may have conflicting types between symbols
    df_sum = df_sum.drop(['Long/Short Ratio (Account)', 'Whale Ind', 'Net Shorts', 'Net Longs'], strict=False)
    
    # Rename overlapping columns to avoid duplicate suffix issues, except key join columns
    df_fp = df_fp.rename({"Candle #": "Candle"})
    df_fp = df_fp.drop(['POC Price', 'Timestamp', 'Price Low', 'Price High', 'time'], strict=False)
    
    # Join on Symbol and Candle
    df = df_sum.join(df_fp, on=['Symbol', 'Candle'], how='inner')
    
    # Parse TimeStamp (naive local) and drop the old time column
    df = df.with_columns(
        pl.col("TimeStamp").str.slice(0, 19).str.to_datetime("%Y-%m-%d %H:%M:%S").alias("datetime")
    ).drop("time")
    
    return df

def compute_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    """Computes all footprint and summary technical/ML features."""
    
    # Fill null values for liquidations and CVD
    df = df.with_columns([
        pl.col("Agg. Liq Long").fill_null(0.0),
        pl.col("Agg. Liq Short").fill_null(0.0),
        pl.col("Volume").fill_null(0.0),
        pl.col("Candle Delta").fill_null(0.0),
        pl.col("Delta USD").fill_null(0.0),
        pl.col("Bid Trades").fill_null(0.0),
        pl.col("Ask Trades").fill_null(0.0),
        pl.col("Bid USD").fill_null(0.0),
        pl.col("Ask USD").fill_null(0.0),
    ])
    
    # 1. Footprint Position / Imbalance features
    df = df.with_columns([
        # POC position: where is POC inside High/Low range
        ((pl.col("POC Price") - pl.col("Low")) / (pl.col("High") - pl.col("Low") + 1e-8)).alias("poc_pos"),
        
        # Delta USD normalized by volume USD (approximated by Volume * Mid Price)
        (pl.col("Delta USD") / (pl.col("Volume") * pl.col("POC Price") + 1e-8)).alias("delta_usd_ratio"),
        
        # Average trade size for bid and ask trades
        (pl.col("Bid USD") / (pl.col("Bid Trades") + 1e-8)).alias("avg_bid_size"),
        (pl.col("Ask USD") / (pl.col("Ask Trades") + 1e-8)).alias("avg_ask_size"),
    ])
    
    df = df.with_columns([
        # Whale size ratio (bids vs asks average size)
        (pl.col("avg_bid_size") / (pl.col("avg_ask_size") + 1e-8)).alias("size_ratio"),
        # Number of trades ratio
        (pl.col("Bid Trades") / (pl.col("Ask Trades") + 1e-8)).alias("trade_ratio"),
    ])
    
    # 2. Rolling Z-scores of liquidations (windows 50 and 200)
    for w in [50, 200]:
        df = df.with_columns([
            ((pl.col("Agg. Liq Long") - pl.col("Agg. Liq Long").rolling_mean(w)) / 
             (pl.col("Agg. Liq Long").rolling_std(w).replace(0, 1e-8))).alias(f"liq_long_z_{w}"),
             
            ((pl.col("Agg. Liq Short") - pl.col("Agg. Liq Short").rolling_mean(w)) / 
             (pl.col("Agg. Liq Short").rolling_std(w).replace(0, 1e-8))).alias(f"liq_short_z_{w}"),
        ])
        
    # 3. CVD rolling z-score (using volume delta)
    for w in [10, 50, 200]:
        df = df.with_columns([
            ((pl.col("CVD") - pl.col("CVD").rolling_mean(w)) / 
             (pl.col("CVD").rolling_std(w).replace(0, 1e-8))).alias(f"cvd_z_{w}")
        ])
        
    # 4. Volatility (Wilder ATR) and macro indicators
    prev_close = pl.col("Close").shift(1)
    tr = pl.max_horizontal([
        pl.col("High") - pl.col("Low"),
        (pl.col("High") - prev_close).abs(),
        (pl.col("Low") - prev_close).abs()
    ]).fill_null(pl.col("High") - pl.col("Low"))
    
    df = df.with_columns(tr.alias("tr"))
    # Wilder EWM smoothing uses alpha = 1 / 14 (equivalent to span = 27)
    df = df.with_columns(pl.col("tr").ewm_mean(span=27, adjust=False).alias("atr"))
    
    df = df.with_columns([
        (pl.col("atr") / pl.col("Close")).alias("atr_ratio"),
        # Close to 200 EMA proxy
        (pl.col("Close") / pl.col("Close").ewm_mean(span=200).replace(0, 1e-8) - 1.0).alias("close_to_ema_200"),
    ])
    
    # Clean all numeric columns of any inf, nan, or null values
    numeric_cols = [c for c, t in df.schema.items() if t.is_numeric()]
    df = df.with_columns([
        pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan() | pl.col(c).is_null())
        .then(0.0)
        .otherwise(pl.col(c))
        .alias(c)
        for c in numeric_cols
    ])
    
    return df

def generate_labels(df: pl.DataFrame, lookahead: int = 16, tp_atr_mult: float = 2.0, sl_atr_mult: float = 1.0) -> pl.DataFrame:
    """
    Generates target labels for ML strategy.
    
    For each trigger row:
    - Long Liquidation Trigger (liq_long_z_200 >= 3.0):
      - We enter LONG (close of trigger bar)
      - SL: Low_t - sl_atr_mult * ATR
      - TP: Close_t + tp_atr_mult * ATR
      - Label: 
        - +1 if price reaches TP before SL (Reversal)
        - -1 if price hits SL before TP (Breakout/Continuation)
        -  0 if holding timeout (neither hit after lookahead bars)
        
    - Short Liquidation Trigger (liq_short_z_200 >= 3.0):
      - We enter SHORT (close of trigger bar)
      - SL: High_t + sl_atr_mult * ATR
      - TP: Close_t - tp_atr_mult * ATR
      - Label:
        - -1 if price reaches TP before SL (Reversal)
        - +1 if price hits SL before TP (Breakout/Continuation)
        -  0 if holding timeout
        
    Returns the dataframe with 'target_label' and 'trigger_type' (1=Long Liq, -1=Short Liq, 0=None).
    """
    # Convert polars to pandas to easily run the forward-looking label logic
    pdf = df.to_pandas()
    n = len(pdf)
    
    target_labels = np.zeros(n, dtype=np.int8)
    trigger_types = np.zeros(n, dtype=np.int8)
    
    close = pdf["Close"].values
    high = pdf["High"].values
    low = pdf["Low"].values
    atr = pdf["atr"].values
    
    # Trigger checks
    liq_long_z = pdf["liq_long_z_200"].values
    liq_short_z = pdf["liq_short_z_200"].values
    
    for i in range(n - lookahead):
        is_long_liq = liq_long_z[i] >= 3.0
        is_short_liq = liq_short_z[i] >= 3.0
        
        if is_long_liq and is_short_liq:
            # Dual liquidation event, skip or take the larger one
            if liq_long_z[i] > liq_short_z[i]:
                is_short_liq = False
            else:
                is_long_liq = False
                
        if is_long_liq:
            trigger_types[i] = 1 # Long Liq Trigger
            entry = close[i]
            sl = low[i] - sl_atr_mult * atr[i]
            tp = entry + tp_atr_mult * atr[i]
            
            # Trace barrier
            label = 0
            for j in range(1, lookahead + 1):
                idx = i + j
                if low[idx] <= sl:
                    label = -1  # Breakout (price continued down through stop)
                    break
                elif high[idx] >= tp:
                    label = 1   # Reversal (price bounced up to TP)
                    break
            # If timeout, set partial return classification
            if label == 0:
                if close[i + lookahead] > entry + 0.5 * atr[i]:
                    label = 1
                elif close[i + lookahead] < entry - 0.5 * atr[i]:
                    label = -1
            target_labels[i] = label
            
        elif is_short_liq:
            trigger_types[i] = -1 # Short Liq Trigger
            entry = close[i]
            sl = high[i] + sl_atr_mult * atr[i]
            tp = entry - tp_atr_mult * atr[i]
            
            # Trace barrier
            label = 0
            for j in range(1, lookahead + 1):
                idx = i + j
                if high[idx] >= sl:
                    label = 1   # Breakout (price continued up through stop)
                    break
                elif low[idx] <= tp:
                    label = -1  # Reversal (price bounced down to TP)
                    break
            # If timeout, set partial return classification
            if label == 0:
                if close[i + lookahead] < entry - 0.5 * atr[i]:
                    label = -1
                elif close[i + lookahead] > entry + 0.5 * atr[i]:
                    label = 1
            target_labels[i] = label
            
    pdf["target_label"] = target_labels
    pdf["trigger_type"] = trigger_types
    return pl.from_pandas(pdf)

if __name__ == "__main__":
    print("Testing data load and feature engineering for BTCUSDT...")
    df = load_and_merge_data("BTCUSDT")
    df = compute_rolling_features(df)
    df = generate_labels(df)
    print("Columns:", df.columns)
    print("Dataset shape:", df.shape)
    trigger_count = df.filter(pl.col("trigger_type") != 0).shape[0]
    print(f"Total trigger events: {trigger_count} ({trigger_count/len(df)*100:.2f}%)")
    revs = df.filter((pl.col("trigger_type") != 0) & (pl.col("target_label") == 1)).shape[0]
    brks = df.filter((pl.col("trigger_type") != 0) & (pl.col("target_label") == -1)).shape[0]
    print(f"Reversals (+1): {revs}, Breakouts (-1): {brks}")
