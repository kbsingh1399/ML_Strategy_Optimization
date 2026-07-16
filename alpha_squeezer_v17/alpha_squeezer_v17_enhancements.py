"""
Alpha Squeezer v17 — Feature & Regime Enhancement Snippets
============================================================
Drop-in pandas feature functions referencing the exact Summary/Footprint
columns described in the strategy spec. These are meant to be merged into
your existing feature pipeline (joined on timestamp across the two parquet
files) prior to LightGBM training.

Assumes a single merged DataFrame `df` per asset, sorted by timestamp,
containing all Summary + Footprint columns, plus a precomputed `ATR` column.
"""

import numpy as np
import pandas as pd


def zscore(series: pd.Series, window: int) -> pd.Series:
    m = series.rolling(window).mean()
    s = series.rolling(window).std(ddof=0)
    return (series - m) / s.replace(0, np.nan)


# ---------------------------------------------------------------------------
# 1A. Institutional Absorption Features
# ---------------------------------------------------------------------------
def add_absorption_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Liquidation volume absorbed relative to aggressive sell flow
    df["abs_ratio_long"] = (
        df["Agg. Liq Long"].rolling(5).sum()
        / (df["Ask USD"].rolling(5).sum() + 1e-9)
    )
    df["abs_ratio_short"] = (
        df["Agg. Liq Short"].rolling(5).sum()
        / (df["Bid USD"].rolling(5).sum() + 1e-9)
    )

    # Delta vs. Liquidation divergence (buyers net aggressive despite long liqs)
    df["liq_long_3bar"] = df["Agg. Liq Long"].rolling(3).sum()
    df["liq_long_pct95"] = df["liq_long_3bar"].rolling(200).quantile(0.95)
    df["delta_3bar"] = df["Candle Delta"].rolling(3).sum()
    df["absorption_flag_long"] = (
        (df["liq_long_3bar"] >= df["liq_long_pct95"]) & (df["delta_3bar"] > 0)
    ).astype(int)

    df["liq_short_3bar"] = df["Agg. Liq Short"].rolling(3).sum()
    df["liq_short_pct95"] = df["liq_short_3bar"].rolling(200).quantile(0.95)
    df["absorption_flag_short"] = (
        (df["liq_short_3bar"] >= df["liq_short_pct95"]) & (df["delta_3bar"] < 0)
    ).astype(int)

    # Whale-confirmed absorption
    df["whale_z"] = zscore(df["Whale Ind"], 20)

    # OI stabilization after liquidation spike
    df["oi_diff_std_3"] = df["Agg. OI"].diff().rolling(3).std()

    # POC anchoring / stability despite wick extremity
    df["poc_stability"] = df["POC Price"].rolling(6).std() / df["ATR"].replace(0, np.nan)

    return df


# ---------------------------------------------------------------------------
# 1B. Orderbook / Volume-Profile Depth Imbalance Features
# ---------------------------------------------------------------------------
def add_depth_imbalance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["depth_imbalance"] = (df["Bid USD"] - df["Ask USD"]) / (
        df["Bid USD"] + df["Ask USD"] + 1e-9
    )
    df["depth_imbalance_z"] = zscore(df["depth_imbalance"], 20)

    # Average trade size on each side (block trades vs. broad retail)
    df["avg_trade_size_bid"] = df["Bid USD"] / df["Bid Trades"].replace(0, np.nan)
    df["avg_trade_size_ask"] = df["Ask USD"] / df["Ask Trades"].replace(0, np.nan)
    df["avg_trade_size_bid_z"] = zscore(df["avg_trade_size_bid"], 50)
    df["avg_trade_size_ask_z"] = zscore(df["avg_trade_size_ask"], 50)

    # POC offset & positional location within the bar's range
    df["poc_offset_atr"] = (df["Close"] - df["POC Price"]) / df["ATR"].replace(0, np.nan)
    df["poc_location"] = (df["POC Price"] - df["Price Low"]) / (
        (df["Price High"] - df["Price Low"]).replace(0, np.nan)
    )

    # Micro (footprint) vs Macro (summary) CVD divergence
    df["footprint_cvd"] = df["Delta USD"].cumsum()
    df["cvd_divergence"] = zscore(df["footprint_cvd"], 50) - zscore(df["CVD"], 50)

    return df


# ---------------------------------------------------------------------------
# 1C. Velocity / Acceleration Features
# ---------------------------------------------------------------------------
def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["trade_count_total"] = df["Bid Trades"] + df["Ask Trades"]
    df["trade_velocity"] = df["trade_count_total"].diff()
    df["trade_accel"] = df["trade_velocity"].diff()

    df["delta_intensity"] = df["Delta Qty"] / df["total_qty"].replace(0, np.nan)
    df["delta_intensity_roc"] = df["delta_intensity"].diff(3)

    df["funding_accel"] = df["Agg. Funding Rate"].diff(4)
    df["oi_velocity"] = df["Agg. OI"].pct_change(4)

    df["liq_velocity_long"] = df["Agg. Liq Long"].diff()
    df["liq_velocity_short"] = df["Agg. Liq Short"].diff()

    return df


# ---------------------------------------------------------------------------
# 2. Market Regime Filter
# ---------------------------------------------------------------------------
def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Directional efficiency ratio (Kaufman-style, needs only Close)
    net_move = df["Close"].diff(20).abs()
    path_sum = df["Close"].diff().abs().rolling(20).sum()
    df["efficiency_ratio"] = net_move / path_sum.replace(0, np.nan)

    # OI trend / velocity, z-scored
    df["oi_trend_z"] = zscore(df["Agg. OI"].diff(16), 96)

    # Funding persistence
    df["funding_z"] = zscore(df["Agg. Funding Rate"], 480)
    df["funding_persistence"] = (
        (df["funding_z"].abs() > 1).rolling(20).mean()
    )

    # Footprint range expansion
    bar_range = df["Price High"] - df["Price Low"]
    df["range_ratio"] = bar_range / bar_range.rolling(50).mean()

    # Volume profile volatility (coefficient of variation)
    total_usd = df["Bid USD"] + df["Ask USD"]
    df["profile_vol"] = (
        total_usd.rolling(20).std() / total_usd.rolling(20).mean().replace(0, np.nan)
    )

    # POC migration speed
    df["poc_migration"] = (
        df["POC Price"].diff().abs().rolling(10).sum() / df["ATR"].replace(0, np.nan)
    )

    # Liquidation clustering vs isolated spike
    liq_total = df["Agg. Liq Long"] + df["Agg. Liq Short"]
    df["liq_cluster"] = (liq_total > 0).rolling(8).sum()

    # Composite regime score (higher = more range/exhaustion-friendly)
    df["regime_score"] = (
        -zscore(df["efficiency_ratio"], 96).fillna(0)
        - zscore(df["range_ratio"], 96).fillna(0)
        - zscore(df["poc_migration"], 96).fillna(0)
        - zscore(df["liq_cluster"], 96).fillna(0)
        + zscore(df["funding_z"].abs().clip(upper=3), 96).fillna(0)
    )

    # Example 3-state regime classification (thresholds must be
    # calibrated via walk-forward search against Calmar / max drawdown)
    df["regime_state"] = pd.cut(
        df["regime_score"],
        bins=[-np.inf, -0.5, 0.5, np.inf],
        labels=["trend_expansion", "transitional", "range_exhaustion"],
    )

    return df


# ---------------------------------------------------------------------------
# 4. Footprint-based SL/TP helpers
# ---------------------------------------------------------------------------
def compute_footprint_stop_long(df: pd.DataFrame, entry_idx: int, buffer_bps: float = 0.0005) -> float:
    """Stop just below the sweep low of the signal bar, with a small buffer."""
    sweep_low = df["Price Low"].iloc[entry_idx]
    buffer = max(sweep_low * buffer_bps, 0.0)
    return sweep_low - buffer


def compute_footprint_stop_short(df: pd.DataFrame, entry_idx: int, buffer_bps: float = 0.0005) -> float:
    sweep_high = df["Price High"].iloc[entry_idx]
    buffer = max(sweep_high * buffer_bps, 0.0)
    return sweep_high + buffer


def nearest_hvn_target(df: pd.DataFrame, entry_idx: int, direction: str,
                        lookback: int = 96, n_bins: int = 50):
    """
    Approximate the nearest High-Volume Node above (long) or below (short)
    the entry price by building a volume-at-price histogram from recent
    footprint bars (Price Low/Price High/total_qty), then locating the
    nearest strong node beyond entry price in the trade direction.
    """
    window = df.iloc[max(0, entry_idx - lookback):entry_idx]
    if window.empty:
        return None

    lo, hi = window["Price Low"].min(), window["Price High"].max()
    if hi <= lo:
        return None

    bins = np.linspace(lo, hi, n_bins + 1)
    vol_at_price = np.zeros(n_bins)

    for _, row in window.iterrows():
        b_lo, b_hi, qty = row["Price Low"], row["Price High"], row["total_qty"]
        if b_hi <= b_lo or qty <= 0:
            continue
        lo_idx = np.searchsorted(bins, b_lo, side="right") - 1
        hi_idx = np.searchsorted(bins, b_hi, side="right") - 1
        lo_idx = np.clip(lo_idx, 0, n_bins - 1)
        hi_idx = np.clip(hi_idx, 0, n_bins - 1)
        span = max(hi_idx - lo_idx + 1, 1)
        vol_at_price[lo_idx:hi_idx + 1] += qty / span

    entry_price = df["Close"].iloc[entry_idx]
    bin_centers = (bins[:-1] + bins[1:]) / 2
    mean_vol = vol_at_price.mean()
    hvn_mask = vol_at_price > mean_vol * 1.5  # simple HVN threshold

    if direction == "long":
        candidates = bin_centers[(bin_centers > entry_price) & hvn_mask]
        return candidates.min() if len(candidates) else None
    else:
        candidates = bin_centers[(bin_centers < entry_price) & hvn_mask]
        return candidates.max() if len(candidates) else None


if __name__ == "__main__":
    print("Enhanced features script successfully saved.")
