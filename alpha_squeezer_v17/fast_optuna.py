"""
fast_optuna.py
Fast hyperparameter optimization using a single Train/Test split.
This allows Optuna to optimize TP_MULT and SL_MULT (which require re-training the ML model)
without taking hours per asset.
Train: 2024-01-01 to 2024-12-31
Test: 2025-01-01 to 2026-07-01
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import warnings

warnings.filterwarnings('ignore')

_DIR = os.path.dirname(os.path.abspath(__file__))
_STRAT_DIR = os.path.dirname(_DIR)
_CORE_DIR = os.path.join(_STRAT_DIR, 'Core')
sys.path.insert(0, _STRAT_DIR)
sys.path.insert(0, _CORE_DIR)

from unified_backtest import load_asset, prep, NUM_LEAVES, MAX_BARS, ACCOUNT_SIZE, MAX_DD_LIMIT, ZENO_DENOM, RISK_CAP, FEE_RATE
import numba

CONFIG_DIR = os.path.join(_DIR, "configs")
os.makedirs(CONFIG_DIR, exist_ok=True)

@numba.njit
def _build_labels_numba(c, h, l, a, direction, tp_mult, sl_mult, max_bars):
    n = len(c)
    labels = np.zeros(n)
    for i in range(n - max_bars - 1):
        atr_val = a[i]
        if np.isnan(atr_val) or atr_val == 0:
            continue
        entry = c[i]
        sl = entry - sl_mult * atr_val if direction == 1 else entry + sl_mult * atr_val
        tp = entry + tp_mult * atr_val if direction == 1 else entry - tp_mult * atr_val
        
        limit = min(i + max_bars + 1, n)
        for j in range(i + 1, limit):
            if direction == 1:
                if l[j] <= sl:
                    break
                if h[j] >= tp:
                    labels[i] = 1
                    break
            else:
                if h[j] >= sl:
                    break
                if l[j] <= tp:
                    labels[i] = 1
                    break
    return labels

def build_labels_fast(df, direction, tp_mult, sl_mult):
    c = df['Close'].values
    h = df['High'].values
    l = df['Low'].values
    a = df['atr'].values
    return _build_labels_numba(c, h, l, a, direction, tp_mult, sl_mult, MAX_BARS)

def objective(trial, df, feats):
    # Parameters
    sl_mult = trial.suggest_float("sl_mult", 0.5, 1.5)
    rr_ratio = trial.suggest_float("rr_ratio", 2.5, 5.0)
    tp_mult = sl_mult * rr_ratio
    conf = trial.suggest_float("confidence", 0.51, 0.65)
    trail_act = trial.suggest_float("trail_act", 1.5, 3.5)
    
    # 1. Labels
    df['lab_long'] = build_labels_fast(df, 1, tp_mult, sl_mult)
    df['lab_short'] = build_labels_fast(df, -1, tp_mult, sl_mult)
    
    # 2. Train/Test Split
    train_mask = (df['ts'] >= '2024-01-01') & (df['ts'] < '2025-01-01')
    test_mask = (df['ts'] >= '2025-01-01')
    
    X_tr = df.loc[train_mask, feats]
    y_tr_long = df.loc[train_mask, 'lab_long']
    y_tr_short = df.loc[train_mask, 'lab_short']
    X_te = df.loc[test_mask, feats]
    
    if len(X_tr) < 100 or len(X_te) < 100:
        return -9999
        
    # 3. Train
    lgb_p = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'learning_rate': 0.05, 'num_leaves': NUM_LEAVES, 'verbose': -1
    }
    ml = lgb.train(lgb_p, lgb.Dataset(X_tr, y_tr_long), 100)
    ms = lgb.train(lgb_p, lgb.Dataset(X_tr, y_tr_short), 100)
    
    # 4. Predict
    p_long = ml.predict(X_te)
    p_short = ms.predict(X_te)
    
    # 5. Simulate
    test_df = df.loc[test_mask].copy()
    test_df['p_long'] = p_long
    test_df['p_short'] = p_short
    
    c = test_df['Close'].values
    h = test_df['High'].values
    l = test_df['Low'].values
    a = test_df['atr'].values
    mac = test_df['macro'].values
    pl = test_df['p_long'].values
    ps = test_df['p_short'].values
    
    equity = ACCOUNT_SIZE
    peak_equity = ACCOUNT_SIZE
    trades = []
    
    i = 0
    n = len(test_df)
    while i < n - MAX_BARS - 1:
        d = 0
        if pl[i] > conf and pl[i] > ps[i] and mac[i] == 1: d = 1
        elif ps[i] > conf and ps[i] > pl[i] and mac[i] == -1: d = -1
        
        if d == 0: i += 1; continue
        atr = a[i]
        if np.isnan(atr) or atr <= 0: i += 1; continue
        
        remaining = MAX_DD_LIMIT - (peak_equity - equity)
        if remaining <= 5: i += 1; continue
            
        risk = min(remaining / ZENO_DENOM, RISK_CAP)
        
        entry = c[i]
        sl = entry - sl_mult * atr if d == 1 else entry + sl_mult * atr
        tp = entry + tp_mult * atr if d == 1 else entry - tp_mult * atr
        exit_p = c[i + MAX_BARS]
        exit_idx = i + MAX_BARS
        trail_buf = 0.5
        
        for j in range(i + 1, i + MAX_BARS + 1):
            if d == 1:
                cur_r = (h[j] - entry) / (atr * sl_mult)
                if cur_r >= trail_act:
                    ns = l[j] - trail_buf * a[j]
                    if ns > sl: sl = ns
                if l[j] <= sl: exit_p = sl; exit_idx = j; break
                if h[j] >= tp: exit_p = tp; exit_idx = j; break
            else:
                cur_r = (entry - l[j]) / (atr * sl_mult)
                if cur_r >= trail_act:
                    ns = h[j] + trail_buf * a[j]
                    if ns < sl: sl = ns
                if h[j] >= sl: exit_p = sl; exit_idx = j; break
                if l[j] <= tp: exit_p = tp; exit_idx = j; break
                
        # Fee-aware risk sizing and narrow stop filter (max 100x leverage / 10% fee limit)
        stop_pct = (atr * sl_mult) / entry
        if stop_pct < 0.0010:
            i += 1
            continue
            
        roundtrip_fee_pct = 2 * entry * FEE_RATE / (atr * sl_mult)
        raw_risk = risk / (1 + roundtrip_fee_pct)
        qty = raw_risk / (atr * sl_mult)
        raw_pnl = (exit_p - entry) * d * qty
        pnl = raw_pnl - ((qty * entry + qty * abs(exit_p)) * FEE_RATE)
        
        equity += pnl
        peak_equity = max(peak_equity, equity)
        trades.append({'pnl': pnl, 'ts': test_df['ts'].iloc[i]})
        i = exit_idx + 1
        
    net_pnl = equity - ACCOUNT_SIZE
    max_dd = peak_equity - equity
    
    if len(trades) < 20: return -9999
    
    tdf = pd.DataFrame(trades)
    tdf['month'] = pd.to_datetime(tdf['ts']).dt.to_period('M')
    mdf = tdf.groupby('month')['pnl'].sum().reset_index()
    losing_months = len(mdf[mdf['pnl'] < 0])
    wr = len(tdf[tdf['pnl'] > 0]) / len(tdf)
    
    score = net_pnl
    if max_dd > MAX_DD_LIMIT: score -= 5000
    score -= (losing_months * 1000)
    
    return score

def run(symbol):
    print(f"Starting Fast Optuna for {symbol}...")
    df = load_asset(symbol)
    btc = load_asset('BTCUSDT')
    if len(df) == 0 or len(btc) == 0:
        print("Data load failed.")
        return
        
    btc_ref = btc[['Close', 'CVD']].copy()
    btc_ref.columns = ['btc_Close', 'btc_CVD']
    df, feats = prep(df.copy(), btc_ref)
    df = df[df.index > '2023-01-01'].reset_index()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, df, feats), n_trials=50, show_progress_bar=True)
    
    best_params = study.best_params.copy()
    best_params['tp_mult'] = best_params['sl_mult'] * best_params.pop('rr_ratio')
    print("Best params mapped:", best_params)
    print("Best score:", study.best_value)
    
    cfg_path = os.path.join(CONFIG_DIR, f"{symbol}.json")
    with open(cfg_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"Saved to {cfg_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('symbol')
    run(parser.parse_args().symbol)
