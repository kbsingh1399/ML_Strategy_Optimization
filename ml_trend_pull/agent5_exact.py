import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import optuna
import joblib
import warnings
from dateutil.relativedelta import relativedelta

warnings.filterwarnings('ignore')

# Add the local directory to sys.path first to override the parent imports
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from unified_backtest import load_asset, prep, NUM_LEAVES, MAX_BARS, ACCOUNT_SIZE, MAX_DD_LIMIT, ZENO_DENOM, RISK_CAP, FEE_RATE
from unified_backtest import build_labels_fast

CONFIG_DIR = os.path.join(_DIR, "agent5_configs")
os.makedirs(CONFIG_DIR, exist_ok=True)

ASSETS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT',
    'TRXUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'NEARUSDT', 'SUIUSDT'
]

def objective(trial, df, feats, test_start, test_end):
    # 1. Hyperparameters
    window_size = trial.suggest_int("window_size", 1, 12)
    ml_model = trial.suggest_categorical("ml_model", ["lightgbm", "xgboost"])
    
    sl_mult = trial.suggest_float("sl_mult", 0.5, 2.0)
    rr_ratio = trial.suggest_float("rr_ratio", 5.0, 8.0)
    tp_mult = sl_mult * rr_ratio
    conf = trial.suggest_float("confidence", 0.50, 0.65)
    trail_act = trial.suggest_float("trail_act", 1.5, 3.5)
    
    # 2. Time boundaries with dynamic fallback for early OOS windows
    df_min_date = pd.to_datetime(df['ts'].min())
    available_history = test_start - df_min_date
    if available_history < pd.Timedelta(days=60):
        train_start = df_min_date
        train_end = test_start - pd.Timedelta(days=5)
        val_start = train_end
        val_end = test_start
    else:
        duration = test_end - test_start
        val_end = test_start
        val_start = val_end - duration
        train_end = val_start
        train_start = train_end - relativedelta(months=window_size)
    
    # Slice the dataframe to contain ONLY the train + val range (+ 2 days buffer)
    # to avoid expensive label building on the entire 225,000 row dataframe
    t_start_str = train_start.strftime('%Y-%m-%d')
    t_end_str = (max(train_end, val_end) + pd.Timedelta(days=2)).strftime('%Y-%m-%d')
    df_slice = df[(df['ts'] >= t_start_str) & (df['ts'] < t_end_str)].copy()
    
    if len(df_slice) < 50:
        return -99999.0
        
    # 3. Build labels on the sliced dataframe
    df_slice['lab_long'] = build_labels_fast(df_slice, 1, tp_mult, sl_mult)
    df_slice['lab_short'] = build_labels_fast(df_slice, -1, tp_mult, sl_mult)
    
    train_mask = (df_slice['ts'] >= train_start.strftime('%Y-%m-%d')) & (df_slice['ts'] < train_end.strftime('%Y-%m-%d'))
    test_mask = (df_slice['ts'] >= val_start.strftime('%Y-%m-%d')) & (df_slice['ts'] < val_end.strftime('%Y-%m-%d'))
    
    X_tr = df_slice.loc[train_mask, feats]
    y_tr_long = df_slice.loc[train_mask, 'lab_long']
    y_tr_short = df_slice.loc[train_mask, 'lab_short']
    X_te = df_slice.loc[test_mask, feats]
    
    if len(X_tr) < 30 or len(X_te) < 15:
        return -99999.0
        
    # 4. Train Model (Forced CPU for fast Optuna trials)
    if ml_model == "lightgbm":
        import lightgbm as lgb
        lgb_p = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': 0.05, 'num_leaves': NUM_LEAVES, 'verbose': -1, 'n_jobs': 1, 'device_type': 'cpu'}
        ml = lgb.train(lgb_p, lgb.Dataset(X_tr, y_tr_long), 40)
        ms = lgb.train(lgb_p, lgb.Dataset(X_tr, y_tr_short), 40)
        p_long = ml.predict(X_te)
        p_short = ms.predict(X_te)
    elif ml_model == "xgboost":
        import xgboost as xgb
        ml = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.05, max_depth=4, n_estimators=40, eval_metric='logloss', tree_method='hist', device='cpu')
        ms = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.05, max_depth=4, n_estimators=40, eval_metric='logloss', tree_method='hist', device='cpu')
        ml.fit(X_tr, y_tr_long)
        ms.fit(X_tr, y_tr_short)
        p_long = ml.predict_proba(X_te)[:, 1]
        p_short = ms.predict_proba(X_te)[:, 1]
    
    # 5. Simulate
    test_df = df_slice.loc[test_mask].copy()
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
        trades.append(pnl)
        i = exit_idx + 1
        
    net_pnl = equity - ACCOUNT_SIZE
    max_dd = peak_equity - equity
    
    # 6. Objective: Maximize Return * Calmar Ratio
    if len(trades) < 3:
        return -99999.0
        
    safe_dd = max(max_dd, 0.001)  # Floor DD to avoid division by zero
    calmar_proxy = net_pnl / safe_dd
    
    # Reward high PnL and high Calmar, strongly penalize negative PnL
    if net_pnl > 0:
        score = net_pnl * calmar_proxy
    else:
        score = net_pnl * 100.0  # Heavy penalty for losses
        
    return score

def optimize_asset(symbol, start_date_str, end_date_str, n_trials=30):
    print(f"[{symbol}] Starting Agent 5 Optimization for {start_date_str} to {end_date_str}...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    try:
        df = load_asset(symbol)
        btc = load_asset('BTCUSDT')
        if df.empty or btc.empty:
            return symbol, False
            
        btc_ref = btc[['Close', 'CVD']].copy()
        btc_ref.columns = ['btc_Close', 'btc_CVD']
        df, feats = prep(df.copy(), btc_ref)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        df = df.reset_index()
        if 'index' in df.columns and 'ts' not in df.columns:
            df = df.rename(columns={'index': 'ts'})
        elif 'Date' in df.columns and 'ts' not in df.columns:
            df = df.rename(columns={'Date': 'ts'})
            
        test_start = pd.to_datetime(start_date_str)
        test_end = pd.to_datetime(end_date_str)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, df, feats, test_start, test_end), n_trials=n_trials, n_jobs=1)
        
        best_params = study.best_params.copy()
        best_params['tp_mult'] = best_params['sl_mult'] * best_params.pop('rr_ratio')
        best_params['target_month'] = f"{start_date_str}_to_{end_date_str}"
        best_params['score'] = study.best_value
        
        cfg_path = os.path.join(CONFIG_DIR, f"{symbol}.json")
        with open(cfg_path, 'w') as f:
            json.dump(best_params, f, indent=4)
            
        print(f"[{symbol}] Done. Window: {best_params['window_size']}m | Model: {best_params['ml_model']} | Score: {best_params['score']:.2f}")
        return symbol, True
    except Exception as e:
        print(f"[{symbol}] ERROR: {e}")
        return symbol, False

def run_all(start_date_str, end_date_str, n_trials=30):
    print(f"--- AGENT 5 SMART OPTIMIZER ---")
    print(f"Target Window: {start_date_str} to {end_date_str}")
    print(f"Assets: {len(ASSETS)}")
    print(f"Running sequentially...")
    
    results = []
    for sym in ASSETS:
        results.append(optimize_asset(sym, start_date_str, end_date_str, n_trials))
    
    success = sum(1 for sym, ok in results if ok)
    print(f"\nOptimization complete. {success}/{len(ASSETS)} assets successfully generated smart configs.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=str, required=True, help='YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, required=True, help='YYYY-MM-DD format')
    parser.add_argument('--n_trials', type=int, default=30, help='Number of trials for Optuna')
    args = parser.parse_args()
    
    run_all(args.start_date, args.end_date, n_trials=args.n_trials)
