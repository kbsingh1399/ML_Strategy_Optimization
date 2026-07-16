import os
import sys
import json
import numpy as np
import pandas as pd

# Add the local directory to sys.path first to override the parent imports
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from unified_backtest import load_asset, prep, MAX_BARS, FEE_RATE, ACCOUNT_SIZE, MAX_DD_LIMIT, ZENO_DENOM, RISK_CAP
from unified_backtest import build_labels_fast
import agent5_exact

TRADING_ASSETS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 
    'DOGEUSDT', 'ADAUSDT', 'TRXUSDT', 'AVAXUSDT', 'DOTUSDT',
    'LINKUSDT', 'LTCUSDT', 'NEARUSDT', 'SUIUSDT'
]

CONFIG_DIR = os.path.join(_DIR, 'agent5_configs')

def simulate_window(start_date_str, end_date_str, btc_ref, all_trades_ever, risk_multiplier=1.0):
    print(f"\n=========================================")
    print(f"=== OOS WINDOW: {start_date_str} to {end_date_str} ===")
    print(f"=========================================")
    
    print(f"--> Running Agent 5 Smart Optimizer for {start_date_str} to {end_date_str}...")
    agent5_exact.run_all(start_date_str, end_date_str, n_trials=30)
    
    target_date = pd.to_datetime(start_date_str)
    test_end = pd.to_datetime(end_date_str)
    
    month_trades = []
    
    # 2. Execute trades using optimal config
    for sym in TRADING_ASSETS:
        cfg_path = os.path.join(CONFIG_DIR, f"{sym}.json")
        if not os.path.exists(cfg_path):
            continue
            
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
            
        if cfg.get('score', -99999) <= 0:
            print(f"[{sym}] Skipping. Optimizer found no profitable edge.")
            continue
            
        print(f"[{sym}] Simulating trades. Model: {cfg['ml_model']} | Window: {cfg['window_size']}m")
        
        # Load and prep data
        df = load_asset(sym)
        if df.empty: continue
        
        df, feats = prep(df.copy(), btc_ref)
        df = df.reset_index()
        if 'index' in df.columns and 'ts' not in df.columns:
            df = df.rename(columns={'index': 'ts'})
        elif 'Date' in df.columns and 'ts' not in df.columns:
            df = df.rename(columns={'Date': 'ts'})
            
        # Build Labels
        sl_mult = cfg['sl_mult']
        tp_mult = cfg['tp_mult']
        df['lab_long'] = build_labels_fast(df, 1, tp_mult, sl_mult)
        df['lab_short'] = build_labels_fast(df, -1, tp_mult, sl_mult)
        
        # Time Filters
        train_end = target_date
        train_start = train_end - pd.DateOffset(months=cfg['window_size'])
        
        train_mask = (df['ts'] >= train_start.strftime('%Y-%m-%d')) & (df['ts'] < train_end.strftime('%Y-%m-%d'))
        test_mask = (df['ts'] >= target_date.strftime('%Y-%m-%d')) & (df['ts'] < test_end.strftime('%Y-%m-%d'))
        
        X_tr = df.loc[train_mask, feats]
        y_tr_long = df.loc[train_mask, 'lab_long']
        y_tr_short = df.loc[train_mask, 'lab_short']
        X_te = df.loc[test_mask, feats]
        
        if len(X_tr) < 100 or len(X_te) < 10:
            print(f"[{sym}] Insufficient data for {start_date_str} to {end_date_str}.")
            continue
            
        # Train ML Model
        ml_model = cfg['ml_model']
        if ml_model == 'lightgbm':
            import lightgbm as lgb
            lgb_params = {
                'objective': 'binary', 'metric': 'binary_logloss',
                'learning_rate': 0.05, 'num_leaves': 16,
                'verbose': -1, 'n_jobs': 1
            }
            # Try GPU acceleration, fallback to CPU
            try:
                lgb_params['device_type'] = 'gpu'
                ml = lgb.train(lgb_params, lgb.Dataset(X_tr, y_tr_long), 100)
                ms = lgb.train(lgb_params, lgb.Dataset(X_tr, y_tr_short), 100)
            except Exception:
                lgb_params['device_type'] = 'cpu'
                ml = lgb.train(lgb_params, lgb.Dataset(X_tr, y_tr_long), 100)
                ms = lgb.train(lgb_params, lgb.Dataset(X_tr, y_tr_short), 100)
            p_long = ml.predict(X_te)
            p_short = ms.predict(X_te)
            
        elif ml_model == 'xgboost':
            import xgboost as xgb
            # Try GPU acceleration, fallback to CPU
            try:
                ml = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, eval_metric='logloss', tree_method='hist', device='cuda')
                ms = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, eval_metric='logloss', tree_method='hist', device='cuda')
                ml.fit(X_tr, y_tr_long)
                ms.fit(X_tr, y_tr_short)
            except Exception:
                ml = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, eval_metric='logloss', tree_method='hist', device='cpu')
                ms = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, eval_metric='logloss', tree_method='hist', device='cpu')
                ml.fit(X_tr, y_tr_long)
                ms.fit(X_tr, y_tr_short)
            p_long = ml.predict_proba(X_te)[:, 1]
            p_short = ms.predict_proba(X_te)[:, 1]
            
        # Simulate testing phase
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
        ts_vals = test_df['ts'].values
        
        conf = cfg['confidence']
        trail_act = cfg['trail_act']
        
        equity = ACCOUNT_SIZE
        peak_equity = ACCOUNT_SIZE
        
        i = 0
        n = len(test_df)
        while i < n - MAX_BARS - 1:
            d = 0
            if pl[i] > conf and pl[i] > ps[i] and mac[i] == 1: d = 1
            elif ps[i] > conf and ps[i] > pl[i] and mac[i] == -1: d = -1
            
            if d == 0: i += 1; continue
                
            atr = a[i]
            if np.isnan(atr) or atr <= 0: i += 1; continue
                
            # Use dynamic risk calculation based on remaining drawdown allowance
            remaining = MAX_DD_LIMIT - (peak_equity - equity)
            if remaining <= 5: i += 1; continue
            risk = min(remaining / ZENO_DENOM, RISK_CAP) * risk_multiplier
            
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
            
            trade = {
                'entry_ts': str(ts_vals[i]),
                'exit_ts': str(ts_vals[exit_idx]),
                'sym': sym,
                'dir': d,
                'pnl': float(pnl)
            }
            month_trades.append(trade)
            i = exit_idx + 1
            
    all_trades_ever.extend(month_trades)
    
    if month_trades:
        m_pnl = sum(t['pnl'] for t in month_trades)
        print(f"--> Window {start_date_str} to {end_date_str} completed! PnL: ${m_pnl:.2f} | Trades: {len(month_trades)}")
    else:
        print(f"--> Window {start_date_str} to {end_date_str} completed! No trades executed.")
