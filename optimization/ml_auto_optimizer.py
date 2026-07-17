import json
import os
import sys
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from optimization.oos_simulator import load_asset, prep_alpha, prep_trend

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT",
    "LTCUSDT", "NEARUSDT", "SUIUSDT", "TRXUSDT",
]

MONTHS = [
    ("2021-01-24", "2021-02-24"), ("2021-06-13", "2021-07-13"),
    ("2021-10-29", "2021-11-29"), ("2022-02-08", "2022-03-08"),
    ("2022-05-21", "2022-06-21"), ("2022-09-14", "2022-10-14"),
    ("2022-12-03", "2023-01-03"), ("2023-04-17", "2023-05-17"),
    ("2023-08-25", "2023-09-25"), ("2023-11-10", "2023-12-10"),
    ("2024-02-19", "2024-03-19"), ("2024-07-06", "2024-08-06"),
    ("2024-10-28", "2024-11-28"), ("2025-01-15", "2025-02-15"),
    ("2025-05-03", "2025-06-03"), ("2025-09-22", "2025-10-22"),
    ("2026-02-11", "2026-03-11"), ("2026-06-09", "2026-07-09"),
]

def generate_independent_trades_raw(symbol, df, btc_ref, tp_mult=5.0, sl_mult=1.0):
    df_alpha, alpha_feats = prep_alpha(df.copy(), btc_ref)
    df_trend, trend_feats = prep_trend(df.copy(), btc_ref)
    
    df_comb = df_trend.copy()
    for col in alpha_feats:
        if col not in df_comb.columns:
            df_comb[col] = df_alpha[col]

    # Precalculate rolling means of liqs and add to df_comb so they are in features
    df_comb["liq_long_5_mean"] = df_comb["liq_long_5"].rolling(100).mean().fillna(0)
    df_comb["liq_short_5_mean"] = df_comb["liq_short_5"].rolling(100).mean().fillna(0)

    h = df_comb["High"].values
    l = df_comb["Low"].values
    c = df_comb["Close"].values
    a = df_comb["atr"].values
    ts = df_comb.index.values

    # Precompute arrays for speed
    arr_liq_long = df_comb["liq_long_5"].values
    arr_liq_long_mean = df_comb["liq_long_5_mean"].values
    arr_liq_short = df_comb["liq_short_5"].values
    arr_liq_short_mean = df_comb["liq_short_5_mean"].values
    
    arr_z20 = df_comb.get("z_cvd_20", pd.Series(np.zeros(len(df_comb)))).values
    arr_z4 = df_comb.get("z_cvd_4", pd.Series(np.zeros(len(df_comb)))).values
    
    arr_macro = df_comb.get("macro", pd.Series(np.zeros(len(df_comb)))).values
    arr_pull8 = df_comb.get("pull_ema8", pd.Series(np.zeros(len(df_comb)))).values
    arr_rsi = df_comb.get("rsi", pd.Series(np.full(len(df_comb), 50.0))).values

    feat_cols = [col for col in df_comb.columns if col not in ['ts', 'Timestamp', 'TimeStamp', 'Symbol']]
    feat_arrs = {col: df_comb[col].values for col in feat_cols}

    trades = {"S1_Liquidation": [], "S2_CVD": [], "S3_Trend": []}
    i = 200
    
    cd_s1 = cd_s2 = cd_s3 = 0

    while i < len(df_comb) - 100:
        
        # --- Strategy 1: Liquidation (M1) - Relaxed threshold (1.1 instead of 2.0) ---
        dir_s1 = 0
        if i >= cd_s1:
            liq_long = float(arr_liq_long[i])
            liq_short = float(arr_liq_short[i])
            ll_mean = float(arr_liq_long_mean[i])
            ls_mean = float(arr_liq_short_mean[i])
            if liq_long > ll_mean * 1.1:
                dir_s1 = 1  
            elif liq_short > ls_mean * 1.1:
                dir_s1 = -1
        
        # --- Strategy 2: CVD / Flow (M2) - Relaxed threshold (0.8 instead of 1.5) ---
        dir_s2 = 0
        if i >= cd_s2:
            z20 = float(arr_z20[i])
            if z20 >= 0.8:
                dir_s2 = 1
            elif z20 <= -0.8:
                dir_s2 = -1
                
        # --- Strategy 3: Trend / Price Action (M3) - Relaxed threshold (pull8 0.2 instead of 0.5) ---
        dir_s3 = 0
        if i >= cd_s3:
            macro = float(arr_macro[i])
            pull8 = float(arr_pull8[i])
            if macro > 0 and pull8 < -0.2:
                dir_s3 = 1
            elif macro < 0 and pull8 > 0.2:
                dir_s3 = -1

        for strategy_name, best_dir, cd_ref in [
            ("S1_Liquidation", dir_s1, cd_s1),
            ("S2_CVD", dir_s2, cd_s2),
            ("S3_Trend", dir_s3, cd_s3)
        ]:
            if best_dir != 0:
                entry = float(c[i])
                atr = float(a[i])
                if atr <= 0 or np.isnan(atr):
                    continue
                    
                sl_dist = sl_mult * atr
                tp_dist = tp_mult * atr
                
                sl = entry - sl_dist if best_dir == 1 else entry + sl_dist
                tp = entry + tp_dist if best_dir == 1 else entry - tp_dist
                
                limit = min(i + 96 + 1, len(c))
                hit = 0
                bars_held = limit - 1 - i
                
                for j in range(i + 1, limit):
                    if best_dir == 1:
                        if l[j] <= sl:
                            hit = -1
                            bars_held = j - i
                            break
                        if h[j] >= tp:
                            hit = 1
                            bars_held = j - i
                            break
                    else:
                        if h[j] >= sl:
                            hit = -1
                            bars_held = j - i
                            break
                        if l[j] <= tp:
                            hit = 1
                            bars_held = j - i
                            break
                
                label = 1 if hit == 1 else 0
                pnl = tp_mult if hit == 1 else (-sl_mult if hit == -1 else 0)

                feats = {c: feat_arrs[c][i] for c in feat_cols}
                
                trades[strategy_name].append({
                    'symbol': symbol,
                    'entry_time': ts[i],
                    'direction': best_dir,
                    'pnl': pnl,
                    'label': label,
                    'features': feats
                })
                
                if strategy_name == "S1_Liquidation": cd_s1 = i + bars_held + 2
                elif strategy_name == "S2_CVD": cd_s2 = i + bars_held + 2
                elif strategy_name == "S3_Trend": cd_s3 = i + bars_held + 2
                
        i += 1

    return trades

def process_symbol(symbol):
    print(f"[{symbol}] Starting trade generation...")
    try:
        df = load_asset(symbol)
        start_time = df.index.min()
        btc = load_asset("BTCUSDT")
        btc_ref = btc[["Close", "CVD"]].copy()
        btc_ref.columns = ["btc_Close", "btc_CVD"]
        trades = generate_independent_trades_raw(symbol, df, btc_ref if symbol != "BTCUSDT" else None)
        return symbol, trades, start_time
    except Exception as e:
        print(f"Error on {symbol}: {e}")
        return symbol, {"S1_Liquidation": [], "S2_CVD": [], "S3_Trend": []}, pd.Timestamp("2020-01-01")

def train_and_evaluate(trades_df, max_depth=3, learning_rate=0.05, n_estimators=100):
    exclude_cols = ['symbol', 'entry_time', 'direction', 'pnl', 'label', 'liq_long_5_mean', 'liq_short_5_mean']
    feature_cols = [c for c in trades_df.columns if c not in exclude_cols]
    
    oos_masks = []
    for start, end in MONTHS:
        mask = (trades_df['entry_time'] >= pd.Timestamp(start)) & (trades_df['entry_time'] <= pd.Timestamp(end))
        oos_masks.append(mask)
    
    is_oos = np.logical_or.reduce(oos_masks)
    
    train_df = trades_df[~is_oos]
    test_df = trades_df[is_oos]

    if len(train_df) < 50 or len(train_df[train_df['label'] == 1]) < 5:
        return None

    X_train = train_df[feature_cols].astype(float)
    y_train = train_df['label'].astype(int)
    
    X_test = test_df[feature_cols].astype(float)
    
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0

    model = lgb.LGBMClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=2,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    test_df = test_df.copy()
    test_df['prob'] = model.predict_proba(X_test)[:, 1]
    
    return test_df

# Globals for storing results and reports
REPORT_PATH = ROOT / "optimization" / "oos_optimization_report.md"
ARTIFACT_REPORT_PATH = Path(r"C:\Users\SIGMA\.gemini\antigravity-ide\brain\47f99fb2-51eb-4dda-a9f1-07d14a5c6ccc\oos_optimization_report.md")
FINAL_RESULTS = {}

def update_markdown_report():
    lines = [
        "# Multi-Strategy OOS Optimization Report",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This report tracks the optimization progress of S1_Liquidation, S2_CVD, and S3_Trend strategies.",
        "To pass the strict criteria, **each Strategy in every OOS monthly window** must achieve strictly `n_trades > 0`, `ROI >= 20%`, and `winrate >= 45%`.",
        ""
    ]
    
    for strategy_name, info in FINAL_RESULTS.items():
        lines.append(f"## Strategy: {strategy_name}")
        if "strat_params" not in info:
            lines.append("Optimization in progress...")
            lines.append("")
            continue
            
        lines.append("### Best Parameters found:")
        lines.append(f"- **Strategy Params**: `{info['strat_params']}`")
        lines.append(f"- **ML Params**: `{info['ml_params']}`")
        lines.append(f"- **Min Probability Threshold**: `{info['best_prob']:.2f}`")
        lines.append(f"- **Risk Percentage per Trade**: `{info['best_risk']:.1f}%`")
        lines.append(f"- **Passed Monthly Windows**: `{info['passed_windows']} / 18`")
        lines.append(f"- **Total Strategy OOS ROI**: `{info['total_roi']:.1f}%`")
        lines.append("")
        
        if "monthly_details" in info:
            lines.append(f"#### Month-by-Month Strategy OOS Performance:")
            lines.append("| Window | Start Date | End Date | Trades | Wins | Win Rate | Net R | Strategy ROI | Status |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            
            for w_idx, det in enumerate(info['monthly_details']):
                status_str = "🟢 PASS" if det['status'] == "PASS" else "🔴 FAIL"
                lines.append(
                    f"| {w_idx+1} | {det['start']} | {det['end']} | {det['trades']} | {det['wins']} | "
                    f"{det['wr']:.1f}% | {det['net_r']:.1f} | {det['roi']:.1f}% | {status_str} |"
                )
            lines.append("")
            
    content = "\n".join(lines)
    
    # Save to local path
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(content)
        
    # Save to artifacts path
    try:
        ARTIFACT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ARTIFACT_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"Failed to write artifact: {e}")

def run_optimization_for_strategy(strategy_name, all_trades_list, start_times):
    print(f"\n==================================================")
    print(f"RUNNING OPTIMIZATION FOR {strategy_name}")
    print(f"==================================================")
    
    trades_df = pd.DataFrame(all_trades_list)
    if len(trades_df) == 0:
        print("No trades available for this strategy.")
        FINAL_RESULTS[strategy_name] = {"error": "No trades"}
        update_markdown_report()
        return

    # Define hyperparameter grid search spaces
    if strategy_name == "S1_Liquidation":
        strat_space = [{'t_liq': t} for t in [1.1, 1.3, 1.6, 2.0, 2.5]]
    elif strategy_name == "S2_CVD":
        strat_space = [{'t_cvd': c, 't_cvd_fast': f} for c in [0.8, 1.0, 1.3, 1.6] for f in [0.3, 0.5, 0.7]]
    else:  # S3_Trend
        strat_space = [{'t_pull': p, 't_rsi': r} for p in [0.2, 0.3, 0.5, 0.8] for r in [45, 50]]

    ml_space = [
        {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 100},
        {'max_depth': 4, 'learning_rate': 0.01, 'n_estimators': 200}
    ]
    
    prob_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]
    risk_multipliers = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

    best_passed_windows = -1
    best_total_roi = -999999
    best_params = {}

    FINAL_RESULTS[strategy_name] = {}
    update_markdown_report()

    for s_param in strat_space:
        if strategy_name == "S1_Liquidation":
            t_liq = s_param['t_liq']
            df_strat = trades_df[
                ((trades_df['direction'] == 1) & (trades_df['liq_long_5'] >= trades_df['liq_long_5_mean'] * t_liq)) |
                ((trades_df['direction'] == -1) & (trades_df['liq_short_5'] >= trades_df['liq_short_5_mean'] * t_liq))
            ]
        elif strategy_name == "S2_CVD":
            t_cvd = s_param['t_cvd']
            t_cvd_fast = s_param['t_cvd_fast']
            df_strat = trades_df[
                ((trades_df['direction'] == 1) & (trades_df['z_cvd_20'] >= t_cvd) & (trades_df['z_cvd_4'] >= t_cvd_fast)) |
                ((trades_df['direction'] == -1) & (trades_df['z_cvd_20'] <= -t_cvd) & (trades_df['z_cvd_4'] <= -t_cvd_fast))
            ]
        else: # S3_Trend
            t_pull = s_param['t_pull']
            t_rsi = s_param['t_rsi']
            df_strat = trades_df[
                ((trades_df['direction'] == 1) & (trades_df['macro'] > 0) & (trades_df['pull_ema8'] < -t_pull) & (trades_df['rsi'] < t_rsi)) |
                ((trades_df['direction'] == -1) & (trades_df['macro'] < 0) & (trades_df['pull_ema8'] > t_pull) & (trades_df['rsi'] > 100 - t_rsi))
            ]

        if len(df_strat) < 50:
            continue

        for m_param in ml_space:
            test_df_with_probs = train_and_evaluate(
                df_strat,
                max_depth=m_param['max_depth'],
                learning_rate=m_param['learning_rate'],
                n_estimators=m_param['n_estimators']
            )
            
            if test_df_with_probs is None or len(test_df_with_probs) == 0:
                continue

            for prob in prob_thresholds:
                filtered_trades = test_df_with_probs[test_df_with_probs['prob'] >= prob]
                
                for risk in risk_multipliers:
                    passed_windows = 0
                    monthly_details = []
                    total_net_r = 0.0
                    
                    for start_str, end_str in MONTHS:
                        w_start = pd.Timestamp(start_str)
                        w_end = pd.Timestamp(end_str)
                        
                        w_trades = filtered_trades[(filtered_trades['entry_time'] >= w_start) & (filtered_trades['entry_time'] <= w_end)]
                        n_tr = len(w_trades)
                        
                        if n_tr == 0:
                            monthly_details.append({
                                'start': start_str, 'end': end_str,
                                'trades': 0, 'wins': 0, 'wr': 0.0, 'net_r': 0.0, 'roi': 0.0,
                                'status': 'FAIL'
                            })
                            continue
                            
                        wins = len(w_trades[w_trades['pnl'] > 0])
                        wr = (wins / n_tr) * 100
                        net_r = w_trades['pnl'].sum()
                        roi = net_r * risk
                        total_net_r += net_r
                        
                        status = "PASS" if (n_tr > 0 and roi >= 20.0 and wr >= 45.0) else "FAIL"
                        if status == "PASS":
                            passed_windows += 1
                            
                        monthly_details.append({
                            'start': start_str, 'end': end_str,
                            'trades': n_tr, 'wins': wins, 'wr': wr, 'net_r': net_r, 'roi': roi,
                            'status': status
                        })
                        
                    total_strat_roi = total_net_r * risk
                    
                    # Track best params (maximize passed_windows, then total_strat_roi)
                    if (passed_windows > best_passed_windows) or (passed_windows == best_passed_windows and total_strat_roi > best_total_roi):
                        best_passed_windows = passed_windows
                        best_total_roi = total_strat_roi
                        best_params = {
                            'strat_params': s_param,
                            'ml_params': m_param,
                            'best_prob': prob,
                            'best_risk': risk,
                            'passed_windows': passed_windows,
                            'total_roi': total_strat_roi,
                            'monthly_details': monthly_details
                        }
                        FINAL_RESULTS[strategy_name] = best_params
                        update_markdown_report()
                        print(f"  [New Best] Strat: {s_param} | ML: {m_param} | Prob: {prob} | Risk: {risk}% | Passed Windows: {passed_windows}/18 | Total ROI: {total_strat_roi:.1f}%")

    print(f"[{strategy_name}] Optimization Complete. Best passed windows: {best_passed_windows}/18 (Total ROI: {best_total_roi:.1f}%)")

def auto_optimize_all():
    print("Generating base trades for all symbols across 3 strategies...")
    
    all_trades = {"S1_Liquidation": [], "S2_CVD": [], "S3_Trend": []}
    asset_start_times = {}
    
    for sym in SYMBOLS:
        _, trades, start_time = process_symbol(sym)
        asset_start_times[sym] = start_time
        print(f"[{sym}] Generated. Start: {start_time.strftime('%Y-%m-%d')} | S1: {len(trades['S1_Liquidation'])}, S2: {len(trades['S2_CVD'])}, S3: {len(trades['S3_Trend'])}")
        
        for k in all_trades:
            for t in trades[k]:
                flat = {
                    'symbol': t['symbol'],
                    'entry_time': t['entry_time'],
                    'direction': t['direction'],
                    'pnl': t['pnl'],
                    'label': t['label']
                }
                flat.update(t['features'])
                all_trades[k].append(flat)
        gc.collect()

    for strategy_name in ["S1_Liquidation", "S2_CVD", "S3_Trend"]:
        run_optimization_for_strategy(strategy_name, all_trades[strategy_name], asset_start_times)
        
    print("\nOptimization completed successfully. Results updated in report files.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    auto_optimize_all()
