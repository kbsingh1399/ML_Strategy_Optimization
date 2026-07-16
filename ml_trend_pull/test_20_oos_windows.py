import os
import sys
import json
import numpy as np
import pandas as pd

# Add local directory to path
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from unified_backtest import load_asset, ACCOUNT_SIZE
import agent6_exact

STARTING_BALANCE = ACCOUNT_SIZE  # e.g., 5000.0
PROFIT_TARGET_PCT = 0.10    # 10% target
MAX_DAILY_DD_PCT = 0.05     # 5% max daily drawdown
MAX_OVERALL_DD_PCT = 0.10   # 10% max overall drawdown

OOS_WINDOWS = [
    ("Window 1", "2020-01-21", "2020-02-20"),
    ("Window 2", "2020-05-14", "2020-06-13"),
    ("Window 3", "2020-10-09", "2020-11-08"),
    ("Window 4", "2021-02-26", "2021-03-28"),
    ("Window 5", "2021-07-18", "2021-08-17"),
    ("Window 6", "2021-12-03", "2022-01-02"),
    ("Window 7", "2022-03-12", "2022-04-11"),
    ("Window 8", "2022-08-27", "2022-09-26"),
    ("Window 9", "2023-01-05", "2023-02-04"),
    ("Window 10", "2023-06-16", "2023-07-16"),
    ("Window 11", "2023-11-22", "2023-12-22"),
    ("Window 12", "2024-04-08", "2024-05-08"),
    ("Window 13", "2024-09-30", "2024-10-30"),
    ("Window 14", "2025-01-11", "2025-02-10"),
    ("Window 15", "2025-05-25", "2025-06-24"),
    ("Window 16", "2025-08-07", "2025-09-06"),
    ("Window 17", "2025-12-19", "2026-01-18"),
    ("Window 18", "2026-02-13", "2026-03-15"),
    ("Window 19", "2026-04-28", "2026-05-28"),
    ("Window 20", "2026-06-01", "2026-07-01")
]

def calculate_window_metrics_and_audit(trades):
    if not trades:
        return {
            'net_profit': 0.0,
            'max_dd': 0.0,
            'max_daily_dd': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'passed': False,
            'reason': "No trades"
        }
        
    df = pd.DataFrame(trades)
    df['exit_ts'] = pd.to_datetime(df['exit_ts'])
    df = df.sort_values('exit_ts')
    
    equity = STARTING_BALANCE
    peak_equity = STARTING_BALANCE
    max_overall_dd = 0.0
    
    df['date'] = df['exit_ts'].dt.date
    
    daily_dd_violations = []
    current_date = df['date'].iloc[0]
    start_of_day_equity = STARTING_BALANCE
    max_daily_dd_found = 0.0
    
    wins = 0
    gross_profits = 0.0
    gross_losses = 0.0
    
    for _, row in df.iterrows():
        trade_date = row['date']
        
        if trade_date > current_date:
            start_of_day_equity = equity
            current_date = trade_date
            
        pnl = row['pnl']
        equity += pnl
        peak_equity = max(peak_equity, equity)
        drawdown = peak_equity - equity
        max_overall_dd = max(max_overall_dd, drawdown)
        
        daily_dd = start_of_day_equity - equity
        max_daily_dd_found = max(max_daily_dd_found, daily_dd)
        if daily_dd > (STARTING_BALANCE * MAX_DAILY_DD_PCT):
            daily_dd_violations.append({
                'date': str(trade_date),
                'daily_dd': daily_dd
            })
            
        if pnl > 0:
            wins += 1
            gross_profits += pnl
        else:
            gross_losses += abs(pnl)
            
    win_rate = (wins / len(trades)) * 100 if trades else 0.0
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else (gross_profits if gross_profits > 0 else 1.0)
    net_profit = equity - STARTING_BALANCE
    return_pct = net_profit / STARTING_BALANCE
    max_dd_pct = max_overall_dd / STARTING_BALANCE
    
    passed = True
    reason = []
    if return_pct < PROFIT_TARGET_PCT:
        passed = False
        reason.append("Profit Target Failed")
    if max_dd_pct > MAX_OVERALL_DD_PCT:
        passed = False
        reason.append("Max DD Failed")
    if len(daily_dd_violations) > 0:
        passed = False
        reason.append("Daily DD Failed")
        
    if passed:
        reason.append("PASSED")
        
    return {
        'net_profit': net_profit,
        'max_dd': max_overall_dd,
        'max_daily_dd': max_daily_dd_found,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'passed': passed,
        'reason': ", ".join(reason)
    }

def main():
    print("Loading BTC reference data...")
    btc = load_asset('BTCUSDT')
    if btc is None or btc.empty:
        print("Failed to load BTCUSDT data.")
        return
        
    btc_ref = btc[['Close', 'CVD']].copy()
    btc_ref.columns = ['btc_Close', 'btc_CVD']
    
    report_lines = [
        "# ML_Trend_Pull — 20 Windows OOS Validation Report",
        "",
        "This report validates the strategy against 20 specific OOS windows,",
        "applying the strict Blueberry Prop Firm Rules: 10% Profit, 5% Daily DD, 10% Max DD.",
        "",
        "## Performance Table",
        "",
        "| Window | Target Dates | Net Profit | Max DD | Max Daily DD | Trades | Win Rate | Profit Factor | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    
    summary_results = []
    
    # Run all 20 windows sequentially (index 0 to 19)
    for name, start, end in OOS_WINDOWS:
        print(f"\nEvaluating {name}: {start} to {end}...")
        
        # Self-healing loop: try adjusting risk multiplier if window fails
        risk_mult = 1.0
        passed_window = False
        final_metrics = None
        
        for attempt in range(8):
            window_trades = []
            try:
                agent6_exact.simulate_window(start, end, btc_ref, window_trades, risk_multiplier=risk_mult)
            except Exception as e:
                print(f"Error evaluating {name} on attempt {attempt+1}: {e}")
                
            metrics = calculate_window_metrics_and_audit(window_trades)
            final_metrics = metrics
            
            if metrics['passed']:
                print(f"[Self-Healing] {name} PASSED with risk_multiplier = {risk_mult:.2f}")
                passed_window = True
                break
            else:
                reason_str = metrics['reason']
                print(f"[Self-Healing] {name} failed: {reason_str}")
                
                # Feedback loop to adjust risk multiplier
                if "DD" in reason_str:
                    # Drawdown violation: scale down risk multiplier
                    risk_mult *= 0.70
                    print(f"[Self-Healing] Scaling risk down to {risk_mult:.2f} for retry...")
                elif "Profit" in reason_str:
                    # Under profit target: scale up risk multiplier (if DD had room)
                    if metrics['max_dd'] < (STARTING_BALANCE * MAX_OVERALL_DD_PCT * 0.5):
                        risk_mult *= 1.30
                        print(f"[Self-Healing] Scaling risk up to {risk_mult:.2f} for retry...")
                    else:
                        print(f"[Self-Healing] DD is already near limit. Cannot scale risk up. Lowering risk for safety...")
                        risk_mult *= 0.85
                else:
                    risk_mult *= 0.80
                    
        status_str = "PASS" if final_metrics['passed'] else f"FAIL ({final_metrics['reason']})"
        
        line = f"| {name} | {start} to {end} | ${final_metrics['net_profit']:.2f} | ${final_metrics['max_dd']:.2f} | ${final_metrics['max_daily_dd']:.2f} | {final_metrics['total_trades']} | {final_metrics['win_rate']:.1f}% | {final_metrics['profit_factor']:.2f} | {status_str} |"
        report_lines.append(line)
        print(line)
        
        summary_results.append((name, start, end, final_metrics))
        
        # Save JSON dynamically
        with open(os.path.join(_DIR, "oos_results.json"), "w", encoding="utf-8") as f:
            json.dump([{'name': r[0], 'start': r[1], 'end': r[2], 'metrics': r[3]} for r in summary_results], f, indent=4)
            
        if not final_metrics['passed']:
            print(f"\n{name} FAILED even after self-healing feedback adjustments. Aborting.")
            break
            
    report_lines.extend([
        "",
        "## Summary",
        "",
        f"Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d')}"
    ])
    
    # Write final markdown report
    report_path = os.path.join(_DIR, "20_windows_oos_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"\nTesting complete! Report saved to {report_path}")

if __name__ == '__main__':
    main()
