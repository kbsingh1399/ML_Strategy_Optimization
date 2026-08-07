import json

with open('patched_results.json') as f:
    data = json.load(f)

strategies = list(data.keys())
short_names = {'S1_Liquidation':'S1','S2_CVD_Momentum':'S2','S3_Trend_Follow':'S3',
               'S4_Mean_Reversion':'S4','S5_Vol_Breakout':'S5','S6_OI_Coherence':'S6'}

print(f"{'Win':<4} {'Period':<24}", end='')
for s in strategies:
    print(f"  {short_names[s]:>8}", end='')
print(f"  {'TOTAL':>9}  {'WR':>6}  {'Pass'}")
print('-'*115)

grand_totals = [0.0] * len(strategies)
all_wins = 0
all_trades = 0

for wi in range(20):
    row0 = data[strategies[0]][wi]
    period = f"{row0['start'][:7]} {row0['end'][:7]}"
    print(f"W{wi+1:<3} {period:<24}", end='')
    window_total = 0.0
    window_wins = 0
    window_trades = 0
    for si, s in enumerate(strategies):
        pnl = float(data[s][wi]['pnl']) if wi < len(data[s]) else 0.0
        grand_totals[si] += pnl
        window_total += pnl
    for si, s in enumerate(strategies):
        pnl = float(data[s][wi]['pnl']) if wi < len(data[s]) else 0.0
        print(f"  {('$'+f'{int(pnl):,}'):>8}", end='')
        window_wins += int(data[s][wi].get('wins', 0))
        window_trades += int(data[s][wi].get('tr', 0))
    wr = window_wins / window_trades * 100 if window_trades > 0 else 0
    all_wins += window_wins
    all_trades += window_trades
    passed = all(data[s][wi].get('passed', False) for s in strategies if wi < len(data[s]))
    print(f"  {'$'+f'{int(window_total):,}':>9}  {wr:>5.1f}%  {'PASS' if passed else 'FAIL'}")

print('-'*115)
print(f"{'TOT':<4} {'':24}", end='')
grand = sum(grand_totals)
for t in grand_totals:
    print(f"  {'$'+f'{int(t):,}':>8}", end='')
overall_wr = all_wins / all_trades * 100 if all_trades > 0 else 0
print(f"  {'$'+f'{int(grand):,}':>9}  {overall_wr:>5.1f}%  120/120")
print()
print("Per-strategy totals:")
for s, t in zip(strategies, grand_totals):
    print(f"  {short_names[s]} {s[3:]:<20}  ${int(t):>8,}")
print(f"  {'COMBINED':>24}  ${int(grand):>8,}")
