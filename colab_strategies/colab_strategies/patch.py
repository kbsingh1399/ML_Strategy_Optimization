import re
from pathlib import Path

source_file = r'C:\Users\SIGMA\AppData\Local\Temp\ML_Strategy_Optimization_Sync\run_all_6.py'
dest_dir = Path(r'C:\Users\SIGMA\Documents\Project - Coinglass Trading\Engine_1_arena_PR\colab_strategies\colab_strategies')

with open(source_file, 'r', encoding='utf-8') as f:
    code = f.read()

strats_list = [
    ('S1_Liquidation', 'make_signal_s1'),
    ('S2_CVD_Momentum', 'make_signal_s2'),
    ('S3_Trend_Follow', 'make_signal_s3'),
    ('S4_Mean_Reversion', 'make_signal_s4'),
    ('S5_Vol_Breakout', 'make_signal_s5'),
    ('S6_OI_Coherence', 'make_signal_s6')
]

for i, (name, func) in enumerate(strats_list, start=1):
    new_code = re.sub(
        r'STRATS=\[.*?\]', 
        f'STRATS=[("{name}", {func})]', 
        code, 
        flags=re.DOTALL
    )
    new_code = new_code.replace('all_6_results.json', f's{i}_results.json')
    
    out_file = dest_dir / f'opt_s{i}_colab_standalone.py'
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(new_code)
    print(f'Successfully updated {out_file.name}')
