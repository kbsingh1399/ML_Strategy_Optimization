import shutil
from pathlib import Path

src = Path(r'C:\Users\SIGMA\AppData\Local\Temp\ML_Strategy_Optimization_Sync\run_all_6.py')
dest = Path(r'C:\Users\SIGMA\Documents\Project - Coinglass Trading\Engine_1_arena_PR\colab_strategies\colab_strategies\run_all_6.py')

shutil.copy(src, dest)

with open(dest, 'r', encoding='utf-8') as f:
    code = f.read()

# Apply patches
code = code.replace("ROOT=Path('.'); DATA=ROOT/'Backtesting_Data'", "ROOT=Path('../..'); DATA=ROOT/'backtesting_data'")
code = code.replace(r'log(f"\u23f1 {name}', r'log(f"TIME {name}')
code = code.replace('⏱', 'TIME')

with open(dest, 'w', encoding='utf-8') as f:
    f.write(code)

print("Successfully copied and patched run_all_6.py")
