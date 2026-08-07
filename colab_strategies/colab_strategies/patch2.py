import os
from pathlib import Path

dest_dir = Path(r'C:\Users\SIGMA\Documents\Project - Coinglass Trading\Engine_1_arena_PR\colab_strategies\colab_strategies')

for i in range(1, 7):
    file_path = dest_dir / f'opt_s{i}_colab_standalone.py'
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Replace the ROOT and DATA path to point two directories up
    new_code = code.replace("ROOT=Path('.'); DATA=ROOT/'Backtesting_Data'", 
                            "ROOT=Path('../..'); DATA=ROOT/'backtesting_data'")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_code)
    
print("Successfully patched data paths for all 6 standalone files.")
