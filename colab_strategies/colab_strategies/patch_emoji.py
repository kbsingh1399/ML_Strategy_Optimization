import os
from pathlib import Path

dest_dir = Path(r'C:\Users\SIGMA\Documents\Project - Coinglass Trading\Engine_1_arena_PR\colab_strategies\colab_strategies')

for i in range(1, 7):
    file_path = dest_dir / f'opt_s{i}_colab_standalone.py'
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Remove the stopwatch emoji causing Windows encode errors
    new_code = code.replace(r'log(f"\u23f1 {name}', r'log(f"TIME {name}')
    new_code = new_code.replace('⏱', 'TIME')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_code)
    
print("Successfully removed emoji from all 6 standalone files.")
