Set-Location "C:\Users\SIGMA\Documents\Project - Coinglass Trading\Engine_1_arena_PR"
$cmdPath = "$env:SystemRoot\System32\cmd.exe"
Start-Process $cmdPath -WindowStyle Maximized -ArgumentList "/k", "cd /d `"C:\Users\SIGMA\Documents\Project - Coinglass Trading\Engine_1_arena_PR`" && python -u Engine_1.py --live --skip-seed --skip-train"
