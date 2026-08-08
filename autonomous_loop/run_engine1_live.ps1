# Clean up any leftover duplicate CMD processes running Engine_1
$engineProcs = Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -like "*Engine_1.py*" -and $_.ProcessId -ne $PID }
foreach ($p in $engineProcs) {
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}

# Launch single maximized CMD console window for Engine_1 Live Trading
Set-Location "C:\Users\SIGMA\Documents\Project - Coinglass Trading\Engine_1_arena_PR"
$cmdPath = "$env:SystemRoot\System32\cmd.exe"
Start-Process $cmdPath -WindowStyle Maximized -ArgumentList "/k", "title Engine_1_Live_Console && cd /d `"C:\Users\SIGMA\Documents\Project - Coinglass Trading\Engine_1_arena_PR`" && python -u Engine_1.py --live --skip-seed --skip-train"
