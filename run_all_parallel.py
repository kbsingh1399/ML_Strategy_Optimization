import subprocess
import threading
import sys
import os
import time
import ctypes
from ctypes import wintypes
from datetime import datetime

LOG_FILE = f"opt_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

scripts = [
    ("S1", "colab_strategies/colab_strategies/opt_s1_colab_standalone.py"),
    ("S2", "colab_strategies/colab_strategies/opt_s2_colab_standalone.py"),
    ("S3", "colab_strategies/colab_strategies/opt_s3_colab_standalone.py"),
    ("S4", "colab_strategies/colab_strategies/opt_s4_colab_standalone.py"),
    ("S5", "colab_strategies/colab_strategies/opt_s5_colab_standalone.py"),
    ("S6", "colab_strategies/colab_strategies/opt_s6_colab_standalone.py"),
]

class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage2', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
    ]

GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
OpenProcess = ctypes.windll.kernel32.OpenProcess
CloseHandle = ctypes.windll.kernel32.CloseHandle
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010

def get_working_set_size(pid):
    handle = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not handle:
        return 0
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    if GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        CloseHandle(handle)
        return counters.WorkingSetSize
    CloseHandle(handle)
    return 0

_log_fh = open(LOG_FILE, 'w', encoding='utf-8', buffering=1)

def tee(msg):
    print(msg, flush=True)
    _log_fh.write(msg + '\n')
    _log_fh.flush()

def stream_output(prefix, proc, mem_stats):
    peak_mem = 0

    def monitor():
        nonlocal peak_mem
        while proc.poll() is None:
            mem = get_working_set_size(proc.pid)
            if mem > peak_mem:
                peak_mem = mem
                mem_stats['peak'] = max(mem_stats['peak'], peak_mem)
            time.sleep(0.5)
        mem = get_working_set_size(proc.pid)
        if mem > peak_mem:
            peak_mem = mem
            mem_stats['peak'] = max(mem_stats['peak'], peak_mem)

    t = threading.Thread(target=monitor)
    t.daemon = True
    t.start()

    for line in iter(proc.stdout.readline, ''):
        if line:
            tee(f"[{prefix}] {line.strip()}")
    proc.stdout.close()
    t.join()

tee(f"Log file: {os.path.abspath(LOG_FILE)}")
tee("Launching 3 Strategies SEQUENTIALLY with memory monitoring...")

for prefix, script in scripts:
    if os.path.exists(script):
        tee(f"\n--- Starting {prefix} ({script}) ---")
        proc = subprocess.Popen(
            [sys.executable, "-u", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        mem_stats = {'peak': 0}
        stream_output(prefix, proc, mem_stats)
        proc.wait()
        peak_mb = mem_stats['peak'] / (1024 * 1024)
        tee(f"[{prefix}] Completed. Exit code: {proc.returncode}. Peak WorkingSet: {peak_mb:.2f} MB")
        if peak_mb > 800:
            tee(f"[WARN] Peak memory ({peak_mb:.2f} MB) exceeded 800 MB soft limit!")
    else:
        tee(f"[WARN] {script} not found — skipping!")

tee("\nALL 3 STRATEGIES COMPLETED.")
_log_fh.close()
