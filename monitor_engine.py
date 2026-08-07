import time
import os

log_file = r"c:\Users\SIGMA\Documents\Project - Coinglass Trading\Engine_1_arena_PR\engine_out.txt"

def follow(thefile):
    # thefile.seek(0, os.SEEK_END) # start from current position instead of end to catch everything
    while True:
        try:
            line = thefile.readline()
            if not line:
                time.sleep(0.5)
                continue
            yield line
        except UnicodeDecodeError:
            yield "<decode_error>\n"
            pass

print("Monitoring engine_out.txt for live phase...", flush=True)

while not os.path.exists(log_file):
    time.sleep(1)

with open(log_file, "r", encoding="utf-16le", errors="ignore") as f:
    for line in follow(f):
        # We will also print everything so it doesn't get lost, but prefix live things
        if "prediction" in line.lower() or "mt5" in line.lower() or "legacy" in line.lower() or "squeezer" in line.lower() or "live" in line.lower() or "seed" in line.lower() or "firing" in line.lower() or "bridge" in line.lower():
            print("[MATCH] " + line.strip(), flush=True)
        elif "[TAB" not in line and "candle" not in line: # avoid spam
            pass
        # To debug if it's printing anything at all
        if "Launch" in line or "Start" in line or "Complete" in line:
            print("[DEBUG] " + line.strip(), flush=True)
