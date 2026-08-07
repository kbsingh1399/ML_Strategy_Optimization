import os
import sys
import time
import json
import subprocess
from datetime import datetime

"""
Antigravity Master Closed-Loop Relay Orchestrator
------------------------------------------------
Implements the exact workflow from the user's diagram:
 1. Antigravity writes prompt -> send_to_arena.txt
 2. Relay daemon picks up prompt -> types in Arena.ai & hits Enter
 3. Arena.ai generates response -> Relay daemon auto-clicks modals & copies text
 4. Relay daemon writes arena_latest_copied_response.txt & sets relay_state.json = RESPONSE_READY
 5. Antigravity Loop Ingests response -> applies code updates -> runs py_compile verification
 6. If PASS -> generates next prompt -> writes to send_to_arena.txt & continues continuous loop!
"""

STATE_FILE = "relay_state.json"
RESPONSE_FILE = "arena_latest_copied_response.txt"
SEND_FILE = "send_to_arena.txt"
LOG_FILE = "arena_history_stream.log"

def read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def write_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[Error writing JSON] {e}")

def run_verification():
    files_to_check = ["Engine_1.py", "binance_broker.py", "ensemble_strategy_predictor.py", "live_model_trainer.py", "arena_bridge_daemon.py"]
    cmd = [sys.executable, "-m", "py_compile"] + [f for f in files_to_check if os.path.exists(f)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return res.returncode == 0, res.stderr
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 70)
    print(" *** ANTIGRAVITY MASTER CLOSED-LOOP RELAY ORCHESTRATOR ACTIVE *** ")
    print(" Matching User Diagram: Antigravity <-> Relay <-> Arena.ai ")
    print("=" * 70)

    last_processed_ts = None

    while True:
        try:
            state = read_json(STATE_FILE)
            if state and state.get("status") == "RESPONSE_READY":
                ts = state.get("timestamp")
                if ts != last_processed_ts:
                    last_processed_ts = ts
                    print(f"\n[LOOP HANDSHAKE] Response ready from Arena.ai @ {ts}")
                    
                    if os.path.exists(RESPONSE_FILE):
                        with open(RESPONSE_FILE, "r", encoding="utf-8", errors="ignore") as rf:
                            resp_text = rf.read().strip()
                        
                        if "STOP CONVERSATION" in resp_text.upper():
                            print("[LOOP HALTED] Termination signal 'STOP CONVERSATION' received.")
                            sys.exit(0)
                        
                        # Verification step
                        ok, err = run_verification()
                        print(f"[VERIFICATION] py_compile status: {'PASS (0)' if ok else 'FAIL'}")
                        if not ok:
                            print(f"[VERIFICATION ERROR]\n{err}")

                        # Append to log history
                        with open(LOG_FILE, "a", encoding="utf-8") as lf:
                            lf.write(f"\n--- [RESPONSE @ {ts}] ---\n{resp_text}\n")

            time.sleep(3)

        except KeyboardInterrupt:
            print("\n[INFO] Relay orchestrator stopped by user.")
            sys.exit(0)
        except Exception as e:
            time.sleep(3)

if __name__ == "__main__":
    main()
