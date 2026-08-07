import time
import os
import sys

# Add root dir to path if needed for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.pipeline_monitor import orchestrate

def run_daemon():
    print("Starting daemon monitor...")
    while True:
        try:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running monitoring cycle...")
            orchestrate()
        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error in monitoring cycle: {e}")
        
        # Sleep for 1 hour
        time.sleep(3600)

if __name__ == "__main__":
    run_daemon()
