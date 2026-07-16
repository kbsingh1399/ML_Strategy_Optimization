# C:\Users\SIGMA\Documents\Project - Coinglass Trading\Engine_1\supervisor.py
import os
import sys
import time
import subprocess
import threading
import signal

# Flag to prevent crash loop if it continuously fails on startup
cooldown_seconds = 5
max_consecutive_latency_alerts = 3

class EngineSupervisor:
    def __init__(self):
        self.process = None
        self.running = True
        self.consecutive_latency = 0
        self.log_file = "supervisor.log"

    def log(self, msg: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [Supervisor] {msg}\n"
        sys.stdout.write(log_line)
        sys.stdout.flush()
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception:
            pass

    def cleanup_chrome(self):
        self.log("Cleaning up orphan Chrome processes...")
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["powershell", "-Command", 'Stop-Process -Name "chrome" -Force -ErrorAction SilentlyContinue'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                subprocess.run(["pkill", "-f", "chrome"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            self.log(f"Failed to clean up Chrome: {e}")

    def terminate_engine(self):
        if self.process:
            self.log("Terminating trading engine...")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.log("Engine did not exit gracefully, killing...")
                self.process.kill()
                self.process.wait()
            except Exception as e:
                self.log(f"Error terminating engine: {e}")
            self.process = None

    def start_engine(self):
        self.log("Starting Engine_1.py...")
        env = os.environ.copy()
        
        # Ensure outputs are flushed immediately
        env["PYTHONUNBUFFERED"] = "1"
        
        # Start Engine_1.py
        self.process = subprocess.Popen(
            [sys.executable, "Engine_1.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=1,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        self.consecutive_latency = 0

    def monitor_loop(self):
        while self.running:
            self.start_engine()
            
            # Start stdout reading thread
            def read_stdout(proc):
                try:
                    for line in iter(proc.stdout.readline, ''):
                        if not self.running:
                            break
                        # Forward output to console
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        
                        # Write raw line to log file
                        try:
                            with open(self.log_file, "a", encoding="utf-8") as f:
                                f.write(line)
                        except Exception:
                            pass

                        # Scan for anomaly alerts
                        if "[Watchdog] [ALERT] [LATENCY_CRITICAL]" in line:
                            self.log("Critical latency limit reached (event loop blocked consecutively). Triggering restart.")
                            self.terminate_engine()
                            break

                        if "[Watchdog] [ALERT] [MEMORY]" in line:
                            self.log("Critical memory alert detected. Triggering restart to prevent crash.")
                            self.terminate_engine()
                            break

                except Exception as e:
                    self.log(f"Error reading engine output: {e}")

            reader_thread = threading.Thread(target=read_stdout, args=(self.process,), daemon=True)
            reader_thread.start()

            # Wait for process to exit
            ret_code = self.process.wait()
            if self.running:
                self.log(f"Engine exited with code {ret_code}.")
                self.terminate_engine()
                self.cleanup_chrome()
                self.log(f"Re-launching engine in {cooldown_seconds} seconds...")
                time.sleep(cooldown_seconds)

    def stop(self):
        self.running = False
        self.terminate_engine()
        self.cleanup_chrome()
        self.log("Supervisor stopped.")

if __name__ == "__main__":
    supervisor = EngineSupervisor()
    
    def signal_handler(sig, frame):
        supervisor.log("Termination signal received. Exiting supervisor...")
        supervisor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        supervisor.monitor_loop()
    except KeyboardInterrupt:
        signal_handler(None, None)
