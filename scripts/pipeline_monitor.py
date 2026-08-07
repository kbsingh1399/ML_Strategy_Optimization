import os
import json
import csv
from datetime import datetime, timedelta
import glob

# Constants
ROOT_DIR = r"C:\Users\SIGMA\Documents\Project - Coinglass Trading\Engine_1_arena_PR"
LIVE_TRADES_CSV = os.path.join(ROOT_DIR, "live_trades_journal.csv")
ENGINE_LOGS_JSON = os.path.join(ROOT_DIR, "Engine_1_trade_logs.json")
SUPERVISOR_LOG = os.path.join(ROOT_DIR, "supervisor.log")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
MANIFEST_PATH = os.path.join(MODELS_DIR, "manifest.json")

# OOS Expected Baselines (derived from opt scripts)
OOS_EXPECTED_WIN_RATE = 50.0 # Alert if dips below 50%
OOS_EXPECTED_PROFIT_FACTOR = 1.1 # Alert if dips below 1.1

class DataIntegrityAgent:
    """Monitors Coinglass/MT5 data feed health by checking recent trades or data."""
    def run(self):
        findings = []
        # In strict token-saving mode, we just check if live_trades_journal.csv was modified recently.
        # But since we killed processes, we just report the current state.
        if os.path.exists(LIVE_TRADES_CSV):
            mod_time = datetime.fromtimestamp(os.path.getmtime(LIVE_TRADES_CSV))
            now = datetime.now()
            hours_since = (now - mod_time).total_seconds() / 3600
            findings.append(f"- `live_trades_journal.csv` last updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')} ({hours_since:.1f} hours ago).")
            if hours_since > 2:
                findings.append("  ⚠️ **ALERT**: No new trades recorded in over 2 hours. Pipeline may be stalled.")
        else:
            findings.append("- ⚠️ `live_trades_journal.csv` not found.")
        return findings

class OOSModelHealthAgent:
    """Monitors live model staleness and alignment with OOS expectations."""
    def run(self):
        findings = []
        # Check manifest
        if os.path.exists(MANIFEST_PATH):
            mod_time = datetime.fromtimestamp(os.path.getmtime(MANIFEST_PATH))
            now = datetime.now()
            hours_since = (now - mod_time).total_seconds() / 3600
            findings.append(f"- `manifest.json` last updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')} ({hours_since:.1f} hours ago).")
            if hours_since > 24:
                findings.append("  ⚠️ **ALERT**: Models have not been retrained in over 24 hours.")
        else:
            findings.append("- ⚠️ `manifest.json` not found in `models/`.")
            
        # OOS Alignment
        if os.path.exists(LIVE_TRADES_CSV):
            total_trades = 0
            winning_trades = 0
            gross_profit = 0.0
            gross_loss = 0.0
            with open(LIVE_TRADES_CSV, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        pnl = float(row.get('pnl_usd', 0))
                        total_trades += 1
                        if pnl > 0:
                            winning_trades += 1
                            gross_profit += pnl
                        else:
                            gross_loss += abs(pnl)
                    except ValueError:
                        pass
            if total_trades > 0:
                win_rate = (winning_trades / total_trades) * 100
                profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
                findings.append(f"- **OOS Alignment**: Live Win Rate: {win_rate:.1f}% (Expected > {OOS_EXPECTED_WIN_RATE}%), Live Profit Factor: {profit_factor:.2f} (Expected > {OOS_EXPECTED_PROFIT_FACTOR})")
                
                if win_rate < OOS_EXPECTED_WIN_RATE:
                    findings.append(f"  ⚠️ **ALERT**: Win Rate ({win_rate:.1f}%) is below OOS baseline ({OOS_EXPECTED_WIN_RATE}%).")
                if profit_factor < OOS_EXPECTED_PROFIT_FACTOR:
                    findings.append(f"  ⚠️ **ALERT**: Profit Factor ({profit_factor:.2f}) is below OOS baseline ({OOS_EXPECTED_PROFIT_FACTOR}).")
        
        return findings

class ExecutionAuditorAgent:
    """Scans logs for runtime anomalies."""
    def run(self):
        findings = []
        # Check tail of supervisor.log
        if os.path.exists(SUPERVISOR_LOG):
            try:
                # Read last 50000 bytes efficiently
                with open(SUPERVISOR_LOG, 'rb') as f:
                    f.seek(0, 2) # EOF
                    size = f.tell()
                    offset = max(0, size - 50000)
                    f.seek(offset)
                    tail = f.read().decode('utf-8', errors='ignore')
                    
                error_count = tail.upper().count('ERROR')
                exception_count = tail.upper().count('EXCEPTION')
                rejected_count = tail.upper().count('REJECTED')
                
                findings.append(f"- `supervisor.log` (Last 50KB check): {error_count} Errors, {exception_count} Exceptions, {rejected_count} Rejections found.")
                if exception_count > 0:
                    findings.append("  ⚠️ **ALERT**: Recent exceptions detected in supervisor.log.")
            except Exception as e:
                findings.append(f"- ⚠️ Could not read supervisor.log: {e}")
        return findings

def orchestrate():
    report_lines = ["# Multi-Agent Pipeline Monitoring Report", ""]
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n## 1. Data Integrity Agent")
    report_lines.extend(DataIntegrityAgent().run())
    
    report_lines.append("\n## 2. OOS & Model Health Agent")
    report_lines.extend(OOSModelHealthAgent().run())
    
    report_lines.append("\n## 3. Execution Auditor Agent")
    report_lines.extend(ExecutionAuditorAgent().run())
    
    report_path = r"C:\Users\SIGMA\.gemini\antigravity\brain\279fba05-6ffc-419e-9a8f-43b6d60f7987\monitoring_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"Monitoring report generated at {report_path}")

if __name__ == "__main__":
    orchestrate()
