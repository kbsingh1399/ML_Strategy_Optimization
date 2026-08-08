"""
Git Auto-Sync — Pure Python replacement for git-autopush.ps1
Watches for file changes, debounces, pre-validates syntax, then commits + pushes.
Handles conflicts via stash → pull --rebase → pop. Retries on network failure.
"""
import os
import sys
import time
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WATCH_INTERVAL = 8          # seconds between change checks
DEBOUNCE_SECONDS = 5        # wait after detecting change before committing
MAX_PUSH_RETRIES = 3
PUSH_RETRY_DELAY = 15       # seconds between push retries

SKIP_PATTERNS = [
    "__pycache__", ".git", "catboost_info", "Seeding",
    "arena_dom.html", "arena_output.txt", "latest_response_text.txt",
    "arena_history_stream.log", "watchdog.log", "supervisor.log",
    ".log", ".png", ".jpg", ".zip", ".xlsx", ".pdf", ".tmp",
    "tab_history.txt", "relay_state.json",
]

SYNTAX_CHECK_FILES = [
    "Engine_1.py",
    "binance_broker.py",
    "ensemble_strategy_predictor.py",
    "live_model_trainer.py",
    "relay_engine.py",
    "arena_bridge_daemon.py",
    "watchdog.py",
    "git_sync.py",
]


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [GIT-SYNC] {msg}", flush=True)


def run_git(args: list, timeout: int = 30) -> tuple:
    try:
        r = subprocess.run(
            ["git"] + args,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except Exception as e:
        return -1, "", str(e)


def syntax_ok() -> bool:
    """Verify syntax of core files before pushing."""
    existing = [f for f in SYNTAX_CHECK_FILES if os.path.exists(os.path.join(BASE_DIR, f))]
    if not existing:
        return True
    try:
        r = subprocess.run(
            [sys.executable, "-m", "py_compile"] + existing,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if r.returncode != 0:
            log(f"Syntax check FAILED — blocking push:\n{r.stderr[:300]}")
            return False
        return True
    except Exception as e:
        log(f"Syntax check exception: {e}")
        return False


def get_changed_files() -> list:
    """Return list of changed files, filtered by skip patterns."""
    code, out, _ = run_git(["status", "--porcelain"])
    if code != 0 or not out:
        return []
    changed = []
    for line in out.splitlines():
        if len(line) < 3:
            continue
        fpath = line[3:].strip().strip('"')
        skip = False
        for pat in SKIP_PATTERNS:
            if pat in fpath:
                skip = True
                break
        if not skip:
            changed.append(fpath)
    return changed


def resolve_conflicts():
    """Stash local changes, pull --rebase, pop stash."""
    log("Attempting conflict resolution: stash → rebase → pop...")
    run_git(["stash"], timeout=15)
    code, _, err = run_git(["pull", "--rebase", "origin"], timeout=30)
    if code != 0:
        log(f"Rebase failed: {err}. Aborting rebase.")
        run_git(["rebase", "--abort"], timeout=10)
    run_git(["stash", "pop"], timeout=15)


def push_with_retry():
    for attempt in range(1, MAX_PUSH_RETRIES + 1):
        code, out, err = run_git(["push"], timeout=45)
        if code == 0:
            log(f"Push successful (attempt {attempt}).")
            return True
        log(f"Push attempt {attempt} failed: {err}")
        if "rejected" in err.lower() or "conflict" in err.lower():
            resolve_conflicts()
        if attempt < MAX_PUSH_RETRIES:
            time.sleep(PUSH_RETRY_DELAY)
    log("All push attempts failed.")
    return False


def commit_and_push(changed_files: list):
    if not syntax_ok():
        log("Skipping commit — syntax errors detected.")
        return

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = ", ".join(os.path.basename(f) for f in changed_files[:5])
    if len(changed_files) > 5:
        summary += f" +{len(changed_files)-5} more"

    msg = f"Auto-sync [{ts}]: {summary}"

    code, _, err = run_git(["add", "-A"], timeout=15)
    if code != 0:
        log(f"git add failed: {err}")
        return

    code, out, err = run_git(["commit", "-m", msg], timeout=15)
    if code != 0:
        if "nothing to commit" in out.lower() or "nothing to commit" in err.lower():
            return
        log(f"git commit failed: {err}")
        return

    log(f"Committed: {msg}")
    push_with_retry()


def main():
    print("=" * 60, flush=True)
    print(" GIT AUTO-SYNC DAEMON — 24/7 ACTIVE", flush=True)
    print(f" Watching: {BASE_DIR}", flush=True)
    print("=" * 60, flush=True)

    last_changed = set()

    while True:
        try:
            changed = get_changed_files()
            changed_set = set(changed)

            if changed_set and changed_set != last_changed:
                log(f"Changes detected: {changed_set}")
                log(f"Debouncing {DEBOUNCE_SECONDS}s...")
                time.sleep(DEBOUNCE_SECONDS)

                # Re-check after debounce
                changed = get_changed_files()
                if changed:
                    commit_and_push(changed)
                    last_changed = set()
                else:
                    last_changed = set()
            else:
                last_changed = changed_set

        except KeyboardInterrupt:
            log("Git sync stopped by user.")
            sys.exit(0)
        except Exception as e:
            log(f"Unexpected error: {e}")

        time.sleep(WATCH_INTERVAL)


if __name__ == "__main__":
    main()
