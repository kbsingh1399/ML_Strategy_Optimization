"""
UNIFIED 24/7 AUTONOMOUS RELAY LOOP — SINGLE PROCESS ARCHITECTURE
Conforming 100% to Master 7-Step Spec + Full Visual Screenshots & Gallery

1. Writes improvement prompt & pushes live code to GitHub.
2. Types prompt into Arena.ai editor (div.tiptap.ProseMirror).
3. Clicks Send button (button.rounded-full.p-0).
4. Waits for Copy button / response completion signal + 10s streaming screenshots.
5. Auto-dismisses feedback popups ("Yes").
6. Dumps response to disk -> extracts code patches.
7. Verifies patches (py_compile + 8s local smoke test).
8. Commits & Pushes changes to GitHub.
9. Loops 24/7.
"""
import asyncio
import os
import re
import sys
import json
import time
import shutil
import subprocess
from datetime import datetime
from playwright.async_api import async_playwright

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
RESPONSE_FILE   = os.path.join(BASE_DIR, "arena_latest_copied_response.txt")
LOG_FILE        = os.path.join(BASE_DIR, "arena_history_stream.log")
CYCLE_LOG       = os.path.join(BASE_DIR, "relay_cycle_log.jsonl")
RECORDINGS_DIR  = os.path.join(BASE_DIR, "recordings")
SHOTS_DIR       = os.path.join(RECORDINGS_DIR, "screenshots")
TRACES_DIR      = os.path.join(RECORDINGS_DIR, "traces")
GALLERY_FILE    = os.path.join(RECORDINGS_DIR, "index.html")
BACKUP_DIR      = os.path.join(BASE_DIR, "patch_backups")

os.makedirs(SHOTS_DIR, exist_ok=True)
os.makedirs(TRACES_DIR, exist_ok=True)

CDP_URL = "http://localhost:19022"
GITHUB_REPO = "https://github.com/kbsingh1399/ML_Strategy_Optimization"

CORE_FILES = [
    "Engine_1.py", "binance_broker.py", "live_model_trainer.py",
    "coinglass_scraper.py", "ensemble_strategy_predictor.py",
    "mt5_broker.py", "run_all_6.py"
]

IMPROVEMENT_TOPICS = [
    {
        "id": "signal_refinement",
        "title": "Signal Refinement & Alpha Generation",
        "prompt": (
            "Review Engine_1.py trading signals (S1, S2, S3, S4, S5). Suggest:\n"
            "1. Improved entry/exit filters to reduce false signals during choppy regimes.\n"
            "2. Dynamic ATR volatility-based thresholds.\n"
            "3. Order flow & CVD divergence confluence filters.\n\n"
            "Provide improved Python code with ```python ... ``` fencing. Label target file at top: # TARGET: Engine_1.py"
        ),
    },
    {
        "id": "risk_management",
        "title": "Dynamic Risk Management",
        "prompt": (
            "Review Engine_1.py risk/position sizing logic. Suggest:\n"
            "1. ATR-based dynamic stop loss tightening during high volatility.\n"
            "2. Max drawdown circuit breaker — pause trading if DD > 5% in 1 hour.\n"
            "3. Position scaling: reduce size on consecutive losses (anti-martingale).\n\n"
            "Provide improved Python code with ```python ... ``` fencing. Label target file at top: # TARGET: Engine_1.py"
        ),
    },
    {
        "id": "fee_optimization",
        "title": "Fee & Execution Cost Optimization",
        "prompt": (
            "Review binance_broker.py and Engine_1.py trade execution. Suggest:\n"
            "1. Post-only limit orders where feasible to earn maker rebates.\n"
            "2. Minimum profit target filter to ensure trade expected value > 2x round-trip fee + slippage.\n"
            "3. Order splitting for large sizes to minimize market impact.\n\n"
            "Provide improved Python code with ```python ... ``` fencing. Label target file at top: # TARGET: binance_broker.py"
        ),
    },
    {
        "id": "order_flow",
        "title": "Order Flow & Microstructure",
        "prompt": (
            "Review Coinglass / footprint data ingestion and signal generation. Suggest:\n"
            "1. Aggregated CVD (Cumulative Volume Delta) imbalance ratio filter.\n"
            "2. Open Interest delta confluence: trip trades only when OI increases alongside price move.\n"
            "3. Liquidation cascade detector: pause shorting into massive long liquidation spikes.\n\n"
            "Provide improved Python code with ```python ... ``` fencing. Label target file at top: # TARGET: Engine_1.py"
        ),
    },
    {
        "id": "ml_tuning",
        "title": "ML Model Hyperparameter Tuning",
        "prompt": (
            "Review live_model_trainer.py and ensemble_strategy_predictor.py. Suggest:\n"
            "1. Better feature engineering for the ML models (log features, rolling stats).\n"
            "2. Ensemble weighting: dynamic weight based on recent model accuracy.\n"
            "3. Online learning: incremental model updates on new candles.\n\n"
            "Provide improved Python code with ```python ... ``` fencing. Label target file: # TARGET: live_model_trainer.py or ensemble_strategy_predictor.py"
        ),
    },
    {
        "id": "latency_optimization",
        "title": "Latency & Event Loop Optimization",
        "prompt": (
            "Review Engine_1.py async loop and ticker processing. Suggest:\n"
            "1. Non-blocking WebSocket tick parsing.\n"
            "2. In-memory circular buffer for 1200-bar candle history (numpy/collections.deque).\n"
            "3. Fast path execution for stop-loss checks.\n\n"
            "Provide improved Python code with ```python ... ``` fencing. Label target file at top: # TARGET: Engine_1.py"
        ),
    },
    {
        "id": "backtest_accuracy",
        "title": "Backtest Accuracy & Walk-Forward Validation",
        "prompt": (
            "Review backtesting and simulation logic. Suggest:\n"
            "1. Realistic fill simulation with slippage model based on ATR/volume.\n"
            "2. Purge/embargo gaps between train and test windows to prevent data leakage.\n"
            "3. Out-of-sample Sharpe and Calmar ratio reporting per window.\n\n"
            "Provide improved Python code with ```python ... ``` fencing. Label target file at top: # TARGET: Engine_1.py"
        ),
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [RELAY] {msg}"
    try:
        print(line.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding), flush=True)
    except Exception:
        print(line.encode('ascii', errors='replace').decode('ascii'), flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


async def wait_for_arena_ready(page) -> bool:
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=15000)
        for _ in range(15):
            ready = await page.evaluate("""() => {
                const ed = document.querySelector('div.tiptap.ProseMirror')
                        || document.querySelector('div.tiptap')
                        || document.querySelector('[contenteditable="true"]');
                return ed !== null && ed.getBoundingClientRect().height > 0;
            }""")
            if ready:
                return True
            await asyncio.sleep(1)
    except Exception as e:
        log(f"Page load wait note: {e}")
    return False


def git_push(msg_label: str = "sync"):
    try:
        subprocess.run(["git", "add", "-A"], cwd=BASE_DIR, capture_output=True, timeout=15)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=BASE_DIR, timeout=10)
        if diff.returncode != 0:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = f"Auto-sync [{msg_label}] @ {ts}"
            subprocess.run(["git", "commit", "-m", msg], cwd=BASE_DIR, capture_output=True, timeout=15)
            log(f"Committed: {msg}")
        subprocess.run(["git", "push", "origin", "arena-seeding-fix"], cwd=BASE_DIR, capture_output=True, timeout=30)
        r = subprocess.run(["git", "push", "ml_strat", "HEAD:main"], cwd=BASE_DIR, capture_output=True, timeout=30)
        subprocess.run(["git", "push", "ml_strat", "HEAD:arena-seeding-fix"], cwd=BASE_DIR, capture_output=True, timeout=30)
        if r.returncode == 0:
            log(f"Git push OK — Live code updated at {GITHUB_REPO}")
    except Exception as e:
        log(f"git_push error: {e}")


def update_visual_gallery(shot_filename: str, label: str, details: str = ""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rel_path = f"screenshots/{shot_filename}"

    card_html = f"""
    <div class="card">
        <div class="card-header">
            <span class="badge">{label}</span>
            <span class="time">{ts}</span>
        </div>
        <div class="card-details">{details}</div>
        <a href="{rel_path}" target="_blank">
            <img src="{rel_path}" alt="{label}" loading="lazy" />
        </a>
    </div>
"""
    try:
        if not os.path.exists(GALLERY_FILE):
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Arena 24/7 Loop — Visual Recording Gallery</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 20px; }}
        h1 {{ color: #38bdf8; border-bottom: 2px solid #334155; padding-bottom: 10px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 20px; margin-top: 20px; }}
        .card {{ background: #1e293b; border-radius: 8px; border: 1px solid #334155; overflow: hidden; padding: 12px; }}
        .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
        .badge {{ background: #0284c7; color: #fff; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; }}
        .time {{ color: #94a3b8; font-size: 0.8em; }}
        .card-details {{ font-size: 0.9em; color: #cbd5e1; margin-bottom: 10px; word-break: break-word; }}
        img {{ width: 100%; border-radius: 6px; border: 1px solid #475569; transition: transform 0.2s; }}
        img:hover {{ transform: scale(1.02); }}
    </style>
</head>
<body>
    <h1>🎬 Arena 24/7 Loop — Visual Recording Gallery</h1>
    <p>Live visual recording ledger capturing every prompt, streaming response, and state transition.</p>
    <div class="grid" id="gallery">
        {card_html}
    </div>
</body>
</html>"""
            with open(GALLERY_FILE, "w", encoding="utf-8") as f:
                f.write(html_content)
        else:
            with open(GALLERY_FILE, "r", encoding="utf-8") as f:
                content = f.read()
            marker = '<div class="grid" id="gallery">'
            if marker in content:
                parts = content.split(marker, 1)
                new_content = parts[0] + marker + card_html + parts[1]
                with open(GALLERY_FILE, "w", encoding="utf-8") as f:
                    f.write(new_content)
    except Exception as e:
        log(f"Gallery update error: {e}")


async def capture_visual(page, step_label: str, details: str = ""):
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts_str}_{step_label}.png"
    filepath = os.path.join(SHOTS_DIR, filename)
    try:
        if page:
            await page.screenshot(path=filepath, full_page=False)
            update_visual_gallery(filename, step_label, details)
            log(f"Visual captured: [{step_label}] -> {filename}")
    except Exception as e:
        log(f"capture_visual error ({step_label}): {e}")


async def auto_dismiss_modals(page):
    try:
        clicked = await page.evaluate("""() => {
            const btns = [...document.querySelectorAll('button')];
            for (const b of btns) {
                const txt = (b.innerText || b.textContent || '').trim();
                if (txt === 'Yes' || txt.startsWith('Yes')) {
                    b.click();
                    return 'Yes';
                }
            }
            return null;
        }""")
        if clicked:
            log(f"Auto-dismissed feedback popup: '{clicked}'")
            # Scroll down fully and attempt clicking copy button
            await page.evaluate("""() => {
                const proseBlocks = [...document.querySelectorAll('div.prose')].filter(d => !d.className.includes('tiptap'));
                if (proseBlocks.length > 0) {
                    const lastBlock = proseBlocks[proseBlocks.length - 1];
                    lastBlock.scrollIntoView({ behavior: 'auto', block: 'end' });
                    let el = lastBlock;
                    while (el && el !== document.body) {
                        if (el.scrollHeight > el.clientHeight && el.clientHeight > 0) {
                            el.scrollTop = el.scrollHeight;
                        }
                        el = el.parentElement;
                    }
                }
                window.scrollTo(0, document.body.scrollHeight);
                if (document.documentElement) document.documentElement.scrollTop = document.documentElement.scrollHeight;

                const copyBtns = [...document.querySelectorAll('button[aria-label="Copy"], button[title="Copy"]')];
                if (copyBtns.length === 0) {
                    const altBtns = [...document.querySelectorAll('button')].filter(b => (b.className||'').includes('hover:bg-accent'));
                    copyBtns.push(...altBtns);
                }
                if (copyBtns.length > 0) {
                    const lastCopy = copyBtns[copyBtns.length - 1];
                    lastCopy.click();
                }
            }""")
            await capture_visual(page, "MODAL_DISMISSED_SCROLLED_COPIED", f"Clicked '{clicked}', scrolled down, and hit Copy button")
    except Exception:
        pass


async def send_prompt(page, msg: str) -> bool:
    try:
        await page.keyboard.press("Escape")
        await asyncio.sleep(0.4)

        focused = await page.evaluate("""() => {
            const editor = document.querySelector('div.tiptap.ProseMirror')
                        || document.querySelector('div.editor-content')
                        || document.querySelector('div.tiptap')
                        || document.querySelector('p.is-editor-empty')
                        || document.querySelector('[contenteditable="true"]');
            if (!editor) return false;
            editor.scrollIntoView({ behavior: 'auto', block: 'center' });
            editor.focus();
            editor.click();
            return true;
        }""")

        if not focused:
            log("Editor box not found.")
            return False

        await asyncio.sleep(0.3)
        # Use insert_text to safely insert multiline text without newlines (\n) triggering premature form submission
        await page.keyboard.insert_text(msg)
        await asyncio.sleep(1)

        clicked = await page.evaluate("""() => {
            const btn = document.querySelector('button.rounded-full.p-0')
                     || document.querySelector('button.rounded-full')
                     || document.querySelector('button[class*="rounded-full"]');
            if (btn) { btn.click(); return true; }
            return false;
        }""")
        if not clicked:
            await page.keyboard.press("Enter")

        return True
    except Exception as e:
        log(f"send_prompt error: {e}")
        return False


async def has_copy_button(page) -> bool:
    try:
        return bool(await page.evaluate("""() => {
            const copyBtns = [...document.querySelectorAll('button[aria-label="Copy"], button[title="Copy"]')];
            if (copyBtns.length === 0) {
                const altBtns = [...document.querySelectorAll('button')].filter(b => (b.className||'').includes('hover:bg-accent'));
                copyBtns.push(...altBtns);
            }
            return copyBtns.length > 0;
        }"""))
    except Exception:
        return False


async def is_generating(page) -> bool:
    try:
        return bool(await page.evaluate("""() => {
            const stopBtn = [...document.querySelectorAll('button')].find(b => {
                const title = (b.getAttribute('title') || '').toLowerCase();
                const aria = (b.getAttribute('aria-label') || '').toLowerCase();
                return title.includes('stop') || aria.includes('stop');
            });
            return !!stopBtn;
        }"""))
    except Exception:
        return False


async def get_response_text(page) -> str:
    try:
        text = await page.evaluate(r"""() => {
            const proseBlocks = [...document.querySelectorAll('div.prose')].filter(d => !d.className.includes('tiptap'));
            if (proseBlocks.length === 0) return '';
            const lastBlock = proseBlocks[proseBlocks.length - 1];

            // Deep scroll into view across all parent overflow containers
            lastBlock.scrollIntoView({ behavior: 'auto', block: 'end' });
            let el = lastBlock;
            while (el && el !== document.body) {
                if (el.scrollHeight > el.clientHeight && el.clientHeight > 0) {
                    el.scrollTop = el.scrollHeight;
                }
                el = el.parentElement;
            }
            window.scrollTo(0, document.body.scrollHeight);
            if (document.documentElement) document.documentElement.scrollTop = document.documentElement.scrollHeight;

            const clone = lastBlock.cloneNode(true);
            const preTags = clone.querySelectorAll('pre');
            preTags.forEach(pre => {
                const codeText = pre.innerText || pre.textContent || '';
                const codeNode = document.createTextNode('\n```python\n' + codeText + '\n```\n');
                pre.parentNode.replaceChild(codeNode, pre);
            });

            return (clone.innerText || clone.textContent || '').trim();
        }""")
        return text or ""
    except Exception:
        return ""


def extract_code_blocks(text: str) -> list:
    results = []
    pattern = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
    known_files = {f.lower(): f for f in CORE_FILES}

    for match in pattern.finditer(text):
        block = match.group(1).strip()
        if not block: continue
        target = None
        for line in block.splitlines()[:5]:
            stripped = line.strip()
            m = re.search(r"TARGET:\s*(\S+\.py)", stripped, re.IGNORECASE)
            if m: target = m.group(1); break
            m = re.search(r"(\w[\w_]*\.py)\b", stripped, re.IGNORECASE)
            if m and m.group(1).lower() in known_files:
                target = known_files[m.group(1).lower()]; break

        if target and block:
            results.append({"file": target, "code": block})

    return results


def run_test_suite() -> tuple:
    existing = [f for f in CORE_FILES if os.path.exists(os.path.join(BASE_DIR, f))]
    try:
        r = subprocess.run([sys.executable, "-m", "py_compile"] + existing, capture_output=True, text=True, timeout=30, cwd=BASE_DIR)
        if r.returncode != 0: return False, f"SYNTAX FAIL:\n{r.stderr}"
    except Exception as e: return False, f"py_compile error: {e}"

    engine_path = os.path.join(BASE_DIR, "Engine_1.py")
    if os.path.exists(engine_path):
        try:
            proc = subprocess.Popen([sys.executable, "-u", engine_path, "--test"], cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            try:
                out, _ = proc.communicate(timeout=8)
                if proc.returncode not in (0, None): return False, f"Engine crashed:\n{out[-300:]}"
            except subprocess.TimeoutExpired:
                proc.kill()
                log("Local 8s execution test PASS — engine ran without crash.")
        except Exception as e:
            return False, f"Execution test error: {e}"

    return True, "All tests PASS"


async def run_unified_loop():
    log("======================================================================")
    log(" UNIFIED 24/7 AUTONOMOUS RELAY LOOP — SINGLE PROCESS ACTIVE")
    log("======================================================================")

    topic_idx = 0
    cycle = 0

    while True:
        cycle += 1
        topic = IMPROVEMENT_TOPICS[topic_idx % len(IMPROVEMENT_TOPICS)]
        topic_idx += 1

        log(f"\n--- CYCLE {cycle}: {topic['title']} ---")

        # Step 2: Build prompt (check send_to_arena.txt override first)
        SEND_FILE = os.path.join(BASE_DIR, "send_to_arena.txt")
        if os.path.exists(SEND_FILE):
            try:
                with open(SEND_FILE, "r", encoding="utf-8") as f:
                    custom_p = f.read().strip()
                if custom_p:
                    prompt_text = custom_p
                    log(f"Using dynamic prompt from send_to_arena.txt ({len(prompt_text)} chars)")
                else:
                    prompt_text = (
                        f"## CONTEXT\n"
                        f"GitHub Repo: {GITHUB_REPO}\n"
                        f"Branch: master\n"
                        f"All files are freshly pushed — you can reference the latest code directly.\n\n"
                        f"# ENGINE_1 AUTONOMOUS IMPROVEMENT CYCLE — {topic['title']}\n\n"
                        f"{topic['prompt']}"
                    )
            except Exception:
                prompt_text = (
                    f"## CONTEXT\n"
                    f"GitHub Repo: {GITHUB_REPO}\n"
                    f"Branch: master\n"
                    f"All files are freshly pushed — you can reference the latest code directly.\n\n"
                    f"# ENGINE_1 AUTONOMOUS IMPROVEMENT CYCLE — {topic['title']}\n\n"
                    f"{topic['prompt']}"
                )
        else:
            prompt_text = (
                f"## CONTEXT\n"
                f"GitHub Repo: {GITHUB_REPO}\n"
                f"Branch: master\n"
                f"All files are freshly pushed — you can reference the latest code directly.\n\n"
                f"# ENGINE_1 AUTONOMOUS IMPROVEMENT CYCLE — {topic['title']}\n\n"
                f"{topic['prompt']}"
            )

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.connect_over_cdp(CDP_URL)
                page = None
                for ctx in browser.contexts:
                    for p in ctx.pages:
                        if "arena" in p.url.lower():
                            page = p
                            break

                if not page:
                    log("Arena tab not found. Retrying in 10s...")
                    await asyncio.sleep(10)
                    continue

                # Ensure page and editor elements are completely loaded
                await wait_for_arena_ready(page)

                # Step 1: Pre-prompt Git Push to GitHub
                log("Step 1: Pushing latest code to GitHub before prompt...")
                git_push(msg_label=f"Pre-prompt-{topic['id']}")
                await capture_visual(page, "STEP1_GIT_PUSH_OK", f"Pre-prompt sync for {topic['id']}")

                # Step 3: Check if Arena is active or prompt is already in progress
                await auto_dismiss_modals(page)

                if await is_generating(page):
                    log("Arena currently active — waiting for current response to complete before sending new prompt...")
                    while await is_generating(page):
                        await auto_dismiss_modals(page)
                        await asyncio.sleep(5)

                log(f"Step 3: Inserting prompt safely into Arena editor ({len(prompt_text)} chars)...")
                await capture_visual(page, "STEP3_TYPING_PROMPT", f"Typing prompt for {topic['title']}")

                sent = await send_prompt(page, prompt_text)
                if not sent:
                    log("Prompt submission failed. Retrying cycle...")
                    await asyncio.sleep(5)
                    continue

                # Clear send_to_arena.txt to prevent re-submitting the same prompt
                if os.path.exists(SEND_FILE):
                    try:
                        shutil.move(SEND_FILE, SEND_FILE + ".done")
                    except Exception:
                        pass

                await capture_visual(page, "STEP4_SUBMITTED", f"Prompt submitted for {topic['title']}")
                log("Step 4: Prompt submitted (Send Button Hit). Waiting for Copy button...")

                # Step 5: Wait for Copy Button + Full Code Response Completion Signal
                await capture_visual(page, "STEP5_WAITING_RESPONSE", "Waiting for response generation...")
                start_wait = time.time()
                last_tick_10s = time.time()
                copy_btn_captured = False
                stable_text = ""
                last_len = 0
                stable_ticks = 0
                STABLE_TICKS_REQUIRED = 6  # 18s of no text change

                while True:
                    await auto_dismiss_modals(page)
                    generating = await is_generating(page)
                    copy_btn = await has_copy_button(page)
                    curr_text = await get_response_text(page)
                    code_patches = extract_code_blocks(curr_text)
                    has_full_patch = any(len(p.get('code', '')) >= 400 for p in code_patches)

                    if copy_btn and not copy_btn_captured:
                        copy_btn_captured = True
                        await capture_visual(page, "STEP5_COPY_BTN_DETECTED", f"Copy button detected ({len(curr_text)} chars)")

                    now = time.time()
                    if now - last_tick_10s >= 10:
                        last_tick_10s = now
                        await capture_visual(page, "STREAMING_TICK_10s", f"Streaming tick — {len(curr_text)} chars ({len(code_patches)} patches)")

                    # Check for active thinking / streaming
                    is_thinking = "thinking" in curr_text.lower() and len(curr_text) < 1500

                    if generating or is_thinking:
                        stable_ticks = 0
                        log(f"Arena streaming/thinking... ({len(curr_text)} chars so far)")
                    elif (len(curr_text) >= 2000 or has_full_patch) and copy_btn and len(curr_text) == last_len:
                        stable_ticks += 1
                        log(f"Response stable tick {stable_ticks}/{STABLE_TICKS_REQUIRED} — {len(curr_text)} chars ({len(code_patches)} patches)")
                    else:
                        stable_ticks = 0

                    last_len = len(curr_text)

                    # Completion gate: must have stable ticks AND (has_full_patch OR len >= 2000 OR 5min timeout)
                    if stable_ticks >= STABLE_TICKS_REQUIRED and (has_full_patch or len(curr_text) >= 2000 or (time.time() - start_wait) > 300):
                        stable_text = curr_text
                        break

                    await asyncio.sleep(3)

                log(f"Step 5 COMPLETE: Response captured ({len(stable_text)} chars).")

                # Step 6: Write to local txt file + Parse & Apply Patches + Push to GitHub
                with open(RESPONSE_FILE, "w", encoding="utf-8") as f:
                    f.write(stable_text)

                patches = extract_code_blocks(stable_text)
                log(f"Step 6: Extracted {len(patches)} code patch candidate(s).")

                if patches:
                    backups = []
                    any_applied = False
                    for patch in patches:
                        target = os.path.join(BASE_DIR, patch["file"])
                        if os.path.exists(target):
                            with open(target, "r", encoding="utf-8") as f:
                                orig_content = f.read()

                            if len(patch["code"]) >= 0.8 * len(orig_content):
                                bak = f"{target}.bak.{datetime.now().strftime('%H%M%S')}"
                                shutil.copy2(target, bak)
                                backups.append((target, bak))
                                with open(target, "w", encoding="utf-8") as f:
                                    f.write(patch["code"])
                                any_applied = True
                                log(f"Applied full-file patch to {patch['file']}")
                            else:
                                pending_dir = os.path.join(BASE_DIR, "pending_patches")
                                os.makedirs(pending_dir, exist_ok=True)
                                patch_name = f"{os.path.basename(target)}_patch_{datetime.now().strftime('%H%M%S')}.py"
                                patch_path = os.path.join(pending_dir, patch_name)
                                with open(patch_path, "w", encoding="utf-8") as f:
                                    f.write(patch["code"])
                                log(f"Saved partial patch for {patch['file']} to {patch_path}")

                    if any_applied:
                        await capture_visual(page, "STEP6_PATCHES_APPLIED", f"Applied {len(backups)} full-file patch(es)")
                    else:
                        await capture_visual(page, "STEP6_PATCHES_SAVED", "Saved partial patch(es) cleanly to pending_patches/")

                    # Test suite
                    passed, report = run_test_suite()
                    log(f"Verification suite: {'PASS' if passed else 'FAIL'}")

                    if passed:
                        await capture_visual(page, "STEP6_TESTS_PASSED", "Verification test suite PASSED")
                        # Commit & Push to GitHub
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
                        git_push(msg_label=f"Cycle-{cycle}-{topic['id']}")
                        log("Step 6 COMPLETE: Changes committed and pushed to GitHub.")
                    else:
                        log(f"Tests failed — rolling back patches:\n{report[:300]}")
                        for orig, bak in backups:
                            if os.path.exists(bak):
                                shutil.copy2(bak, orig)

                else:
                    log("Step 6: No code patches in response. Pushing sync commit...")
                    git_push(msg_label=f"Cycle-{cycle}-discussion")

                # Step 7: Completed. Next topic in loop...
                log(f"Step 7 COMPLETE: Cycle {cycle} finished. Starting next topic...\n")
                await capture_visual(page, "STEP7_CYCLE_COMPLETE", f"Cycle {cycle} complete for topic '{topic['id']}'")
                await asyncio.sleep(5)

                if "--single-run" in sys.argv:
                    log("Single run mode active. Terminating loop.")
                    return

        except Exception as e:
            log(f"Cycle exception: {e}")
            if "--single-run" in sys.argv:
                log("Single run mode active (with exception). Terminating.")
                return


if __name__ == "__main__":
    asyncio.run(run_unified_loop())
