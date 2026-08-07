import asyncio
import os
import time
import json
from datetime import datetime
from playwright.async_api import async_playwright

STATE_FILE = "relay_state.json"
RESPONSE_FILE = "arena_latest_copied_response.txt"
SEND_FILE = "send_to_arena.txt"
LOG_FILE = "arena_history_stream.log"

def set_state(status: str, extra: dict = None):
    data = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
    }
    if extra:
        data.update(extra)
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[State Error] {e}")

async def daemon():
    set_state("IDLE")
    async with async_playwright() as pw:
        try:
            print("Connecting to Chrome on port 19222...")
            browser = await pw.chromium.connect_over_cdp('http://localhost:19222')
            target_page = None
            for ctx in browser.contexts:
                for page in ctx.pages:
                    if 'arena' in page.url.lower():
                        target_page = page
                        break
                if target_page:
                    break
            
            if not target_page:
                print("Could not find Arena tab. Exiting.")
                return

            print("Connected to Arena tab! Closed-Loop Relay Daemon Active...")

            while True:
                # 1. Auto-click 'Yes' modal if present
                try:
                    yes_btn = target_page.get_by_role("button", name="Yes")
                    if await yes_btn.is_visible():
                        print("Modal detected! Auto-clicking 'Yes'...")
                        await yes_btn.click()
                        await asyncio.sleep(0.5)
                except Exception:
                    pass

                # 2. Scroll all scrollable elements to bottom to keep latest text in view
                try:
                    await target_page.evaluate("""() => {
                        window.scrollTo(0, document.body.scrollHeight);
                        const scrollables = document.querySelectorAll('*');
                        scrollables.forEach(el => {
                            if (el.scrollHeight > el.clientHeight && el.clientHeight > 0) {
                                el.scrollTop = el.scrollHeight;
                            }
                        });
                    }""")
                except Exception:
                    pass

                # 3. Check for prompt dispatch from Antigravity
                if os.path.exists(SEND_FILE):
                    try:
                        with open(SEND_FILE, "r", encoding="utf-8") as f:
                            msg = f.read().strip()
                        if msg:
                            set_state("GENERATING", {"prompt_snippet": msg[:100]})
                            print(f"[RELAY] Sending prompt to Arena.ai: {msg[:40]}...")
                            chat_input = target_page.locator('[contenteditable="true"]').first
                            await chat_input.fill(msg)
                            await asyncio.sleep(0.5)
                            await target_page.keyboard.press("Enter")
                            os.remove(SEND_FILE)
                            await asyncio.sleep(5)  # Wait for generation to start
                    except Exception as ex:
                        print(f"[RELAY ERROR] {ex}")

                # 4. Click response copy button to extract clean markdown
                try:
                    buttons = await target_page.locator('button').all()
                    for b in reversed(buttons):
                        html = await b.inner_html()
                        if 'copy' in html.lower() or 'M8' in html or 'M16' in html or 'M19' in html:
                            if await b.is_visible():
                                await b.click()
                                await asyncio.sleep(0.3)
                                clip_text = await target_page.evaluate("navigator.clipboard.readText()")
                                if clip_text and len(clip_text) > 100:
                                    with open(RESPONSE_FILE, "w", encoding="utf-8") as f:
                                        f.write(clip_text)
                                    set_state("RESPONSE_READY", {"length": len(clip_text)})
                                break
                except Exception:
                    pass

                await asyncio.sleep(2)

        except Exception as e:
            print(f"Daemon error: {e}")

if __name__ == "__main__":
    asyncio.run(daemon())
