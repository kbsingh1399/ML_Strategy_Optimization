from PIL import ImageGrab
import os

out_path = r"C:\Users\SIGMA\.gemini\antigravity\brain\11e71ddd-8b3c-47ec-8aa3-8505db9c824f\desktop_terminal_screenshot.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
try:
    img = ImageGrab.grab()
    img.save(out_path)
    print("Screenshot saved to:", out_path)
except Exception as e:
    print("Screenshot error:", e)
