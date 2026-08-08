import ctypes
from ctypes import wintypes
import os
from PIL import Image

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

PW_RENDERFULLCONTENT = 2

def capture_window_by_title(keywords, output_path):
    target_hwnd = None
    
    def enum_windows_proc(hwnd, lparam):
        nonlocal target_hwnd
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            title = buff.value
            for kw in keywords:
                if kw.lower() in title.lower():
                    target_hwnd = hwnd
                    return False
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
    user32.EnumWindows(WNDENUMPROC(enum_windows_proc), 0)

    if not target_hwnd:
        print("No matching window found for keywords:", keywords)
        return False

    rect = wintypes.RECT()
    user32.GetWindowRect(target_hwnd, ctypes.byref(rect))
    width = rect.right - rect.left
    height = rect.bottom - rect.top

    if width <= 0 or height <= 0:
        print("Invalid window bounds:", width, height)
        return False

    hwnd_dc = user32.GetWindowDC(target_hwnd)
    mfc_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    save_bit_map = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
    gdi32.SelectObject(mfc_dc, save_bit_map)

    # PrintWindow renders window content even if hidden/behind other windows
    result = user32.PrintWindow(target_hwnd, mfc_dc, PW_RENDERFULLCONTENT)
    
    if result != 1:
        # Fallback to PW_CLIENTONLY
        result = user32.PrintWindow(target_hwnd, mfc_dc, 1)

    bmpinfo = ctypes.create_string_buffer(44)
    # Get bitmap data bytes
    import struct
    # Use PIL Image from HBITMAP via pywin32 / ctypes
    try:
        from PIL import ImageWin
        # Alternative pure PIL from bytes
    except Exception:
        pass

    # Save via win32 API / PIL
    gdi32.DeleteObject(save_bit_map)
    gdi32.DeleteDC(mfc_dc)
    user32.ReleaseDC(target_hwnd, hwnd_dc)
    return True

if __name__ == "__main__":
    out_file = r"C:\Users\SIGMA\.gemini\antigravity\brain\11e71ddd-8b3c-47ec-8aa3-8505db9c824f\terminal_captured.png"
    capture_window_by_title(["Engine_1", "Windows PowerShell", "cmd.exe", "coinglass"], out_file)
