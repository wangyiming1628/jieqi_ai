import sys
import time
import random
from typing import Optional, Tuple

if sys.platform == "win32":
    import ctypes
    import pyautogui
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    SetCursorPos = user32.SetCursorPos
    SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
    mouse_event = user32.mouse_event
    mouse_event.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD]

    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004

    def click(x: int, y: int):
        pyautogui.click(x, y)
        time.sleep(0.05)

    def drag(from_x: int, from_y: int, to_x: int, to_y: int, duration: float = 0.15):
        pyautogui.moveTo(from_x, from_y)
        pyautogui.mouseDown()
        steps = max(int(duration / 0.01), 5)
        for i in range(1, steps + 1):
            t = i / steps
            cur_x = int(from_x + (to_x - from_x) * t)
            cur_y = int(from_y + (to_y - from_y) * t)
            pyautogui.moveTo(cur_x, cur_y, duration=0.005)
        time.sleep(0.02)
        pyautogui.mouseUp()
        time.sleep(0.02)

    def find_window(title_contains: str) -> Optional[int]:
        hwnd = user32.FindWindowW(None, None)
        windows = []
        def enum_callback(hwnd, lparam):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                if title_contains.lower() in buf.value.lower():
                    windows.append(hwnd)
            return True
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        user32.EnumWindows(WNDENUMPROC(enum_callback), 0)
        return windows[0] if windows else None

    def get_window_rect(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        rect = wintypes.RECT()
        if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return (rect.left, rect.top, rect.right, rect.bottom)
        return None

    def capture_window(hwnd: int) -> Optional[bytes]:
        import mss
        rect = get_window_rect(hwnd)
        if rect is None:
            return None
        left, top, right, bottom = rect
        monitor = {"top": top, "left": left, "width": right - left, "height": bottom - top}
        with mss.mss() as sct:
            img = sct.grab(monitor)
            return img

elif sys.platform == "darwin":
    import pyautogui
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGWindowListOptionOnScreenOnly,
        kCGNullWindowID,
    )

    def click(x: int, y: int):
        pyautogui.click(x, y)
        time.sleep(0.05)

    def drag(from_x: int, from_y: int, to_x: int, to_y: int, duration: float = 0.15):
        pyautogui.moveTo(from_x, from_y)
        pyautogui.mouseDown()
        pyautogui.moveTo(to_x, to_y, duration=duration)
        pyautogui.mouseUp()
        time.sleep(0.02)

    def find_window(title_contains: str) -> Optional[dict]:
        infos = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
        for info in infos:
            name = info.get("kCGWindowName", "")
            if title_contains.lower() in name.lower():
                return info
        return None

    def get_window_rect(info: dict) -> Optional[Tuple[int, int, int, int]]:
        bounds = info.get("kCGWindowBounds", {})
        x = int(bounds.get("X", 0))
        y = int(bounds.get("Y", 0))
        w = int(bounds.get("Width", 0))
        h = int(bounds.get("Height", 0))
        return (x, y, x + w, y + h)

    def capture_window(info: dict) -> Optional[bytes]:
        import mss
        rect = get_window_rect(info)
        if rect is None:
            return None
        left, top, right, bottom = rect
        monitor = {"top": top, "left": left, "width": right - left, "height": bottom - top}
        with mss.mss() as sct:
            img = sct.grab(monitor)
            return img

else:
    raise OSError("不支持的操作系统")


class Controller:
    """自动操控 — 连接窗口 + 点击/拖拽走子"""

    def __init__(self, window_title: str = "天天象棋"):
        self.window_title = window_title
        self.hwnd = None
        self.window_info = None
        self.rect = None
        self.board_offset = (0, 0)
        self.cell_size = 0

    def connect(self) -> bool:
        if sys.platform == "win32":
            self.hwnd = find_window(self.window_title)
            if self.hwnd is None:
                return False
            self.rect = get_window_rect(self.hwnd)
        else:
            self.window_info = find_window(self.window_title)
            if self.window_info is None:
                return False
            self.rect = get_window_rect(self.window_info)
        if self.rect is None:
            return False
        return True

    def capture(self) -> Optional[bytes]:
        if sys.platform == "win32":
            return capture_window(self.hwnd)
        else:
            return capture_window(self.window_info)

    def set_board_offset(self, x: int, y: int, cell_size: int):
        self.board_offset = (x, y)
        self.cell_size = cell_size

    def cell_to_screen(self, row: int, col: int) -> Tuple[int, int]:
        bx, by = self.board_offset
        if self.rect is None:
            return (0, 0)
        sx = self.rect[0] + bx + col * self.cell_size + self.cell_size // 2
        sy = self.rect[1] + by + row * self.cell_size + self.cell_size // 2
        return (sx, sy)

    def make_move(self, from_row: int, from_col: int, to_row: int, to_col: int):
        fx, fy = self.cell_to_screen(from_row, from_col)
        tx, ty = self.cell_to_screen(to_row, to_col)
        drag(fx, fy, tx, ty)
        time.sleep(0.3 + random.uniform(0.05, 0.25))

    def reconnect(self):
        self.connect()
