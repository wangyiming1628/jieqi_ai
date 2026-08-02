"""
揭棋 AI 主程序 - YOLO+OCR 识别 + 引擎对抗
"""
import sys, os, time, cv2, numpy as np
from collections import Counter
from controller import Controller
from board_recognizer import BoardRecognizer
from engine import UCCIEngine
from game_state import GameState, BOARD_ROWS, BOARD_COLS

WINDOW_TITLE = "天天象棋"
SCAN_INTERVAL = 0.8
BOARD_OFFSET_X = 13
BOARD_OFFSET_Y = 6

def main():
    print("[*] 揭棋 AI v1.3 启动中...")
    
    controller = Controller(WINDOW_TITLE)
    if not controller.connect():
        print(f"[!] 未找到窗口: {WINDOW_TITLE}")
        return

    print(f"[+] 已连接: {WINDOW_TITLE}")
    
    print("[*] 初始化识别器...")
    recognizer = BoardRecognizer()
    print("[*] 识别器就绪")
    engine = None
    engine_path = UCCIEngine.default_engine_path()
    if engine_path:
        engine = UCCIEngine(engine_path)
        print(f"[+] 引擎: {engine.name}")
    else:
        print("[!] 未找到引擎, 仅识别模式")

    my_side = None
    last_board = None
    last_hash = ""
    stable_count = 0

    print("[*] 监控中...\n")

    while True:
        try:
            img = controller.capture()
            if img is None:
                time.sleep(SCAN_INTERVAL)
                continue

            h, w = img.shape[:2]
            cx, cy = w // 2, h // 2
            x1 = max(0, cx - 1035 // 2)
            y1 = max(0, cy - 1143 // 2)
            x2 = min(w, cx + 1035 // 2)
            y2 = min(h, cy + 1143 // 2)
            img = img[y1:y2, x1:x2]

            snapshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshot")
            os.makedirs(snapshot_dir, exist_ok=True)
            ts = time.strftime("%H%M%S")
            cv2.imencode(".png", img)[1].tofile(os.path.join(snapshot_dir, f"{ts}.png"))

            h, w = img.shape[:2]
            cx, cy = w // 2, h // 2
            x1 = max(0, cx - 1035 // 2)
            y1 = max(0, cy - 1143 // 2)
            x2 = min(w, cx + 1035 // 2)
            y2 = min(h, cy + 1143 // 2)
            img = img[y1:y2, x1:x2]

            board = recognizer.detect(img)
            board_hash = "|".join("".join(row) for row in board)

            if my_side is None:
                for r in range(5, 10):
                    for c in range(9):
                        if board[r][c] in ("r帅","r帥"):
                            my_side = "r"; break
                    if my_side: break
                if my_side is None: my_side = "b"
                print(f"[+] 己方: {'红' if my_side=='r' else '黑'}方")

            if board_hash != last_hash:
                print("\n" + recognizer.board_to_string(board))
            last_hash = board_hash

        except KeyboardInterrupt:
            print("\n[*] 停止")
            break
        except Exception as e:
            print(f"[!] {e}")
            import traceback; traceback.print_exc()
            time.sleep(SCAN_INTERVAL)

    if engine:
        engine.quit()

def find_diff(prev, curr):
    changes = []
    for r in range(10):
        for c in range(9):
            if prev[r][c] != curr[r][c]:
                changes.append(f"({r},{c}): {prev[r][c]} -> {curr[r][c]}")
    return ", ".join(changes[:3]) if changes else None

if __name__ == "__main__":
    main()
