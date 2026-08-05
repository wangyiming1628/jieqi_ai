"""
揭棋 AI 主程序 - 状态框检测 + jieqiai 引擎 + 自动走子
"""
import sys, os, time, subprocess, json, cv2, numpy as np
import pyautogui
from board_recognizer import BoardRecognizer

SCAN_INTERVAL = 1.0
MY_BOX_X1, MY_BOX_Y1 = 2708, 1461
MY_BOX_X2, MY_BOX_Y2 = 2867, 1617
CROP_W, CROP_H = (1529, 1695) if sys.platform == "darwin" else (1035, 1143)

PIECE_NAME = {
    "r帥": "帥", "r仕": "仕", "r相": "相", "r馬": "馬", "r車": "車", "r炮": "炮", "r兵": "兵",
    "b將": "將", "b士": "士", "b象": "象", "b馬": "馬", "b車": "車", "b炮": "炮", "b卒": "卒",
    "r?": "?", "b?": "?",
}
RED_ROW_NAMES = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
BLACK_ROW_NAMES = ["十", "九", "八", "七", "六", "五", "四", "三", "二", "一"]
COL_NAMES = ["九", "八", "七", "六", "五", "四", "三", "二", "一"]

FEN_MAP = {
    "r帥": "K", "r仕": "A", "r相": "E", "r馬": "H", "r車": "R", "r炮": "C", "r兵": "P",
    "b將": "k", "b士": "a", "b象": "e", "b馬": "h", "b車": "r", "b炮": "c", "b卒": "p",
    "r?": "X", "b?": "x",
}


def get_box_border_color(img, x1, y1, x2, y2, bw=4):
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    px = []
    if y1 + bw <= y2: px.append(img[y1:y1 + bw, x1:x2].reshape(-1, 3))
    if y2 - bw >= y1: px.append(img[y2 - bw:y2, x1:x2].reshape(-1, 3))
    if x1 + bw <= x2: px.append(img[y1:y2, x1:x1 + bw].reshape(-1, 3))
    if x2 - bw >= x1: px.append(img[y1:y2, x2 - bw:x2].reshape(-1, 3))
    if not px: return (0, 0, 0)
    a = np.vstack(px)
    return (np.mean(a[:, 2]), np.mean(a[:, 1]), np.mean(a[:, 0]))


def is_green(r, g, b):
    return g > r * 1.15 and g > b * 1.15 and g > 80


def is_my_turn(img):
    r, g, b = get_box_border_color(img, MY_BOX_X1, MY_BOX_Y1, MY_BOX_X2, MY_BOX_Y2)
    return is_green(r, g, b)


def crop_board(img):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    return img[cy - CROP_H // 2:cy + CROP_H // 2, cx - CROP_W // 2:cx + CROP_W // 2]


def board_to_fen(board, my_side):
    # 每方棋子不能超过16个
    rc = sum(1 for r in range(10) for c in range(9) if board[r][c].startswith("r"))
    bc = sum(1 for r in range(10) for c in range(9) if board[r][c].startswith("b"))
    for r in range(10):
        for c in range(9):
            if rc > 16 and board[r][c] == "r?": board[r][c] = "."; rc -= 1
            if bc > 16 and board[r][c] == "b?": board[r][c] = "."; bc -= 1
    # 引擎内部：FEN 第一段 → 引擎 rank 9，FEN 最后一段 → 引擎 rank 0
    # 引擎 row 0-4 = 红方半场，row 5-9 = 黑方半场
    # 需要：红方棋子 → 引擎 rank 0-4，黑方棋子 → 引擎 rank 5-9
    # 判断红方在 board 的上半还是下半来决定 FEN 方向
    red_in_bottom = any(board[r][c].startswith("r") for r in range(5, 10) for c in range(9))
    rows = []
    row_range = range(10) if red_in_bottom else range(9, -1, -1)
    for r in row_range:
        s = ""; e = 0
        for c in range(9):
            p = board[r][c]
            if p == ".": e += 1
            else:
                if e: s += str(e); e = 0
                s += FEN_MAP.get(p, p)
        if e: s += str(e)
        rows.append(s)
    return "/".join(rows) + f" -:- {my_side} {my_side}"


def uci_to_human(uci, board, my_side):
    is_reveal = uci.startswith('+')
    u = uci.lstrip('+')
    if len(u) < 4 or not u[1].isdigit() or not u[3].isdigit(): return uci
    fc, engine_fr = ord(u[0]) - 97, int(u[1])
    tc, engine_tr = ord(u[2]) - 97, int(u[3])
    # 引擎 rank → board row：取决于红方在 board 的上半还是下半
    # my_side == "r" 时红方在下半（正序FEN）：board_row = 9 - engine_rank
    # my_side == "b" 时红方在上半（倒序FEN）：board_row = engine_rank
    if my_side == "r":
        fr, tr = 9 - engine_fr, 9 - engine_tr
    else:
        fr, tr = engine_fr, engine_tr
    piece = board[fr][fc]
    pn = PIECE_NAME.get(piece, "?")
    is_hidden = piece.endswith("?")
    rn = RED_ROW_NAMES if my_side == "r" else BLACK_ROW_NAMES
    steps = abs(fr - tr)
    step_names = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九"]

    if fc == tc and fr == tr: act, pos = "平", rn[tr]
    elif fc == tc: act, pos = ("进" if (my_side == "r" and fr > tr) or (my_side == "b" and fr > tr) else "退"), step_names[steps] if steps < len(step_names) else str(steps)
    elif fr == tr: act, pos = "平", COL_NAMES[tc]
    elif (my_side == "r" and fr > tr) or (my_side == "b" and fr > tr): act, pos = "进", COL_NAMES[tc]
    else: act, pos = "退", COL_NAMES[tc]
    pre = "[揭]" if is_reveal else ("[暗]" if is_hidden else "")
    return f"{pre}{pn}{COL_NAMES[fc]}{act}{pos}"


TITLEBAR_OFFSET = 68  # macOS 标题栏在 Retina 2x 下的高度（逻辑 34pt）


def _get_window_position():
    """AppleScript 获取微信/天天象棋窗口的 position（逻辑坐标，包含标题栏）"""
    try:
        r = subprocess.run([
            "osascript", "-e",
            'tell application "System Events" to get position of '
            'first window of process "微信" whose name is "天天象棋"'
        ], capture_output=True, text=True, timeout=3)
        parts = r.stdout.strip().split(",")
        if len(parts) == 2:
            return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        pass
    return None


def _activate_window():
    """AppleScript 激活微信到前台"""
    try:
        subprocess.run([
            "osascript", "-e",
            'tell application "System Events" to set frontmost of process "微信" to true'
        ], timeout=3)
    except Exception:
        pass


def board_pos_to_screen(row, col, win_img_w, win_img_h, win_pos):
    """窗口内容图坐标 → 屏幕绝对逻辑坐标
    win_img: 窗口内容截图（Retina 2x）
    win_pos: (x, y) 窗口 position（逻辑坐标，包含标题栏）
    棋盘在窗口内容图中居中
    """
    win_x, win_y = win_pos
    ox = (win_img_w - CROP_W) // 2
    oy = (win_img_h - CROP_H) // 2
    cw, ch = CROP_W / 9, CROP_H / 10
    wx = ox + col * cw + cw / 2
    wy = oy + row * ch + ch / 2
    # Retina 2x → 逻辑坐标：除以 2
    return ((win_x * 2 + int(wx)) // 2, (win_y * 2 + TITLEBAR_OFFSET + int(wy)) // 2)


def execute_move(uci, recognizer, my_side):
    u = uci.lstrip('+')
    if len(u) < 4 or not u[1].isdigit() or not u[3].isdigit():
        print(f"[!] 无效着法: {uci}"); return
    fc, engine_fr = ord(u[0]) - 97, int(u[1])
    tc, engine_tr = ord(u[2]) - 97, int(u[3])
    # 引擎 rank → board row
    if my_side == "r":
        fr, tr = 9 - engine_fr, 9 - engine_tr
    else:
        fr, tr = engine_fr, engine_tr

    # 1. 获取窗口位置（逻辑坐标）
    win_pos = _get_window_position()
    if win_pos is None:
        print("[!] 获取窗口位置失败"); return
    print(f"[*] 窗口 position: {win_pos}")

    # 2. 激活窗口到前台
    _activate_window()
    time.sleep(0.3)

    # 3. 截窗口内容图（Retina 2x）
    win_img = recognizer._capture._screencapture_raw()
    if win_img is None:
        print("[!] 窗口截图失败"); return
    h, w = win_img.shape[:2]
    print(f"[*] 窗口截图: {w}x{h}")

    # 4. 计算屏幕坐标
    fx, fy = board_pos_to_screen(fr, fc, w, h, win_pos)
    tx, ty = board_pos_to_screen(tr, tc, w, h, win_pos)
    print(f"[*] 走子: ({fr},{fc})→({tr},{tc}) 屏幕: ({fx},{fy})→({tx},{ty})")

    # 5. 走子
    pyautogui.click(fx, fy); time.sleep(0.5)
    pyautogui.click(tx, ty); time.sleep(0.3)
    print(f"[+] 已走子")


def main():
    print("[*] 揭棋 AI v2.0 启动中...")
    recognizer = BoardRecognizer()
    print("[*] 识别器就绪")

    engine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engines", "xiangqi-ai")
    if not os.path.exists(engine_path):
        print("[!] 未找到 jieqiai 引擎"); return
    print(f"[+] 引擎: xiangqi-ai")

    my_side = None
    last_was_green = False
    print(f"[*] 状态框检测启动 (间隔 {SCAN_INTERVAL}s)...\n")

    while True:
        try:
            full_img = recognizer._capture.capture_full()
            if full_img is None: time.sleep(SCAN_INTERVAL); continue

            is_green_now = is_my_turn(full_img)
            if is_green_now and not last_was_green:
                print("\n[+] 轮到我方走棋！等待稳定...")
                time.sleep(0.5)
                full_img = recognizer._capture.capture_full()
                if full_img is None: continue
                board_img = crop_board(full_img)

                sd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshot")
                os.makedirs(sd, exist_ok=True)
                cv2.imencode(".png", board_img)[1].tofile(os.path.join(sd, f"{time.strftime('%H%M%S')}.png"))

                # 先识别（无side），用于判断阵营
                board = recognizer.detect(board_img, my_side=my_side)
                if my_side is None:
                    # 己方半场是 row 5-9（图片下半部分），己方帅/将在己方底线 row 9 附近
                    hs = any(board[r][c] == "r帥" for r in range(5, 10) for c in range(9))
                    hj = any(board[r][c] == "b將" for r in range(5, 10) for c in range(9))
                    my_side = "r" if hs else ("b" if hj else ("b" if any(board[r][c] == "r帥" for r in range(5) for c in range(9)) else "r"))
                    print(f"[+] 己方: {'红' if my_side == 'r' else '黑'}方")
                    # 用正确side重新识别
                    board = recognizer.detect(board_img, my_side=my_side)

                print(recognizer.board_to_string(board))
                fen = board_to_fen(board, my_side)
                print(f"[*] FEN: {fen}")

                r = subprocess.run([engine_path, "best", "--fen", fen, "--strategy", "it2", "--time-limit", "2", "--json"],
                                   capture_output=True, text=True, timeout=30)
                if r.returncode == 0:
                    d = json.loads(r.stdout)
                    if d.get("ok") and d.get("moves"):
                        uci = d["moves"][0]["move"]
                        print(f"[+] 推荐: {uci} → {uci_to_human(uci, board, my_side)} (分数:{d['moves'][0]['score']:.0f} 深度:{d['depth']})")
                        print("[*] 自动走子...")
                        execute_move(uci, recognizer, my_side)
                    else:
                        print("[!] 引擎无着法")
                else:
                    print(f"[!] 引擎错误: {r.stderr}")

                print("-" * 40)

            last_was_green = is_green_now
            time.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            print("\n[*] 停止"); break
        except Exception as e:
            print(f"[!] {e}"); import traceback; traceback.print_exc()
            time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
