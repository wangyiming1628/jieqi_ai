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
# 标准开局每个位置对应的棋子名
# 己方半场（row 5-9）：己方底线 row9=車馬相仕帥仕相馬車, row7=炮, row6=兵/卒
# 对方半场（row 0-4）：对方底线 row0=車馬象士將士象馬車, row2=炮, row3=兵/卒
_STD_SELF = {
    9: ["車", "馬", "相", "仕", "帥", "仕", "相", "馬", "車"],
    7: {1: "炮", 7: "炮"},
    6: {0: "兵", 2: "兵", 4: "兵", 6: "兵", 8: "兵"},
}
_STD_SELF_BLACK = {
    9: ["車", "馬", "象", "士", "將", "士", "象", "馬", "車"],
    7: {1: "炮", 7: "炮"},
    6: {0: "卒", 2: "卒", 4: "卒", 6: "卒", 8: "卒"},
}
_STD_OPPO = {
    0: ["車", "馬", "象", "士", "將", "士", "象", "馬", "車"],
    2: {1: "炮", 7: "炮"},
    3: {0: "卒", 2: "卒", 4: "卒", 6: "卒", 8: "卒"},
}
_STD_OPPO_RED = {
    0: ["車", "馬", "相", "仕", "帥", "仕", "相", "馬", "車"],
    2: {1: "炮", 7: "炮"},
    3: {0: "兵", 2: "兵", 4: "兵", 6: "兵", 8: "兵"},
}

def _get_hidden_name(row, col, my_side):
    """根据棋盘位置获取暗子的标准棋子名"""
    if row >= 5:  # 己方半场
        std = _STD_SELF if my_side == "r" else _STD_SELF_BLACK
    else:  # 对方半场
        std = _STD_OPPO if my_side == "r" else _STD_OPPO_RED
    if row in std:
        pos_map = std[row]
        if isinstance(pos_map, dict):
            return pos_map.get(col, "?")
        elif isinstance(pos_map, list) and col < len(pos_map):
            return pos_map[col]
    return "?"
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
    # 天天象棋：己方永远在 board 下半 (row 5-9)
    # 引擎：rank 0-4=红方半场, rank 5-9=黑方半场（固定）
    # FEN 段0=引擎rank9, 段9=引擎rank0（第一段=棋盘最上方）
    # 需要：红方棋子→rank 0-4(FEN段5-9), 黑方棋子→rank 5-9(FEN段0-4)
    # my_side=="r": 己方(红)在row 5-9 → FEN段5-9(rank 0-4) → 正序
    # my_side=="b": 己方(黑)在row 5-9 → FEN段0-4(rank 5-9) → 倒序
    if my_side == "r":
        row_range = range(10)  # 正序：row0→段0(rank9黑方半场), row9→段9(rank0红方半场)
    else:
        row_range = range(9, -1, -1)  # 倒序：row9→段0(rank9黑方半场), row0→段9(rank0红方半场)
    rows = []
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
    # my_side=="r": FEN正序 → board_row = 9 - engine_rank
    # my_side=="b": FEN倒序 → board_row = engine_rank
    if my_side == "r":
        fr, tr = 9 - engine_fr, 9 - engine_tr
    else:
        fr, tr = engine_fr, engine_tr
    piece = board[fr][fc]
    is_hidden = piece.endswith("?")
    if is_hidden:
        pn = _get_hidden_name(fr, fc, my_side)
    else:
        pn = PIECE_NAME.get(piece, "?")
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


def _activate_window():
    """AppleScript 激活微信到前台"""
    try:
        subprocess.run([
            "osascript", "-e",
            'tell application "System Events" to set frontmost of process "微信" to true'
        ], timeout=3)
    except Exception:
        pass


def board_pos_to_screen(row, col, win_img, win_img_w, win_img_h):
    """棋盘 (row, col) → 屏幕绝对逻辑坐标
    win_img: 全屏截图（Retina 2x），直接像素÷2=逻辑坐标
    棋盘 = 全屏截图居中 1529x1695 区域，按 10x9 网格划分
    """
    ox = (win_img_w - CROP_W) // 2
    oy = (win_img_h - CROP_H) // 2
    cw, ch = CROP_W / 9, CROP_H / 10
    # 像素坐标（Retina 2x），÷2 转屏幕逻辑坐标
    px = int((ox + col * cw + cw / 2) / 2)
    py = int((oy + row * ch + ch / 2) / 2)
    return (px, py)


def execute_move(uci, recognizer, my_side):
    u = uci.lstrip('+')
    if len(u) < 4 or not u[1].isdigit() or not u[3].isdigit():
        print(f"[!] 无效着法: {uci}"); return
    fc, engine_fr = ord(u[0]) - 97, int(u[1])
    tc, engine_tr = ord(u[2]) - 97, int(u[3])
    # my_side=="r": FEN正序 → board_row = 9 - engine_rank
    # my_side=="b": FEN倒序 → board_row = engine_rank
    if my_side == "r":
        fr, tr = 9 - engine_fr, 9 - engine_tr
    else:
        fr, tr = engine_fr, engine_tr

    # 1. 激活窗口到前台
    _activate_window()
    time.sleep(0.3)

    # 2. 全屏截图（Retina 2x），直接像素÷2=屏幕逻辑坐标，无需 win_pos
    import pyautogui as pag
    ss = pag.screenshot()
    win_img = cv2.cvtColor(np.array(ss), cv2.COLOR_RGB2BGR)
    h, w = win_img.shape[:2]
    print(f"[*] 全屏截图: {w}x{h}")

    # 3. 计算屏幕坐标
    fx, fy = board_pos_to_screen(fr, fc, win_img, w, h)
    tx, ty = board_pos_to_screen(tr, tc, win_img, w, h)
    print(f"[*] 走子: ({fr},{fc})→({tr},{tc}) 屏幕: ({fx},{fy})→({tx},{ty})")

    # 4.5 保存调试截图：在截图上画出所有格子中心和走子标记
    debug_img = win_img.copy()
    ox = (w - CROP_W) // 2
    oy = (h - CROP_H) // 2
    cw, ch = CROP_W / 9, CROP_H / 10
    # 画 10x9 网格中心点
    for rr in range(10):
        for cc in range(9):
            px = int(ox + cc * cw + cw / 2)
            py = int(oy + rr * ch + ch / 2)
            color = (0, 255, 0)  # 绿点
            if rr == fr and cc == fc:
                color = (0, 0, 255)  # 起点红点
            elif rr == tr and cc == tc:
                color = (255, 0, 0)  # 终点蓝点
            cv2.circle(debug_img, (px, py), 5, color, -1)
    # 画箭头从起点到终点
    fx_px = int(ox + fc * cw + cw / 2)
    fy_px = int(oy + fr * ch + ch / 2)
    tx_px = int(ox + tc * cw + cw / 2)
    ty_px = int(oy + tr * ch + ch / 2)
    cv2.arrowedLine(debug_img, (fx_px, fy_px), (tx_px, ty_px), (0, 255, 255), 3)
    # 画截图中心点
    cv2.circle(debug_img, (w // 2, h // 2), 10, (0, 165, 255), -1)  # 橙色大圆
    cv2.putText(debug_img, f"center({w//2},{h//2})", (w // 2 + 15, h // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    # 画棋盘裁剪区域
    cv2.rectangle(debug_img, (ox, oy), (ox + CROP_W, oy + CROP_H), (255, 255, 0), 2)
    sd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshot")
    os.makedirs(sd, exist_ok=True)
    dbg_path = os.path.join(sd, f"debug_move_{time.strftime('%H%M%S')}.png")
    cv2.imencode(".png", debug_img)[1].tofile(dbg_path)
    print(f"[*] 调试截图已保存: {dbg_path}")

    # 5. 走子
    pyautogui.click(fx, fy); time.sleep(0.5)
    pyautogui.click(tx, ty); time.sleep(0.3)
    print(f"[+] 已走子")

    # 6. 切回 Terminal，确保下次能检测到状态框
    subprocess.run([
        "osascript", "-e",
        'tell application "Terminal" to activate'
    ], timeout=3)


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
                board = recognizer.detect(board_img, my_side=None)
                # 每局重新判断阵营：己方半场 row 5-9，看帥/將在哪个半场
                hs = any(board[r][c] == "r帥" for r in range(5, 10) for c in range(9))
                hj = any(board[r][c] == "b將" for r in range(5, 10) for c in range(9))
                new_side = "r" if hs else ("b" if hj else my_side)
                if new_side is None:
                    new_side = "r"
                if new_side != my_side:
                    my_side = new_side
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
