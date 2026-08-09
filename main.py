"""
揭棋 AI 主程序 - 状态框检测 + 引擎 + 自动走子
"""
import sys, os, time, subprocess, json, cv2, numpy as np, threading, queue, re
import pyautogui
from board_recognizer import BoardRecognizer

SCAN_INTERVAL = 0.3
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


FEN_MAP = {
    "r帥": "K", "r仕": "A", "r相": "E", "r馬": "H", "r車": "R", "r炮": "C", "r兵": "P",
    "b將": "k", "b士": "a", "b象": "e", "b馬": "h", "b車": "r", "b炮": "c", "b卒": "p",
    "r?": "X", "b?": "x",
}
# PikaJieQi 使用标准 UCI 棋子字母: B=相, N=馬
PIKAJIEQI_FEN_MAP = {
    "r帥": "K", "r仕": "A", "r相": "B", "r馬": "N", "r車": "R", "r炮": "C", "r兵": "P",
    "b將": "k", "b士": "a", "b象": "b", "b馬": "n", "b車": "r", "b炮": "c", "b卒": "p",
    "r?": "X", "b?": "x",
}
PIECE_MAP = {
    "r帥": "R", "r仕": "A", "r相": "B", "r馬": "N", "r車": "R", "r炮": "C", "r兵": "P",
    "b將": "k", "b士": "a", "b象": "b", "b馬": "n", "b車": "r", "b炮": "c", "b卒": "p",
}
PIECE_COUNTS = {
    "r": {"R": 2, "A": 2, "B": 2, "N": 2, "C": 2, "P": 5},
    "b": {"r": 2, "a": 2, "b": 2, "n": 2, "c": 2, "p": 5},
}

def board_to_pikajieqi_fen(board, my_side):
    # 使用 PikaJieQi 的标准 UCI 棋子字母 (B=相/象, N=馬)
    rc = sum(1 for r in range(10) for c in range(9) if board[r][c].startswith("r"))
    bc = sum(1 for r in range(10) for c in range(9) if board[r][c].startswith("b"))
    for r in range(10):
        for c in range(9):
            if rc > 16 and board[r][c] == "r?": board[r][c] = "."; rc -= 1
            if bc > 16 and board[r][c] == "b?": board[r][c] = "."; bc -= 1
    if my_side == "r":
        row_range = range(10)
    else:
        row_range = range(9, -1, -1)
    rows = []
    for r in row_range:
        s = ""; e = 0
        for c in range(9):
            p = board[r][c]
            if p == ".": e += 1
            else:
                if e: s += str(e); e = 0
                s += PIKAJIEQI_FEN_MAP.get(p, p)
        if e: s += str(e)
        rows.append(s)
    placement = "/".join(rows)
    side_char = "w" if my_side == "r" else "b"
    # 统计已翻开的各类型棋子数量
    revealed = {}
    for r in range(10):
        for c in range(9):
            k = PIECE_MAP.get(board[r][c])
            if k:
                revealed[k] = revealed.get(k, 0) + 1
    # 剩余暗子数 = 总数 - 已翻开
    rest_parts = []
    for color_key in ["r", "b"]:
        counts = PIECE_COUNTS[color_key]
        for key, total in counts.items():
            revealed_count = revealed.get(key, 0)
            if total >= revealed_count:
                rest_parts.append(f"{key}{total - revealed_count}")
            else:
                rest_parts.append(f"{key}0")
    return f"{placement} {side_char} {' '.join(rest_parts)} 0 1"

class PikaJieQiEngine:
    def __init__(self, engine_path, nnue_path=None, threads=1, hash_mb=64, use_wine=False):
        self.engine_path = engine_path
        self.nnue_path = nnue_path or os.path.join(os.path.dirname(engine_path), "pikafish.nnue")
        self.threads = threads
        self.hash_mb = hash_mb
        self.use_wine = use_wine
        self._proc = None
        self._q = queue.Queue()

    def _reader(self):
        for line in iter(self._proc.stdout.readline, ""):
            s = line.strip()
            if not s:
                continue
            # 过滤 Wine/MoltenVK/Vulkan 启动信息
            if s.startswith("[mvk-") or s.startswith("[wine-") or s.startswith("Vulkan"):
                continue
            if "VK_" in s and "supported" in s:
                continue
            if s.startswith("GPU ") or s.startswith("pipelineCache"):
                continue
            if s.startswith("Metal ") or "memory used" in s.lower():
                continue
            self._q.put(s)

    def _start(self):
        if self._proc is not None:
            return
        cmd = ["wine", self.engine_path] if self.use_wine else [self.engine_path]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding="utf-8", errors="ignore", bufsize=1,
            env={**os.environ, "WINEDEBUG": "-all"},
        )
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()
        self._wait_for("uciok")
        self._send(f"setoption name EvalFile value {self.nnue_path}")
        self._send(f"setoption name Threads value {self.threads}")
        self._send(f"setoption name Hash value {self.hash_mb}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd):
        try:
            self._proc.stdin.write(cmd + "\n")
            self._proc.stdin.flush()
        except Exception:
            pass

    def _wait_for(self, keyword, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self._q.get(timeout=0.1)
                if keyword in line:
                    return True
            except queue.Empty:
                pass
        return False

    def get_best_move(self, fen, movetime_ms=2000):
        self._start()
        self._send("ucinewgame")
        time.sleep(0.05)
        self._send(f"position fen {fen}")
        self._send(f"go movetime {movetime_ms}")
        bestmove = None
        score = 0
        depth = 0
        deadline = time.time() + (movetime_ms / 1000) + 5
        while time.time() < deadline:
            try:
                line = self._q.get(timeout=0.1)
                if line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) >= 2:
                        bestmove = parts[1]
                    break
                if line.startswith("info"):
                    m = re.search(r"score cp (-?\d+)", line)
                    if m:
                        score = int(m.group(1))
                    m = re.search(r"depth (\d+)", line)
                    if m:
                        depth = int(m.group(1))
            except queue.Empty:
                pass
        return {"move": bestmove, "score": score, "depth": depth}

    def close(self):
        if self._proc:
            try:
                self._send("quit")
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.terminate()
            self._proc = None

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
    # sd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshot")
    # os.makedirs(sd, exist_ok=True)
    # dbg_path = os.path.join(sd, f"debug_move_{time.strftime('%H%M%S')}.png")
    # cv2.imencode(".png", debug_img)[1].tofile(dbg_path)
    # print(f"[*] 调试截图已保存: {dbg_path}")

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
    print("[*] 揭棋 AI v3.0 启动中...")
    recognizer = BoardRecognizer()
    print("[*] 识别器就绪")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    engine_dir = os.path.join(base_dir, "engines")

    xiangqi_ai_path = os.path.join(engine_dir, "xiangqi-ai")
    pikajieqi_nnue = os.path.join(engine_dir, "pikafish.nnue")

    # 引擎选择：优先 揭棋原生引擎 (纯算法, 无需 NNUE) → PikaJieQi → xiangqi-ai
    engine_type = None
    pikajieqi_engine = None
    jieqi_engine = None
    use_wine = False

    # 0) 优先使用 miaosiSari 揭棋纯算法引擎 (PST 评估, 无需权重)
    try:
        from jieqi_engine import JieQiEngine
        jieqi_engine = JieQiEngine()
        engine_type = "jieqi"
        print(f"[+] 使用引擎: 揭棋纯算法引擎 (PST, 无需 NNUE)")
    except Exception as e:
        print(f"[!] 揭棋引擎加载失败: {e}")

    # 1) 尝试原生 macOS 二进制
    pikajieqi_native = os.path.join(engine_dir, "PikaJieQi")
    # 2) 尝试 Windows exe (通过 Wine)
    pikajieqi_exe = os.path.join(engine_dir, "PikaJieQi-modern.exe")

    if engine_type is None:
        for candidate, wine in [(pikajieqi_native, False), (pikajieqi_exe, True)]:
            if not os.path.exists(candidate):
                continue
            label = f"Wine+{os.path.basename(candidate)}" if wine else "PikaJieQi"
            print(f"[+] 检测到 {label}")
            if os.path.exists(pikajieqi_nnue):
                try:
                    pikajieqi_engine = PikaJieQiEngine(candidate, pikajieqi_nnue, use_wine=wine)
                    engine_type = "pikajieqi"
                    use_wine = wine
                    print(f"[+] 使用引擎: {label} (UCI)")
                    break
                except Exception as e:
                    print(f"[!] {label} 启动失败: {e}")
            else:
                print(f"[!] 未找到 NNUE 文件, {label} 棋力较弱")

    if engine_type is None and os.path.exists(xiangqi_ai_path):
        engine_type = "xiangqi-ai"
        print(f"[+] 使用引擎: xiangqi-ai")
    if engine_type is None:
        print("[!] 未找到任何引擎"); return

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

                if engine_type == "jieqi" and jieqi_engine:
                    uci, score, depth = jieqi_engine.get_best_move(board, my_side, think_time=2.0)
                    if uci:
                        # 引擎 UCI 为己方视角 (rank = 9 - board_row)，等价于 execute_move/uci_to_human 的 'r' 分支
                        print(f"[+] 推荐: {uci} → {uci_to_human(uci, board, 'r')} (分数:{score} 深度:{depth})")
                        print("[*] 自动走子...")
                        execute_move(uci, recognizer, "r")
                    else:
                        print("[!] 揭棋引擎无着法")
                elif engine_type == "pikajieqi" and pikajieqi_engine:
                    fen = board_to_pikajieqi_fen(board, my_side)
                    print(f"[*] FEN: {fen}")
                    result = pikajieqi_engine.get_best_move(fen, movetime_ms=2000)
                    if result.get("move"):
                        uci = result["move"]
                        print(f"[+] 推荐: {uci} → {uci_to_human(uci, board, my_side)} (分数:{result['score']} 深度:{result['depth']})")
                        print("[*] 自动走子...")
                        execute_move(uci, recognizer, my_side)
                    else:
                        print("[!] PikaJieQi 无着法")
                else:
                    fen = board_to_fen(board, my_side)
                    print(f"[*] FEN: {fen}")
                    r = subprocess.run([xiangqi_ai_path, "best", "--fen", fen, "--strategy", "it2", "--time-limit", "2", "--json"],
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

    if pikajieqi_engine:
        pikajieqi_engine.close()


if __name__ == "__main__":
    main()
