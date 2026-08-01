"""
模板采集工具
用法:
  python capture_templates.py                   # 实时窗口采集
  python capture_templates.py chess.png         # 从截图采集
"""
import cv2
import numpy as np
import os
import sys

from board_detector import BoardDetector, BOARD_ROWS, BOARD_COLS, PIECE_CHARS_RED, PIECE_CHARS_BLACK


def main():
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    os.makedirs(assets_dir, exist_ok=True)

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        if not os.path.exists(img_path):
            print(f"[!] 文件不存在: {img_path}")
            return
        img = cv2.imread(img_path)
        if img is None:
            print("[!] 读取图片失败")
            return
    else:
        from controller import Controller
        controller = Controller("天天象棋")
        if not controller.connect():
            input("未找到窗口，按回车退出...")
            return
        raw = controller.capture()
        if raw is None:
            print("截图失败")
            return
        img = np.array(raw)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    cv2.imencode(".png", img)[1].tofile(os.path.join(assets_dir, "_screenshot.png"))

    detector = BoardDetector(assets_dir)
    rect = detector.find_board_corners(img)
    if rect is None:
        print("[!] 未检测到棋盘，使用全图作为棋盘")
        h, w = img.shape[:2]
        rect = (0, 0, w, h)

    bx, by, bw, bh = rect
    cell_w = bw // BOARD_COLS
    cell_h = bh // BOARD_ROWS
    radius = min(cell_w, cell_h) // 2 - 2

    ref_path = os.path.join(assets_dir, "cell_ref.txt")
    with open(ref_path, "w") as f:
        f.write(f"{cell_w} {cell_h}")

    print(f"[+] 棋盘: ({bx},{by}) {bw}x{bh}  格子: {cell_w}x{cell_h}  半径: {radius}")
    print()

    while True:
        cmd = input("输入 <side> <piece> <row> <col> (如 r 车 9 0) 或 q 退出: ").strip()
        if cmd.lower() == "q":
            break
        parts = cmd.split()
        if len(parts) != 4:
            print("  格式: side piece row col")
            continue

        side = parts[0]
        piece = parts[1]
        try:
            row = int(parts[2])
            col = int(parts[3])
        except ValueError:
            print("  行/列必须是数字")
            continue

        if side not in ("r", "b"):
            print("  side 必须是 r 或 b")
            continue
        if piece == "h":
            piece = "?"
        if not (0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS):
            print(f"  行 0-9, 列 0-8")
            continue

        cx = bx + int((col + 0.5) * cell_w)
        cy = by + int((row + 0.5) * cell_h)
        x1, y1 = max(cx - radius, 0), max(cy - radius, 0)
        x2, y2 = min(cx + radius, img.shape[1]), min(cy + radius, img.shape[0])
        roi = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        filename = f"{side}{piece}.png"
        filepath = os.path.join(assets_dir, filename)
        ok, buf = cv2.imencode(".png", gray)
        if ok:
            with open(filepath, "wb") as f:
                f.write(buf.tobytes())
            print(f"  [OK] {filename} ({os.path.getsize(filepath)} bytes)")
        else:
            print(f"  [!] 编码失败")

    print(f"\n[+] 共采集 {len(os.listdir(assets_dir)) - 1} 个模板")


if __name__ == "__main__":
    main()
