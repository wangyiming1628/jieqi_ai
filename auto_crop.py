"""
自动从截图切分棋子模板
用法: python auto_crop.py [截图路径]
"""
import os, sys, cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board_detector import BoardDetector, BOARD_ROWS, BOARD_COLS


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Desktop/chess.png")
    img = cv2.imread(img_path)
    if img is None:
        print(f"[!] 读取失败: {img_path}")
        return

    h, w = img.shape[:2]
    print(f"[*] 图片: {w}x{h}")

    assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    os.makedirs(assets, exist_ok=True)

    # 保存参考尺寸
    detector = BoardDetector()
    rect = detector.find_board_corners(img)
    if rect is None:
        rect = (0, 0, w, h)
    bx, by, bw, bh = rect
    cell_w = bw // BOARD_COLS
    cell_h = bh // BOARD_ROWS
    radius = min(cell_w, cell_h) // 2 - 2
    print(f"[+] 棋盘: ({bx},{by}) {bw}x{bh}  格子: {cell_w}x{cell_h}")

    with open(os.path.join(assets, "cell_ref.txt"), "w") as f:
        f.write(f"{cell_w} {cell_h}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    midline = by + bh // 2

    saved = 0
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            cx = bx + int((col + 0.5) * cell_w)
            cy = by + int((row + 0.5) * cell_h)
            x1, y1 = max(cx - radius, 0), max(cy - radius, 0)
            x2, y2 = min(cx + radius, w), min(cy + radius, h)
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            mean_val = np.mean(roi)
            if mean_val > 200:
                continue

            side = "r" if cy < midline else "b"
            filename = f"{side}_hidden_{row:02d}_{col:02d}.png"
            filepath = os.path.join(assets, filename)
            ok, buf = cv2.imencode(".png", roi)
            if ok:
                with open(filepath, "wb") as f:
                    f.write(buf.tobytes())
                saved += 1

    print(f"\n[+] 共保存 {saved} 个棋子图片到 assets/")
    print("    暗棋模板已生成, 翻开的棋子模板需要从翻棋后的截图采集")
    print("    从 r_hidden_*.png 和 b_hidden_*.png 中各选一个清晰的")
    print("    重命名为 r?.png 和 b?.png")


if __name__ == "__main__":
    main()
