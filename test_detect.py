"""
棋盘标定测试
用法:
  python test_detect.py                    # 自动检测
  python test_detect.py 10 8               # 自动检测+整体偏移(dx,dy)
  python test_detect.py 20 30 830 930      # 手动指定四角(x1,y1,x2,y2)
"""
import os
import sys
import cv2
from board_detector import BoardDetector, BOARD_ROWS, BOARD_COLS


def main():
    args = sys.argv[1:]

    img_path = os.path.join(os.path.expanduser("~"), "Desktop", "chess.png")
    img = cv2.imread(img_path)
    if img is None:
        print("[!] 读取图片失败")
        return

    h, w = img.shape[:2]
    print(f"[*] 图片尺寸: {w}x{h}")

    detector = BoardDetector()

    if len(args) >= 4:
        x1, y1, x2, y2 = int(args[0]), int(args[1]), int(args[2]), int(args[3])
        bx, by, bw, bh = x1, y1, x2 - x1, y2 - y1
    else:
        rect = detector.find_board_corners(img)
        if rect is None:
            print("[!] 未检测到棋盘")
            return
        bx, by, bw, bh = rect
        if len(args) >= 2:
            bx += int(args[0])
            by += int(args[1])

    print(f"[+] 棋盘: ({bx},{by}) w={bw} h={bh}")

    cell_w = bw // BOARD_COLS
    cell_h = bh // BOARD_ROWS
    print(f"[+] 格子: {cell_w}x{cell_h}")

    display = img.copy()
    cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (255, 0, 0), 3)

    board = detector.detect_board(img, (bx, by, bw, bh))
    if not board or not board[0]:
        print("[!] 检测失败")
        return

    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            gx = bx + col * cell_w
            gy = by + row * cell_h
            cv2.rectangle(display, (gx, gy), (gx + cell_w, gy + cell_h), (0, 255, 0), 1)
            cell = board[row][col]
            if not cell.is_empty:
                cx = bx + int((col + 0.5) * cell_w)
                cy = by + int((row + 0.5) * cell_h)
                cv2.circle(display, (cx, cy), cell_w // 3, (0, 0, 255), 2)
                cv2.putText(display, cell.piece, (cx - 12, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    out = os.path.join(os.path.expanduser("~"), "Desktop", "chess_detected.png")
    cv2.imwrite(out, display)
    print(f"[+] 已保存: {out}")

    print("\n[*] 棋盘:")
    print("   " + "".join(f"{c:^4}" for c in range(BOARD_COLS)))
    for row in range(BOARD_ROWS):
        line = f"{row:2d} "
        for col in range(BOARD_COLS):
            line += f" {board[row][col].piece:>3}"
        print(line)

    r = sum(1 for row in board for c in row if c.piece.startswith("r"))
    b = sum(1 for row in board for c in row if c.piece.startswith("b"))
    e = sum(1 for row in board for c in row if c.is_empty)
    print(f"\n[*] 红方{r}  黑方{b}  空位{e}")


if __name__ == "__main__":
    main()
