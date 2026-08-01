import sys
import os
import time
import cv2
import numpy as np
from typing import Optional
from collections import Counter

from controller import Controller
from board_detector import BoardDetector, Cell, BOARD_ROWS, BOARD_COLS
from game_state import GameState
from engine import UCCIEngine


WINDOW_TITLE = "天天象棋"
SCAN_INTERVAL = 1.0
CONFIDENCE_THRESHOLD = 3
BOARD_OFFSET_X = 13
BOARD_OFFSET_Y = 6


def main():
    print("[*] 揭棋 AI 启动中...")

    controller = Controller(WINDOW_TITLE)
    if not controller.connect():
        print(f"[!] 未找到窗口: {WINDOW_TITLE}")
        print("    请打开天天象棋并进入揭棋对局")
        input("    按回车重试...")
        return

    print(f"[+] 已连接窗口: {WINDOW_TITLE}")

    detector = BoardDetector(offset_x=BOARD_OFFSET_X, offset_y=BOARD_OFFSET_Y)
    game = GameState()
    engine = None

    engine_path = UCCIEngine.default_engine_path()
    if engine_path:
        try:
            engine = UCCIEngine(engine_path)
            print(f"[+] 引擎已加载: {engine.name} by {engine.author}")
        except Exception as e:
            print(f"[!] 引擎加载失败: {e}")
            print("    请将皮卡鱼(pikafish.exe)放入 engines/ 目录")
    else:
        print("[!] engines/ 目录未找到引擎文件")
        print("    请下载皮卡鱼并放入 engines/ 目录")

    my_side: Optional[str] = None
    last_board_hash = ""
    stable_count = 0

    print("[*] 开始监控棋盘...")

    while True:
        try:
            raw = controller.capture()
            if raw is None:
                time.sleep(SCAN_INTERVAL)
                continue

            img = np.array(raw)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            board = detector.detect_board(img)

            if not board or not board[0]:
                time.sleep(SCAN_INTERVAL)
                continue

            if my_side is None:
                my_side = _detect_my_side(board)
                if my_side:
                    print(f"[+] 己方颜色: {'红方' if my_side == 'r' else '黑方'}")

            board_hash = _board_hash(board)
            if board_hash == last_board_hash and game.turn != my_side:
                stable_count += 1
                if stable_count < CONFIDENCE_THRESHOLD:
                    time.sleep(0.3)
                    continue
            else:
                stable_count = 0
            last_board_hash = board_hash

            internal_board = _to_internal_board(board, my_side)
            game.update_from_detection(internal_board)

            if game.is_game_over():
                print("[*] 对局结束")
                break

            if game.turn == my_side and engine:
                has_hidden = any(
                    game.board[r][c].endswith("?")
                    for r in range(BOARD_ROWS) for c in range(BOARD_COLS)
                )
                best_move = _search_best(game, engine, has_hidden)
                if best_move:
                    fr, fc, tr, tc = _parse_uci(best_move)
                    sfr, sfc = _internal_to_screen(fr, fc, my_side)
                    str_, stc = _internal_to_screen(tr, tc, my_side)
                    print(f"  [→] 走子: {best_move} → 屏幕坐标({sfr},{sfc})→({str_},{stc})")
                    game.apply_move(fr, fc, tr, tc)
                    controller.make_move(sfr, sfc, str_, stc)
                    time.sleep(0.6)
                else:
                    print("  [!] 引擎未返回走法，等待...")

            time.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            print("\n[*] 手动停止")
            break
        except Exception as e:
            print(f"[!] 错误: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(SCAN_INTERVAL)

    if engine:
        engine.quit()


def _search_best(game, engine, has_hidden: bool) -> Optional[str]:
    if not has_hidden:
        fen = game.to_fen()
        print(f"  [*] FEN: {fen}")
        best_move, _ = engine.search(fen, movetime=2000)
        return best_move if best_move else None

    samples = 5
    moves = []
    for i in range(samples):
        fen = game.to_masked_fen()
        print(f"  [{i+1}/{samples}] {fen[:50]}...")
        move, _ = engine.search(fen, movetime=400)
        if move:
            moves.append(move)
    if not moves:
        return None
    counter = Counter(moves)
    best = counter.most_common(1)[0]
    print(f"  [*] 采样结果: {dict(counter)} → 最优: {best[0]}")
    return best[0]


def _detect_my_side(board) -> Optional[str]:
    raw = input("[?] 己方颜色 (r=红方/b=黑方): ").strip().lower()
    if raw in ("r", "b"):
        return raw
    return None


def _board_hash(board) -> str:
    return "|".join(
        "".join(c.piece for c in row)
        for row in board
    )


def _to_internal_board(screen_board, my_side: str):
    opponent = "b" if my_side == "r" else "r"
    result = []
    for r in range(BOARD_ROWS):
        row = []
        for c in range(BOARD_COLS):
            if my_side == "b":
                src_row = BOARD_ROWS - 1 - r
                src_col = BOARD_COLS - 1 - c
            else:
                src_row = r
                src_col = c
            cell = screen_board[src_row][src_col]
            piece = cell.piece
            if piece.startswith("d"):
                piece = my_side + piece[1:]
            elif piece.startswith("t"):
                piece = opponent + piece[1:]
            row.append(Cell(row=r, col=c, x=cell.x, y=cell.y, piece=piece))
        result.append(row)
    return result


def _internal_to_screen(internal_r: int, internal_c: int, my_side: str):
    """内部标准坐标转屏幕点击坐标"""
    if my_side == "b":
        return (BOARD_ROWS - 1 - internal_r, BOARD_COLS - 1 - internal_c)
    return (internal_r, internal_c)


def _parse_uci(move: str):
    fc = ord(move[0]) - ord('a')
    fr = 9 - int(move[1])
    tc = ord(move[2]) - ord('a')
    tr = 9 - int(move[3])
    return fr, fc, tr, tc


if __name__ == "__main__":
    main()
