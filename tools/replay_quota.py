# -*- coding: utf-8 -*-
"""In-process 复刻 referee 走子循环, 用新引擎驱动红方, 监测是否走出超配额将军着。
直接 import jieqi_engine 跑搜索(无子进程开销), 复现 seed 的失败对局。
"""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import referee as R
import jieqi_engine as E

RED_POOL = R.RED_POOL
BLACK_POOL = R.BLACK_POOL
INIT_SQUARES = R.INIT_SQUARES

# 直接用新引擎(己方视角)跑, 不再经子进程
new_engine = E.JieQiEngine()
import jieqi_engine_v514 as E514
old_engine = E514.JieQiEngine()


def new_game(seed):
    rng = random.Random(seed)
    board = [["."] * 9 for _ in range(10)]
    secret = {}
    board[9][4], board[0][4] = "r帥", "b將"
    for side, pool in (("r", RED_POOL), ("b", BLACK_POOL)):
        pool = list(pool)
        rng.shuffle(pool)
        for (r, c), t in zip(INIT_SQUARES[side], pool):
            board[r][c] = side + "?"
            secret[(r, c)] = t
    return board, secret


def flip_board(board):
    return [[board[9 - r][8 - c] for c in range(9)] for r in range(10)]


def uci_to_view_rc(uci):
    u = uci.lstrip("+")
    src = (9 - int(u[1]), ord(u[0]) - 97)
    dst = (9 - int(u[3]), ord(u[2]) - 97)
    return src, dst


def replay(seed, think=1.0, new_is_red=True):
    board, secret = new_game(seed)
    trackers = {s: R.PerpCheckTracker() for s in "rb"}
    for ply in range(400):
        side = "r" if ply % 2 == 0 else "b"
        view = board if side == "r" else flip_board(board)
        # 合法着法检查
        if not R.legal_engine_moves(view, side):
            print(f"[ply {ply+1}] {side} 无合法着法")
            return
        engine = new_engine if (side == "r") == new_is_red else old_engine
        cs = trackers[side].state_for()
        uci, score, depth = engine.get_best_move(view, side, think_time=think,
                                                  check_state=cs)
        if uci is None:
            print(f"[ply {ply+1}] {side} 引擎无着法")
            return
        src_v, dst_v = uci_to_view_rc(uci)
        if not R.is_legal(view, side, src_v, dst_v):
            print(f"[ply {ply+1}] {side} 非法着法 {uci}")
            return
        src = src_v if side == "r" else R.rc_flip(src_v)
        dst = dst_v if side == "r" else R.rc_flip(dst_v)
        captured, reveal = R.apply_move(board, secret, side, src, dst)

        # 更新将军追踪
        next_side = "b" if side == "r" else "r"
        for s in "rb":
            trackers[s].on_any_move(src, dst, captured != ".")
        if R.in_check(board, next_side):
            before = trackers[side].count
            trackers[side].deliver_check(dst)
            after = trackers[side].count
            if trackers[side].exceeded():
                n_pieces = len(trackers[side].squares) + trackers[side].retired
                st = E.State.from_string(E.board_to_engine_string(view, side))
                mv = (E._row_col_to_engine_idx(*src_v), E._row_col_to_engine_idx(*dst_v))
                gc = E.JieQiEngine()._gives_check(st, mv)
                print(f"[VIOLATION] ply {ply+1} {side} 走 {uci}: "
                      f"连将 {before}->{after} 次, 配额 min(3*{n_pieces},9)={trackers[side].quota}")
                print(f"  check_state 下发的是: count={cs['count']}, "
                      f"squares={cs.get('squares')}, retired={cs.get('retired')}")
                print(f"  引擎 gives_check({uci})= {gc}  (False=漏判, 导致未剪枝)")
                print(f"  裁判 in_check(next={next_side})= True")
                print("  局面(view, {side}视角, 红在下):")
                for r in range(10):
                    print("   ", " ".join(view[r]))
                # 同时验证: 该 move 是否真的让 next_side 被将军 (用引擎自身 make/unmake 测)
                st2 = E.State.from_string(E.board_to_engine_string(view, side))
                st2.make(mv[0], mv[1])
                chk2 = st2.can_capture_king(st2.stm)
                print(f"  [交叉验证] make后 can_capture_king(stm={st2.stm})={chk2}")
                return
        else:
            trackers[side].reset()

        if captured in R.KING.values():
            print(f"[结束] ply {ply+1} {side} 吃王获胜")
            return
        if ply > 160:
            print("超过160着, 中断观测")
            return
    print("达到 max_ply")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=330013)
    ap.add_argument("--think", type=float, default=1.0)
    ap.add_argument("--black", action="store_true", help="新引擎执黑")
    a = ap.parse_args()
    replay(a.seed, a.think, new_is_red=not a.black)
