# -*- coding: utf-8 -*-
"""差分测试: 对大量可达局面, 比较 State.gives_check(move) 与独立"走子后对方是否被将军"判定。
独立判定 = 用 referee.in_check(apply_move 后的规范棋盘, 对方)。任何 gives_check=False 但独立=True 的漏判,
正是导致搜索内长将剪枝失效、引擎走出超配额将军着的元凶。
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import referee as R
import jieqi_engine as E

RED_POOL = R.RED_POOL
BLACK_POOL = R.BLACK_POOL
INIT_SQUARES = R.INIT_SQUARES
KING = R.KING


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


def engine_idx(src, dst):
    return (E._row_col_to_engine_idx(*src), E._row_col_to_engine_idx(*dst))


def independent_gives_check(board, side, src, dst, secret):
    """走完 move 后, side 是否将军了对方 (规范棋盘视角)。"""
    import copy
    nb = [row[:] for row in board]
    ns = dict(secret)
    R.apply_move(nb, ns, side, src, dst)
    next_side = "b" if side == "r" else "r"
    return R.in_check(nb, next_side)


_ENGINE = None

_DUMP = []

def test_one_position(board, secret, side):
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = E.JieQiEngine()
    view = board if side == "r" else flip_board(board)
    estr = E.board_to_engine_string(view, side)
    # 初始化全局评估表 (value/_pstats 需要), 与真实 get_best_move 一致
    _ENGINE._pool.observe(estr)
    _ENGINE._pool.apply(estr)
    _ENGINE.searcher.calc_average()
    est = E.State.from_string(estr)
    mism = []
    for mv in est.gen_moves():
        src_i, dst_i = mv
        src_rc = E._engine_idx_to_row_col(src_i)
        dst_rc = E._engine_idx_to_row_col(dst_i)
        if side == "r":
            csrc, cdst = src_rc, dst_rc
        else:
            csrc = R.rc_flip(src_rc)
            cdst = R.rc_flip(dst_rc)
        eng_gc = est.gives_check(mv)
        ref_gc = independent_gives_check(board, side, csrc, cdst, secret)
        if eng_gc != ref_gc:
            mism.append((src_rc, dst_rc, eng_gc, ref_gc))
            if len(_DUMP) < 25:
                _DUMP.append({
                    "board": board, "secret": {str(k): v for k, v in secret.items()},
                    "side": side,
                    "csrc": list(csrc), "cdst": list(cdst),
                    "eng_gc": eng_gc, "ref_gc": ref_gc,
                    "view": view,
                })
    return mism
    mism = []
    for mv in est.gen_moves():
        src_i, dst_i = mv
        # 引擎视角 rc
        src_rc = E._engine_idx_to_row_col(src_i)
        dst_rc = E._engine_idx_to_row_col(dst_i)
        # 转规范坐标 (裁判视角)
        if side == "r":
            csrc, cdst = src_rc, dst_rc
        else:
            csrc = R.rc_flip(src_rc)
            cdst = R.rc_flip(dst_rc)
        eng_gc = est.gives_check(mv)
        ref_gc = independent_gives_check(board, side, csrc, cdst, secret)
        if eng_gc != ref_gc:
            mism.append((src_rc, dst_rc, eng_gc, ref_gc))
    return mism


def main():
    random.seed(12345)
    total_mismatch = 0
    positions_tested = 0
    for trial in range(400):
        seed = 1000 + trial
        board, secret = new_game(seed)
        # 随机走若干合法着法, 形成中局局面
        for ply in range(random.randint(5, 40)):
            side = "r" if ply % 2 == 0 else "b"
            view = board if side == "r" else flip_board(board)
            moves = list(R.legal_engine_moves(view, side))
            if not moves:
                break
            m = random.choice(list(moves))
            src_v, dst_v = R.uci_to_view_rc(E._engine_idx_to_uci(m[0]) + E._engine_idx_to_uci(m[1]))
            src = src_v if side == "r" else R.rc_flip(src_v)
            dst = dst_v if side == "r" else R.rc_flip(dst_v)
            cap, rev = R.apply_move(board, secret, side, src, dst)
            if cap in KING.values():
                break
        # 在最终局面测双方
        for side in ("r", "b"):
            mism = test_one_position(board, secret, side)
            positions_tested += 1
            if mism:
                total_mismatch += len(mism)
                if total_mismatch <= 500:
                    print(f"[MISMATCH] seed={seed} side={side} n={len(mism)}")
                    for s, d, eg, rg in mism[:5]:
                        print(f"   move view {s}->{d}: engine gives_check={eg}, referee={rg}")
    print(f"\n完成: 测试局面 {positions_tested}, 漏判/误判总数 {total_mismatch}")
    import json as _json
    with open("tools/games/gc_dump.json", "w", encoding="utf-8") as _f:
        _json.dump(_DUMP, _f, ensure_ascii=False)


if __name__ == "__main__":
    main()
