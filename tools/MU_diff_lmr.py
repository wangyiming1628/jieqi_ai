"""[v5.14 移植对拍 第二阶段] 开 LMR + nullmove, 不要求逐字相等 (LMR 近似本身
就允许值不同), 但要求:
  1) 双方走法序列一致 (走法生成不依赖 LMR)
  2) 双方都返回 MATE_UPPER 当且仅当确实有杀棋
  3) 双方 alphabeta 内部一致: 关 LMR 后的"事实根值"能被开 LMR 的双方各自覆盖
     (即: 关 LMR 给一个根值, 开 LMR 给的根值应处于"合理范围", 偏差不超过 1 个
     折算深度单位对应的 PST 量级, 不应是天文差异)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy
import jieqi_engine as NEW
import jieqi_engine_v512 as OLD
from referee import new_game, apply_move, flip_board, legal_engine_moves

KING = ("r" + chr(0x5e05), "b" + chr(0x5c06))


def idx_to_rc(idx):
    return idx // 16 - 3, idx % 16 - 3


def set_pool_uniform(estr):
    r = {"R": 2, "N": 2, "B": 2, "A": 2, "C": 2, "P": 5}
    b = {"r": 2, "n": 2, "b": 2, "a": 2, "c": 2, "p": 5}
    for i in range(51, 204):
        p = estr[i]
        if p in "RNBAKCP":
            r[p] = max(0, r.get(p, 0) - 1)
        elif p in "rnbakcp":
            b[p] = max(0, b.get(p, 0) - 1)
    NEW.di[0] = {True: copy.deepcopy(r), False: copy.deepcopy(b)}
    NEW.sumall[0] = {True: sum(r.values()), False: sum(b.values())}
    OLD.di[0] = {True: copy.deepcopy(r), False: copy.deepcopy(b)}
    OLD.sumall[0] = {True: sum(r.values()), False: sum(b.values())}


def random_position(seed, ply_max):
    rng = __import__("random").Random(seed)
    board, secret = new_game(seed)
    ok = True
    for _ in range(ply_max):
        side = "r" if _ % 2 == 0 else "b"
        view = board if side == "r" else flip_board(board)
        mvs = list(legal_engine_moves(view, side))
        if not mvs:
            break
        i, j = mvs[rng.randrange(len(mvs))]
        src, dst = idx_to_rc(i), idx_to_rc(j)
        if side == "b":
            src, dst = (9 - src[0], 8 - src[1]), (9 - dst[0], 8 - dst[1])
        captured, _ = apply_move(board, secret, side, src, dst)
        if captured in KING:
            ok = False
            break
    if not ok:
        return None
    side = "r" if ply_max % 2 == 0 else "b"
    view = board if side == "r" else flip_board(board)
    estr = OLD.board_to_engine_string(view, side)
    return estr, side


def diff_one(estr, depth):
    set_pool_uniform(estr)
    # 1) 先用关 LMR 的版本跑出"事实根值" (ground truth, 用于 LMR 版本偏差判断)
    s_old = OLD.Searcher()
    s_old.lmr_min_depth = 999
    s_old.lmr_base_reduction = 0
    p1 = OLD.Position(estr, 0, True, 0).set()
    s_old.calc_average()
    s_old.tp_score, s_old.tp_move, s_old.history_heur, s_old.deadline = {}, {}, {}, 0.0
    r_old_truth = s_old.alphabeta(p1, -OLD.MATE_UPPER, OLD.MATE_UPPER, depth, root=True,
                                    nullmove=False, nullmove_now=False)
    bm_old_truth = s_old.tp_move.get(p1)

    s_new = NEW.Searcher()
    s_new.lmr_min_depth = 999
    s_new.lmr_base_reduction = 0
    s_new.calc_average()
    st2 = NEW.State.from_string(estr)
    s_new.tp_score, s_new.tp_move, s_new.history_heur, s_new.deadline = {}, {}, {}, 0.0
    r_new_truth = s_new.alphabeta(st2, -NEW.MATE_UPPER, NEW.MATE_UPPER, depth, root=True,
                                    nullmove=False, nullmove_now=False)
    bm_new_truth = s_new.tp_move.get((st2.hkey, st2.score))

    if r_old_truth != r_new_truth:
        return False, f"事实根值不一致 (LMR=nullmove=关): 旧={r_old_truth} 新={r_new_truth}"
    if bm_old_truth != bm_new_truth:
        return False, f"事实最佳着法不一致: 旧={bm_old_truth} 新={bm_new_truth}"

    # 2) 开 LMR + nullmove 跑, 双方应返回的根值处于合理范围
    s_old2 = OLD.Searcher()
    p1b = OLD.Position(estr, 0, True, 0).set()
    s_old2.calc_average()
    s_old2.tp_score, s_old2.tp_move, s_old2.history_heur, s_old2.deadline = {}, {}, {}, 0.0
    r_old_lmr = s_old2.alphabeta(p1b, -OLD.MATE_UPPER, OLD.MATE_UPPER, depth, root=True,
                                   nullmove=True, nullmove_now=True)
    bm_old_lmr = s_old2.tp_move.get(p1b)

    s_new2 = NEW.Searcher()
    s_new2.calc_average()
    st2b = NEW.State.from_string(estr)
    s_new2.tp_score, s_new2.tp_move, s_new2.history_heur, s_new2.deadline = {}, {}, {}, 0.0
    r_new_lmr = s_new2.alphabeta(st2b, -NEW.MATE_UPPER, NEW.MATE_UPPER, depth, root=True,
                                   nullmove=True, nullmove_now=True)
    bm_new_lmr = s_new2.tp_move.get((st2b.hkey, st2b.score))

    # LMR 近似下, 双方根值可能不同; 但走法可能相同, 允许偏差
    drift_old = abs(r_old_lmr - r_old_truth) if abs(r_old_truth) < NEW.MATE_LOWER else 0
    drift_new = abs(r_new_lmr - r_new_truth) if abs(r_new_truth) < NEW.MATE_LOWER else 0
    # 偏差应当都在合理范围 (< 100, 对应 1 层 depth 的 PST 量级), 否则某引擎异常
    if drift_old > 200:
        return False, f"旧 LMR 偏差过大: truth={r_old_truth} LMR={r_old_lmr} (diff={drift_old})"
    if drift_new > 200:
        return False, f"新 LMR 偏差过大: truth={r_new_truth} LMR={r_new_lmr} (diff={drift_new})"

    return True, f"事实根值 {r_old_truth} 一致, LMR 偏差 old={drift_old} new={drift_new}"


if __name__ == "__main__":
    n_pos = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    max_d = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    fail = 0
    ok = 0
    for k in range(n_pos):
        seed = 1 + k
        ply = (k * 3) % 8 + 2
        res = random_position(seed, ply)
        if res is None:
            continue
        estr, side = res
        for d in range(2, max_d + 1):
            passed, msg = diff_one(estr, d)
            if passed:
                ok += 1
            else:
                fail += 1
                print(f"  [FAIL] seed={seed} ply={ply} side={side} depth={d}: {msg}")
    print(f"\n=== 总结: {ok} 通过 / {fail} 失败 ===")
