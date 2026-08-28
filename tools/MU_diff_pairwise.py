"""[v5.14 移植对拍] 跨实现 alphabeta 逐字对比: 新版 jieqi_engine (State + make/unmake)
vs 旧版 jieqi_engine_v512 (Position + put/rotate)。

按 tools/MAKE_UNMAKE_notes_20260827.md 的方法学:
  1) 关 LMR + 关 nullmove, 跑固定深度, 逐着法对比 value() (期望逐字相等)
  2) 跑 alphabeta(root=True), 对比根值 (期望逐字相等)
  3) 跑 alphabeta(root=False) 单节点, 对比单节点值
  4) 跑若干随机局面, 上面三个全过才算移植成功

注意: 两版的 di/sumall/average 计算口径一致 (P0 后的 v5.12 = v5.13 风险惩罚默认关闭
与 v5.14 风险惩罚默认关闭完全一致), 所以只要双方都用 _update_distribution(P0 之前
的逻辑) 调用, di 即可对齐, 而 v5.14 默认会走 DarkPoolTracker 路径, 需注意切换。
本次对拍两边都先手工重置 di 为 P0 之前的形式以保证一致。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy
import jieqi_engine as NEW
import jieqi_engine_v512 as OLD
from referee import new_game, apply_move, flip_board, legal_engine_moves

# --- 帮助函数: 用双方一致的 (P0 之前) 方式更新 di/sumall, 保证双方读到同一分布 ---
KING = ("r" + chr(0x5e05), "b" + chr(0x5c06))


def idx_to_rc(idx):
    return idx // 16 - 3, idx % 16 - 3


def set_pool_uniform(estr):
    """手动把 di/sumall 重置为开局标准池 (用于双方引擎对拍时池分布对齐)"""
    global NEW_di, OLD_di
    # v512 的 _update_distribution 是单帧反推; 新版覆盖为 P0 后, 这里两边都强制重置 di
    r = {"R": 2, "N": 2, "B": 2, "A": 2, "C": 2, "P": 5}
    b = {"r": 2, "n": 2, "b": 2, "a": 2, "c": 2, "p": 5}
    # 从盘面扣除明子, 与 P0 之前算法一致
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
    """构造一个随机局面: 从初始盘面走 ply_max 步, 返回 engine_board_str"""
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


def M(x):
    return 254 - x


def diff_one(estr, side, depth):
    """关 LMR + 关 nullmove, 对比两版: 走法序列 / value 逐字 / alphabeta(root) / 节点值"""
    set_pool_uniform(estr)
    # 关 LMR + nullmove
    s_old = OLD.Searcher()
    s_old.lmr_min_depth = 999
    s_old.lmr_base_reduction = 0
    p1 = OLD.Position(estr, 0, True, 0).set()
    s_old.calc_average()   # value() 依赖 average 全局

    s_new = NEW.Searcher()
    s_new.lmr_min_depth = 999
    s_new.lmr_base_reduction = 0
    s_new.calc_average()
    if NEW.RISK_LAMBDA:
        s_new.calc_variance()
    st2 = NEW.State.from_string(estr)

    # 1) 走法序列: depth=0 层, 不需要映射 (两边根都是大写=己方视角, 不旋转)
    m1 = list(p1.gen_moves())
    m2 = list(st2.gen_moves())
    if sorted(m1) != sorted(m2):
        return False, f"走法序列不一致 (关LMR):\n  旧={m1[:5]}\n  新={m2[:5]}"

    # 2) 每个着法的 value 逐字对比
    for mv in m1:
        v1 = p1.value(mv)
        v2 = st2.value(mv[0], mv[1])
        if abs(v1 - v2) > 1e-9:
            return False, f"value 不一致 mv={mv}: 旧={v1} 新={v2}"

    # 3) 固定深度 alphabeta root 逐字对比
    s_old.tp_score, s_old.tp_move, s_old.history_heur, s_old.deadline = {}, {}, {}, 0.0
    s_new.tp_score, s_new.tp_move, s_new.history_heur, s_new.deadline = {}, {}, {}, 0.0
    r1 = s_old.alphabeta(p1, -OLD.MATE_UPPER, OLD.MATE_UPPER, depth, root=True,
                         nullmove=False, nullmove_now=False)
    r2 = s_new.alphabeta(st2, -NEW.MATE_UPPER, NEW.MATE_UPPER, depth, root=True,
                         nullmove=False, nullmove_now=False)
    if r1 != r2:
        return False, f"根值不一致 depth={depth}: 旧={r1} 新={r2}"
    bm1 = s_old.tp_move.get(p1)
    bm2 = s_new.tp_move.get((st2.hkey, st2.score))
    if bm1 != bm2:
        return False, f"最佳着法不一致 depth={depth}: 旧={bm1} 新={bm2}"

    # 4) 每个子节点的 alphabeta 逐字对比 (确保子树决策一致)
    for mv in m1:
        s_old.tp_score, s_old.tp_move, s_old.history_heur, s_old.deadline = {}, {}, {}, 0.0
        s_new.tp_score, s_new.tp_move, s_new.history_heur, s_new.deadline = {}, {}, {}, 0.0
        v1c = -s_old.alphabeta(p1.move(mv), -OLD.MATE_UPPER, OLD.MATE_UPPER, depth - 1,
                                root=False, nullmove=False, nullmove_now=False)
        st2.make(mv[0], mv[1])
        v2c = -s_new.alphabeta(st2, -NEW.MATE_UPPER, NEW.MATE_UPPER, depth - 1,
                                root=False, nullmove=False, nullmove_now=False)
        st2.unmake()
        if v1c != v2c:
            return False, f"子节点 alphabeta 不一致 mv={mv} depth={depth-1}: 旧={v1c} 新={v2c}"

    return True, f"逐字相等 (depth={depth}, moves={len(m1)}, value+alphabeta 全过)"


if __name__ == "__main__":
    n_pos = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    max_d = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    fail = 0
    ok = 0
    for k in range(n_pos):
        seed = 1 + k
        ply = (k * 3) % 8 + 2   # 2-9 步开局
        res = random_position(seed, ply)
        if res is None:
            print(f"  [SKIP] seed={seed} ply={ply} (国吃)")
            continue
        estr, side = res
        for d in range(2, max_d + 1):
            passed, msg = diff_one(estr, side, d)
            if passed:
                ok += 1
            else:
                fail += 1
                print(f"  [FAIL] seed={seed} ply={ply} side={side} depth={d}: {msg}")
    print(f"\n=== 总结: {ok} 通过 / {fail} 失败 ===")
