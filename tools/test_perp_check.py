# -*- coding: utf-8 -*-
"""长将检测回归测试 (分支 fix/perp-check-v514)

锁定三处修复, 任何一处回退都会让本测试变红:

  1. Position.rotate() 垫片补回"翻盘 + swapcase"。
     此前只翻 stm/turn 不动盘面, 导致 referee.in_check 判据
     `board[m[1]] == "k"` 变成在问"对方能否吃到自己的将" —— 恒为 False。
     后果: 裁判连将配额永不累计, 长将判负从不触发, 所有"规则零判负"
     的对局结论都失去意义。

  2. _gives_check 语义修正 + 暗子真身穷举。
     - 语义: can_capture_king(side) 是"side 能否吃掉对方王", make() 后 stm
       已指向对方, 原写法 can_capture_king(st.stm) 算的是"我方走完是否被将
       军", 与函数名相反。对合法着法几乎恒为 False, 长将安检被
       `if cand_chk and ...` 短路跳过。
     - 暗子: 暗子(DEFGHI)与 U 子走完会被写成 U, 而 gen_moves 跳过 U, 其将军
       完全不可见。揭棋绝大多数着法都是暗子首着, 漏判面积极大。

  3. _memory_quota_blocks 按轨迹认子, 与裁判 PerpCheckTracker 同口径。
     原实现只对候选着的那个子做轨迹前移, 段内其余将军子按落点格子计数 ——
     同一子换格照将被记成多个将军子, 配额由 3 虚高到 9, 恰好放过最典型的
     "双子交替来回照将"长将形态。

跑法:
    .venv/bin/python tools/test_perp_check.py
"""
import os
import sys
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

import referee as R          # noqa: E402
import jieqi_engine as E     # noqa: E402

_FAILS = []


def check(name, cond, detail=""):
    print(("  [OK]   " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        _FAILS.append(name)


def mk(pieces):
    """用 {(row, col): piece} 造一块 10x9 棋盘 (规范视角, 红在下)。"""
    b = [["."] * 9 for _ in range(10)]
    for (r, c), p in pieces.items():
        b[r][c] = p
    return b


def warm(eng, board, side="r"):
    """初始化全局评估表 (State._pstats 依赖 average), 返回引擎串。"""
    estr = E.board_to_engine_string(board, side)
    eng._pool.observe(estr)
    eng._pool.apply(estr)
    eng.searcher.calc_average()
    return estr


def rc(r, c):
    return E._row_col_to_engine_idx(r, c)


def indep_gives_check(board, secret, side, src, dst):
    """真值基准: 在裁判掌握真身的真实棋盘上走完, 对方是否被将军。"""
    nb = [row[:] for row in board]
    ns = dict(secret)
    R.apply_move(nb, ns, side, src, dst)
    return R.in_check(nb, "b" if side == "r" else "r")


# ---------------------------------------------------------------- 1. 裁判将军判定
def test_referee_in_check():
    print("\n[1] referee.in_check (依赖 Position.rotate 垫片翻盘)")
    b = mk({(9, 4): "r帥", (0, 4): "b將", (4, 4): "r車"})
    check("黑将(0,4) 被红车(4,4) 照将 -> True", R.in_check(b, "b") is True)
    check("红帅(9,4) 未被将军     -> False", R.in_check(b, "r") is False)


# ---------------------------------------------------------------- 2. _gives_check 语义
def test_gives_check_semantics():
    print("\n[2] _gives_check 语义 (我方将军对方, 而非我方被将军)")
    eng = E.JieQiEngine()

    # 2a 明子照将
    b = mk({(9, 4): "r帥", (0, 4): "b將", (5, 4): "r車"})
    estr = warm(eng, b)
    st = E.State.from_string(estr)
    mv = (rc(5, 4), rc(4, 4))
    check("红车 (5,4)->(4,4) 照将黑将 -> True",
          eng._gives_check(E.State.from_string(estr), mv) is True,
          f"(着法合法={mv in set(st.gen_moves())})")

    # 2b 明子让开导致自己被将 —— 不是将军, 应为 False
    b = mk({(9, 3): "r帥", (0, 4): "b將", (5, 3): "b車", (7, 3): "r車"})
    estr = warm(eng, b)
    st = E.State.from_string(estr)
    mv = (rc(7, 3), rc(7, 5))
    check("红车 (7,3)->(7,5) 让开致红帅被照 -> False (非将军)",
          eng._gives_check(E.State.from_string(estr), mv) is False,
          f"(着法合法={mv in set(st.gen_moves())})")


# ---------------------------------------------------------------- 3. 暗子真身穷举
def test_dark_piece_reveal():
    """暗子首着走完变 U 并被 gen_moves 跳过, 必须穷举真身, 否则整棵漏判。"""
    print("\n[3] 暗子首着的将军判定 (穷举可能真身 RNBACP)")
    eng = E.JieQiEngine()

    # 3a (6,4) 是合法暗子位 (暗兵 I); 真身若是車, 走到 (5,4) 即照将
    b = mk({(9, 4): "r帥", (0, 4): "b將", (6, 4): "r?"})
    estr = warm(eng, b)
    st = E.State.from_string(estr)
    mv = (rc(6, 4), rc(5, 4))
    legal = mv in set(st.gen_moves())
    check("暗子 (6,4)->(5,4): 真身可能是車, 应判将军 -> True",
          (not legal) or eng._gives_check(E.State.from_string(estr), mv) is True,
          f"(着法合法={legal})")

    # 3b (6,0) 走到 (5,0): R/N/B/A/C/P 无一能照到 (0,3) 的黑将。
    #     黑将放 (0,3) 而非 (0,4): 避免与红帅 (9,4) 同列构成"白脸将" —— 那样
    #     第 5 列一空, 任何红方着法都会因对脸判将, 用例失去区分力。
    b = mk({(9, 4): "r帥", (0, 3): "b將", (6, 0): "r?"})
    estr = warm(eng, b)
    st = E.State.from_string(estr)
    mv = (rc(6, 0), rc(5, 0))
    legal = mv in set(st.gen_moves())
    check("暗子 (6,0)->(5,0): 无真身可照将 -> False",
          (not legal) or eng._gives_check(E.State.from_string(estr), mv) is False,
          f"(着法合法={legal})")


# ---------------------------------------------------------------- 4. 差分: 漏判必须为 0
def test_differential(trials=60):
    """对大量可达局面, 与裁判真值比对。

    暗子真身引擎不可知, 故采用"只多判不漏判"策略: 允许误判(引擎说将军而真值
    不是), 但**漏判必须为 0** —— 漏判即长将失控。
    """
    print(f"\n[4] 差分测试 ({trials} 局随机中局, 与裁判真值比对)")
    eng = E.JieQiEngine()
    random.seed(20260818)
    total = missed = falsepos = ref_true = 0
    for trial in range(trials):
        board, secret = R.new_game(5000 + trial)
        for ply in range(random.randint(5, 30)):
            side = "r" if ply % 2 == 0 else "b"
            view = board if side == "r" else R.flip_board(board)
            moves = list(R.legal_engine_moves(view, side))
            if not moves:
                break
            m = random.choice(moves)
            sv, dv = R.uci_to_view_rc(E._engine_idx_to_uci(m[0]) + E._engine_idx_to_uci(m[1]))
            src = sv if side == "r" else R.rc_flip(sv)
            dst = dv if side == "r" else R.rc_flip(dv)
            cap, _ = R.apply_move(board, secret, side, src, dst)
            if cap in R.KING.values():
                break

        for side in ("r", "b"):
            view = board if side == "r" else R.flip_board(board)
            estr = warm(eng, view, side)
            for mv in list(E.State.from_string(estr).gen_moves()):
                src_rc = E._engine_idx_to_row_col(mv[0])
                dst_rc = E._engine_idx_to_row_col(mv[1])
                csrc = src_rc if side == "r" else R.rc_flip(src_rc)
                cdst = dst_rc if side == "r" else R.rc_flip(dst_rc)
                ref = indep_gives_check(board, secret, side, csrc, cdst)
                got = eng._gives_check(E.State.from_string(estr), mv)
                total += 1
                if ref:
                    ref_true += 1
                if ref and not got:
                    missed += 1
                elif got and not ref:
                    falsepos += 1

    # 基准自检: 真值必须真的判出过将军。若 referee.in_check 恒为 False
    # (rotate 垫片漏翻盘时的症状), 样本里 ref_true 会是 0, 于是"漏判=0"
    # 变成恒真的假阴性 —— 必须挡住。
    check(f"真值基准自检: 样本中确有将军着法 (>0, 否则基准不可信)",
          ref_true > 0, f"真值判出将军 {ref_true} 次")
    check(f"漏判数 == 0 (共比对 {total} 个着法)", missed == 0 and ref_true > 0,
          f"漏判={missed}, 误判(允许)={falsepos} ({falsepos / max(total, 1) * 100:.1f}%)")


# ---------------------------------------------------------------- 5. 配额按轨迹认子
def _mem_engine(moves):
    """手工构造实战模式记忆链: moves 全是连续将军着。"""
    eng = E.JieQiEngine()
    n = len(moves)
    eng._mem_side = "r"
    eng._mem_boards = [""] * (n + 1)
    eng._mem_moves = list(moves)
    eng._mem_my_checks = [True] * n
    eng._mem_oppo_checks = [False] * (n + 1)
    eng._mem_oppo_dsts = [None] * (n + 1)
    return eng


def test_quota_trajectory():
    """同一子换格照将只应算 1 个将军子 (配额 3), 不得按格子累加虚高到 9。"""
    print("\n[5] _memory_quota_blocks 按轨迹认子 (与裁判 PerpCheckTracker 同口径)")

    # 5a 单子连将 3 次 + 候选 = 4 > min(3x1, 9) -> 拦截
    eng = _mem_engine([(rc(5, 0), rc(5, 1)), (rc(5, 1), rc(5, 2)), (rc(5, 2), rc(5, 3))])
    check("单子连将 4 次 -> 拦截 (配额 3)",
          eng._memory_quota_blocks((rc(5, 3), rc(5, 4)), True) is True)

    # 5b 只连将 2 次 -> 不拦截, 避免误伤正常进攻
    eng = _mem_engine([(rc(5, 0), rc(5, 1))])
    check("单子连将 2 次 -> 不拦截",
          eng._memory_quota_blocks((rc(5, 1), rc(5, 2)), True) is False)

    # 5c 双子交替连将 9 次: 真实将军子 2 个 -> 配额 6, 应拦截。
    #    修复前段内非候选子的 4 次移动被记成 4 个将军子, 配额 min(3x5,9)=9,
    #    count=9 不满足 9>9 -> 放过。这正是"双子交替长将"逃逸的路径。
    seq = []
    for k in range(4):
        seq.append((rc(5, k), rc(5, k + 1)))       # A 子沿 row5
        seq.append((rc(6, k), rc(6, k + 1)))       # B 子沿 row6
    eng = _mem_engine(seq)
    check("双子交替连将 9 次 -> 拦截 (配额 6, 修复前配额虚高到 9 而放过)",
          eng._memory_quota_blocks((rc(5, 4), rc(5, 5)), True) is True)


def main():
    print("=" * 68)
    print("长将检测回归测试")
    print("=" * 68)
    test_referee_in_check()
    test_gives_check_semantics()
    test_dark_piece_reveal()
    test_differential()
    test_quota_trajectory()
    print()
    print("=" * 68)
    if _FAILS:
        print(f"失败 {len(_FAILS)} 项: " + "; ".join(_FAILS))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
