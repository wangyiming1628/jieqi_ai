"""[v5.14 移植微基准] 逐节点成本对比: 跑一整局 1 步, 测 make/unmake 单次成本
(提案: 35.6μs 降到 ~5μs 应该是 5~7x 提升; 旧版实测也可能更小, 因为 PyPy 极擅长
整串切片)。
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jieqi_engine as NEW
import jieqi_engine_v512 as OLD


def opening_estr():
    board = [["."] * 9 for _ in range(10)]
    board[9][4] = "r帥"; board[0][4] = "b將"
    board[7][1] = "r車"; board[9][1] = "r馬"; board[7][7] = "r炮"
    board[6][2] = "r兵"; board[6][6] = "r兵"; board[9][2] = "r相"; board[9][6] = "r相"; board[9][3] = "r仕"
    board[2][1] = "b車"; board[0][1] = "b馬"; board[2][7] = "b炮"
    board[3][2] = "b卒"; board[3][6] = "b卒"; board[0][2] = "b象"; board[0][3] = "b士"
    board[6][4] = "r?"; board[9][0] = "r?"; board[3][4] = "b?"; board[0][7] = "b?"
    return OLD.board_to_engine_string(board, "r")


def setup_old(estr):
    OLD._update_distribution(estr)
    s = OLD.Searcher()
    s.calc_average()
    p = OLD.Position(estr, 0, True, 0).set()
    return s, p


def setup_new(estr):
    import copy
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
    s = NEW.Searcher()
    s.calc_average()
    st = NEW.State.from_string(estr)
    return s, st


def bench_old():
    estr = opening_estr()
    s, p = setup_old(estr)
    moves = list(p.gen_moves())
    m = moves[0]
    # 热身
    for _ in range(2000):
        for mv in moves[:5]:
            p.move(mv)
    N = 50000
    t0 = time.perf_counter()
    for _ in range(N):
        p.move(m)
    t1 = time.perf_counter()
    # 整局 make + 撤销 (模拟 alphabeta 进/退)
    t2 = time.perf_counter()
    for _ in range(N):
        p.move(m)
        p.move(moves[1])
        p.move(moves[2])
    t3 = time.perf_counter()
    print(f"  旧版 move() 单次: {(t1-t0)/N*1e6:.2f} μs")
    print(f"  旧版 move×3 一次 (累积): {(t3-t2)/N*1e6:.2f} μs")


def bench_new():
    estr = opening_estr()
    s, st = setup_new(estr)
    moves = list(st.gen_moves())
    m = moves[0]
    # 热身
    for _ in range(2000):
        for mv in moves[:5]:
            st.make(mv[0], mv[1])
            st.unmake()
    N = 50000
    t0 = time.perf_counter()
    for _ in range(N):
        st.make(m[0], m[1])
    t1 = time.perf_counter()
    t2 = time.perf_counter()
    for _ in range(N):
        st.make(m[0], m[1])
        st.make(moves[1][0], moves[1][1])
        st.make(moves[2][0], moves[2][1])
        st.unmake()
        st.unmake()
        st.unmake()
    t3 = time.perf_counter()
    print(f"  新版 make() 单次: {(t1-t0)/N*1e6:.2f} μs")
    print(f"  新版 make×3 + unmake×3 (累积): {(t3-t2)/N*1e6:.2f} μs")


if __name__ == "__main__":
    print("=== per-node 微基准 (PyPy JIT 热态) ===")
    bench_old()
    bench_new()
