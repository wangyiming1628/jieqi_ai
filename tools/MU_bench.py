"""[v5.14 移植吞吐基准] 固定深度 6, 对比新旧 jieqi_engine 节点吞吐 (knps)。
    - 用同一开局局面, 全新 TT, 不限时
    - 双方都用各自 Searcher (类型不同), 各跑 3 次取中位数避免抖动
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jieqi_engine as NEW
import jieqi_engine_v512 as OLD

DEPTH = 8
RUNS = 3


def midgame_estr():
    """更深的中局: 双方各剩 10 子, 走子已发生很多, 暗子翻开率高"""
    board = [["."] * 9 for _ in range(10)]
    board[9][4] = "r帥"; board[0][4] = "b將"
    board[7][4] = "r車"; board[6][4] = "r馬"; board[5][3] = "r炮"
    board[4][2] = "r兵"; board[3][2] = "r兵"; board[2][2] = "r兵"
    board[9][2] = "r相"; board[9][6] = "r相"
    board[2][4] = "b車"; board[3][4] = "b馬"; board[4][5] = "b炮"
    board[5][6] = "b卒"; board[5][2] = "b卒"; board[6][6] = "b卒"
    board[0][2] = "b象"; board[0][6] = "b象"
    board[1][3] = "b?"; board[1][5] = "b?"; board[8][1] = "r?"; board[8][7] = "r?"
    return OLD.board_to_engine_string(board, "r")


def median(lst):
    s = sorted(lst)
    return s[len(s) // 2]


def run_old(estr):
    times, nodes_list = [], []
    for _ in range(RUNS):
        OLD._update_distribution(estr)   # P0 之前的单帧池, 与 bench_v2.py 一致
        s = OLD.Searcher()
        s.calc_average()
        s.tp_score, s.tp_move, s.history_heur, s.deadline = {}, {}, {}, 0.0
        p = OLD.Position(estr, 0, True, 0).set()
        t0 = time.time()
        s.alphabeta(p, -OLD.MATE_UPPER, OLD.MATE_UPPER, DEPTH, nullmove=True, nullmove_now=True)
        dt = time.time() - t0
        times.append(dt)
        nodes_list.append(s.nodes)
    return times, nodes_list


def run_new(estr):
    """新版走与旧版完全一致的池初始化路径 (P0 之前的单帧反推), 以便公平对比
    节点吞吐, 不引入 P0 DarkPoolTracker 的额外开销。"""
    import copy
    times, nodes_list = [], []
    for _ in range(RUNS):
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
        s.tp_score, s.tp_move, s.history_heur, s.deadline = {}, {}, {}, 0.0
        st = NEW.State.from_string(estr)
        t0 = time.time()
        s.alphabeta(st, -NEW.MATE_UPPER, NEW.MATE_UPPER, DEPTH, nullmove=True, nullmove_now=True)
        dt = time.time() - t0
        times.append(dt)
        nodes_list.append(s.nodes)
    return times, nodes_list


if __name__ == "__main__":
    estr = midgame_estr()
    print(f"=== 固定深度 {DEPTH} 吞吐对比 (深中局示意局面) ===")
    t_old, n_old = run_old(estr)
    t_new, n_new = run_new(estr)
    print(f"  旧版 (v512): times={[f'{t:.2f}s' for t in t_old]} nodes={n_old}")
    print(f"  新版 (v514): times={[f'{t:.2f}s' for t in t_new]} nodes={n_new}")
    t_old_med = median(t_old)
    t_new_med = median(t_new)
    n_old_med = median(n_old)
    n_new_med = median(n_new)
    knps_old = n_old_med / t_old_med / 1000
    knps_new = n_new_med / t_new_med / 1000
    print(f"\n  旧版 中位: time={t_old_med:.2f}s nodes={n_old_med} knps={knps_old:.1f}")
    print(f"  新版 中位: time={t_new_med:.2f}s nodes={n_new_med} knps={knps_new:.1f}")
    print(f"\n  节点数差异: 新版/旧版 = {n_new_med/n_old_med:.3f}x (期望 ~1)")
    print(f"  节点吞吐提升: 新版/旧版 = {knps_new/knps_old:.2f}x")
    print(f"  单次节点时间: 旧 {1e6/knps_old/1000:.2f}μs -> 新 {1e6/knps_new/1000:.2f}μs")
