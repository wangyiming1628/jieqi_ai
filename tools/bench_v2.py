"""引擎搜索基准: 直接驱动 Searcher 逐层计时, 对比 原版(每层清TT) vs 优化版(TT持久化)
   的纯搜索提速; 末尾附同 think_time 的端到端 (engine_server 子进程链路) 对比。
用法: pypy3 tools\bench_v2.py"""
import sys, os, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import jieqi_engine as e1
import jieqi_engine_v2 as e2


def make_board():
    # 合成中局: 双方各剩 11-12 个子, 明暗混合, 暗子均在合法初始格上
    b = [["."] * 9 for _ in range(10)]
    b[9][4] = "r帥"; b[0][4] = "b將"
    b[7][1] = "r車"; b[9][1] = "r馬"; b[7][7] = "r炮"
    b[6][2] = "r兵"; b[6][6] = "r兵"; b[9][2] = "r相"; b[9][6] = "r相"; b[9][3] = "r仕"
    b[2][1] = "b車"; b[0][1] = "b馬"; b[2][7] = "b炮"
    b[3][2] = "b卒"; b[3][6] = "b卒"; b[0][2] = "b象"; b[0][3] = "b士"
    b[6][4] = "r?"; b[9][0] = "r?"; b[3][4] = "b?"; b[0][7] = "b?"
    return b


def run_id(module, label, clear_each_iter, max_depth=7, quiet=False):
    board = make_board()
    estr = module.board_to_engine_string(board, "r")
    module._update_distribution(estr)
    s = module.Searcher()
    s.calc_average()
    s.deadline = 0.0  # v2: 关闭硬时限, 只测纯搜索
    pos = module.Position(estr, 0, True, 0).set()
    total, rows = 0.0, []
    for depth in range(2, max_depth + 1):
        if clear_each_iter:
            s.tp_score = {}; s.tp_move = {}; s.history_heur = {}
        s.nodes = 0
        t0 = time.time()
        s.alphabeta(pos, -module.MATE_UPPER, module.MATE_UPPER, depth, nullmove=True, nullmove_now=True)
        dt = time.time() - t0
        total += dt
        rows.append((depth, dt, s.nodes, s.tp_move.get(pos)))
    if not quiet:
        print(f"--- {label} ---")
        for depth, dt, nodes, mv in rows:
            print(f"  depth {depth}: {dt:6.3f}s  nodes={nodes:8d}  move={mv}")
        print(f"  total 2..{max_depth}: {total:.2f}s")
    return total, rows


if __name__ == "__main__":
    run_id(e1, "warmup", True, max_depth=4, quiet=True)   # JIT 预热
    t1, r1 = run_id(e1, "original (TT clear per iter)", True, max_depth=7)
    t2, r2 = run_id(e2, "v2 (TT persist)", False, max_depth=7)
    print(f"\nspeedup total: {t1 / t2:.2f}x")
    print("depth | original |    v2")
    for (d1, dt1, n1, _), (d2, dt2, n2, _) in zip(r1, r2):
        print(f"  {d1:2d}  | {dt1:7.3f}s | {dt2:7.3f}s  ({dt1/dt2:.1f}x)")

    # 端到端 (同 think_time, 真实对局链路)
    from engine_client import PypyEngineClient
    print("\n=== end-to-end via server, think_time=1.0 ===")
    board = make_board()
    for name, server in [("original", "engine_server.py"),
                         ("v2      ", "engine_server_v2.py")]:
        c = PypyEngineClient(prefer_pypy=True, server_path=os.path.join(REPO, server))
        c.get_best_move(board, "r", think_time=1.0)  # 预热
        t0 = time.time()
        uci, score, depth = c.get_best_move(board, "r", think_time=1.0)
        print(f"  {name}: uci={uci} score={score} depth={depth} elapsed={time.time()-t0:.2f}s")
        c.close()
