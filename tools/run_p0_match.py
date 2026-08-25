"""
P0 (暗子池跨回合跟踪) A/B 对局套件: jieqi_engine (P0) vs jieqi_engine_v511 (基线)

用法:
    python tools/run_p0_match.py                       # 10 局, 每着 1.0s
    python tools/run_p0_match.py --games 10 --think-time 1.0
    python tools/run_p0_match.py --out tools/games/p0_match.json

设计:
  - 交替先后手 (奇数局 P0 执红, 偶数局 P0 执黑), 消除先手偏差
  - 每局独立随机种子 (暗子洗牌不同), 双方等墙钟思考预算
  - 复用 tools/referee.py 的完整规则实现 (胜负/长将配额/重复判和)
  - 汇总: 胜平负、按颜色拆分、用时、步数、终局原因分布
"""
import sys, os, time, json, argparse, io, contextlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import referee  # noqa: E402

NEW = "pypy"        # jieqi_engine.py = P0 修复版
OLD = "pypy511"     # jieqi_engine_v511.py = 修复前基线


class Args:
    """referee.play() 需要的参数容器。"""

    def __init__(self, red, black, think, seed, max_ply, no_cap):
        self.red = red
        self.black = black
        self.think_time = think
        self.red_think = think
        self.black_think = think
        self.seed = seed
        self.max_ply = max_ply
        self.no_cap_draw = no_cap


def run_one(idx, new_is_red, think, seed, max_ply, no_cap, verbose):
    red = NEW if new_is_red else OLD
    black = OLD if new_is_red else NEW
    args = Args(red, black, think, seed, max_ply, no_cap)
    label = f"P0执{'红' if new_is_red else '黑'}"
    print(f"\n{'=' * 68}\n[对局 {idx}] {label}  红={red} 黑={black}  "
          f"seed={seed}  {think}s/着\n{'=' * 68}", flush=True)
    t0 = time.perf_counter()
    if verbose:
        board, records, stats, result, wall = referee.play(args)
    else:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            board, records, stats, result, wall = referee.play(args)
    elapsed = time.perf_counter() - t0

    w = result["winner"]
    if w is None:
        outcome = "draw"
    elif (w == "r") == new_is_red:
        outcome = "win"
    else:
        outcome = "loss"
    new_side = "r" if new_is_red else "b"
    old_side = "b" if new_is_red else "r"
    ns, os_ = stats[new_side], stats[old_side]
    print(f"  结果: P0 {outcome.upper():4s} | {result['reason']}")
    print(f"  步数: {len(records)}  墙钟 {elapsed:.0f}s")
    print(f"  用时: P0 {ns['total']:.0f}s/{ns['n']}着 (均 "
          f"{ns['total'] / max(ns['n'], 1):.2f}s, 最长 {ns['max']:.2f}s) | "
          f"v511 {os_['total']:.0f}s/{os_['n']}着 (均 "
          f"{os_['total'] / max(os_['n'], 1):.2f}s, 最长 {os_['max']:.2f}s)",
          flush=True)

    depths = {"new": [], "old": []}
    for rec in records:
        key = "new" if rec["side"] == new_side else "old"
        d = rec.get("depth")
        if isinstance(d, int) and d > 0:
            depths[key].append(d)

    return {
        "idx": idx, "seed": seed, "p0_side": new_side,
        "outcome": outcome, "winner": w, "reason": result["reason"],
        "plies": len(records), "wall_s": round(elapsed, 1),
        "p0_time": round(ns["total"], 1), "p0_moves": ns["n"],
        "p0_avg": round(ns["total"] / max(ns["n"], 1), 3),
        "p0_max": round(ns["max"], 2),
        "v511_time": round(os_["total"], 1), "v511_moves": os_["n"],
        "v511_avg": round(os_["total"] / max(os_["n"], 1), 3),
        "v511_max": round(os_["max"], 2),
        "p0_avg_depth": round(sum(depths["new"]) / len(depths["new"]), 2)
                        if depths["new"] else 0,
        "v511_avg_depth": round(sum(depths["old"]) / len(depths["old"]), 2)
                          if depths["old"] else 0,
        "moves": records,
    }


def summarize(rows, think):
    n = len(rows)
    W = sum(1 for r in rows if r["outcome"] == "win")
    L = sum(1 for r in rows if r["outcome"] == "loss")
    D = sum(1 for r in rows if r["outcome"] == "draw")
    print(f"\n\n{'#' * 68}\n# P0 (暗子池跟踪) vs v5.11 基线 — {n} 局汇总 "
          f"({think}s/着, 交替先后手)\n{'#' * 68}")
    print(f"\n战绩 (P0 视角): {W}胜 {L}负 {D}平   "
          f"得分 {W + 0.5 * D:.1f}/{n} = {(W + 0.5 * D) / n * 100:.0f}%")

    for side, name in (("r", "P0 执红"), ("b", "P0 执黑")):
        sub = [r for r in rows if r["p0_side"] == side]
        if sub:
            w = sum(1 for r in sub if r["outcome"] == "win")
            l = sum(1 for r in sub if r["outcome"] == "loss")
            d = sum(1 for r in sub if r["outcome"] == "draw")
            print(f"  {name}: {w}胜 {l}负 {d}平 ({len(sub)} 局)")

    print(f"\n{'局':>3s} {'P0方':>5s} {'结果':>5s} {'步数':>5s} "
          f"{'P0均时':>7s} {'v511均时':>9s} {'P0均深':>7s} {'v511均深':>9s}  原因")
    for r in rows:
        print(f"{r['idx']:3d} {('红' if r['p0_side'] == 'r' else '黑'):>4s} "
              f"{r['outcome']:>6s} {r['plies']:5d} "
              f"{r['p0_avg']:7.2f} {r['v511_avg']:9.2f} "
              f"{r['p0_avg_depth']:7.2f} {r['v511_avg_depth']:9.2f}  {r['reason'][:34]}")

    tp = sum(r["p0_time"] for r in rows)
    to = sum(r["v511_time"] for r in rows)
    mp = sum(r["p0_moves"] for r in rows)
    mo = sum(r["v511_moves"] for r in rows)
    dp = [r["p0_avg_depth"] for r in rows if r["p0_avg_depth"]]
    do = [r["v511_avg_depth"] for r in rows if r["v511_avg_depth"]]
    print(f"\n用时对称性校验 (必须接近, 否则算力不等):")
    print(f"  P0   总 {tp:7.0f}s / {mp:4d} 着 = 均 {tp / max(mp, 1):.3f}s/着")
    print(f"  v511 总 {to:7.0f}s / {mo:4d} 着 = 均 {to / max(mo, 1):.3f}s/着")
    print(f"  偏差 {abs(tp / max(mp, 1) - to / max(mo, 1)) / (to / max(mo, 1)) * 100:.1f}%")
    if dp and do:
        print(f"\n平均搜索深度: P0 {sum(dp) / len(dp):.2f}  "
              f"v511 {sum(do) / len(do):.2f}  "
              f"(差 {sum(dp) / len(dp) - sum(do) / len(do):+.2f})")
    print(f"\n平均步数: {sum(r['plies'] for r in rows) / n:.0f}")
    reasons = {}
    for r in rows:
        k = ("吃王" if "吃掉" in r["reason"] else
             "长将超配额" if "配额" in r["reason"] else
             "重复判和" if "次出现" in r["reason"] else
             "无吃子判和" if "无吃子" in r["reason"] else
             "无合法着法" if "无合法" in r["reason"] else
             "步数上限" if "最大步数" in r["reason"] else "其他")
        reasons[k] = reasons.get(k, 0) + 1
    print("终局原因分布: " + ", ".join(f"{k}×{v}" for k, v in
                                       sorted(reasons.items(), key=lambda x: -x[1])))
    return {"games": n, "win": W, "loss": L, "draw": D,
            "score_pct": round((W + 0.5 * D) / n * 100, 1),
            "p0_avg_s": round(tp / max(mp, 1), 3),
            "v511_avg_s": round(to / max(mo, 1), 3),
            "p0_avg_depth": round(sum(dp) / len(dp), 2) if dp else 0,
            "v511_avg_depth": round(sum(do) / len(do), 2) if do else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--think-time", type=float, default=1.0)
    ap.add_argument("--max-ply", type=int, default=400)
    ap.add_argument("--no-cap-draw", type=int, default=120)
    ap.add_argument("--base-seed", type=int, default=520120)
    ap.add_argument("--verbose", action="store_true", help="打印每一着")
    ap.add_argument("--out", default=os.path.join(REPO, "tools", "games",
                                                 "p0_vs_v511_match.json"))
    a = ap.parse_args()

    print(f"[*] P0 A/B 对局: {a.games} 局, {a.think_time}s/着, 交替先后手")
    print(f"[*] 新版 {NEW} (jieqi_engine.py, P0 暗子池跟踪)")
    print(f"[*] 基线 {OLD} (jieqi_engine_v511.py, git 8ca26ed 快照)", flush=True)

    rows = []
    t0 = time.perf_counter()
    for i in range(1, a.games + 1):
        row = run_one(i, new_is_red=(i % 2 == 1), think=a.think_time,
                      seed=a.base_seed + i, max_ply=a.max_ply,
                      no_cap=a.no_cap_draw, verbose=a.verbose)
        rows.append(row)
        s = summarize.__doc__  # noqa
        w = sum(1 for r in rows if r["outcome"] == "win")
        l = sum(1 for r in rows if r["outcome"] == "loss")
        d = sum(1 for r in rows if r["outcome"] == "draw")
        print(f"  [累计 {i}/{a.games}] P0 {w}胜 {l}负 {d}平", flush=True)

    agg = summarize(rows, a.think_time)
    print(f"\n总墙钟: {(time.perf_counter() - t0) / 60:.1f} 分钟")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"config": {"games": a.games, "think_time": a.think_time,
                              "base_seed": a.base_seed, "new": NEW, "old": OLD},
                   "summary": agg, "results": rows}, f, ensure_ascii=False, indent=1)
    print(f"[*] 结果已保存: {a.out}")


if __name__ == "__main__":
    main()
