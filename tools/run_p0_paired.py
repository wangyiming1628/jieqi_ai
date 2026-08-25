"""
配对对局 (paired / gauntlet-swap) A/B: P0 vs v5.11

动机: 揭棋先手优势极大 (前两批 20 局实测红方得分 75%), 且每局暗子洗牌不同,
      两者共同制造巨大方差, 淹没评估改动的信号。
      配对设计把"种子 + 颜色"这两个混淆因素成对抵消: 同一个暗子布局各打两遍,
      两个引擎各执红一次。每对的净结果只反映引擎差异。

每对判定:
    P0 执红胜 + P0 执黑胜  →  P0 净胜 (+1)   两种布局都赢, 与颜色无关
    P0 执红负 + P0 执黑负  →  P0 净负 (-1)
    一胜一负 / 含和棋      →  该对打平 ( 0)   通常意味着"红方都赢"= 纯颜色效应

用法:
    python tools/run_p0_paired.py --pairs 10 --think-time 1.0
"""
import sys, os, time, json, argparse, io, contextlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import referee  # noqa: E402

NEW, OLD = "pypy", "pypy511"


class Args:
    def __init__(self, red, black, think, seed, max_ply, no_cap):
        self.red, self.black = red, black
        self.think_time = self.red_think = self.black_think = think
        self.seed, self.max_ply, self.no_cap_draw = seed, max_ply, no_cap


def one(new_is_red, think, seed, max_ply, no_cap):
    a = Args(NEW if new_is_red else OLD, OLD if new_is_red else NEW,
             think, seed, max_ply, no_cap)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        board, recs, stats, res, wall = referee.play(a)
    w = res["winner"]
    out = ("draw" if w is None else
           "win" if (w == "r") == new_is_red else "loss")
    ns = stats["r" if new_is_red else "b"]
    os_ = stats["b" if new_is_red else "r"]
    dn = [r["depth"] for r in recs
          if r["side"] == ("r" if new_is_red else "b")
          and isinstance(r.get("depth"), int) and r["depth"] > 0]
    do = [r["depth"] for r in recs
          if r["side"] == ("b" if new_is_red else "r")
          and isinstance(r.get("depth"), int) and r["depth"] > 0]
    return {"outcome": out, "reason": res["reason"], "plies": len(recs),
            "p0_time": round(ns["total"], 1), "p0_moves": ns["n"],
            "v511_time": round(os_["total"], 1), "v511_moves": os_["n"],
            "p0_depth": round(sum(dn) / len(dn), 2) if dn else 0,
            "v511_depth": round(sum(do) / len(do), 2) if do else 0,
            "moves": recs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--think-time", type=float, default=1.0)
    ap.add_argument("--max-ply", type=int, default=400)
    ap.add_argument("--no-cap-draw", type=int, default=120)
    ap.add_argument("--base-seed", type=int, default=910000)
    ap.add_argument("--out", default=os.path.join(REPO, "tools", "games",
                                                 "p0_paired.json"))
    a = ap.parse_args()

    print(f"[*] 配对对局: {a.pairs} 对 = {a.pairs * 2} 局, {a.think_time}s/着")
    print(f"[*] 每对同一 seed 打两遍, 两引擎各执红一次 (抵消先手+布局方差)")
    print(f"[*] 新版 {NEW} (P0)  vs  基线 {OLD} (v5.11)", flush=True)

    pairs = []
    t0 = time.perf_counter()
    for i in range(1, a.pairs + 1):
        seed = a.base_seed + i * 7
        print(f"\n{'=' * 66}\n[对 {i}/{a.pairs}] seed={seed}", flush=True)
        g1 = one(True, a.think_time, seed, a.max_ply, a.no_cap_draw)
        print(f"  A) P0 执红: {g1['outcome']:4s} ({g1['plies']} 着) {g1['reason'][:40]}",
              flush=True)
        g2 = one(False, a.think_time, seed, a.max_ply, a.no_cap_draw)
        print(f"  B) P0 执黑: {g2['outcome']:4s} ({g2['plies']} 着) {g2['reason'][:40]}",
              flush=True)
        sc = {"win": 1.0, "draw": 0.5, "loss": 0.0}
        net = sc[g1["outcome"]] + sc[g2["outcome"]] - 1.0
        verdict = "P0 净胜" if net > 0 else "P0 净负" if net < 0 else "该对打平"
        print(f"  → {verdict} (net {net:+.1f})", flush=True)
        pairs.append({"pair": i, "seed": seed, "net": net,
                      "p0_red": g1, "p0_black": g2})
        nw = sum(1 for p in pairs if p["net"] > 0)
        nl = sum(1 for p in pairs if p["net"] < 0)
        nd = sum(1 for p in pairs if p["net"] == 0)
        print(f"  [累计] 净胜 {nw} 对, 净负 {nl} 对, 打平 {nd} 对", flush=True)

    # ---- 汇总 ----
    games = [p["p0_red"] for p in pairs] + [p["p0_black"] for p in pairs]
    W = sum(1 for g in games if g["outcome"] == "win")
    L = sum(1 for g in games if g["outcome"] == "loss")
    D = sum(1 for g in games if g["outcome"] == "draw")
    n = len(games)
    nw = sum(1 for p in pairs if p["net"] > 0)
    nl = sum(1 for p in pairs if p["net"] < 0)
    nd = sum(1 for p in pairs if p["net"] == 0)

    print(f"\n\n{'#' * 66}")
    print(f"# 配对对局汇总: {a.pairs} 对 / {n} 局 ({a.think_time}s/着)")
    print(f"{'#' * 66}")
    print(f"\n逐局战绩 (P0 视角): {W}胜 {L}负 {D}平  "
          f"得分 {W + .5 * D:.1f}/{n} = {(W + .5 * D) / n * 100:.1f}%")
    print(f"配对结果:  P0 净胜 {nw} 对 | 净负 {nl} 对 | 打平 {nd} 对")

    red_w = sum(1 for p in pairs for g, isred in ((p["p0_red"], True),
                                                  (p["p0_black"], False))
                if g["outcome"] != "draw"
                and ((g["outcome"] == "win") == isred))
    print(f"\n颜色效应校验: 红方共胜 {red_w}/{n - D} 决胜局 "
          f"({red_w / max(n - D, 1) * 100:.0f}%)  ← 配对设计已将其抵消")

    print(f"\n{'对':>3s} {'seed':>8s} {'P0执红':>7s} {'P0执黑':>7s} {'净':>5s}")
    for p in pairs:
        print(f"{p['pair']:3d} {p['seed']:8d} {p['p0_red']['outcome']:>8s} "
              f"{p['p0_black']['outcome']:>8s} {p['net']:+5.1f}")

    tp = sum(g["p0_time"] for g in games); mp = sum(g["p0_moves"] for g in games)
    to = sum(g["v511_time"] for g in games); mo = sum(g["v511_moves"] for g in games)
    dn = [g["p0_depth"] for g in games if g["p0_depth"]]
    do = [g["v511_depth"] for g in games if g["v511_depth"]]
    print(f"\n算力对称性: 用时/着 P0 {tp / mp:.3f}s vs v511 {to / mo:.3f}s "
          f"(偏差 {abs(tp / mp - to / mo) / (to / mo) * 100:.1f}%)")
    print(f"            平均深度 P0 {sum(dn) / len(dn):.2f} vs v511 "
          f"{sum(do) / len(do):.2f} (差 {sum(dn) / len(dn) - sum(do) / len(do):+.2f})")
    print(f"\n总墙钟 {(time.perf_counter() - t0) / 60:.1f} 分钟")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"config": {"pairs": a.pairs, "think_time": a.think_time,
                              "base_seed": a.base_seed},
                   "summary": {"games": n, "win": W, "loss": L, "draw": D,
                               "pair_win": nw, "pair_loss": nl, "pair_draw": nd,
                               "p0_avg_s": round(tp / mp, 3),
                               "v511_avg_s": round(to / mo, 3),
                               "p0_depth": round(sum(dn) / len(dn), 2),
                               "v511_depth": round(sum(do) / len(do), 2)},
                   "pairs": pairs}, f, ensure_ascii=False, indent=1)
    print(f"[*] 已保存 {a.out}")


if __name__ == "__main__":
    main()
