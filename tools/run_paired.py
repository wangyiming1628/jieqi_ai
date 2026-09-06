"""
配对对局 (paired) A/B 套件 - 通用版

动机: 揭棋先手优势极大, 且每局暗子洗牌不同, 两者共同制造巨大方差, 淹没评估改动
      的信号。实测同样两个引擎, 未配对的两批 10 局给出 75% 和 55%; 20 局合并后
      按颜色拆分为"执红 90% / 执黑 40%", 红方总得分 75% —— 颜色是最强的单一因素。
      配对设计把"种子 + 颜色"这两个混淆因素成对抵消: 同一暗子布局各打两遍,
      两引擎各执红一次。改用配对后同一对比从 65% 回落到真实的 52.5%。

每对判定:
    新版执红胜 + 新版执黑胜  →  净胜 (+1)   两种布局都赢, 与颜色无关
    新版执红负 + 新版执黑负  →  净负 (-1)
    一胜一负 / 含和棋        →  打平 ( 0)   通常意味着"红方都赢"= 纯颜色效应

判据说明: 20 配对局的可分辨下限约 200+ Elo, 只能检出巨大改动。次要项改动
      (只改变个位数百分比的决策) 测不出显著性属预期, 不要据此宣称"有提升"。

用法:
    python tools/run_paired.py --pairs 10
    # 指定基线 kind (需 referee.make_engine 支持, 旧版本引擎已随清理移除):
    python tools/run_paired.py --new pypy --old pypy --pairs 10
"""
import sys, os, time, json, argparse, io, contextlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import referee  # noqa: E402


class Args:
    def __init__(self, red, black, think, seed, max_ply, no_cap):
        self.red, self.black = red, black
        self.think_time = self.red_think = self.black_think = think
        self.seed, self.max_ply, self.no_cap_draw = seed, max_ply, no_cap


def one(new_kind, old_kind, new_is_red, think, seed, max_ply, no_cap):
    a = Args(new_kind if new_is_red else old_kind,
             old_kind if new_is_red else new_kind,
             think, seed, max_ply, no_cap)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        board, recs, stats, res, wall = referee.play(a)
    w = res["winner"]
    out = ("draw" if w is None else
           "win" if (w == "r") == new_is_red else "loss")
    ns = stats["r" if new_is_red else "b"]
    os_ = stats["b" if new_is_red else "r"]
    ds = {"new": [], "old": []}
    for r in recs:
        k = "new" if (r["side"] == ("r" if new_is_red else "b")) else "old"
        d = r.get("depth")
        if isinstance(d, int) and d > 0:
            ds[k].append(d)
    return {"outcome": out, "reason": res["reason"], "plies": len(recs),
            "new_time": round(ns["total"], 1), "new_moves": ns["n"],
            "old_time": round(os_["total"], 1), "old_moves": os_["n"],
            "new_depth": round(sum(ds["new"]) / len(ds["new"]), 2) if ds["new"] else 0,
            "old_depth": round(sum(ds["old"]) / len(ds["old"]), 2) if ds["old"] else 0,
            "moves": recs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="pypy", help="新版引擎 kind (referee 的选项名)")
    ap.add_argument("--old", default="pypy", help="基线引擎 kind")
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--think-time", type=float, default=1.0)
    ap.add_argument("--max-ply", type=int, default=400)
    ap.add_argument("--no-cap-draw", type=int, default=120)
    ap.add_argument("--base-seed", type=int, default=330000)
    ap.add_argument("--tag", default="paired")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_path = a.out or os.path.join(REPO, "tools", "games", f"{a.tag}.json")

    print(f"[*] 配对对局 [{a.tag}]: {a.pairs} 对 = {a.pairs * 2} 局, {a.think_time}s/着")
    print(f"[*] 新版 {a.new}  vs  基线 {a.old}")
    print(f"[*] 每对同一 seed 打两遍, 两引擎各执红一次", flush=True)

    pairs = []
    t0 = time.perf_counter()
    for i in range(1, a.pairs + 1):
        seed = a.base_seed + i * 13
        print(f"\n{'=' * 66}\n[对 {i}/{a.pairs}] seed={seed}", flush=True)
        g1 = one(a.new, a.old, True, a.think_time, seed, a.max_ply, a.no_cap_draw)
        print(f"  A) 新版执红: {g1['outcome']:4s} ({g1['plies']} 着) {g1['reason'][:38]}",
              flush=True)
        g2 = one(a.new, a.old, False, a.think_time, seed, a.max_ply, a.no_cap_draw)
        print(f"  B) 新版执黑: {g2['outcome']:4s} ({g2['plies']} 着) {g2['reason'][:38]}",
              flush=True)
        sc = {"win": 1.0, "draw": 0.5, "loss": 0.0}
        net = sc[g1["outcome"]] + sc[g2["outcome"]] - 1.0
        verd = "新版净胜" if net > 0 else "新版净负" if net < 0 else "该对打平"
        print(f"  → {verd} (net {net:+.1f})", flush=True)
        pairs.append({"pair": i, "seed": seed, "net": net,
                      "new_red": g1, "new_black": g2})
        nw = sum(1 for p in pairs if p["net"] > 0)
        nl = sum(1 for p in pairs if p["net"] < 0)
        nd = sum(1 for p in pairs if p["net"] == 0)
        print(f"  [累计] 净胜 {nw} | 净负 {nl} | 打平 {nd}", flush=True)

    games = [p["new_red"] for p in pairs] + [p["new_black"] for p in pairs]
    W = sum(1 for g in games if g["outcome"] == "win")
    L = sum(1 for g in games if g["outcome"] == "loss")
    D = sum(1 for g in games if g["outcome"] == "draw")
    n = len(games)
    nw = sum(1 for p in pairs if p["net"] > 0)
    nl = sum(1 for p in pairs if p["net"] < 0)
    nd = sum(1 for p in pairs if p["net"] == 0)

    print(f"\n\n{'#' * 66}")
    print(f"# 配对汇总 [{a.tag}]: {a.pairs} 对 / {n} 局 ({a.think_time}s/着)")
    print(f"# 新版 {a.new}  vs  基线 {a.old}")
    print(f"{'#' * 66}")
    print(f"\n逐局战绩 (新版视角): {W}胜 {L}负 {D}平  "
          f"得分 {W + .5 * D:.1f}/{n} = {(W + .5 * D) / n * 100:.1f}%")
    print(f"配对结果: 净胜 {nw} 对 | 净负 {nl} 对 | 打平 {nd} 对")
    red_w = sum(1 for p in pairs for g, isred in ((p["new_red"], True),
                                                  (p["new_black"], False))
                if g["outcome"] != "draw" and ((g["outcome"] == "win") == isred))
    print(f"颜色效应校验: 红方胜 {red_w}/{n - D} 决胜局 "
          f"({red_w / max(n - D, 1) * 100:.0f}%)  ← 配对已抵消")
    print(f"\n{'对':>3s} {'seed':>8s} {'新执红':>8s} {'新执黑':>8s} {'净':>5s}")
    for p in pairs:
        print(f"{p['pair']:3d} {p['seed']:8d} {p['new_red']['outcome']:>9s} "
              f"{p['new_black']['outcome']:>9s} {p['net']:+5.1f}")
    tn = sum(g["new_time"] for g in games); mn = sum(g["new_moves"] for g in games)
    to = sum(g["old_time"] for g in games); mo = sum(g["old_moves"] for g in games)
    dn = [g["new_depth"] for g in games if g["new_depth"]]
    do = [g["old_depth"] for g in games if g["old_depth"]]
    print(f"\n算力对称性: 用时/着 新 {tn / mn:.3f}s vs 基线 {to / mo:.3f}s "
          f"(偏差 {abs(tn / mn - to / mo) / (to / mo) * 100:.1f}%)")
    print(f"            平均深度 新 {sum(dn) / len(dn):.2f} vs 基线 "
          f"{sum(do) / len(do):.2f} (差 {sum(dn) / len(dn) - sum(do) / len(do):+.2f})")
    print(f"平均步数 {sum(g['plies'] for g in games) / n:.0f}")
    print(f"\n总墙钟 {(time.perf_counter() - t0) / 60:.1f} 分钟")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(a),
                   "summary": {"games": n, "win": W, "loss": L, "draw": D,
                               "pair_win": nw, "pair_loss": nl, "pair_draw": nd,
                               "new_avg_s": round(tn / mn, 3),
                               "old_avg_s": round(to / mo, 3),
                               "new_depth": round(sum(dn) / len(dn), 2),
                               "old_depth": round(sum(do) / len(do), 2)},
                   "pairs": pairs}, f, ensure_ascii=False, indent=1)
    print(f"[*] 已保存 {out_path}")


if __name__ == "__main__":
    main()
