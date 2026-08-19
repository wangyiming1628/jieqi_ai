"""汇总 tools/games/ 下的对局 JSON 记录: 胜负 + 每方用时/深度统计
用法: python tools/summarize_match.py game_xxx.json [game_yyy.json ...]
不带参数时默认汇总最新 3 局"""
import sys, os, json, glob

GAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games")


def load(paths):
    if not paths:
        paths = sorted(glob.glob(os.path.join(GAMES_DIR, "game_*.json")))[-3:]
    return [json.load(open(p, encoding="utf-8")) for p in paths]


def reason_en(reason):
    if "吃掉" in reason:
        return f"king captured ({reason})"
    if "无吃子" in reason:
        return "draw: 120 half-moves without capture"
    if "上限" in reason:
        return "draw: max ply reached"
    if "无着法" in reason or "判负" in reason:
        return f"loss by rule ({reason})"
    return reason


def side_stats(rec, side):
    ms = [m for m in rec["moves"] if m["side"] == side]
    ts = [m["time_s"] for m in ms]
    ds = [m["depth"] for m in ms if m["depth"] > 0]   # depth=0 是开局库着法, 不计
    return {
        "moves": len(ms),
        "total": sum(ts), "avg": sum(ts) / len(ts), "max": max(ts),
        "avg_depth": sum(ds) / len(ds) if ds else 0, "max_depth": max(ds) if ds else 0,
    }


def fmt(s):
    return (f"{s['moves']:3d} moves | total {s['total']:7.1f}s | avg {s['avg']:5.2f}s "
            f"| max {s['max']:5.2f}s | avg depth {s['avg_depth']:5.2f} | max depth {s['max_depth']:2d}")


def main(paths):
    tally = {}
    for rec in load(paths):
        cfg = rec["config"]
        res = rec["result"]
        rt = cfg.get("red_think", cfg.get("think_time"))
        bt = cfg.get("black_think", cfg.get("think_time"))
        think_note = f"think r={rt}s b={bt}s" if rt != bt else f"think={rt}s"
        print(f"\n=== seed={cfg.get('seed')} red={cfg['red']} black={cfg['black']} {think_note} ===")
        winner = res.get("winner")
        label = {"r": f"RED ({cfg['red']})", "b": f"BLACK ({cfg['black']})"}.get(winner, "DRAW")
        print(f"  result: {label}  |  {reason_en(res['reason'])}")
        print(f"  plies: {len(rec['moves'])}")
        for side in ("r", "b"):
            engine = cfg["red" if side == "r" else "black"]
            st = side_stats(rec, side)
            print(f"  {side.upper():5s} {engine:6s}: {fmt(st)}")
            t = tally.setdefault(engine, {"wins": 0, "draws": 0, "losses": 0,
                                          "moves": 0, "time": 0.0, "depths": [], "games": 0})
            t["games"] += 1
            t["moves"] += st["moves"]
            t["time"] += st["total"]
            t["depths"].append(st["avg_depth"])
            if winner == side:
                t["wins"] += 1
            elif winner is None:
                t["draws"] += 1
            else:
                t["losses"] += 1
    print("\n" + "=" * 70)
    print("MATCH TOTALS")
    print("=" * 70)
    for engine, t in tally.items():
        avg_t = t["time"] / t["moves"] if t["moves"] else 0
        print(f"  {engine:6s}: {t['wins']}W-{t['losses']}L-{t['draws']}D | "
              f"{t['moves']} moves | total {t['time']:.1f}s | avg/move {avg_t:.2f}s | "
              f"avg-of-avg depth {sum(t['depths'])/len(t['depths']):.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
