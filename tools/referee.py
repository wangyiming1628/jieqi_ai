"""
揭棋引擎裁判程序 - 让两个引擎完整对局一局, 汇报双方用时与对局胜负

用法:
  python tools/referee.py                          # java(红) vs pypy(黑), 每着 1.0s
  python tools/referee.py --red pypy --black java  # 交换先后手
  python tools/referee.py --red pypy2 --black pypy # 优化版(TT保留+双时限) vs 原版
  python tools/referee.py --think-time 0.5 --seed 42
  # 等墙钟时间对局: 双方单独指定思考预算 (原版软超时会超标, 优化版硬上限不会)
  python tools/referee.py --red pypy2 --red-think 2.5 --black pypy --black-think 1.0

设计:
  - 裁判维护真实棋盘: 暗子真身随机洗牌后只有裁判知道, 引擎只收到公共视野(暗子显示为 ?),
    双方信息对称, 不存在偷看
  - 棋盘约定与 main.py 一致: 发给引擎的棋盘永远是"引擎自己总在 rows 5-9(下半场)";
    黑方走子前裁判把规范棋盘(红在下)180° 翻转, 收到着法后再翻回规范坐标
  - 着法校验: 用 jieqi_engine.gen_moves 在公共视野上校验(暗子走法只与所在格有关, 与真身无关)
  - 胜负判定: 吃掉对方帥/將获胜; 轮到走棋却无合法着法判负; 引擎连续两次无响应判负;
          连续 --no-cap-draw 个半着无吃子、或总步数达 --max-ply 判和
  - 汇报: 每着用时、双方总/平均/最长用时、胜负原因; 棋谱(JSON+文本)存 tools/games/
"""
import sys, os, time, json, random, argparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from engine_client import JavaEngineClient, PypyEngineClient  # noqa: E402
from jieqi_engine import (  # noqa: E402
    Position, board_to_engine_string, _row_col_to_engine_idx,
)

try:
    # 保持控制台本地编码(如 GBK), 只把无法编码的字符替换掉; 棋谱文件单独用 UTF-8 写
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

# ---------------- 揭棋开局与规则常量 ----------------
RED_POOL = ["車", "車", "馬", "馬", "相", "相", "仕", "仕", "炮", "炮", "兵", "兵", "兵", "兵", "兵"]
BLACK_POOL = ["車", "車", "馬", "馬", "象", "象", "士", "士", "炮", "炮", "卒", "卒", "卒", "卒", "卒"]
# 每方除帥/將外的 15 个暗子初始格 (帥/將明置在 (9,4)/(0,4))
INIT_SQUARES = {
    "r": [(9, c) for c in range(9) if c != 4] + [(7, 1), (7, 7)] + [(6, c) for c in (0, 2, 4, 6, 8)],
    "b": [(0, c) for c in range(9) if c != 4] + [(2, 1), (2, 7)] + [(3, c) for c in (0, 2, 4, 6, 8)],
}
KING = {"r": "r帥", "b": "b將"}
SIDE_NAME = {"r": "红", "b": "黑"}


def new_game(seed):
    """初始局面: 帥/將明置, 其余 15 子/方随机洗入各自初始格; 真身只存在裁判的 secret."""
    rng = random.Random(seed)
    board = [["."] * 9 for _ in range(10)]
    secret = {}
    board[9][4], board[0][4] = "r帥", "b將"
    for side, pool in (("r", RED_POOL), ("b", BLACK_POOL)):
        pool = list(pool)
        rng.shuffle(pool)
        for (r, c), t in zip(INIT_SQUARES[side], pool):
            board[r][c] = side + "?"
            secret[(r, c)] = t
    return board, secret


def flip_board(board):
    """180° 翻转棋盘, 让走子方总在 rows 5-9."""
    return [[board[9 - r][8 - c] for c in range(9)] for r in range(10)]


def rc_flip(rc):
    r, c = rc
    return (9 - r, 8 - c)


def uci_to_view_rc(uci):
    """引擎 UCI (走子方视角, board_row = 9 - uci_rank) → 视图坐标 (row, col)."""
    u = uci.lstrip("+")
    src = (9 - int(u[1]), ord(u[0]) - 97)
    dst = (9 - int(u[3]), ord(u[2]) - 97)
    return src, dst


def legal_engine_moves(view, side):
    """公共视野上的合法着法集合(引擎坐标). 暗子按所在格走法生成, 与真身无关."""
    estr = board_to_engine_string(view, side)
    return set(Position(estr, 0, True, 0).gen_moves())


def is_legal(view, side, src, dst):
    m = (_row_col_to_engine_idx(*src), _row_col_to_engine_idx(*dst))
    return m in legal_engine_moves(view, side)


def apply_move(board, secret, side, src, dst):
    """在真实棋盘上执行着法. 返回 (被吃子真身或 None, 走子方翻开真身或 None)."""
    piece = board[src[0]][src[1]]
    reveal = None
    if piece[1] == "?":                       # 暗子首次移动 → 翻开
        reveal = secret.pop(src)
        piece = side + reveal
    captured = board[dst[0]][dst[1]]
    cap_true = None
    if captured != ".":
        cap_true = secret.pop(dst, None)      # 吃暗子: 真身只对裁判揭晓
        if cap_true:
            captured = captured[0] + cap_true
    board[dst[0]][dst[1]] = piece
    board[src[0]][src[1]] = "."
    return captured, reveal


def board_str(board):
    lines = ["    " + " ".join(str(c) for c in range(9))]
    for r in range(10):
        lines.append(f"{r:2d} | " + " ".join(p if p != "." else "·" for p in board[r]))
    return "\n".join(lines)


def in_check(board, side):
    """side 的王是否可被对方一步吃掉 (被将军)。board 为规范棋盘(红在下)。"""
    view = board if side == "r" else flip_board(board)
    estr = board_to_engine_string(view, side)
    oppo = Position(estr, 0, True, 0).set().rotate()
    return any(oppo.board[m[1]] == "k" for m in oppo.gen_moves())


class PerpCheckTracker:
    """单方连续将军计数器 (亚洲规则配额制: 容忍 6×将军子数 次连续将军)。
    count: 不间断将军累计着数; squares: 参与将军的棋子当前格集合
    (按子的轨迹认子, 不按格); retired: 已参与后被吃掉的子数 (配额只增不减)。
    只有该方走出非将军着法才整体重置; 吃子不重置。"""

    def __init__(self):
        self.count = 0
        self.squares = {}
        self.retired = 0

    @property
    def quota(self):
        return 6 * (len(self.squares) + self.retired)

    def on_any_move(self, src, dst, has_capture):
        """任一方走子后更新 tracked 子的轨迹 (被吃/移动)。必须先查吃子再挪子。"""
        if has_capture and dst in self.squares:
            del self.squares[dst]
            self.retired += 1
        if src in self.squares:
            del self.squares[src]
            self.squares[dst] = True

    def deliver_check(self, dst):
        """本方走出一着将军: 计数并纳入将军子。"""
        self.count += 1
        if dst not in self.squares:
            self.squares[dst] = True

    def exceeded(self):
        return self.count > self.quota

    def reset(self):
        self.count = 0
        self.squares = {}
        self.retired = 0

    def state_for(self):
        """下发给引擎的已方状态 (规范坐标)。"""
        return {"count": self.count,
                "squares": [list(rc) for rc in self.squares],
                "retired": self.retired}


def adjudicate_long_check(loop, side):
    """长将定罪检查。loop: [(board, side_to_move), ...] 从键首次记录时刻到当前候选着
    的闭环 (含当前着)。side 为触发方: 其环内着法全部是将、且对方不是全部是将
    (互将豁免) → 判 side 负; 否则返回 None。"""
    oppo = "b" if side == "r" else "r"
    my_chks, oppo_chks = [], []
    for board, s in loop:
        mover = "b" if s == "r" else "r"      # 形成该局面的走子方
        (my_chks if mover == side else oppo_chks).append(in_check(board, s))
    if my_chks and all(my_chks) and not (oppo_chks and all(oppo_chks)):
        return {"winner": oppo,
                "reason": f"{SIDE_NAME[side]}方长将 (键重复且环内{SIDE_NAME[side]}方每着均将军), 判负"}
    return None


def make_engine(kind):
    if kind == "java":
        return JavaEngineClient(), "java (Makinuohara, expectiminimax)"
    if kind == "pypy2":
        server = os.path.join(REPO, "engine_server_v2.py")
        return PypyEngineClient(prefer_pypy=True, server_path=server), \
            "pypy2 (miaosiSari 优化版: TT保留+双时限)"
    if kind == "pypy3":
        server = os.path.join(REPO, "engine_server_v3.py")
        return PypyEngineClient(prefer_pypy=True, server_path=server), \
            "pypy3 (v5.6 基线+真静态搜索)"
    if kind == "pypy57":
        server = os.path.join(REPO, "engine_server_v57.py")
        return PypyEngineClient(prefer_pypy=True, server_path=server), \
            "pypy57 (v5.7 基线, 无重复感知)"
    return PypyEngineClient(prefer_pypy=True), "pypy (miaosiSari 原版, alpha-beta)"


def ask_engine(engine, view, side, think_time, pos_history=None, check_state=None):
    """向引擎要一步棋, 返回 (uci, 用时秒). 失败重试一次(客户端会自动重启子进程).
    pos_history: 开局以来的完整局面历史 [(规范棋盘, 轮到方), ...] (引擎需要跨吃子的
    全量历史才能正确做长将归属; 裁判自己的重复判和仍用吃子清零的短历史);
    check_state: 已方连续将军计数状态, 供引擎配额规避。"""
    last_err = None
    for attempt in (1, 2):
        t0 = time.perf_counter()
        try:
            uci, score, depth = engine.get_best_move(
                view, side, think_time=think_time,
                pos_history=pos_history, check_state=check_state)
        except Exception as e:
            uci, score, depth, last_err = None, 0, 0, repr(e)
        dt = time.perf_counter() - t0
        if uci:
            return uci, score, depth, dt, attempt
        print(f"    [!] 引擎第 {attempt} 次无响应 (耗时 {dt:.1f}s"
              + (f", 异常: {last_err}" if last_err else "") + ")", flush=True)
    return None, 0, 0, 0.0, 2


def play(args):
    print(f"[*] 裁判启动: {args.red} 执红先行(每着 {args.red_think}s) "
          f"vs {args.black} 执黑(每着 {args.black_think}s), 随机种子 {args.seed}", flush=True)
    engines, labels = {}, {}
    for side in "rb":
        kind = args.red if side == "r" else args.black
        engines[side], labels[side] = make_engine(kind)
    print(f"[*] 红方引擎: {labels['r']}")
    print(f"[*] 黑方引擎: {labels['b']}", flush=True)

    board, secret = new_game(args.seed)
    stats = {s: {"total": 0.0, "n": 0, "max": 0.0, "times": []} for s in "rb"}
    records = []
    no_cap = 0            # 连续无吃子半着数
    # [重复裁决] 短历史: 吃子清零, 用于三次重复判和 (循环不可能跨越吃子点)
    pos_history = [([row[:] for row in board], "r")]
    # [长将键表] 每方的 (将军前局面签名, src, dst) → 首次记录时的历史锚点索引;
    # 键第二次出现即触发环检测 (锚点保证环从首次记录算起, 一将一闲里的闲着不会丢)
    full_history = [([row[:] for row in board], "r")]
    check_keys = {"r": {}, "b": {}}
    # [配额裁决] 双方连续将军计数器 (吃子不重置, 非将军着才重置)
    trackers = {s: PerpCheckTracker() for s in "rb"}
    result = None         # {"winner": "r"/"b"/None, "reason": str}
    t_start = time.perf_counter()

    for ply in range(args.max_ply):
        side = "r" if ply % 2 == 0 else "b"
        view = board if side == "r" else flip_board(board)

        # 先检查走子方是否还有合法着法
        if not legal_engine_moves(view, side):
            result = {"winner": "b" if side == "r" else "r", "reason": f"{SIDE_NAME[side]}方无合法着法, 判负"}
            break

        uci, score, depth, dt, tries = ask_engine(
            engines[side], view, side,
            args.red_think if side == "r" else args.black_think,
            pos_history=full_history, check_state=trackers[side].state_for())
        stats[side]["total"] += dt
        stats[side]["n"] += 1
        stats[side]["max"] = max(stats[side]["max"], dt)
        stats[side]["times"].append(round(dt, 3))

        if uci is None:
            result = {"winner": "b" if side == "r" else "r",
                      "reason": f"{SIDE_NAME[side]}方引擎连续 {tries} 次无着法响应, 判负"}
            break

        src_v, dst_v = uci_to_view_rc(uci)
        if not is_legal(view, side, src_v, dst_v):
            result = {"winner": "b" if side == "r" else "r",
                      "reason": f"{SIDE_NAME[side]}方走出非法着法 {uci}, 判负"}
            records.append({"ply": ply + 1, "side": side, "uci": uci, "illegal": True})
            break

        src = src_v if side == "r" else rc_flip(src_v)     # 规范坐标(红在下)
        dst = dst_v if side == "r" else rc_flip(dst_v)
        captured, reveal = apply_move(board, secret, side, src, dst)

        tags = []
        if reveal:
            tags.append(f"翻开={reveal}")
        if captured != ".":
            tags.append(f"吃={captured}")
            no_cap = 0
        else:
            no_cap += 1
        info = " ".join(tags)
        print(f"[{ply + 1:3d}] {SIDE_NAME[side]} {uci} {src}->{dst} {info} "
              f"用时 {dt:.2f}s (score={score} depth={depth})", flush=True)
        records.append({
            "ply": ply + 1, "side": side, "uci": uci,
            "src": list(src), "dst": list(dst),
            "reveal": reveal, "captured": None if captured == "." else captured,
            "score": score, "depth": depth, "time_s": round(dt, 3),
        })

        if captured in KING.values():                       # 吃掉帥/將 → 胜
            result = {"winner": side, "reason": f"第 {ply + 1} 着吃掉 {'將' if side == 'r' else '帥'}, 获胜"}
            break

        # [配额+键表] 更新将军子轨迹 → 判定本着是否将军 → 超额/键重复裁决
        next_side = "b" if side == "r" else "r"
        for s in "rb":
            trackers[s].on_any_move(src, dst, captured != ".")
        if in_check(board, next_side):
            trackers[side].deliver_check(dst)
            if trackers[side].exceeded():
                result = {"winner": next_side,
                          "reason": f"第 {ply + 1} 着后 {SIDE_NAME[side]}方连续将军 "
                                    f"{trackers[side].count} 次, 超过配额 "
                                    f"(6×{len(trackers[side].squares) + trackers[side].retired} 将军子), 判负"}
                break
            # 键表: 键 = (将军前局面, 着法); 第二次出现 → 从锚点起环检测
            pre_sig = tuple(tuple(r) for r in full_history[-1][0])
            key = (pre_sig, src, dst)
            if key in check_keys[side]:
                anchor = check_keys[side][key]
                loop = full_history[anchor + 1:] + [([row[:] for row in board], next_side)]
                adj = adjudicate_long_check(loop, side)
                if adj is not None:
                    adj["reason"] = f"第 {ply + 1} 着后: {adj['reason']}"
                    result = adj
                    break
            else:
                check_keys[side][key] = len(full_history) - 1   # 锚点 = 将军前局面的索引
        else:
            trackers[side].reset()

        # [重复裁决] 记录新局面; 短历史吃子清零重记, 全量历史只增不减
        snap = [row[:] for row in board]
        if captured != ".":
            pos_history = [(snap, next_side)]
        else:
            pos_history.append((snap, next_side))
        full_history.append((snap, next_side))
        cur = pos_history[-1]
        cnt = sum(1 for e in pos_history if e == cur)
        if cnt >= 3:
            result = {"winner": None, "reason": f"第 {ply + 1} 着后局面第 {cnt} 次出现, 判和"}
            break

        if no_cap >= args.no_cap_draw:
            result = {"winner": None, "reason": f"连续 {no_cap} 个半着无吃子, 判和"}
            break
    else:
        result = {"winner": None, "reason": f"达到最大步数 {args.max_ply}, 判和"}

    wall = time.perf_counter() - t_start
    for s in "rb":
        engines[s].close()
    return board, records, stats, result, wall


def report(args, board, records, stats, result, wall):
    n = len(records)
    print("\n" + "=" * 56)
    print("对局结果")
    print("=" * 56)
    if result["winner"]:
        w = result["winner"]
        print(f"胜者: {args.red if w == 'r' else args.black} ({SIDE_NAME[w]}方)")
    else:
        print("和棋")
    print(f"原因: {result['reason']}")
    print(f"总步数: {n} 着, 全局墙钟 {wall:.1f}s")
    print("\n双方用时:")
    for s in "rb":
        st = stats[s]
        name = args.red if s == "r" else args.black
        avg = st["total"] / st["n"] if st["n"] else 0.0
        print(f"  {name:5s} ({SIDE_NAME[s]}方): {st['n']:3d} 着  总用时 {st['total']:7.1f}s  "
              f"平均 {avg:5.2f}s  最长 {st['max']:6.2f}s")

    # 棋谱落盘
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games")
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, time.strftime("game_%Y%m%d_%H%M%S"))
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({
            "config": {"red": args.red, "black": args.black,
                       "think_time": args.think_time,
                       "red_think": args.red_think, "black_think": args.black_think,
                       "seed": args.seed,
                       "max_ply": args.max_ply, "no_cap_draw": args.no_cap_draw},
            "result": result, "stats": stats, "moves": records,
            "final_board": board,
        }, f, ensure_ascii=False, indent=1)
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write(f"{args.red}(红, think={args.red_think}s) vs "
                f"{args.black}(黑, think={args.black_think}s)  seed={args.seed}\n")
        f.write(f"结果: {result}\n\n终局棋盘 (红方在下):\n{board_str(board)}\n\n着法记录:\n")
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[*] 棋谱已保存: {base}.json / .txt")
    print("\n终局棋盘 (红方在下):")
    print(board_str(board))
    return base


def main():
    ap = argparse.ArgumentParser(description="揭棋引擎裁判: java vs pypy 完整对局")
    ap.add_argument("--red", choices=["java", "pypy", "pypy2", "pypy3", "pypy57"], default="java", help="红方引擎 (默认 java)")
    ap.add_argument("--black", choices=["java", "pypy", "pypy2", "pypy3", "pypy57"], default="pypy", help="黑方引擎 (默认 pypy)")
    ap.add_argument("--think-time", type=float, default=1.0, help="双方每着思考秒数 (默认 1.0)")
    ap.add_argument("--red-think", type=float, default=None, help="红方每着思考秒数 (缺省用 --think-time)")
    ap.add_argument("--black-think", type=float, default=None, help="黑方每着思考秒数 (缺省用 --think-time)")
    ap.add_argument("--max-ply", type=int, default=400, help="最大半着数, 超出判和 (默认 400)")
    ap.add_argument("--no-cap-draw", type=int, default=120, help="连续无吃子和棋半着数 (默认 120)")
    ap.add_argument("--seed", type=int, default=20260818, help="暗子洗牌随机种子")
    args = ap.parse_args()
    # 未单独指定时, 双方使用共同的 --think-time
    args.red_think = args.red_think if args.red_think is not None else args.think_time
    args.black_think = args.black_think if args.black_think is not None else args.think_time
    board, records, stats, result, wall = play(args)
    report(args, board, records, stats, result, wall)


if __name__ == "__main__":
    main()
