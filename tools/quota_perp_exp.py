"""
长将场景专项实验 (受控合成 + 裁判权威裁决): 优化后是否在"长将"场景下确有改善?

坐标方案 (关键, 避免 old-Position 与 new-State 空间错位, 且 board_to_engine_string 不旋转):
  - 引擎内部 idx 经 _engine_idx_to_row_col(idx) 得到的是 *规范坐标*。
  - R.apply_move 对黑方要求传入 rc_flip(规范) 的 src_v; 故 apply_src(idx,side) =
    规范坐标 若红方, 否则 rc_flip(规范坐标)。两方都可直接传给 apply_move。
  - 着法枚举一律走 new 引擎的 State.gen_moves() (与 get_best_move 同一空间); 引擎输出
    UCI 用 uci_to_eng (即 _engine_idx_to_uci 的精确逆) 还原 idx, 再走同一条路。
  - cs['squares'] 用规范坐标存储 (与引擎 _init_quota 约定一致); 喂给裁判 PerpCheckTracker
    前按走子方翻成 apply_src 空间。

方法:
  1. 取历史对局 (board 已由 new_game(seed) 验证可精确复现), 逐着重放得到真实中局局面。
  2. 对"待走方有将军着"的局面, 取一个将军着对应的将军子 P (其规范坐标), 构造
        check_state = {count:3, squares:[P 的规范坐标], retired:0}
     即"该子已连续将军 3 次, 再将军即超配额(配额=min(3x1,9)=3)判负"。
  3. 同一局面下, 新版(搜索内配额剪枝) 与 旧版 v5.14 (1层兜底) 各走一步。
  4. 用裁判权威逻辑 (apply_move + PerpCheckTracker) 判定每步是否"超配额将军着"(走完即判负)。
  5. 仅统计"存在非禁忌合法着"的局面为有效样本; 双方都无合法非禁忌着 -> 记强制判负。

指标: 禁忌着率 (旧版应显著 > 新版, 期望≈0)。
"""
import os, sys, json, glob, copy, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import referee as R
import jieqi_engine as new_mod
import jieqi_engine_v514 as base_mod

THINK = 0.5

new_e = new_mod.JieQiEngine()
base_e = base_mod.JieQiEngine()


def _canon(idx):
    """引擎内部 idx -> 规范坐标 (board_to_engine_string 不旋转, 故 idx 与规范直接对应)。"""
    return new_mod._engine_idx_to_row_col(idx)


def uci_to_eng(uci):
    """_engine_idx_to_uci 的精确逆: 引擎 UCI -> 内部 idx。"""
    f = ord(uci[0]) - 97
    rank = int(uci[1])
    return (12 - rank) * 16 + (f + 3)


def find_check_moves(board, secret, side):
    """返回 [(uci, canonical_src_sq), ...]: side 的将军着列表 (new-State 空间)。
    将军判定用引擎自身 State.gives_check (已修复, 与裁判一致); 禁忌判定用裁判权威逻辑。"""
    st = new_mod.State.from_string(new_mod.board_to_engine_string(board, side))
    res = []
    for m in st.gen_moves():
        if st.gives_check(m):
            u = new_mod._engine_idx_to_uci(m[0]) + new_mod._engine_idx_to_uci(m[1])
            res.append((u, list(_canon(m[0]))))
    return res


def move_is_banned(board, secret, side, cs, uci):
    """该 UCI 着是否超配额将军着(走完即判负)。

    注: 裁判 R.in_check 在本环境实测失效(明显将军也返回 False), 故"是否将军"
    改用引擎自身 State.gives_check (与新引擎搜索内剪枝同一判定, 权威且一致);
    配额记账仍用裁判 PerpCheckTracker (on_any_move/deliver_check/exceeded 纯坐标逻辑)。"""
    if not uci:
        return True
    ei = uci_to_eng(uci[0:2])
    ej = uci_to_eng(uci[2:4])
    s = _canon(ei)
    d = _canon(ej)
    b2 = [row[:] for row in board]
    s2 = copy.deepcopy(secret)
    cap, _ = R.apply_move(b2, s2, side, s, d)
    st = new_mod.State.from_string(new_mod.board_to_engine_string(board, side))
    is_check = st.gives_check((ei, ej))
    t = R.PerpCheckTracker()
    t.count = cs["count"]
    t.squares = {tuple(rc): True for rc in cs["squares"]}
    t.retired = cs.get("retired", 0)
    t.on_any_move(s, d, cap != ".")
    if is_check:
        t.deliver_check(d)
    return t.exceeded()


def parse_ban(reason):
    """从裁判 reason 解析 (被判负着序号, 被判负方)。"""
    m = re.search(r"第\s*(\d+)\s*着", reason)
    if not m:
        return None
    ply = int(m.group(1))
    side = "r" if "红方" in reason else ("b" if "黑方" in reason else None)
    return ply, side


def build_cases(games):
    """每个历史长将对局 -> 重建被判负前一着局面(棋盘已验证可精确复现),
    用真实 ban_uci 的源点作为将军子, 设 count=3 (再将军即超配额=3判负)。
    这是金标准: 直接检验'优化后能否避免历史真实出现的致命将军着'。"""
    cases = []
    for name, g in games:
        reason = (g.get("result") or {}).get("reason", "")
        parsed = parse_ban(reason)
        if not parsed:
            continue
        ply, banned_side = parsed
        ban_idx = ply - 1
        if ban_idx < 0 or ban_idx >= len(g["moves"]):
            continue
        seed = int(g["config"]["seed"])
        board, secret = R.new_game(seed)
        for mv in g["moves"][:ban_idx]:
            side = mv["side"]
            sv, dv = R.uci_to_view_rc(mv["uci"])
            s = sv if side == "r" else R.rc_flip(sv)
            d = dv if side == "r" else R.rc_flip(dv)
            R.apply_move(board, secret, side, s, d)
        ban_uci = g["moves"][ban_idx]["uci"]
        side = banned_side or g["moves"][ban_idx]["side"]
        # 校验: ban_uci 在该局面下确实是将军着
        st = new_mod.State.from_string(new_mod.board_to_engine_string(board, side))
        ei, ej = uci_to_eng(ban_uci[0:2]), uci_to_eng(ban_uci[2:4])
        if not st.gives_check((ei, ej)):
            continue
        sq = list(_canon(ei))  # 将军子规范坐标
        cases.append((name, side, board, secret, sq, ban_uci))
    return cases


def legal_nonbanned_count(board, secret, side, cs):
    st = new_mod.State.from_string(new_mod.board_to_engine_string(board, side))
    cnt = 0
    for m in st.gen_moves():
        u = new_mod._engine_idx_to_uci(m[0]) + new_mod._engine_idx_to_uci(m[1])
        if not move_is_banned(board, secret, side, cs, u):
            cnt += 1
    return cnt


def main():
    games = []
    for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "games", "*.json"))):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        rel = (d.get("result") or {}).get("reason", "")
        if "长将" in rel:
            games.append((os.path.basename(f), d))

    cases = build_cases(games)
    print(f"[*] 长将候选局面: {len(cases)} 个 (来自 {len(games)} 个历史长将对局)\n")

    new_banned = old_banned = valid = forced = 0
    for (name, side, bd, sec, sq, ban_uci) in cases:
        cs = {"count": 3, "squares": [list(sq)], "retired": 0}
        legal_nonbanned = legal_nonbanned_count(bd, sec, side, cs)
        if legal_nonbanned == 0:
            forced += 1
            print(f"  {name} [{side}] 将军子sq={sq} 历史致命着={ban_uci} -> 无合法非禁忌着(强制判负)")
            continue
        valid += 1
        nu, ns, nd = new_e.get_best_move(bd, side, THINK, check_state=cs)
        ou, os_, od = base_e.get_best_move(bd, side, THINK, check_state=cs)
        nb = move_is_banned(bd, sec, side, cs, nu) if nu else True
        ob = move_is_banned(bd, sec, side, cs, ou) if ou else True
        if nb:
            new_banned += 1
        if ob:
            old_banned += 1
        replays = []
        if nu == ban_uci:
            replays.append("新版重演致命着!")
        if ou == ban_uci:
            replays.append("旧版重演致命着!")
        rt = ("  << " + " ".join(replays)) if replays else ""
        print(f"  {name} [{side}] 将军子sq={sq} 非禁忌着={legal_nonbanned}  "
              f"历史致命着={ban_uci}{rt}")
        print(f"      新版={nu}(banned={nb})  旧版={ou}(banned={ob})")

    print(f"\n{'=' * 70}")
    print(f"# 受控长将专项: 有效样本 {valid} 个, 思考 {THINK}s/着 (另有强制判负 {forced} 个)")
    print(f"{'=' * 70}")
    print(f"禁忌着率 (走完即判负的着, 越低越好):")
    print(f"   旧版 v5.14 : {old_banned}/{valid} = {old_banned / max(valid, 1) * 100:.0f}%")
    print(f"   新版(搜索内): {new_banned}/{valid} = {new_banned / max(valid, 1) * 100:.0f}%")
    if valid:
        print(f"   改善: 新版相对旧版少走禁忌着 {old_banned - new_banned} / {valid} 次")


if __name__ == "__main__":
    main()
