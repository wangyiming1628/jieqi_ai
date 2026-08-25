"""
[v5.12 P0] 暗子池跨回合跟踪 - 单元测试

运行:
    python tools/test_dark_pool.py            # 全部
    python tools/test_dark_pool.py -v         # 详细
    python tools/test_dark_pool.py TestInvariant   # 单个 TestCase

测试分组:
  TestPoolInvariant      核心不变量 sum(池) == 盘面朝下子数 (构造用例 + 全棋谱回放)
  TestRevealEvents       翻开事件识别 (我方/对方/暗着被吃/断链兜底)
  TestProportionalShrink 暗着被吃的按比例缩放语义 (贝叶斯边缘)
  TestReplayAccuracy     全棋谱回放的池精度, 与 v5.11 及信息论上界对比
  TestBugRegression      针对 v5.11 具体 bug 的回归测试 (幽灵暗车)
  TestEngineIntegration  引擎级集成 (裁判/实战两模式、换边重置、开局库路径)
  TestEvalImpact         评估量的实际变化 (average 表 / possible_che / value())
"""
import sys, os, json, glob, unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import jieqi_engine as E                    # P0 修复版
import jieqi_engine_v511 as OLD             # 修复前基线

TL = E.TYPE_LETTER
INIT = dict(E.POOL_INIT)


def _is_game_log(path):
    """棋谱文件判定: tools/games/ 下还有对局套件产出的汇总文件 (p0_*.json),
    它们的 schema 不同 (无顶层 moves), 必须排除, 否则回放测试会 KeyError。"""
    if os.path.basename(path).startswith("p0_"):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return False
    return isinstance(d, dict) and isinstance(d.get("moves"), list)


GAMES = [p for p in sorted(glob.glob(os.path.join(REPO, "tools", "games", "*.json")))
         if _is_game_log(p)]

INIT_SQ = {
    "r": [(9, c) for c in range(9) if c != 4] + [(7, 1), (7, 7)]
         + [(6, c) for c in (0, 2, 4, 6, 8)],
    "b": [(0, c) for c in range(9) if c != 4] + [(2, 1), (2, 7)]
         + [(3, c) for c in (0, 2, 4, 6, 8)],
}


# ---------------------------------------------------------------- 工具
def fresh_board():
    """标准揭棋初始盘面 (规范坐标: 红在下), 帥/將明置, 其余 15 子/方朝下。"""
    bd = [["."] * 9 for _ in range(10)]
    bd[9][4], bd[0][4] = "r帥", "b將"
    for side in "rb":
        for r, c in INIT_SQ[side]:
            bd[r][c] = side + "?"
    return bd


def mkboard(spec):
    """spec: {(row, col): "r車" / "b?" / ...} → 规范棋盘。"""
    bd = [["."] * 9 for _ in range(10)]
    for rc, p in spec.items():
        bd[rc[0]][rc[1]] = p
    return bd


def dark_count(estr, mine):
    dark = E.DARK_UPPER if mine else E.DARK_LOWER
    return sum(1 for i in range(51, 204) if estr[i] in dark)


def replay(gpath, viewer="r"):
    """回放一局棋谱。yield 每个 viewer 回合的
    (ply, estr, 真实池(viewer), 真实池(对方), 真实rev(viewer), 我方着法idx)。
    棋谱里 reveal/captured 是裁判记录的真身, 作为 ground truth。"""
    g = json.load(open(gpath, encoding="utf-8"))
    board = fresh_board()
    tpool = {"r": dict(INIT), "b": dict(INIT)}
    trev = {"r": {}, "b": {}}
    opp_of = {"r": "b", "b": "r"}
    for rec in g["moves"]:
        if rec.get("illegal"):
            break
        side = rec["side"]
        opp = opp_of[side]
        s_, d_ = tuple(rec["src"]), tuple(rec["dst"])
        emit = None
        if side == viewer:
            estr = E.board_to_engine_string(board, viewer)
            if "K" not in estr or "k" not in estr:
                break
            emit = (rec["ply"], estr,
                    dict(tpool[viewer]), dict(tpool[opp_of[viewer]]),
                    dict(trev[viewer]),
                    (E._row_col_to_engine_idx(*s_), E._row_col_to_engine_idx(*d_)))
        piece = board[s_[0]][s_[1]]
        dark_victim = board[d_[0]][d_[1]].endswith("?")
        if rec.get("reveal"):
            piece = side + rec["reveal"]
            t = TL[rec["reveal"]]
            trev[side][t] = trev[side].get(t, 0) + 1
            tpool[side][t] -= 1
        cap = rec.get("captured")
        if cap and dark_victim:
            tpool[opp][TL[cap[1:]]] -= 1
        board[d_[0]][d_[1]] = piece
        board[s_[0]][s_[1]] = "."
        if emit:
            yield emit


def drive(gpath, viewer="r"):
    """用 DarkPoolTracker 完整跟踪一局, yield (ply, estr, tracker, 真实池, 真实rev)。"""
    trk = E.DarkPoolTracker()
    for ply, estr, tp_my, tp_op, trev, mymove in replay(gpath, viewer):
        trk.observe(estr)
        yield ply, estr, trk, tp_my, tp_op, trev
        trk.commit(mymove)


# ---------------------------------------------------------------- 不变量
class TestPoolInvariant(unittest.TestCase):
    """核心不变量: sum(池) 必须等于盘面上该方朝下子数。

    依据: 每方 15 子入池, 每子恰以两种方式离开池子 —— 被翻开(类型公开) 或
    仍朝下时被吃(类型不公开)。因此"池里还剩几个"永远等于"盘面上还有几个朝下的"。
    """

    def test_initial_position(self):
        estr = E.board_to_engine_string(fresh_board(), "r")
        trk = E.DarkPoolTracker()
        trk.observe(estr)
        for mine in (True, False):
            pool = trk.pool(estr, mine)
            self.assertAlmostEqual(sum(pool.values()), 15.0, places=9)
            self.assertEqual(dark_count(estr, mine), 15)
        # 初始时池 == 先验
        self.assertEqual({k: round(v) for k, v in trk.pool(estr, True).items()}, INIT)

    def test_empty_pool_all_revealed(self):
        """无朝下子 → 池必须全 0 (v5.11 在此返回非零幽灵子)。"""
        spec = {(9, 4): "r帥", (0, 4): "b將", (5, 4): "r車", (3, 4): "b車"}
        estr = E.board_to_engine_string(mkboard(spec), "r")
        trk = E.DarkPoolTracker()
        trk.observe(estr)
        for mine in (True, False):
            self.assertEqual(dark_count(estr, mine), 0)
            self.assertAlmostEqual(sum(trk.pool(estr, mine).values()), 0.0, places=9)

    def test_single_dark_left(self):
        """只剩 1 个朝下子 → 池和必须恰为 1。"""
        spec = {(9, 4): "r帥", (0, 4): "b將", (9, 0): "r?"}
        estr = E.board_to_engine_string(mkboard(spec), "r")
        trk = E.DarkPoolTracker()
        trk.observe(estr)
        self.assertAlmostEqual(sum(trk.pool(estr, True).values()), 1.0, places=9)
        self.assertAlmostEqual(sum(trk.pool(estr, False).values()), 0.0, places=9)

    def test_invariant_over_all_games(self):
        """全棋谱逐局面断言不变量 (这是 P0 的硬验收标准)。"""
        self.assertTrue(GAMES, "找不到棋谱, 无法回放验证")
        checked = 0
        for gpath in GAMES:
            for ply, estr, trk, tp_my, tp_op, trev in drive(gpath):
                for mine in (True, False):
                    pool = trk.pool(estr, mine)
                    self.assertAlmostEqual(
                        sum(pool.values()), dark_count(estr, mine), places=9,
                        msg=f"{os.path.basename(gpath)} ply{ply} mine={mine} 不变量被破坏")
                    for t, v in pool.items():
                        self.assertGreaterEqual(v, 0.0, "池计数不得为负")
                        self.assertLessEqual(v, INIT[t] + 1e-9,
                                             f"类型 {t} 超过初始个数 {INIT[t]}")
                checked += 1
        self.assertGreater(checked, 3000, "回放局面数异常偏少")
        print(f"\n    [不变量] {checked} 个局面全部满足 sum(池)==朝下子数")

    def test_v511_violates_the_same_invariant(self):
        """对照: v5.11 在同样的局面上大面积违反不变量 (证明测试有区分力)。"""
        bad = tot = 0
        for gpath in GAMES:
            for ply, estr, tp_my, tp_op, trev, mv in replay(gpath):
                OLD._update_distribution(estr)
                s_my = sum(OLD.di[0][True].values())
                s_op = sum(OLD.di[0][False].values())
                tot += 1
                if s_my != dark_count(estr, True) or s_op != dark_count(estr, False):
                    bad += 1
        self.assertGreater(bad / tot, 0.9, "v5.11 应大面积违反不变量")
        print(f"    [对照] v5.11 违反不变量 {bad}/{tot} ({bad / tot * 100:.0f}%)")


# ---------------------------------------------------------------- 翻开事件
class TestRevealEvents(unittest.TestCase):
    """翻开事件的识别: 我方 / 对方 / 暗着被吃 / 断链兜底。"""

    def test_own_reveal_detected_next_observation(self):
        """我方暗子走动 → 下次观测时读到真身并扣减对应类型。"""
        trk = E.DarkPoolTracker()
        b1 = fresh_board()
        e1 = E.board_to_engine_string(b1, "r")
        trk.observe(e1)
        self.assertEqual(trk.rev[True], {})            # 尚无翻开
        mv = (E._row_col_to_engine_idx(9, 0), E._row_col_to_engine_idx(8, 0))
        trk.commit(mv)
        # 真身是車; 对方也走一步 (暗馬 b0->b1 不翻开, 用明子避免干扰)
        b2 = fresh_board()
        b2[9][0] = "."
        b2[8][0] = "r車"                                # 我方翻出車
        b2[0][0] = "."
        b2[1][0] = "b車"                                # 对方翻出車
        e2 = E.board_to_engine_string(b2, "r")
        trk.observe(e2)
        self.assertEqual(trk.rev[True].get("R"), 1, "我方翻出的車未被登记")
        self.assertEqual(trk.stats["own_revealed"], 1)
        pool = trk.pool(e2, True)
        self.assertAlmostEqual(pool["R"], 1.0 * 14 / 14, places=6)
        self.assertAlmostEqual(sum(pool.values()), 14.0, places=9)

    def test_oppo_reveal_detected(self):
        """对方暗子走动 → 从 diff 定位其 src/dst 并读到真身。"""
        trk = E.DarkPoolTracker()
        b1 = fresh_board()
        trk.observe(E.board_to_engine_string(b1, "r"))
        mv = (E._row_col_to_engine_idx(6, 0), E._row_col_to_engine_idx(5, 0))
        trk.commit(mv)
        b2 = fresh_board()
        b2[6][0] = "."
        b2[5][0] = "r兵"
        b2[0][8] = "."
        b2[1][8] = "b車"                                # 对方翻出車
        trk.observe(E.board_to_engine_string(b2, "r"))
        self.assertEqual(trk.rev[False].get("R"), 1, "对方翻出的車未被登记")
        self.assertEqual(trk.stats["oppo_revealed"], 1)

    def test_own_reveal_lost_when_captured_immediately(self):
        """我方暗子走动后立刻被吃 → 真身永不可知, 记入 own_lost,
        但池大小仍须正确 (由重标定吸收)。这是信息论下界, 不是 bug。"""
        trk = E.DarkPoolTracker()
        b1 = fresh_board()
        trk.observe(E.board_to_engine_string(b1, "r"))
        src = E._row_col_to_engine_idx(6, 0)
        dst = E._row_col_to_engine_idx(5, 0)
        trk.commit((src, dst))
        b2 = fresh_board()
        b2[6][0] = "."
        b2[3][0] = "."
        b2[5][0] = "b卒"          # 对方暗卒吃掉我方刚翻开的子并占位
        e2 = E.board_to_engine_string(b2, "r")
        trk.observe(e2)
        self.assertEqual(trk.stats["own_lost"], 1, "应记录一次不可知的自方翻开")
        self.assertEqual(trk.rev[True], {}, "不可知类型不应臆测")
        self.assertAlmostEqual(sum(trk.pool(e2, True).values()),
                               dark_count(e2, True), places=9,
                               msg="类型不可知也必须保持池大小正确")

    def test_floor_recovers_from_broken_chain(self):
        """断链兜底: 完全不喂着法, 单调下界仍能从可见明子恢复翻开数。"""
        trk = E.DarkPoolTracker()
        spec = {(9, 4): "r帥", (0, 4): "b將",
                (5, 0): "r車", (5, 1): "r車", (5, 2): "r炮",
                (9, 0): "r?", (9, 1): "r?",
                (3, 0): "b車", (0, 0): "b?"}
        estr = E.board_to_engine_string(mkboard(spec), "r")
        trk.observe(estr)                     # 无 commit, 无历史
        self.assertEqual(trk.rev[True].get("R"), 2, "两个可见車应推出已翻开 2 个")
        self.assertEqual(trk.rev[True].get("C"), 1)
        self.assertEqual(trk.rev[False].get("R"), 1)
        pool = trk.pool(estr, True)
        self.assertAlmostEqual(pool["R"], 0.0, places=9, msg="車已全部翻开, 池中不应再有")
        self.assertAlmostEqual(sum(pool.values()), 2.0, places=9)

    def test_floor_is_monotone_never_decreases(self):
        """明子被吃后从盘面消失, 但"曾被翻开"永久成立 → rev 不得回落。
        这正是 v5.11 的错误所在。"""
        trk = E.DarkPoolTracker()
        spec = {(9, 4): "r帥", (0, 4): "b將", (5, 0): "r車", (9, 0): "r?", (0, 0): "b?"}
        e1 = E.board_to_engine_string(mkboard(spec), "r")
        trk.observe(e1)
        self.assertEqual(trk.rev[True].get("R"), 1)
        # 車被吃掉, 从盘面消失
        spec2 = {(9, 4): "r帥", (0, 4): "b將", (9, 0): "r?", (0, 0): "b?"}
        e2 = E.board_to_engine_string(mkboard(spec2), "r")
        trk.observe(e2)
        self.assertEqual(trk.rev[True].get("R"), 1, "车已被吃, 但翻开记录必须保留")
        self.assertAlmostEqual(trk.pool(e2, True)["R"], 1.0 / 14, places=6)

    def test_reveal_capped_at_initial_count(self):
        """识别错误导致同类明子超过初始个数时, 不得让池出现负数。"""
        trk = E.DarkPoolTracker()
        spec = {(9, 4): "r帥", (0, 4): "b將", (9, 0): "r?"}
        for c in range(5):
            spec[(5, c)] = "r車"              # 5 个車 (超过上限 2), 人为矛盾
        estr = E.board_to_engine_string(mkboard(spec), "r")
        trk.observe(estr)
        self.assertLessEqual(trk.rev[True]["R"], INIT["R"])
        pool = trk.pool(estr, True)
        for t, v in pool.items():
            self.assertGreaterEqual(v, 0.0)
        self.assertAlmostEqual(sum(pool.values()), 1.0, places=9)

    def test_all_types_exhausted_falls_back_to_prior(self):
        """极端矛盾: 所有类型都已翻开满, 但盘面仍有朝下子 → 退回先验比例,
        绝不返回全 0 或负数 (否则 sumall==0 会让 value() 分母为 0)。"""
        trk = E.DarkPoolTracker()
        trk.rev[True] = dict(INIT)             # 声称 15 子全翻开
        spec = {(9, 4): "r帥", (0, 4): "b將", (9, 0): "r?", (9, 1): "r?"}
        estr = E.board_to_engine_string(mkboard(spec), "r")
        pool = trk.pool(estr, True)
        self.assertAlmostEqual(sum(pool.values()), 2.0, places=9)
        for v in pool.values():
            self.assertGreaterEqual(v, 0.0)


# ---------------------------------------------------------------- 比例缩放
class TestProportionalShrink(unittest.TestCase):
    """暗着被吃 (类型不可观测) 的处理: 按当前比例缩放。

    这是无信息条件下的正确贝叶斯边缘: 一个未知类型的子离开池子时,
    每种类型按其占比承担损失, 类型比例保持不变, 池和精确减 1。
    """

    def test_shrink_preserves_ratio(self):
        trk = E.DarkPoolTracker()
        # 15 子全朝下时的比例
        spec = {(9, 4): "r帥", (0, 4): "b將"}
        for rc in INIT_SQ["r"]:
            spec[rc] = "r?"
        e15 = E.board_to_engine_string(mkboard(spec), "r")
        trk.observe(e15)
        p15 = trk.pool(e15, True)
        # 拿掉 5 个朝下子 (视作暗着被吃), 无任何翻开事件
        spec2 = dict(spec)
        for rc in INIT_SQ["r"][:5]:
            del spec2[rc]
        e10 = E.board_to_engine_string(mkboard(spec2), "r")
        trk.observe(e10)
        p10 = trk.pool(e10, True)
        self.assertAlmostEqual(sum(p10.values()), 10.0, places=9)
        # 比例不变
        for t in INIT:
            self.assertAlmostEqual(p10[t] / 10.0, p15[t] / 15.0, places=9,
                                   msg=f"类型 {t} 的比例在缩放后改变了")

    def test_reveal_then_shrink_composition(self):
        """先翻开一个車, 再暗着被吃 2 子: 車的剩余比例应体现翻开事实。"""
        trk = E.DarkPoolTracker()
        trk.rev[True] = {"R": 1}               # 已翻开 1 個車
        spec = {(9, 4): "r帥", (0, 4): "b將", (5, 0): "r車"}
        for rc in INIT_SQ["r"][:10]:
            spec[rc] = "r?"
        estr = E.board_to_engine_string(mkboard(spec), "r")
        pool = trk.pool(estr, True)
        self.assertAlmostEqual(sum(pool.values()), 10.0, places=9)
        # base: R=1, N=2, B=2, A=2, C=2, P=5 → 共 14, 缩放到 10
        self.assertAlmostEqual(pool["R"], 1 * 10 / 14, places=9)
        self.assertAlmostEqual(pool["P"], 5 * 10 / 14, places=9)


# ---------------------------------------------------------------- 精度
class TestReplayAccuracy(unittest.TestCase):
    """回放全部棋谱, 与 v5.11 及信息论上界 (ORACLE) 对比池精度。"""

    @classmethod
    def setUpClass(cls):
        cls.err_p0 = cls.err_old = cls.err_oracle = 0.0
        cls.worst_p0 = cls.worst_old = 0.0
        cls.n = 0
        for gpath in GAMES:
            trk = E.DarkPoolTracker()
            oracle_rev = {}
            for ply, estr, tp_my, tp_op, trev, mv in replay(gpath):
                trk.observe(estr)
                oracle_rev = dict(trev)        # 全知者: 真实翻开记录
                D = dark_count(estr, True)
                p0 = trk.pool(estr, True)
                OLD._update_distribution(estr)
                old = {t: float(OLD.di[0][True].get(t, 0)) for t in INIT}
                base = {t: max(0, INIT[t] - oracle_rev.get(t, 0)) for t in INIT}
                S = sum(base.values())
                orc = ({t: base[t] * D / S for t in INIT} if S and D
                       else {t: 0.0 for t in INIT})
                e0 = sum(abs(p0[t] - tp_my[t]) for t in INIT)
                eo = sum(abs(old[t] - tp_my[t]) for t in INIT)
                ec = sum(abs(orc[t] - tp_my[t]) for t in INIT)
                cls.err_p0 += e0
                cls.err_old += eo
                cls.err_oracle += ec
                cls.worst_p0 = max(cls.worst_p0, e0)
                cls.worst_old = max(cls.worst_old, eo)
                cls.n += 1
                trk.commit(mv)

    def test_p0_much_better_than_v511(self):
        a, b = self.err_p0 / self.n, self.err_old / self.n
        red = (1 - a / b) * 100
        print(f"\n    [精度] 平均 L1 池误差: v5.11 {b:.3f} → P0 {a:.3f} (降低 {red:.1f}%)")
        print(f"    [精度] 最坏 L1 误差:   v5.11 {self.worst_old:.2f} → P0 {self.worst_p0:.2f}")
        self.assertLess(a, b * 0.4, "P0 误差应显著低于 v5.11")

    def test_p0_close_to_information_floor(self):
        a = self.err_p0
        c = self.err_oracle
        b = self.err_old
        closed = (1 - (a - c) / (b - c)) * 100
        print(f"    [精度] 信息论上界 (ORACLE) {c / self.n:.3f}; "
              f"P0 弥合了与 v5.11 差距的 {closed:.1f}%")
        self.assertGreater(closed, 85.0, "P0 应接近信息论上界")
        self.assertGreaterEqual(a, c - 1e-6, "不可能优于全知者")


# ---------------------------------------------------------------- 回归
class TestBugRegression(unittest.TestCase):
    """针对 v5.11 具体症状的回归测试。"""

    def test_ghost_dark_rook_in_endgame(self):
        """残局幽灵暗车: 双方明子全被吃、朝下子极少时, v5.11 仍认为池里有暗车。"""
        spec = {(9, 4): "r帥", (0, 4): "b將", (9, 0): "r?"}
        estr = E.board_to_engine_string(mkboard(spec), "r")
        OLD._update_distribution(estr)
        old_sum = sum(OLD.di[0][True].values())
        self.assertEqual(old_sum, 15, "v5.11 应认为池里仍有 15 子 (幽灵)")
        self.assertEqual(OLD.di[0][True]["R"], 2, "v5.11 应认为仍有 2 个暗车")
        trk = E.DarkPoolTracker()
        trk.observe(estr)
        pool = trk.pool(estr, True)
        self.assertAlmostEqual(sum(pool.values()), 1.0, places=9)
        self.assertLessEqual(pool["R"], 2.0 / 15 + 1e-9, "P0 的暗车期望应 ≤ 1 子×先验比例")
        print(f"\n    [回归] 残局 1 子朝下: v5.11 池和={old_sum} (暗车 2.0) → "
              f"P0 池和={sum(pool.values()):.0f} (暗车 {pool['R']:.3f})")

    def test_worst_case_from_logged_game(self):
        """复现分析中最坏的一例: v5.11 认为池里 10 子而盘面只剩 1 子朝下。"""
        worst = None
        for gpath in GAMES:
            for ply, estr, tp_my, tp_op, trev, mv in replay(gpath):
                OLD._update_distribution(estr)
                gap = sum(OLD.di[0][True].values()) - dark_count(estr, True)
                if worst is None or gap > worst[0]:
                    worst = (gap, os.path.basename(gpath), ply, estr)
        gap, name, ply, estr = worst
        self.assertGreaterEqual(gap, 5, "应能找到显著的幽灵子案例")
        trk = E.DarkPoolTracker()
        trk.observe(estr)
        self.assertAlmostEqual(sum(trk.pool(estr, True).values()),
                               dark_count(estr, True), places=9)
        print(f"    [回归] 最坏案例 {name} ply{ply}: v5.11 多算 {gap} 个幽灵暗子, P0 为 0")

    def test_no_zombie_king_key(self):
        """v5.11 的 `p in \"RNBAKCP\"` 会给池塞一个 K:0 僵尸键; P0 不应有。"""
        estr = E.board_to_engine_string(fresh_board(), "r")
        OLD._update_distribution(estr)
        self.assertIn("K", OLD.di[0][True], "v5.11 应存在 K 僵尸键")
        trk = E.DarkPoolTracker()
        trk.observe(estr)
        self.assertNotIn("K", trk.pool(estr, True), "P0 不应出现 K 键")


# ---------------------------------------------------------------- 引擎集成
class TestEngineIntegration(unittest.TestCase):
    """引擎级: 两种模式都必须启用池跟踪; 换边重置; 全局 di/sumall 正确写入。"""

    def test_pool_active_in_referee_mode(self):
        """裁判模式 (下发 check_state) 同样必须跟踪池 —— v5.11 的 live 门控
        会让 A/B 实验恰好测不到修复。"""
        eng = E.JieQiEngine()
        cs = {"count": 0, "squares": [], "retired": 0}
        uci, sc, dp = eng.get_best_move(fresh_board(), "r", think_time=0.3,
                                        check_state=cs)
        self.assertIsNotNone(uci)
        self.assertIsNotNone(eng._pool._my_move, "裁判模式下 commit 未被调用")
        self.assertEqual(sum(E.di[0][True].values()), 15)

    def test_pool_active_in_live_mode(self):
        eng = E.JieQiEngine()
        uci, sc, dp = eng.get_best_move(fresh_board(), "r", think_time=0.3)
        self.assertIsNotNone(uci)
        self.assertIsNotNone(eng._pool._my_move)

    def test_global_di_matches_invariant_after_call(self):
        """get_best_move 之后, 全局 di/sumall 必须满足不变量 (评估读的是它们)。"""
        eng = E.JieQiEngine()
        spec = {(9, 4): "r帥", (0, 4): "b將", (9, 0): "r?", (9, 1): "r?",
                (5, 0): "r車", (0, 0): "b?", (3, 4): "b車"}
        bd = mkboard(spec)
        estr = E.board_to_engine_string(bd, "r")
        eng.get_best_move(bd, "r", think_time=0.3)
        self.assertAlmostEqual(E.sumall[0][True], dark_count(estr, True), places=9)
        self.assertAlmostEqual(E.sumall[0][False], dark_count(estr, False), places=9)

    def test_side_switch_resets_pool(self):
        eng = E.JieQiEngine()
        eng.get_best_move(fresh_board(), "r", think_time=0.2)
        eng._pool.rev[True] = {"R": 2}
        eng.get_best_move(fresh_board(), "b", think_time=0.2)
        self.assertEqual(eng._pool.rev[True], {}, "换边应重置池")

    def test_multi_turn_sequence_keeps_invariant(self):
        """连续多回合驱动引擎, 每回合校验全局不变量 (端到端)。"""
        eng = E.JieQiEngine()
        board = fresh_board()
        secret_r = ["車", "車", "馬", "馬", "相", "相", "仕", "仕",
                    "炮", "炮", "兵", "兵", "兵", "兵", "兵"]
        smap = {rc: t for rc, t in zip(INIT_SQ["r"], secret_r)}
        bsecret = ["車", "車", "馬", "馬", "象", "象", "士", "士",
                   "炮", "炮", "卒", "卒", "卒", "卒", "卒"]
        bmap = {rc: t for rc, t in zip(INIT_SQ["b"], bsecret)}
        for turn in range(6):
            estr = E.board_to_engine_string(board, "r")
            uci, sc, dp = eng.get_best_move(board, "r", think_time=0.2)
            self.assertIsNotNone(uci, f"第 {turn} 回合无着法")
            self.assertAlmostEqual(E.sumall[0][True], dark_count(estr, True),
                                   places=9, msg=f"第 {turn} 回合我方池和错误")
            self.assertAlmostEqual(E.sumall[0][False], dark_count(estr, False),
                                   places=9, msg=f"第 {turn} 回合对方池和错误")
            # 应用我方着法 (翻开真身)
            s_ = (9 - int(uci[1]), ord(uci[0]) - 97)
            d_ = (9 - int(uci[3]), ord(uci[2]) - 97)
            piece = board[s_[0]][s_[1]]
            if piece.endswith("?"):
                piece = "r" + smap.pop(s_)
            board[d_[0]][d_[1]] = piece
            board[s_[0]][s_[1]] = "."
            # 对方走一步暗子 (翻开)
            for rc in list(bmap):
                r2 = rc[0] + 1
                if r2 <= 9 and board[r2][rc[1]] == ".":
                    board[r2][rc[1]] = "b" + bmap.pop(rc)
                    board[rc[0]][rc[1]] = "."
                    break

    def test_deprecated_update_distribution_now_satisfies_invariant(self):
        """旧入口 _update_distribution 被保留 (bench/工具仍调用), 改为等价重标定,
        必须同样满足不变量, 不再返回幽灵子。"""
        spec = {(9, 4): "r帥", (0, 4): "b將", (9, 0): "r?", (5, 0): "r車"}
        estr = E.board_to_engine_string(mkboard(spec), "r")
        E._update_distribution(estr)
        self.assertAlmostEqual(E.sumall[0][True], dark_count(estr, True), places=9)
        self.assertAlmostEqual(E.sumall[0][False], dark_count(estr, False), places=9)


# ---------------------------------------------------------------- 评估影响
class TestEvalImpact(unittest.TestCase):
    """确认修复真的传导到了评估量 (否则只是改了个没人读的数)。

    注意用例构造: v5.11 的 bug 只在"曾翻开、随后被吃"的子上显现 —— 仍在盘面上的
    明子它会正确扣减。因此这些用例必须让明子从盘面消失 (被吃), 才能暴露差异。
    这也解释了为什么 bug 越到中残局越严重: 被吃的明子越积越多。
    """

    @staticmethod
    def _p0_with_history(estr, rev_my, rev_op=None):
        """构造一个"已知这些类型被翻开过"的 P0 池并写入全局 (模拟事件历史)。"""
        trk = E.DarkPoolTracker()
        trk.rev[True] = dict(rev_my)
        if rev_op:
            trk.rev[False] = dict(rev_op)
        trk.observe(estr)          # floor 只会上调, 不会抹掉历史
        trk.apply(estr)
        return trk

    def test_average_table_changes(self):
        """average[..][False] = 单个暗子期望值, 由池组成决定。

        场景: 我方两个車和两个炮都曾翻开并已被吃掉 (盘面无痕)。
        v5.11 认为它们还在池里 → 暗子期望值虚高; P0 知道它们已出池。
        """
        spec = {(9, 4): "r帥", (0, 4): "b將"}
        for rc in INIT_SQ["r"][:6]:
            spec[rc] = "r?"                 # 只剩 6 子朝下
        for rc in INIT_SQ["b"][:6]:
            spec[rc] = "b?"
        estr = E.board_to_engine_string(mkboard(spec), "r")
        OLD._update_distribution(estr)
        OLD.Searcher().calc_average()
        old_avg = OLD.average[0][True][False]
        # 真相: 2車 2炮 5兵 已翻开并被吃, 池里只剩 2馬 2相 2仕 → 但盘面只有 6 子
        self._p0_with_history(estr, {"R": 2, "C": 2, "P": 5})
        E.Searcher().calc_average()
        new_avg = E.average[0][True][False]
        print(f"\n    [评估] 单暗子期望值 avgU (2車2炮5兵已翻开且被吃): "
              f"v5.11 {old_avg} → P0 {new_avg}")
        self.assertNotEqual(old_avg, new_avg, "池修复应改变暗子期望值")
        self.assertLess(new_avg, old_avg,
                        "高价值子已出池, 剩余暗子期望值应下降")

    def test_possible_che_changes(self):
        """possible_che (暗车期望数) 直接驱动 value() 的暗车风险项。

        场景: 我方两個車都曾翻开、随后被吃 → 盘面上看不到任何車。
        v5.11 因此认为"池里还有 2 个暗车"(幽灵); P0 知道暗车期望为 0。
        """
        spec = {(9, 4): "r帥", (0, 4): "b將"}
        for rc in INIT_SQ["r"][:5]:
            spec[rc] = "r?"
        for rc in INIT_SQ["b"][:5]:
            spec[rc] = "b?"
        estr = E.board_to_engine_string(mkboard(spec), "r")
        D = dark_count(estr, True)
        OLD._update_distribution(estr)
        old_e = D * OLD.di[0][True]["R"] / sum(OLD.di[0][True].values())
        trk = self._p0_with_history(estr, {"R": 2})
        new_e = trk.pool(estr, True)["R"]
        print(f"    [评估] E[暗车数] (两車皆已翻开且被吃): "
              f"v5.11 {old_e:.3f} → P0 {new_e:.3f}")
        self.assertAlmostEqual(new_e, 0.0, places=9,
                               msg="車已全部出池, 暗车期望应为 0")
        self.assertGreater(old_e, 0.3, "v5.11 应虚高出幽灵暗车")

    def test_move_value_changes(self):
        """Position.value() 对同一着法的打分应因池修复而改变。"""
        spec = {(9, 4): "r帥", (0, 4): "b將"}
        for rc in INIT_SQ["r"][:5]:
            spec[rc] = "r?"
        for rc in INIT_SQ["b"][:5]:
            spec[rc] = "b?"
        bd = mkboard(spec)
        estr = E.board_to_engine_string(bd, "r")
        OLD._update_distribution(estr)
        OLD.Searcher().calc_average()
        op = OLD.Position(estr, 0, True, 0).set()
        omv = next(m for m in op.gen_moves() if op.board[m[0]] in "DEFGHI")
        oldv = op.value(omv)
        self._p0_with_history(estr, {"R": 2, "C": 2, "P": 5},
                             {"R": 2, "C": 2, "P": 5})
        E.Searcher().calc_average()
        np_ = E.Position(estr, 0, True, 0).set()
        newv = np_.value(omv)
        print(f"    [评估] 同一暗子着法 value(): v5.11 {oldv:.1f} → P0 {newv:.1f} "
              f"(差 {newv - oldv:+.1f})")
        self.assertNotEqual(round(oldv, 6), round(newv, 6),
                            "池修复应改变暗子着法的评估分")

    def test_search_still_returns_legal_move(self):
        """回归: 池改为 float 后 calc_average / value / search 全链路仍正常。"""
        eng = E.JieQiEngine()
        for side in ("r", "b"):
            uci, sc, dp = eng.get_best_move(fresh_board(), side, think_time=0.4)
            self.assertIsNotNone(uci)
            self.assertEqual(len(uci), 4)
            self.assertGreater(dp, 0)

    def test_float_pool_does_not_break_calc_average(self):
        """池是 float (缩放结果), calc_average 内有 round/除法, 确认无异常且有界。"""
        trk = E.DarkPoolTracker()
        spec = {(9, 4): "r帥", (0, 4): "b將"}
        for rc in INIT_SQ["r"][:7]:
            spec[rc] = "r?"
        for rc in INIT_SQ["b"][:3]:
            spec[rc] = "b?"
        estr = E.board_to_engine_string(mkboard(spec), "r")
        trk.observe(estr)
        trk.apply(estr)
        avg = E.Searcher().calc_average()
        self.assertTrue(all(isinstance(v, (int, float))
                            for v in avg[True][True].values()))
        self.assertLess(abs(avg[True][False]), 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
