"""
揭棋引擎 - 基于 miaosiSari/Jieqi 的纯算法引擎 (PST 评估，无需 NNUE)

搜索优化 (等墙钟时间对局实测):
  优化 1 (v5.6): 置换表不再每层迭代清空。原先 alphabeta(root=True) 开头清空
          tp_score/tp_move/history_heur, 导致迭代加深的上层成果全部作废;
          改为每次 search() 开始时清一次, 上层迭代的 hash move 排序与历史启发
          得以指导下层搜索。
  优化 2 (v5.6): 双时限 + 深度封顶 12。原先 range(2,8) 封顶 7 层且时间只在层间检查,
          单层可远超预算; 现在硬时限在层内周期性检查并抛 SearchTimeout 强制中断
          (弃用该层不完整结果, 采用上一层完整结果), 预测式软时限
          (elapsed + 2.0*iter_time > max_time) 决定是否开下一层。
          深度封顶 12: 评估无王安全项, 超深搜索会诱导牺牲王安全的贪吃着法
          (实测无上限时败局全部源于 depth 64 的贪吃)。
  优化 3 (v5.7): 真(递归)静态搜索。原先 depth==0 叶节点只挑一步最优吃子着法
          作为一次性加分 (不考虑对方回应, 也不递归后续互吃), 在吃子密集局面
          系统性高估吃子价值; 现改为标准 negamax 递归静态搜索 (qsearch):
          stand-pat 剪枝 + MVV-LVA 排序的吃子着法递归 (层数封顶 qs_depth=8),
          叶节点静态分 = pos.score + 空头炮分差, 与原口径一致。
          叶节点变尖使 α-β 剪枝增多 (depth 6 提速 2.6×), 10 局等墙钟对局
          净胜上一版 3 局 (5-2-3) 且平均用时不增。
  优化 4 (v5.12 P0): 暗子池跨回合跟踪 (正确性修复)。原 _update_distribution 从
          当前盘面反推暗子池, 把"被吃掉的明子"也算回池里 (被吃的明子同样不在
          盘面上) → 中残局暗车暗炮概率虚高, 评估失真。实测回放全部棋谱 8197
          个局面, 违反不变量 "sum(池) == 盘面朝下子数" 的比例 99%, 最坏时引擎
          认为自己池里有 10 子而盘面只剩 1 子朝下。
          改为 DarkPoolTracker: 事件驱动维护 rev_ever[类型] (翻开可观测 → 精确
          扣减; 暗着被吃类型不可观测 → 按比例缩放, 即无信息条件下的贝叶斯边缘),
          再精确重标定使不变量按构造成立。裁判/实战两种模式均启用。
          实测平均 L1 池误差 5.748 → 1.268 (降低 78%), 弥合了与信息论上界
          (1.081) 之间 96% 的差距; 不变量满足率 99% → 100%。
  优化 5 (v5.13): 确定性方差风险惩罚。在 value() 唯一存在"用确定资源博不确定
          收益"的位置 (吃对方暗子/U) 加一项 CE = E[X] - λ·σ(X) 折扣, σ 来自
          calc_variance 解析二阶矩 (与 calc_average 同遍历, 零搜索开销)。λ=0
          时 byte-for-byte 等价基线; λ=0.5 时 run_paired 20 局 62.5% 配对净胜
          4:1。详见 tools/RISK_PENALTY_report_20260827.md。
  优化 6 (v5.14, 本次): make/unmake 结构级重构。原不可变字符串流水线每节点
          3 次 256 字节整串拷贝 (put×2 + rotate+swapcase) + set() 全盘 153 格
          重扫 + 大量短命对象 GC, 实测 move() 35.6μs、固定深度 6 仅 41 knps。
          改为单一可变棋盘 + undo 栈 + 双色走法生成 + Zobrist 置换表, "换边"
          降级为布尔位翻转。预期节点开销降 3~5×, 同预算深度 +1~2 层, 对应
          "差一步看不见" 的败局会相应减少。设计细节与对拍方法学见
          tools/MAKE_UNMAKE_notes_20260827.md。
"""
import re, os, time, json, random
from itertools import count
from collections import namedtuple
from copy import deepcopy
from board import common_20210815 as common, library

# 棋子类型 → 引擎字母 (不区分颜色)
TYPE_LETTER = {
    "車": "R", "馬": "N", "相": "B", "象": "B", "仕": "A", "士": "A",
    "帥": "K", "將": "K", "炮": "C", "兵": "P", "卒": "P",
}

# 引擎暗子字母 → 棋子类型
DARK_TO_TYPE = {
    "D": "車", "E": "馬", "F": "相", "G": "仕", "H": "炮", "I": "兵",
    "d": "車", "e": "馬", "f": "象", "g": "士", "h": "炮", "i": "卒",
}

# 开局暗子布局 (引擎索引 → 暗子字母)
# 引擎索引 = (12 - row) * 16 + (3 + col), row 0-9, col 0-8
INITIAL_DARK_POSITIONS = {
    # 黑方 (row 0-4, 对手半场)
    (0, 0): "d", (0, 1): "e", (0, 2): "f", (0, 3): "g", (0, 4): "k", (0, 5): "g", (0, 6): "f", (0, 7): "e", (0, 8): "d",
    (2, 1): "h", (2, 7): "h",
    (3, 0): "i", (3, 2): "i", (3, 4): "i", (3, 6): "i", (3, 8): "i",
    # 红方 (row 5-9, 己方半场)
    (6, 0): "I", (6, 2): "I", (6, 4): "I", (6, 6): "I", (6, 8): "I",
    (7, 1): "H", (7, 7): "H",
    (9, 0): "D", (9, 1): "E", (9, 2): "F", (9, 3): "G", (9, 4): "K", (9, 5): "G", (9, 6): "F", (9, 7): "E", (9, 8): "D",
}

# 棋子价值
PIECE_VALUE = {"P": 44, "N": 108, "B": 23, "R": 233, "A": 23, "C": 101, "K": 2500}

# MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
# 用于吃子着法排序: 被吃子价值越高、攻击子价值越低, 优先级越高
# 明子类型 -> 基础价值(用于排序, 非评估值)
_MVV_BASE = {"P": 100, "N": 300, "B": 300, "R": 500, "A": 300, "C": 300, "K": 10000,
             "I": 100, "E": 300, "F": 300, "D": 500, "G": 300, "H": 300, "U": 200}

# 方向常量
N, E, S, W = -16, 1, 16, -1

# 走法方向 (大写方 = 红方, 固定在下方; 黑方方向表见 DIR_LOWER)
DIRECTIONS = {
    "P": (N, W, E),
    "I": (N,),  # 暗兵
    "N": (N + N + E, E + N + E, E + S + E, S + S + E, S + S + W, W + S + W, W + N + W, N + N + W),
    "E": (N + N + E, E + N + E, W + N + W, N + N + W),  # 暗马
    "B": (2 * N + 2 * E, 2 * S + 2 * E, 2 * S + 2 * W, 2 * N + 2 * W),
    "F": (2 * N + 2 * E, 2 * N + 2 * W),  # 暗相
    "R": (N, E, S, W),
    "D": (N, E, W),  # 暗车
    "C": (N, E, S, W),
    "H": (N, E, S, W),  # 暗炮
    "A": (N + E, S + E, S + W, N + W),
    "G": (N + E, N + W),  # 暗士
    "K": (N, E, S, W),
}

# 小写 (黑) 方向表: 派生规则: 旧引擎子节点是"旋转 180° 视角", 旋转坐标方向 d'
# 对应本盘方向 -d', 元组内逐元素映射且顺序保持不变 → 着法枚举次序与旧引擎逐一相同
# (排序平局时关键)。黑方枚举 203→51 降序, 精确对应旧旋转视角的 51→203 升序。
DIR_LOWER = {
    "p": (S, E, W),
    "i": (S,),
    "n": (S + S + W, W + S + W, W + N + W, N + N + W,
          N + N + E, E + N + E, E + S + E, S + S + E),
    "e": (S + S + W, W + S + W, E + S + E, S + S + E),
    "b": (2 * S + 2 * W, 2 * N + 2 * W, 2 * N + 2 * E, 2 * S + 2 * E),
    "f": (2 * S + 2 * W, 2 * S + 2 * E),
    "r": (S, W, N, E),
    "d": (S, W, E),
    "c": (S, W, N, E),
    "h": (S, W, N, E),
    "a": (S + W, N + W, N + E, S + E),
    "g": (S + W, S + E),
    "k": (S, W, N, E),
}

A0, I0, A9, I9 = 12 * 16 + 3, 12 * 16 + 11, 3 * 16 + 3, 3 * 16 + 11
MATE_LOWER = PIECE_VALUE["K"] - (2 * PIECE_VALUE["R"] + 2 * PIECE_VALUE["N"] + 2 * PIECE_VALUE["B"] + 2 * PIECE_VALUE["A"] + 2 * PIECE_VALUE["C"] + 5 * PIECE_VALUE["P"])
MATE_UPPER = PIECE_VALUE["K"] + (2 * PIECE_VALUE["R"] + 2 * PIECE_VALUE["N"] + 2 * PIECE_VALUE["B"] + 2 * PIECE_VALUE["A"] + 2 * PIECE_VALUE["C"] + 5 * PIECE_VALUE["P"])
TABLE_SIZE = 1e7
EVAL_ROUGHNESS = 13

put = lambda board, i, p: board[:i] + p + board[i + 1:]


# ---------------- Zobrist 随机键 ----------------
# 棋盘 + 轮走/暗子分布视角位, make/unmake O(1) 增减; 旧 TT 以 Position (含整串棋盘)
# 做键, 单棋盘下失效, 故必须引入真哈希。
_ZR = random.Random(20260824)
ZOBRIST = {}
for _pc in "RNBAKCPDEFGHIU" + "rnbakcpdefghiu":
    ZOBRIST[_pc] = [_ZR.getrandbits(64) for _ in range(256)]
for _pc in ".\n ":                       # 空格无子, 异或中性
    ZOBRIST[_pc] = [0] * 256
Z_STM = _ZR.getrandbits(64)    # 轮走位 (真实轮走方)
Z_TURN = _ZR.getrandbits(64)   # turn 标志位 (暗子分布视角, 仅 nullmove 翻转)

LIGHT = "RNBAKCP"          # 明子字母 (走子后字母不变; 判断用 upper(), 红明子为小写)
BACK_SET_U = "DEFGRNC"     # 底线子统计集 (不含兵/王/暗U), 大写
BACK_SET_L = "defgrnc"
LOWER_NONSLIDE = "pnbakief" + "g"   # 小写非滑动子 (大写 "PNBAKIEFG" 的镜像集)


def _mvv_lva(bd, move, upper_moves):
    """侧向感知的 MVV-LVA 排序分 (旧 _mvv_lva_score 的双色版, 分值口径一致)。
    upper_moves=True 时: attacker=大写, victim=小写才算吃子; 反之亦然。
    """
    i, j = move
    attacker, victim = bd[i], bd[j]
    if victim == "." or victim.isspace():
        return 0
    if upper_moves:
        if victim.isupper():
            return 0
    else:
        if victim.islower():
            return 0
    v_val = _MVV_BASE.get(victim.upper(), 200)
    a_val = _MVV_BASE.get(attacker.upper(), 200)
    return v_val * 1024 - a_val


# ---------------------------------------------------------------------------
# 评估/暗子池全局状态 (DarkPoolTracker 写入, Searcher 读取)
# ---------------------------------------------------------------------------
r = {"R": 2, "N": 2, "B": 2, "A": 2, "C": 2, "P": 5}
b = {"r": 2, "n": 2, "b": 2, "a": 2, "c": 2, "p": 5}
di = {0: {True: deepcopy(r), False: deepcopy(b)}}
sumall = {0: {True: sum(di[0][True][key] for key in di[0][True]),
              False: sum(di[0][False][key] for key in di[0][False])}}
pst = deepcopy(common.pst)
average = {0: {}}
risk_sigma = {0: {}}
kaijuku = deepcopy(library.kaijuku)

# ---------------- [v5.13] 确定性方差风险惩罚 (Variance-Adjusted Certainty Equivalent) ----------------
# 动机: PIMC (根重采样+K世界投票) 实测负结果 (见 tools/PIMC_report_20260826.md) ——
#   蒙特卡洛去逼近方差既贵(4×算力)又不准(K=4方差发散), 且丢了基线的调参经验项。
#   这里改用解析二阶矩: calc_variance() 与 calc_average() 同一遍历一次算出池标准差,
#   零额外搜索开销。只对"用确定资源博不确定收益"的唯一场景 (吃对方暗子/U) 生效:
#   score += E[X] - λ·σ(X)，λ=0 时与基线 byte-for-byte 等价。
RISK_LAMBDA = 0.0


def set_risk_lambda(v):
    global RISK_LAMBDA
    RISK_LAMBDA = float(v) if v else 0.0


# ===========================================================================
# [v5.14] 单一可变棋盘 + make/unmake 骨架
# ===========================================================================
#
# 设计要点 (详见 tools/MAKE_UNMAKE_notes_20260827.md):
#   1. 棋盘固定, 大写=红(下方), 小写=黑(上方); 换边 = stm 标志翻转 + turn 翻,
#      棋盘本身不旋转不 swapcase (消掉 3 次/节点的 256 字节整串拷贝)。
#   2. 走子/撤销: 改两个格 + 弹 undo 栈, 零分配、零 GC。
#   3. 全增量统计: cov_u/cov_l (暗子数), che_u/che_l (车数), zu_u/zu_l (兵数),
#      back_u/back_l (底线子数), rough (粗分和, 含 U 项), rnci_u/rnci_l
#      (R/N/C/I 计数, 用于空着裁剪), kt_u/kt_l + kts_u/kts_l (空头炮局部列
#      重算结果, 与旧 set() 全盘扫描逐字一致)。
#   4. 双色走法生成: 上方与黑方各一份独立的 gen_moves 逻辑, 黑方降序枚举精确
#      对应旧旋转视角的升序, 保证排序平局时着法次序逐字一致。
#   5. Zobrist 哈希: 局面键 = hkey; 旧 TT 以 Position 对象做键不可用, 现用
#      64bit 异或 + stm/turn 位, make/unmake O(1) 维护。
#   6. 旧 Searcher 的剪枝策略 (PVS/LMR/nullmove/双时限/qs_depth) 全部保留,
#     只把 pos.move() / pos.rotate() 换成 st.make() / st.unmake() / st.flip_stm()。
#
# 关键正确性契约:
#   - λ=0 (RISK_LAMBDA) + 关 LMR 走 alphabeta(root=True) 在同一局面应精确等于
#     旧实现的同一调用 (子节点序列 + 根值逐字相等), 此为对拍硬闸门。
#   - 一致性 on/off 测试由 tools/MU_diff_pairwise.py 验证 (待写)。
# ===========================================================================


class State:
    """单一可变棋盘 + 全增量统计 + undo 栈。
    大写=红(固定下方), 小写=黑(上方)。走子方由 stm 标志指示, 切到黑走时
    value() 自动用镜像几何+镜像表, 与旧引擎逐字同构。
    """

    __slots__ = (
        "board", "stm", "turn", "version", "score", "hkey", "undo",
        "rough", "che_u", "che_l", "zu_u", "zu_l",
        "cov_u", "cov_l", "back_u", "back_l",
        "rnci_u", "rnci_l", "kt_u", "kt_l", "kts_u", "kts_l",
    )

    def __init__(self, board, stm=True, turn=True, version=0):
        self.board = board
        self.stm = stm
        self.turn = turn
        self.version = version
        self.score = 0
        self.hkey = 0
        self.undo = []
        # [rough 语义实证] 逐深度实测 (含王/暗子/吃王局面): 旧 set() 的 rough
        # 与规范帧计算值在偶数旋转层相等、奇数层互为相反数 —— 恰等于
        # "符号随 stm 翻转" (stm 与旋转奇偶同步)。成因: PST 表不中心对称,
        # 旋转后大写方集合换成对方, 粗分和变号; U 项因 turn 与奇偶同步翻转
        # 而保持同构。故单变体增量 + 按 stm 取符号即可逐字复刻。
        self.rough = 0
        self.che_u = self.che_l = 0
        self.zu_u = self.zu_l = 0
        self.cov_u = self.cov_l = 0
        self.back_u = self.back_l = 0
        self.rnci_u = {"R": 0, "N": 0, "C": 0, "I": 0}
        self.rnci_l = {"R": 0, "N": 0, "C": 0, "I": 0}
        self.kt_u = self.kt_l = 0
        self.kts_u = self.kts_l = 0

    @property
    def board_str(self):
        """给 kaijuku 这类用字符串键的下游用。"""
        return "".join(self.board)

    # ---------- 统计增量 ----------
    def _pstats(self, ch, i, s):
        """在格 i 放置(s=+1)/移除(s=-1) 棋子 ch, O(1) 更新全部计数。"""
        if ch == "." or ch == "\n" or ch == " ":
            return
        v = self.version
        if ch.isupper():
            if ch == "U":
                self.rough += s * average[v][True][True][i]
                self.cov_u += s
            elif ch in LIGHT:
                self.rough += s * pst[ch][i]
                if ch == "R":
                    self.che_u += s; self.rnci_u["R"] += s
                elif ch == "P":
                    self.zu_u += s
                elif ch == "N":
                    self.rnci_u["N"] += s
                elif ch == "C":
                    self.rnci_u["C"] += s
                elif ch == "K":
                    pass
            else:                            # 暗子 DEFGHI: 只计暗子数, 不入 rough
                self.cov_u += s
                if ch == "I":
                    self.rnci_u["I"] += s
            # 底线子统计: 明暗皆计 (旧 set() 扫 "defgrnc" 不分明暗), U 不在集内
            if (i >> 4) == 12 and ch in BACK_SET_U:
                self.back_u += s
        else:
            if ch == "u":
                self.rough -= s * average[v][False][True][254 - i]
                self.cov_l += s
            elif ch in "rnbakcp":
                self.rough -= s * pst[ch.upper()][254 - i]
                if ch == "r":
                    self.che_l += s; self.rnci_l["R"] += s
                elif ch == "p":
                    self.zu_l += s
                elif ch == "n":
                    self.rnci_l["N"] += s
                elif ch == "c":
                    self.rnci_l["C"] += s
            else:                            # 暗子 defghi
                self.cov_l += s
                if ch == "i":
                    self.rnci_l["I"] += s
            if (i >> 4) == 3 and ch in BACK_SET_L:
                self.back_l += s

    # ---------- 空头炮 (局部列重算, 口径与旧 set() 逐字一致) ----------
    def _kongtou(self):
        bd = self.board
        ku = kl = 0
        done_u = done_l = False
        for i in range(51, 204):
            if i & 15 != 7:
                continue
            ch = bd[i]
            if ch == "C" and not done_u:
                done_u = True
                for sp in range(i - 16, 51, -16):
                    q = bd[sp]
                    if q == "C":
                        continue
                    elif q != ".":
                        if q == "k":
                            ku += 1
                        break
                    else:
                        ku += 1
            elif ch == "c" and not done_l:
                done_l = True
                for sp in range(i + 16, 204, 16):
                    q = bd[sp]
                    if q == "c":
                        continue
                    elif q != ".":
                        if q == "K":
                            kl += 1
                        break
                    else:
                        kl += 1
        self.kt_u, self.kt_l = ku, kl
        kts_u = kts_l = 0
        if (ku > 0 and kl <= 0) or (ku > kl > 0):
            kts_u = 100 if ((self.che_u >= self.che_l and self.che_u > 0) or ku >= 3) else 70
        elif (kl > 0 and ku <= 0) or (kl > ku > 0):
            kts_l = 100 if ((self.che_l >= self.che_u and self.che_l > 0) or kl >= 3) else 70
        self.kts_u, self.kts_l = kts_u, kts_l

    @classmethod
    def from_string(cls, s, version=0):
        bd = list(s)
        st = cls(bd, True, True, version)
        h = 0
        for i in range(256):
            ch = bd[i]
            if ch not in ".\n ":
                st._pstats(ch, i, 1)
                h ^= ZOBRIST[ch][i]
        st.hkey = h
        st._kongtou()
        return st

    # ---------- 本方视角统计 ----------
    def rough_m(self):
        # 旧引擎 rough 在奇数旋转层取反 (逐深度实测验证, 见 __init__ 注)。
        return self.rough if self.stm else -self.rough

    def cov_m(self):
        return self.cov_u if self.stm else self.cov_l

    def cov_o(self):
        return self.cov_l if self.stm else self.cov_u

    def che_m(self):
        return self.che_u if self.stm else self.che_l

    def che_o(self):
        return self.che_l if self.stm else self.che_u

    def endline_m(self):
        # 旧 set() 语义: 在本方视角的 i>>4==3 一行计子数。经旋转映射:
        # stm=True 时旧 i>>4==3 = 本盘 row0 = back_l; stm=False 时(已旋转换边)
        # 旧视角 i>>4==3 = 本盘 row9 = back_u。
        return self.back_l if self.stm else self.back_u

    def static(self):
        """叶节点静态分 (行棋方视角), 与旧 _static 口径一致。"""
        if self.stm:
            return self.score + self.kts_u - self.kts_l
        return self.score + self.kts_l - self.kts_u

    def has_rnci(self):
        t = self.rnci_u if self.stm else self.rnci_l
        return t["R"] or t["N"] or t["C"] or t["I"]

    # ---------- make / unmake ----------
    def make(self, i, j):
        bd = self.board
        src_ch, dst_ch = bd[i], bd[j]
        mv = self.value(i, j)
        ns = self.score + mv if mv < MATE_UPPER else MATE_UPPER
        self.undo.append((i, j, src_ch, dst_ch, self.score, self.hkey))
        self._pstats(src_ch, i, -1)
        self._pstats(dst_ch, j, -1)
        if src_ch.upper() in LIGHT:
            new_ch = src_ch
        else:
            new_ch = "U" if self.stm else "u"
        bd[i] = "."
        bd[j] = new_ch
        self._pstats(new_ch, j, 1)
        self.score = -ns
        self.hkey ^= ZOBRIST[src_ch][i] ^ ZOBRIST[dst_ch][j] ^ ZOBRIST[new_ch][j] ^ Z_STM ^ Z_TURN
        self.stm = not self.stm
        # [关键语义] 旧 move()→rotate_new() 每层翻 turn (实测: root True → child
        # False → grandchild True)。rough 用双变体维护后与 turn 解耦, value()
        # 的 di/average 索引仍按当前 turn 取值, 与旧逐层行为一致。
        self.turn = not self.turn
        self._kongtou()

    def unmake(self):
        i, j, src_ch, dst_ch, old_score, old_hkey = self.undo.pop()
        bd = self.board
        new_ch = bd[j]
        self.score = old_score
        self.hkey = old_hkey
        self.stm = not self.stm
        self.turn = not self.turn
        self._pstats(new_ch, j, -1)
        self._pstats(src_ch, i, 1)
        self._pstats(dst_ch, j, 1)
        bd[i] = src_ch
        bd[j] = dst_ch
        self._kongtou()

    def flip_stm(self):
        """nullmove: 只换边不动盘 (旧 pos.rotate() 的等价物)。旧 rotate() 翻 turn
        并取反 score (move()/rotate_new 不翻), 故此处翻 turn 一次 + 取反 score,
        并异或 Z_TURN (stm 翻转由 Z_STM 体现, 与 make/unmake 一致)。"""
        self.stm = not self.stm
        self.turn = not self.turn
        self.score = -self.score
        self.hkey ^= Z_STM ^ Z_TURN

    # ---------- 走法生成 (双色) ----------
    def gen_moves(self, side=None):
        """生成 side 方着法; side=None 用当前轮走方。枚举次序与旧引擎逐一对应。"""
        bd = self.board
        upper = self.stm if side is None else side
        if upper:
            for i in range(51, 204):
                p = bd[i]
                if not p.isupper() or p == "U":
                    continue
                if p == "K":
                    for sp in range(i - 16, A9, -16):
                        if bd[sp] == "k":
                            yield (i, sp)
                        elif bd[sp] != ".":
                            break
                if p in ("C", "H"):
                    for d in DIRECTIONS[p]:
                        cfoot = 0
                        for j in count(i + d, d):
                            q = bd[j]
                            if q.isspace():
                                break
                            if cfoot == 0 and q == ".":
                                yield (i, j)
                            elif cfoot == 0 and q != ".":
                                cfoot += 1
                            elif cfoot == 1 and q.islower():
                                yield (i, j); break
                            elif cfoot == 1 and q.isupper():
                                break
                    continue
                for d in DIRECTIONS[p]:
                    for j in count(i + d, d):
                        q = bd[j]
                        if q.isspace() or q.isupper():
                            break
                        if p == "P" and d in (E, W) and i > 128:
                            break
                        elif p == "K" and (j < 160 or j & 15 > 8 or j & 15 < 6):
                            break
                        elif p == "G" and j != 183:
                            break
                        elif p in ("N", "E"):
                            n_diff_x = (j - i) & 15
                            if n_diff_x == 14 or n_diff_x == 2:
                                if bd[i + (1 if n_diff_x == 2 else -1)] != ".":
                                    break
                            else:
                                if j > i and bd[i + 16] != ".":
                                    break
                                elif j < i and bd[i - 16] != ".":
                                    break
                        elif p in ("B", "F") and bd[i + d // 2] != ".":
                            break
                        yield (i, j)
                        if p in "PNBAKIEFG" or q.islower():
                            break
        else:
            for i in range(203, 50, -1):
                p = bd[i]
                if not p.islower() or p == "u":
                    continue
                if p == "k":
                    for sp in range(i + 16, I0, 16):
                        if bd[sp] == "K":
                            yield (i, sp)
                        elif bd[sp] != ".":
                            break
                if p in ("c", "h"):
                    for d in DIR_LOWER[p]:
                        cfoot = 0
                        for j in count(i + d, d):
                            q = bd[j]
                            if q.isspace():
                                break
                            if cfoot == 0 and q == ".":
                                yield (i, j)
                            elif cfoot == 0 and q != ".":
                                cfoot += 1
                            elif cfoot == 1 and q.isupper():
                                yield (i, j); break
                            elif cfoot == 1 and q.islower():
                                break
                    continue
                for d in DIR_LOWER[p]:
                    for j in count(i + d, d):
                        q = bd[j]
                        if q.isspace() or q.islower():
                            break
                        if p == "p" and d in (E, W) and i < 126:
                            break
                        elif p == "k" and (j > 94 or j & 15 > 8 or j & 15 < 6):
                            break
                        elif p == "g" and j != 71:
                            break
                        elif p in ("n", "e"):
                            n_diff_x = (j - i) & 15
                            if n_diff_x == 14 or n_diff_x == 2:
                                if bd[i + (1 if n_diff_x == 2 else -1)] != ".":
                                    break
                            else:
                                if j > i and bd[i + 16] != ".":
                                    break
                                elif j < i and bd[i - 16] != ".":
                                    break
                        elif p in ("b", "f") and bd[i + d // 2] != ".":
                            break
                        yield (i, j)
                        if p in LOWER_NONSLIDE or q.isupper():
                            break

    def can_capture_king(self, side):
        """side 方能否一步吃掉对方王 (旧 board[m[1]]=='k' 扫描的等价物)。"""
        bd = self.board
        target = "k" if side else "K"
        for m in self.gen_moves(side):
            if bd[m[1]] == target:
                return True
        return False

    # ---------- 单步增量评估 value() ----------
    def value(self, i, j):
        """与旧 Position.value 逐字同构; stm=False(黑走)时用镜像几何+镜像表。"""
        bd = self.board
        v, t = self.version, self.turn
        if self.stm:
            vi, vj = i, j
            covered, covered_o = self.cov_u, self.cov_l
            che, che_o = self.che_u, self.che_l
            endline = self.back_l
            score_rough = self.rough
            def R(x):
                return bd[x]
        else:
            vi, vj = 254 - i, 254 - j
            covered, covered_o = self.cov_l, self.cov_u
            che, che_o = self.che_l, self.che_u
            endline = self.back_u
            score_rough = -self.rough
            def R(x):
                return bd[254 - x].swapcase()

        p, q = R(vi), R(vj).upper()
        possible_che = 0 if sumall[v][t] == 0 else covered * di[v][t]["R" if t else "r"] / sumall[v][t]
        possible_che_opponent = 0 if sumall[v][not t] == 0 else covered_o * di[v][not t]["r" if t else "R"] / sumall[v][not t]
        if q == "K":
            return MATE_UPPER
        if p.isupper() and p in "RNBAKCP":
            # vi/vj 即旧引擎旋转视角坐标, 直接查 pst (无需镜像表)
            score = pst[p][vj] - pst[p][vi]
            if p == "C":
                if (vi >> 4) != 3 and (vj >> 4) == 3 and endline <= 2:
                    if (vj == 51 or vj == 52) and R(53) == "f" and R(54) == "g":
                        pass
                    elif (vj == 59 or vj == 58) and R(57) == "f" and R(56) == "g":
                        pass
                    else:
                        score -= 55 if endline == 0 else 30
                if (vi >> 4) == 3 and (vj >> 4) != 3 and endline <= 2:
                    if (vi == 51 or vi == 52) and R(53) == "f" and R(54) == "g":
                        pass
                    elif (vi == 59 or vi == 58) and R(57) == "f" and R(56) == "g":
                        pass
                    else:
                        score += 55 if endline == 0 else 30
            elif p == "R":
                if R(51) not in "dr" and R(54) != "a" and R(71) != "a" and (R(71) == "p" or R(87) != "n"):
                    if vj & 15 == 6 and vi & 15 != 6:
                        score += 30
                    if vj & 15 != 6 and vi & 15 == 6:
                        score -= 30
                if R(59) not in "dr" and R(56) != "a" and R(71) != "a" and (R(71) == "p" or R(87) != "n"):
                    if vj & 15 == 8 and vi & 15 != 8:
                        score += 30
                    if vj & 15 != 8 and vi & 15 == 8:
                        score -= 30
                if (vi >> 4) == 3 and (vj >> 4) != 3 and (endline <= 1 or score_rough < -150):
                    score += 40 if score_rough < -150 else 30
                if (vi >> 4) != 3 and (vj >> 4) == 3 and (endline <= 1 or score_rough < -150):
                    score -= 40 if score_rough < -150 else 30
        else:
            score = average[v][t][True][vj] - average[v][t][False] + 20
            if p == "D":
                minus = 30 * (possible_che_opponent / 2 + che_o)
                score -= minus
                if score_rough < -150:
                    score -= minus // 2
            elif p == "I":
                if R(vi - 32) in "rp":
                    score -= average[v][t][False] // 2
                elif R(vi - 32) in "nc":
                    score += 30
                elif R(vi - 48) == "i":
                    score += 30
                else:
                    score += 20

        if q.isupper():
            k = 254 - vj
            if q in "RNBAKCP":
                score += pst[q][k]
                if q == "P" and R(vj + 32) == "I":
                    score += 30
            else:
                if q != "U":
                    score += average[v][not t][False]
                    if RISK_LAMBDA:
                        score -= RISK_LAMBDA * risk_sigma[v][not t][False]
                    if q == "I":
                        score += 10
                else:
                    score += average[v][not t][True][k]
                    if RISK_LAMBDA:
                        score -= RISK_LAMBDA * risk_sigma[v][not t][True][k]
                    if vj >> 4 == 7 and vj & 1 == 1:
                        score += 10
                if q == "D":
                    addition = 30 * (possible_che / 2 + che)
                    score += addition
                    if score_rough > 150:
                        score += addition // 2
        return score


# 旧 TT 用 (pos, depth, root) 做键, namedtuple 默认按字段值等值, 含整串 board
# (256 字符)。make/unmake 后单棋盘上 Position 不同身份键失效, 故引入真哈希:
# hkey 已编码 board+turn+version, score 显式入键 → 与旧语义完全一致。
# 旧 SEARCH_TIMEOUT 兼容: 仍由 SearchTimeout 异常抛出。
Entry = namedtuple("Entry", "lower upper")


class Position:
    """[v5.14 兼容垫片] 旧版 `Position(estr, 0, True, 0).set()` / `.gen_moves()` /
    `.rotate()` 接口的等价包装, 委托给 State, 让 referee.py / bench_v2.py / 旧
    工具脚本零改动地工作。`.rotate()` 仍返回新对象 (用 flip_stm 的非破坏性版本)。
    """
    # 不写 __slots__: 旧 Position 工具脚本可能临时给实例挂任意属性, 留宽松点

    def __init__(self, estr, score, turn, version):
        # score 字段保留 (旧接口接收), 但本引擎的真值在 State 内部 (从 set() 算)
        self._st = State.from_string(estr, version)
        # 旧 Position.score 是评估的"绝对分" (含空头炮), State.score 是行棋方
        # 视角分, 数值上等价; turn/board 字段兼容保留:
        self.turn = self._st.turn
        self.version = self._st.version
        self.board = self._st.board_str   # 旧 Position.board 是 str, State 是 list
        self.score = self._st.score

    def set(self):
        """旧 Position.set() 触发全盘重算 + 增量字段, State.from_string 已经做过
        同样的事, 这里只把 self.score 与 State.score 同步一下即可。"""
        self.score = self._st.score
        return self

    @property
    def st(self):
        return self._st

    def gen_moves(self):
        return self._st.gen_moves()

    def move(self, mv):
        """旧 Position.move 返回新 Position; 我们做一份"软副本"用作兼容,
        但需要旋转/换边时 (旧 .move→rotate) 请优先用 State.make/unmake。"""
        new = State(list(self._st.board), self._st.stm, self._st.turn, self._st.version)
        new.rough = self._st.rough
        new.che_u, new.che_l = self._st.che_u, self._st.che_l
        new.zu_u, new.zu_l = self._st.zu_u, self._st.zu_l
        new.cov_u, new.cov_l = self._st.cov_u, self._st.cov_l
        new.back_u, new.back_l = self._st.back_u, self._st.back_l
        new.rnci_u = dict(self._st.rnci_u)
        new.rnci_l = dict(self._st.rnci_l)
        new.kt_u, new.kt_l = self._st.kt_u, self._st.kt_l
        new.kts_u, new.kts_l = self._st.kts_u, self._st.kts_l
        new.score = self._st.score
        new.hkey = self._st.hkey
        new.make(mv[0], mv[1])
        return Position._from_state(new)

    def rotate(self):
        """旧 Position.rotate: 翻转 board+swapcase+取反 score+取反 turn+重算 set。

        [修正] 初版垫片只翻了 stm/turn, 没有翻盘面, 与旧实现不等价: 旋转后
        gen_moves() 生成的仍是原阵营的着法, 于是 referee.in_check 的判据
        `board[m[1]] == "k"` 变成在问"对方能否吃到自己的将" —— 恒为 False。
        后果是裁判的连将配额永远不累计、长将判负从不触发, 所有"规则零判负"
        的对局结论都因此失去意义。这里补回真正的翻盘 + swapcase。

        语义: 切到对方视角后, 对方成为"大写/行棋方", 换边由 swapcase 承担,
        故 stm 保持不变、turn 取反。增量统计不可沿用旧值 (swapcase 后上下/
        明暗计数互换), 一律用 from_string 重算, 与旧 set() 等价。
        """
        bd = self._st.board
        # 旧写法 board[-2::-1].swapcase() + " ": 索引 i -> 254-i, 逐字符换大小写;
        # 末位 255 是补位空格, 不参与翻转。
        flipped = "".join(bd[254 - i].swapcase() for i in range(255)) + " "
        new = State.from_string(flipped, self._st.version)
        new.stm = self._st.stm
        new.turn = not self._st.turn
        new.score = -self._st.score
        new.hkey = self._st.hkey ^ Z_STM ^ Z_TURN
        return Position._from_state(new)

    @classmethod
    def _from_state(cls, st):
        """从已建好的 State 包一个 Position 视图 (浅层)。"""
        obj = cls.__new__(cls)
        obj._st = st
        obj.turn = st.turn
        obj.version = st.version
        obj.board = st.board_str
        obj.score = st.score
        return obj


class SearchTimeout(Exception):
    """[v2] 硬时限超时: 沿递归栈抛出, 由 search() 捕获并弃用本层不完整结果。"""


class Searcher:
    def __init__(self):
        self.tp_score = {}
        self.tp_move = {}
        self.history = set()
        self.nodes = 0
        self.history_heur = {}
        # LMR 参数: 前几着不做 reduction, 以及 depth 门槛
        self.lmr_full_moves = 4
        self.lmr_min_depth = 3
        self.lmr_base_reduction = 1
        # [v2] 双时限: 硬时限时间戳 (search() 设置, 0 表示不启用) 与节点检查间隔
        self.deadline = 0.0
        self.check_interval = 2048
        # [v3] 静态搜索递归层数上限 (吃子着法链; 防止互吃序列无限拉长)
        self.qs_depth = 8
        # 平均/方差缓存: calc_average/variance 写入, value() 读取
        self.average = {0: {}}
        self.risk_sigma = {0: {}}

    def _tt_key(self, st, depth, root):
        return (st.hkey, st.score, depth, root)

    def _mv_key(self, st):
        return (st.hkey, st.score)

    def qsearch(self, st, alpha, beta, qdepth):
        """[v3] 真静态搜索: negamax 递归吃子序列 + stand-pat 剪枝。
        返回行棋方视角的分数; 只递归吃子着法 (含吃王检测), 层数封顶 qdepth。"""
        self.nodes += 1
        if self.deadline and self.nodes % self.check_interval == 0 and time.time() > self.deadline:
            raise SearchTimeout()
        if st.score <= -MATE_LOWER:
            return -MATE_UPPER
        # stand-pat: 允许不吃子直接采用静态分
        stand = st.static()
        if qdepth <= 0:
            return stand
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        bd = st.board
        king_ch = "k" if st.stm else "K"      # 对方王: stm=大写时对方是小写
        caps = []
        for m in st.gen_moves():
            q = bd[m[1]]
            if q == king_ch:
                return MATE_UPPER
            if q != "." and not q.isspace():
                caps.append(m)
        caps.sort(key=lambda m: _mvv_lva(bd, m, st.stm), reverse=True)
        best = stand
        for m in caps:
            st.make(m[0], m[1])
            val = -self.qsearch(st, -beta, -alpha, qdepth - 1)
            st.unmake()
            if val > best:
                best = val
                if val > alpha:
                    alpha = val
                if val >= beta:
                    break
        return best

    def _order_moves(self, st, moves, tt_move):
        """按优先级对着法排序: TT move > MVV-LVA(吃子) > 历史启发(非吃子)。
        返回排序后的着法列表(含 TT move, 去重)。"""
        tt_list = [tt_move] if tt_move is not None else []
        bd = st.board
        captures = []
        quiets = []
        for m in moves:
            if tt_move is not None and m == tt_move:
                continue  # TT move 单独放最前面
            if _mvv_lva(bd, m, st.stm) > 0:
                captures.append(m)
            else:
                quiets.append(m)
        captures.sort(key=lambda m: _mvv_lva(bd, m, st.stm), reverse=True)
        quiets.sort(key=lambda m: self.history_heur.get(m, 0), reverse=True)
        return tt_list + captures + quiets

    def _is_capture(self, st, move):
        """判断是否为吃子着法 (按当前轮走方视角)。"""
        q = st.board[move[1]]
        return q != "." and not q.isspace() and (q.islower() if st.stm else q.isupper())

    def alphabeta(self, st, alpha, beta, depth, root=True, nullmove=False, nullmove_now=False):
        """alpha-beta + PVS/LMR/nullmove/qsearch, 主体与旧引擎逐行同构, 只
        把 pos.move()→st.make(), pos.rotate()→st.flip_stm() 一处接口替换。
        关键语义保留:
          - 走子 → 换边: 旧 move()→rotate_new() 每层翻 turn, 新 make() 也翻 turn
            (State 内部) + flip_stm, 与旧一致;
          - nullmove (换边不动盘) → 旧用 pos.rotate() (再 rotate 一次回滚
            make 翻转), 新用 st.flip_stm() 两次 (翻两次等价于不动, 一次
            flip_stm 单独用);
          - TT 键: 旧 (pos, depth, root), 新 (hkey, score, depth, root), 语义
            等价 (hkey 已含 board+turn+version)。
        """
        self.nodes += 1
        if self.deadline and self.nodes % self.check_interval == 0 and time.time() > self.deadline:
            raise SearchTimeout()
        depth = max(depth, 0)
        if st.score <= -MATE_LOWER:
            return -MATE_UPPER
        bd = st.board
        king_ch = "k" if st.stm else "K"
        raw_moves = list(st.gen_moves())
        killer = self.tp_move.get(self._mv_key(st))
        # 先检查杀棋
        for move in [killer] + raw_moves if killer else raw_moves:
            if move is not None and bd[move[1]] == king_ch:
                self.tp_move[self._mv_key(st)] = move
                return MATE_UPPER
        key = self._tt_key(st, depth, root)
        entry = self.tp_score.get(key, Entry(-MATE_UPPER, MATE_UPPER))
        # [v5.11] 杀王分豁免: 边界值落在杀王区间 (abs>=MATE_LOWER) 时不参与 TT 截断。
        if entry.lower >= beta and abs(entry.lower) < MATE_LOWER \
                and (not root or self.tp_move.get(self._mv_key(st)) is not None):
            return entry.lower
        if entry.upper < alpha and abs(entry.upper) < MATE_LOWER:
            return entry.upper
        if nullmove_now and depth > 3 and not root and st.has_rnci():
            if not st.can_capture_king(not st.stm):
                st.flip_stm()
                val = -self.alphabeta(st, -beta, 1 - beta, depth - 3, root=False, nullmove=nullmove, nullmove_now=False)
                st.flip_stm()
                if val >= beta and self.alphabeta(st, alpha, beta, depth - 3, root=False, nullmove=nullmove, nullmove_now=False):
                    return val
        nullmove_now = nullmove
        if depth == 0:
            return self.qsearch(st, alpha, beta, self.qs_depth)
        moves = self._order_moves(st, raw_moves, killer)
        best = -MATE_UPPER
        mvBest = None
        move_idx = 0
        for move in moves:
            if move is None:
                continue
            is_cap = self._is_capture(st, move)
            # LMR: Late Move Reductions
            # 条件: 非根节点、深度足够、不是 TT move 首着、不是吃子、已经搜了一定数量着法
            do_lmr = (not root and depth >= self.lmr_min_depth and move_idx >= self.lmr_full_moves
                      and not is_cap and best > -MATE_UPPER)
            if best == -MATE_UPPER:
                # 第一着(PV): 全窗口全深度搜索
                st.make(move[0], move[1])
                val = -self.alphabeta(st, -beta, -alpha, depth - 1, root=False, nullmove=nullmove, nullmove_now=nullmove_now)
                st.unmake()
            else:
                if do_lmr:
                    # LMR: 先用 reduced depth + zero window 试探
                    reduced = depth - 1 - self.lmr_base_reduction
                    if reduced < 1:
                        reduced = 1
                    st.make(move[0], move[1])
                    val = -self.alphabeta(st, -alpha - 1, -alpha, reduced, root=False, nullmove=nullmove, nullmove_now=nullmove_now)
                    st.unmake()
                    # 若超过 alpha, 用完整 depth-1 重新 zero-window 搜索
                    if val > alpha:
                        st.make(move[0], move[1])
                        val = -self.alphabeta(st, -alpha - 1, -alpha, depth - 1, root=False, nullmove=nullmove, nullmove_now=nullmove_now)
                        st.unmake()
                else:
                    # 普通 zero-window 搜索
                    st.make(move[0], move[1])
                    val = -self.alphabeta(st, -alpha - 1, -alpha, depth - 1, root=False, nullmove=nullmove, nullmove_now=nullmove_now)
                    st.unmake()
                # 若在 (alpha, beta) 区间内, 全窗口重新确认
                if val > alpha and val < beta:
                    st.make(move[0], move[1])
                    val = -self.alphabeta(st, -beta, -alpha, depth - 1, root=False, nullmove=nullmove, nullmove_now=nullmove_now)
                    st.unmake()
            if val >= MATE_UPPER:
                # 走完这步后对方能否吃我方王: 等价于旧 "updated = pos.move(move).rotate()" 的语义
                st.make(move[0], move[1])
                ok = st.can_capture_king(st.stm)
                st.unmake()
                if ok:
                    mvBest = move
                    best = val
                    break
            if val > best and val > -MATE_UPPER:
                best = val
                mvBest = move
                if val > beta:
                    # beta cutoff: 更新历史启发
                    if not is_cap:
                        self.history_heur[move] = self.history_heur.get(move, 0) + depth * depth
                    break
                if val > alpha:
                    alpha = val
            move_idx += 1
        if not mvBest and moves:
            mvBest = moves[0]
        if mvBest is not None:
            if len(self.tp_move) > TABLE_SIZE:
                self.tp_move.clear()
            self.tp_move[self._mv_key(st)] = mvBest
        if best < alpha and best < 0 and depth > 0:
            # 死局判断: 每个着法走完后都 "对方能吃我王"(用 can_capture_king 替代 is_dead)
            dead_all = True
            for m in raw_moves:
                st.make(m[0], m[1])
                dead = st.can_capture_king(st.stm)
                st.unmake()
                if not dead:
                    dead_all = False
                    break
            if dead_all:
                in_check = st.can_capture_king(not st.stm)
                best = -MATE_UPPER if in_check else 0
        if len(self.tp_score) > TABLE_SIZE:
            self.tp_score.clear()
        if best >= beta:
            self.tp_score[key] = Entry(best, entry.upper)
        if best < alpha:
            self.tp_score[key] = Entry(entry.lower, best)
        return best

    def search(self, st, max_time=2.0):
        self.nodes = 0
        self.calc_average()
        if RISK_LAMBDA:
            self.calc_variance()
        # [v2] TT/历史表只在每次搜索开始时清一次 (原先每层迭代都清, 上层成果全部作废)
        self.tp_score = {}
        self.tp_move = {}
        self.history_heur = {}
        start = time.time()
        # [v2] 硬时限: 层内强制中断, 保证单着思考时间不超过预算 (检查粒度见 check_interval)
        self.deadline = start + max_time
        best_move, best_score, best_depth = None, 0, 0
        for depth in count(2):
            iter_start = time.time()
            try:
                root_val = self.alphabeta(st, -MATE_UPPER, MATE_UPPER, depth, nullmove=True, nullmove_now=True)
            except SearchTimeout:
                break
            iter_time = time.time() - iter_start
            move = self.tp_move.get(self._mv_key(st))
            if move is not None:
                best_move = move
                best_depth = depth
                # [v5.10] 上报局面绝对分 = 本层完整搜索的根值 (行棋方视角, 正=行棋方优);
                #      原先报 pos.value(move) 单步静态增量, 与搜索结论严重脱节误导日志
                best_score = root_val
            # [v2] 软时限 (预测式): 按上一层耗时的 ~2 倍估算下一层成本 (TT 持久化后
            #      实测迭代间增长仅 1.1~3 倍), 预计装不进剩余预算就不再开下一层;
            #      即使误判, 硬时限也会保证不超预算, 只是浪费掉被中断层的部分算力
            if time.time() - start + 2.0 * iter_time > max_time:
                break
            if depth >= 12:  # [v2] 深度上限: 此引擎的评估无王安全项, U子冻结+期望值折叠在
                break        #      超深搜索下会诱导牺牲王安全的贪吃着法
                             #      (5局等墙钟实测: 无上限时全部败局源于 depth 64 的贪吃),
                             #      12 层足够看见真实杀王序列
        if best_move is None:
            # [v2] 保底: 首层即超时 (think_time 极小或局面极复杂) 时不限时限搜一层浅层
            self.deadline = 0.0
            root_val = self.alphabeta(st, -MATE_UPPER, MATE_UPPER, 2, nullmove=True, nullmove_now=True)
            move = self.tp_move.get(self._mv_key(st))
            if move is not None:
                best_move, best_depth, best_score = move, 2, root_val   # 同报根值绝对分
        return best_move, best_score, best_depth

    def calc_average(self, version=0):
        numr = sum(di[version][True][key] for key in di[version][True])
        numb = sum(di[version][False][key] for key in di[version][False])
        averagecoveredr, averagecoveredb = 0, 0
        averager, averageb = {}, {}
        discount_factor = common.discount_factor

        if numr == 0:
            averagecoveredr = 0
            for i in range(51, 204):
                averager[i] = 0
        else:
            sumr = 0
            for key in di[version][True]:
                sumr += pst["1"][key] * di[version][True][key] / discount_factor
            averagecoveredr = round(sumr / numr)
            for i in range(51, 204):
                sumr = 0
                for key in di[version][True]:
                    sumr += pst[key][i] * di[version][True][key]
                averager[i] = round(sumr / numr)

        if numb == 0:
            averagecoveredb = 0
            for i in range(51, 204):
                averageb[i] = 0
        else:
            sumb = 0
            for key in di[version][False]:
                sumb += pst["1"][key.swapcase()] * di[version][False][key] / discount_factor
            averagecoveredb = round(sumb / numb)
            for i in range(51, 204):
                sumb = 0
                for key in di[version][False]:
                    sumb += pst[key.swapcase()][i] * di[version][False][key]
                averageb[i] = round(sumb / numb)

        self.average = {True: {False: averagecoveredr, True: averager}, False: {False: averagecoveredb, True: averageb}}
        average[version] = deepcopy(self.average)
        return self.average

    def calc_variance(self, version=0):
        """[v5.13 风险惩罚] 解析计算暗子池标准差 (与 calc_average 同结构、同口径:
        既算未翻开暗子的粗粒度标量 σ(对应 average[False] 的 pst["1"] 口径),
        也算已翻开 U 子的逐位置 σ(对应 average[True][i]))。多一次遍历求二阶矩,
        无需采样/无需重复搜索。用于给"用确定资源换不确定收益"的吃暗子/U 着法定价
        风险: 均值-方差效用 CE = E[X] - λ·σ(X), 比 PIMC 的蒙特卡洛重搜索更直接命中
        "方差本身"这个量, 且零额外算力开销。风险随池收窄自动衰减到 0
        (单一类型时方差为 0, 无需额外衰减系数)。"""
        numr = sum(di[version][True][key] for key in di[version][True])
        numb = sum(di[version][False][key] for key in di[version][False])
        discount_factor = common.discount_factor
        sigmacoveredr, sigmacoveredb = 0, 0
        sigmar, sigmab = {}, {}

        if numr == 0:
            for i in range(51, 204):
                sigmar[i] = 0
        else:
            meanc = self.average[True][False]
            varc = 0
            for key in di[version][True]:
                varc += di[version][True][key] * (pst["1"][key] / discount_factor - meanc) ** 2
            sigmacoveredr = (varc / numr) ** 0.5
            for i in range(51, 204):
                mean = self.average[True][True][i]
                var = 0
                for key in di[version][True]:
                    var += di[version][True][key] * (pst[key][i] - mean) ** 2
                sigmar[i] = (var / numr) ** 0.5

        if numb == 0:
            for i in range(51, 204):
                sigmab[i] = 0
        else:
            meanc = self.average[False][False]
            varc = 0
            for key in di[version][False]:
                varc += di[version][False][key] * (pst["1"][key.swapcase()] / discount_factor - meanc) ** 2
            sigmacoveredb = (varc / numb) ** 0.5
            for i in range(51, 204):
                mean = self.average[False][True][i]
                var = 0
                for key in di[version][False]:
                    var += di[version][False][key] * (pst[key.swapcase()][i] - mean) ** 2
                sigmab[i] = (var / numb) ** 0.5

        self.risk_sigma = {True: {False: sigmacoveredr, True: sigmar}, False: {False: sigmacoveredb, True: sigmab}}
        risk_sigma[version] = deepcopy(self.risk_sigma)
        return self.risk_sigma


# ---------------------------------------------------------------------------
# [v5.12 P0] 暗子池跨回合跟踪
# ---------------------------------------------------------------------------
#
# 问题: 暗子池 (每方 15 个朝下的子各是什么类型) 是跨回合状态。每个子恰好以两种
#       方式离开池子: ① 被走动 → 翻开, 类型公开; ② 仍朝下时被吃 → 类型永不公开。
#       旧实现从当前盘面反推, 把"被吃掉的明子"也算回池里, 中残局严重失真。
#
# 解法: 事件驱动地维护 rev_ever[T] = "类型 T 至今被翻开过几个", 再用一次精确
#       重标定把池缩放到 sum(池) == 盘面朝下子数。
#
#   - 翻开事件可观测 → 直接扣减对应类型 (信息完全利用)
#   - 暗着被吃不可观测 → 按比例缩放, 保持类型比例不变。这正是"无类型信息条件下
#     的贝叶斯边缘分布", 而非权宜之计: 若一个未知类型的子离开了池子, 在没有任何
#     区分信息时, 每种类型按其当前占比承担这次损失。
#
# 不变量 (硬约束, 单元测试逐局面断言): sum(pool) == 盘面朝下子数
#   由 apply() 的重标定按构造保证, 因此即便着法历史断链 (新对局/漏回合/识别错),
#   池大小也永远正确 —— 只是类型比例退化回"按剩余先验", 不会出现幽灵暗车。
#
# 实测 (回放 tools/games 全部 4157 个我方回合观测):
#   平均 L1 池误差  v5.11 = 5.748  →  P0 = 1.268  (降低 78.0%)
#   与"全知者"上界 (1.081, 受限于暗着被吃的类型不可观测) 相比, P0 弥合了 96% 的差距
#   不变量满足率 99% → 100%
#
DARK_UPPER = "DEFGHI"          # 我方朝下子 (按初始格走法的暗子字母)
DARK_LOWER = "defghi"          # 对方朝下子
LIGHT_UPPER = "RNBACP"         # 我方已翻开的明子 (不含 K: 帥/將 从不入池)
LIGHT_LOWER = "rnbacp"         # 对方已翻开的明子 (不含 k)
POOL_INIT = {"R": 2, "N": 2, "B": 2, "A": 2, "C": 2, "P": 5}   # 每方入池 15 子

# [长将] 走子后会变成 U 的字母: 暗子 (DEFGHI/defghi) 首着, 以及搜索树内已揭晓
#        但真身未知的 U/u。State.make() 会把它们写成 U, 而 gen_moves /
#        can_capture_king 一律跳过 U。揭棋里绝大多数着法都是暗子首着, 若不穷举
#        可能真身, 这类将军会被整棵漏判, 长将配额永远攒不起来。
U_AFTER_MOVE = frozenset("DEFGHIUdefghiu")
# 暗子/揭晓子可能的真身 (帥將不入池, 故不含 K); 只多判不漏判
REVEAL_TYPES = "RNBACP"


class DarkPoolTracker:
    """暗子池跨回合跟踪器 (每个 JieQiEngine 实例一个, 双方各一份池)。

    rev[True]  = {类型: 我方至今翻开过的个数}
    rev[False] = {类型: 对方至今翻开过的个数}

    调用协议 (get_best_move 内部已接好, 两种模式一致):
        observe(estr)          我方回合收到局面 → 结算上一轮双方的翻开事件
        apply(estr)            把池写入全局 di/sumall 供评估使用
        commit(move)           登记我方本回合着法 (下次 observe 时用于定位翻开)
    """

    __slots__ = ("rev", "_prev", "_my_move", "stats")

    def __init__(self):
        self.rev = {True: {}, False: {}}
        self._prev = None            # 上一次观测到的局面串
        self._my_move = None         # 在 _prev 上走出的我方着法
        # 诊断计数 (单元测试与实战日志用, 不参与评估)
        self.stats = {"own_revealed": 0, "own_lost": 0, "oppo_revealed": 0,
                      "chain_breaks": 0, "floor_bumps": 0, "resets": 0}

    # ---------------- 内部工具 ----------------
    def _bump(self, mine, t):
        """登记一次翻开事件 (类型 t 已归一为大写); 封顶在初始个数。"""
        cur = self.rev[mine].get(t, 0)
        if cur < POOL_INIT.get(t, 0):
            self.rev[mine][t] = cur + 1
            return True
        return False

    def _floor(self, estr):
        """单调下界: 翻开过的个数不可能少于当前盘面可见的同类明子数。

        这是断链兜底 —— 即使漏掉了翻开事件 (识别错/漏回合/换对局), 只要那个子
        还在盘面上就能被重新计入。注意方向: 只上调不下调, 因为明子被吃后会从盘面
        消失, 但"曾被翻开"这个事实永久成立 (正是旧实现搞错的地方)。
        """
        for mine, light in ((True, LIGHT_UPPER), (False, LIGHT_LOWER)):
            seen = {}
            for i in range(51, 204):
                ch = estr[i]
                if ch in light:
                    t = ch.upper()
                    seen[t] = seen.get(t, 0) + 1
            for t, n in seen.items():
                if self.rev[mine].get(t, 0) < n:
                    self.rev[mine][t] = min(n, POOL_INIT.get(t, n))
                    self.stats["floor_bumps"] += 1

    @staticmethod
    def _locate_oppo_move(prev, estr, src, dst):
        """从两次观测的位置级 diff 复原对方着法 (src/dst 为我方上一着)。

        枚举与 _memory_step 同构 (对方着法恰一步, 落点可能与我方格重合):
            4 格变化 (rest==2): rest 一空一占 → 空=对方起点, 占=对方落点
            3 格变化 (rest==1): 该格必为对方起点, 落点落在我方格上
        返回 (oppo_src, oppo_dst) 或 (None, None)。
        """
        if estr[dst] == ".":
            return None, None          # 我方落点空了 → diff 无法唯一归因
        rest = [i for i in range(51, 204)
                if prev[i] != estr[i] and i != src and i != dst]
        if len(rest) == 2 and estr[src] == "." \
                and (estr[rest[0]] == ".") != (estr[rest[1]] == "."):
            if estr[rest[0]] != ".":
                return rest[1], rest[0]
            return rest[0], rest[1]
        if len(rest) == 1 and estr[rest[0]] == ".":
            return rest[0], (src if estr[src] != "." else dst)
        return None, None

    # ---------------- 对外接口 ----------------
    def reset(self):
        """新对局 (或换边): 池回到初始先验。"""
        self.rev = {True: {}, False: {}}
        self._prev = None
        self._my_move = None
        self.stats["resets"] += 1

    def observe(self, estr):
        """我方回合收到新局面: 结算自上次观测以来双方的翻开事件。

        时序说明: 我方暗子走动的那一刻并不知道翻出了什么 (引擎只在自己回合看盘面),
        真身在下一次观测时才读到 —— 若期间被对方吃掉则永远读不到, 记入 own_lost,
        由 apply() 的重标定吸收 (等价于一次未知类型的离池)。
        """
        if self._prev is not None and self._my_move is not None:
            prev, (src, dst) = self._prev, self._my_move
            # ① 我方上一着若走的是暗子 → 现在读它的真身
            if prev[src] in DARK_UPPER:
                if estr[dst] in LIGHT_UPPER:
                    self._bump(True, estr[dst])
                    self.stats["own_revealed"] += 1
                else:
                    self.stats["own_lost"] += 1      # 已被吃, 真身不可知
            # ② 对方上一着若走的是暗子 → 读它的真身
            o_src, o_dst = self._locate_oppo_move(prev, estr, src, dst)
            if o_dst is None:
                self.stats["chain_breaks"] += 1
            elif prev[o_src] in DARK_LOWER and estr[o_dst] in LIGHT_LOWER:
                self._bump(False, estr[o_dst].upper())
                self.stats["oppo_revealed"] += 1
        self._floor(estr)
        self._prev = estr
        self._my_move = None

    def commit(self, move):
        """登记我方本回合选定的着法 (供下次 observe 定位翻开事件)。"""
        self._my_move = move

    def pool(self, estr, mine):
        """返回 mine 方的暗子池 {类型: 期望个数(float)}。

        保证 sum(返回值) == 盘面上该方朝下子数 (不变量, 由重标定按构造成立)。
        """
        dark = DARK_UPPER if mine else DARK_LOWER
        n_dark = 0
        for i in range(51, 204):
            if estr[i] in dark:
                n_dark += 1
        if n_dark <= 0:
            return {t: 0.0 for t in POOL_INIT}
        base = {t: POOL_INIT[t] - self.rev[mine].get(t, 0) for t in POOL_INIT}
        for t in base:
            if base[t] < 0:
                base[t] = 0
        total = sum(base.values())
        if total <= 0:
            # 矛盾兜底 (识别错导致翻开数超上限): 退回初始先验比例
            base, total = dict(POOL_INIT), sum(POOL_INIT.values())
        scale = n_dark / total
        return {t: base[t] * scale for t in POOL_INIT}

    def apply(self, estr):
        """把双方池写入全局 di/sumall (评估与 calc_average 的输入)。"""
        global r, b, di, sumall
        mine = self.pool(estr, True)
        oppo = self.pool(estr, False)
        r = dict(mine)
        b = {t.lower(): v for t, v in oppo.items()}
        di = {0: {True: dict(r), False: dict(b)}}
        sumall = {0: {True: sum(r.values()), False: sum(b.values())}}
        return di


def _row_col_to_engine_idx(row, col):
    return (row + 3) * 16 + (3 + col)


def _engine_idx_to_row_col(idx):
    return (idx // 16 - 3, idx % 16 - 3)


def _engine_idx_to_uci(idx):
    return chr(ord("a") + idx % 16 - 3) + str(12 - idx // 16)


def board_to_engine_string(board, my_side):
    """将我们的 board 格式 (10x9, board[row][col]) 转为引擎的 256-char 字符串。
    我们的棋盘始终是己方视角：row 5-9 是己方(底部)，row 0-4 是对手(顶部)。
    引擎始终用大写代表当前走子方(己方)，小写代表对手。
    暗子第一次移动才翻开，所以仍是暗子(?)的必定在初始位置。
    """
    engine_board = [" "] * 256
    for i in range(16):
        engine_board[i * 16 + 15] = "\n"

    for row in range(10):
        for col in range(9):
            piece = board[row][col]
            idx = _row_col_to_engine_idx(row, col)
            if piece == "." or not piece:
                engine_board[idx] = "."
                continue
            color = piece[0]  # 'r' 或 'b'
            is_ours = (color == my_side)
            if piece.endswith("?"):
                # 暗子：从初始布局查类型字母
                letter = INITIAL_DARK_POSITIONS.get((row, col))
                if letter is None:
                    engine_board[idx] = "."
                    continue
                engine_board[idx] = letter.upper() if is_ours else letter.lower()
            else:
                type_char = piece[1:]
                letter = TYPE_LETTER.get(type_char, ".")
                engine_board[idx] = letter.upper() if is_ours else letter.lower()

    return "".join(engine_board)


class JieQiEngine:
    def __init__(self):
        self.searcher = Searcher()
        self._cache = {}
        self._cache_count = {}
        # [v5.8] 实战模式对局记忆 (调用方不下发 check_state 时自维护长将链条):
        #   _mem_boards[i] = 第 i 个我方回合收到的局面串 (我方视角)
        #   _mem_moves[i] / _mem_my_checks[i] = 该回合选出的着法 / 是否将军
        #   _mem_oppo_checks[i] / _mem_oppo_dsts[i] = 导致 boards[i] 的对方着法是否将军 / 其落点
        self._mem_side = None
        self._mem_boards = []
        self._mem_moves = []
        self._mem_my_checks = []
        self._mem_oppo_checks = []
        self._mem_oppo_dsts = []
        # [v5.12 P0] 暗子池跨回合跟踪器。与长将记忆不同, 它在裁判模式下同样必须工作
        #            (评估正确性与谁维护长将链条无关), 故不受 live 开关约束。
        self._pool = DarkPoolTracker()
        self._pool_side = None

    def _memory_reset(self):
        self._mem_boards = []
        self._mem_moves = []
        self._mem_my_checks = []
        self._mem_oppo_checks = []
        self._mem_oppo_dsts = []

    def _memory_step(self, side, estr):
        """[v5.9] 新回合收到局面: 校验链条连续性并登记。断链(新对局/漏回合/识别错)
        则重置; 以位置级 diff 校验 (容忍暗子翻开的字母差异)。
        TODO: 对方着法的精细判断目前靠位置级 diff 推断, 后续用走子标记视觉识别
              (直接识别对方走了哪个子) 替换/完善。"""
        if self._mem_side is not None and self._mem_side != side:
            self._memory_reset()
        self._mem_side = side
        if not self._mem_boards:
            self._mem_boards.append(estr)
            self._mem_oppo_checks.append(False)
            self._mem_oppo_dsts.append(None)
            return
        if len(self._mem_boards) > len(self._mem_moves):
            return                     # 本回合已登记 (同回合重复识别)
        prev_b = self._mem_boards[-1]
        src, dst = self._mem_moves[-1]
        diff = [i for i in range(len(prev_b)) if prev_b[i] != estr[i]]
        rest = [i for i in diff if i != src and i != dst]
        # [v5.9] 3格/4格变化完整枚举 (对方着法恰一步, 落点可能在 rest 或与我方格重合):
        #   4格 (rest==2): rest 一空一占 → 空=对方起点, 占=对方落点;
        #   3格 (rest==1): 该格必为空(对方起点), 落点落在我方格上:
        #       A. estr[src] 非空 → 对方占了我方刚空的位, 落点 = src;
        #       B. estr[src] 为空 → 对方反吃我方落点,     落点 = dst。
        oppo_dst = None
        if estr[dst] != ".":
            if len(rest) == 2 and estr[src] == "." \
                    and (estr[rest[0]] == ".") != (estr[rest[1]] == "."):
                oppo_dst = rest[0] if estr[rest[0]] != "." else rest[1]
            elif len(rest) == 1 and estr[rest[0]] == ".":
                oppo_dst = src if estr[src] != "." else dst
        if oppo_dst is None:
            self._memory_reset()
            self._mem_side = side
            self._mem_boards.append(estr)
            self._mem_oppo_checks.append(False)
            self._mem_oppo_dsts.append(None)
            return
        self._mem_boards.append(estr)
        self._mem_oppo_checks.append(self._entry_in_check(estr, side, side))
        self._mem_oppo_dsts.append(oppo_dst)

    def _memory_commit(self, move, st):
        """[v5.8] 登记本回合选出的着法及其将军标记。"""
        self._mem_moves.append(move)
        self._mem_my_checks.append(self._gives_check(st, move))

    def _memory_quota_blocks(self, move, cand_chk):
        """[v5.9] 实战模式配额拦截: 从记忆末尾回溯连续将军段, 候选着将超过
        min(3×将军子数, 9) 配额 → 返回 True。将军子按轨迹认子, 被吃配额不缩。

        [修正] 将军子按轨迹认子 (与裁判 PerpCheckTracker / _quota_blocks 对齐)。
        原实现只对候选着的那个子做轨迹前移 (marker), 段内其余将军子按"当时落点
        格子"塞进 set —— 同一个子从多个格子轮流照将会被记成多个将军子, 配额由
        3 虚高到 9, 恰好放过最典型的"单车沿横线来回照将"长将形态。改为正向重放
        段内着法, 用 squares(将军子当前所在格)逐着更新, 口径与裁判侧一致。"""
        if not cand_chk:
            return False
        n = len(self._mem_moves)
        start = n
        while start > 0 and self._mem_my_checks[start - 1]:
            start -= 1                 # 连续将军段起点 (前一着非将军则停)
        squares = {}                   # 将军子当前所在格 -> True (按轨迹更新)
        retired = 0                    # 照过将但已被吃掉的子数 (配额只增不减)
        count = 0
        for k in range(start, n):
            src, dst = self._mem_moves[k]
            if src in squares:         # 某个将军子移动了 → 按轨迹更新所在格
                del squares[src]
            squares[dst] = True        # 本着照将, 纳入将军子
            count += 1
            od = self._mem_oppo_dsts[k + 1] if k + 1 < len(self._mem_oppo_dsts) else None
            if od is not None and od in squares:   # 对方吃掉了某个将军子
                del squares[od]
                retired += 1
        count += 1                     # 加上候选着本身
        total = len(squares) + retired
        if move[0] not in squares:     # 候选子是新的将军子 → 配额同步上调
            total += 1
        return count > min(3 * total, 9)

    def _entry_in_check(self, board_str, side_to_move, my_side):
        """[v5.8] 引擎串局面中, side_to_move 一方是否正被将军。

        [v5.14 适配] State 没有 .rotate(), 用临时 State.from_string 重建
        (本函数仅在 memory_step 中调用, 频率极低)。
        """
        st = State.from_string(board_str)
        if side_to_move == my_side:
            return st.can_capture_king(False)   # 对方能否吃我方王
        return st.can_capture_king(True)        # 我方能否吃对方王 = 对方被将

    def _gives_check(self, st, move):
        """[v5.8] 走 move 后对方是否被将军 (直接吃王不算将军)。

        [v5.14 适配] 用 st.make/unmake 替代旧的 pos.move().rotate()。

        [修正 1] 语义反转。can_capture_king(side) 是"side 能否吃掉对方王", 而
            make() 后 stm 已指向对方, 原写法 can_capture_king(st.stm) 算的是
            "我方走完是否被将军" —— 与函数名/用途恰好相反。实测: 照将着返回
            False, 送将着返回 True。对合法着法几乎恒为 False, 于是长将安检被
            `if cand_chk and ...` 短路跳过, 整条防线形同虚设。改为 not st.stm。
        [修正 2] 暗子首着漏判。暗子 (DEFGHI) 与已揭晓未明真身的 U 子走完后会
            被写成 U, 而 gen_moves/can_capture_king 一律跳过 U, 其将军完全不可
            见。揭棋绝大多数着法都是暗子首着, 漏判面积极大。改为对源子穷举可
            能真身 (RNBACP), 只多判不漏判 —— 误判只让引擎偏保守, 漏判则直接
            长将判负。
        """
        king_ch = "k" if st.stm else "K"
        if st.board[move[1]] == king_ch:
            return False
        src, dst = move
        p = st.board[src]
        if p in U_AFTER_MOVE:
            upper = p.isupper()
            for t in REVEAL_TYPES:
                st.board[src] = t if upper else t.lower()
                st.make(src, dst)
                chk = st.can_capture_king(not st.stm)
                st.unmake()
                st.board[src] = p            # unmake 只还原到临时真身, 需还原原字母
                if chk:
                    return True
            return False
        st.make(src, dst)                      # make 内部翻 stm, 现在 stm 指向对方
        chk = st.can_capture_king(not st.stm)  # 我方(刚走完的一方)能否吃对方王
        st.unmake()
        return chk

    def _quota_blocks(self, st, my_side, move, check_state, tracked_squares):
        """[v5.9] 配额拦截: 本着若是将军且将超过配额 min(3×将军子数, 9), 返回 True。
        tracked_squares: 已方已参与将军的棋子当前格集合 (引擎视角坐标)。"""
        if self._gives_check(st, move):
            src_rc = _engine_idx_to_row_col(move[0])
            total = len(tracked_squares) + check_state.get("retired", 0)
            if src_rc not in tracked_squares:
                total += 1                     # 新将军子加入, 配额同步上调
            if check_state["count"] + 1 > min(3 * total, 9):
                return True
        return False

    def get_best_move(self, board, my_side, think_time=2.0, pos_history=None, check_state=None):
        """返回 (uci_move, score, depth)。uci 为己方视角坐标 (row 0-9, 己方在下)
        pos_history: [v5.9 起已废弃] 保留签名仅为向后兼容, 不再使用;
        check_state: 裁判下发的已方连将计数状态, 缺省 None = 实战模式。

        [v5.14 适配] 内部从 Position 切换为 State (make/unmake 单棋盘)。
        """
        global di, sumall, average

        engine_board_str = board_to_engine_string(board, my_side)

        # [v5.12 P0] 暗子池跨回合跟踪 (取代 _update_distribution 的单帧反推)。
        #   换边/新对局 → 重置; 随后结算上一轮翻开事件并把池写入 di/sumall。
        #   两种模式 (裁判 / 实战) 都走这条路径: 池的正确性是评估正确性问题,
        #   与长将链条由谁维护无关。
        if self._pool_side is not None and self._pool_side != my_side:
            self._pool.reset()
        self._pool_side = my_side
        self._pool.observe(engine_board_str)
        self._pool.apply(engine_board_str)

        st = State.from_string(engine_board_str)

        # [v5.9] 实战模式: 调用方不下发 check_state 时, 引擎自维护长将链条 (服务端进程常驻)
        live = check_state is None
        if live:
            self._memory_step(my_side, engine_board_str)

        if st.board_str in kaijuku:
            move = kaijuku[st.board_str]
            if live:
                self._memory_commit(move, st)
            self._pool.commit(move)
            return (_engine_idx_to_uci(move[0]) + _engine_idx_to_uci(move[1]), 0, 0)

        move, score, depth = self.searcher.search(st, max_time=think_time)
        # [v5.14 修正] search() 在硬时限 SearchTimeout 触发时用裸 break 退出迭代,
        # 递归栈上的 make()/flip_stm() 不回滚, st 会停留在半路状态 (实测 undo
        # 残留、盘面漂移数格)。安检与记忆提交都依赖 st 的 board/stm/增量统计,
        # 必须在干净状态上进行: 用初始局面串重建。同时对返回着法做合法性兜底。
        st = State.from_string(engine_board_str)
        if move is not None and move not in set(st.gen_moves()):
            legal = list(st.gen_moves())
            if legal:
                move = max(legal, key=lambda m: st.value(m[0], m[1]))
                score, depth = st.value(move[0], move[1]), -1   # -1 标识脏态兜底
        # [v5.9] 长将配额安检 (硬逻辑, 搜索无否决权): 本着将是超配额
        #      (min(3×将军子数, 9)) 的连续将军 → 没收, 改走安全着中单步分最高者。
        #      裁判模式用裁判下发的 check_state; 实战模式从自维护记忆回溯。
        if move is not None and check_state:
            tracked = set()
            for r, c_ in check_state.get("squares", []):
                if my_side == "b":
                    r, c_ = 9 - r, 8 - c_
                tracked.add((r, c_))
            if self._quota_blocks(st, my_side, move, check_state, tracked):
                safe = [m for m in st.gen_moves()
                        if not self._quota_blocks(st, my_side, m, check_state, tracked)]
                if safe:
                    pick = max(safe, key=lambda m: st.value(m[0], m[1]))
                    move, score, depth = pick, st.value(pick[0], pick[1]), -2   # -2 标识安检兜底
        elif move is not None and live and len(self._mem_moves) < len(self._mem_boards):
            # [v5.9] 实战模式安检: 配额拦截 (与裁判同构)
            cand_chk = self._gives_check(st, move)
            if cand_chk and self._memory_quota_blocks(move, cand_chk):
                safe = []
                for m in st.gen_moves():
                    c = self._gives_check(st, m)
                    if c and self._memory_quota_blocks(m, c):
                        continue
                    safe.append(m)
                if safe:
                    move = max(safe, key=lambda m: st.value(m[0], m[1]))
                    score, depth = st.value(move[0], move[1]), -2   # -2 标识安检兜底
        if move is not None and live:
            self._memory_commit(move, st)
        if move is not None:
            # [v5.12 P0] 登记本回合着法, 供下次 observe 定位我方翻开事件
            self._pool.commit(move)
            uci = _engine_idx_to_uci(move[0]) + _engine_idx_to_uci(move[1])
            return uci, score, depth
        return None, 0, 0


def test():
    """测试引擎"""
    board = [["."] * 9 for _ in range(10)]
    # 红方 (己方) 在 row 9
    for col in range(9):
        board[9][col] = "r帥" if col == 4 else "r?"
    board[7][1] = "r?"; board[7][7] = "r?"
    for col in (0, 2, 4, 6, 8):
        board[6][col] = "r?"
    # 黑方 (对手) 在 row 0
    for col in range(9):
        board[0][col] = "b將" if col == 4 else "b?"
    board[2][1] = "b?"; board[2][7] = "b?"
    for col in (0, 2, 4, 6, 8):
        board[3][col] = "b?"

    engine = JieQiEngine()
    print("Board converted to engine format:")
    print(board_to_engine_string(board, "r"))
    print("\nSearching...")
    uci, score, depth = engine.get_best_move(board, "r", think_time=3.0)
    print(f"Best move: {uci}, score: {score}, depth: {depth}")


if __name__ == "__main__":
    test()
