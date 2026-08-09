"""
揭棋引擎 - 基于 miaosiSari/Jieqi 的纯算法引擎 (PST 评估，无需 NNUE)
"""
import re, os, time, json
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

# 方向常量
N, E, S, W = -16, 1, 16, -1

# 走法方向
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

A0, I0, A9, I9 = 12 * 16 + 3, 12 * 16 + 11, 3 * 16 + 3, 3 * 16 + 11
MATE_LOWER = PIECE_VALUE["K"] - (2 * PIECE_VALUE["R"] + 2 * PIECE_VALUE["N"] + 2 * PIECE_VALUE["B"] + 2 * PIECE_VALUE["A"] + 2 * PIECE_VALUE["C"] + 5 * PIECE_VALUE["P"])
MATE_UPPER = PIECE_VALUE["K"] + (2 * PIECE_VALUE["R"] + 2 * PIECE_VALUE["N"] + 2 * PIECE_VALUE["B"] + 2 * PIECE_VALUE["A"] + 2 * PIECE_VALUE["C"] + 5 * PIECE_VALUE["P"])
TABLE_SIZE = 1e7
EVAL_ROUGHNESS = 13

put = lambda board, i, p: board[:i] + p + board[i + 1:]


class Position(namedtuple("Position", "board score turn version")):
    def set(self):
        self.che = 0
        self.che_opponent = 0
        self.zu = 0
        self.covered = 0
        self.covered_opponent = 0
        self.endline = 0
        self.score_rough = 0
        self.kongtoupao = 0
        self.kongtoupao_opponent = 0
        self.kongtou_score = 0
        self.kongtou_score_opponent = 0

        for i in range(51, 204):
            if i >> 4 == 3:
                if self.board[i] in "defgrnc":
                    self.endline += 1
            p = self.board[i]
            if p in "RNBAKCP":
                self.score_rough += pst[p][i]
            elif p in "DEFGHI":
                self.covered += 1
            elif p in "U":
                self.score_rough += average[self.version][self.turn][True][i]
                self.covered += 1
            elif p in "rnbakcp":
                self.score_rough -= pst[p.upper()][254 - i]
            elif p in "defghi":
                self.covered_opponent += 1
            elif p in "u":
                self.score_rough -= average[self.version][not self.turn][True][254 - i]
                self.covered_opponent += 1
            if p == "R":
                self.che += 1
            if p == "r":
                self.che_opponent += 1
            if p == "P":
                self.zu += 1
            if p == "C" and i & 15 == 7:
                self.check_kongtoupao(i, True)
            if p == "c" and i & 15 == 7:
                self.check_kongtoupao(i, False)

        if (self.kongtoupao > 0 and self.kongtoupao_opponent <= 0) or (self.kongtoupao > self.kongtoupao_opponent > 0):
            if (self.che >= self.che_opponent and self.che > 0) or self.kongtoupao >= 3:
                self.kongtou_score += 100
            else:
                self.kongtou_score += 70
        elif (self.kongtoupao <= 0 and self.kongtoupao_opponent > 0) or (self.kongtoupao_opponent > self.kongtoupao > 0):
            if (self.che_opponent >= self.che and self.che_opponent > 0) or self.kongtoupao_opponent >= 3:
                self.kongtou_score_opponent += 100
            else:
                self.kongtou_score_opponent += 70
        return self

    def check_kongtoupao(self, pos, t):
        cannon = "C" if t else "c"
        king = "k" if t else "K"
        if t:
            if self.kongtoupao:
                return
            for scanpos in range(pos - 16, 51, -16):
                if self.board[scanpos] == cannon:
                    continue
                elif self.board[scanpos] != ".":
                    if self.board[scanpos] == king:
                        self.kongtoupao += 1
                    return
                else:
                    self.kongtoupao += 1
        else:
            if self.kongtoupao_opponent:
                return
            for scanpos in range(pos + 16, 204, 16):
                if self.board[scanpos] == cannon:
                    continue
                elif self.board[scanpos] != ".":
                    if self.board[scanpos] == king:
                        self.kongtoupao_opponent += 1
                    return
                else:
                    self.kongtoupao_opponent += 1

    def gen_moves(self):
        for i in range(51, 204):
            p = self.board[i]
            if not p.isupper() or p == "U":
                continue
            if p == "K":
                for scanpos in range(i - 16, A9, -16):
                    if self.board[scanpos] == "k":
                        yield (i, scanpos)
                    elif self.board[scanpos] != ".":
                        break
            if p in ("C", "H"):
                for d in DIRECTIONS[p]:
                    cfoot = 0
                    for j in count(i + d, d):
                        q = self.board[j]
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
                    q = self.board[j]
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
                            if self.board[i + (1 if n_diff_x == 2 else -1)] != ".":
                                break
                        else:
                            if j > i and self.board[i + 16] != ".":
                                break
                            elif j < i and self.board[i - 16] != ".":
                                break
                    elif p in ("B", "F") and self.board[i + d // 2] != ".":
                        break
                    yield (i, j)
                    if p in "PNBAKIEFG" or q.islower():
                        break

    def rooted(self):
        rooted_chesses = set()
        for i in range(51, 204):
            p = self.board[i]
            if not p.isupper() or p == "U":
                continue
            if p in ("C", "H"):
                for d in DIRECTIONS[p]:
                    cfoot = 0
                    for j in count(i + d, d):
                        q = self.board[j]
                        if q.isspace():
                            break
                        if cfoot == 0 and q == ".":
                            continue
                        elif cfoot == 0 and q != ".":
                            cfoot += 1
                        elif cfoot == 1 and q.islower():
                            break
                        elif cfoot == 1 and q.isupper():
                            rooted_chesses.add(j); break
                continue
            for d in DIRECTIONS[p]:
                for j in count(i + d, d):
                    q = self.board[j]
                    if q.isspace() or q.islower():
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
                            if self.board[i + (1 if n_diff_x == 2 else -1)] != ".":
                                break
                        else:
                            if j > i and self.board[i + 16] != ".":
                                break
                            elif j < i and self.board[i - 16] != ".":
                                break
                    elif p in ("B", "F") and self.board[i + d // 2] != ".":
                        break
                    if q.isupper():
                        rooted_chesses.add(j); break
                    if p in "PNBAKIEFG":
                        break
        return rooted_chesses

    def rotate(self):
        p = Position(self.board[-2::-1].swapcase() + " ", -self.score, not self.turn, self.version)
        p.set()
        return p

    @staticmethod
    def rotate_new(board, score, turn, version):
        p = Position(board[-2::-1].swapcase() + " ", -score, not turn, version)
        p.set()
        return p

    def move(self, move):
        i, j = move
        movevalue = self.value(move)
        score = self.score + movevalue if movevalue < MATE_UPPER else MATE_UPPER
        if self.board[i] in "RNBAKCP":
            board = put(self.board, j, self.board[i])
        else:
            board = put(self.board, j, "U")
        board = put(board, i, ".")
        return Position.rotate_new(board, score, self.turn, self.version)

    def value(self, move):
        i, j = move
        p, q = self.board[i], self.board[j].upper()
        possible_che = 0 if sumall[self.version][self.turn] == 0 else self.covered * di[self.version][self.turn]["R" if self.turn else "r"] / sumall[self.version][self.turn]
        possible_che_opponent = 0 if sumall[self.version][not self.turn] == 0 else self.covered_opponent * di[self.version][not self.turn]["r" if self.turn else "R"] / sumall[self.version][not self.turn]
        if q == "K":
            return MATE_UPPER
        if p in "RNBAKCP":
            score = pst[p][j] - pst[p][i]
            if p == "C":
                if (i >> 4) != 3 and (j >> 4) == 3 and self.endline <= 2:
                    if (j == 51 or j == 52) and self.board[53] == "f" and self.board[54] == "g":
                        pass
                    elif (j == 59 or j == 58) and self.board[57] == "f" and self.board[56] == "g":
                        pass
                    else:
                        score -= 55 if self.endline == 0 else 30
                if (i >> 4) == 3 and (j >> 4) != 3 and self.endline <= 2:
                    if (i == 51 or i == 52) and self.board[53] == "f" and self.board[54] == "g":
                        pass
                    elif (i == 59 or i == 58) and self.board[57] == "f" and self.board[56] == "g":
                        pass
                    else:
                        score += 55 if self.endline == 0 else 30
            elif p == "R":
                if self.board[51] not in "dr" and self.board[54] != "a" and self.board[71] != "a" and (self.board[71] == "p" or self.board[87] != "n"):
                    if j & 15 == 6 and i & 15 != 6:
                        score += 30
                    if j & 15 != 6 and i & 15 == 6:
                        score -= 30
                if self.board[59] not in "dr" and self.board[56] != "a" and self.board[71] != "a" and (self.board[71] == "p" or self.board[87] != "n"):
                    if j & 15 == 8 and i & 15 != 8:
                        score += 30
                    if j & 15 != 8 and i & 15 == 8:
                        score -= 30
                if (i >> 4) == 3 and (j >> 4) != 3 and (self.endline <= 1 or self.score_rough < -150):
                    score += 40 if self.score_rough < -150 else 30
                if (i >> 4) != 3 and (j >> 4) == 3 and (self.endline <= 1 or self.score_rough < -150):
                    score -= 40 if self.score_rough < -150 else 30
        else:
            score = average[self.version][self.turn][True][j] - average[self.version][self.turn][False] + 20
            if p == "D":
                minus = 30 * (possible_che_opponent / 2 + self.che_opponent)
                score -= minus
                if self.score_rough < -150:
                    score -= minus // 2
            elif p == "I":
                if self.board[i - 32] in "rp":
                    score -= average[self.version][self.turn][False] // 2
                elif self.board[i - 32] in "nc":
                    score += 30
                elif self.board[i - 48] == "i":
                    score += 30
                else:
                    score += 20

        if q.isupper():
            k = 254 - j
            if q in "RNBAKCP":
                score += pst[q][k]
                if q == "P" and self.board[j + 32] == "I":
                    score += 30
            else:
                if q != "U":
                    score += average[self.version][not self.turn][False]
                    if q == "I":
                        score += 10
                else:
                    score += average[self.version][not self.turn][True][k]
                    if j >> 4 == 7 and j & 1 == 1:
                        score += 10
                if q == "D":
                    addition = 30 * (possible_che / 2 + self.che)
                    score += addition
                    if self.score_rough > 150:
                        score += addition // 2
        return score


Entry = namedtuple("Entry", "lower upper")


class Searcher:
    def __init__(self):
        self.tp_score = {}
        self.tp_move = {}
        self.history = set()
        self.nodes = 0

    def quiescence(self, pos, moves, oppo):
        score = 0
        maxscore = 0
        oppo_rooted_set = oppo.rooted()
        oppo_rooted_set = set(map(lambda x: 254 - x, oppo_rooted_set))
        argmax = None
        for move in moves:
            p = pos.board[move[0]]
            q = pos.board[move[1]]
            if q == ".":
                continue
            if p == "D":
                if q == "r":
                    if 240 > maxscore:
                        argmax = move
                        maxscore = 240
                else:
                    continue
            j = move[1]
            k = 254 - j
            if q in "rcnabpk":
                score += pst[q.upper()][k]
            elif q in "defghi":
                score += average[oppo.version][oppo.turn][False]
            elif q == "u":
                score += average[oppo.version][oppo.turn][True][k]
            if move[1] in oppo_rooted_set:
                if p in "RCNABPK":
                    score -= pst[p][j]
                if p in "EFGHI":
                    score -= average[pos.version][pos.turn][False]
            if score > maxscore:
                argmax = move
                maxscore = score
        if argmax and pos.board[argmax[1]] == "c" and argmax[1] & 15 == 7 and pos.kongtou_score_opponent > 0 and pos.kongtoupao_opponent > 0:
            pos.kongtou_score_opponent = 0
        return maxscore, argmax

    def alphabeta(self, pos, alpha, beta, depth, root=True, nullmove=False, nullmove_now=False):
        global debug_var
        oppo = pos.rotate()
        if root:
            self.tp_score = {}
            self.tp_move = {}
        self.nodes += 1
        depth = max(depth, 0)
        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER
        moves = sorted(pos.gen_moves(), key=pos.value, reverse=True)
        killer = self.tp_move.get(pos)
        for move in [killer] + moves:
            if (move is not None) and pos.board[move[1]] == "k":
                self.tp_move[pos] = move
                return MATE_UPPER
        entry = self.tp_score.get((pos, depth, root), Entry(-MATE_UPPER, MATE_UPPER))
        if entry.lower >= beta and (not root or self.tp_move.get(pos) is not None):
            return entry.lower
        if entry.upper < alpha:
            return entry.upper
        if nullmove_now and depth > 3 and not root and any(c in pos.board for c in "RNCI"):
            if all(oppo.board[m[1]] != "k" for m in oppo.gen_moves()):
                val = -self.alphabeta(pos.rotate(), -beta, 1 - beta, depth - 3, root=False, nullmove=nullmove, nullmove_now=False)
                if val >= beta and self.alphabeta(pos, alpha, beta, depth - 3, root=False, nullmove=nullmove, nullmove_now=False):
                    return val
        nullmove_now = nullmove
        if depth == 0:
            score = self.quiescence(pos, moves, oppo)
            return pos.score + pos.kongtou_score - pos.kongtou_score_opponent + score[0]
        best = -MATE_UPPER
        mvBest = None
        for move in [killer] + moves:
            if (move is not None) and (depth > 0):
                if best == -MATE_UPPER:
                    val = -self.alphabeta(pos.move(move), -beta, -alpha, depth - 1, root=False, nullmove=nullmove, nullmove_now=nullmove_now)
                else:
                    val = -self.alphabeta(pos.move(move), -alpha - 1, -alpha, depth - 1, root=False, nullmove=nullmove, nullmove_now=nullmove_now)
                    if val > alpha and val < beta:
                        val = -self.alphabeta(pos.move(move), -beta, -alpha, depth - 1, root=False, nullmove=nullmove, nullmove_now=nullmove_now)
                if val >= MATE_UPPER:
                    updated = pos.move(move).rotate()
                    if any(updated.board[m[1]] == "k" for m in updated.gen_moves()):
                        mvBest = move
                        best = val
                        break
                if val > best and val > -MATE_UPPER:
                    best = val
                    mvBest = move
                    if val > beta:
                        break
                    if val > alpha:
                        alpha = val
        if not mvBest and moves:
            mvBest = moves[0]
        if mvBest is not None:
            if len(self.tp_move) > TABLE_SIZE:
                self.tp_move.clear()
            self.tp_move[pos] = mvBest
        if best < alpha and best < 0 and depth > 0:
            is_dead = lambda pos: any(pos.value(m) >= MATE_LOWER for m in pos.gen_moves())
            if all(is_dead(pos.move(m)) for m in pos.gen_moves()):
                in_check = is_dead(pos.rotate())
                best = -MATE_UPPER if in_check else 0
        if len(self.tp_score) > TABLE_SIZE:
            self.tp_score.clear()
        if best >= beta:
            self.tp_score[pos, depth, root] = Entry(best, entry.upper)
        if best < alpha:
            self.tp_score[pos, depth, root] = Entry(entry.lower, best)
        return best

    def search(self, pos, max_time=2.0):
        self.nodes = 0
        self.calc_average()
        pos.set()
        start = time.time()
        best_move, best_score, best_depth = None, 0, 0
        for depth in range(5, 8):
            self.alphabeta(pos, -MATE_UPPER, MATE_UPPER, depth, nullmove=True, nullmove_now=True)
            move = self.tp_move.get(pos)
            if move is not None:
                best_move = move
                best_depth = depth
                # 重新评估该着法得到分数
                best_score = pos.value(move)
            if time.time() - start > max_time:
                break
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


# 全局状态
r = {"R": 2, "N": 2, "B": 2, "A": 2, "C": 2, "P": 5}
b = {"r": 2, "n": 2, "b": 2, "a": 2, "c": 2, "p": 5}
di = {0: {True: deepcopy(r), False: deepcopy(b)}}
sumall = {0: {True: sum(di[0][True][key] for key in di[0][True]), False: sum(di[0][False][key] for key in di[0][False])}}
pst = deepcopy(common.pst)
average = {0: {}}
kaijuku = deepcopy(library.kaijuku)


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


def _update_distribution(engine_board_str):
    """根据引擎棋盘字符串更新 di 分布"""
    global r, b, di, sumall
    r = {"R": 2, "N": 2, "B": 2, "A": 2, "C": 2, "P": 5}
    b = {"r": 2, "n": 2, "b": 2, "a": 2, "c": 2, "p": 5}

    for i in range(51, 204):
        p = engine_board_str[i]
        if p in "RNBAKCP":
            r[p] = max(0, r.get(p, 0) - 1)
        elif p in "rnbakcp":
            k = p.upper()
            b[p] = max(0, b.get(p, 0) - 1)
        elif p == "U":
            pass
        elif p == "u":
            pass

    di = {0: {True: deepcopy(r), False: deepcopy(b)}}
    sumall = {0: {True: sum(di[0][True][key] for key in di[0][True]), False: sum(di[0][False][key] for key in di[0][False])}}


class JieQiEngine:
    def __init__(self):
        self.searcher = Searcher()
        self._cache = {}
        self._cache_count = {}

    def get_best_move(self, board, my_side, think_time=2.0):
        """返回 (uci_move, score, depth)。uci 为己方视角坐标 (row 0-9, 己方在下)"""
        global di, sumall, average

        engine_board_str = board_to_engine_string(board, my_side)
        _update_distribution(engine_board_str)

        pos = Position(engine_board_str, 0, True, 0).set()

        if pos.board in kaijuku:
            move = kaijuku[pos.board]
            return (_engine_idx_to_uci(move[0]) + _engine_idx_to_uci(move[1]), 0, 0)

        move, score, depth = self.searcher.search(pos, max_time=think_time)
        if move is not None:
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