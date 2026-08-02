import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set

BOARD_ROWS = 10
BOARD_COLS = 9


@dataclass
class Cell:
    row: int
    col: int
    x: int
    y: int
    piece: str = "."

    @property
    def is_empty(self) -> bool:
        return self.piece == "."

    @property
    def is_hidden(self) -> bool:
        return self.piece.endswith("?")

    @property
    def side(self) -> str:
        if self.is_empty:
            return ""
        return "r" if self.piece.startswith("r") else "b"


KNOWN_PIECES = {
    "r": ["帥", "仕", "相", "馬", "車", "炮", "兵"],
    "b": ["將", "士", "象", "馬", "車", "炮", "卒"],
}

POSITION_PIECE_MAP = {
    "r": [
        ["車","馬","相","仕","帥","仕","相","馬","車"],
        [  "",  "",  "",  "",  "",  "",  "",  "",  ""],
        [  "","炮",  "",  "",  "",  "",  "","炮",  ""],
        ["兵",  "","兵",  "","兵",  "","兵",  "","兵"],
        [  "",  "",  "",  "",  "",  "",  "",  "",  ""],
    ],
    "b": [
        ["車","馬","象","士","將","士","象","馬","車"],
        [  "",  "",  "",  "",  "",  "",  "",  "",  ""],
        [  "","炮",  "",  "",  "",  "",  "","炮",  ""],
        ["卒",  "","卒",  "","卒",  "","卒",  "","卒"],
        [  "",  "",  "",  "",  "",  "",  "",  "",  ""],
    ],
}

INIT_BOARD_PIECES = {
    (0,0):"r車",(0,1):"r馬",(0,2):"r相",(0,3):"r仕",(0,4):"r帥",(0,5):"r仕",(0,6):"r相",(0,7):"r馬",(0,8):"r車",
    (2,1):"r炮",(2,7):"r炮",
    (3,0):"r兵",(3,2):"r兵",(3,4):"r兵",(3,6):"r兵",(3,8):"r兵",
    (9,0):"b車",(9,1):"b馬",(9,2):"b象",(9,3):"b士",(9,4):"b將",(9,5):"b士",(9,6):"b象",(9,7):"b馬",(9,8):"b車",
    (7,1):"b炮",(7,7):"b炮",
    (6,0):"b卒",(6,2):"b卒",(6,4):"b卒",(6,6):"b卒",(6,8):"b卒",
}


class GameState:
    """揭棋游戏状态管理器 — 追踪暗棋信息 + 候选集"""

    def __init__(self):
        self.board: List[List[str]] = [["."] * BOARD_COLS for _ in range(BOARD_ROWS)]
        self.hidden_candidates: dict = {}
        self.revealed: Set[Tuple[int, int]] = set()
        self.turn = "r"
        self.move_history: List[str] = []
        self._init_hidden()

    def _init_hidden(self):
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                key = (r, c)
                if key in INIT_BOARD_PIECES:
                    full = INIT_BOARD_PIECES[key]
                    side = full[0]
                    self.board[r][c] = f"{side}?"
                    self.hidden_candidates[key] = self._init_candidates(side, r, c)
                else:
                    self.board[r][c] = "."
                    self.hidden_candidates[key] = []

    def _init_candidates(self, side: str, row: int, col: int) -> List[str]:
        side_pieces = KNOWN_PIECES[side]
        local_row = row if side == "r" else 9 - row
        if local_row < 0 or local_row >= 5:
            return side_pieces.copy()
        expected = POSITION_PIECE_MAP[side][local_row][col]
        if expected:
            return [expected]
        return side_pieces.copy()

    def update_from_detection(self, detected: List[List[Cell]]):
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                cell = detected[r][c]
                old = self.board[r][c]
                new = cell.piece
                if old != new:
                    self._handle_piece_change(r, c, old, new)

    def _handle_piece_change(self, r: int, c: int, old: str, new: str):
        key = (r, c)
        if old.startswith("r?") or old.startswith("b?"):
            self.board[r][c] = new
            self.revealed.add(key)
            if not new.endswith("?") and len(new) >= 2:
                self._narrow_candidates_from_reveal(r, c, new)
        elif new == ".":
            if key in self.hidden_candidates:
                del self.hidden_candidates[key]
            if key in self.revealed:
                self.revealed.discard(key)
            self.board[r][c] = "."
        else:
            self.board[r][c] = new

    def _narrow_candidates_from_reveal(self, r: int, c: int, revealed_piece: str):
        if len(revealed_piece) < 2:
            return
        side = revealed_piece[0]
        piece_name = revealed_piece[1]
        for k, cands in self.hidden_candidates.items():
            if k[1] == c or k[0] == r:
                continue
            if piece_name in cands:
                cands_copy = cands.copy()
                cands_copy.remove(piece_name)
                self._enforce_candidates(k, cands_copy, side)

    def _enforce_candidates(self, key, candidates, side):
        old = self.hidden_candidates.get(key, [])
        new = [x for x in candidates if x in old]
        if not new and old:
            new = old
        self.hidden_candidates[key] = new

    def apply_move(self, from_r: int, from_c: int, to_r: int, to_c: int, reveal: str = ""):
        from_key = (from_r, from_c)
        to_key = (to_r, to_c)
        moving = self.board[from_r][from_c]
        if moving.endswith("?") and reveal:
            moving = f"{moving[0]}{reveal}"
            self.revealed.add(from_key)
            self._narrow_candidates_from_reveal(from_r, from_c, moving)
        captured = self.board[to_r][to_c]
        self.board[to_r][to_c] = moving
        self.board[from_r][from_c] = "."
        if to_key in self.hidden_candidates:
            del self.hidden_candidates[to_key]
        ucimove = self._to_uci(from_r, from_c, to_r, to_c)
        self.move_history.append(ucimove)
        self.turn = "b" if self.turn == "r" else "r"
        return captured

    def _to_uci(self, fr: int, fc: int, tr: int, tc: int) -> str:
        return f"{chr(ord('a') + fc)}{9 - fr}{chr(ord('a') + tc)}{9 - tr}"

    def is_game_over(self) -> bool:
        has_r_king = any("帥" in self.board[r][c] or self.board[r][c] == "r?"
                         for r in range(10) for c in range(9))
        has_b_king = any("將" in self.board[r][c] or self.board[r][c] == "b?"
                         for r in range(10) for c in range(9))
        return not has_r_king or not has_b_king

    def to_fen(self) -> str:
        rows = []
        for r in range(BOARD_ROWS):
            row_str = ""
            empty_cnt = 0
            for c in range(BOARD_COLS):
                p = self.board[r][c]
                if p == ".":
                    empty_cnt += 1
                else:
                    if empty_cnt > 0:
                        row_str += str(empty_cnt)
                        empty_cnt = 0
                    row_str += self._piece_to_fen(p)
            if empty_cnt > 0:
                row_str += str(empty_cnt)
            rows.append(row_str)
        rows.reverse()
        fen = "/".join(rows)
        return f"{fen} {'w' if self.turn == 'r' else 'b'} - - 0 1"

    def to_masked_fen(self) -> str:
        assignment = {}
        used = {"r": {}, "b": {}}
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                p = self.board[r][c]
                if not p.endswith("?"):
                    if p not in (".",):
                        side = p[0]
                        name = p[1:]
                        used[side][name] = used[side].get(name, 0) + 1
                    continue
                side = p[0]
                key = (r, c)
                cands = self.hidden_candidates.get(key, KNOWN_PIECES[side])
                available = []
                for cand in cands:
                    max_cnt = 2 if cand in ("車", "馬", "炮") else \
                              5 if cand in ("兵", "卒") else \
                              2 if cand in ("仕", "士", "相", "象") else 1
                    if used[side].get(cand, 0) < max_cnt:
                        available.append(cand)
                if not available:
                    available = cands
                chosen = random.choice(available)
                used[side][chosen] = used[side].get(chosen, 0) + 1
                assignment[(r, c)] = f"{side}{chosen}"

        rows = []
        for r in range(BOARD_ROWS):
            row_str = ""
            empty_cnt = 0
            for c in range(BOARD_COLS):
                p = assignment.get((r, c), self.board[r][c])
                if p == ".":
                    empty_cnt += 1
                else:
                    if empty_cnt > 0:
                        row_str += str(empty_cnt)
                        empty_cnt = 0
                    row_str += self._piece_to_fen(p)
            if empty_cnt > 0:
                row_str += str(empty_cnt)
            rows.append(row_str)
        rows.reverse()
        fen = "/".join(rows)
        return f"{fen} {'w' if self.turn == 'r' else 'b'} - - 0 1"

    def _piece_to_fen(self, piece: str) -> str:
        if piece.endswith("?"):
            return piece
        mapping = {
            "r帥": "K", "r仕": "A", "r相": "B", "r馬": "N", "r車": "R", "r炮": "C", "r兵": "P",
            "b將": "k", "b士": "a", "b象": "b", "b馬": "n", "b車": "r", "b炮": "c", "b卒": "p",
        }
        return mapping.get(piece, piece)
