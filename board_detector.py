"""
纯颜色识别 - 不依赖模板，根据棋子颜色判定红方/黑方/暗棋
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import os

PIECE_CHARS_RED = "帅仕相马车炮兵"
PIECE_CHARS_BLACK = "将士象马车砲卒"

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


class BoardDetector:
    """棋盘检测器 - 纯颜色识别方案"""

    def __init__(self, assets_dir: str = None, offset_x: int = 0, offset_y: int = 0):
        self.assets_dir = assets_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        self.offset_x = offset_x
        self.offset_y = offset_y

    def find_board_corners(self, screenshot: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        h, w = screenshot.shape[:2]
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        h_proj = np.sum(binary, axis=1) / 255
        v_proj = np.sum(binary, axis=0) / 255

        h_thresh = np.max(h_proj) * 0.15
        v_thresh = np.max(v_proj) * 0.15
        h_lines = np.where(h_proj > h_thresh)[0]
        v_lines = np.where(v_proj > v_thresh)[0]

        if len(h_lines) < 20 or len(v_lines) < 18:
            margin = int(min(w, h) * 0.02)
            return (margin, margin, w - 2 * margin, h - 2 * margin)

        h_groups = self._cluster_1d(h_lines, gap=int(h * 0.01))
        v_groups = self._cluster_1d(v_lines, gap=int(w * 0.01))

        if len(h_groups) < 6 or len(v_groups) < 5:
            margin = int(min(w, h) * 0.02)
            return (margin, margin, w - 2 * margin, h - 2 * margin)

        h_medians = [int(np.median(g)) for g in h_groups]
        v_medians = [int(np.median(g)) for g in v_groups]
        h_medians.sort()
        v_medians.sort()

        top, bottom = h_medians[0], h_medians[-1]
        left, right = v_medians[0], v_medians[-1]
        top = max(top - 20, 0)
        bottom = min(bottom + 20, h)
        left = max(left - 20, 0)
        right = min(right + 20, w)

        return (left, top, right - left, bottom - top)

    def _cluster_1d(self, values: np.ndarray, gap: int) -> List[List[int]]:
        if len(values) == 0:
            return []
        values = sorted(values)
        groups = [[values[0]]]
        for v in values[1:]:
            if v - groups[-1][-1] <= gap:
                groups[-1].append(v)
            else:
                groups.append([v])
        return groups

    def detect_board(self, screenshot: np.ndarray,
                     board_rect: Optional[Tuple[int, int, int, int]] = None) -> List[List[Cell]]:
        if board_rect is None:
            board_rect = self.find_board_corners(screenshot)
        if board_rect is None:
            return [[]]
        bx, by, bw, bh = board_rect
        bx += self.offset_x
        by += self.offset_y
        midline = by + bh // 2
        cell_w = bw // BOARD_COLS
        cell_h = bh // BOARD_ROWS

        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        circles = self._find_circles(screenshot, (bx, by, bw, bh), cell_w, cell_h)

        board = []
        for row in range(BOARD_ROWS):
            row_cells = []
            for col in range(BOARD_COLS):
                cx = bx + int((col + 0.5) * cell_w)
                cy = by + int((row + 0.5) * cell_h)
                piece = "."

                has_circle = False
                for (px, py, pr) in circles:
                    dist = np.sqrt(float(px - cx) ** 2 + float(py - cy) ** 2)
                    if dist < min(cell_w, cell_h) * 0.4:
                        has_circle = True
                        break

                if not has_circle:
                    r = min(cell_w, cell_h) // 3
                    x1, y1 = max(cx - r, 0), max(cy - r, 0)
                    x2, y2 = min(cx + r, gray.shape[1]), min(cy + r, gray.shape[0])
                    if x2 > x1 and y2 > y1:
                        roi = gray[y1:y2, x1:x2]
                        mean_val = np.mean(roi)
                        std_val = np.std(roi)
                        if mean_val < 205 and std_val < 50:
                            has_circle = True

                if has_circle:
                    side = "t" if cy < midline else "d"
                    piece = f"{side}?"

                row_cells.append(Cell(row=row, col=col, x=cx, y=cy, piece=piece))
            board.append(row_cells)
        return board

    def _find_circles(self, img: np.ndarray, rect: Tuple, cw: int, ch: int) -> list:
        bx, by, bw, bh = rect
        board_roi = img[by:by+bh, bx:bx+bw]
        gray = cv2.cvtColor(board_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 20, 60)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        min_area = (min(cw, ch) * 0.4) ** 2 * 3.14 * 0.5
        max_area = (min(cw, ch) * 0.6) ** 2 * 3.14 * 1.5
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            (px, py), pr = cv2.minEnclosingCircle(cnt)
            pr = int(pr)
            if pr == 0:
                continue
            fill_ratio = area / (pr * pr * np.pi)
            if 0.4 < fill_ratio < 1.3:
                gx, gy = int(bx + px), int(by + py)
                result.append((gx, gy, pr))
        return result

    def _classify_cell(self, row: int, col: int, pieces: list,
                       cw: int, ch: int, midline: int, bx: int, by: int,
                       hsv: np.ndarray, gray: np.ndarray) -> str:
        cx = bx + int((col + 0.5) * cw)
        cy = by + int((row + 0.5) * ch)

        nearest_dist = float('inf')
        for (px, py, pr) in pieces:
            dist = np.sqrt(float(px - cx) ** 2 + float(py - cy) ** 2)
            if dist < min(cw, ch) * 0.4 and dist < nearest_dist:
                nearest_dist = dist

        if nearest_dist == float('inf'):
            return "."

        side = "t" if cy < midline else "d"
        return f"{side}?"
