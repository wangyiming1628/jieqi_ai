"""
棋盘识别器 - 封装 YOLO + PaddleOCR 识别逻辑
"""
import os, cv2, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_detector import YOLOChessDetector
from paddleocr import PaddleOCR

CHESS = "帥仕相馬車炮兵將士象卒"
RED_ONLY = set("帥仕相兵")
BLACK_ONLY = set("將士象卒")

class BoardRecognizer:
    def __init__(self):
        self.detector = YOLOChessDetector()
        self.ocr = PaddleOCR(lang="ch", use_angle_cls=False, show_log=False)
        self.rec = self.ocr.text_recognizer
        self.char_list = self.rec.postprocess_op.character
        self.max_wh = self.rec.rec_image_shape[2] / self.rec.rec_image_shape[1]
        self._cache = {}  # (col, row) -> (cx, cy, char, conf, side)

    def _raw_ocr(self, crop_bgr):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        norm = self.rec.resize_norm_img(rgb, self.max_wh)
        norm_batch = np.expand_dims(norm, 0).copy()
        input_names = self.rec.predictor.get_input_names()
        handle = self.rec.predictor.get_input_handle(input_names[0])
        handle.copy_from_cpu(norm_batch)
        self.rec.predictor.run()
        output_handles = [self.rec.predictor.get_output_handle(n) for n in self.rec.predictor.get_output_names()]
        probs = np.array(output_handles[0].copy_to_cpu())
        best_ch, best_p = "", 0.0
        for pos in range(probs.shape[1]):
            for ch in CHESS:
                idx = self.char_list.index(ch) if ch in self.char_list else -1
                if idx >= 0:
                    p = probs[0, pos, idx]
                    if p > best_p:
                        best_p = p
                        best_ch = ch
        return best_ch, best_p

    def _detect_side(self, crop_img, ocr_char):
        if ocr_char in RED_ONLY:
            return "r"
        if ocr_char in BLACK_ONLY:
            return "b"
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        mid = max(crop_img.shape[0] // 4, 1)
        core = hsv[mid:-mid, mid:-mid] if crop_img.shape[0] > mid * 2 else hsv
        r1 = cv2.inRange(core, (0, 40, 40), (15, 255, 255))
        r2 = cv2.inRange(core, (160, 40, 40), (180, 255, 255))
        red_ratio = (np.sum(r1 > 0) + np.sum(r2 > 0)) / max(core.size // 3, 1)
        dark = cv2.inRange(core, (0, 0, 0), (180, 255, 80))
        dark_ratio = np.sum(dark > 0) / max(core.size // 3, 1)
        if red_ratio > 0.05:
            return "r"
        if dark_ratio > 0.2:
            return "b"
        return None

    def detect(self, image: np.ndarray):
        """返回 10x9 棋盘: '.'=空, 'r?'=暗红, 'b?'=暗黑, 'r帥'=红帥等"""
        h, w = image.shape[:2]
        mid_y = h / 2
        dets = self.detector.detect(image)
        
        board = [["."] * 9 for _ in range(10)]
        
        for i, d in enumerate(dets):
            cx, cy = d["x"], d["y"]
            col = int(cx / w * 9)
            row = int(cy / h * 10)
            row, col = max(0, min(9, row)), max(0, min(8, col))
            key = (row, col)

            # Check cache
            if key in self._cache:
                pcx, pcy, pch, pconf, pside, _ = self._cache[key]
                if abs(cx - pcx) < 10 and abs(cy - pcy) < 10:
                    board[row][col] = pside + pch if pch else pside + "?"
                    continue
            x1 = max(0, int(cx - d["w"] / 2))
            y1 = max(0, int(cy - d["h"] / 2))
            x2 = min(w, int(cx + d["w"] / 2))
            y2 = min(h, int(cy + d["h"] / 2))
            full = image[y1:y2, x1:x2]
            if full.size == 0: continue

            # HoughCircles
            gray = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)
            blurred = cv2.medianBlur(gray, 5)
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.0, minDist=80,
                                        param1=80, param2=30, minRadius=25, maxRadius=55)
            if circles is not None:
                pcx, pcy = map(int, circles[0][0][:2])
            else:
                pcx, pcy = full.shape[1] // 2, full.shape[0] // 2

            S = int(min(d["w"], d["h"]) * 0.58)
            cx2 = max(0, int(x1 + pcx - S // 2))
            cy2 = max(0, int(y1 + pcy - S // 2))
            cx3 = min(w, int(x1 + pcx + S // 2))
            cy3 = min(h, int(y1 + pcy + S // 2))
            crop = image[cy2:cy3, cx2:cx3]

            ch, conf = self._raw_ocr(crop)

            # Check if piece is face-down (uniform texture -> low std)
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size > 0 else None
            # Check if piece is face-down: uniform -> low std on ORIGINAL YOLO crop
            full_gray = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)
            is_hidden = np.std(full_gray) < 40

            side = "r" if cy < mid_y else "b"
            if ch and conf > 0.005 and not is_hidden:
                detected = self._detect_side(crop, ch)
                if detected:
                    side = detected
                board[row][col] = side + ch
            else:
                board[row][col] = "?"

            self._cache[key] = (cx, cy, ch, conf, side, board[row][col])

        # Determine hidden piece sides from king position
        bottom_king = ""
        for r in range(5, 10):
            for c in range(9):
                p = board[r][c]
                if p == "r帥": bottom_king = "r"; break
                if p == "b將": bottom_king = "b"; break
            if bottom_king: break

        for r in range(10):
            for c in range(9):
                if board[r][c] == "?":
                    side = bottom_king if r >= 5 else ("r" if bottom_king == "b" else "b")
                    if not side: side = "b"
                    board[r][c] = side + "?"

        king_row = next((r for r in range(5,10) for c in range(9) if board[r][c] not in (".","?","b?","r?") and board[r][c][0] in "rb"), -1)
        if king_row >= 0:
            player_side = "r" if king_row >= 5 else "b"

        return board

    def board_to_string(self, board):
        lines = ["   " + "".join(f"{c:^5}" for c in range(9))]
        for row in range(10):
            l = f"{row:2d}|"
            for col in range(9):
                p = board[row][col]
                l += f" {p:<4}" if p != "." else "  .  "
            lines.append(l)
        return "\n".join(lines)
