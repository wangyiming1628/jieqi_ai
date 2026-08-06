"""
棋盘识别器 - 封装 YOLO + PaddleOCR 识别逻辑
"""
import os, cv2, sys, time, subprocess, numpy as np
from typing import Optional, List
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_detector import YOLOChessDetector
from paddleocr import PaddleOCR

CHESS = "帥仕相馬車炮兵將士象卒"
RED_ONLY = set("帥仕相兵")
BLACK_ONLY = set("將士象卒")


class ScreenCapture:
    """macOS 后台窗口截屏器"""

    def __init__(self, target_title: str = "天天象棋", target_owner: str = "微信"):
        if sys.platform != "darwin":
            raise OSError("ScreenCapture 仅支持 macOS")
        self.target_title = target_title
        self.target_owner = target_owner
        self.window_id = None
        self.window_bounds = None  # (x, y, width, height)
        self.window_name = ""

    def find_window(self) -> bool:
        infos = self._list_windows()
        for info in infos:
            owner = info.get("kCGWindowOwnerName", "")
            name = info.get("kCGWindowName", "")
            if owner == self.target_owner and name == self.target_title:
                return self._set_window(info)
        candidates = []
        for info in infos:
            owner = info.get("kCGWindowOwnerName", "")
            if owner != self.target_owner:
                continue
            name = info.get("kCGWindowName", "")
            layer = info.get("kCGWindowLayer", 0)
            bounds = info.get("kCGWindowBounds", {})
            w = int(bounds.get("Width", 0))
            h = int(bounds.get("Height", 0))
            if name == "微信":
                continue
            if layer > 0:
                continue
            if w < 700 or h < 400:
                continue
            candidates.append((w * h, info))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return self._set_window(candidates[0][1])
        candidates = []
        for info in infos:
            owner = info.get("kCGWindowOwnerName", "")
            if owner != self.target_owner:
                continue
            name = info.get("kCGWindowName", "")
            bounds = info.get("kCGWindowBounds", {})
            w = int(bounds.get("Width", 0))
            h = int(bounds.get("Height", 0))
            if name == "微信":
                continue
            if w < 400 or h < 300:
                continue
            candidates.append((w * h, info))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return self._set_window(candidates[0][1])
        print(f"[!] 未找到窗口: [{self.target_owner}] \"{self.target_title}\"")
        return False

    def _set_window(self, info: dict) -> bool:
        self.window_id = info.get("kCGWindowNumber")
        bounds = info.get("kCGWindowBounds", {})
        self.window_bounds = (
            int(bounds.get("X", 0)),
            int(bounds.get("Y", 0)),
            int(bounds.get("Width", 0)),
            int(bounds.get("Height", 0)),
        )
        self.window_name = info.get("kCGWindowName", "")
        owner = info.get("kCGWindowOwnerName", "")
        print(f"[+] 目标窗口: [{owner}] \"{self.window_name}\" "
              f"ID:{self.window_id} {self.window_bounds[2]}x{self.window_bounds[3]}")
        return True

    def _list_windows(self) -> List[dict]:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGWindowListOptionOnScreenOnly,
            kCGNullWindowID,
        )
        return CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)

    def _screencapture_raw(self) -> Optional[np.ndarray]:
        """截取完整窗口截图"""
        if self.window_id is None:
            if not self.find_window():
                return None
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "snapshot", "_tmp_capture.png"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            subprocess.run(
                ["screencapture", f"-l{self.window_id}", "-x", output_path],
                check=True,
                timeout=5,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            return None
        if not os.path.exists(output_path):
            return None
        img = cv2.imread(output_path)
        try:
            os.remove(output_path)
        except OSError:
            pass
        return img

    def capture(self, output_path: Optional[str] = None) -> Optional[np.ndarray]:
        img = self._screencapture_raw()
        if img is None:
            return None
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        if sys.platform == "darwin":
            crop_w, crop_h = 1529, 1695
        else:
            crop_w, crop_h = 1035, 1143
        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        return img[y1:y2, x1:x2]

    def capture_full(self) -> Optional[np.ndarray]:
        return self._screencapture_raw()

    def capture_region(self, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
        img = self.capture()
        if img is None:
            return None
        return img[y:y + h, x:x + w]

    def save_snapshot(self, output_dir: str = None) -> Optional[str]:
        img = self.capture()
        if img is None:
            return None
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshot")
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"capture_{ts}.png")
        cv2.imencode(".png", img)[1].tofile(path)
        print(f"[+] 截图已保存: {path}")
        return path

    def list_windows(self) -> List[dict]:
        infos = self._list_windows()
        windows = []
        for info in infos:
            windows.append({
                "owner": info.get("kCGWindowOwnerName", ""),
                "name": info.get("kCGWindowName", ""),
                "id": info.get("kCGWindowNumber", 0),
                "bounds": info.get("kCGWindowBounds", {}),
                "layer": info.get("kCGWindowLayer", 0),
            })
        return windows


class BoardRecognizer:
    def __init__(self):
        self.detector = YOLOChessDetector()
        self.ocr = PaddleOCR(lang="ch", use_angle_cls=False, show_log=False)
        self.rec = self.ocr.text_recognizer
        self.char_list = self.rec.postprocess_op.character
        self.max_wh = self.rec.rec_image_shape[2] / self.rec.rec_image_shape[1]
        # 优化1：预计算棋子字在字典中的索引，解码时向量化取值，
        # 避免每次对 6000+ 长度的 char_list 做 list.index() 线性查找
        self._chess_chars = list(CHESS)
        self._chess_idx = np.array([self.char_list.index(c) for c in self._chess_chars])
        self._capture = None  # macOS: ScreenCapture 实例
        if sys.platform == "darwin":
            self._capture = ScreenCapture(target_title="天天象棋", target_owner="微信")

    def capture_screen(self):
        """macOS: 从后台天天象棋窗口截取中心 1529x1695 区域；Windows: 返回 None（由 controller 截取）"""
        if self._capture is not None:
            return self._capture.capture()
        return None

    def _raw_ocr_batch(self, crops_bgr):
        """优化2：批量推理。一次前向处理多张棋子图，摊薄固定调用开销。
        优化1：向量化解码，只在预计算的棋子字索引上取 argmax。
        返回 [(char, prob), ...]，顺序与输入对齐。"""
        if not crops_bgr:
            return []
        # 统一 resize/pad 到相同尺寸后堆叠成一个 batch
        norms = []
        for crop in crops_bgr:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            norms.append(self.rec.resize_norm_img(rgb, self.max_wh))
        norm_batch = np.stack(norms, axis=0).copy()

        input_names = self.rec.predictor.get_input_names()
        handle = self.rec.predictor.get_input_handle(input_names[0])
        handle.copy_from_cpu(norm_batch)
        self.rec.predictor.run()
        output_handles = [self.rec.predictor.get_output_handle(n) for n in self.rec.predictor.get_output_names()]
        probs = np.array(output_handles[0].copy_to_cpu())  # (N, pos, vocab)

        # 只在棋子字对应的列上取值 → (N, pos, len(CHESS))
        sub = probs[:, :, self._chess_idx]
        results = []
        for i in range(sub.shape[0]):
            si = sub[i]  # (pos, len(CHESS))
            flat = int(si.argmax())
            char_j = flat % len(self._chess_chars)
            results.append((self._chess_chars[char_j], float(si.flat[flat])))
        return results

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

    def detect(self, image: np.ndarray, my_side: str = None):
        """返回 10x9 棋盘。my_side 用于暗棋阵营推断。"""
        h, w = image.shape[:2]
        mid_y = h / 2
        dets = self.detector.detect(image)

        board = [["."] * 9 for _ in range(10)]

        # 同格去重：多个检测框映射到同一格子时，只保留置信度最高的
        cell_best = {}
        for d in dets:
            cx, cy = d["x"], d["y"]
            col = int(cx / w * 9)
            row = int(cy / h * 10)
            row, col = max(0, min(9, row)), max(0, min(8, col))
            key = (row, col)
            if key not in cell_best or d["conf"] > cell_best[key]["conf"]:
                cell_best[key] = d

        # 第一遍：收集每格的裁剪图与元数据（不逐个推理）
        cells = []  # [(row, col, crop, is_hidden), ...]
        crops = []
        for (row, col), d in cell_best.items():
            cx, cy = d["x"], d["y"]

            x1 = max(0, int(cx - d["w"] / 2))
            y1 = max(0, int(cy - d["h"] / 2))
            x2 = min(w, int(cx + d["w"] / 2))
            y2 = min(h, int(cy + d["h"] / 2))
            full = image[y1:y2, x1:x2]
            if full.size == 0: continue

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
            if crop.size == 0: continue

            is_hidden = np.std(gray) < 40
            cells.append((row, col, crop, is_hidden))
            crops.append(crop)

        # 优化2：一次批量推理所有棋子（替代逐格串行调用）
        ocr_results = self._raw_ocr_batch(crops)

        # 第二遍：用批量结果填表
        for (row, col, crop, is_hidden), (ch, conf) in zip(cells, ocr_results):
            if ch and conf > 0.001:
                detected = self._detect_side(crop, ch)
                if detected:
                    side = detected
                else:
                    side = "?"  # 不确定阵营，后面统一修正
                board[row][col] = side + ch
            else:
                board[row][col] = "?"

        # 根据帥/將位置确定阵营：找到红帥和黑將所在半场
        red_half = None  # "top" (row 0-4) 或 "bottom" (row 5-9)
        for r in range(10):
            for c in range(9):
                if board[r][c] == "r帥":
                    red_half = "top" if r < 5 else "bottom"
                elif board[r][c] == "b將":
                    # 黑將所在半场 = 黑方，另一半 = 红方
                    red_half = "bottom" if r < 5 else "top"

        # 修正所有暗子阵营
        for r in range(10):
            for c in range(9):
                p = board[r][c]
                if p == "?" or (len(p) == 2 and p[1] == "?"):
                    if red_half is not None:
                        is_red_half = (r < 5 and red_half == "top") or (r >= 5 and red_half == "bottom")
                        board[r][c] = ("r" if is_red_half else "b") + "?"
                    elif my_side is not None:
                        opp = "b" if my_side == "r" else "r"
                        board[r][c] = (my_side if r >= 5 else opp) + "?"

        # 暗子位置合法性校验：引擎只允许暗子出现在标准开局棋子所在行和列
        # 根据实际阵营朝向动态构建校验表（红方可能在棋盘上方 row0-4 或下方 row5-9）
        _HIDDEN_VALID = {}
        if red_half == "top":
            # 红方在 row0-4（上方），黑方在 row5-9（下方）
            _HIDDEN_VALID = {
                ("r", 0): set(range(9)),
                ("r", 2): {1, 7},
                ("r", 3): {0, 2, 4, 6, 8},
                ("b", 9): set(range(9)),
                ("b", 7): {1, 7},
                ("b", 6): {0, 2, 4, 6, 8},
            }
        elif red_half == "bottom":
            # 红方在 row5-9（下方），黑方在 row0-4（上方）
            _HIDDEN_VALID = {
                ("r", 9): set(range(9)),
                ("r", 7): {1, 7},
                ("r", 6): {0, 2, 4, 6, 8},
                ("b", 0): set(range(9)),
                ("b", 2): {1, 7},
                ("b", 3): {0, 2, 4, 6, 8},
            }
        else:
            # 无法确定阵营朝向时：两边都允许（保守策略，避免误删合法暗子）
            _HIDDEN_VALID = {
                ("r", 0): set(range(9)), ("r", 2): {1, 7}, ("r", 3): {0, 2, 4, 6, 8},
                ("r", 9): set(range(9)), ("r", 7): {1, 7}, ("r", 6): {0, 2, 4, 6, 8},
                ("b", 0): set(range(9)), ("b", 2): {1, 7}, ("b", 3): {0, 2, 4, 6, 8},
                ("b", 9): set(range(9)), ("b", 7): {1, 7}, ("b", 6): {0, 2, 4, 6, 8},
            }
        for r in range(10):
            for c in range(9):
                p = board[r][c]
                if len(p) == 2 and p[1] == "?":
                    side = p[0]
                    valid_cols = _HIDDEN_VALID.get((side, r))
                    if valid_cols is None or c not in valid_cols:
                        board[r][c] = "."

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
