"""
YOLO 象棋棋子识别 - 基于 dffge552/xiangqi-pwa-offline ONNX 模型
双模型架构：检测器 + 分类器
"""
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional

# 7 种棋子类型
PIECE_NAMES = ["K", "A", "B", "N", "R", "C", "P"]
PIECE_CHINESE_RED = {"K": "帥", "A": "仕", "B": "相", "N": "馬", "R": "車", "C": "炮", "P": "兵"}
PIECE_CHINESE_BLACK = {"K": "將", "A": "士", "B": "象", "N": "馬", "R": "車", "C": "炮", "P": "卒"}


class YOLOChessDetector:
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engines")
        import onnxruntime as ort
        self.detector = ort.InferenceSession(
            os.path.join(model_dir, "online_xiangqi_piece_detector.onnx"),
            providers=["CPUExecutionProvider"]
        )
        self.classifier = ort.InferenceSession(
            os.path.join(model_dir, "online_xiangqi_classifier.onnx"),
            providers=["CPUExecutionProvider"]
        )
        self.input_size = 640
        self.clf_size = 64
        self.conf_threshold = 0.3
        self.iou_threshold = 0.5

    def detect(self, image: np.ndarray) -> List[dict]:
        """检测所有棋子位置和类型"""
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size))
        inp = resized.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))
        inp = np.expand_dims(inp, 0)

        out = self.detector.run(None, {"images": inp})[0]
        boxes = out[0]  # [5, 8400]

        raw_detections = []
        for i in range(boxes.shape[1]):
            x, y, bw_, bh_, conf = boxes[:5, i]
            if conf > self.conf_threshold:
                raw_detections.append({
                    "x": float(x / self.input_size * w),
                    "y": float(y / self.input_size * h),
                    "w": float(bw_ / self.input_size * w),
                    "h": float(bh_ / self.input_size * h),
                    "conf": float(conf),
                })

        filtered = self._nms(raw_detections)
        filtered = self._size_filter(filtered)
        return self._classify(image, filtered)

    def _size_filter(self, detections: List[dict]) -> List[dict]:
        if len(detections) < 5:
            return detections
        areas = [d["w"] * d["h"] for d in detections]
        aspects = [d["w"] / max(d["h"], 1) for d in detections]
        med_area = np.median(areas)
        med_aspect = np.median(aspects)
        return [
            d for d in detections
            if 0.5 * med_area < d["w"] * d["h"] < 2.0 * med_area
            and 0.5 < d["w"] / max(d["h"], 1) < 2.0
            and d["w"] > 40 and d["h"] > 40  # 绝对最小尺寸，过滤过小的误检
        ]

    def _nms(self, detections: List[dict]) -> List[dict]:
        detections.sort(key=lambda d: d["conf"], reverse=True)
        result = []
        while detections:
            best = detections.pop(0)
            result.append(best)
            detections = [
                d for d in detections
                if self._iou(best, d) < self.iou_threshold
            ]
        return result

    def _iou(self, a: dict, b: dict) -> float:
        ax1, ay1 = a["x"] - a["w"] / 2, a["y"] - a["h"] / 2
        ax2, ay2 = a["x"] + a["w"] / 2, a["y"] + a["h"] / 2
        bx1, by1 = b["x"] - b["w"] / 2, b["y"] - b["h"] / 2
        bx2, by2 = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
        iw = min(ax2, bx2) - max(ax1, bx1)
        ih = min(ay2, by2) - max(ay1, by1)
        if iw <= 0 or ih <= 0:
            return 0
        inter = iw * ih
        area_a = a["w"] * a["h"]
        area_b = b["w"] * b["h"]
        return inter / (area_a + area_b - inter)

    def _classify(self, image: np.ndarray, detections: List[dict]) -> List[dict]:
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = []
        for d in detections:
            x1 = max(0, int(d["x"] - d["w"] / 2))
            y1 = max(0, int(d["y"] - d["h"] / 2))
            x2 = min(w, int(d["x"] + d["w"] / 2))
            y2 = min(h, int(d["y"] + d["h"] / 2))
            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0:
                d["piece"] = "?"
                results.append(d)
                continue
            crop64 = cv2.resize(crop, (self.clf_size, self.clf_size))
            clf_inp = crop64.astype(np.float32) / 255.0
            clf_inp = np.transpose(clf_inp, (2, 0, 1))
            clf_inp = np.expand_dims(clf_inp, 0)
            probs = self.classifier.run(None, {"images": clf_inp})[0][0]
            class_id = int(np.argmax(probs))
            d["piece"] = PIECE_NAMES[class_id]
            d["probs"] = probs.tolist()
            results.append(d)
        return results

    def to_board(self, image: np.ndarray, board_rect: Tuple[int, int, int, int],
                 midline: int) -> List[List[str]]:
        """將检测结果映射到 10x9 棋盘"""
        detections = self.detect(image)
        board = [["."] * 9 for _ in range(10)]
        bx, by, bw, bh = board_rect
        cw, ch = bw / 9, bh / 10

        for d in detections:
            col = int((d["x"] - bx) / cw)
            row = int((d["y"] - by) / ch)
            if 0 <= row < 10 and 0 <= col < 9:
                side = "r" if d["y"] < midline else "b"
                name_map = PIECE_CHINESE_RED if side == "r" else PIECE_CHINESE_BLACK
                name = name_map.get(d["piece"], "?")
                board[row][col] = f"{side}{name}"
        return board
