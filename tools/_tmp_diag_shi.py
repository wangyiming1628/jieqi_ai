"""一次性诊断: 微信图片局面 — 中路双士为何漏识别 (跑完即删)"""
import sys, os, cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

IMG = sys.argv[1] if len(sys.argv) > 1 else r"c:\Users\yimingwang218185\Desktop\微信图片_20260823202113_160_17.png"

img = cv2.imdecode(np.fromfile(IMG, dtype=np.uint8), cv2.IMREAD_COLOR)
h, w = img.shape[:2]
print(f"原图尺寸: {w}x{h}")

# 按目测比例裁棋盘区
x1, x2 = int(w * 0.283), int(w * 0.715)
y1, y2 = int(h * 0.108), int(h * 0.925)
crop = img[y1:y2, x1:x2]
print(f"裁剪区: ({x1},{y1})-({x2},{y2}) = {crop.shape[1]}x{crop.shape[0]}")

from board_recognizer import BoardRecognizer
rec = BoardRecognizer()
board = rec.detect(crop)
print(rec.board_to_string(board))

print("\n=== 调试信息: 每格 YOLO/Hough/OCR ===")
for cell in sorted(rec.last_debug["cells"], key=lambda c: (c["row"], c["col"])):
    print(f"({cell['row']},{cell['col']}) yolo_conf={cell['yolo_conf']:.2f} "
          f"hough={cell['hough_found']} r={cell['hough_r']} "
          f"ocr={cell['ocr_char']} conf={cell['ocr_conf']:.3f} hidden={cell['is_hidden']}")

# 原始 YOLO 检测 (不经识别器后处理): 看中路区域有没有框
print("\n=== 原始 YOLO 检测框 (按 y 排序) ===")
dets = rec.detector.detect(crop)
ch, cw = crop.shape[0] / 10, crop.shape[1] / 9
for d in sorted(dets, key=lambda t: (t["y"], t["x"])):
    r, c = int(d["y"] / ch), int(d["x"] / cw)
    print(f"grid({r},{c}) conf={d['conf']:.2f} cls={d['piece']} "
          f"box=({d['x']:.0f},{d['y']:.0f},{d['w']:.0f}x{d['h']:.0f})")
