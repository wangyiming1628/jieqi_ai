"""
YOLO检测 + PaddleOCR原始概率 — 从6625类中取象棋字最高分
"""
import os, cv2, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_detector import YOLOChessDetector
from paddleocr import PaddleOCR

img = cv2.imread(os.path.expanduser(r"~\Desktop\chess_fuck.png"))
mid_y = img.shape[0] / 2

detector = YOLOChessDetector()
dets = detector.detect(img)

ocr = PaddleOCR(lang="ch", use_angle_cls=False, show_log=False)
rec = ocr.text_recognizer
char_list = rec.postprocess_op.character
max_wh = rec.rec_image_shape[2] / rec.rec_image_shape[1]

CHESS = "帅帥仕相马馬车車炮兵将士士象砲卒将將"

def raw_ocr(crop_bgr):
    """Return best chess character + probability from raw model output"""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    norm = rec.resize_norm_img(rgb, max_wh)
    norm_batch = np.expand_dims(norm, 0).copy()
    
    input_names = rec.predictor.get_input_names()
    handle = rec.predictor.get_input_handle(input_names[0])
    handle.copy_from_cpu(norm_batch)
    rec.predictor.run()
    
    output_handles = [rec.predictor.get_output_handle(n) for n in rec.predictor.get_output_names()]
    probs = np.array(output_handles[0].copy_to_cpu())  # (1, 40, 6625)
    
    best_ch, best_p = "", 0.0
    for pos in range(probs.shape[1]):
        for ch in CHESS:
            idx = char_list.index(ch) if ch in char_list else -1
            if idx >= 0:
                p = probs[0, pos, idx]
                if p > best_p:
                    best_p = p
                    best_ch = ch
    return best_ch, best_p

lines = [f"Pieces: {len(dets)}\n"]
board = [["."] * 9 for _ in range(10)]

for d in dets:
    cx, cy = d["x"], d["y"]
    x1 = max(0, int(cx - d["w"]/2))
    y1 = max(0, int(cy - d["h"]/2))
    x2 = min(img.shape[1], int(cx + d["w"]/2))
    y2 = min(img.shape[0], int(cy + d["h"]/2))
    full = img[y1:y2, x1:x2]
    if full.size == 0: continue
    h, w = full.shape[:2]

    gray = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pcx, pcy = w // 2, h // 2
    if contours:
        largest = max(contours, key=cv2.contourArea)
        (px, py), _ = cv2.minEnclosingCircle(largest)
        pcx, pcy = int(px), int(py)

    S = int(min(d["w"], d["h"]) * 0.58)
    cx2 = max(0, int(x1 + pcx - S//2))
    cy2 = max(0, int(y1 + pcy - S//2))
    cx3 = min(img.shape[1], int(x1 + pcx + S//2))
    cy3 = min(img.shape[0], int(y1 + pcy + S//2))
    crop = img[cy2:cy3, cx2:cx3]

    ch, conf = raw_ocr(crop)
    
    col = int(cx / img.shape[1] * 9)
    row = int(cy / img.shape[0] * 10)
    row, col = max(0, min(9, row)), max(0, min(8, col))

    if ch and conf > 0.01:
        side = "红" if cy < mid_y else "黑"
        board[row][col] = side + ch
    else:
        board[row][col] = ("暗红" if cy < mid_y else "暗黑")

    lines.append(f"  ({row},{col}) {ch or '?'} p={conf:.4f} -> {board[row][col]}")

lines.append(f"\n   " + "".join(f"{c:^5}" for c in range(9)))
for row in range(10):
    l = f"{row:2d}|"
    for col in range(9):
        p = board[row][col]
        l += f" {p:<4}" if p != "." else "  .  "
    lines.append(l)

open(os.path.expanduser(r"~\Desktop\ocr_result.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("Done -> Desktop/ocr_result.txt")
