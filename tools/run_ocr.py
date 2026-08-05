import sys, os, cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from board_recognizer import BoardRecognizer

img_path = os.path.join(ROOT, "snapshot", "212353.png")
img = cv2.imread(img_path)
if img is None:
    print(f"[!] 无法读取图片: {img_path}")
    sys.exit(1)
print(f"[*] image: {img.shape}")

rec = BoardRecognizer()
board = rec.detect(img)

# 构建输出（不再依赖缓存，无概率信息）
lines = []
lines.append("   " + "".join(f"{c:^6}" for c in range(9)))
for row in range(10):
    l = f"{row:2d}|"
    for col in range(9):
        p = board[row][col]
        l += f" {p:<5}" if p != "." else "  .   "
    lines.append(l)
result = "\n".join(lines)
print(result)

out_path = os.path.join(ROOT, "tools", "ocr_result.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(result + "\n")
print(f"[+] saved: {out_path}")

# 保存 YOLO 候选框可视化
h, w = img.shape[:2]
dets = rec.detector.detect(img)
vis = img.copy()
for d in dets:
    cx, cy = d["x"], d["y"]
    x1 = max(0, int(cx - d["w"] / 2))
    y1 = max(0, int(cy - d["h"] / 2))
    x2 = min(w, int(cx + d["w"] / 2))
    y2 = min(h, int(cy + d["h"] / 2))
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis, f"{d.get('piece','?')} {d.get('conf',0):.2f}",
                (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

out_vis = os.path.join(ROOT, "snapshot", f"{os.path.splitext(os.path.basename(img_path))[0]}_yolo_boxes.png")
cv2.imwrite(out_vis, vis)
print(f"[+] YOLO 候选框: {out_vis} ({len(dets)} 个)")
