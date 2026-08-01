"""
标注工具 - 网格固定尺寸
py -3.12 label.py
"""
import os, cv2, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_detector import YOLOChessDetector

img = cv2.imread(os.path.expanduser(r"~\Desktop\chess_fuck.png"))
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
assets = os.path.dirname(os.path.abspath(__file__)) + "/assets"

detector = YOLOChessDetector()
dets = detector.detect(img)

# Map to grid
cells = {}
for d in dets:
    col = int(d["x"] / w * 9)
    row = int(d["y"] / h * 10)
    key = (max(0, min(9, row)), max(0, min(8, col)))
    cells[key] = d

bx, by = 0, 0
cw, ch = w / 9, h / 10
S = 64

# Generate boxes image
display = img.copy()
labeled = set()
print(f"\n检测到 {len(cells)} 个棋子:\n")
for i, (pos, d) in enumerate(sorted(cells.items())):
    row, col = pos
    cx = int(bx + (col + 0.5) * cw)
    cy = int(by + (row + 0.5) * ch)
    x1 = max(0, cx - S//2)
    y1 = max(0, cy - S//2)
    x2 = min(w, cx + S//2)
    y2 = min(h, cy + S//2)
    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(display, str(i), (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    print(f" {i:>2}  ({row},{col})")
cv2.imencode(".png", display)[1].tofile(os.path.expanduser(r"~\Desktop\chess_fuck_boxes.png"))

valid = {"红帅","红仕","红相","红马","红车","红炮","红兵","黑将","黑士","黑象","黑马","黑车","黑砲","黑卒","暗红","暗黑"}
name_map = {
    "红帅":"r帅","红仕":"r仕","红相":"r相","红马":"r马","红车":"r车","红炮":"r炮","红兵":"r兵",
    "黑将":"b将","黑士":"b士","黑象":"b象","黑马":"b马","黑车":"b车","黑砲":"b砲","黑卒":"b卒",
    "暗红":"r_hidden","暗黑":"b_hidden",
}

# Clean old
for fn in os.listdir(assets):
    if "_hidden_" in fn: os.remove(os.path.join(assets, fn))

sorted_keys = sorted(cells.items())
print(f"\n格式: 编号 棋子  如: 0红车  5黑马  10暗红")
print("s=跳过 q=退出\n")

while True:
    cmd = input("> ").strip()
    if cmd == "q": break
    if cmd == "s": continue
    parts = cmd.split()
    if len(parts) < 2:
        i = 0
        while i < len(cmd) and cmd[i].isdigit(): i += 1
        if i > 0:
            parts = [cmd[:i], cmd[i:]]
    if len(parts) < 2:
        print("  格式: 编号+棋子名")
        continue
    idx = int(parts[0])
    name = parts[1]
    if idx < 0 or idx >= len(sorted_keys):
        print(f"  编号 0-{len(sorted_keys)-1}")
        continue
    if name not in valid:
        print(f"  可选: {sorted(valid)}")
        continue
    if idx in labeled:
        print(f"  #{idx} 已标过")
        continue

    pos, _ = sorted_keys[idx]
    row, col = pos
    cx = int(bx + (col + 0.5) * cw)
    cy = int(by + (row + 0.5) * ch)
    x1 = max(0, cx - S//2)
    y1 = max(0, cy - S//2)
    x2 = min(w, cx + S//2)
    y2 = min(h, cy + S//2)
    crop = np.zeros((S, S), dtype=np.uint8)
    h2, w2 = y2 - y1, x2 - x1
    if h2 > 0 and w2 > 0:
        crop[S//2-(cy-y1):S//2-(cy-y1)+h2, S//2-(cx-x1):S//2-(cx-x1)+w2] = gray[y1:y2, x1:x2]

    fn = f"{name_map[name]}.png"
    dst = os.path.join(assets, fn)
    cv2.imencode(".png", crop)[1].tofile(dst)
    labeled.add(idx)
    print(f"  #{idx} {name} -> {fn} ({crop.shape[1]}x{crop.shape[0]})")

print(f"\n已标 {len(labeled)} 个，现在运行 python final.py 测试。")
