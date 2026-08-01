import os, cv2, numpy as np, sys
sys.path.insert(0, os.path.expanduser(r"~\Desktop\jieqi_ai"))
from board_detector import BoardDetector, BOARD_ROWS, BOARD_COLS

img = cv2.imread(os.path.expanduser(r"~\Desktop\chess_fuck.png"))
h, w = img.shape[:2]
d = BoardDetector()
rect = d.find_board_corners(img)
if rect is None: rect = (0, 0, w, h)
bx, by, bw, bh = rect
bx += d.offset_x
by += d.offset_y
cw, ch = bw // BOARD_COLS, bh // BOARD_ROWS
mid = by + bh // 2

# Load face-up templates from assets (from chess.png)
assets = os.path.expanduser(r"~\Desktop\jieqi_ai\assets")
templates = {}
ref_cw, ref_ch = 92, 92
with open(os.path.join(assets, "cell_ref.txt")) as f:
    parts = f.read().strip().split()
    if len(parts) >= 2:
        ref_cw, ref_ch = int(parts[0]), int(parts[1])

for fn in os.listdir(assets):
    if not fn.endswith(".png"): continue
    if "_hidden" in fn.lower(): continue
    if fn == "cell_ref.txt": continue
    name = fn.replace(".png", "")
    if name[0] in ("r", "b") and len(name) >= 2:
        path = os.path.join(assets, fn)
        tpl = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
        if tpl is not None:
            templates[name] = tpl

# Scale templates to current cell size
scale_w = cw / ref_cw
scale_h = ch / ref_ch
scaled_tpl = {}
for name, tpl in templates.items():
    nw = max(int(tpl.shape[1] * scale_w), 1)
    nh = max(int(tpl.shape[0] * scale_h), 1)
    scaled_tpl[name] = cv2.resize(tpl, (nw, nh))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"Templates: {len(scaled_tpl)}  Board:({bx},{by}){bw}x{bh}  Cell:{cw}x{ch}")
print()
print("   " + "".join(f"{c:^5}" for c in range(9)))

for row in range(BOARD_ROWS):
    line = f"{row:2d} |"
    for col in range(BOARD_COLS):
        cx = bx + int((col + 0.5) * cw)
        cy = by + int((row + 0.5) * ch)

        # Check if there's a piece (mean + std)
        r_small = min(cw, ch) // 3
        x1, y1 = max(cx - r_small, 0), max(cy - r_small, 0)
        x2, y2 = min(cx + r_small, w), min(cy + r_small, h)
        if x2 <= x1 or y2 <= y1:
            line += "  .  "
            continue
        roi_small = gray[y1:y2, x1:x2]
        m = np.mean(roi_small)
        s = np.std(roi_small)
        if m > 200 or s > 55:
            line += "  .  "
            continue

        # Template matching with search window
        exp = "r" if cy < mid else "b"
        pad = min(cw, ch) // 4
        r_big = min(cw, ch) // 2 + pad
        sx1, sy1 = max(cx - r_big, 0), max(cy - r_big, 0)
        sx2, sy2 = min(cx + r_big, w), min(cy + r_big, h)
        roi = gray[sy1:sy2, sx1:sx2]

        best_name, best_score = "", 0
        for name, tpl in scaled_tpl.items():
            if name[0] != exp: continue
            if roi.shape[0] < tpl.shape[0] or roi.shape[1] < tpl.shape[1]: continue
            res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            mcx = loc[0] + tpl.shape[1] // 2
            mcy = loc[1] + tpl.shape[0] // 2
            ecx = cx - sx1
            ecy = cy - sy1
            dist = np.sqrt((mcx - ecx)**2 + (mcy - ecy)**2)
            if dist < pad * 1.5 and score > best_score:
                best_score = score
                best_name = name

        if best_score > 0.4:
            line += f" {best_name:<4}"
        else:
            if best_score > 0.3:
                line += f" {best_name:<4}"
            else:
                tag = "t?" if cy < mid else "d?"
                line += f" {tag:<4}"
    print(line)
