import sys, os, cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from board_recognizer import BoardRecognizer

img_path = os.path.join(ROOT, "snapshot", "102644.png")
img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
print(f"[*] image: {img.shape}")

rec = BoardRecognizer()
board = rec.detect(img)
result = rec.board_to_string(board)
print(result)

out_path = os.path.join(ROOT, "tools", "ocr_result.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(result + "\n")
print(f"[+] saved: {out_path}")
