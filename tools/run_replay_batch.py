import sys, io, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import replay_quota as RQ

seeds = [330013, 330026, 330065, 330104, 330117, 330130, 330052, 330200, 330333]
buf = io.StringIO()
old = sys.stdout
sys.stdout = buf
for s in seeds:
    try:
        RQ.replay(s, think=0.15, new_is_red=True)
    except Exception as e:
        buf.write(f"[ERROR seed {s}] {e!r}\n")
    buf.write(f"=== end seed {s} ===\n")
sys.stdout = old
out = buf.getvalue()
# write ascii-safe summary
with open(os.path.join(os.path.dirname(__file__), "games", "replay_batch.log"), "w", encoding="utf-8") as f:
    f.write(out)
# print ascii summary
for line in out.splitlines():
    if "VIOLATION" in line or "=== end seed" in line or "ERROR" in line:
        print(line)
print("DONE")
