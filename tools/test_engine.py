import subprocess, os, threading, time, queue

exe = os.path.join(os.path.expanduser("~"), "Desktop", "jieqi_ai", "engines", "pikafish-bmi2.exe")
engine_dir = os.path.dirname(exe)

p = subprocess.Popen(
    exe, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    encoding="utf-8", errors="ignore", bufsize=1, cwd=engine_dir,
)

q = queue.Queue()
def reader():
    for line in iter(p.stdout.readline, ""):
        if line.strip():
            q.put(line.strip())
threading.Thread(target=reader, daemon=True).start()

def send(cmd):
    try:
        p.stdin.write(cmd + "\n")
        p.stdin.flush()
    except:
        pass

def wait_for(keyword, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            line = q.get(timeout=0.1)
            print("  <<", line)
            if keyword in line:
                return True
        except queue.Empty:
            pass
    print(f"  [!] 超时: {keyword}")
    return False

send("uci")
wait_for("uciok")

print("[*] 尝试使用内置 HCE (传统评估)...")
send("setoption name EvalFile value <empty>")
time.sleep(0.2)

send("isready")
wait_for("readyok")

send("position fen rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1")
send("go movetime 2000")
wait_for("bestmove", timeout=5)

send("quit")
time.sleep(0.3)
try: p.terminate()
except: pass
print("[+] 完成")
