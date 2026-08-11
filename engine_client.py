"""
揭棋引擎客户端 - 在主进程 (CPython, 带 cv2/paddleocr) 中启动引擎子进程 (优先 PyPy)。

设计:
  - 优先用 pypy3 运行 engine_server.py (4~5 倍加速)，找不到则回退 CPython。
  - 子进程常驻，通过 stdin/stdout 逐行 JSON 通信 (避免每步启停 + JIT 反复预热)。
  - get_best_move() 与原 JieQiEngine 接口一致，主程序改动最小。
  - 读响应带超时保护；子进程崩溃/超时会自动重启，并在本次调用返回 None。
"""
import sys, os, json, shutil, subprocess, threading, queue, time


class JieQiEngineClient:
    def __init__(self, prefer_pypy=True):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.server_path = os.path.join(self.base_dir, "engine_server.py")
        self.prefer_pypy = prefer_pypy
        self.python_bin = self._pick_interpreter()
        self.runtime_label = "PyPy" if "pypy" in os.path.basename(self.python_bin).lower() else "CPython"
        self._proc = None
        self._q = queue.Queue()
        self._reader_thread = None
        self._start()

    def _pick_interpreter(self):
        if self.prefer_pypy:
            for name in ("pypy3", "pypy3.11", "pypy"):
                path = shutil.which(name)
                if path:
                    return path
        # 回退: 当前 CPython
        return sys.executable

    def _reader(self, proc):
        """后台线程: 把子进程 stdout 的每一行放进队列。"""
        for line in iter(proc.stdout.readline, ""):
            s = line.strip()
            if s:
                self._q.put(s)

    def _start(self):
        # 清空旧队列
        self._q = queue.Queue()
        self._proc = subprocess.Popen(
            [self.python_bin, self.server_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            encoding="utf-8", errors="ignore", bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._reader, args=(self._proc,), daemon=True)
        self._reader_thread.start()
        # 等待就绪信号 (含预热时间)
        ready = self._read_json(timeout=30)
        if ready and ready.get("ready"):
            print(f"[+] 引擎子进程就绪 ({self.runtime_label}: {os.path.basename(self.python_bin)})")
        else:
            print(f"[!] 引擎子进程就绪信号异常: {ready}")

    def _read_json(self, timeout):
        """从队列读一行并解析 JSON；超时返回 None。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self._q.get(timeout=0.1)
            except queue.Empty:
                if self._proc.poll() is not None:  # 子进程已退出
                    return None
                continue
            try:
                return json.loads(line)
            except Exception:
                continue  # 忽略非 JSON 行
        return None

    def _send(self, obj):
        try:
            self._proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            return True
        except Exception:
            return False

    def get_best_move(self, board, my_side, think_time=2.0):
        """返回 (uci, score, depth)；失败返回 (None, 0, 0)。接口与 JieQiEngine 一致。"""
        # 子进程若已死，先重启
        if self._proc is None or self._proc.poll() is not None:
            print("[!] 引擎子进程不在运行，重启中...")
            self._start()

        if not self._send({"cmd": "go", "board": board, "my_side": my_side, "think_time": think_time}):
            print("[!] 发送失败，重启引擎子进程...")
            self._restart()
            return None, 0, 0

        # 读响应: 给足 think_time + 缓冲 (搜索是软限制，可能略超)
        resp = self._read_json(timeout=think_time + 15)
        if resp is None:
            print("[!] 引擎响应超时/子进程异常，重启...")
            self._restart()
            return None, 0, 0
        if not resp.get("ok"):
            print(f"[!] 引擎错误: {resp.get('error')}")
            return None, 0, 0
        return resp.get("uci"), resp.get("score", 0), resp.get("depth", 0)

    def _restart(self):
        self.close()
        self._start()

    def close(self):
        if self._proc:
            try:
                self._send({"cmd": "quit"})
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
            self._proc = None


if __name__ == "__main__":
    # 自测: 启动客户端，跑一个开局局面
    c = JieQiEngineClient()
    b = [["."] * 9 for _ in range(10)]
    for col in range(9):
        b[9][col] = "r帥" if col == 4 else "r?"
        b[0][col] = "b將" if col == 4 else "b?"
    b[7][1] = "r?"; b[7][7] = "r?"
    b[2][1] = "b?"; b[2][7] = "b?"
    for col in (0, 2, 4, 6, 8):
        b[6][col] = "r?"; b[3][col] = "b?"
    t0 = time.time()
    print("结果:", c.get_best_move(b, "r", think_time=2.0), f"耗时 {time.time()-t0:.2f}s")
    c.close()
