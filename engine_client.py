"""
揭棋引擎客户端 - 在主进程 (CPython, 带 cv2/paddleocr) 中启动引擎子进程。

支持两种引擎, 二选一:
  - "pypy" : miaosiSari/Jieqi 纯算法引擎 (PyPy/CPython 子进程, alpha-beta 搜索)   [默认]
  - "java" : Makinuohara/2026-jieqi-AI 引擎 (Java 子进程, expectiminimax 搜索)

设计:
  - 子进程常驻, stdin/stdout 逐行 JSON 通信 (避免每步启停 + JIT/JVM 反复启动开销)。
  - get_best_move() 接口统一, 主程序改动最小。
  - 读响应带超时保护; 子进程崩溃/超时会自动重启, 本次调用返回 None。
"""
import sys, os, json, shutil, subprocess, threading, queue, time


class _EngineClientBase:
    """通用子进程客户端: 负责启动/通信/超时/重启。子类只需提供启动命令与标签。"""

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self._proc = None
        self._q = queue.Queue()
        self._reader_thread = None

    # --- 子类实现 ---
    def _launch_cmd(self):
        raise NotImplementedError

    def _label(self):
        raise NotImplementedError

    # --- 通用逻辑 ---
    def _reader(self, proc):
        for line in iter(proc.stdout.readline, ""):
            s = line.strip()
            if s:
                self._q.put(s)

    def _start(self):
        self._q = queue.Queue()
        cmd = self._launch_cmd()
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            encoding="utf-8", errors="ignore", bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._reader, args=(self._proc,), daemon=True)
        self._reader_thread.start()
        ready = self._read_json(timeout=30)
        if ready and ready.get("ready"):
            print(f"[+] 引擎子进程就绪 ({self._label()})")
        else:
            print(f"[!] 引擎子进程就绪信号异常: {ready}")

    def _read_json(self, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self._q.get(timeout=0.1)
            except queue.Empty:
                if self._proc.poll() is not None:
                    return None
                continue
            try:
                return json.loads(line)
            except Exception:
                continue
        return None

    def _send(self, obj):
        try:
            self._proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            return True
        except Exception:
            return False

    def get_best_move(self, board, my_side, think_time=2.0):
        """返回 (uci, score, depth); 失败返回 (None, 0, 0)。"""
        if self._proc is None or self._proc.poll() is not None:
            print("[!] 引擎子进程不在运行, 重启中...")
            self._start()

        if not self._send({"cmd": "go", "board": board, "my_side": my_side, "think_time": think_time}):
            print("[!] 发送失败, 重启引擎子进程...")
            self._restart()
            return None, 0, 0

        # 读响应上限: 引擎的 max_time 是软限制(每层搜完才检查时间), 开放中局单层
        # 可能远超预算, 故留足余量, 避免把"算得慢但算得出"误判为异常而杀掉重启。
        resp = self._read_json(timeout=think_time + 30)
        if resp is None:
            print("[!] 引擎响应超时/子进程异常, 重启...")
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


class JavaEngineClient(_EngineClientBase):
    """Makinuohara 引擎 (Java 子进程, expectiminimax)。"""

    def __init__(self):
        super().__init__()
        self.classes_dir = os.path.join(self.base_dir, "engines", "mak", "classes")
        self.main_class = "edu.bupt.jieqi.bridge.EngineBridge"
        self.java_bin = self._pick_java()
        if self.java_bin is None:
            raise RuntimeError("未找到 java, 无法启动 Java 引擎 (请安装 JDK 21)")
        if not os.path.isdir(self.classes_dir):
            raise RuntimeError(f"未找到引擎 classes 目录: {self.classes_dir}")
        self._start()

    def _pick_java(self):
        # 候选顺序: 环境变量 JAVA_HOME -> Homebrew openjdk@21 -> PATH 里的 java。
        # 注意: macOS 的 /usr/bin/java 可能是 stub (未装真正 JDK 时无法运行),
        # 所以逐个候选实际执行 `java -version` 验证可用性。
        candidates = []
        jh = os.environ.get("JAVA_HOME")
        if jh:
            candidates.append(os.path.join(jh, "bin", "java"))
        candidates += [
            "/opt/homebrew/opt/openjdk@21/bin/java",
            "/usr/local/opt/openjdk@21/bin/java",
        ]
        w = shutil.which("java")
        if w:
            candidates.append(w)
        for cand in candidates:
            if cand and os.path.exists(cand) and self._java_works(cand):
                return cand
        return None

    @staticmethod
    def _java_works(java_bin):
        try:
            r = subprocess.run([java_bin, "-version"], capture_output=True, timeout=10)
            return r.returncode == 0
        except Exception:
            return False

    def _launch_cmd(self):
        return [self.java_bin, "-cp", self.classes_dir, self.main_class]

    def _label(self):
        return f"Makinuohara/Java: {os.path.basename(self.java_bin)}"


class PypyEngineClient(_EngineClientBase):
    """miaosiSari 纯算法引擎 (PyPy 优先, 回退 CPython)。"""

    def __init__(self, prefer_pypy=True):
        super().__init__()
        self.server_path = os.path.join(self.base_dir, "engine_server.py")
        self.prefer_pypy = prefer_pypy
        self.python_bin = self._pick_interpreter()
        self.runtime_label = "PyPy" if "pypy" in os.path.basename(self.python_bin).lower() else "CPython"
        self._start()

    def _pick_interpreter(self):
        if self.prefer_pypy:
            for name in ("pypy3", "pypy3.11", "pypy"):
                p = shutil.which(name)
                if p:
                    return p
        return sys.executable

    def _launch_cmd(self):
        return [self.python_bin, self.server_path]

    def _label(self):
        return f"{self.runtime_label}: {os.path.basename(self.python_bin)}"


def create_engine(engine_type="pypy", prefer_pypy=True):
    """引擎工厂。engine_type: "pypy"/"pypy3"(默认) 或 "java"。
    java 不可用(无 JDK 或无 classes)时自动回退到 pypy。"""
    engine_type = engine_type.lower()
    if engine_type == "java":
        try:
            return JavaEngineClient()
        except Exception as e:
            print(f"[!] Java 引擎不可用 ({e}), 回退 miaosiSari 引擎")
            return PypyEngineClient(prefer_pypy=prefer_pypy)
    # "pypy" 和 "pypy3" 都走 miaosiSari 引擎
    if engine_type in ("pypy", "pypy3"):
        return PypyEngineClient(prefer_pypy=True)
    # 未知引擎类型，默认走 pypy
    print(f"[!] 未知引擎类型 '{engine_type}'，回退 miaosiSari 引擎")
    return PypyEngineClient(prefer_pypy=prefer_pypy)


# 向后兼容: 旧名保留 (miaosiSari PyPy 引擎)
class JieQiEngineClient(PypyEngineClient):
    pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="pypy", choices=["pypy", "java"])
    args = ap.parse_args()

    c = create_engine(args.engine)
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
