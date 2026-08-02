import os
import stat
import subprocess
import sys
import threading
import time
import queue
from typing import Optional, Tuple


class UCCIEngine:
    """皮卡鱼 UCI 引擎封装"""

    def __init__(self, engine_path: str):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"引擎不存在: {engine_path}")
        self.engine_path = engine_path
        self.process = None
        self.name = ""
        self.author = ""
        self._q = queue.Queue()
        self._reader_thread = None
        self._start()

    def _start(self):
        engine_dir = os.path.dirname(self.engine_path)
        self.process = subprocess.Popen(
            self.engine_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='ignore',
            bufsize=1,
            cwd=engine_dir,
        )
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

        self._send("uci")
        self._wait_for("uciok")

        self._send("isready")
        self._wait_for("readyok")

    def _read_loop(self):
        for line in iter(self.process.stdout.readline, ""):
            stripped = line.strip()
            if stripped:
                if "id name" in stripped and not self.name:
                    self.name = stripped.split("id name ")[-1]
                elif "id author" in stripped and not self.author:
                    self.author = stripped.split("id author ")[-1]
                self._q.put(stripped)

    def _send(self, cmd: str):
        try:
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()
        except (OSError, ValueError):
            pass

    def _wait_for(self, keyword: str, timeout: float = 5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self._q.get(timeout=0.1)
                if keyword in line:
                    return
            except queue.Empty:
                pass

    def _drain(self):
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def search(self, fen: str, movetime: int = 2000) -> Tuple[str, str]:
        self._send("stop")
        time.sleep(0.1)
        self._drain()
        self._send(f"position fen {fen}")
        self._send(f"go movetime {movetime}")
        bestmove = ""
        ponder = ""
        deadline = time.time() + movetime / 1000.0 + 3
        while time.time() < deadline:
            try:
                line = self._q.get(timeout=0.2)
                if line.startswith("bestmove"):
                    parts = line.split()
                    bestmove = parts[1] if len(parts) > 1 else ""
                    ponder = parts[3] if len(parts) > 3 else ""
                    break
            except queue.Empty:
                pass
        return bestmove, ponder

    def quit(self):
        if self.process:
            self._send("quit")
            time.sleep(0.2)
            try:
                self.process.terminate()
            except Exception:
                pass
            self.process = None

    def __del__(self):
        self.quit()

    @staticmethod
    def default_engine_path() -> Optional[str]:
        engine_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engines")
        if not os.path.exists(engine_dir):
            return None
        candidates = sorted(os.listdir(engine_dir))

        if sys.platform == "win32":
            # Windows: 匹配 .exe 文件，优先 bmi2 变体
            for f in candidates:
                lower = f.lower()
                if lower.endswith(".exe") and ("pikafish" in lower or "pika" in lower):
                    if "bmi2" in lower:
                        return os.path.join(engine_dir, f)
            for f in candidates:
                lower = f.lower()
                if lower.endswith(".exe") and "pikafish" in lower:
                    return os.path.join(engine_dir, f)
        else:
            # macOS / Linux: 匹配非 .exe 的 pikafish 二进制文件
            # 优先选择不带指令集后缀的通用版本，其次 bmi2、avx2
            priorities = ["pikafish", "bmi2", "avx2"]
            for priority in priorities:
                for f in candidates:
                    lower = f.lower()
                    if lower.endswith(".exe") or lower.endswith(".nnue"):
                        continue
                    if "pikafish" in lower and priority in lower:
                        path = os.path.join(engine_dir, f)
                        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
                        return path
            # 兜底：任意包含 pikafish 的非 .exe 文件
            for f in candidates:
                lower = f.lower()
                if lower.endswith(".exe") or lower.endswith(".nnue"):
                    continue
                if "pikafish" in lower:
                    path = os.path.join(engine_dir, f)
                    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
                    return path
        return None
