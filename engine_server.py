"""
揭棋引擎服务端 - 常驻进程，通过 stdin/stdout 逐行 JSON 通信。
用 PyPy 运行可获得 4~5 倍加速；也可用 CPython 运行 (回退)。

协议 (每行一个 JSON 对象):
  请求  <- stdin:  {"cmd": "go", "board": [[...]], "my_side": "r", "think_time": 2.0}
                   {"cmd": "ping"}
                   {"cmd": "quit"}
  响应  -> stdout: {"ok": true, "uci": "b2b9", "score": 154, "depth": 5}
                   {"ok": true, "pong": true}
                   {"ok": false, "error": "..."}

注意: 所有非协议输出 (预热日志等) 一律走 stderr，保持 stdout 纯净只放 JSON。
"""
import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jieqi_engine import JieQiEngine


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def main():
    engine = JieQiEngine()
    runtime = "PyPy" if hasattr(sys, "pypy_version_info") else "CPython"
    _log(f"[engine_server] 就绪 ({runtime} {sys.version.split()[0]})")

    # JIT 预热 (PyPy 首次搜索有编译开销，先跑一次让热点编译)
    try:
        warm = [["."] * 9 for _ in range(10)]
        for c in range(9):
            warm[9][c] = "r帥" if c == 4 else "r?"
            warm[0][c] = "b將" if c == 4 else "b?"
        for c in (0, 2, 4, 6, 8):
            warm[6][c] = "r?"; warm[3][c] = "b?"
        engine.get_best_move(warm, "r", think_time=0.5)
        _log("[engine_server] 预热完成")
    except Exception as e:
        _log(f"[engine_server] 预热跳过: {e}")

    # 就绪信号 (客户端可据此确认服务端已可用)
    print(json.dumps({"ok": True, "ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            print(json.dumps({"ok": False, "error": f"bad json: {e}"}), flush=True)
            continue

        cmd = req.get("cmd", "go")
        if cmd == "quit":
            break
        if cmd == "ping":
            print(json.dumps({"ok": True, "pong": True}), flush=True)
            continue

        try:
            board = req["board"]
            my_side = req["my_side"]
            think_time = float(req.get("think_time", 2.0))
            uci, score, depth = engine.get_best_move(board, my_side, think_time=think_time)
            print(json.dumps({"ok": True, "uci": uci, "score": score, "depth": depth}), flush=True)
        except Exception as e:
            import traceback
            _log(traceback.format_exc())
            print(json.dumps({"ok": False, "error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
