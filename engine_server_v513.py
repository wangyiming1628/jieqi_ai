"""揭棋引擎服务 - v5.13 基线 (main, P0 暗子池已修 + 风险惩罚默认关闭, 不可变 Position 管线)。

与 engine_server.py 同协议, 仅加载 jieqi_engine_v513 (main 版本) 用于配对对照。
"""
import sys, os, json, importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def main(engine_module="jieqi_engine_v513"):
    mod = importlib.import_module(engine_module)
    _knobs = (("JIEQI_DETERMINIZE", "set_determinize_mode"),
              ("JIEQI_DET_SIDES", "set_determinize_sides"),
              ("JIEQI_RISK_LAMBDA", "set_risk_lambda"))
    _applied = []
    for _env, _setter in _knobs:
        _val = os.environ.get(_env)
        if _val and hasattr(mod, _setter):
            getattr(mod, _setter)(_val)
            _applied.append(f"{_setter.split('_', 1)[1]}={_val}")
    engine = mod.JieQiEngine()
    runtime = "PyPy" if hasattr(sys, "pypy_version_info") else "CPython"
    _extra = (", " + ", ".join(_applied)) if _applied else ""
    _log(f"[engine_server_v513] 就绪 ({runtime} {sys.version.split()[0]}, "
         f"engine={engine_module}{_extra})")

    try:
        warm = [["."] * 9 for _ in range(10)]
        for c in range(9):
            warm[9][c] = "r帥" if c == 4 else "r?"
            warm[0][c] = "b將" if c == 4 else "b?"
        for c in (0, 2, 4, 6, 8):
            warm[6][c] = "r?"; warm[3][c] = "b?"
        engine.get_best_move(warm, "r", think_time=0.5)
        _log("[engine_server_v513] 预热完成")
    except Exception as e:
        _log(f"[engine_server_v513] 预热跳过: {e}")

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
            history = req.get("history")
            pos_history = [(b, s) for b, s in history] if history else None
            check_state = req.get("check_state")
            uci, score, depth = engine.get_best_move(
                board, my_side, think_time=think_time,
                pos_history=pos_history, check_state=check_state)
            print(json.dumps({"ok": True, "uci": uci, "score": score, "depth": depth}), flush=True)
        except Exception as e:
            import traceback
            _log(traceback.format_exc())
            print(json.dumps({"ok": False, "error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
