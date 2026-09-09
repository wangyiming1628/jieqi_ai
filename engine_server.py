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
import sys, os, json, importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def main(engine_module="jieqi_engine"):
    """engine_module: 引擎模块名, 默认 jieqi_engine。"""
    mod = importlib.import_module(engine_module)
    # 可选环境变量: 引擎旋钮注入 (A/B 消融实验用)。
    #   引擎若提供对应 setter 才生效, 否则静默忽略 —— 这样同一份服务端可以驱动
    #   不同版本的引擎, 无需为每个实验分支改服务端。
    #   目前约定的旋钮 (见各特性分支):
    #     JIEQI_DETERMINIZE=weighted|pessimistic  -> set_determinize_mode
    #     JIEQI_DET_SIDES=both|mine|oppo          -> set_determinize_sides
    _knobs = (("JIEQI_DETERMINIZE", "set_determinize_mode"),
              ("JIEQI_DET_SIDES", "set_determinize_sides"))
    _applied = []
    for _env, _setter in _knobs:
        _val = os.environ.get(_env)
        if _val and hasattr(mod, _setter):
            getattr(mod, _setter)(_val)
            _applied.append(f"{_setter.split('_', 1)[1]}={_val}")
    engine = mod.JieQiEngine()
    runtime = "PyPy" if hasattr(sys, "pypy_version_info") else "CPython"
    _extra = (", " + ", ".join(_applied)) if _applied else ""
    _log(f"[engine_server] 就绪 ({runtime} {sys.version.split()[0]}, "
         f"engine={engine_module}{_extra})")

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
            # 可选: 局面历史 (最近一次吃子以来) 与已方连续将军计数状态,
            # 供引擎做重复/长将规避
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
    # 可选命令行参数: 引擎模块名 (默认 jieqi_engine)
    main(sys.argv[1] if len(sys.argv) > 1 else "jieqi_engine")
