"""
v5.7 基线引擎服务端 (对局对照用) - 加载 git e425620 的 jieqi_engine (无重复感知)。
协议与 engine_server.py 一致; 旧引擎不认识 pos_history 参数, 此处用适配器吞掉。
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jieqi_engine_v57 as m

_orig = m.JieQiEngine.get_best_move


def _patched(self, board, my_side, think_time=2.0, pos_history=None, check_state=None):
    return _orig(self, board, my_side, think_time=think_time)


m.JieQiEngine.get_best_move = _patched

from engine_server import main  # noqa: E402

if __name__ == "__main__":
    main("jieqi_engine_v57")
