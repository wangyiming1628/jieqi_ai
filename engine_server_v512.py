"""
v5.12 基线引擎服务端 (P1 U解冻对照用) - 加载 P1 之前的 jieqi_engine 快照
(jieqi_engine_v512.py, 取自 git d88406c = v5.12 P0 暗子池跟踪)。
协议与 engine_server.py 一致。
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine_server import main  # noqa: E402

if __name__ == "__main__":
    main("jieqi_engine_v512")
