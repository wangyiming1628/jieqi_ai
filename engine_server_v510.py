"""
v5.10 基线引擎服务端 (对局对照用) - 加载修复前 jieqi_engine (git HEAD 快照 jieqi_engine_v510.py)。
协议与 engine_server.py 一致。
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine_server import main  # noqa: E402

if __name__ == "__main__":
    main("jieqi_engine_v510")
