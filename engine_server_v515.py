"""
v5.15 基线引擎服务端 (方案 A 二次搜索对照用) - 加载实现方案 A 前的 jieqi_engine 快照
(jieqi_engine_v515.py)。协议与 engine_server.py 一致。
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine_server import main  # noqa: E402

if __name__ == "__main__":
    main("jieqi_engine_v515")
