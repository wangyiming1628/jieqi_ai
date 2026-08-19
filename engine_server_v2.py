"""
优化版引擎服务端 - 加载 jieqi_engine_v2 (搜索优化: TT 不清空 + 双时限去上限)。
协议与 engine_server.py 完全一致 (stdin/stdout 逐行 JSON), 仅加载的引擎模块不同。
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_server import main

if __name__ == "__main__":
    main("jieqi_engine_v2")
