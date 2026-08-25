"""
v5.11 基线引擎服务端 (P0 暗子池对照用) - 加载 P0 修复前的 jieqi_engine 快照
(jieqi_engine_v511.py, 取自 git 8ca26ed)。协议与 engine_server.py 一致。
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine_server import main  # noqa: E402

if __name__ == "__main__":
    main("jieqi_engine_v511")
