import os

# 风险惩罚实验服务: 在 v5.12 引擎上开启确定性方差风险惩罚 (均值-方差效用)。
# λ=0.5 为 run_paired 网格消融 (0/0.2/0.35/0.5/0.6/0.75) 实测最优档
# (20局配对净胜4:1, 得分62.5%; 见 tools/RISK_PENALTY_report_20260827.md)。
# JIEQI_RISK_LAMBDA 可被外部环境变量覆盖。
os.environ["JIEQI_RISK_LAMBDA"] = os.environ.get("JIEQI_RISK_LAMBDA", "0.5")

import engine_server

if __name__ == "__main__":
    engine_server.main("jieqi_engine")
