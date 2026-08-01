"""
揭棋 AI — 天天象棋自动连线程序
================================
支持的平台: Windows / macOS

使用步骤:
  1. 安装依赖:
     pip install -r requirements.txt

  2. 采集棋子模板（首次使用必须）:
     python capture_templates.py
     → 打开天天象棋，进入揭棋对局，按提示采集每个棋子模板

  3. 下载引擎:
     将皮卡鱼(pikafish.exe)放入 engines/ 目录
          或象眼(ElephantEye.exe)放入 engines/ 目录

  4. 运行 AI:
     python main.py
     → 打开天天象棋揭棋对局，程序自动识别并走棋

模块说明:
  - main.py              程序入口
  - controller.py        窗口连接 + 截图 + 鼠标操控
  - board_detector.py    OpenCV 棋盘检测 + 棋子模板匹配
  - game_state.py        揭棋状态管理 + 暗棋候选追踪
  - engine.py            UCCI 协议引擎封装
  - capture_templates.py 棋子模板采集工具
  - assets/              棋子模板图片
  - engines/             象棋引擎文件
"""
