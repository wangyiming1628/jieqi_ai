"""
揭棋 AI — 天天象棋自动识别与走子
================================
一个学习交流用的 AI 项目：自动识别揭棋棋盘 → 纯算法揭棋引擎算最佳着法 → 模拟鼠标走子。
当前主流程在 macOS 上运行（截图与窗口操控为 macOS 专属实现）。

整体流程:
  截图 → YOLO+OCR 识别棋盘 → 揭棋引擎计算着法 → 模拟点击走子 → 循环

使用步骤:
  1. 安装依赖:
     pip install -r requirements.txt

  2. 准备识别模型（放入 engines/ 目录）:
     - online_xiangqi_piece_detector.onnx   棋子检测模型
     - online_xiangqi_classifier.onnx        棋子分类模型
     （揭棋引擎为纯算法实现，无需任何权重文件）

  3. 运行:
     python main.py
     → 打开天天象棋揭棋对局，程序检测到轮到己方时自动识别并走棋

模块说明:
  - main.py              程序入口：状态框检测 + 坐标换算 + 自动走子
  - jieqi_engine.py      揭棋引擎：PST 评估 + alpha-beta/PVS + 静态搜索 + 空着裁剪（基于 miaosiSari/Jieqi，无需 NNUE）
  - board/               引擎依赖：PST 评估表(common_20210815) + 开局库(library)
  - board_recognizer.py  棋盘识别：YOLO 检测 + PaddleOCR 文字识别 + 暗子阵营/位置校验
  - yolo_detector.py     YOLO 双模型（检测器 + 分类器）ONNX 推理
  - engines/             YOLO ONNX 识别模型
"""
