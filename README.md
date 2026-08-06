"""
揭棋 AI — 天天象棋自动识别与走子
================================
一个学习交流用的 AI 项目：自动识别揭棋棋盘 → 调用象棋引擎算最佳着法 → 模拟鼠标走子。
当前主流程在 macOS 上运行（截图与窗口操控为 macOS 专属实现）。

整体流程:
  截图 → YOLO+OCR 识别棋盘 → 生成 FEN → 引擎计算着法 → 模拟点击走子 → 循环

使用步骤:
  1. 安装依赖:
     pip install -r requirements.txt

  2. 准备模型与引擎（放入 engines/ 目录）:
     - online_xiangqi_piece_detector.onnx   棋子检测模型
     - online_xiangqi_classifier.onnx        棋子分类模型
     - xiangqi-ai                            揭棋着法引擎（best --fen ... --json）

  3. 运行:
     python main.py
     → 打开天天象棋揭棋对局，程序检测到轮到己方时自动识别并走棋

模块说明:
  - main.py              程序入口：状态框检测 + FEN 生成 + 坐标换算 + 自动走子
  - board_recognizer.py  棋盘识别：YOLO 检测 + PaddleOCR 文字识别 + 暗子阵营/位置校验
  - yolo_detector.py     YOLO 双模型（检测器 + 分类器）ONNX 推理
  - engines/             ONNX 模型 + 象棋引擎二进制
"""
