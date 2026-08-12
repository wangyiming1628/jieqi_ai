# Makinuohara 揭棋引擎 (抽取自第三方项目)

来源: https://github.com/Makinuohara/2026-jieqi-AI  (Java, 北邮 2026 大作业)

本目录是从该项目的 jieqi-model / jieqi-rules / jieqi-ai 三个 Maven 模块抽取并编译的
引擎最小闭包 (无外部运行时依赖, 仅模块间互相依赖 + JDK 标准库), 用于本项目通过
Java 子进程调用。不含 server/gui/protocol/app 等无关模块。

引擎特点:
  - ExpectiminimaxAgent: 期望极小极大搜索 + 迭代加深 + Alpha-Beta 剪枝 + 置换表 + 静态搜索。
  - 处理暗子不确定性: 用 expectiminimax 对暗子做概率期望, 明确不偷看暗子真身
    (仅用公开信息 + 历史观察重建概率局面), 契合"对手暗子未知"场景。

目录:
  classes/                        编译产物 (JDK 21, javac 直接编译, 无需 Maven)
    edu/bupt/jieqi/model/...       数据模型 (Board/Piece/Position/Move/GameState/...)
    edu/bupt/jieqi/rules/...       规则引擎 (合法着法生成 / 将军判定 / apply)
    edu/bupt/jieqi/ai/...          AI 引擎 (ExpectiminimaxAgent/MaterialEvaluator/...)
    edu/bupt/jieqi/bridge/         本项目自写的桥接 server (EngineBridge)
  bridge/EngineBridge.java        桥接源码 (stdin/stdout JSON 通信 + board 格式转换)

运行入口: java -cp classes edu.bupt.jieqi.bridge.EngineBridge

数据模型:
  引擎坐标: Position(file 0-8, rank 0-9), RED 在 rank0-3(底部), BLACK 在 rank6-9(顶部)。
  本项目己方(board row5-9)统一映射为 RED, 对手映射为 BLACK; 引擎 rank = 9 - board_row。
  暗子(hidden)用初始摆位类型 virtualType 估值; 翻开后 actualType 为真身。

编译方式 (classes/ 不入版本库, 首次使用需自行编译):

  1. 安装 JDK 21:
     brew install openjdk@21

  2. 克隆原仓库 (任意临时目录):
     git clone --depth 1 https://github.com/Makinuohara/2026-jieqi-AI /tmp/jieqi-mak

  3. 编译引擎三模块 + 本项目桥接:
     JAVA=/opt/homebrew/opt/openjdk@21/bin
     SRC=/tmp/jieqi-mak
     DST=<本项目根>/engines/mak
     $JAVA/javac -d "$DST/classes" $(find \
        "$SRC/jieqi-model/src/main/java" \
        "$SRC/jieqi-rules/src/main/java" \
        "$SRC/jieqi-ai/src/main/java" -name '*.java')
     $JAVA/javac -cp "$DST/classes" -d "$DST/classes" "$DST/bridge/EngineBridge.java"

  说明: 三模块无外部运行时依赖(仅模块间互相依赖 + JDK 标准库), 故用 javac 直接编译,
       无需 Maven。编译产物约 38 个 .class / 208KB。

  4. 验证:
     $JAVA/java -cp "$DST/classes" edu.bupt.jieqi.bridge.EngineBridge
     (应输出 {"ok":true,"ready":true}, Ctrl-C 退出)

注: 本引擎为可选备选引擎, 默认引擎是 miaosiSari(PyPy)。若未编译 classes/,
    engine_client 会在选择 "java" 时自动回退到默认引擎, 不影响程序运行。
