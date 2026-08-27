# make/unmake 重构: 现状诊断笔记 (留给下次续做)

日期: 2026-08-27
背景: `feature/v5.11.1-make-unmake` 分支 (commit `e5b68a2`) 之前尝试过一次结构级重构
(单一可变棋盘 + make/unmake + undo栈 + 双色走法生成 + Zobrist 置换表), 因对拍失败
被搁置未合并。本次重新诊断, 结论: **旧分支的核心实现质量是过关的, 失败原因跟
make/unmake 本身无关**, 下次续做时不需要重新怀疑这套设计, 直接移植+对齐即可。

## 结论一句话版

旧分支能用, 之前对拍失败是三个跟 make/unmake 无关的原因造成的假象:
1. P0 暗子池算法新旧不一致 (分叉时间点问题)
2. 诊断脚本自己的 bug (`sumall` 没跟 `di` 同步更新)
3. LMR 近似搜索的正常统计噪声被误判成"bug"

预计成本: **0.5~1 天** (移植 + 重新对拍), 不是最初估的 1.5~3 天, 因为最难的双色生成
+ 可变棋盘骨架已经写过一次且验证是对的, 剩下是移植体力活。

## 详细排查过程与证据

### 第一层: `sumall` 未同步 (诊断脚本的 bug, 不是引擎的 bug)

`jieqi_engine_v511_1.py` 里 `sumall` 是从模块级全局 `di` 派生的缓存, 只有调用
`_update_distribution()` 时才重算。诊断时为了屏蔽 P0 池算法差异手动覆盖了
`NEW.di[0] = OLD.di[0]`, 忘了同步 `NEW.sumall[0]`, 导致 `possible_che` 计算错误,
产生大量误报。修正后, **150 个随机局面、深度2-3 对拍失败清零** (原 9条噪声+2条真实
分歧 → 0)。

### 第二层: P0 暗子池算法差异 (已知的分叉问题, 非 bug)

旧分支从 P0 修复 (`d88406c`) 之前的 v5.11 分叉, 自带旧版 `_update_distribution`
(单帧反推, 会把被吃明子计回池里)。这在旧分支自己的 commit message 里已经写明预警。
下次移植时必须把 `DarkPoolTracker` (P0 的事件驱动池跟踪) 一起搬过去, 不能直接拿
旧分支代码替换 `jieqi_engine.py`, 否则会静默回退 P0。

同理, 风险惩罚 (`RISK_LAMBDA`/`calc_variance`, v5.13, commit `94b4100`) 也是在旧分支
分叉之后才加的, 移植时要一并搬。

### 第三层 (关键): LMR 不是 bug, 是设计上的近似, 不影响 make/unmake 的可验证性

深度4+ 对拍仍有分歧, 逐层排查定位到 LMR (Late Move Reduction)。做了最小可复现实验:

- 关掉 LMR (`lmr_min_depth=999`) 后, alphabeta 精确等于朴素 minimax (无剪枝的
  ground truth), 逐字相等 (71.0=71.0 深度3, -40=-40 深度4)。**证明 alpha-beta 剪枝 +
  置换表 + 空着裁剪骨架本身是精确的, 不引入误差。**
- 开着 LMR, 同一节点在不同调用场景 (不同 alpha/beta 窗口、不同 TT 命中状态) 会给出
  不同近似值——这是 LMR 设计上就允许发生的正常统计噪声 (用速度换精度的权衡), 不是
  代码缺陷。把 `lmr_base_reduction` 从1改成0 (LMR代码路径照走, 但不削减探测深度),
  结果精确等于朴素 minimax (40=40), **证明 LMR 的探测→重搜升级逻辑本身是对的**,
  差异纯粹来自"探测深度比正常深度浅1层"这一件事本身的固有风险, 不是实现错误。

**LMR 与 make/unmake 之间没有本质冲突**: LMR 影响的是"搜索该不该被信任"(搜索层),
make/unmake 要验证的是"局面表示和增量计算是否正确"(表示层), 两者是独立维度。

**对拍方法学的教训 (下次直接用这个, 不要重复踩坑)**:
1. **先关 LMR (连同空着裁剪) 做对拍, 要求根值逐字相等**——这时两边都是精确
   alpha-beta, 搜索路径完全确定。这一步验证局面表示、增量统计、双色生成、Zobrist
   全部正确。这是最该做、也最有效的第一道闸门。
2. **LMR 开着时, 不要直接比根值**——LMR 的触发条件 (`do_lmr` 判断) 只依赖搜索控制
   变量 (alpha, beta, depth, move_idx, is_cap), 不依赖局面表示方式。只要两边收到
   完全相同的这些参数, LMR 决策序列、递归子节点序列就会完全一致, 根值自然逐字相等。
   如果 LMR 开着时分歧, 说明是别处 (局面表示/增量计算) 有偏差导致某个中间节点算错,
   LMR 只是把这个偏差放大暴露出来, 不是 LMR 本身的问题。可以做"搜索路径逐节点比对"
   (比对每一层调用的 alpha/beta/depth 序列, 而不是只看最终数字) 作为补充验证。

## 下次续做的步骤

1. 建独立分支 (例如 `feature/make-unmake-v2`), 从当前 main 出发。
2. 把 `feature/v5.11.1-make-unmake` 里的 `State` 类 (可变棋盘+increment统计)、
   双色 `gen_moves`、Zobrist 置换表移植到当前 main 的基础上。
3. 移植时补上两块新增内容 (旧分支没有):
   - P0 `DarkPoolTracker` (事件驱动暗子池跟踪, 见 `jieqi_engine.py` 里
     `# [v5.12 P0]` 注释处)
   - 风险惩罚 `RISK_LAMBDA`/`calc_variance` (见 `jieqi_engine.py` L700-748,
     以及 `value()` 里 `if RISK_LAMBDA:` 的两处调用点)
4. 对拍验证, 严格按上面"方法学教训"的两步走 (先关LMR锁定逐字相等, 再开LMR做路径级
   对比), 不要只比根值就下结论。
5. 吞吐基准 (`tools/_tmp_bench_mu.py` 里已经有现成脚本, 旧分支保留着, 可以直接抄)。
6. `run_paired` 等墙钟对局, 确认棋力有无回归/提升。

## 相关文件/分支索引

- `feature/v5.11.1-make-unmake` (commit `e5b68a2`): 旧实现, 保留作为移植底稿,
  未合并、未删除。
- 旧分支里的诊断工具 (`tools/_tmp_diag_mu*.py`, `tools/_tmp_diff_mu.py`,
  `tools/_tmp_bench_mu.py`): 对拍/基准脚本底稿, 移植时可以直接复用改造
  (注意: 用之前先修正 `sumall` 同步问题, 并按上面方法学先关 LMR)。
- `jieqi_engine.py` (当前 main, HEAD 见 `git log`): 移植的目标基础版本,
  含 P0 (`DarkPoolTracker`) + v5.13 风险惩罚 (`RISK_LAMBDA`)。
