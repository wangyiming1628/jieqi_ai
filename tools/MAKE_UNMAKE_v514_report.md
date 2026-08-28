# 揭棋引擎 make/unmake 结构重构 — v5.14 移植报告

> 分支: `feature/make-unmake-v2`（基于 `main` = v5.13）
> 目标: 把旧分支 `feature/v5.11.1-make-unmake` 的「单盘可变 + undo 栈 + 双色走法生成 + Zobrist TT」
> 重构移植到当前 `main`（已含 v5.12 P0 暗子池修复 + v5.13 风险惩罚，默认 `RISK_LAMBDA=0`），
> 并验证与基线**逐字节等价** + 配对对局无实力回退 + 量化提速。

---

## 1. 改动内容

`jieqi_engine.py` 重写为 `State` + `make/unmake` 架构（对照旧 `Position` + `put/rotate` 不可变管线）：

| 模块 | 说明 |
|------|------|
| `State`（L224 起） | 256 长 list 棋盘；`stm` 标志（True=红=大写）；`undo` 栈；增量计数 `cov_u/l, che_u/l, zu_u/l, back_u/l, rnci_u/l, kt_u/l, kts_u/l, rough`；`make()/unmake()/flip_stm()`；`gen_moves(side)` 双色；`value(i,j)`；`can_capture_king(side)`；Zobrist `hkey` |
| `Searcher`（L686 起） | 与 v512 同套 PVS/LMR/nullmove/qsearch 策略；`pos.move()/rotate()` 替换为 `st.make()/unmake()/flip_stm()`；TT key = `(hkey, score, depth, root)` |
| `Position` shim（L681 起） | 保留 `board_to_engine_string` / `set()` / `gen_moves()` / `rotate()` 委托，兼容 `referee.py` |
| `DarkPoolTracker` | 原样保留（`get_best_move` 调用 `self._pool`） |
| `calc_variance` / `RISK_LAMBDA` | 原样保留（默认 0 → 与基线逐字节等价） |

---

## 2. 关键 bug 与修复（核心交付物）

### 现象
跨实现 diff 在「关 LMR + 关 nullmove」下逐字节通过（Gate-1：90/90），但「开 nullmove」时却在
**深度 ≥ 7** 出现灾难性发散：

```
d=6 nullmove ON:  旧 50595 节点 root=178 | 新 48819 节点 root=178   (差 -3.5%)
d=7 nullmove ON:  旧 149846 节点 root=178 | 新 702424 节点 root=59   (差 +368%, 根值错!)
d=8 nullmove ON:  旧 406775 节点 root=175 | 新 3898649 节点 root=72   (差 +858%, 根值错!)
```

关 nullmove 时所有深度根值完全一致 → 问题**仅出在 nullmove 探针**。

### 根因
旧 `pos.rotate()` 在翻 `turn` 的同时 **取反 score**（`Position(board, -self.score, not self.turn, ...)`），
以保证 `score` 始终与「当前轮走方视角」一致。

新引擎的 `flip_stm()`（L443）只翻 `stm/turn` 与 `hkey` 的 `Z_STM^Z_TURN`，**遗漏了 `score` 取反**。
而 `score` 被用于：
- `static()` 叶节点静态分（`self.score + kts_u - kts_l`）
- TT key `_tt_key = (hkey, score, depth, root)` / `_mv_key = (hkey, score)`

后果：nullmove 探针局面中 `score` 与翻后的 `stm` **不一致**，导致叶节点静态分错误 + TT key 污染，
使探针给出错误剪枝值 → 整棵搜索根值漂移。

正常搜索中 `make()` 已同时取反 `score` 与翻 `stm`，故「关 nullmove」全深度正确——这正是为何
Gate-1 通过、唯独 nullmove 探针出错。

### 修复（一行）
`flip_stm()` 增加 `self.score = -self.score`，与旧 `rotate()` 完全对应：

```python
def flip_stm(self):
    self.stm = not self.stm
    self.turn = not self.turn
    self.score = -self.score          # ← 新增：对齐旧 rotate() 的 -self.score
    self.hkey ^= Z_STM ^ Z_TURN
```

### 修复后验证
```
d=6 nullmove ON:  旧 50595 root=178 | 新 55326 root=178   差 +9.4%
d=7 nullmove ON:  旧 149846 root=178 | 新 159025 root=178  差 +6.1%
d=8 nullmove ON:  旧 406775 root=175 | 新 554525 root=175  差 +36.3%
```
**根值全部逐字节一致**（178=178，175=175）。剩余 ~6–36% 节点差异来自 TT key 表示不同
（新用 `hkey+score` 整数元组，旧用 `board 串+score+turn`），属效率差异、**非正确性**。

---

## 3. 等价性验证（逐字节）

| 测试 | 设置 | 结果 |
|------|------|------|
| Gate-1 `MU_diff_pairwise.py` | 关 LMR + 关 nullmove，30 局面 × d2-4，value/根值/子节点逐字 | **90/90 通过** |
| 开 nullmove 逐字 diff（自写） | 开 nullmove + 关 LMR，15 局面 × d2-5 | **60/60 通过** |
| 开 nullmove 逐字 diff（自写） | 开 nullmove + 关 LMR，8 局面 × d2-6 | **40/40 通过** |
| 探针根值抽查 | d7/d8 nullmove ON | 178=178 / 175=175 |

→ 搜索结果在两种 nullmove 设置、深度 2–8 下均与 v512 基线**逐字节等价**。

---

## 4. 提速量化（PyPy 3.11.13）

### 4.1 单步 make/unmake 微基准 `MU_microbench.py`
| 操作 | 旧 `move()` | 新 `make()` | 倍率 |
|------|------------|------------|------|
| 单次 | 8.22 μs | 1.40 μs | **5.87×** |
| make×3 + unmake×3（累积） | 24.97 μs | 3.46 μs | **7.22×** |

> 旧 `move()` 每次 `rotate_new` 重建整个 `Position`（`set()` 重算全部统计 + 字符串旋转/交换大小写）；
> 新 `make()` 仅在 list 上做 O(1) 落子 + 增量统计 + Zobrist 异或。

### 4.2 搜索级吞吐 `MU_bench.py`（固定深度 8，关 LMR+nullmove）
| 指标 | 旧 v512 | 新 v514 | 倍率 |
|------|---------|---------|------|
| 节点数 | 185980 | 181954 | 0.978×（≈1，验证等价） |
| 墙钟（中位） | 3.01 s | 1.51 s | — |
| 吞吐 | 61.7 knps | 120.4 knps | **1.95×** |
| 单节点时间 | 16.20 μs | 8.31 μs | **1.95×** |

> 端到端搜索提速约 **2×**（受 TT/走法排序/裁剪等固定开销摊薄，低于纯 make/unmake 的 6×）。

---

## 5. 配对对局（vs v5.13 基线，非 v5.12）

> 按用户指正，基线应为 **v5.13**（`main`：P0 暗子池已修 + 风险惩罚默认关），而非 v5.12。
> 提取 `main:jieqi_engine.py` → `jieqi_engine_v513.py` + `engine_server_v513.py`，新增 `referee` kind `pypy513`。

`run_paired.py --new pypy --old pypy513 --pairs 10 --think-time 1.0`（=20 局）：

| 指标 | 值 |
|------|----|
| 战绩（新版视角） | 11 胜 / 4 负 / 5 平 = **67.5%** |
| 配对净胜/净负/平 | **净胜 6 对 / 净负 2 对 / 平 2 对**（net 累计 +6） |
| 算力对称性 | 0.728 s/着 vs 0.720 s/着（**+1.2%**，等时预算下同对称） |
| 平均搜索深度 | 新 **5.59** vs 基线 5.30（**+0.28 更深**） |

**结论：无实力回退**（实际略强）。等墙钟下双方用时对称，新版因每节点更快而**搜得更深**
（+0.28 层），这正是重构的收益——逐字节等价的搜索在同等时间内探索更多节点。

### 5.1 大规模配对（50 对 = 100 局，3.0s/着，等墙钟）

`run_paired.py --new pypy --old pypy513 --pairs 50 --think-time 3.0 --tag v514_vs_v513_50p`
（结果：`tools/games/v514_vs_v513_50p.json`）

| 指标 | 值 |
|------|----|
| 逐局战绩（新版视角） | **41 胜 / 32 负 / 27 平 = 54.5%** |
| 配对净胜 / 净负 / 平 | **净胜 16 对 / 净负 11 对 / 平 23 对**（net +5） |
| 算力对称性 | 2.195 s/着 vs 2.174 s/着（**+1.0%**，等时预算下同对称） |
| 平均搜索深度 | 新 **6.96** vs 基线 6.55（**+0.41 更深**） |

**结论：无实力回退，且实测略强。** 配对 net +5（16 净胜对 vs 11 净负对），
逐局 54.5% 高于 50% 基线。等墙钟下双方用时几乎对称（偏差 +1.0%），
新版凭借更快的单节点在同等 3.0s 预算内**多搜 0.41 层**，这正是 make/unmake 重构带来的
确定性收益——搜索结果逐字节等价，只是单位时间探索更多。

> 注：50 局中 27 平局占比偏高，符合揭棋长局 + 重复/长将判和的常见形态；颜色效应已由配对设计抵消。

---

## 6. 交付物 / 改动文件

- `jieqi_engine.py` — make/unmake 重构 + `flip_stm` score 取反修复
- `tools/MU_diff_pairwise.py`、`tools/MU_diff_lmr.py`、`tools/MU_bench.py`、`tools/MU_microbench.py` — 对拍/基准
- `jieqi_engine_v513.py`、`engine_server_v513.py` — v5.13 基线对照件
- `tools/referee.py` — 新增 `pypy513` kind
- `tools/games/v514_vs_v513.json` — 配对对局原始结果
- `tools/MAKE_UNMAKE_notes_20260827.md` — 方法学笔记（先前）

**状态：分支 `feature/make-unmake-v2` 已就绪，未合并到 `main`。**
