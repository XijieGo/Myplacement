# Abacus 合法化：最小位移、可复现对照与边界

## 为什么做它

全局布局已经对线长、密度和（可选的）拥塞代理做过优化。合法化若只是先把单元贪心分配到某行、最后再
逐行打包，早期进入一行的单元不会因后来单元而重新调整，容易在最终布局引入不必要的位移和 HPWL 损失。

本项目原有的 `greedy` 路径正是这样的“行分配 + isotonic 打包”基线。它保留在代码中，供每次 QoR
比较使用；它不是论文中的 canonical Tetris 实现，因此下面的实验只声称“相对项目旧基线”的收益。

## 调研依据与适配范围

算法依据是 Spindler、Schlichtmann、Johannes 的
[Abacus: Fast Legalization of Standard Cell Circuits with Minimal Movement](https://portal.fis.tum.de/en/publications/abacus-fast-legalization-of-standard-cell-circuits-with-minimal-m/)
（ISPD 2008；课程资料中也保存了原始 PDF）。其关键不是简单按行塞单元，而是：

1. 按全局 x 坐标处理标准单元；
2. 临时插入每个候选行，再以 cluster 动态规划重放该行；
3. 对相交 cluster 合并，并以加权平方位移最优位置
   `x = clamp(q / e, xmin, xmax - w)` 放置；
4. 选择当前单元位移最小的候选行。论文还建议比较正、反两个 x 排序方向。

我也检查了 [OpenROAD 的 DPL 文档](https://github.com/The-OpenROAD-Project/OpenROAD/blob/master/src/dpl/README.md)
来确认工程侧需要考虑行碎片、混合高度单元和合法性检查；本实现没有复制 OpenROAD 代码。当前迁移范围刻意
限定为**标准单元**：已有宏单元预处理保留不变，固定单元和已合法的宏会先切分出无障碍的 `RowSlot`，再在
每个 slot 内执行 Abacus。

## 实现要点

- 默认策略变为 `LegalizationStrategy::Abacus`；`greedy`/`legacy`/`isotonic` 仍可作为 A/B 基线。
- 每个 slot 保存 append-only 的 cluster 状态 `(first_entry, e, w, q, x)`。候选试插只重建会与新单元
  相交的 cluster 后缀，不复制整行，也不在试插阶段改变数据库。
- 最终 x 坐标会对齐该行 site 栅格；单元宽度按 site spacing 向上取整，因而 slot 边界、相邻单元和
  输出坐标都保持可制造的行约束。
- 候选行由最近行向上下展开，使用“只移动垂直方向”的非负代价作下界；局部范围无容量时才回退扫描全部行。
- 默认执行正、反 x 两个 pass，按标准单元总欧氏位移选择较优者；`overview.txt` 会记录
  `legalization_reverse_pass_selected`。
- 新增 `standard_cell_total_displacement`、加权平方位移、最大位移和合法化耗时，避免只看最终 HPWL。

相关入口：

```text
--legalizer abacus     # 默认
--legalizer greedy     # 旧的 greedy + isotonic 基线
```

受控单元测试位于 `tests/LegalizerTests.cpp`：两个宽度为 10 的重叠单元从 x=`6, 8` 进入一条 30-site
行后，Abacus 的 cluster collapse 得到 x=`2, 12`。这验证了“后插单元会重新移动先前已插入单元”的核心
行为，而不仅是检查最后不重叠。

## 同条件 A/B 实验（2026-09-02）

协议：BookShelf 内置 `adaptec1`（A1）和 `adaptec4`（A4）；`quadratic` 初始布局、`seed=2026`、
Neumann、`64x64` bins、CPU、关闭 BMP/GDS/详细放置。A1 使用 140 次全局迭代上限，A4 使用 280 次。
两种策略除 `--legalizer` 外参数完全一致。

两次运行的 `global_history.csv` 均逐字节一致，故全局阶段没有成为混杂变量：

- A1 SHA-256：`3f3eae65d53c04ad7b498feb23437d77ac189d1acce3bb4a13a8538a4a554825`
- A4 SHA-256：`554fa6552bd857a3669deb1cc8281c6eb5e49705f4df24b48b27496d1cfc694a`

| 指标 | A1 Greedy | A1 Abacus | A1 变化 | A4 Greedy | A4 Abacus | A4 变化 |
|---|---:|---:|---:|---:|---:|---:|
| 最终 HPWL | 145,351,310 | 123,505,082 | **-15.030%** | 1,244,354,029 | 1,142,521,511 | **-8.184%** |
| 标准单元总位移 | 25,328,269.63 | 9,101,986.20 | **-64.064%** | 143,301,242.79 | 83,078,510.82 | **-42.025%** |
| 加权平方位移 | 1.860e12 | 1.579e11 | **-91.508%** | 3.498e13 | 1.088e13 | **-68.890%** |
| 合法化耗时 | 0.275 s | 1.447 s | 5.264x | 0.913 s | 3.384 s | 3.705x |
| 合法性 | PASS / 0 / 0 / 0 | PASS / 0 / 0 / 0 | — | PASS / 0 / 0 / 0 | PASS / 0 / 0 / 0 | — |

“合法性”的四项依次是 `legal / overlap_pairs / off_row_modules / unplaced_standard_cells`。两例均选择了
反向 x pass。A4 的自适应全局器在第 255 次已停止，最佳检查点仍为第 244 次；这两项也在对照中一致。

### 必须如实说明的 trade-off

Abacus 的直接目标是最小位移，不是密度再优化。因此不应只挑 HPWL 报告：

- A1 合法化后的 `normalized_overflow` 为 `0.0430756 -> 0.0442364`（+2.695%）；
- A4 为 `0.0431739 -> 0.0431391`（-0.081%，基本持平）。

也就是说，这次改动已在两个大基准上稳定改善最终 HPWL 与位移，但 A1 的课程密度指标有小幅回退。
若后续把“合法化后 overflow”设成硬门槛，应按 HPWL、overflow、RUDY 留出代理一起筛选，不能无条件用
Abacus 覆盖全部正式基线。这也是将旧策略保留为命令行对照，而不是删除它的原因。

## 复跑

脚本会拒绝比较全局历史不同的两次运行：

```bash
./scripts/run_abacus_ablation.sh \
  'resources/course_materials/任务2参考/BookShelf格式的解析/adaptec1/adaptec1.aux' \
  140 out/scratch/abacus-a1

./scripts/run_abacus_ablation.sh \
  'resources/course_materials/任务2参考/BookShelf格式的解析/adaptec4/adaptec4.aux' \
  280 out/scratch/abacus-a4
```

常规正确性检查：

```bash
cmake --build build -j 8
ctest --test-dir build --output-on-failure
```

本次原始运行输出在 `out/scratch/abacus-legalization-study-20260902/`；`out/scratch` 是非交付目录，
正式报告需要在独立复跑后再冻结到 `out/verified`。
