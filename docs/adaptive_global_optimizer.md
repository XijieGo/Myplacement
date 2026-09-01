# 自适应闭环全局布局器

默认的 `adaptive` 优化器将全局布局看作一个受反馈控制的连续优化过程，而不是按固定日程
反复乘以某个参数。它仍然优化同一个核心目标：

```text
f(x, y) = Wγ(x, y) + λ Eρ(x, y)
```

- `Wγ` 是 log-sum-exp 平滑线长；`γ` 越小，越接近真实 HPWL。
- `Eρ` 是 DCT/Neumann 静电势能；密度过高的位置会产生把单元推开的梯度。其电荷、溢出率和
  分母统一采用 [`course_eplace_v1`](density_model.md) 的分层定义。
- `λ` 决定此刻“降低拥挤”相对于“缩短连线”的权重。

它与旧 `legacy` 模式共存，因而任何结论都可以用同一输入、种子、初始布局和迭代上限重跑
验证。

## 一次迭代如何闭环

1. 先测量真实 HPWL、平滑线长、带 filler 的优化密度、去掉 filler 的真实设计密度和静电能。
2. 根据当前溢出率决定平滑度：拥挤时使用较大的 `γ`，布局接近可行时逐渐降到较小的 `γ`。
   因此它跟随布局状态，而不跟随“已经跑到第几轮”。
3. 以相邻状态的位移 `s` 和预条件梯度变化 `y` 估计局部曲率
   `L ≈ ||y|| / ||s||`，再取步长约为 `1/L`。如果没有可靠历史，使用由期望平均移动距离
   推出的初始步长。
4. 每个单元的梯度按引脚数、面积和当前密度权重缩放。高扇出或大单元不会再单独把全局步长
   压得极小。
5. 生成 Nesterov 前瞻候选解；若动量与负梯度相冲突，或前瞻目标明显变坏，清空动量并从当前点
   重新尝试。
6. 候选解必须满足 Armijo 型下降条件；仅当溢出有实质改善时，才允许很小的目标上升作为密度
   交换。否则按比例回溯缩短步长并重算，最多 `maximum_backtracks` 次。
7. 根据本轮 HPWL 增长和溢出改善动态改变 `λ`。密度长期停滞时增强排斥力；密度已经可控但
   HPWL 明显恶化时降低它。参数变化后会丢弃过期的曲率历史，避免将不同目标函数的梯度混用。
8. 保存“达到真实设计密度约束的所有状态中 HPWL 最小者”；若尚不可行，则保存溢出最低者。
   结束时恢复该检查点，而不是盲目输出最后一次迭代。

`GlobalPlacementOptions` 中的主要安全/控制参数都位于
[`include/myplacement/placement/GlobalPlacer.hpp`](../include/myplacement/placement/GlobalPlacer.hpp)：
回溯次数、回溯比例、最大动量、密度惩罚上下界、溢出平滑指数和可行后的细化轮数均可由 C++
调用方调整。命令行默认选择 `adaptive`；`--global-optimizer legacy` 可选择原来的开环版本。

## 诊断文件

`global_history.csv` 在自适应模式下每轮写一行，字段如下：

```text
iteration, hpwl, optimizer_overflow, design_overflow,
smooth_wirelength, density_energy, objective, penalty, smoothing,
step_size, maximum_displacement, gradient_norm, curvature, backtracks,
momentum_restarted, accepted, best_checkpoint
```

其中 `optimizer_overflow` 包含 filler，反映优化器实际看到的空间压力；`design_overflow` 排除
filler，反映最终真实单元的密度。二者都保留固定端子和暗区的密度障碍，也都只用标准单元与
目标密度缩放后的可移动宏块作为分母，详见 [density_model.md](density_model.md)。`overview.txt`
还汇总接受/拒绝次数、动量重启数、恢复的最佳检查点以及溢出分子/分母。这样可以直接回答一次
性能变化来自何处，而不是只比较最后一个 HPWL。

## 可复现实验

下面的脚本会在同一资源封套内运行开环/闭环 A/B：最多 16 个逻辑核、16 GiB 地址空间，且不
启用 BMP/GDSII 导出时尤其适合大基准。

```bash
scripts/run_optimizer_ablation.sh \
  build \
  'resources/course_materials/任务2参考/BookShelf格式的解析/adaptec1/adaptec1.aux' \
  out/scratch/adaptec1-adaptive-vs-legacy-160 \
  --initial quadratic --iterations 160 --bins 64 --seed 2026 \
  --density-field neumann --no-bmp --no-gds
```

### 指标版本与复现实验

`overview.txt` 的 `density_metric=course_eplace_v1` 是数值是否可比较的前提。项目中早于该字段的
历史输出使用了旧分母：它把固定端子面积混入归一化分母，因此不能再用来证明 0.10 课程约束或与
新运行作数值比较。保留这些输出仅供排查开发过程，正式 A/B 结论必须用上面的脚本重新执行。

修正后的 `adaptec1` 复测（`quadratic` 初始、64×64、160 轮上限、Neumann、seed 2026）在第 140
轮选择全局检查点：HPWL 为 `116,932,383.577`，真实设计溢出率为 `0.0918172`。经同一合法化器后，
布局合法，最终 HPWL 为 `145,351,310`，真实设计溢出率为 `0.0430756`。这组数值说明修正并非只改
报表：检查点和停止规则实际依照同一门槛工作。

大基准仍应分别报告“给定迭代预算下的结果”和“是否达到密度门槛”。例如 `adaptec4` 在相同模型的
160 轮压力复测中设计溢出率为 `0.141372`，尚不满足 0.10；不能把旧口径的较小数字当作已经达标。
将预算提升到默认的 280 后，运行在第 244 轮保存 HPWL `1,064,016,726.86`、设计溢出率
`0.0995075` 的最佳可行检查点，并在第 255 轮早停。经合法化后布局完全合法，最终 HPWL
`1,244,354,029`、设计溢出率 `0.0431739`。因此 280 不是为了盲目多跑：它覆盖了该规模设计达到
课程约束所需的过渡段，而可行后的细化和早停仍限制了实际工作量。
