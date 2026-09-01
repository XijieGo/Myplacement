# 课程 ePlace 密度模型

## 为什么要单独定义“密度”

全局布局不是把所有矩形面积机械相加。标准单元可以在连续阶段自由移动；宏、固定端子和行外区域则
是已经占用或不可用的容量。若把固定端子也放进归一化溢出率的分母，会让固定端子很多的基准看起来
“不拥挤”，即使可移动标准单元仍严重重叠。因此本项目将老师讲义中的密度口径作为唯一实现定义，
并在报告中标记为 `course_eplace_v1`。

## 每个 bin 的定义

对一个面积为 `A_bin`、目标密度为 `t` 的 bin，先计算参与静电场的等效电荷面积：

```text
Q_bin = A_standard + t × (A_movable_macro + A_fixed + A_filler + A_dark)
```

- `A_standard`：可移动标准单元的矩形重叠面积，按完整面积计；
- `A_movable_macro`：可移动宏块的重叠面积；
- `A_fixed`：固定端子、固定宏等障碍的重叠面积；
- `A_filler`：只在全局布局内部存在的虚拟 filler 面积；
- `A_dark`：core bounding box 中未被任一 `SiteRow` 覆盖的面积，例如行间空洞或 subrow 缺口。

密度场的源项是 `Q_bin / A_bin - t`。因此暗区本身已经消耗 `t × A_dark` 的容量，移动单元会被
从中推开，而不是只在更新后被坐标夹回 core。

课程要求的归一化溢出率是：

```text
overflow = Σ max(0, Q_bin - t × A_bin)
           / (Σ A_standard + t × Σ A_movable_macro)
```

固定端子、暗区和 filler 会影响分子（即真实拥挤），但**绝不进入分母**。这正是防止固定对象把
指标“稀释”的关键。

## 同一模型如何贯穿流程

`DensityMap` 是唯一存放上述分层面积和计算指标的接口。

```text
BookShelf SiteRows ──> placeable mask ──> dark area
模块 / filler     ──> DensityMap layers ──> Q_bin / overflow
                                           ├─> DCT-Neumann density source
                                           ├─> checkpoint / stopping rule
                                           └─> calculateDensity / overview.txt
```

因此自适应优化器、保留的 `legacy` 基线、检查点选择和最终 `overview.txt` 不会各自采用不同的密度
定义。`clearDynamic()` 只清除标准单元、宏和 filler；固定对象与暗区掩膜在整个全局布局中保持不变。

宏和 filler 的电荷是 `t` 倍物理面积，因此其静电梯度也乘以 `t`；标准单元的梯度保留完整面积。
这使“密度能量—梯度—位置更新”保持同一尺度，而不是只改最终报表。

## filler 的生成

filler 不是最终设计的一部分，只是让连续密度场在稀疏区也有足够的可移动电荷。设 `Q_static` 为固定
对象和暗区的等效电荷，初始可移动对象的电荷为

```text
Q_movable = A_standard + t × A_movable_macro
```

则所需 filler 的**物理面积**为：

```text
A_filler = max(0, t × A_core - Q_static - Q_movable) / t
```

除以 `t` 很重要：filler 本身进入密度时也会被 `t` 缩放。生成数量仍有 `maximum_fillers` 上限，避免
大基准意外消耗过多内存。

## 两个诊断指标

`global_history.csv` 中保留两条溢出率：

- `optimizer_overflow`：包含 filler，表示优化器这一轮实际看到的压力；
- `design_overflow`：排除 filler，表示真实设计单元的课程溢出率。

二者共享固定端子、宏、暗区、分母和网格定义。最终 `overview.txt` 使用后者所属的 `calculateDensity`
结果，并额外输出 `total_overflow_area`、`density_normalization_area`、`density_charge_area`、
`placeable_area` 与 `dark_area`，便于复核公式。

## 自动验证

`tests/DensityModelTests.cpp` 使用一个 `2×2` 人工网格精确验证：SiteRow 暗区掩膜、标准/宏/固定/filler 的
缩放、溢出分子、课程分母以及重叠 SiteRow 的拒绝行为。混合尺寸 fixture 还验证了宏在实际
`calculateDensity` 中按 `t × area` 进入分母。Release 和 AddressSanitizer 的 `ctest` 均覆盖这些
测试。
