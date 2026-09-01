# 架构与数据流

## 一条内部数据主线

所有算法只看 `PlacementDatabase`：

```text
BookshelfParser
  ├── Module：名称、尺寸、中心坐标、方向、固定/宏属性
  ├── Pin：所属 Module、所属 Net、相对中心偏移
  ├── Net：引脚索引与权重
  └── SiteRow：可放置行、site 宽度、步长、可放置范围
          ↓
Metrics / InitialPlacer / GlobalPlacer / Legalizer / Renderer / GdsWriter
```

该设计刻意避免让算法保存解析器指针或文件路径。输入格式是前端，布局器核心是后端；未来可写
`DefParser`，把 LEF 中的库尺寸和 DEF 中的实例/网络转换到同一个数据库。

## 坐标约定

- BookShelf `.pl` 记录的是单元左下角；读入时立即转换为中心坐标。
- `.nets` 引脚偏移相对于所属单元中心。
- 线长和优化均以“中心 + 经方向变换后的引脚偏移”计算。
- 绘制 BMP 时 y 轴翻转，因为布局原点在左下、屏幕原点在左上。

## 每层的责任

| 层 | 只负责 | 不负责 |
|---|---|---|
| `io` | 文本到对象图、格式校验 | 线长、坐标优化 |
| `model` | 对象关系、坐标变换、派生区域 | 文件格式细节 |
| `metrics` | HPWL、bin 面积、溢出率 | 修改布局位置 |
| `placement` | 修改可移动对象位置 | 图像、文件导出 |
| `export` | 将当前数据库写成 BMP/GDSII | 决定布局位置 |
| `app` | 命令行参数、阶段编排、报告文件 | 算法细节 |

## 全局布局迭代

```text
当前位置
  ↓（由当前溢出率确定平滑度）
平滑网络线长梯度 + 课程 ePlace 分层 bin 密度 → ElectrostaticField → DCT-Neumann 场 → 密度梯度
  ↓（按引脚数、面积、密度权重预条件）
闭环估计局部曲率与步长
  ↓（Nesterov 前瞻）
候选位置
  ↓
目标函数/密度检查 ──变坏→ 回溯缩步长或动量重启
  ↓
接受后记录 HPWL / 两类溢出率 / λ / 曲率，并动态更新 λ
  ↓
恢复满足真实密度约束时 HPWL 最小的检查点
```

`DensityMap` 先由 `SiteRow` 建立可放置掩膜：未被行覆盖的 core 面积成为暗区。标准单元按完整面积，
宏、固定对象、filler 和暗区按目标密度缩放；所有阶段共用相同的溢出分子/分母定义，详见
[density_model.md](density_model.md)。filler 只保留在 `GlobalPlacer` 的局部工作集，因此不会污染
`PlacementDatabase`、合法化或 GDSII。

接口定义在 `include/myplacement/placement/GlobalPlacer.hpp`。实现进一步分成四层：

```text
GlobalPlacer.cpp                 公开 facade：解析 optimizer 选择并分发
GlobalPlacementInternal.hpp      src 私有的 adaptive / legacy 入口声明
GlobalPlacementSupport.*         共用的静态密度、dark mask、filler、密度源与电荷尺度
AdaptiveGlobalPlacer.cpp         闭环控制器
LegacyGlobalPlacer.cpp           保留的开环 A/B 基线
CudaPlacementBackend.*           可选 CUDA 数值后端的接口、CPU stub 与实现
```

这保证 adaptive 与 legacy 不会各自复制、漂移课程密度逻辑，同时优化器内部工作集不泄漏到解析、
合法化或导出层。详细控制律见 [adaptive_global_optimizer.md](adaptive_global_optimizer.md)。

`ElectrostaticField` 是一个独立、无数据库依赖的数值接口：输入按行优先排列的 bin 密度偏差，
输出势和电场。它内部封装 FFTW 缓冲区与计划，默认使用闭边界 DCT；`periodic` 版本只用于 A/B
回归。这样全局布局只负责把 `DensityMap` 转成密度源和把电场转换成单元梯度，数值求解可以被
单独做解析解、DC 去除和边界条件测试。

当选择 `--compute-backend cuda` 时，Adaptive 控制器仍在 CPU 上执行，但平滑线长、分层密度投影、
DCT-Neumann 求解和电场采样都由 CUDA 后端完成。该后端只服务于 `adaptive + neumann`，不会让
`legacy` 或 `periodic` 的 A/B 基线悄悄改变计算路径；详细内存边界和验证见 [cuda_backend.md](cuda_backend.md)。

因此项目并不存在“CPU 版算法”和“GPU 版算法”两份需要分别维护的业务实现：`GlobalPlacer` 只选择
一条正式控制律，CUDA 后端仅替换其中的数值评估。`LegacyGlobalPlacer` 和 periodic 场也不是候选交付
版本，而是隔离在同一 facade 后的回归/消融工具。

## 合法化

1. 先按面积从大到小处理可移动宏块，在固定障碍和已放宏块之间搜索最近可行位置。
2. 从固定对象和宏块中扣除每条 `SiteRow` 的不可用区间，得到若干可用 row slot。
3. 对每个标准单元，只在其附近若干行中选择有剩余容量的 slot。
4. 在每个 slot 内按目标 x 顺序，用加权 isotonic 回归得到不重叠的连续坐标，再对齐到 site。
5. 独立检查边界、行/site 对齐以及重叠。

这让全局布局的“连续近似位置”成为可实际放置的“离散行位置”。
