# 当前验证基线

本页只记录采用 `course_eplace_v1` 密度口径的结果。该口径将固定端子、暗区和 filler 作为密度障碍，
但不允许它们稀释归一化溢出率分母；完整公式见 [density_model.md](density_model.md)。

## 可交付验证结果

| 基准 | 配置 | 全局布局最佳可行检查点 | 最终合法化结果 |
|---|---|---:|---:|
| `adaptec1` | adaptive / Neumann / 64×64 | overflow `0.0918172`（第 140 轮） | 合法，overflow `0.0430756` |
| `adaptec4` | adaptive / Neumann / 默认 280 轮 / 64×64 | overflow `0.0995075`（第 244 轮） | 合法，overflow `0.0431739` |
| `thin1` | adaptive / Neumann / 默认 280 轮 / 64×64 | 初始检查点可行 | 合法，overflow `0` |

对应的原始报告位于：

- `out/verified/course-eplace-v1/adaptec1/adaptive-neumann-legalized/`；
- `out/verified/course-eplace-v1/adaptec4/adaptive-neumann-default-legalized/`；
- `out/verified/course-eplace-v1/thin1/adaptive-neumann-default-legalized/`。

## RUDY 拥塞代理的最终验证

RUDY 是方向需求热点代理，不是技术相关 router overflow；公式、容量冻结及其物理边界见
[rudy_routability.md](rudy_routability.md)。本表只列完成合法化后的最终数据库评分，验证栅格为
128×128、验证容量系数为 1.5：

| 基准 | 最终配置 | 相对 monitor 的最终 proxy overflow | 相对 monitor 的最大利用率 | 合法化 |
|---|---|---:|---:|---|
| `adaptec1` | Softplus-L2 / α=1.0 / s=0.60 / 96×96 | `0.025305 → 0.009242`（-63.5%） | `2.416 → 1.729`（-28.4%） | PASS |
| `adaptec4` | Hinge-L4 / α=0.65 / s=0.95 / 112×112 | `0.846109 → 0.788076`（-6.9%） | `10.184 → 6.867`（-32.6%） | PASS |

两项结果的 HPWL 也都下降（A1 -5.0%，A4 -3.8%）。完整参数扫描、被淘汰的“全局好看但合法化后变差”
候选和可复跑命令见 [rudy_experiments.md](rudy_experiments.md)。原始最终报告在：

- `out/verified/rudy-v2/a1_softplus_c100w060_legal/`；
- `out/verified/rudy-v2/a4_hinge_l4_c065w095_grid112_legal/`。

## 验证规则

- `overview.txt` 是最终布局的权威报告；其中必须有 `density_metric=course_eplace_v1`。
- `global_history.csv` 的 `optimizer_overflow` 包含虚拟 filler；`design_overflow` 排除 filler，
  是检查点和课程约束使用的真实设计指标。
- `out/diagnostics/course-eplace-v1/cuda-validation/` 保存 CPU/CUDA 数值一致性检查；
  `cuda-performance/` 保存已整理的单卡性能/显存记录。二者均不能代替最终交付结果。
- `out/scratch/` 是任意新试跑的默认位置，不应直接引用到报告中。
- `out/archive/` 保存旧口径、已替代或冻结前未整理的输出；这些数据不能用于性能或质量结论。

## 自动测试

当前代码要求四个层级同时通过：常规 CPU CTest、Release CPU CTest、AddressSanitizer CTest 和
启用 CUDA 的构建 CTest。数值层、课程密度层和端到端流程分别位于
`tests/DensityFieldTests.cpp`、`tests/DensityModelTests.cpp` 和 `tests/PlacementFlowTests.cpp`；
共享 GPU 的数值一致性与性能检查保留为显式的诊断运行，不会被 CTest 自动抢占设备。
