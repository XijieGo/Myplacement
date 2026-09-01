# DCT 闭边界静电场：实现、验证与消融

## 问题的物理直觉

密度优化把每个可移动单元看成一小块“电荷”。局部单元面积超过目标密度时，该处是正密度偏差，
应当把单元推向稀疏区。令 `φ` 为势、`E` 为电场，代码求解的是：

```text
ρ' = ρ - mean(ρ)
-∇²φ = ρ'
E = -∇φ
```

`mean(ρ)` 必须先去掉：闭边界下 Poisson 方程的常数模式没有唯一势，也不能产生有意义的力。
有限芯片不能把单元从左边推到区域外、再从右边绕回，因此边界使用：

```text
∂φ/∂n = 0    （等价于 n·E = 0）
```

这里的 `n` 是边界的外法线。这不是“把坐标硬夹回 core”代替边界条件；夹回只是保护更新位置，
DCT/Neumann 则是在计算排斥力时从根本上消除穿越边界的法向分量。

## 离散方法

密度在 bin 中心采样：

```text
x_i = x_min + (i + 1/2) Δx
y_j = y_min + (j + 1/2) Δy
```

余弦基 `cos(kπx/Lx) cos(lπy/Ly)` 的法向导数在四条边界均为零，因而天然满足闭边界。实现位于
[`ElectrostaticField.hpp`](../include/myplacement/placement/ElectrostaticField.hpp) 和
[`ElectrostaticField.cpp`](../src/placement/ElectrostaticField.cpp)：

1. 用二维 DCT-II 将 `ρ'` 变为余弦系数。
2. 对每个非 DC 模式，令
   `φ̂[k,l] = ρ̂[k,l] / ((kπ/Lx)² + (lπ/Ly)²)`；`φ̂[0,0] = 0`。
3. 用 DCT-III 重建势。
4. 由 `E=-∇φ` 重建电场：`E_x` 是“x 方向正弦、y 方向余弦”，所以使用 DST-III × DCT-III；
   `E_y` 则使用 DCT-III × DST-III。
5. 在任意单元位置采样时，对势使用偶镜像，对 `E_x` 的 x 方向和 `E_y` 的 y 方向使用奇镜像。
   因此两个镜像值在法向边界相消，线性插值仍严格满足 `n·E=0`。

FFTW 的 r2r 变换未归一化。一个二维 DCT-II/DCT-III 往返带来 `4*Nx*Ny` 的系数，因此代码在
频域除法前显式乘 `1/(4*Nx*Ny)`。这项看似细小，但漏掉后电场量级会随 bin 数变化，无法稳定调参。

`periodic` 版本也在同一接口中保留：它使用旧有 r2c/c2r FFT 和波数 `2πk/L`，仅作为回归与消融
基线，不能作为有限芯片的默认物理模型。

## 接口边界

```text
DensityMap
   ↓  buildDensityDeviation()
vector<double> density deviation
   ↓  ElectrostaticField::solve()
potential grid + electric-field grid
   ↓  sampleField(module.center)
GlobalPlacer density gradient
```

`GlobalPlacer` 不再管理 FFTW 缓冲区或频域索引；`ElectrostaticField` 也不知道模块、网络或文件格式。
这使数值层能独立验证，而布局层只处理“密度 → 单元梯度”的业务转换。

## 自动验证

`tests/DensityFieldTests.cpp` 添加了以下确定性测试，全部包含在 `ctest` 中：

- 二维解析模式 `(kx, ky)=(2, 3)`：势、`E_x`、`E_y` 与解析值逐点比较；
- 两个轴向模式 `(0, 3)` 与 `(2, 0)`：覆盖 DCT 的常数轴和 DST 的索引平移；
- 均匀密度：DC 去除后势与场均为零；
- 四条边：中心采样位置沿法向场为零；
- 超过 `1,048,576` 个 bin：构造函数在分配前拒绝请求；
- `closed`/`neumann` 和 `fft`/`periodic` 命令行解析别名。

密度层还通过课程 ePlace 的精确 `2×2` 算例验证 SiteRow 暗区、宏/固定/filler 缩放、溢出分子和
归一化分母；定义见 [density_model.md](density_model.md)。

在当前实现中，解析模式与边界测试的绝对误差上限为 `2e-12`。这比“布局结果看上去合理”更能发现
变换类型、归一化、符号或 DST 系数下标的错误。

## 安全边界

| 防护 | 实现 |
|---|---|
| 密度网格 | 代码在 `DensityMap` 分配前限制总 bin 数不超过 `1,048,576`；方形命令行网格最大为 `1024×1024`。 |
| FFT 工作区 | DCT 路径最多保留 7 个双精度实数组，`1024×1024` 时约 56 MiB；周期基线约 64 MiB，均不含设计数据库。 |
| 线程 | FFTW 未初始化线程接口；安全脚本固定 CPU affinity 为 0–15，并设定常见数值库的线程变量为 16。 |
| 内存 | `scripts/run_safe_benchmark.sh` 为整个子进程设置 16 GiB `RLIMIT_AS`，远低于服务器总内存。 |

边界压力检查也已实际运行：`thin1` 使用最大允许的 `1024×1024` 网格、一次 Neumann 全局迭代，
在 **1 GiB** `RLIMIT_AS` 下成功完成，峰值 RSS 为 **194 MiB**、无 swap。这同时验证了上限附近的
DCT 工作区路径，而不是只验证“超限会报错”。

## 已运行的同条件消融

命令条件应固定为：`adaptec1`、`quadratic` 初始布局、相同迭代数、`64×64` bin、`seed=2026`、关闭
BMP/GDS/合法化，单进程、CPU 0–15、16 GiB 地址空间上限。报告至少应同时记录最终 HPWL、
`design_overflow`、最大密度、运行时间与 RSS。

项目中曾保存过一张早于 `course_eplace_v1` 的数值表；其溢出率使用了已废弃的固定端子分母，不能
再用于闭边界/周期边界质量比较。闭边界的解析模式、DC 去除和零法向场测试仍然有效，但任何质量
消融都必须用当前版本脚本重新生成两侧的 `overview.txt` 与 `global_history.csv`。`thin1` 只有 3 个
可移动单元，适合作为快速数值 smoke test，不适合作为质量结论。

复现实验：

```bash
scripts/run_density_field_ablation.sh \
  build \
  'resources/course_materials/任务2参考/BookShelf格式的解析/adaptec1/adaptec1.aux' \
  out/scratch/adaptec1-neumann-vs-periodic-80 \
  --initial quadratic --iterations 80 --bins 64 --seed 2026 --no-bmp --no-gds --no-legalize
```
