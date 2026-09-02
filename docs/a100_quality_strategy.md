# 从“完成作业”到可答辩的 95+：A100/A800 质量路线

## 结论先行

本项目不该把 GPU 当成“同一算法跑得快一点”的装饰。课程主线（BookShelf 解析、三种初始布局、全局
布局、展示）已经完整，真正会让答辩变得有说服力的是同时回答三个问题：

1. 最终导出的**合法**布局是否真的更好，而非只让全局阶段的一个中间数字好看？
2. GPU 是否做了 CPU 路径难以在同一时间预算内完成的质量搜索？
3. 对拥塞、显存和共享机器的结论是否有清晰边界，而非夸大 BookShelf 所没有的信息？

当前推荐的正式路线是：

```text
adaptive + Neumann 全局布局 + 已验证的 RUDY 热点代理
                    ↓
              行级合法化
                    ↓
  A100/A800：四单元、24 个全排列的 warp 级精确详细放置
                    ↓
       CPU 精确复核每一步 + 最终合法性复核
```

这里的 GPU 重点不是改写已有的全局优化器，而是在合法解上批量搜索大量局部排序。四个连续单元恰有
`4! = 24` 个候选；一个 32-lane warp 可同时评估它们。这是把 A100/A800 的并行宽度直接变成最终
HPWL 质量，而不是只减少日志里的运行秒数。

本服务器实际查询到的卡为 **NVIDIA A800-SXM4-80GB**（Ampere、A100 同代计算能力），所以本文只把
已经跑出的数据称为 CPU 结果；在卡空闲前不虚构“A100 实测加速”。CUDA 源码已在 `sm_80` 配置下成功
编译并通过测试，GPU 实跑有严格的空闲卡门槛，见后文。

## 老师要求与本次提升的边界

课程任务 PDF 明确列出：Linux/C++ 环境、BookShelf/DEF/LEF 理解、BookShelf 解析、至少三种初始布局、
全局布局、CImg + GDSII 两种展示方式；布局合法化是选做项。本项目已经覆盖这些主线，并额外拥有行级
合法化、CUDA 数值后端和 RUDY 代理。因此冲击 95+ 的重点不应再是“再加一个格式解析器”或“再画一张图”，
而应是拿出最终合法结果的 QoR 证据，并说明 A100/A800 在其中做了什么不可替代的工作。

## 为什么要把创新放在合法化之后

课程要求的全局布局并不等于最终布局。以固定的 A1 正式 CPU 基线为例：全局 checkpoint 的 HPWL 是
`116,932,383.577`，行级合法化后的 HPWL 是 `145,351,310`。这是约 24.3% 的落差；如果只展示前者，
就会把最该解决的质量损失藏起来。

详细放置的约束恰好适合可验证的局部搜索：一个窗口内的单元重新排序后，仍占用同一行、同一连续
site 区间，宽度总和不变。因此每次接受的 move 都不会制造重叠、越行或出 core。该性质让“GPU 大规模
候选评估 + CPU 精确接受”比在非法位置上盲目学习一个黑盒位移更可靠，也更适合课程答辩。

## 三个候选方向，以及为什么不能只堆功能

| 候选方向 | GPU 的不可替代作用 | 实际价值与实验状态 | 决策 |
|---|---|---|---|
| 1. warp 级精确窗口详细放置 | 一个窗口一个 CUDA block；24 个线程各评估一个四单元排列，硬件一次批量给出 argmin | 已实现，A1 完整 160 轮及 RUDY 叠加均有合法化后对照 | **主方案** |
| 2. 多 GPU、留出指标驱动的 Pareto 多起点搜索 | 四张卡同时跑不同初始解/RUDY 模型/权重，而不是人为押注一组参数 | RUDY 的参数筛选与 128×128 留出验证已证明“没有万能参数”；并行调度尚待空闲 GPU 运行 | **第二阶段方案** |
| 3. GPU 路由代理约束的详细放置 | 对同一 24 个排列同时计算局部 `ΔHPWL` 与 `ΔRUDY`，做 Pareto/约束接受而非只贪婪 HPWL | 数学和数据布局可行，但 BookShelf 缺少真实层/轨道容量；当前不把它冒充为 router | **有条件储备** |

### 1. 四单元 warp 详细放置：选择它的技术理由

实现位于 [详细放置公共接口](../include/myplacement/placement/DetailedPlacer.hpp)、
[CPU 参考实现](../src/placement/DetailedPlacer.cpp) 和
[CUDA 后端](../src/placement/cuda/CudaDetailedPlacementBackend.cu)。工作方式如下：

1. CPU 从已合法化的行中切出连续、无障碍的单元段，跳过触及高扇出 net 的窗口；
2. GPU 给每个恰好四单元的独立窗口分配一个 32-thread block，前 24 个线程枚举全部排列，计算所有受影响
   net 的精确局部 HPWL；
3. GPU 只返回最优排列提示。CPU 在**当前**数据库上重新计算该 move 的精确 HPWL，只有严格改善才写回；
4. 每一 pass 后仍执行 `Legalizer::check`，最终 `overview.txt` 记录 HPWL、候选数、接受数、后端和显存预算。

第 3 步很关键：不同窗口可能通过同一条 net 间接耦合，GPU 并发评分使用的是 pass 开始时的位置。CPU
复核避免了陈旧评分导致退化；即使 CUDA 出错，后端会恢复进入详细放置前的 module center/orientation，
`auto` 才会安全回退到 CPU。它不是“GPU 结果神秘更好”，而是一条有单调质量保证的异构搜索路径。

四单元不是拍脑袋选出的数字。五单元需要 `5! = 120` 个候选，至少要跨四个 warp；六单元为 720 个，
候选数已经压过局部窗口变大的边际质量价值。四单元恰好既可一 warp 穷举，又明显强于相邻交换。

### 2. 多 GPU Pareto 多起点：让硬件换来“更会选”而非“更快停”

课程已经要求 random、cluster、quadratic 三种初始布局，但一个正常流程通常只把其中一条送入最终优化。
在本机四张允许使用的 Ampere 卡空闲时，应把可并行的独立轨迹分发为一个小型、受控的 population：

```text
3 种 initial × {RUDY model, capacity, weight, grid} × 若干 seed
             ↓  （每张 GPU 仅一个进程）
合法性 / 课程密度硬门槛
             ↓
128×128 留出 RUDY proxy、最大利用率、最终 HPWL 的 Pareto 前沿
             ↓
只把前沿解送入详细放置与最终展示
```

这不是“多跑几次然后挑最低 HPWL”。RUDY 实验已经说明，A1 上看似很强的某些全局 checkpoint 在合法化后
会失去优势；A4 的最佳 penalty、capacity 和 grid 又与 A1 不同。详见
[RUDY 实验记录](rudy_experiments.md)。所以选择规则必须是：

- `legal=true`、overlap/off-row/unplaced 都为零；
- 课程设计溢出率满足门槛；
- 用从不进入目标函数的 128×128 留出 RUDY 图比较 proxy overflow 与最大利用率；
- 在上述约束下比较最终 HPWL，而不是优化栅格自己的能量。

当前 CUDA 全局后端已经在 A1/A4 的固定短跑上给出约 4.35×/4.17× 的单卡数值加速，记录在
[CUDA 后端验证](cuda_backend.md)。四卡的价值不是让同一条轨迹占满四卡，而是在同样交付时间内把
“参数靠直觉”换成“Pareto 前沿有数据”。该方向应在四张卡真的空闲时执行；共享机器上绝不为追求
吞吐而多进程同卡共驻。

### 3. RUDY 约束的 GPU 详细放置：可做，但必须先守住物理边界

更激进的版本会把候选窗口的比较从单一 HPWL 扩展为：

```text
优先最小 ΔHPWL；若候选近似并列，则最小化 held-out local ΔRUDY；
或只接受 ΔHPWL < 0 且 ΔRUDY 不恶化超过预先声明阈值的排列。
```

实现上并不需要大型模型：当前 CUDA 后端已经把 window-module、net-pin incidence、pin offset 和候选
center 扁平化。对每个候选增量维护受影响 net 的 bbox，再把 bbox 对小型 RUDY 栅格的 overlap 归约即可。
四单元、最大 net 度 64 的边界使共享内存与寄存器使用量可预估，A100/A800 很适合做这类高算术密度的
候选筛选。

但它不应立刻成为默认：BookShelf 没有金属层、方向 capacity、track、blockage layer 或真实 global
router，RUDY 只能是 demand-hotspot proxy，不能叫“可布通”或 router overflow。当前已验证的全局 RUDY
扩展保留了这一边界，见 [定义与边界](rudy_routability.md)。只有在以下条件同时满足时，才升级该候选：

1. 至少 A1/A4、至少三个 seed 的 held-out proxy 均不退化；
2. 最终 HPWL 和合法性不弱于纯 HPWL 四单元版本；
3. 若拿到 LEF/DEF 与真实路由器，再以真实 overflow 复核代理改善。

这不是放弃创新，而是避免在答辩时被一句“你的路由容量从哪里来？”直接击穿。

## 已完成的受控实验

### 统一协议

- BookShelf `adaptec1`（A1），quadratic initial，`seed=2026`；
- `adaptive + neumann`、64×64 密度图、最多 160 全局迭代；
- 同一运行内先全局布局、再合法化、最后才运行详细放置；关闭 BMP/GDS，避免导出时间混入；
- 所有结论读取最终 `overview.txt`，并以 `legal=true`、零 overlap/off-row/unplaced 为硬条件；
- CPU 实验通过 `scripts/run_safe_benchmark.sh` 限制为 16 逻辑核和 16 GiB 虚拟地址空间。

结果保存在 `out/diagnostics/a100-detailed-placement/`。该目录是诊断/研究证据，不会覆盖
`out/verified/` 的课程交付基线。

### A1：窗口大小的完整 160 轮对照

| 最后阶段 | 最终 HPWL | 相对无详细放置 | 详细阶段时间 | 枚举排列数 | 合法 |
|---|---:|---:|---:|---:|---|
| 无 | 145,351,310 | — | — | — | 是 |
| 3-cell | 142,873,396 | -1.7048% | 1.179 s | 719,406 | 是 |
| **4-cell** | **142,007,730** | **-2.3003%** | **2.884 s** | **1,991,916** | 是 |
| 5-cell | 141,462,121 | -2.6757% | 13.855 s | 7,595,516 | 是 |

五单元的绝对 HPWL 最低，但四单元已经获得五单元总改善量的 **86.0%**，只花其 **20.8%** 的详细阶段
时间。由四到五单元只额外减少 545,609 HPWL（0.3754 个百分点），却额外增加 10.971 秒和约 5.60M
候选。因此：

- `window=4, passes=2` 是默认的 GPU 质量/吞吐甜点；
- `window=5` 是允许更长离线时间时的“绝对 QoR”选项，不是默认 GPU 映射；
- 窗口 3 可作为极低延迟基线，不能替代 4 的协同重排能力。

### A1：与已验证 RUDY 全局优化的叠加对照

为排除 CPU/GPU 浮点路径的差异，以下两行均使用 **CPU**、相同 RUDY 参数
`softplus_l2, grid=96, capacity=1.0, weight=0.60`、相同 seed 和 160 轮；唯一差别是最后是否运行四单元
详细放置。留出指标来自不反传的 128×128 RUDY 图。

| 最后阶段 | 最终 HPWL | 留出 proxy overflow | 留出最大利用率 | 合法 |
|---|---:|---:|---:|---|
| RUDY + 合法化 | 160,754,363 | 0.00885701 | 1.749981 | 是 |
| RUDY + 合法化 + **4-cell** | **157,700,337** | **0.00625190** | **1.704799** | 是 |
| 相对变化 | **-1.8998%** | **-29.4130%** | **-2.5818%** | — |

这个结果很有答辩价值：详细放置并没有为了缩短 HPWL 把热点推得更糟；在这个独立留出 proxy 上两者反而
同时改善。它仍不是实际 router 结果，但至少不是“只优化自己目标”的自证循环。

### A4：规模冒烟验证

在 40 轮、同一 CPU 协议下，A4 的四单元版本从 2,855,591,259 降至 2,839,471,685（-0.5645%），详细
阶段 7.671 s，考察 5,415,460 个排列，且合法。它证明了窗口数量随设计规模扩展时仍可工作；它**不是**
A4 280 轮的最终 QoR 宣称。完整 A4 GPU 配对会等空闲卡出现后按下一节条件运行，避免把共享 CPU 时间
耗在与已有正式 A4 基线重复的长跑上。

## CUDA 实现、资源边界和待完成的 GPU 证据

CUDA 版本通过两条构建都已验证：

```bash
cmake --build build --parallel 12
ctest --test-dir build --output-on-failure
cmake --build build-cuda --parallel 4
ctest --test-dir build-cuda --output-on-failure
```

它的资源/正确性设计如下：

- 仅接受 `window`、`window_size=4` 的 CUDA 请求；其他模式可用 `auto` 安全回退 CPU；
- GPU 1--4 的编号限制与全局 CUDA 后端一致；启动前保留至少 4 GiB 空闲显存，所有显式数组计入独立预算；
- 高扇出 net 不近似，而是跳过，避免用不精确的局部评估换取表面吞吐；
- GPU 只提供候选，CPU 以当前状态精确 HPWL 重新验收，因此每个被写回的 move 严格改善其受影响 net；
- CUDA 异常时恢复所有 module 的位置和 orientation，再由 `auto` 回退，不留下半个 pass 的脏布局。

共享 GPU 的可复现启动器是：

```bash
# 只会在同一张卡连续两次都 <=5% 利用率，且在 2 GiB 预算外仍留 15 GiB 时启动。
DETAIL_ITERATIONS=160 DETAIL_GPU_MEMORY_LIMIT_GIB=2 \
scripts/run_detailed_gpu_study.sh build-cuda \
  'resources/course_materials/任务2参考/BookShelf格式的解析/adaptec1/adaptec1.aux' \
  out/diagnostics/a100-detailed-placement/a1-window4-cuda-160 1
```

当前卡处于 100% 利用率时，该脚本会在 preflight 直接退出，不分配显存、不发射 kernel。这条规则比只看
free VRAM 更重要：当前 30 多 GiB 空闲显存并不代表一张 100% 忙的卡可用。

GPU 结果写入正式研究结论前，必须同时满足：

1. 与同输入 CPU 四单元对照相比，最终 HPWL 不出现实质性 QoR 回退；
2. `legal=true`、overlap/off-row/unplaced 全为零；
3. 详细阶段 GPU 时间明显低于 CPU 参考，并记录 `detailed_compute_backend_used=cuda`、GPU 编号和
   预算峰值；
4. A1 通过后，再对 A4 280 轮跑相同的 CPU/GPU 配对，不能仅凭小基准宣布成功。

在这些证据出现前，最严谨的表述是：**CUDA 详细放置已实现并完成编译/单元测试，CPU QoR 已被完整
验证；GPU 运行时速度与 QoR 一致性待空闲 A800 上的受控复跑。**

## 答辩时应如何讲这件事

可以用一条非常直观的叙事，而不是堆术语：

> 全局布局像决定“哪些人住在哪个街区”，合法化把人塞进合法的座位后会拉长连线。我们没有只展示
> 合法化前的漂亮数字，而是利用 GPU 同时尝试每个小街区的 24 种合法座位顺序；CPU 再逐一复核，保证
> 每一个接受的调整都缩短真实 HPWL、不会破坏合法性。对于拥塞，我们只把 BookShelf 可支持的 RUDY
> demand proxy 作为额外证据，并用留出栅格防止模型自我打分。

这比“我们调用了 GPU”“我们有一个 AI/拥塞模块”更经得起追问：有明确的瓶颈、有硬件映射、有最终
质量数据、有失败边界，也有下一步真实 router 验证计划。

## 参考依据

1. [DREAMPlace, DAC 2019](https://research.nvidia.com/publication/2019-06_dreamplace-deep-learning-toolkit-enabled-gpu-acceleration-modern-vlsi-placement)：GPU 并行可把现代布局中的大规模数值/搜索工作移到设备端。
2. [LegalGPU, DATE 2022](https://www.cse.cuhk.edu.hk/~byu/papers/C136-DATE2022-LegalGPU.pdf)：GPU 合法化/后合法阶段的并行潜力与正确性边界。
3. [RUDY 及本项目实现边界](rudy_routability.md)：本项目的方向性需求代理、冻结容量和留出验证定义。
