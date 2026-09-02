# CUDA 单卡全局布局后端

## 目标与边界

CUDA 后端不是第二套布局器，而是 `adaptive` 全局布局器的可选数值后端。CPU 仍保存
`PlacementDatabase`、执行回溯/动量重启/检查点选择和最终合法化；GPU 负责每轮中规模最大的规则
数值计算：

```text
GPU：引脚位置 → 平滑线长梯度 → 密度 bin 投影 → DCT-Neumann 场 → 密度梯度
CPU：闭环控制、候选接受、坐标边界夹紧、检查点、合法化、导出
```

这样 CPU 与 GPU 共用同一份课程密度定义、同一套 Adaptive 控制律和同一个输出格式。GPU 不会改变
`course_eplace_v1` 的宏、固定对象、暗区或 filler 口径。

这是一条已经冻结的数值加速路径，而不是待维护的第二个算法版本：后续若有课程创新，应先在这条
正式控制律上验证质量，再决定新热点是否值得下沉为 CUDA 核心。

当前 CUDA 仅支持默认的 `adaptive + neumann` 组合。`legacy` 和 `periodic` 是 CPU A/B 基线：显式
请求 `--compute-backend cuda` 时会给出明确错误；`auto` 则安全回退到 CPU。

## CUDA 合法化后详细放置

CUDA 构建还包含一条独立的详细放置后端：在已经合法的连续行窗口中，对四个标准单元的 `4! = 24` 个
排序并行评分。一个窗口对应一个 32-thread CUDA block，前 24 个线程各计算一个排列受影响 net 的局部
HPWL。GPU 返回最优候选后，CPU 会在当前数据库上重新计算精确 HPWL，只有严格改善才写回；因此不同
窗口的并发评分不会因为跨窗口 net 的陈旧信息造成质量退化。

```bash
./build-cuda/myplace examples/tiny/tiny.aux \
  --detailed-placement window --detailed-backend cuda \
  --detailed-passes 2 --detailed-window 4 --gpu-device 1
```

`cuda` 详细后端只支持 `window + size=4`；`auto` 在 CUDA 不可用或窗口不兼容时回退到 CPU，`cpu` 则是
全排列参考实现。GPU 异常会恢复进入该阶段前的 module 位置和 orientation，避免回退路径接到半完成布局。
`overview.txt` 会记录 `detailed_backend_requested`、实际使用后端、GPU 编号、显存预算、窗口数和候选数。

共享服务器的大基准不要直接执行上述 tiny 命令模式。应使用
`scripts/run_detailed_gpu_study.sh`：它会连续两次确认允许的 GPU 1--4 或 7 的利用率低于阈值并保留显存余量，当前
高利用率卡会被拒绝而不是抢占。完整 QoR 对照与 GPU 运行时证据边界见
[a100_quality_strategy.md](a100_quality_strategy.md)。

## 构建与运行

普通 CPU 构建不依赖 NVIDIA 环境：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 16
```

在有 CUDA 工具链的机器上，从**同一份源码**构建双后端程序：

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMYPLACEMENT_ENABLE_CUDA=ON
cmake --build build-cuda --parallel 16
```

运行时选择后端，而不是切换项目：

```bash
# 可重复的 CPU 参考结果
./build-cuda/myplace examples/tiny/tiny.aux --compute-backend cpu

# 显式使用一张允许的 GPU
./build-cuda/myplace examples/tiny/tiny.aux \
  --compute-backend cuda --gpu-device 1 --gpu-memory-limit-gib 1

# CUDA 可用且安全时使用 CUDA；否则回退 CPU
./build-cuda/myplace examples/tiny/tiny.aux --compute-backend auto
```

`overview.txt` 会同时记录 `compute_backend_requested`、`compute_backend_used`、实际 GPU 编号和
`cuda_reserved_memory_bytes`，因此不会出现“以为跑了 GPU，实际跑了 CPU”的歧义。

## 共享服务器安全规则

本服务器当前允许布局任务使用物理 GPU `1` 到 `4` 或 `7`。开始 GPU 大基准前先人工检查状态，并显式选择
空闲卡：

```bash
nvidia-smi -i 1,2,3,4,7 \
  --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader
```

程序也会执行以下防护：

- 拒绝 `1` 到 `4` 及 `7` 之外的 GPU 编号；
- 默认显式 CUDA 分配上限为 40 GiB，命令行上限同样不能超过 40 GiB；
- 创建后端前读取当前空闲显存，并额外保留 4 GiB 给共享任务；
- 不使用 Unified Memory（统一内存）或隐式显存溢出；所有持久数组和 cuFFT 工作区均计入预算；
- 密度网格仍受项目的 `1024×1024` 上限约束；
- `scripts/run_safe_benchmark.sh` 同时限制 CPU 到 16 个逻辑核、主机虚拟地址空间到 16 GiB。

## 显存估算与实测上界

这不是 LLM 那样按模型参数量估算显存的程序；它没有常驻权重，显存由设计规模和密度网格决定。对
当前双精度数据布局，持久算法数据可按下式保守估算：

```text
M_payload = 76P + 40I + 92N + 240B²
            + 16 × ceil(max(N, B²) / 512) + W_cuFFT    bytes
```

- `P` 是可移动模块和 filler 的总数；默认 filler 上限为 50,000；
- `I` 是引脚数，`N` 是网络数，`B` 是方形密度网格边长；
- `W_cuFFT` 是 cuFFT 计划要求的显式工作区，后端会与所有其他分配一起计入预算。

在本项目常用的 `64` 与 `1024` 网格上，`W_cuFFT` 为零；公式与下列实测 `cuda_reserved_memory_bytes`
完全一致。排队或与其他任务共享 GPU 时，应在该数值外再留至少 1 GiB 给 CUDA 上下文和驱动。

| 实测用例 | 显式算法数据 | `nvidia-smi` 进程峰值 | 结论 |
|---|---:|---:|---|
| `adaptec1`，64×64 | 75 MiB | 未单独采样 | 1 GiB 显式预算下通过。 |
| `adaptec4`，64×64 | 159 MiB | 未单独采样 | 最大附带设计的常规网格。 |
| `adaptec4`，1024×1024、16 轮、GPU 4 | 398 MiB | 845 MiB | 2 GiB 显式预算下通过，任务结束后显存回到 0 MiB。 |

因此当前课程资料离 40 GiB 的硬上限很远，但这不是取消保护的理由：显存仍会随引脚/网络规模线性增长、
随网格边长平方增长，且程序会在任何 `cudaMalloc` 前执行预算检查。

## CUDA 内核如何对应课程算法

| 课程计算 | CUDA 实现 |
|---|---|
| 平滑线长 | 每个网络一个线程块，先稳定地求最大/最小引脚坐标与指数和，再并行写回引脚梯度。单引脚网络被显式跳过。 |
| 密度 | 每个可移动模块或 filler 并行计算与 bin 的矩形重叠，用双精度原子累加到标准单元、宏和 filler 分层数组。 |
| 闭边界电场 | 将 DCT-II/DCT-III 与 DST-III 的偶/奇对称扩展映射到 cuFFT 双精度复数变换，保持 CPU 版 Neumann 零法向场定义。 |
| 电场采样 | 在 GPU 上按与 CPU 相同的镜像奇偶性插值，直接得到每个粒子的密度梯度。 |
| 控制器 | CPU 只接收每轮标量指标及粒子梯度，执行现有的曲率估计、回溯、动量重启和检查点逻辑。 |

静态网络、引脚偏移、模块尺寸和固定/暗区密度层只上传一次。每次评估只上传当前粒子坐标，并下载
梯度和少量诊断值；这使控制逻辑保持可读，同时把主要算量留在 GPU。

## 已完成验证

所有结果均写入 `out/diagnostics/`，不替代 `out/verified/` 中的课程交付结果。

| 用例 | 条件 | 结论 |
|---|---|---|
| `tiny` | 8×8、12 轮 | CPU/GPU 的 `global_history.csv` 逐行一致。 |
| `mixed` | 含宏、固定对象，8×8、16 轮 | 宏的目标密度缩放、固定密度和闭环历史与 CPU 一致。 |
| `tiny` | 1024×1024、1 轮 | CPU/GPU 历史逐行一致；CUDA 显式分配约 240 MiB。 |
| `adaptec1` | 随机初始、64×64、16 轮、无 BMP/GDS/合法化 | CPU/GPU 历史逐行一致；GPU 全局阶段 1.127 s，CPU 为 4.908 s，约 4.35 倍加速。 |
| `adaptec4` | 随机初始、64×64、16 轮、无 BMP/GDS/合法化、GPU 4 | 质量与控制决策一致；GPU 为 2.497 s，CPU 为 10.406 s，约 4.17 倍加速。 |
| `adaptec4` | 随机初始、1024×1024、16 轮、GPU 4 | 显式数据 398 MiB，实测进程峰值 845 MiB；未接近 2 GiB 测试上限。 |
| `adaptec1` | 当前 Abacus/reverse、160 轮、GPU 7，CPU-global/GPU-detail 配对 | GPU 详细阶段 0.580 s，CPU 为 2.865 s，4.94×；同起点最终 HPWL 差 0.0682%，均合法。 |
| `adaptec1` | 当前 Abacus/reverse、160 轮、全 CUDA、GPU 7 | 全局 6.127 s、详细 0.593 s；阶段合计相对 CPU 5.16×，最终 HPWL 低 0.8347%。 |
| `adaptec4` | 当前 Abacus/reverse、280 轮、全 CUDA、GPU 7 | 全局 24.771 s、详细 1.541 s，最终合法 HPWL 1,132,194,492；全局/详细显式预算约 159/75 MiB。 |

GPU 并行归约在一般输入上不承诺逐比特相同，但它必须保持解析测试、课程密度指标、合法性、控制决策
和最终质量一致。`tiny`、`mixed` 与 16 轮 `adaptec1` 的受控历史逐行一致；更大的 `adaptec4` 只出现浮点
归约末位差异（约 `1e-12` 相对量级）。160 轮 A1 端到端实测则按最终合法 QoR 与运行时比较，见上表。后续任何
CUDA 优化都应继续以 CPU 后端为基准。
