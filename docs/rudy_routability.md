# RUDY 布线需求热点代理

## 结论边界

BookShelf 输入只提供模块、引脚、网表、行和区域；没有金属层、走线方向、轨道数、障碍物层或
全局路由器。因此本项目绝不把下面的指标称为真实 `route overflow`、DRC 数或可布通保证。它是一个
可微、方向区分的 **RUDY routing-demand hotspot proxy**：适合在全局布局阶段避免把大量连线压进同一
片区域，但必须在拥有 LEF/DEF、技术规则和全局路由器时再用真实路由结果复核。

这一区分不是措辞问题。已有工作既显示 RUDY 可作为布局阶段的需求估计，也显示仅优化代理指标未必
改善真实路由器的 overflow。因此本实现输出字段一律使用 `rudy_proxy_*`，并将它作为布局决策和
实验筛选的一个证据，而非签核结论。

## 定义

将 core 划分为 `B` 个 RUDY bin。对权重为 `q_n`、引脚包围盒为
`[x_l,x_r] × [y_b,y_t]` 的 net `n`，令

```text
w_n = x_r - x_l,        h_n = y_t - y_b
A_{n,b} = area(b ∩ bbox(n))
```

则 bin `b` 上的水平方向与垂直方向 RUDY 需求为

```text
H_b = Σ_n q_n A_{n,b} / h_n
V_b = Σ_n q_n A_{n,b} / w_n
```

这是矩形 overlap 的精确离散化，而不是把每条 net 简单平均撒到 bbox 内。极窄 net 在栅格化时用
`minimum_span_in_bins` 扩张以消除 `1/w_n` 或 `1/h_n` 的奇异性；扩张轴只传播真实的整体平移梯度，
不会产生人为的拉伸/收缩梯度。bbox
边界恰好落在 bin 分界线时，使用相邻 bin 各一半的 Clarke 次梯度，避免二次初始布局把 net 固定在
零梯度的格线上。

## 固定参考容量与拥塞能量

在设计密度首次满足 `routability_start_overflow` 时刻 `t_a`，从当时的 RUDY 图校准一次方向容量：

```text
C_H = α · mean_b(H_b(t_a))
C_V = α · mean_b(V_b(t_a))
```

其中 `α = rudy_capacity_factor`（默认 1.0）。`C_H` 和 `C_V` 随后冻结，绝不随布局更新。若每一轮都用当前平均
需求重新校准容量，优化器可以通过改变参考值而非消除热点来降低自身分数，目标函数就失去含义。

归一化利用率为 `u_H=H_b/C_H`、`u_V=V_b/C_V`，RUDY 能量为

```text
E_RUDY = (1/B) Σ_b [ φ(u_H - 1) + φ(u_V - 1) ]
```

可选的 `φ` 包括：

- `rudy_hinge_l2`：`φ(z)=1/2·max(0,z)^2`；
- `rudy_softplus_l2`：用温度受控的 softplus 平滑上式；
- `rudy_hinge_l4`：`φ(z)=1/4·max(0,z)^4`。

`proxy_overflow` 是两个方向 `max(0,u-1)` 的 bin 平均；`maximum_utilization` 与
`p95_utilization` 分别描述最坏热点和长尾。它们均是相对 RUDY 需求，不是工艺容量。

## 如何进入布局目标

自适应全局布局的完整目标为

```text
F_t = W_γ + λ_t E_density + η_t E_RUDY
```

`W_γ`、`E_density` 和 `λ_t` 保持原有定义。激活 RUDY 时先计算一次原始梯度，再作 L1 范数匹配：

```text
η* = s · ( ||∇W_γ||₁ + λ_t ||∇E_density||₁ ) / max(||∇E_RUDY||₁, ε)
η_t = η* · min(1, (k + 1) / ramp_iterations)
```

其中 `s=routability_weight_scale`，`k` 是 RUDY 激活后的已接受步数。这样 `s=0.45` 的含义是“RUDY
初始满权重的总力约为线长加密度总力的 45%”，而不是一个依赖实例尺度的任意大数。升温期间会丢弃
过期曲率历史并重启动量，避免把不同权重的梯度混在同一条 Barzilai–Borwein 估计中。

达到课程密度门槛的 checkpoint 使用静态的 `HPWL + η* E_RUDY` 作比较；因 `η*` 固定，不会因升温过程
而偏向早期或晚期状态。未达到密度门槛时仍优先最小真实设计溢出率。

## 留出验证而非自我打分

`--rudy-validation-bins 128` 会在同一个激活参考状态校准另一套 128×128 RUDY 容量，并且从不把它的
能量或梯度加入 `F_t`。最终 `overview.txt` 中的 `rudy_validation_*` 因而是不同分辨率的留出代理评估。
`rudy_validation_capacity_factor` 又独立于优化容量因子（默认固定 1.5），使容量因子扫描能在统一阈值下
比较，不能通过改评分标尺“获胜”。这仍不是独立全局路由器，但能发现只针对某一布局栅格过拟合的情况。
`GlobalPlacementResult` 保存的是恢复后的全局布局 checkpoint；若随后运行合法化，`overview.txt` 会以
`rudy_evaluation_stage=final_database` 明确表示重新在最终数据库上测得的 `rudy_*`，并单列
`*_global_checkpoint_*` 以免将合法化前后的指标混为一谈。

## CUDA 边界与资源

CUDA 后端继续负责高吞吐的平滑线长和静电密度求值；RUDY overlap、固定容量校准及解析次梯度在 CPU
上计算，以保证其公式与测试完全一致。后者只使用若干 96×96/128×128 双精度图及线性 module 梯度，
不会显著增加设备显存。运行时仍应显式设置共享服务器限额。

已完成合法化确认的课程基准配置（完整筛选过程见
[rudy_experiments.md](rudy_experiments.md)）是：

```bash
# adaptec1：最终 128×128 留出 proxy overflow 降 63.5%
build-cuda/myplace \
  'resources/course_materials/任务2参考/BookShelf格式的解析/adaptec1/adaptec1.aux' \
  --output out/scratch/adaptec1-rudy \
  --initial quadratic --iterations 160 --bins 64 --density-field neumann --seed 2026 \
  --routability-model rudy_softplus_l2 --rudy-bins 96 --rudy-validation-bins 128 \
  --rudy-capacity-factor 1.0 --rudy-validation-capacity-factor 1.5 \
  --rudy-start-overflow 0.20 --rudy-ramp-iters 24 --rudy-weight 0.60 \
  --compute-backend cuda --gpu-device 1 --gpu-memory-limit-gib 10 --no-bmp --no-gds

# adaptec4：最终 128×128 最大利用率降 32.6%
build-cuda/myplace \
  'resources/course_materials/任务2参考/BookShelf格式的解析/adaptec4/adaptec4.aux' \
  --output out/scratch/adaptec4-rudy \
  --initial quadratic --iterations 280 --bins 64 --density-field neumann --seed 2026 \
  --routability-model rudy_hinge_l4 --rudy-bins 112 --rudy-validation-bins 128 \
  --rudy-capacity-factor 0.65 --rudy-validation-capacity-factor 1.5 \
  --rudy-start-overflow 0.20 --rudy-ramp-iters 24 --rudy-weight 0.95 \
  --compute-backend cuda --gpu-device 1 --gpu-memory-limit-gib 10 --no-bmp --no-gds
```

`gpu-memory-limit-gib=10` 是硬上限；共享机应只在选定卡的空闲显存至少为 `10 GiB + 15 GiB` 时启动。

## 主要依据

1. [Spindler and Johannes, *Fast and Accurate Routing Demand Estimation for Efficient Routability-Driven Placement*, DATE 2007](https://doi.org/10.1109/DATE.2007.364463)：提出 RUDY 的矩形需求估计。
2. [Liu et al., *Global Placement with Deep Learning-Enabled Explicit Routability Optimization*, DATE 2021](https://past.date-conference.com/proceedings-archive/2021/pdf/1290.pdf)：给出 `W + λD + ηL` 形式，并讨论 RUDY 硬 bbox 梯度的次梯度处理。
3. [Cheng et al., *RePlAce: Advancing Solution Quality and Routability Validation in Global Placement*, IEEE TCAD](https://vlsicad.ucsd.edu/Publications/Journals/j126.pdf)：说明真实 routability 需要路由器需求/容量与后续 inflation，而不是把 RUDY 误报为物理 overflow。
4. [Li et al., *DCGP: Direct Congestion-Driven Global Placement*, DAC 2025](https://xingquan-li.github.io/docs/paper/25-DAC25-DCGP.pdf)：使用额外拥塞能量并按梯度范数校准多目标权重，启发了此处的尺度归一化策略。
5. [C3PO, APSDAC 2026](https://hhsiao30.github.io/papers/yichen_apsdac26__Camera_Ready_eXpress.pdf)：给出方向区分的 RUDY 需求与硬 bbox 导数推导，可交叉核对这里的 overlap 公式和次梯度实现。
