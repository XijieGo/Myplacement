# 文档导航

## 先确认项目状态

当前稳定基线只有 `adaptive + neumann + course_eplace_v1`。CPU 与 CUDA 是这一基线的两个数值后端；
`legacy` 与 `periodic` 仅用于 A/B 回归，不能替代正式结果。正式结果只在 `out/verified/`，个人试跑
写入 `out/scratch/`，已整理但不交付的验证写入 `out/diagnostics/`。

项目文档按“先理解边界，再理解算法，再查看验证”的顺序组织：

1. [architecture.md](architecture.md)：模块边界、数据流和目录职责；
2. [density_model.md](density_model.md)：课程 `course_eplace_v1` 密度定义、暗区和 filler；
3. [dct_neumann.md](dct_neumann.md)：闭边界 DCT 静电场的离散方法与数值验证；
4. [adaptive_global_optimizer.md](adaptive_global_optimizer.md)：闭环全局优化器、诊断字段和控制逻辑；
5. [rudy_routability.md](rudy_routability.md)：RUDY 拥塞能量、容量冻结、留出代理验证及其物理边界；
6. [rudy_experiments.md](rudy_experiments.md)：RUDY 的 V2 参数扫描、合法化后筛选和可复跑协议；
7. [cuda_backend.md](cuda_backend.md)：CPU/CUDA 后端边界、显存限制和已验证的单卡加速；
8. [a100_quality_strategy.md](a100_quality_strategy.md)：A100/A800 质量候选、窗口详细放置实验、
   共享 GPU 运行边界与答辩叙事；
9. [abacus_legalization.md](abacus_legalization.md)：Abacus 合法化的算法来源、实现边界和 A/B 验证；
10. [verification.md](verification.md)：当前可比较的验证基线与结果归档规则。

任何布局质量数字都应先确认对应 `overview.txt` 含有
`density_metric=course_eplace_v1`。缺少该字段的历史结果只用于开发追溯，不能用于课程密度门槛或
新旧版本比较。
