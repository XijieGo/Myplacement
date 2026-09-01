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
5. [cuda_backend.md](cuda_backend.md)：CPU/CUDA 后端边界、显存限制和已验证的单卡加速；
6. [verification.md](verification.md)：当前可比较的验证基线与结果归档规则。

任何布局质量数字都应先确认对应 `overview.txt` 含有
`density_metric=course_eplace_v1`。缺少该字段的历史结果只用于开发追溯，不能用于课程密度门槛或
新旧版本比较。
