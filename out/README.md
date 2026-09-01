# 运行结果目录

该目录只存放可再生成的运行产物，源码与课程资料不依赖其中任何文件。结果按结论资格而非按
“跑了第几次”组织，避免多个相似目录被误认为并列正式版本。

```text
verified/course-eplace-v1/                唯一可交付、已复核的课程结果
diagnostics/course-eplace-v1/
  cuda-validation/                         CPU/CUDA 数值一致性检查
  cuda-performance/                        已整理的单卡速度与显存记录
scratch/                                   新试跑的默认输出；不作为证据
archive/pre-course-eplace-v1/              旧密度口径历史
archive/cuda-development/                  已知无效、替代的 CUDA 开发记录
archive/pre-freeze-diagnostics/            冻结前正确但未整理的重复诊断
```

只有 `verified/course-eplace-v1/` 可直接用于课程报告。所有可比较结果的 `overview.txt` 都必须包含
`density_metric=course_eplace_v1`；详细规则见 [`docs/verification.md`](../docs/verification.md)。

运行新实验时，先写入 `out/scratch/<purpose>/`。只有完成参数、密度口径、合法性和可复现性复核后，才
将一份明确命名的结果提升到 `diagnostics/` 或 `verified/`。同一结论只保留一个已整理目录；被替代的
结果移动到 `archive/`，不删除原始证据也不混入当前结论。

CUDA 后端发现的已知错误位于 `archive/cuda-development/known-invalid-single-pin-gradient/`；早期但已被
替代的 smoke 位于 `archive/cuda-development/superseded-smoke-runs/`。二者都不能用于速度或质量结论。
