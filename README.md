# MyPlacement

`MyPlacement` 是“工业软件创新训练 III”的一个可构建、可测试的 C++ 自动布局器。它以
BookShelf 基准格式为输入，依次完成数据解析、三种初始布局、全局布局、标准单元合法化、
位图渲染和 GDSII 导出。

它的目标不是复刻商业 EDA 工具，而是把课程中的完整布局主线实现为可运行的软件：

```text
BookShelf (.aux/.nodes/.nets/.pl/.scl/.wts)
        ↓
PlacementDatabase
        ↓
random / cluster / quadratic initial placement
        ↓
smooth wirelength + DCT-Neumann density-field global placement
        ↓
row-based legalization
        ↓
BMP (CImg) + GDSII + CSV/TXT reports
```

## 当前稳定基线

项目只有一条正式算法路径：`adaptive + neumann + course_eplace_v1`。它是课程报告、
`out/verified/` 和后续创新的共同基线。

- `cpu` 与 `cuda` 是同一条算法路径的数值执行后端，不是两份布局器；运行时通过
  `--compute-backend` 选择。
- `legacy` 优化器与 `periodic` 密度场只保留为可复现的 A/B 回归基线，不能作为正式交付配置。
- `build/`、`build-cuda/` 和 `build-asan/` 都是可再生成的构建产物，不属于源代码版本分支。

这一区分使项目既保留可验证的对照能力，又不会出现多个“都像正式版本”的实现。

## 项目结构

```text
Myplacement/
├── CMakeLists.txt                 # 构建、依赖、测试入口
├── include/myplacement/
│   ├── core/Geometry.hpp           # 二维点、矩形、重叠面积
│   ├── model/PlacementDatabase.hpp # Module / Pin / Net / SiteRow
│   ├── io/BookshelfParser.hpp      # BookShelf 前端
│   ├── metrics/Metrics.hpp         # HPWL、密度网格、溢出率
│   ├── placement/                  # 初始布局、全局布局、合法化接口
│   └── export/                     # BMP 与 GDSII 导出接口
├── src/
│   ├── model/                      # 统一内部数据库实现
│   ├── io/                         # 解析器实现
│   ├── metrics/                    # 指标实现
│   ├── placement/
│   │   ├── GlobalPlacer.cpp         # 公开全局布局 facade / 策略分发
│   │   ├── GlobalPlacementInternal.hpp # placement 私有算法入口
│   │   ├── GlobalPlacementSupport.hpp/.cpp # adaptive 与 legacy 共用的密度/filler 服务
│   │   ├── AdaptiveGlobalPlacer.cpp # 闭环优化器实现
│   │   ├── LegacyGlobalPlacer.cpp   # 保留的开环 A/B 基线
│   │   ├── CudaPlacementBackend.*    # CUDA 后端接口与无 CUDA 时的安全 stub
│   │   ├── cuda/CudaPlacementBackend.cu # GPU 线长、密度与 DCT 数值内核
│   │   └── ...                      # 初始布局、DCT 场、合法化
│   ├── export/                     # CImg / GDSII 实现
│   └── app/main.cpp                # 命令行编排层
├── docs/                           # 文档导航、架构、算法与验证基线
├── examples/tiny/                  # 可直接运行的自包含 BookShelf 示例
├── resources/course_materials/     # 老师提供的完整课程资料、PDF、代码与大规模基准
├── tests/
│   ├── fixtures/                   # 自包含微型 BookShelf 用例
│   ├── TestAssertions.hpp           # 测试断言工具
│   ├── TestCases.hpp                # 测试组入口声明
│   ├── DensityFieldTests.cpp        # DCT/Neumann 数值测试
│   ├── DensityModelTests.cpp        # 课程密度口径测试
│   └── PlacementFlowTests.cpp       # 解析到导出的流程测试
├── out/
│   ├── verified/                   # 唯一可交付的结果
│   ├── diagnostics/                # 已整理的验证与性能诊断
│   ├── scratch/                    # 默认试跑输出，不参与结论
│   ├── archive/                    # 只读历史追溯，不参与结论
│   └── README.md                   # 结果提升与归档规则
└── third_party/CImg/CImg.h         # 课程材料提供的单头文件绘图库
```

层之间只通过头文件中定义的小接口通信。布局算法不读取文本文件，解析器也不依赖任何
优化器；未来增加 DEF 读取器时，只需要再实现一个“输入格式 → `PlacementDatabase`”的前端。

## 已实现的课程任务

| 课程任务 | MyPlacement 中的实现 |
|---|---|
| 1. Linux / C++ 工程 | CMake、Ninja、CTest、GDB、严格编译警告 |
| 2. BookShelf / DEF / LEF 理解 | 运行核心使用 BookShelf；内部数据库为扩展 DEF/LEF 前端预留边界 |
| 3. BookShelf 解析 | `.aux/.nodes/.nets/.pl/.scl/.wts`，并核对声明的节点/网络/引脚数 |
| 4. 初始布局 | 随机、连接聚类、重加权二次解析三种方法，可输出比较 CSV |
| 5. 全局布局 | 平滑线长梯度 + DCT/Neumann 频域密度场 + 自适应闭环 Nesterov + filler |
| 6. 结果展示 | CImg 生成 BMP；自包含 GDSII 矩形导出，可用 KLayout 打开 |
| 7. 合法化 | 宏块贪心避障 + 标准单元行分配 + 行内等距/二次（isotonic）压缩 |

### 三种初始布局

- `random`：在核心区域内随机放置，是程序正确性的基线。
- `cluster`：把有连接关系的可移动单元聚成有上限的簇，在簇图上松弛后再展开。
- `quadratic`：以网络引脚对构造稀疏二次方程，采用 Eigen 的 `BiCGSTAB` 迭代求解横、纵坐标。
  高扇出网络自动退化为星形近似，避免无界的二次展开。

### 全局布局

全局布局最小化“平滑线长 + 密度惩罚”。线长采用可求导的 log-sum-exp 近似；密度将核心区域
划分为 `bin`，默认使用 DCT 的闭边界 Poisson 求解：先去除不可解的平均密度，求
`-∇²φ = ρ - mean(ρ)`，并令四条边界的法向场为零。于是过密区域产生向外的电场，但不会从
芯片左边“穿出”后又从右边进入。DCT-II 做密度分析，DCT-III 重建势，两个电场分量分别用
DCT/DST 混合重建；插值时按场的奇偶性生成镜像点，所以在边界处严格得到零法向场。

`--density-field periodic` 保留旧的周期性 FFT 版本，只用于同条件 A/B 对照；默认的
`neumann` 才是课程设计中有限芯片区域的物理模型。filler 是只参与密度计算、不会写入最终
设计的虚拟单元。密度采用老师讲义的 `course_eplace_v1` 口径：标准单元按完整面积，宏、固定端子、
filler 和 `SiteRow` 外暗区按目标密度缩放；固定/暗区/filler 不会进入归一化溢出率分母。完整定义、
filler 方程和验证见 [docs/density_model.md](docs/density_model.md)，闭边界 DCT 的离散公式见
[docs/dct_neumann.md](docs/dct_neumann.md)。

默认的 `adaptive` 全局优化器则根据实际的目标函数、HPWL、溢出率和梯度变化闭环地调节步长、
密度惩罚、线长平滑度及 Nesterov 动量；更新变坏时会回溯并在必要时重启动量。旧的固定参数版本
仍以 `--global-optimizer legacy` 保留，用于公平 A/B 对照。原理、诊断字段和已复现的大基准结果见
[docs/adaptive_global_optimizer.md](docs/adaptive_global_optimizer.md)。默认迭代预算为 280，但一旦真实
设计密度连续满足约束就会提前停止；实验若需要严格同预算对照，应显式传入 `--iterations`。

### 合法化

标准单元先在少量邻近行中选择有剩余容量的候选槽，再在每一行中做加权 isotonic 回归。这等价于
在“无重叠、保持顺序”的约束下最小化与全局布局目标位置的偏移，属于 Abacus 思路的行内二次优化。

## 构建

已验证的依赖：GCC 11+、CMake 3.22+、Eigen 3.4、FFTW3、pthread。CImg 已随项目附带。

`build/` 是标准 CPU Release 构建目录；`build-cuda/` 是同一份源码的可选 CUDA Release 构建；
`build-asan/` 仅用于内存安全验证。它们不是实现分支。

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 16
ctest --test-dir build --output-on-failure
```

在本服务器启用单卡 CUDA 后端（仍是同一个项目、同一个可执行程序接口）：

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMYPLACEMENT_ENABLE_CUDA=ON
cmake --build build-cuda --parallel 16
ctest --test-dir build-cuda --output-on-failure
```

需要调试内存问题时：

```bash
cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DMYPLACEMENT_ENABLE_SANITIZERS=ON
cmake --build build-asan --parallel 16
ctest --test-dir build-asan --output-on-failure
```

## 运行

查看命令行帮助：

```bash
./build/myplace --help
```

对项目内的 `thin1` 基准完整运行，并比较三种初始布局：

```bash
./build/myplace \
  'resources/course_materials/任务2参考/BookShelf格式的解析/thin1/thin1.aux' \
  --output out/scratch/thin1-manual-80 \
  --initial all --iterations 80 --bins 32 --seed 2026
```

快速试运行可使用更小的自包含样例：

```bash
./build/myplace examples/tiny/tiny.aux \
  --output out/scratch/tiny-quick-run --initial all
```

对项目内的 `adaptec1` 大基准运行（不生成大尺寸图像和 GDSII 时更适合快速验证）：

```bash
./build/myplace \
  'resources/course_materials/任务2参考/BookShelf格式的解析/adaptec1/adaptec1.aux' \
  --output out/scratch/adaptec1-manual-120 \
  --initial quadratic --iterations 120 --bins 64 \
  --density-field neumann --no-bmp --no-gds
```

也仍可对任意外部 BookShelf 设计运行：

```bash
./build/myplace \
  /path/to/your-design.aux \
  --output out/scratch/your-design \
  --initial quadratic --iterations 120 --bins 64 --seed 2026
```

项目不依赖绝对路径：老师提供的完整资料包已保存在 `resources/course_materials/`，其中包含
`thin1`、`adaptec1`、`adaptec4` 三套 BookShelf 基准。`examples/tiny/` 则是快速演示用的小样例。
外部 `.aux` 输入仍受支持，但不再是运行项目自带大基准的前提。

常用选项：

```text
--initial random|cluster|quadratic|all
--iterations <count>       全局布局最多迭代次数
--quadratic-iters <count>  二次布局的重加权外层迭代次数
--quadratic-solver-iters <count>  每次稀疏线性求解的迭代上限
--bins <count>             密度网格横、纵尺寸
--density-field neumann|periodic  默认闭边界 DCT；周期性 FFT 仅作 A/B 基线
--global-optimizer adaptive|legacy  默认闭环自适应；legacy 用于开环基线对照
--compute-backend cpu|cuda|auto  默认 CPU；CUDA 仅支持 adaptive + neumann
--gpu-device <1..4>       本共享服务器允许的 GPU 编号，显式 CUDA 时建议先检查空闲卡
--gpu-memory-limit-gib <n>  CUDA 显式分配上限，范围 1..40，默认 40
--target-density <value>   目标密度
--no-global                只运行初始布局
--no-legalize              不运行合法化
--no-bmp / --no-gds        关闭相应导出
--parse-only               只解析并检查输入
```

未传入 `--output` 时，程序默认写入 `out/scratch/`。这里适合个人试跑，且不会与已整理的
`diagnostics/` 或正式 `verified/` 结果混在一起。

## 输出文件

```text
01_initial_random.bmp      初始布局比较图片（使用 --initial all 时）
01_initial_cluster.bmp
01_initial_quadratic.bmp
02_initial_selected.bmp    选中方法的初始布局
03_global.bmp              全局布局结果
04_legalized.bmp           合法化结果
initial_comparison.csv     三种初始方法的 HPWL、时间、迭代次数
global_history.csv         每轮目标函数、HPWL、两类课程口径溢出率、步长、曲率、回溯、重启和检查点
placement.gds              以不同层表示标准单元 / 宏 / 固定单元的 GDSII 文件
overview.txt               最终对象数、HPWL、密度口径/分子/分母/暗区、合法性摘要
```

BMP 可直接在服务器上生成，不需要图形桌面。`placement.gds` 可下载后在 KLayout 打开；当前
GDSII 导出的是课程布局结果的矩形可视化，而不是含标准单元细节与真实布线的可制造版图。

## 规模与实现注意

`adaptec1` 约有 21 万个可移动单元和 94 万个引脚。项目始终使用稀疏矩阵、网络邻接关系和局部
密度网格；不要改为 `N × N` 稠密矩阵。解析和随机/聚类初始化可快速完成；二次解析和全局布局
的运行时间取决于迭代次数，建议先用 `thin1` 调试，再逐步增加大基准的迭代次数。

为避免实验意外占满服务器，`ElectrostaticField` 会在分配前拒绝超过 `1,048,576` 个密度 bin
的网格（命令行的方形网格即 `--bins <= 1024`）。DCT/FFTW 计划没有启用内部多线程。运行大基准
建议使用项目脚本，它会将进程限制在 0–15 号 CPU（最多 16 逻辑核）以及 16 GiB 虚拟地址空间：

```bash
scripts/run_safe_benchmark.sh build \
  'resources/course_materials/任务2参考/BookShelf格式的解析/adaptec1/adaptec1.aux' \
  out/scratch/adaptec1-safe-80 \
  --initial quadratic --iterations 80 --bins 64 --seed 2026 --no-bmp --no-gds
```

同一输入、同一参数下的 Neumann/periodic 消融可以直接运行：

```bash
scripts/run_density_field_ablation.sh \
  build \
  'resources/course_materials/任务2参考/BookShelf格式的解析/adaptec1/adaptec1.aux' \
  out/scratch/adaptec1-neumann-vs-periodic-80 \
  --initial quadratic --iterations 80 --bins 64 --seed 2026 --no-bmp --no-gds --no-legalize
```

同一输入、同一参数下的开环/闭环优化器消融可以运行：

```bash
scripts/run_optimizer_ablation.sh \
  build \
  'resources/course_materials/任务2参考/BookShelf格式的解析/adaptec1/adaptec1.aux' \
  out/scratch/adaptec1-adaptive-vs-legacy-160 \
  --initial quadratic --iterations 160 --bins 64 --seed 2026 \
  --density-field neumann --no-bmp --no-gds
```

文档入口为 [docs/README.md](docs/README.md)；更完整的设计边界和数据流见
[docs/architecture.md](docs/architecture.md)。
