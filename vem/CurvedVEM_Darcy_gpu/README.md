# CurvedVEM_Darcy_gpu

基于 PETSc/CUDA 的混合 H(div)–L² 虚拟元工程。原 `../CurvedVEM_Darcy` 保持原样；本目录的 CPU 网格、几何映射与算例模块来自该目录的当前快照。GPU 实现覆盖 Darcy 算例 1/2、三组分 Maxwell–Stefan 算例 1/2/3/4、Euler/BDF2、法向通量边界和误差/质量计算。

代数层使用 PETSc `Mat/Vec/KSP/PC`，默认 GPU COO 稀疏装配、PETSc FGMRES 与 CPU ILU(0) 因子（规避 cuSPARSE 重复更新时的主机内存增长）；MS 用一次 ND 图排序，Darcy 用通量先消元的 natural 顺序。详细对象所有权、GPU 数组访问和模式选择见 [PETSc GPU 重构](docs/PETSc_GPU重构.md)。

优先阅读：[数学与算法](docs/数学与GPU算法.md)、[GPU 分配与内存](docs/GPU分配与内存.md)、[验收与性能](docs/验收与性能.md)、[工作日志](docs/工作日志.md)。

## 独立主函数：一次运行一个算例

主函数全部放在 `main/`，`src/` 只放公共求解与实验工具。参数、网格序列、终止时间和输出名称直接在对应主函数中修改；不需要附加运行参数。

| 主函数 | 内容 | 默认网格参数 n | 默认时间设置 |
|---|---|---|---|
| [Darcy_curved_main.cpp](main/Darcy_curved_main.cpp) | Darcy 1 曲边压力/通量收敛阶 | 1、2、4、8 | 稳态 |
| [Darcy_straight_main.cpp](main/Darcy_straight_main.cpp) | Darcy 2 恒等映射收敛阶 | 1、2、4、8 | 稳态 |
| [MS_curved_bdf2_main.cpp](main/MS_curved_bdf2_main.cpp) | MS 2 正弦曲边制造解收敛阶 | 2、4、8、16 | T=0.1，dt 随 h、k 调整 |
| [MS_triblock_bdf2_main.cpp](main/MS_triblock_bdf2_main.cpp) | MS 4 Q8 制造解收敛阶 | 2、4、8、16、32、64 | T=0.1，dt 随 h、k 调整 |
| [MS_discontinuous_main.cpp](main/MS_discontinuous_main.cpp) | MS 1 间断初值扩散/守恒 | 8 | T=0.5，目标 dt=0.001 |
| [MS_half_annulus_main.cpp](main/MS_half_annulus_main.cpp) | MS 3 半圆环扩散/守恒 | 8 | T=0.5，目标 dt=0.001 |

例如只编译并运行 MS 算例 2：

```bash
cd /home/lzccode/code/vem/CurvedVEM_Darcy_gpu
make -j4 MS_curved_bdf2
./build/MS_curved_bdf2
```

`make -j4` 只编译所有独立入口，不执行算例；`make list` 显示入口名称。新建 `main/名字_main.cpp` 后，Makefile 自动生成 `build/名字`。`make run` 只启动 MS 2 的网格实验；可选旧批量功能保留为 `make run-all`，不会在正常编译或独立算例运行时调用。

当前默认 k=1，可在主函数修改到 0..4。`meshes={32}` 表示本次只计算这一层；只有一层时无法计算空间收敛阶，表中显示 `-`。`square_nxn.msh` 实际是 `(2n)×(2n)` 个单元，程序读取实际网格后测量尺度，不从文件名猜测单元数。Q8 算例的网格序列从 n=2 开始。

需要 C++17、MPI 编译器、CUDA Toolkit 和启用 CUDA 的双精度实数 PETSc。本机 PETSc 路径为 `/home/lzccode/petsc`，架构为 `arch-linux-c-opt`，可在 Makefile 配置。Makefile 默认 CUDA 13.0、`sm_120`；更换 GPU 时修改构建配置后重编译，不将算例参数放入编译命令。

## 时间步与空间尺度

制造解实验使用同一终止时间 T。按照原 CPU 主函数，默认计算域尺度为 `h=sqrt(area/Ncells)`，本项目结构网格上等于 `1/(2n)`；同时输出实际最大单元直径 `h_max`。对局部加密或非均匀多边形网格，可在 main 选择 `MeshScale::MaxDiameter`，并确认所选网格确实构成加密序列。

BDF2 取 `dt_target=C*h^((k+1)/2)`，Euler 取 `dt_target=C*h^(k+1)`，默认 C=0.1。该选择匹配时间误差 `O(dt^p)` 与空间误差 `O(h^(k+1))` 的阶，不保证任意粗网格或不光滑解已进入理论渐近区。更严格分离空间误差时可减小 C，并比较结果是否稳定。

计算 `Nmin=ceil(T/dt_target)`，BDF2 至少两步，再选择不小于 Nmin 的最小 `2^a*5^b` 形式步数 N，最后令 `dt=T/N`。这样不放大目标步长，所有网格在同一时刻比较误差，也不向 T 增加半步。达到 `max_steps` 限制时明确报错，不使用错误的时间尺度继续计算。`smooth_steps=false` 可只使用向上取整的整数步数。

在主函数将 `study.time.policy` 改为 `FixedTarget` 可保留固定目标时间步的实验；细网格上要留意时间误差对空间阶的影响。MS 1/3 无解析解，默认固定目标步长做扩散和守恒实验，不报告伪造的解析误差/收敛阶。

## 终端与文件输出

每个网格开始时显示文件名、实际单元数、h、h_max、目标 dt、实际 dt 和步数。运行中输出选定时间步的 Picard/FGMRES 次数、残差、单步与累计墙钟耗时；每层结束输出所有组分的计算域/物理域标量、通量与散度误差，并立即刷新累计误差/收敛阶表和计时表。MS 1/3 改为逐组分的初始质量、最终质量与漂移。

累计表默认使用**计算域绝对 L2**，与原 CPU `MSSolver::computeConcentrationL2Error/computeFluxL2Error` 一致。可在 main 选择 `ErrorDomain::Physical`；每次实验始终使用同一范数。阶为 `log(e_old/e_new)/log(h_old/h_new)`，首层/无真解/不可比较时为 `-`，真实零阶、负阶均照实保留。

以 MS 2 为例，main 明确指定：

```cpp
const std::string output_root = "output/MS_curved_bdf2_k" + std::to_string(k);
study.table_file = output_root + "/convergence.csv";
study.terminal_file = output_root + "/results.txt";
// 每一层的独立输出目录：output_root + "/mesh_" + label。
options.save_solution = false; // 收敛实验默认不下载/写出最终大向量。
options.error_interval = 0;    // 仅最终步计算误差；设为 10 可每 10 步诊断。
options.print_interval = 10;   // 首步、每 10 步和最终步输出进度。
```

`convergence.csv` 按网格/组分保存两种域的原始误差、选定范数的阶、质量、dt/h、残差及耗时。`results.txt` 保存实验设置和逐层累计表。后续网格失败时停止当前实验，保留已完成表格并记录失败原因。总耗时使用 `steady_clock` 墙钟，包含读网格、GPU 初始化/准备、求解、后处理、保存和资源释放；准备、迭代、误差及解保存子计时单独列出，不包含所有调度/日志开销，因此不应要求子计时之和恰好等于总耗时。

每层仍有独立的参数、迭代日志、history 和 JSON 报告。原始最终解只在 `save_solution=true` 时生成。修改参数后重复使用目录会更新相应报告；若之前保存过解、后来关闭保存，旧解文件不会自动删除，不应将旧解当作当次结果。

| 方程 | 编号 | 物理/几何内容 |
|---|---:|---|
| Darcy | 1 | 正弦压力真解，正弦曲边映射，K=I |
| Darcy | 2 | 同一解析解，恒等映射，K=I |
| MS | 1 | 正弦曲边，分块间断初值，无源零通量 |
| MS | 2 | 正弦曲边，三组分制造解 |
| MS | 3 | 半圆环，径向分层初值，无源零通量 |
| MS | 4 | 原 TriBlockSwirl 逐单元 Q8 映射，制造解 |

输出文件名在 main 中的 `options.files`（或 `common.files`）逐项设置。默认包含：

- `parameters.txt`：实际生效的完整配置，含解析后的积分点数、时间、容差、GPU 分片参数及全部输出文件名。
- `iterations.log`（兼容入口为 `run.log`）：GPU 资源、内存量、每次 Picard 与 FGMRES 真实残差。
- `history.csv`：逐时间步、逐组分的计算域/物理域误差、散度误差与物理质量。
- `solution.csv`：完整全局自由度；排列与原 CPU 相同——各组分通量块在前，随后各组分浓度块。Darcy 最后一个自由度为压力均值乘子。
- `report.json`：最终状态、误差、初始/最终质量、耗时和矩阵缓存字节数。无真解的 MS 1/3 不报告伪造的 L² 真解误差，JSON 对应值为 `null`，CSV 为 `nan`。

算例 1/3 应比较各组分 `initial_mass` 与 `mass`；制造解各组分的质量可以随时间变化。FGMRES 或 Picard 不收敛时返回非零退出码，不写 `converged` 报告。重复使用同一输出目录会覆盖上次文件。

## 保留的性能及兼容入口

[main/vem_gpu_benchmark_main.cpp](main/vem_gpu_benchmark_main.cpp) 保留源码配置性能计时，`make benchmark` 运行它；`benchmark_only=true` 仅测算子，不代表 PDE 求解收敛。

[main/vem_gpu_main.cpp](main/vem_gpu_main.cpp) 保留原单网格示例；[main/vem_gpu_all_main.cpp](main/vem_gpu_all_main.cpp) 为可选旧批量入口。日常实验使用上方按算例命名的主函数。

[main/vem_gpu_cli_main.cpp](main/vem_gpu_cli_main.cpp) 保留已有脚本的命令行接口：

```bash
./build/vem_gpu_cli --equation ms --example 4 --scheme bdf2 \
  --steps 3 --dt 0.001 --output output/triblock_cli
```

`basis_ms` 为基础矩阵重建；`update_ms` 为初值模式直接积分并构造预条件对角；`picard_update_ms` 为浓度多项式模式的更新；`apply_ms` 为一次带边界 mask 的 matrix-free 算子。计时先预热，CUDA event 包括这组调用之间的主机调度间隙。`benchmark_only=true` 不求解 PDE。

MS 多项式模式使用精确等价的几何矩预积分收缩，额外缓存预算为 256 MiB，超过后自动使用直接 GPU 积分。基础 G/B/H/H* 构造后释放；正式迭代只保留 D/P/W/H⁻¹ 及方程因子。默认在 GPU 生成 COO 单元条目，并由 PETSc 形成原顺序与排序后的 CSR；matrix-free MatShell 模式不形成全局 CSR。

## 实现边界

- 单 GPU、双精度，支持 k=0..4。k=3 默认 12 点三角形规则；k=4 使用新增 36 点 Duffy 规则。原 CPU 的 k=4 完整基础矩阵路径存在 P5 基限制，因此四阶采用独立再现性和直接积分等价验收，而不宣称与原 CPU 四阶完整求解逐项对照。
- CPU 保留网格读取/三角剖分/固定积分几何与 J；G/B/D、投影、系数与右端积分、边矩、Krylov 向量、时间历史与误差归约在 GPU。控制循环和小 Hessenberg 系统在 CPU。
- 当前前端提供原算例的全部法向通量边界路径。新增不同渗透张量、组分数、PDE 数据或压力 Dirichlet 类型，需要扩展设备方程函数；不能把任意 CPU `std::function` 直接传进 CUDA 核。
- 分片是按边数分桶后的 **kernel launch 分片**，不是磁盘/CPU–GPU 的显存外流式计算。网格因子和 Krylov 向量仍需容纳于显存。
- 默认预条件为 PETSc CPU ILU(0)，稀疏矩阵乘仍在 GPU，MS 使用 ND 排序、Darcy 使用 natural 消元顺序；MatShell 模式保留近似对角预条件。可在 main 设置 `petsc_pc`、`petsc_ordering`、`ilu_levels` 和 `restart`。本轮不实现多 GPU/MPI 分布式求解。
- 本机 WDDM 调试接口未开启，Compute Sanitizer 未能工作；不能将普通 GPU 运行通过等同于 sanitizer 通过，具体记录见验收文档。

## 文件组织与测试约定

`HdivGpu` 是几何/投影父类；`MixedGpu` 实现混合方程算子，`DarcyGpu`/`MSGpu` 为方程入口；`Krylov` 通过 PETSc KSP 管理 FGMRES，应用代码不维护 Hessenberg/Arnoldi 或自写 BLAS 向量代数。CPU 原模块保留在 `mesh/`、`core/`、`lib/`、`examples/`，只补充注释，数学代码保持原样。所有正式 C++/CUDA 源文件与头文件都有中文模块说明；GPU 核额外解释矩阵维数、局部/全局方向、共享内存布局、时间历史与误差归约。`.clang-format` 固定四空格缩进和分行格式。

按本次要求，各阶段临时测试源在通过后删除；保留测试结果、实际运行命令与报告，不把测试逻辑混入正式主程序。验收记录见 `docs/validation/` 和 `output/`。重新运行上述正式命令可以重现求解、误差、质量与性能结果。

构建依赖已启用 CUDA 的双精度实数 PETSc 与 MPI 编译器。Makefile 默认 `PETSC_DIR=/home/lzccode/petsc`、`PETSC_ARCH=arch-linux-c-opt`；算例参数仍在 main 中设置。
