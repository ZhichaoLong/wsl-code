# PETSc GPU 重构验收（2026-09-10）

本轮把持久矩阵、向量与代数求解迁至 PETSc，保留原 CPU 网格及混合 H(div)–L² 数学。最终默认使用 `MATSEQAIJCUSPARSE` / `VECSEQCUDA` / PETSc FGMRES / cuSPARSE ILU(0)。Darcy 用 natural 顺序，MS 用 ND 排序；ND 只在 CPU 建一次图，数值装配、分解、三角求解及向量置换在 GPU。

环境：RTX 5060 Laptop（26 SM、约 8 GiB），CUDA 13.0；PETSc `/home/lzccode/petsc/arch-linux-c-opt`，版本 `v3.25.1-150-ga03911edb92`，双精度实数、32 位 PetscInt、CUDA、PETSC_USE_LOG，非 debug 构建。系统 OpenMPI 非 GPU-aware，单进程启动显式设置 `-use_gpu_aware_mpi 0`。

## 最终数值检查

| 检查 | 结果 | 证据 |
|---|---|---|
| k=0..4、两类方程、四边形/三角形/悬挂边、两次系数更新 | 32 组通过；COO 与 MatShell 相对差 ≤4.109e-15 | [operators_final.log](operators_final.log) |
| `R A Rᵀ` 与 VecScatter 正反置换 | 相对差 ≤4.131e-15；覆盖 ND/RCM/natural | 同上 |
| 投影再现 `P D=I` | 最大偏差 5.224e-12 | 同上 |
| Darcy 1/2 稳态、MS 1..4 三步 BDF2 | 全部通过，误差/质量与旧实现一致 | [regression_final.log](regression_final.log) |
| MS4 两步 Euler，稀疏与 MatShell | 两条路径均通过，结果一致 | 同上、operators_final.log |
| k=4 Darcy2/MS2 完整求解 | 均通过，与原 k=4 报告比较通过 | [high_order_final.log](high_order_final.log) |
| 115 次线性求解真实残差 | 最大 9.965e-13，均小于用户 1e-11 | regression_final.log |
| MS1/MS3 三步守恒 | 最大组分质量漂移分别 0、8.882e-16 | regression_final.log |

误差对照用 `abs(new-old) <= 5e-7*max(abs(old),1e-3)`；质量使用绝对差 ≤1e-8，守恒另按 ≤1e-9 检查。k=4 Darcy 旧报告压力均值约 −4.381e-10，新结果更接近零，因此含质量项的最大缩放差约 4.381e-7，不能把它误标成所有误差范数的相对变化。输出在 `output/petsc_validation/final/`。

## 首步性能

同一 MS4、k=1、BDF2 首步（Euler 启动），`dt=0.1/side`、Picard 容差 1e-6、真实线性容差 1e-11、restart=320。新测试每层只运行一步，maxit=4000；旧长跑上限为 20000，双方均按容差收敛，未触及各自上限。关闭解保存。

比较的是系数更新、右端、全部 Picard/线性迭代的首步墙钟时间，包含新路径首次 COO/CSR 与排序/PC 准备；不含误差积分与初始网格/基础准备。旧基线来自已停止长跑的 `build/MS_triblock_bdf2_k1_run.log`，新记录见 [performance_final.log](performance_final.log)、[performance_final.csv](performance_final.csv)。不是多次重复平均，也不是完整 T=0.1 的端到端加速。

| 实际网格 | 自由度 | 旧首步 s | PETSc 首步 s | 比值（旧/新） | 旧线性迭代总数 | 新线性迭代总数 |
|---|---:|---:|---:|---:|---:|---:|
| 32×32 | 31104 | 9.080641 | 3.614255 | 2.51× | 8670 | 626 |
| 64×64 | 123648 | 46.540158 | 20.339129 | 2.29× | 12926 | 1943 |
| 128×128 | 493056 | 264.916778 | 135.569404 | 1.95× | 20093 | 4039 |

新测试总墙钟（含准备、误差、析构等）分别为约 3.900、20.553、136.306 秒。表中累计迭代数包含一个时间步的多个 Picard 线性系统；4039 大于单次上限 4000 并不意味着超限。

新三层最终非线性残差分别约 4.315e-8、1.195e-7、3.715e-9，均满足 Picard 容差；它们不是线性残差。性能 CSV 的 residual 列也是非线性残差。原本停止的完整 128×128 长跑没有恢复，本轮不提供它的 T=0.1 最终误差或收敛阶。

## 生命周期与内存

[最终内存日志](memory_final.log)：64 单元、4 轮预热之后连续 24 轮创建/销毁。每轮执行 4 次方程系数、右端、Mat 数值更新、KSP 求解，并交替计算误差；覆盖默认 ND、RCM 和 MatShell。

- 预热后 PETSc 跟踪的当前主机分配始终 **54192 B**。
- 预热后 `cudaMemGetInfo` 空闲显存始终 **7301234688 B**。
- 每轮 `PetscObjectsDump` 检查无 Mat/Vec/KSP/PC/IS/VecScatter/PetscSF 遗留；`final_objects_*.log` 保留快照。`final_objects_-1.log` 的活 Vec 正控制证明检测器能发现未释放对象。
- 覆盖借用 CUDA 数组期间抛异常、派生构造失败、故意 KSP 不收敛，以及同尺寸不同方程间往返复用 KSP/置换的清理。
- 初始化用 `-malloc_debug -malloc_dump -objects_dump all`；正常 Finalize 后未报告未释放对象/分配。Finalize 前的 PetscContainer/PetscDeviceContext 是 PETSc 上下文，不冒充为应用矩阵泄漏。

这说明上述重复和异常路径中未观察到泄漏，不是对任意 GPU 故障和任意参数的绝对证明。Compute Sanitizer 在本 WSL/WDDM 环境此前因 debugger interface 未启用而未能初始化，不能把该工具的未运行检查当成通过。现有证据是 PETSc 对象/分配跟踪、CUDA 显存稳定性和实际数值回归。

## 实现与复现

关键实现：`include/petsc_gpu.h`（所有权与 Get/Restore）、`src/krylov.cu`（KSP/PC/真实残差）、`src/equation_gpu.cu`（GPU COO 与 PETSc 图/数值复用）、`src/solver_driver.cpp`（Vec 历史与输出）。完整规则见 [PETSc_GPU重构.md](../../PETSc_GPU重构.md)。

`make -j4` 已编译全部 10 个 main，见 [build_all_final.log](build_all_final.log)。编译返回 0，无 C++/CUDA warning/error；沙箱内 MPI 编译器的 `opal_ifinit: socket() failed errno=1` 属环境信息，实际 MPI/GPU 运行已在沙箱外完成。

正式运行仍在 `main/*_main.cpp` 配置。有限首步性能可在临时 main 中设置 `example=4,k=1,steps=1,dt=.1/side,picard_tol=1e-6,restart=320,maxit=4000,save_solution=false`，逐层用 `square_(side/2)x(side/2).msh` 调用 `run_case`。单例小规模回归也可使用兼容入口，例如：

```bash
./build/vem_gpu_cli --equation ms --example 4 --steps 3 --output output/replay_petsc
./build/vem_gpu_cli --equation darcy --example 2 --k 4 --mesh mesh/data/square_1x1.msh --restart 320 --output output/replay_petsc_k4
```

本轮通过的临时测试源码已按要求删除；日志、数值报告和旧二进制基线保留。早期 natural、RCM+GMRES、RCM+FGMRES 超时、RCM+ILU(1) 失败及 Darcy+ND 零主元的日志均保留用于解释默认排序选择，不将其列为成功验收。

最终链接产物另通过 CLI 的 MS4 三步冒烟运行及 `--help` 正常退出检查，日志为 `cli_smoke_final.log` / `cli_help_final.log`。已断言 JSON 的 PETSc 类型、排序、COO/CSR 规模有效，并检查临时测试源已清理；最终源码指纹见 `source_sha256_final.txt`。
