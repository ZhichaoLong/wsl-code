# CPU 因子后端规避验证（2026-09-10）

`src/krylov.cu` 将 ILU/LU 的因子后端明确设为 `MATSOLVERPETSC`。全局算子仍为 `seqaijcusparse`、向量仍为 `seqcuda`；实际因子为 `seqaij`，数值分解/三角回代在 CPU。PETSc 同步数值与向量镜像，PC/KSP 管理并销毁因子，没有手工创建 CSR 或每步重建 KSP。

原项目的 `MatCUSPARSESetUseCPUSolve` 不能直接覆盖本机 PETSc 的 natural + ILU(0) 专用 GPU 分解路径。此修改规避整个 cuSPARSE 因子更新路径；对照测试支持该规避有效，但没有定位到 cuSPARSE 库内部的具体泄漏分配点。

## 固定对象重复更新测试

MS4 k=1、实际 8×8，同一方程/KSP、固定稀疏结构，160 次更新系数和求解，从零初猜解已知解系统。线性容差 1e-11；第 21 次开始统计预热后窗口。两个进程串行运行，均启用 PETSc malloc_debug/malloc_dump/objects_dump。

| 后端 | 首次 RSS MiB | 末次 RSS MiB | 预热后增长 MiB | 预热后显存变化 MiB | 平均 KSP 秒 |
|---|---:|---:|---:|---:|---:|
| 原 cuSPARSE 因子 | 483.445 | 513.164 | 26.141 | 0 | 0.07736 |
| PETSc CPU 因子 | 407.695 | 407.836 | 0.043 | 0 | 0.06933 |

两者 PETSc 跟踪的存活分配均不增长，故该计数单独不能检测原后端的 RSS 问题。修复后最大真实残差 9.32e-13。正常 Finalize 无遗留代数对象/分配报告。原始日志 `repeat_baseline.log`、`repeat_cpu.log`，机器可读统计 `memory_comparison.json`。该小网格计时不代表大网格加速比。

## 数值回归

- Darcy 1/2 与 MS 1/2/3/4，MS 为三步 BDF2：全部成功，新旧全局解最大差 1.679e-13。逐项核对质量、计算域/物理域误差与无真解的 null 字段，见 `checks.log` 和 `baseline/`、`cpu/`。
- MS4 实际 16×16、k=1、16 步、dt=0.00625、T=0.1、Picard 1e-6：误差与此前正式运行一致；第 4 至 16 步 RSS 增长 0.003906 MiB。推进 18.929 秒，其中 KSP 18.286 秒。见 `time_cpu.log`、`cpu/time_ms4/`。
- 可选 LU 与 ILU(1) 三步 MS4 通过，实际因子均为 CPU seqaij，见 `factors_cpu.log`。原全 GPU 后端的历史性能数据不适用于当前后端。
- 所有主函数重新编译成功，见 `build_all.log`；主函数输出配置更新后另编译 `build_run_main.log`。沙箱编译期间 OpenMPI 输出 socket 权限提示，但编译链接退出码为 0，真实 GPU 测试均在沙箱外成功。

临时 `tests/test_stage.cpp` 和测试构建产物在通过后删除，保留日志、报告、参考结果和修复前 `krylov_before.cu`。用户此前停止的六层运行未被视作通过；当前重新授权的正式六层运行另见 `output/MS_triblock_bdf2_k1/`。
