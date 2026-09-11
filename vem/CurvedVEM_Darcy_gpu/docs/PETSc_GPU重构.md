# PETSc GPU 代数层

本轮将持久代数对象及求解交给 PETSc：`Mat`、`Vec`、`KSP`、`PC`、`IS`、`VecScatter`。CPU 网格读取、H(div) 数学定义与各算例保持一致。结果仍由 `main/` 中的参数控制，数学积分与误差归约仍在 GPU。

## 配置与实际后端

```cpp
vgpu::PetscSession petsc; // main 中最先创建，全部代数对象释放后才 Finalize
vgpu::SolverOptions options;
options.assembled = true;
options.petsc_pc = "ilu";
options.petsc_ordering = "auto"; // Darcy: natural；MS: nd
options.ilu_levels = 0;
```

默认全局算子为 `MATSEQAIJCUSPARSE`，向量为 `VECSEQCUDA`，线性求解为 PETSc 右预条件 FGMRES。主函数仍直接设置容差、重启维数、步数和输出目录。`PetscInitialize` 对当前非 GPU-aware OpenMPI 显式设置 `-use_gpu_aware_mpi 0`，本工程仅支持单进程单 GPU，不进行 GPU MPI 通信。

混合方程的消元顺序影响 ILU(0) 稳定性和求解效率。默认规则按方程区分：Darcy 保留 natural 顺序（先通量，再零质量压力块与均值乘子）；MS 使用 ND（nested dissection，嵌套剖分）的一次图排序。显式选择 RCM/natural/ND 可用于比较，但不能认为任意排序都适合鞍点系统。MS 稀疏流程如下：

1. CPU 从原网格自由度建立 COO 图，PETSc 计算 ND 排序 R；图仅建立一次，数学数值不在 CPU 组装。
2. GPU 计算原顺序 COO 数值。两套预分配索引分别生成 A 和 R A Rᵀ；每次仅调用 `MatSetValuesCOO(...,INSERT_VALUES)` 更新数值，保留稀疏结构。
3. PETSc `VecScatter` 在 GPU 置换右端和初猜，KSP 解排序后的方程，再将解还原到原 H(div) 自由度顺序。
4. 在已经排序的矩阵上，PC 的内部排序使用 natural，ILU(0) 数值分解与三角求解由 PETSc 在 CPU 执行；GPU MatMult 保留。
5. 接受解之前用原顺序 A 独立复算真实相对残差。KSP 内部阈值取用户线性容差的 0.1 倍。必要时最多做两次 `Aδ=b−Ax` 残差校正，计入同一个 maxit 上限；最终仍严格检查用户容差，失败返回非零状态。

`ilu_levels>=0` 和 `petsc_pc="lu"` 均显式选择 `MATSOLVERPETSC`，因子为 CPU `MATSEQAIJ`，由 PC/KSP 拥有和销毁。PETSc 在数值分解时通过 `MatSeqAIJGetArrayRead` 更新 GPU 矩阵的主机镜像，在 CPU 三角回代时同步输入/输出向量；不把全局 Mat/Vec 绑定到 CPU，也不重复创建因子或手动拷贝 CSR。每个算例首次求解后检查并打印实际因子类型 `factor_mat=seqaij numeric_factorization=CPU triangular_solve=CPU`。

这是对当前 cuSPARSE 路径主机内存增长的规避。仅设置 `MatCUSPARSESetUseCPUSolve` 不足以规避安装版本的 natural + ILU(0) 专用 GPU 分解路径，因为该分支不检查此标记。CPU 预条件会增加主机因子存储及迭代中的 CPU/GPU 传输，不能沿用旧 GPU ILU 的性能结论。原 LU/ILU(k>0) 的 shift 设置保留；不另行改变 ILU(0) 的预条件参数与真实残差门限。复现与修复测试见 [CPU 因子验证](validation/cpu_factor_20260910/README.md)。

`options.assembled=false` 保留 `MatShell` 的 matrix-free 路径，其 MatMult 调用原 GPU 单元因子作用核；Krylov 基、向量范数与正交化仍由 PETSc 管理。该路径使用原近似对角预条件与 `VecPointwiseDivide`，大网格性能弱于适合本系统的稀疏预条件，主要用于对照与内存受限场景。

## 对象、数组与释放

| 数据 | PETSc 类型/所有者 | 释放与访问规则 |
|---|---|---|
| G/H/H*/D/B/P/W/H⁻¹、AG/M、几何矩 | `PetscDense` → `MATSEQDENSECUDA` | 批量打包 Mat；单元形状/偏移由 `Cell` 描述，禁止把打包存储视图当全局算子做 MatMult |
| 解、右端、边界、历史、系数/诊断数组、COO 数值 | `PetscVector` → `VECSEQCUDA` | PETSc 分配和释放；GPU 核借用数组，不转移所有权 |
| 原顺序和排序后的稀疏矩阵 | `PetscMat` → `MATSEQAIJCUSPARSE` | 一次 COO 预分配，多次数值更新；析构 `MatDestroy` |
| 线性求解和预条件器 | `PetscKsp` → `KSP` / KSP 内部 PC | `KSPDestroy`；`KSPGetPC`、`PCFactorGetMatrix` 返回借用对象，禁止自行 Destroy |
| 排序和置换 | `PetscIs` / `PetscScatter` | `ISDestroy` / `VecScatterDestroy`，GPU 向量置换由 PETSc 管理 |
| 节点、积分几何、单元编号等 | 原 `Buffer<T>` | 不是代数 Mat/Vec；CUDA RAII 缓冲区保持原 CPU 网格数据结构 |

为了避免为数万个小矩阵创建数十万个 PETSc 对象，基础矩阵采用少量批量 Mat 存储，继续由每单元 block 的数学核处理。共享内存中的消元/积分临时数组属于 CUDA 核工作区，不是拥有独立生命周期的代数对象。PETSc 负责持久矩阵/向量的分配和求解，VEM 的积分与局部投影公式仍由专用核计算。

`data()` 开始 `VecCUDAGetArray` / `MatDenseCUDAGetArray` 借用，多个专用核可复用同一借用。`raw()`、`restore()`、`zero()`、`download()` 在 PETSc 代数操作前配对 Restore。借用期间禁止同时调用该对象的 PETSc 运算。获取/归还与默认流核之间使用同步保证可见性；MatShell 回调另外使用配对的 Read/Write 接口，异常也归还数组。

所有句柄禁止复制；析构中不抛异常。释放顺序为 KSP/Scatter 工作对象 → 方程矩阵与向量 → 基础数据 → `PetscFinalize`。会话只释放自己初始化的 PETSc，不替外部调用者 Finalize。没有依赖进程结束才能释放的全局求解器缓存。

## 显存与性能边界

MS 默认稀疏路径保留原矩阵和排序矩阵两份 CSR（Darcy natural 仅一份），以及一份 GPU COO 数值数组和 PETSc 预分配的 COO 汇总映射。相比 matrix-free 使用更多显存，换取成熟稀疏预条件器。COO 内重复共享边项由 PETSc 在 GPU 汇总；方向系数仍严格为内部反向边第 r 阶矩的 `(-1)^(r+1)`。

每个 CUDA block 负责一个单元，线程跨局部矩阵条目循环，按桶和 tile 分片 launch。局部矩阵数值写到设备 COO 数组，图与显存持久复用；不在每次 Picard 创建 Mat/KSP，也不逐项在 CPU 调用 MatSetValues。基础积分父类 `HdivGpu` 与方程派生类 `MixedGpu`/`DarcyGpu`/`MSGpu` 的职责保持分离。

报告单独保存 COO 条目数/数值字节数和两份 CSR 非零数；`basis_bytes`、`operator_bytes` 仍只表示基础/方程因子，不是全工程显存。COO 分配前检查显存预算；高阶或大多边形可能需要更多显存。超预算明确失败，不静默改用 CPU 数值装配。

## 验证记录

日志在 `docs/validation/petsc_refactor_20260910/`。最终配置的 32 组算子/置换对照、六算例、Euler/BDF2、MatShell 和 k=4 完整求解通过。MS4 的实际 32/64/128 首步通过，迭代耗时分别约 3.614/20.339/135.569 秒；这是首步比较，不是完整 T=0.1 的加速结论。具体误差、残差、生命周期及性能证据见 [验收记录](validation/petsc_refactor_20260910/README.md)。
