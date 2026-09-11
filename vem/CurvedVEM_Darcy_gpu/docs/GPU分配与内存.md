# GPU 分配与内存布局

## 调度粒度

当前单 GPU 为 RTX 5060 Laptop，26 个 SM，约 8 GiB 显存。每 SM 65536 个 32 位寄存器，shared memory 102400 字节。硬件调度单元是 warp/block，不将某个单元固定绑在某个 CUDA core。

1. CPU 从原 `StraightMeshData` 提取局部 CCW 边和全局边号，按 `nodes_per_cell` 分桶；保持原单元编号和输出自由度编号；MS 的 PETSc 稀疏求解另外维护 ND 排序，通过 VecScatter 往返置换。
2. 桶内每个单元对应一个 block；默认 128 线程，4 个 warps。每桶以最多 4096 个单元为一个 launch 分片。
3. 基础积分中，线程 t 处理 z=t,t+blockDim,… 的矩阵元素，每个元素的积分和保留在标量寄存器中。
4. 逆/投影核使用 shared memory 中的带主元消元。pivot、归一化、消元间明确 block 同步；整个单元矩阵不塞进每个线程的寄存器/栈。
5. matrix-free 中每线程处理局部向量行；阶段间同步，公共边使用双精度 atomicAdd。
6. 误差中线程按积分点分工，先 block 归约，再独立归约核计算组分范数与质量。只有最终少量标量传回 CPU。

不同桶共享相同 k，p/q 相同，n 随边数变化。数组采用 ragged offset，没有用最大边数给所有单元填充大型矩阵。SM 居留的块数量由寄存器/shared memory/硬件限制共同决定，occupancy 上限不等于实测利用率。

## 持久数据

| 数组 | 形状（每单元） | 布局/职责 |
|---|---|---|
| D | n×p | 行主序，局部自由度作用于多项式基 |
| P | p×n | 行主序，L² 多项式投影 |
| W | q×n | 散度矩 |
| H⁻¹ | q×q | 散度多项式重构 |
| AG | s×s×p×p | 组分耦合的加权 Gram；不含完整 n×n 刚度 |
| α | s | 各组分对角稳定化系数 |
| M | q×q | 物理标量质量矩阵 |
| C（可选） | q×p×p | ∫m_r g_aᵀTg_b，MS 更新的几何矩缓存 |
| T_m（可选） | q×4 | ∫m_r T 的张量矩，用于 α |
| local-to-global/sign | n | 阶次优先边矩编号和方向符号 |
| 固定几何 | 体/边积分点 | ξ/η、权重、物理坐标、J、detJ、边 t 和外法向 |

G/B/H/H* 只在构造/校验阶段保留，正式求解调用 `compact_basis()` 释放。后续 `build_basis()` 会重新分配它们，可用于基准测试。这些持久数值矩阵由批量 `MATSEQDENSECUDA` 持有，向量为 `VECSEQCUDA`，几何/编号仍是 CUDA RAII 缓冲区。MS 的 C/T_m 缓存总预算为 256 MiB，超出自动采用直接积分路径；两条路径逐阶对照通过。

四边形 k=1 时 n=11、p=6、q=3、s=3：

- 持久基础因子 174 个 double，即 **1392 B/单元**。
- 方程 AG/α/M 共 **2688 B/单元**。
- 可选几何矩 **960 B/单元**。
- 完整局部三组分混合块若存储为 (3n+3q)²，会是 **14112 B/单元**；MatShell 路径不保存它；PETSc 稀疏路径将其条目保存到 COO 值数组并汇总成 CSR。

以上只计算矩阵/因子，不含网格、积分数据、源项缓存和求解向量，不能当作全部显存用量。PETSc FGMRES 的两组 Krylov 基存储量约为 8N(2*restart+1) 字节，N 为全局自由度数；还有工作向量。源项/真解每时间步在 GPU 采样一次，系数/右端工作数组反复复用，不在每次 Picard 中重复 malloc/free。

## matrix-free 共享内存

每个单元的临时共享数组为 u(sn)、z(sp)、r(sn)、Dᵀr(sp)、Gz(sp)，合计 8s(2n+3p) 字节。k=1 三组分四边形为 **960 B/block**。稳定项按

z=Pu，r=u−Dz，t=Dᵀr，y_stab=α(r−Pᵀt)

计算，不形成 Π_dof 或 (I−Π_dof)ᵀ(I−Π_dof)。对一致项先在 p 维投影空间完成组分耦合，再乘 Pᵀ。

基础消元最多 2p² 个 double shared memory，加少量 pivot 标量。四阶 p=30 时约 14.4 KiB，仍无需每线程 n² 局部数组。超过当前 48 KiB 算子动态 shared budget 的超大边数单元显式报错。

## 寄存器实测与取舍

每次运行 `run.log` 用 `cudaFuncGetAttributes` 和 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 记录最终编译内核的资源需求。以下为 k=1、128 线程的代表值，重新编译可能变化：

| 核 | registers/thread | local B/thread | shared B/block | 可驻留 blocks/SM 上限 |
|---|---:|---:|---:|---:|
| integrate_base | 62 | 0 | 0 | 8 |
| projection | 78 | 0 | 608 | 6 |
| integrate_equation | 72 | 0 | 0 | 7 |
| contract_moments | 48 | 0 | 0 | 10 |
| element_apply | 40 | 0 | 960 | 12 |
| sample_exact | 92 | 136 | 0 | 5 |
| error_kernel | 90 | 40 | 6360 | 5 |

核心基础/算子核无 local memory；制造解与误差核仍有少量 local memory，不能声称全工程零 spill。将制造解采样从误差积分中拆出，已减少误差核的寄存器压力，并避免每个 Picard 重复评估制造源项。没有用 `--maxrregcount` 强压寄存器数，也没有使用降低精度的 fast-math。

4096 单元实测 32/64/128 线程的 matvec 差距很小（约 0.127–0.130 ms），256 线程变慢至约 0.187 ms。保持默认 128；用户可以针对阶数/多边形边数调整为 32、64、128、256。当前不声称此分配对所有显卡/阶数最优。

## PETSc 稀疏路径的额外存储

每个单元 block 按线程跨步生成局部混合块 COO 值（矩阵包含两端 H(div) 方向系数和边界消元）。PETSc 用一次 `MatSetPreallocationCOO` 建立汇总映射，每次 `MatSetValuesCOO` 在 GPU 汇总成 CSR，不在 Picard 中重复建 Mat/KSP。MS 维护原顺序与 ND 顺序两份 CSR，Darcy natural 只需原顺序一份；PETSc 的 COO 汇总映射、PC 分解与工作区也占内存。

`report.json` 的 `coo_entries`、`coo_value_bytes`、`csr_nnz_original`、`csr_nnz_ordered` 单独报告这些规模；不能只看原来的 `basis_bytes/operator_bytes` 判断总显存。128×128 的 N=493056、restart=320 时，两组 FGMRES 基约 2.53 GB（十进制），还需稀疏图/值、预条件与几何等存储。大网格/高阶应按这些项共同选择 restart。

PETSc 对象唯一所有权和 Get/Restore 同步规则见 [代数层说明](PETSc_GPU重构.md)。早期短期重复释放/异常路径测试未观察到泄漏，但之后长寿命 KSP 多次更新出现主机 RSS 持续增长；当前 ILU/LU 已改为 CPU 因子以规避 cuSPARSE 更新路径，需计入主机因子和向量镜像的内存。早期记录和工具检查范围见 [本轮验收](validation/petsc_refactor_20260910/README.md)。
