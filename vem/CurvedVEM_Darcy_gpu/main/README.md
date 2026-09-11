# 主函数使用约定

每个 `*_main.cpp` 单独编译成 `build/名字`，只运行对应的一个方程算例。

1. 选择 `Darcy_curved`、`Darcy_straight`、`MS_curved_bdf2`、`MS_triblock_bdf2`、`MS_discontinuous` 或 `MS_half_annulus`。
2. 在主函数集中修改 k、网格序列、终止时间、时间系数/目标步长、迭代容差及输出名称。
3. `make -j4 名字` 后运行 `./build/名字`。只算一个网格时将 meshes 改为单元素列表。
4. 每层完成立即显示并保存误差/收敛阶与计时表。MS 1/3 无解析解，只显示质量漂移。

BDF2 默认 dt_target=0.1*h^((k+1)/2)，Euler 为 0.1*h^(k+1)，步数向上取整后 dt=T/N。不要在终止时间加半步。结构网格 `square_nxn` 实际为 (2n)² 单元，h 由真实网格测量。只完成一层不能给出空间收敛阶。

`vem_gpu*_main.cpp` 是保留的单网格、批量、性能、命令行兼容入口；默认工作流不依赖批量执行六个算例。

完整参数、输出列及计时解释见 [工程说明](../README.md)。

PETSc 入口首先创建 `PetscSession`，算例中配置 `assembled=true`、`petsc_pc="ilu"`、`petsc_ordering="auto"`（Darcy→natural，MS→nd；各命名主函数显式填写对应排序）、`ilu_levels=0`。矩阵/向量/求解器释放后再 Finalize。详情见 [PETSc GPU 代数层](../docs/PETSc_GPU重构.md)。
