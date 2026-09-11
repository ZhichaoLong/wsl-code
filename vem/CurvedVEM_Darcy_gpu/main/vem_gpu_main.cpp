/**
 * 单算例主程序：在下面的“配置参数”区域修改算例、网格、时间和输出文件。
 * 编译运行：make -j4 && ./build/vem_gpu（从工程根目录运行）。
 * Darcy 算例：1 正弦曲边，2 恒等映射。
 * MS 算例：1 间断初值，2 正弦曲边制造解，3 半圆环，4 TriBlockSwirl Q8 制造解。
 */
#include "solver_driver.h"
#include <exception>
#include <iostream>

int main(int argc, char**) {
    vgpu::PetscSession petsc; // 必须晚于全部 Mat/Vec/KSP 释放再 Finalize。
    // 本入口只读取下方源码配置；需要旧命令行方式时运行 vem_gpu_cli。
    if (argc != 1) {
        std::cerr << "请在 main/vem_gpu_main.cpp 中修改参数；命令行兼容入口为 build/vem_gpu_cli。\n";
        return 1;
    }
    try {
        // ==================== 1. 算例与网格 ====================
        vgpu::SolverOptions options;
        // PETSc GPU 稀疏矩阵、向量和 FGMRES/ILU(0)；可选 assembled=false 使用 MatShell。
        options.assembled = true;
        options.petsc_pc = "ilu";
        options.petsc_ordering = "auto"; // 随下方 equation 选择 Darcy natural / MS nd。
        options.ilu_levels = 0;
        options.equation = "ms";                   // 方程："darcy" 或 "ms"。
        options.example = 2;                       // 当前运行 MS 算例 2。
        options.mesh = "mesh/data/square_2x2.msh"; // 此文件实际为 16 个单元。
        options.k = 1;                             // 多项式阶数，0..4。
        options.nq = 0;                            // 0 自动选择；也可指定 9/12/36。
        options.ng = 9;                            // 每边积分点数，至少 k+2。

        // ==================== 2. 时间与求解器 ====================
        options.dt = 0.001;         // 时间步长（Darcy 忽略时间推进设置）。
        options.steps = 3;          // 最终时间 T = dt * steps = 0.003。
        options.bdf2 = true;        // true: BDF2，首步 Euler；false: Euler。
        options.picard = 50;        // 每步最大 Picard 次数。
        options.picard_tol = 1e-9;  // 非线性收敛容差。
        options.maxit = 4000;       // 最大 FGMRES 迭代次数。
        options.restart = 80;       // FGMRES 重启长度。
        options.linear_tol = 1e-11; // 线性求解容差。

        // ==================== 3. GPU 与性能计时 ====================
        options.threads = 128;          // 每块线程数：32/64/128/256。
        options.tile = 4096;            // 每次分片 launch 的最大单元数。
        options.benchmark = 0;          // 0 不计时；设为正数可先计时再求解。
        options.benchmark_only = false; // true 只测性能，不推进时间。

        // ==================== 4. 输出目录与文件名 ====================
        options.output = "output/ms2_main";
        options.files.log = "run.log";
        options.files.history = "history.csv";
        options.files.solution = "solution.csv";
        options.files.report = "report.json";
        options.files.parameters = "parameters.txt";

        // 公共驱动先检查配置，随后构造 GPU 算子、求解并保存到上述目录。
        return vgpu::run_case(options);
    } catch (const std::exception& e) {
        std::cerr << "算例运行失败：" << e.what() << '\n';
        return 1;
    }
}
