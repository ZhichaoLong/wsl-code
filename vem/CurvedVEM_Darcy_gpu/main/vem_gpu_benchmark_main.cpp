/**
 * GPU 性能测试主程序：在 main 中设置网格、阶次、线程及重复次数。
 * 编译运行：make benchmark；默认测量 4096 单元 MS 2 算子的准备和应用耗时。
 * benchmark_only=true 不求解 PDE，报告状态为 benchmark。
 */
#include "solver_driver.h"
#include <exception>
#include <iostream>

int main(int argc, char**) {
    vgpu::PetscSession petsc; // 必须晚于全部 Mat/Vec/KSP 释放再 Finalize。
    if (argc != 1) {
        std::cerr << "请在 main/vem_gpu_benchmark_main.cpp 中修改性能测试参数。\n";
        return 1;
    }
    try {
        vgpu::SolverOptions options;
        // PETSc GPU 稀疏矩阵、向量和 FGMRES/ILU(0)；可选 assembled=false 使用 MatShell。
        options.assembled = true;
        options.petsc_pc = "ilu";
        options.petsc_ordering = "auto"; // 随下方 equation 选择 Darcy natural / MS nd。
        options.ilu_levels = 0;
        options.equation = "ms";
        options.example = 2;
        options.mesh = "mesh/data/square_32x32.msh";
        options.k = 1;
        options.nq = 0;
        options.ng = 9;
        options.dt = 0.001; // 质量块系数为 1/dt。
        options.steps = 3;
        options.bdf2 = true;
        options.picard = 50;
        options.picard_tol = 1e-9;
        options.maxit = 4000;
        options.restart = 80;
        options.linear_tol = 1e-11;
        options.threads = 128;
        options.tile = 4096;
        options.benchmark = 20;        // 预热后重复计时的次数。
        options.benchmark_only = true; // 改为 false 可在计时后继续完整求解。

        options.output = "output/benchmark_main";
        options.files.log = "run.log";
        options.files.history = "history.csv";
        options.files.solution = "solution.csv"; // 仅完整求解时生成。
        options.files.report = "report.json";
        options.files.parameters = "parameters.txt";
        return vgpu::run_case(options);
    } catch (const std::exception& e) {
        std::cerr << "性能测试失败：" << e.what() << '\n';
        return 1;
    }
}
