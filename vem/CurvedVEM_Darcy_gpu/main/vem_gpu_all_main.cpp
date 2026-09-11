/**
 * 可选的旧批量入口（非默认运行方式）：共同参数集中设置，各算例的编号、输出目录及独立设置在 main 中列出。
 * 编译运行：make -j4 && ./build/vem_gpu_all。
 * 在 cases 列表中增删算例；可对任意一例单独设置 k、mesh、dt、steps 和输出文件名。
 */
#include "solver_driver.h"
#include <exception>
#include <iostream>
#include <vector>

int main(int argc, char**) {
    vgpu::PetscSession petsc; // 必须晚于全部 Mat/Vec/KSP 释放再 Finalize。
    if (argc != 1) {
        std::cerr << "请在 main/vem_gpu_all_main.cpp 中修改共同参数和各算例配置。\n";
        return 1;
    }
    try {
        // ==================== 1. 共同参数 ====================
        vgpu::SolverOptions common;
        // PETSc 默认按方程选择消元顺序；每个算例仍可单独改这些参数。
        common.assembled = true;
        common.petsc_pc = "ilu";
        common.petsc_ordering = "auto"; // Darcy: natural；MS: nd。
        common.ilu_levels = 0;
        common.mesh = "mesh/data/square_2x2.msh"; // 16 单元，沿用 CPU 网格读取器。
        common.k = 1;
        common.nq = 0; // 随 k 自动选积分点数。
        common.ng = 9;
        common.dt = 0.001;
        common.steps = 3; // 四个 MS 算例均推进到 0.003。
        common.bdf2 = true;
        common.picard = 50;
        common.picard_tol = 1e-9;
        common.maxit = 4000;
        common.restart = 80;
        common.linear_tol = 1e-11;
        common.threads = 128;
        common.tile = 4096;
        common.benchmark = 0;
        common.benchmark_only = false; // 批量入口执行完整求解。

        // 每个算例的子目录内使用以下文件名；也可在各例下单独覆盖。
        common.files.log = "run.log";
        common.files.history = "history.csv";
        common.files.solution = "solution.csv";
        common.files.report = "report.json";
        common.files.parameters = "parameters.txt";
        const std::string output_root = "output/all_examples_main";
        const std::string summary_file = output_root + "/summary.csv";

        // ==================== 2. Darcy 两个算例 ====================
        vgpu::SolverOptions darcy1 = common;
        darcy1.equation = "darcy";
        darcy1.example = 1; // 正弦曲边映射，解析压力与通量。
        darcy1.output = output_root + "/darcy1";

        vgpu::SolverOptions darcy2 = common;
        darcy2.equation = "darcy";
        darcy2.example = 2; // 恒等映射，与算例 1 使用同一物理解析解。
        darcy2.output = output_root + "/darcy2";

        // ==================== 3. MS 四个算例 ====================
        vgpu::SolverOptions ms1 = common;
        ms1.equation = "ms";
        ms1.example = 1; // 分块间断初值、无源零通量；观察质量守恒。
        ms1.output = output_root + "/ms1";

        vgpu::SolverOptions ms2 = common;
        ms2.equation = "ms";
        ms2.example = 2; // 正弦曲边制造解；计算浓度、通量和散度误差。
        ms2.output = output_root + "/ms2";

        vgpu::SolverOptions ms3 = common;
        ms3.equation = "ms";
        ms3.example = 3; // 半圆环与径向分层初值；观察质量守恒。
        ms3.output = output_root + "/ms3";
        // 如需单独加密时间，可写 ms3.dt = 0.0005; ms3.steps = 6;

        vgpu::SolverOptions ms4 = common;
        ms4.equation = "ms";
        ms4.example = 4; // TriBlockSwirl 逐单元 Q8 曲边制造解。
        ms4.output = output_root + "/ms4";

        // ==================== 4. 按列表顺序求解 ====================
        const std::vector<vgpu::SolverOptions> cases = {darcy1, darcy2, ms1, ms2, ms3, ms4};
        return vgpu::run_examples(cases, summary_file);
    } catch (const std::exception& e) {
        std::cerr << "批量算例运行失败：" << e.what() << '\n';
        return 1;
    }
}
