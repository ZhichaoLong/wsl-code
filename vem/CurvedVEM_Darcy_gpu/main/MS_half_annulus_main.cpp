/**
 * MS 算例 3：半圆环扩散与组分质量守恒。
 * 此算例无解析解：终端输出每组分质量、初始质量及漂移，不伪造误差和收敛阶。
 * 默认只计算一套网格的长时间过程，定期输出进度/质量，最终输出耗时。
 * 编译：make MS_half_annulus；运行：./build/MS_half_annulus。
 * 所有配置都在本 main 内修改，本程序不运行其他方程算例。
 */
#include "mesh_study.h"
#include <exception>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char**) {
    vgpu::PetscSession petsc; // 必须晚于全部 Mat/Vec/KSP 释放再 Finalize。
    if (argc != 1) {
        std::cerr << "请在 main/MS_half_annulus_main.cpp 中修改参数。\n";
        return 1;
    }
    try {
        // ========== 算例、空间阶与积分规则 ==========
        const int problem_index = 3;
        const int k = 1;               // 可改 0..4；高阶可增大下方 FGMRES restart。
        const int gauss_point_num = 0; // 0 按 k 自动选择 9/12/36。
        const int gauss_point_num_1d = 9;
        const std::vector<int> meshes = {8};
        // square_nxn.msh 实际有 (2n)² 个单元；真实单元数/尺度由读取后的网格测量。
        // ========== 长时间扩散的时间配置 ==========
        // 无解析解，不从网格误差计算收敛阶；用每组分质量漂移检查守恒。
        // 可在 meshes 中只保留当前要运行的一套网格。
        const double final_time = 0.5;
        const double delta_t = 0.001;
        const bool use_bdf2 = true;

        // ========== 线性/非线性求解与 GPU 参数 ==========
        vgpu::SolverOptions options;
        // PETSc GPU 稀疏矩阵、向量和 FGMRES/ILU(0)；可选 assembled=false 使用 MatShell。
        options.assembled = true;
        options.petsc_pc = "ilu";
        options.petsc_ordering = "nd";
        options.ilu_levels = 0;
        options.equation = "ms";
        options.example = problem_index;
        options.k = k;
        options.nq = gauss_point_num;
        options.ng = gauss_point_num_1d;
        options.picard = 50;
        options.picard_tol = 1e-9;
        options.maxit = 4000;
        options.restart = 80;
        options.linear_tol = 1e-11;
        options.threads = 128;
        options.tile = 4096;
        options.save_solution = true; // 保存最终解，不在每一步下载大向量。
        options.error_interval = 10;  // 每 10 步诊断一次质量，最终步必做。
        options.print_interval = 10;  // 首步、每 10 步及最终步打印进度和耗时。

        // ========== 输出目录与累计表名称 ==========
        const std::string output_root = "output/MS_half_annulus_k" + std::to_string(k);
        options.files.log = "iterations.log";
        options.files.history = "history.csv";
        options.files.report = "report.json";
        options.files.parameters = "parameters.txt";
        options.files.solution = "solution.csv";
        vgpu::StudyOptions study;
        study.title = "MS 算例 3：半圆环扩散与组分质量守恒";
        study.scale = vgpu::MeshScale::Equivalent;             // 非均匀/局部加密网格请考虑 MaxDiameter。
        study.error_domain = vgpu::ErrorDomain::Computational; // 与原 CPU MS 的 L2 接口一致。
        study.table_file = output_root + "/mass_history.csv";
        study.terminal_file = output_root + "/results.txt";
        options.bdf2 = use_bdf2;
        study.time.final_time = final_time;
        study.time.policy = vgpu::TimePolicy::FixedTarget;
        study.time.fixed_dt = delta_t;
        study.time.smooth_steps = true;
        study.time.max_steps = 1000000;

        // ========== 当前算例的网格列表 ==========
        // 单独运行一层时将 meshes 改为 {需要的 n}；首层的收敛阶显示 '-'。
        for (int n : meshes) {
            const std::string label = std::to_string(2 * n) + "x" + std::to_string(2 * n);
            const std::string mesh_file =
                "mesh/data/square_" + std::to_string(n) + "x" + std::to_string(n) + ".msh";
            study.meshes.push_back({label, mesh_file, output_root + "/mesh_" + label});
        }
        // 逐网格测量 h、确定 dt/N、求解、打印累计表；本次失败时保留前面已完成的结果。
        return vgpu::run_mesh_study(options, study);
    } catch (const std::exception& e) {
        std::cerr << "MS_half_annulus 运行失败：" << e.what() << '\n';
        return 1;
    }
}
