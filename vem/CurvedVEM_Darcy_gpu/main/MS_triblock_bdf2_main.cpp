/**
 * MS 算例 4：TriBlockSwirl Q8 制造解 BDF2 空间收敛实验。
 * 每完成一层网格，立即输出所有组分的误差、累计空间收敛阶和分阶段墙钟耗时。
 * h=sqrt(计算域面积/实际单元数)，结构网格 square_nxn 的 h=1/(2n)。
 * 编译：make MS_triblock_bdf2；运行：./build/MS_triblock_bdf2。
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
        std::cerr << "请在 main/MS_triblock_bdf2_main.cpp 中修改参数。\n";
        return 1;
    }
    try {
        // ========== 算例、空间阶与积分规则 ==========
        const int problem_index = 4;
        const int k = 1;               // 可改 0..4；高阶可增大下方 FGMRES restart。
        const int gauss_point_num = 0; // 0 按 k 自动选择 9/12/36。
        const int gauss_point_num_1d = 9;
        const std::vector<int> meshes = {64};
        // square_nxn.msh 实际有 (2n)² 个单元；真实单元数/尺度由读取后的网格测量。
        // ========== 同一终止时间与随空间阶选择的时间尺度 ==========
        // BDF2: O(dt²) 与 O(h^(k+1)) 匹配，dt_target=C*h^((k+1)/2)。
        // 改 use_bdf2=false 时自动改为 Euler 的 dt_target=C*h^(k+1)。
        // 不同网格都推进到同一个 T；不向 T 增加半步。
        const double final_time = 0.1;
        const double time_coefficient = 0.1;
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
        options.picard_tol = 1e-6;
        // PETSc ND + CPU ILU(0)（GPU MatMult）；保留当前主函数的迭代上限与 Krylov 空间。
        options.maxit = 200000;
        options.restart = 320;
        options.linear_tol = 1e-11;
        options.threads = 128;
        options.tile = 4096;
        options.save_solution = false; // 收敛实验只保存误差/日志；改 true 才下载最终全局解。
        options.error_interval = 10;   // 每 10 步及最终步记录误差/质量，便于运行中监控。
        options.print_interval = 10;   // 首步、每 10 步及最终步打印进度和耗时。

        // ========== 输出目录与累计表名称 ==========
        const std::string output_root = "output/MS_triblock_bdf2_k" + std::to_string(k);
        options.files.log = "iterations.log";
        options.files.history = "history.csv";
        options.files.report = "report.json";
        options.files.parameters = "parameters.txt";
        options.files.solution = "solution.csv";
        vgpu::StudyOptions study;
        study.title = "MS 算例 4：TriBlockSwirl Q8 制造解 BDF2 空间收敛实验";
        study.scale = vgpu::MeshScale::Equivalent;             // 非均匀/局部加密网格请考虑 MaxDiameter。
        study.error_domain = vgpu::ErrorDomain::Computational; // 与原 CPU MS 的 L2 接口一致。
        study.table_file = output_root + "/convergence.csv";
        study.terminal_file = output_root + "/results.txt";
        options.bdf2 = use_bdf2;
        study.time.final_time = final_time;
        study.time.policy = vgpu::TimePolicy::Balanced;
        study.time.coefficient = time_coefficient;
        study.time.smooth_steps = true; // 取不小于 ceil(T/dt_target) 的最小 2^a*5^b。
        study.time.max_steps = 1000000; // 超过上限报错，不暗中放大时间步。

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
        std::cerr << "MS_triblock_bdf2 运行失败：" << e.what() << '\n';
        return 1;
    }
}
