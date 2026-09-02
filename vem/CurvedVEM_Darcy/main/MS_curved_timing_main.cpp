/**
 * CurvedVEM MS - 单网格组装性能剖析主程序
 * 功能：只跑一套网格、少量时间步，统计 assembleStiffnessMatrix 内各阶段耗时，
 *       重点确认 A(u) 系数矩阵（AG）是否为主要瓶颈。
 *       不计算收敛阶，不保存数据文件。
 *
 * 默认配置：网格文件 mesh/data/square_16x16.msh（= 32x32 个曲边单元）
 *           k = 1, dt = 1e-3, 3 个时间步, GPU 求解
 *
 * 用法：./build/MS_curved_timing [nx] [k] [n_steps] [cpu]
 *   nx       网格文件编号，读取 square_{nx}x{nx}.msh，默认 16
 *   k        多项式阶数，默认 1
 *   n_steps  时间步数，默认 3
 *   cpu      第 4 个参数写 cpu 则用 CPU 求解器，默认 GPU
 *
 * 示例：
 *   ./build/MS_curved_timing              # 16x16 网格文件, k=1, 3 步
 *   ./build/MS_curved_timing 8 1 3        # 8x8 网格文件
 *   ./build/MS_curved_timing 16 2 2       # k=2
 */

#include "../solver/MSSolver.h"

#include <petsc.h>

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Stage {
    std::string name;
    double secs;
};

// 打印累计计时表：按耗时降序，附占比与每次组装均值
void printAccumTable(const MSSolver::AssemblyTiming& t,
                     int num_cells,
                     double time_stepping_secs) {
    int n = t.n_calls > 0 ? t.n_calls : 1;

    std::vector<Stage> stages;
    stages.push_back(Stage{"AG 矩阵 A(u)",   t.ag});
    stages.push_back(Stage{"父类单元矩阵",   t.parent_mats});
    stages.push_back(Stage{"局部到全局",     t.to_global});
    stages.push_back(Stage{"K_ac",           t.k_ac});
    stages.push_back(Stage{"稳定化",         t.stab});
    stages.push_back(Stage{"子块插入",       t.insert_local});
    stages.push_back(Stage{"右端项",         t.rhs});
    stages.push_back(Stage{"G_cmin",         t.g_cmin});
    stages.push_back(Stage{"H_curved",       t.h_curved});
    stages.push_back(Stage{"方向调整",       t.direction});
    stages.push_back(Stage{"提取浓度系数",   t.extract_conc});
    stages.push_back(Stage{"全局最终装配",   t.global_final});

    // 各阶段之和（可能略小于 total，差值为未插桩的零碎开销）
    double sum = 0.0;
    for (size_t i = 0; i < stages.size(); ++i) sum += stages[i].secs;

    std::sort(stages.begin(), stages.end(),
              [](const Stage& a, const Stage& b) { return a.secs > b.secs; });

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  组装各阶段累计耗时（" << t.n_calls << " 次组装，"
              << num_cells << " 单元）\n";
    std::cout << "================================================================\n\n";
    std::cout << std::left << std::setw(18) << "阶段"
              << std::right << std::setw(12) << "累计(s)"
              << std::setw(10) << "占比"
              << std::setw(14) << "每次(s)"
              << std::setw(14) << "每单元(ms)" << "\n";
    std::cout << std::string(68, '-') << "\n";

    for (size_t i = 0; i < stages.size(); ++i) {
        double s = stages[i].secs;
        double per_call = s / n;
        double per_cell_ms = num_cells > 0
                           ? per_call / num_cells * 1000.0 : 0.0;
        std::cout << std::left << std::setw(18) << stages[i].name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << s
                  << std::setw(9) << std::setprecision(1)
                  << (t.total > 0.0 ? 100.0 * s / t.total : 0.0) << "%"
                  << std::setw(14) << std::setprecision(4) << per_call
                  << std::setw(14) << std::setprecision(3) << per_cell_ms
                  << "\n";
    }

    std::cout << std::string(68, '-') << "\n";
    std::cout << std::left << std::setw(18) << "各阶段之和"
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(12) << sum
              << std::setw(9) << std::setprecision(1)
              << (t.total > 0.0 ? 100.0 * sum / t.total : 0.0) << "%"
              << std::setw(14) << std::setprecision(4) << (sum / n) << "\n";
    std::cout << std::left << std::setw(18) << "组装函数总计"
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(12) << t.total
              << std::setw(9) << std::setprecision(1) << 100.0 << "%"
              << std::setw(14) << std::setprecision(4) << (t.total / n)
              << std::setw(14) << std::setprecision(3)
              << (num_cells > 0 ? t.total / n / num_cells * 1000.0 : 0.0)
              << "\n";
    std::cout << "\n";

    std::cout << "  时间推进总墙钟耗时  : " << std::fixed
              << std::setprecision(3) << time_stepping_secs << " s\n";
    std::cout << "  其中组装占          : " << std::setprecision(1)
              << (time_stepping_secs > 0.0
                      ? 100.0 * t.total / time_stepping_secs : 0.0)
              << " %\n";
    std::cout << "  其余（求解+边界等） : " << std::setprecision(3)
              << (time_stepping_secs - t.total) << " s\n";
    std::cout << "\n================================================================\n\n";
}

}  // namespace

int main(int argc, char** argv) {
    // 命令行参数在 PetscInitialize 之前解析，避免 PETSc 改写 argc/argv
    int nx = 16;
    int k = 1;
    int n_steps = 3;
    bool use_gpu = true;
    if (argc > 1) nx = std::atoi(argv[1]);
    if (argc > 2) k = std::atoi(argv[2]);
    if (argc > 3) n_steps = std::atoi(argv[3]);
    if (argc > 4 && std::string(argv[4]) == "cpu") use_gpu = false;

#if defined(PETSC_HAVE_CUDA)
    setenv("CUDA_VISIBLE_DEVICES", "0", 1);
    PetscOptionsSetValue(NULL, "-use_gpu_aware_mpi", "0");
    PetscOptionsSetValue(NULL, "-device_type", "cuda");
    PetscOptionsSetValue(NULL, "-device_select", "0");
#endif

    PetscErrorCode error = PetscInitialize(&argc, &argv, NULL, NULL);
    if (error != 0) return static_cast<int>(error);

    int status = 0;
    try {
        const int gauss = 9;
        const int picard_max = 50;
        const double picard_tol = 1e-6;
        const double dt = 1e-3;
        const double final_time = dt * n_steps + 0.5 * dt;

        std::string mesh_file = "mesh/data/square_" + std::to_string(nx)
                              + "x" + std::to_string(nx) + ".msh";
        int n = 2 * nx;  // square_nxn.msh 实际为 2n × 2n 个曲边单元

        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "  MS 单网格组装性能剖析\n";
        std::cout << "================================================================\n";
        std::cout << "  网格文件  : " << mesh_file << "\n";
        std::cout << "  曲边单元  : " << n << "x" << n
                  << " = " << n * n << " 个\n";
        std::cout << "  k         : " << k << "\n";
        std::cout << "  高斯点数  : " << gauss << "\n";
        std::cout << "  dt        : " << std::scientific << std::setprecision(1)
                  << dt << "   步数: " << n_steps << "\n";
        std::cout << "  时间格式  : BDF2（第 1 步 BDF1 启动）\n";
        std::cout << "  求解器    : " << (use_gpu ? "GPU" : "CPU") << "\n";
        std::cout << "================================================================\n";

        vem::StraightMeshReader reader;
        if (!reader.read_mesh(mesh_file))
            throw std::runtime_error("无法读取网格: " + mesh_file);

        MaxwellStefan::MSProblem problem;
        problem.set_problem_index(2);
        if (!problem.init_problem())
            throw std::runtime_error("无法初始化 MS 算例");

        std::clock_t t_init0 = std::clock();
        MSSolver solver(reader, problem, dt, final_time,
                        picard_max, picard_tol,
                        gauss, gauss, k,
                        false,   // 不保存数据
                        "",
                        true);   // use_bdf2
        if (!solver.initialize())
            throw std::runtime_error("求解器初始化失败");
        double init_secs = double(std::clock() - t_init0) / CLOCKS_PER_SEC;

        std::cout << "\n  initialize() 耗时: " << std::fixed
                  << std::setprecision(3) << init_secs << " s\n";
        std::cout << "  全局自由度       : " << solver.getTotalDofAll()
                  << "  (通量 " << solver.getTotalDofFluxAll()
                  << " + 浓度 " << solver.getTotalDofConcAll() << ")\n";

        solver.setTimingEnabled(true);

        std::clock_t t_ts0 = std::clock();
        PetscErrorCode ierr = solver.solveTimeStepping("ms_timing", use_gpu);
        if (ierr != 0)
            throw std::runtime_error("时间推进失败");
        double ts_secs = double(std::clock() - t_ts0) / CLOCKS_PER_SEC;

        printAccumTable(solver.getAssemblyTimingAccum(), n * n, ts_secs);

    } catch (const std::exception& e) {
        std::cerr << "\n  错误：" << e.what() << "\n\n";
        status = 1;
    }

    PetscFinalize();
    return status;
}
