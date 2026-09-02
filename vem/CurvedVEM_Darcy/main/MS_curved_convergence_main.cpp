/**
 * CurvedVEM MS - 收敛阶主程序入口（浓度 + 通量）
 * 功能：Maxwell-Stefan 制造解算例，多套网格下计算浓度与通量 L2 误差及收敛阶
 * 配置：k=2, dt=0.0001, 共 50 步, 网格 1x1~16x16, GPU 求解
 *       不保存任何数据文件，仅输出误差表与收敛阶
 */

#include "../solver/MSSolver.h"

#include <petsc.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Result {
    std::string label;
    int num_cells;
    double h;
    std::vector<double> conc_err;
    std::vector<double> conc_order;
    std::vector<double> flux_err;
    std::vector<double> flux_order;
};

Result runCase(int nx, double dt, double final_time, int k,
               int gauss, int picard_max, double picard_tol,
               int n_comp, bool use_gpu) {
    std::string mesh_file = "mesh/data/square_" + std::to_string(nx)
                          + "x" + std::to_string(nx) + ".msh";
    int n = 2 * nx;  // square_nxn.msh 实际为 2n × 2n 个曲边单元
    std::string label = std::to_string(n) + "x" + std::to_string(n);

    vem::StraightMeshReader reader;
    if (!reader.read_mesh(mesh_file))
        throw std::runtime_error("无法读取网格: " + mesh_file);

    MaxwellStefan::MSProblem problem;
    problem.set_problem_index(2);
    if (!problem.init_problem())
        throw std::runtime_error("无法初始化 MS 算例");

    // 不保存任何数据
    MSSolver solver(reader, problem, dt, final_time,
                    picard_max, picard_tol,
                    gauss, gauss, k,
                    false,    // 不保存
                    "");
    if (!solver.initialize())
        throw std::runtime_error("求解器初始化失败: " + label);

    PetscErrorCode ierr = solver.solveTimeStepping("ms_convergence", use_gpu);
    if (ierr != 0)
        throw std::runtime_error("时间推进失败: " + label);

    Result r;
    r.label = label;
    r.num_cells = n * n;
    r.h = 1.0 / static_cast<double>(n);
    r.conc_err = solver.computeConcentrationL2Error(gauss);
    r.flux_err = solver.computeFluxL2Error(gauss);
    r.conc_order.assign(n_comp, 0.0);
    r.flux_order.assign(n_comp, 0.0);
    return r;
}

void computeOrders(std::vector<Result>& results) {
    for (size_t i = 1; i < results.size(); ++i) {
        double log_h = std::log(results[i-1].h / results[i].h);
        for (int c = 0; c < (int)results[i].conc_err.size(); ++c) {
            double log_e = std::log(
                results[i-1].conc_err[c] / results[i].conc_err[c]);
            results[i].conc_order[c] = log_e / log_h;
        }
        for (int c = 0; c < (int)results[i].flux_err.size(); ++c) {
            double log_e = std::log(
                results[i-1].flux_err[c] / results[i].flux_err[c]);
            results[i].flux_order[c] = log_e / log_h;
        }
    }
}

void printTable(const std::vector<Result>& results, int n_comp) {
    // ===== 浓度误差表 =====
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "  Maxwell-Stefan 浓度 L2 误差收敛阶（绝对误差，制造解，k=1，曲边）\n";
    std::cout << "============================================================\n\n";
    std::cout << std::setw(10) << "网格"
              << std::setw(10) << "单元数"
              << std::setw(10) << "h";
    for (int c = 0; c < n_comp; ++c)
        std::cout << std::setw(16) << ("c" + std::to_string(c) + " 误差");
    for (int c = 0; c < n_comp; ++c)
        std::cout << std::setw(10) << ("c" + std::to_string(c) + " 阶");
    std::cout << "\n";
    std::cout << std::string(30 + n_comp * 26, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::setw(10) << r.label
                  << std::setw(10) << r.num_cells
                  << std::setw(10) << std::fixed
                  << std::setprecision(4) << r.h;
        for (int c = 0; c < n_comp; ++c) {
            std::cout << std::setw(16) << std::scientific
                      << std::setprecision(4) << r.conc_err[c];
        }
        for (int c = 0; c < n_comp; ++c) {
            if (r.conc_order[c] == 0.0)
                std::cout << std::setw(10) << "  -  ";
            else
                std::cout << std::setw(10) << std::fixed
                          << std::setprecision(2) << r.conc_order[c];
        }
        std::cout << "\n";
    }
    std::cout << "\n============================================================\n";

    // ===== 通量误差表 =====
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "  Maxwell-Stefan 通量 L2 误差收敛阶（绝对误差，制造解，k=1，曲边）\n";
    std::cout << "============================================================\n\n";
    std::cout << std::setw(10) << "网格"
              << std::setw(10) << "单元数"
              << std::setw(10) << "h";
    for (int c = 0; c < n_comp; ++c)
        std::cout << std::setw(16) << ("J" + std::to_string(c) + " 误差");
    for (int c = 0; c < n_comp; ++c)
        std::cout << std::setw(10) << ("J" + std::to_string(c) + " 阶");
    std::cout << "\n";
    std::cout << std::string(30 + n_comp * 26, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::setw(10) << r.label
                  << std::setw(10) << r.num_cells
                  << std::setw(10) << std::fixed
                  << std::setprecision(4) << r.h;
        for (int c = 0; c < n_comp; ++c) {
            std::cout << std::setw(16) << std::scientific
                      << std::setprecision(4) << r.flux_err[c];
        }
        for (int c = 0; c < n_comp; ++c) {
            if (r.flux_order[c] == 0.0)
                std::cout << std::setw(10) << "  -  ";
            else
                std::cout << std::setw(10) << std::fixed
                          << std::setprecision(2) << r.flux_order[c];
        }
        std::cout << "\n";
    }
    std::cout << "\n============================================================\n\n";
}

}  // namespace

int main(int argc, char** argv) {
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
        const int k = 1;
        const int gauss = 9;
        const int n_comp = 3;
        const int picard_max = 50;
        const double picard_tol = 1e-6;
        const bool use_gpu = true;

        // 时间步长和步数
        const int n_steps = 100;
        const double dt = 0.00001;
        const double final_time = dt * n_steps;

        // 测试网格序列
        std::vector<int> meshes = {1, 2, 4, 8, 16, 32};

        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════╗\n";
        std::cout << "║  MS 收敛阶测试（制造解，曲边 H(div) 混合虚元）        ║\n";
        std::cout << "║  k = " << k
                  << "  dt = " << dt
                  << "  steps = " << n_steps
                  << "  T = " << final_time
                  << "           ║\n";
        std::cout << "║  GPU = " << (use_gpu ? "on" : "off")
                  << "  误差类型：绝对 L2"
                  << "                           ║\n";
        std::cout << "║  网格：";
        for (size_t i = 0; i < meshes.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << meshes[i] << "x" << meshes[i];
        }
        std::cout << "                    ║\n";
        std::cout << "╚════════════════════════════════════════════════════════╝\n\n";

        std::vector<Result> results;
        for (size_t i = 0; i < meshes.size(); ++i) {
            int nx = meshes[i];
            std::cout << "[" << (i+1) << "/" << meshes.size()
                      << "]  网格 " << nx << "x" << nx << " ... ";
            std::cout.flush();

            Result r = runCase(nx, dt, final_time, k, gauss,
                               picard_max, picard_tol, n_comp, use_gpu);
            results.push_back(r);

            std::cout << "done (" << r.num_cells << " cells, "
                      << "c0=" << std::scientific << std::setprecision(2)
                      << r.conc_err[0]
                      << ", J0=" << std::scientific << std::setprecision(2)
                      << r.flux_err[0] << ")\n";
        }

        computeOrders(results);
        printTable(results, n_comp);

    } catch (const std::exception& e) {
        std::cerr << "\n  错误：" << e.what() << "\n\n";
        status = 1;
    }

    PetscFinalize();
    return status;
}
