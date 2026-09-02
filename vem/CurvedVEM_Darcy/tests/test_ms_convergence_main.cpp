/**
 * MS 方程收敛阶测试（制造解算例）
 *
 * 在 1x1、2x2、4x4、8x8 网格上分别求解 Maxwell-Stefan 方程，
 * 计算每个组分的浓度与通量 L2 误差，并估计收敛阶。
 *
 * 时间步长统一取足够小（dt=0.001，终止时间 0.1），保证空间误差主导。
 */

#include "../solver/MSSolver.h"

#include <petsc.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ConvergenceResult {
    std::string mesh_label;    // 例如 "4x4"
    int num_cells;             // 实际单元数（正方形网格 = 2*nx*ny）
    double h;                  // 特征网格尺寸
    std::vector<double> conc_err;  // 各组分浓度 L2 误差
    std::vector<double> flux_err;  // 各组分通量 L2 误差
    std::vector<double> conc_order; // 各组分浓度收敛阶
    std::vector<double> flux_order; // 各组分通量收敛阶
};

std::string getMeshPath(int nx) {
    return "mesh/data/square_" + std::to_string(nx) + "x" + std::to_string(nx) + ".msh";
}

ConvergenceResult runCase(int nx, int k, double dt, double final_time,
                          int picard_max_iter, double picard_tol,
                          int gauss_point_num, int gauss_point_num_1d,
                          int n_comp) {
    const std::string mesh_file = getMeshPath(nx);
    const std::string label = std::to_string(nx) + "x" + std::to_string(nx);

    vem::StraightMeshReader reader;
    if (!reader.read_mesh(mesh_file)) {
        throw std::runtime_error("无法读取网格: " + mesh_file);
    }

    MaxwellStefan::MSProblem problem;
    problem.set_problem_index(2);  // 制造解算例
    if (!problem.init_problem()) {
        throw std::runtime_error("无法初始化 MS 算例");
    }

    // 数据子目录名按网格区分，避免冲突
    std::string subfolder = "conv_test_" + label;

    MSSolver solver(reader, problem,
                    dt, final_time,
                    picard_max_iter, picard_tol,
                    gauss_point_num, gauss_point_num_1d,
                    k,
                    false,  // 不保存每步解
                    subfolder);

    if (!solver.initialize()) {
        throw std::runtime_error("求解器初始化失败: " + label);
    }

    PetscErrorCode ierr = solver.solveTimeStepping("ms_convergence");
    if (ierr != 0) {
        throw std::runtime_error("时间推进求解失败: " + label);
    }

    ConvergenceResult result;
    result.mesh_label = label;
    result.num_cells = solver.getMeshData()->num_cells;
    result.h = 1.0 / static_cast<double>(nx);  // 单元尺寸 ~ 1/nx
    result.conc_err = solver.computeConcentrationL2Error(gauss_point_num);
    result.flux_err = solver.computeFluxL2Error(gauss_point_num);
    result.conc_order.assign(n_comp, 0.0);
    result.flux_order.assign(n_comp, 0.0);

    return result;
}

void computeOrders(std::vector<ConvergenceResult>& results) {
    for (size_t i = 1; i < results.size(); ++i) {
        double log_h_ratio = std::log(results[i-1].h / results[i].h);
        for (size_t c = 0; c < results[i].conc_err.size(); ++c) {
            double log_c = std::log(
                results[i-1].conc_err[c] / results[i].conc_err[c]);
            results[i].conc_order[c] = log_c / log_h_ratio;

            double log_f = std::log(
                results[i-1].flux_err[c] / results[i].flux_err[c]);
            results[i].flux_order[c] = log_f / log_h_ratio;
        }
    }
}

void printTable(const std::vector<ConvergenceResult>& results,
                int n_comp) {
    std::cout << "\n";
    std::cout << "========================================================================\n";
    std::cout << "  Maxwell-Stefan 方程收敛阶（制造解，k=1，曲边映射）\n";
    std::cout << "========================================================================\n";
    std::cout << "\n  浓度 L2 误差：\n";
    std::cout << std::setw(10) << "网格"
              << std::setw(10) << "单元数"
              << std::setw(10) << "h";
    for (int c = 0; c < n_comp; ++c) {
        std::cout << std::setw(16) << ("c" + std::to_string(c) + " 误差");
    }
    for (int c = 0; c < n_comp; ++c) {
        std::cout << std::setw(10) << ("c" + std::to_string(c) + " 阶");
    }
    std::cout << "\n";
    std::cout << std::string(
        30 + static_cast<size_t>(n_comp) * 26, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::setw(10) << r.mesh_label
                  << std::setw(10) << r.num_cells
                  << std::setw(10) << std::fixed
                  << std::setprecision(4) << r.h;
        for (int c = 0; c < n_comp; ++c) {
            std::cout << std::setw(16) << std::scientific
                      << std::setprecision(4) << r.conc_err[c];
        }
        for (int c = 0; c < n_comp; ++c) {
            if (r.conc_order[c] == 0.0) {
                std::cout << std::setw(10) << "  -  ";
            } else {
                std::cout << std::setw(10) << std::fixed
                          << std::setprecision(2) << r.conc_order[c];
            }
        }
        std::cout << "\n";
    }

    std::cout << "\n  通量 L2 误差：\n";
    std::cout << std::setw(10) << "网格"
              << std::setw(10) << "单元数"
              << std::setw(10) << "h";
    for (int c = 0; c < n_comp; ++c) {
        std::cout << std::setw(16) << ("J" + std::to_string(c) + " 误差");
    }
    for (int c = 0; c < n_comp; ++c) {
        std::cout << std::setw(10) << ("J" + std::to_string(c) + " 阶");
    }
    std::cout << "\n";
    std::cout << std::string(
        30 + static_cast<size_t>(n_comp) * 26, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::setw(10) << r.mesh_label
                  << std::setw(10) << r.num_cells
                  << std::setw(10) << std::fixed
                  << std::setprecision(4) << r.h;
        for (int c = 0; c < n_comp; ++c) {
            std::cout << std::setw(16) << std::scientific
                      << std::setprecision(4) << r.flux_err[c];
        }
        for (int c = 0; c < n_comp; ++c) {
            if (r.flux_order[c] == 0.0) {
                std::cout << std::setw(10) << "  -  ";
            } else {
                std::cout << std::setw(10) << std::fixed
                          << std::setprecision(2) << r.flux_order[c];
            }
        }
        std::cout << "\n";
    }

    std::cout << "\n========================================================================\n\n";
}

}  // namespace

int main(int argc, char** argv) {
    PetscErrorCode error = PetscInitialize(&argc, &argv, NULL, NULL);
    if (error != 0) {
        return static_cast<int>(error);
    }

    int status = 0;
    try {
        // ========== 配置参数 ==========
        const int k = 1;
        const int gauss_point_num = 9;
        const int gauss_point_num_1d = 9;

        const double final_time = 0.5;
        const double dt = 0.01;        // 统一时间步长，所有网格同一时刻比较
        const int picard_max_iter = 50;
        const double picard_tol = 1e-10;
        const int n_comp = 3;

        // 测试网格序列
        std::vector<int> mesh_sizes = {1, 2, 4, 8};

        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  MS 收敛阶测试 — 制造解算例（曲边 H(div) 混合虚元）            ║\n";
        std::cout << "║    k = " << k
                  << "  dt = " << dt
                  << "  T = " << final_time
                  << "  Picard tol = " << picard_tol << "      ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

        // ========== 依次计算每一套网格 ==========
        std::vector<ConvergenceResult> results;
        for (size_t i = 0; i < mesh_sizes.size(); ++i) {
            int nx = mesh_sizes[i];
            std::cout << "[" << (i+1) << "/" << mesh_sizes.size()
                      << "]  网格 " << nx << "x" << nx << " ...\n";
            std::cout.flush();

            ConvergenceResult r = runCase(nx, k, dt, final_time,
                                          picard_max_iter, picard_tol,
                                          gauss_point_num, gauss_point_num_1d,
                                          n_comp);
            results.push_back(r);

            std::cout << "     完成。单元数 " << r.num_cells
                      << "  浓度 c0=" << std::scientific
                      << std::setprecision(2) << r.conc_err[0]
                      << "  通量 J0=" << r.flux_err[0] << "\n\n";
        }

        // ========== 计算收敛阶并输出表格 ==========
        computeOrders(results);
        printTable(results, n_comp);

    } catch (const std::exception& exception) {
        std::cerr << "\n  错误：" << exception.what() << "\n\n";
        status = 1;
    }

    PetscFinalize();
    return status;
}
