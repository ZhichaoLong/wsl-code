/**
 * CurvedVEM Darcy - 主程序入口
 * 功能：计算算例1（曲边映射）在正方形四边形网格上的收敛阶
 * 网格密度：1x1, 2x2, 4x4, 8x8, 16x16, 32x32
 */

#include "../solver/DarcySolver.h"

#include <petsc.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>

namespace {

// 收敛结果数据结构
struct ConvergenceResult {
    std::string mesh_name;      // 网格文件名
    int num_nodes;              // 节点数
    int num_cells;              // 单元数
    int num_edges;              // 边数
    int system_size;            // 系统规模（自由度）
    double h;                   // 特征网格尺寸
    double pressure_error;      // 压力 L2 误差
    double flux_error;          // 通量 L2 误差
    double pressure_order;      // 压力收敛阶
    double flux_order;          // 通量收敛阶
    double solve_time;          // 求解时间（秒）
    int iterations;             // 迭代次数
};

/**
 * 获取网格文件名
 */
std::string getMeshName(int n) {
    return "mesh/data/square_" + std::to_string(n) + "x" + std::to_string(n) + ".msh";
}

/**
 * 运行单个算例并返回结果
 */
ConvergenceResult runTestCase(const std::string& mesh_file,
                               int problem_index,
                               int gauss_point_num,
                               int gauss_point_num_1d,
                               int k,
                               bool use_gpu = false) {
    vem::StraightMeshReader reader;
    if (!reader.read_mesh(mesh_file)) {
        throw std::runtime_error("无法读取网格: " + mesh_file);
    }

    Darcy::DarcyProblem problem;
    problem.set_problem_index(problem_index);
    if (!problem.init_problem()) {
        throw std::runtime_error("无法初始化算例");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    DarcySolver solver(reader, problem, gauss_point_num, gauss_point_num_1d, k);
    bool ok = use_gpu ? solver.solveGPU() : solver.solve();
    if (!ok) {
        throw std::runtime_error("Darcy 方程求解失败");
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    const auto& mesh = reader.get_mesh_data();
    double h = 1.0 / std::sqrt(static_cast<double>(mesh.num_cells));

    ConvergenceResult result;
    result.mesh_name = mesh_file;
    result.num_nodes = mesh.num_nodes;
    result.num_cells = mesh.num_cells;
    result.num_edges = mesh.num_edges;
    result.system_size = solver.getTotalDofFlux() + solver.getTotalDofConc() + 1;
    result.h = h;
    result.pressure_error = solver.computeL2ErrorPressure(gauss_point_num);
    result.flux_error = solver.computeL2ErrorFlux(gauss_point_num);
    result.pressure_order = 0.0;
    result.flux_order = 0.0;
    result.solve_time = elapsed.count();
    result.iterations = 0;  // PETSc迭代次数在solve中内部处理

    return result;
}

/**
 * 计算收敛阶（基于相邻网格的误差比）
 */
void computeConvergenceOrders(std::vector<ConvergenceResult>& results) {
    for (size_t i = 1; i < results.size(); ++i) {
        double log_h_ratio = std::log(results[i-1].h / results[i].h);
        double log_p_ratio = std::log(results[i-1].pressure_error / results[i].pressure_error);
        double log_f_ratio = std::log(results[i-1].flux_error / results[i].flux_error);
        results[i].pressure_order = log_p_ratio / log_h_ratio;
        results[i].flux_order = log_f_ratio / log_h_ratio;
    }
}

/**
 * 打印美观的收敛阶表格
 */
void printResults(const std::vector<ConvergenceResult>& results,
                  const std::string& title) {
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(100, '=') << "\n\n";

    // 表头
    std::cout << std::setw(16) << "网格"
              << std::setw(10) << "节点数"
              << std::setw(10) << "单元数"
              << std::setw(12) << "h"
              << std::setw(16) << "压力L2误差"
              << std::setw(12) << "压力阶"
              << std::setw(16) << "通量L2误差"
              << std::setw(12) << "通量阶"
              << std::setw(12) << "求解时间(s)"
              << "\n";

    std::cout << std::string(100, '-') << "\n";

    // 数据行
    for (const auto& r : results) {
        // 提取网格名称（去掉路径和后缀）
        std::string mesh_short = r.mesh_name.substr(10, r.mesh_name.size() - 14);

        std::cout << std::setw(16) << mesh_short
                  << std::setw(10) << r.num_nodes
                  << std::setw(10) << r.num_cells
                  << std::setw(12) << std::scientific << std::setprecision(4) << r.h
                  << std::setw(16) << std::scientific << std::setprecision(6) << r.pressure_error
                  << std::setw(12) << std::fixed << std::setprecision(4)
                  << (r.pressure_order > 0 ? r.pressure_order : 0.0)
                  << std::setw(16) << std::scientific << std::setprecision(6) << r.flux_error
                  << std::setw(12) << std::fixed << std::setprecision(4)
                  << (r.flux_order > 0 ? r.flux_order : 0.0)
                  << std::setw(12) << std::fixed << std::setprecision(4) << r.solve_time
                  << "\n";
    }

    std::cout << std::string(100, '-') << "\n\n";
}

/**
 * 打印总结信息
 */
void printSummary(const std::vector<ConvergenceResult>& results) {
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "  收敛阶总结\n";
    std::cout << std::string(100, '=') << "\n\n";

    std::cout << "  理论收敛阶（k=3）：\n";
    std::cout << "    - 压力：O(h⁴)  (k+1阶)\n";
    std::cout << "    - 通量：O(h⁴)  (k+1阶)\n\n";

    std::cout << "  实际观测收敛阶（最密网格）：\n";
    if (results.size() >= 2) {
        const auto& r = results.back();
        std::cout << "    - 压力：" << std::fixed << std::setprecision(4)
                  << r.pressure_order << " 阶\n";
        std::cout << "    - 通量：" << std::fixed << std::setprecision(4)
                  << r.flux_order << " 阶\n";
    }
    std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    // 如果有CUDA支持，提前设置 GPU 相关选项
#if defined(PETSC_HAVE_CUDA)
    setenv("CUDA_VISIBLE_DEVICES", "0", 1);
    PetscOptionsSetValue(NULL, "-use_gpu_aware_mpi", "0");
    PetscOptionsSetValue(NULL, "-device_type", "cuda");
    PetscOptionsSetValue(NULL, "-device_select", "0");
#endif

    // 初始化PETSc
    PetscErrorCode error = PetscInitialize(&argc, &argv, NULL, NULL);
    if (error != 0) {
        return static_cast<int>(error);
    }

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              Darcy_curved_main - 曲边虚单元法 Darcy 求解器       ║\n";
    std::cout << "║           算例1：正弦扰动曲边映射 + 正弦真解                      ║\n";
    std::cout << "║           网格：正方形四边形网格 (1x1 ~ 32x32)                   ║\n";
    std::cout << "║           多项式阶数：k = 3                                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

    try {
        // 配置参数
        const int problem_index = 1;      // 算例1：曲边映射
        const int k = 3;                   // 多项式阶数
        const int gauss_point_num = 12;    // 二维高斯积分点数 (k=3 需要更高精度)
        const int gauss_point_num_1d = 9;  // 一维高斯积分点数

        // 网格加密序列
        std::vector<int> mesh_sizes = {1, 2, 4, 8, 16, 32};
        const bool use_gpu = false;  // 使用 CPU 求解

        std::cout << "\n  开始计算收敛阶...\n";
        std::cout << "  求解器：" << (use_gpu ? "GPU (CUDA FGMRES+BJACOBI)" : "CPU (FGMRES+ILU)") << "\n";
        std::cout << "  二维积分点数：" << gauss_point_num << "\n";
        std::cout << "  一维积分点数：" << gauss_point_num_1d << "\n";
        std::cout << "  网格序列：1x1 → 2x2 → 4x4 → 8x8 → 16x16 → 32x32\n\n";

        // 运行所有算例
        std::vector<ConvergenceResult> results;
        for (size_t i = 0; i < mesh_sizes.size(); ++i) {
            int n = mesh_sizes[i];
            std::cout << "  [" << (i+1) << "/" << mesh_sizes.size() << "] "
                      << "计算网格 " << n << "x" << n << "... ";
            std::cout.flush();

            results.push_back(runTestCase(
                getMeshName(n), problem_index,
                gauss_point_num, gauss_point_num_1d, k, use_gpu
            ));

            std::cout << "✓ (" << results.back().num_cells << " 单元, "
                      << results.back().system_size << " 自由度)\n";
        }

        // 计算收敛阶
        computeConvergenceOrders(results);

        // 打印结果表格
        printResults(results, "算例1（曲边映射）收敛阶结果 - 正方形四边形网格, k=3, GPU求解");

        // 打印总结
        printSummary(results);

    } catch (const std::exception& exception) {
        std::cerr << "\n  错误：" << exception.what() << "\n\n";
        PetscFinalize();
        return 1;
    }

    std::cout << "\n  计算完成！\n\n";
    PetscFinalize();
    return 0;
}
