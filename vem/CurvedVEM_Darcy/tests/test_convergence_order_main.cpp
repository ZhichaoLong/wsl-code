#include "../solver/DarcySolver.h"

#include <petsc.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ConvergenceResult {
    std::string mesh_name;
    int num_cells;
    double h;
    double pressure_error;
    double flux_error;
    double pressure_order;
    double flux_order;
};

std::string getMeshName(int n) {
    return "/home/lzccode/code/vem/gmesh_py/mesh_data/square_" + std::to_string(n) + "x" + std::to_string(n) + ".msh";
}

ConvergenceResult runTestCase(const std::string& mesh_file, int problem_index,
                               int gauss_point_num, int gauss_point_num_1d, int k) {
    vem::StraightMeshReader reader;
    if (!reader.read_mesh(mesh_file)) {
        throw std::runtime_error("无法读取网格: " + mesh_file);
    }

    Darcy::DarcyProblem problem;
    problem.set_problem_index(problem_index);
    if (!problem.init_problem()) {
        throw std::runtime_error("无法初始化算例");
    }

    DarcySolver solver(reader, problem, gauss_point_num, gauss_point_num_1d, k);
    if (!solver.solve()) {
        throw std::runtime_error("Darcy 方程求解失败");
    }

    const auto& mesh = reader.get_mesh_data();
    double h = 1.0 / std::sqrt(static_cast<double>(mesh.num_cells));

    ConvergenceResult result;
    result.mesh_name = mesh_file;
    result.num_cells = mesh.num_cells;
    result.h = h;
    result.pressure_error = solver.computeL2ErrorPressure(gauss_point_num);
    result.flux_error = solver.computeL2ErrorFlux(gauss_point_num);
    result.pressure_order = 0.0;
    result.flux_order = 0.0;

    return result;
}

void computeConvergenceOrders(std::vector<ConvergenceResult>& results) {
    for (size_t i = 1; i < results.size(); ++i) {
        double log_h_ratio = std::log(results[i-1].h / results[i].h);
        double log_p_ratio = std::log(results[i-1].pressure_error / results[i].pressure_error);
        double log_f_ratio = std::log(results[i-1].flux_error / results[i].flux_error);
        results[i].pressure_order = log_p_ratio / log_h_ratio;
        results[i].flux_order = log_f_ratio / log_h_ratio;
    }
}

void printResults(const std::vector<ConvergenceResult>& results,
                  const std::string& title) {
    std::cout << "\n========================================\n";
    std::cout << title << "\n";
    std::cout << "========================================\n";
    std::cout << std::setw(12) << "网格"
              << std::setw(10) << "单元数"
              << std::setw(12) << "h"
              << std::setw(16) << "压力L2误差"
              << std::setw(12) << "压力阶"
              << std::setw(16) << "通量L2误差"
              << std::setw(12) << "通量阶" << "\n";
    std::cout << std::string(100, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::setw(12) << r.mesh_name.substr(17)
                  << std::setw(10) << r.num_cells
                  << std::setw(12) << std::scientific << std::setprecision(4) << r.h
                  << std::setw(16) << std::scientific << std::setprecision(6) << r.pressure_error
                  << std::setw(12) << std::fixed << std::setprecision(4)
                  << (r.pressure_order > 0 ? r.pressure_order : 0.0)
                  << std::setw(16) << std::scientific << std::setprecision(6) << r.flux_error
                  << std::setw(12) << std::fixed << std::setprecision(4)
                  << (r.flux_order > 0 ? r.flux_order : 0.0) << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    PetscErrorCode error = PetscInitialize(&argc, &argv, NULL, NULL);
    if (error != 0) {
        return static_cast<int>(error);
    }

    try {
        int problem_index = 1;  // 算例1（曲边）
        int k = 1;  // 默认多项式阶数
        int gauss_point_num = 9;
        int gauss_point_num_1d = 9;

        // 网格加密序列
        std::vector<int> mesh_sizes = {1, 2, 4, 8, 16, 32};  // 到32x32，64x64耗时较长

        std::vector<ConvergenceResult> results;
        for (int n : mesh_sizes) {
            std::cout << "正在计算网格: " << n << "x" << n << "...\n";
            results.push_back(runTestCase(getMeshName(n), problem_index,
                                          gauss_point_num, gauss_point_num_1d, k));
        }

        computeConvergenceOrders(results);
        printResults(results, "Darcy 方程收敛阶测试 (算例" +
                     std::to_string(problem_index) + ", k=" + std::to_string(k) + ")");

    } catch (const std::exception& exception) {
        std::cerr << "收敛阶测试失败: " << exception.what() << std::endl;
        PetscFinalize();
        return 1;
    }

    PetscFinalize();
    return 0;
}
