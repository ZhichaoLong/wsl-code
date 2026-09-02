/**
 * @file test_ms_stab_coeff_main.cpp
 * @brief 测试稳定化系数（c* 线性项 + A_ij 非线性项 init/poly 两种模式）
 *
 * 使用 square_1x1.msh 网格，输出第 0 个单元的稳定化系数，
 * 并验证 poly 模式与 init 模式在分段常数初值下一致。
 */
#include "../solver/MSSolver.h"
#include "../examples/ms_problem.h"
#include "../mesh/straight_mesh.h"

#include <petsc.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>

int main(int argc, char* argv[]) {
    PetscInitialize(&argc, &argv, NULL, NULL);

    std::string mesh_file = "mesh/data/square_1x1.msh";
    if (argc > 1) mesh_file = argv[1];

    vem::StraightMeshReader mesh;
    if (!mesh.read_mesh(mesh_file)) {
        std::cerr << "无法读取网格: " << mesh_file << std::endl;
        PetscFinalize();
        return 1;
    }

    const auto& md = mesh.get_mesh_data();

    MaxwellStefan::MSProblem problem;
    problem.set_problem_index(1);
    problem.init_problem();

    const double dt = 0.01;
    const double T = 1.0;
    const int picard_max = 10;
    const double picard_tol = 1e-8;
    const int k = 1;

    MSSolver solver(mesh, problem, dt, T, picard_max, picard_tol,
                    9, 9, k);
    solver.initialize();

    const auto& pde = problem.get_pde_data();
    int n_comp = pde.n_components;
    int dof_conc = solver.getDofPerElementConc();
    int elem = 0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "k = " << k << ", 单元 = " << elem << "\n";
    std::cout << "单元面积 = " << md.cell_area[elem]
              << ", 直径 = " << md.cell_diameter[elem] << "\n";
    std::cout << "c* = " << pde.c_star << "\n";

    // c* 线性项稳定化系数
    double alpha_cmin = solver.getStabilizationCoeffCmin(elem);
    std::cout << "\n========== 稳定化系数 ==========\n";
    std::cout << "alpha_cmin (c* 线性项) = " << alpha_cmin << "\n";

    // 构造多项式系数（分段常数初值）
    std::vector<double> u_ploy_coeff(n_comp * dof_conc, 0.0);
    for (int c = 0; c < n_comp; ++c) {
        double c0 = pde.initial_concentration_comp(
            c, md.cell_centroid_x[elem], md.cell_centroid_y[elem]);
        u_ploy_coeff[c * dof_conc + 0] = c0;
    }
    std::cout << "\n浓度初值（质心处）: ";
    for (int c = 0; c < n_comp; ++c) {
        if (c > 0) std::cout << ", ";
        std::cout << "c" << c << "=" << u_ploy_coeff[c * dof_conc];
    }
    std::cout << "\n";

    // A_ij 非线性项稳定化系数（init 模式）
    std::cout << "\n--- alpha_A (A_ij 非线性项, init 模式) ---\n";
    for (int i = 0; i < n_comp; ++i) {
        for (int j = 0; j < n_comp; ++j) {
            double alpha = solver.getStabilizationCoeffInit(elem, i, j);
            double Aij = pde.evaluate_A_comp_initial(
                i, j, md.cell_centroid_x[elem], md.cell_centroid_y[elem]);
            std::cout << "  alpha[" << i << "," << j << "] = "
                      << std::setw(12) << alpha
                      << "  (A_ij ≈ " << std::setw(8) << Aij << ")\n";
        }
    }

    // 验证 poly 模式与 init 模式一致
    std::cout << "\n--- 验证 poly 模式 vs init 模式 ---\n";
    double max_diff = 0.0;
    for (int i = 0; i < n_comp; ++i) {
        for (int j = 0; j < n_comp; ++j) {
            double a_init = solver.getStabilizationCoeffInit(elem, i, j);
            double a_poly = solver.getStabilizationCoeffPoly(elem, i, j, u_ploy_coeff);
            double diff = std::abs(a_init - a_poly);
            max_diff = std::max(max_diff, diff);
            std::cout << "  [" << i << "," << j << "]: |init - poly| = "
                      << diff << "\n";
        }
    }
    std::cout << "最大误差: " << max_diff;
    if (max_diff < 1e-12) {
        std::cout << "  ✓ 一致\n";
    } else {
        std::cout << "  ✗ 不一致\n";
    }

    // 额外验证：若 A_ij = 常数，则 alpha / alpha_cmin ≈ A_ij / c*
    std::cout << "\n--- 比值验证（alpha_A / alpha_cmin ≈ A_ij / c*） ---\n";
    for (int i = 0; i < n_comp; ++i) {
        for (int j = 0; j < n_comp; ++j) {
            double alpha = solver.getStabilizationCoeffInit(elem, i, j);
            double Aij = pde.evaluate_A_comp_initial(
                i, j, md.cell_centroid_x[elem], md.cell_centroid_y[elem]);
            double ratio_alpha = alpha / alpha_cmin;
            double ratio_A = Aij / pde.c_star;
            std::cout << "  [" << i << "," << j << "]: "
                      << "alpha_ratio = " << std::setw(10) << ratio_alpha
                      << ", A/c* = " << std::setw(10) << ratio_A;
            if (std::abs(Aij) > 1e-14) {
                double rel = std::abs(ratio_alpha - ratio_A) / std::abs(ratio_A);
                std::cout << ", 相对误差 = " << rel;
            }
            std::cout << "\n";
        }
    }

    PetscFinalize();
    return 0;
}
