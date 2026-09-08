/**
 * @file test_ms_ag_poly_main.cpp
 * @brief 测试 AG_poly 矩阵（数值解模式，用多项式系数还原浓度计算 A_ij）
 *
 * 验证方法：构造初值对应的浓度多项式系数，调用 AG_poly，
 * 结果应与 AG_init（直接用初值函数）一致。
 * 算例 1 的初值是分段常数，正好是 P_k 的常数项，
 * 高阶单项式系数为 0。
 */
#include "../solver/MSSolver.h"
#include "../examples/ms_problem.h"
#include "../mesh/straight_mesh.h"

#include <petsc.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>

void printMatrix(const AutoPetscMat& mat, const char* name) {
    Mat raw = vem::get_raw(mat);
    PetscInt m, n;
    MatGetSize(raw, &m, &n);

    std::cout << "\n===== " << name << " (" << m << " x " << n << ") =====\n";
    std::cout << std::scientific << std::setprecision(6);

    for (PetscInt i = 0; i < m; ++i) {
        for (PetscInt j = 0; j < n; ++j) {
            PetscScalar val;
            MatGetValue(raw, i, j, &val);
            if (j > 0) std::cout << " ";
            if (val >= 0.0) std::cout << " ";
            std::cout << val;
        }
        std::cout << "\n";
    }
}

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
    int dof_conc = solver.getDofPerElementConc();  // dim(P_k)
    int elem = 0;  // 第 0 个单元

    std::cout << "k = " << k << ", dim(P_k) = " << dof_conc << "\n";
    std::cout << "组分数 = " << n_comp << "\n";
    std::cout << "测试单元 = " << elem << "\n";
    std::cout << "单元质心 = (" << md.cell_centroid_x[elem]
              << ", " << md.cell_centroid_y[elem] << ")\n";

    // 构造初值多项式系数：
    // 算例 1 是分段常数初值，第 0 个单元的浓度在质心处求值就是常数项
    // P_k 单项式中第 0 个是常数项 1，其余高阶项系数为 0
    std::vector<double> u_ploy_coeff(n_comp * dof_conc, 0.0);
    for (int c = 0; c < n_comp; ++c) {
        double c0 = pde.initial_concentration_comp(
            c, elem, md.cell_centroid_x[elem], md.cell_centroid_y[elem]);
        u_ploy_coeff[c * dof_conc + 0] = c0;  // 常数项
        // 高阶项（x, y, x^2, ...）系数为 0
    }

    std::cout << "\n构造的浓度多项式系数 (按组分连续):\n";
    for (int c = 0; c < n_comp; ++c) {
        std::cout << "  comp " << c << ": ";
        for (int m = 0; m < dof_conc; ++m) {
            if (m > 0) std::cout << ", ";
            std::cout << u_ploy_coeff[c * dof_conc + m];
        }
        std::cout << "\n";
    }

    // 对比 AG_init 与 AG_poly
    std::cout << "\n========== 验证 AG_poly 与 AG_init 一致性 ==========\n";
    double max_diff_total = 0.0;

    for (int i = 0; i < n_comp; ++i) {
        for (int j = 0; j < n_comp; ++j) {
            AutoPetscMat AG_init = solver.getMatrixAG_init(elem, i, j);
            AutoPetscMat AG_poly = solver.getMatrixAG_poly(elem, i, j, u_ploy_coeff);

            Mat raw_init = vem::get_raw(AG_init);
            Mat raw_poly = vem::get_raw(AG_poly);
            PetscInt dim;
            MatGetSize(raw_init, &dim, &dim);

            double max_diff = 0.0;
            for (PetscInt r = 0; r < dim; ++r) {
                for (PetscInt c = 0; c < dim; ++c) {
                    PetscScalar v1, v2;
                    MatGetValue(raw_init, r, c, &v1);
                    MatGetValue(raw_poly, r, c, &v2);
                    max_diff = std::max(max_diff, std::abs(v1 - v2));
                }
            }

            double Aij = pde.evaluate_A_comp_initial(
                i, j, elem, md.cell_centroid_x[elem], md.cell_centroid_y[elem]);
            std::cout << "AG[" << i << "," << j << "]: "
                      << "max|AG_init - AG_poly| = " << max_diff
                      << "  (A_ij ≈ " << Aij << ")\n";
            max_diff_total = std::max(max_diff_total, max_diff);
        }
    }

    std::cout << "\n所有组分对的最大误差: " << max_diff_total << "\n";
    if (max_diff_total < 1e-12) {
        std::cout << "✓ AG_poly 与 AG_init 一致（分段常数初值下应精确相等）\n";
    } else {
        std::cout << "✗ AG_poly 与 AG_init 不一致！\n";
    }

    PetscFinalize();
    return 0;
}
