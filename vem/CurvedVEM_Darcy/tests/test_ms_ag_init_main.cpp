/**
 * @file test_ms_ag_init_main.cpp
 * @brief 测试 AG_init 矩阵（初值模式下 A_ij 加权 Gram 矩阵，各向异性张量形式）
 *
 * 使用 square_1x1.msh 网格，输出第 0 个单元的 AG_init 矩阵，
 * 并与 G_cmin（常数 c* 加权）对比，验证结构一致性。
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
    double cstar = pde.c_star;

    std::cout << "多项式次数 k = " << k << "\n";
    std::cout << "组分数 n_comp = " << n_comp << "\n";
    std::cout << "c* = " << cstar << "\n";
    std::cout << "H(div) 向量多项式维数: "
              << vem::basis::BasisFunctionPloy(k).getDimHdiv() << "\n";

    // 输出 c* 的 G_cmin 作为参考
    AutoPetscMat G_cmin = solver.getMatrixG_cmin(0);
    printMatrix(G_cmin, "G_cmin (c* 加权)");

    // 输出所有 (i,j) 对的 AG_init
    std::cout << "\n========== AG_init 矩阵 (A_ij 加权，初值模式) ==========\n";
    for (int i = 0; i < n_comp; ++i) {
        for (int j = 0; j < n_comp; ++j) {
            AutoPetscMat AG = solver.getMatrixAG_init(0, i, j);

            // 计算 A_ij 的平均值（质心处的值，粗略参考）
            double Aij_centroid = pde.evaluate_A_comp_initial(
                i, j, 0, md.cell_centroid_x[0], md.cell_centroid_y[0]);

            char name[128];
            snprintf(name, sizeof(name), "AG_init[%d,%d]  (A_ij ≈ %.6f)",
                     i, j, Aij_centroid);
            printMatrix(AG, name);

            // 验证对称性
            Mat raw = vem::get_raw(AG);
            PetscInt dim;
            MatGetSize(raw, &dim, &dim);
            double max_asym = 0.0;
            for (PetscInt r = 0; r < dim; ++r) {
                for (PetscInt c = r + 1; c < dim; ++c) {
                    PetscScalar v1, v2;
                    MatGetValue(raw, r, c, &v1);
                    MatGetValue(raw, c, r, &v2);
                    max_asym = std::max(max_asym, std::abs(v1 - v2));
                }
            }
            std::cout << "  最大不对称误差: " << max_asym << "\n";
        }
    }

    // 特别验证：若 A_ij = c*（常数），AG_init 应等于 G_cmin
    // 用一个常系数的特殊情况做一致性检验（对角元比值应为 A_ij/cstar 量级）
    std::cout << "\n========== 数值一致性检查 ==========\n";
    std::cout << "对比 G_cmin 与 AG_init[0,0] 的 Frobenius 范数比值（近似 A_00/c*）:\n";
    Mat raw_G = vem::get_raw(G_cmin);
    PetscInt dim;
    MatGetSize(raw_G, &dim, &dim);
    double frob_G = 0.0, frob_AG = 0.0;
    AutoPetscMat AG00 = solver.getMatrixAG_init(0, 0, 0);
    Mat raw_AG = vem::get_raw(AG00);
    for (PetscInt r = 0; r < dim; ++r) {
        for (PetscInt c = 0; c < dim; ++c) {
            PetscScalar gv, av;
            MatGetValue(raw_G, r, c, &gv);
            MatGetValue(raw_AG, r, c, &av);
            frob_G += gv * gv;
            frob_AG += av * av;
        }
    }
    frob_G = std::sqrt(frob_G);
    frob_AG = std::sqrt(frob_AG);
    std::cout << "  ||G_cmin||_F = " << frob_G << "\n";
    std::cout << "  ||AG[0,0]||_F = " << frob_AG << "\n";
    std::cout << "  比值 = " << frob_AG / frob_G << "\n";
    std::cout << "  （若 A_00 为常数，则应等于 A_00/c* = "
              << (pde.evaluate_A_comp_initial(0, 0, 0, md.cell_centroid_x[0],
                                               md.cell_centroid_y[0]) / cstar)
              << "，此处为近似参考，因为 A_00 不是常数）\n";

    PetscFinalize();
    return 0;
}
