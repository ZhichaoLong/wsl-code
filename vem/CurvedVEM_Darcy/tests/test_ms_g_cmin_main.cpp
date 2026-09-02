/**
 * @file test_ms_g_cmin_main.cpp
 * @brief 测试 G_cmin 矩阵（c* 加权 Gram 矩阵，各向异性张量形式）
 *
 * 使用 square_1x1.msh 网格，输出第 0 个单元的 G_cmin 矩阵，
 * 同时输出无权 G 矩阵作为对比参考。
 */
#include "../solver/MSSolver.h"
#include "../examples/ms_problem.h"
#include "../mesh/straight_mesh.h"

#include <petsc.h>
#include <iostream>
#include <iomanip>
#include <string>

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
            // 正数前面加空格对齐
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
    std::cout << "网格信息：\n";
    std::cout << "  节点数: " << md.num_nodes << "\n";
    std::cout << "  单元数: " << md.num_cells << "\n";
    std::cout << "  边数: " << md.num_edges << "\n";
    std::cout << "  第 0 个单元节点数: " << md.nodes_per_cell[0] << "\n";
    std::cout << "  第 0 个单元质心: (" << md.cell_centroid_x[0]
              << ", " << md.cell_centroid_y[0] << ")\n";
    std::cout << "  第 0 个单元直径: " << md.cell_diameter[0] << "\n";
    std::cout << "  第 0 个单元面积: " << md.cell_area[0] << "\n";

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

    std::cout << "\n多项式次数 k = " << k << "\n";
    std::cout << "H(div) 单组分单元自由度: " << solver.getDofPerElementFlux(0) << "\n";
    std::cout << "H(div) 向量多项式维数: "
              << vem::basis::BasisFunctionPloy(k).getDimHdiv() << "\n";
    std::cout << "c* = " << problem.get_pde_data().c_star << "\n";

    // 输出 G_cmin 矩阵
    AutoPetscMat G_cmin = solver.getMatrixG_cmin(0);
    printMatrix(G_cmin, "G_cmin (c* 加权 Gram, 各向异性)");

    // 对比：无权 G 矩阵（基类）
    AutoPetscMat G_plain = solver.getMatrixG(0);
    printMatrix(G_plain, "G (无权 Gram, 参考)");

    // 验证对称性
    Mat raw = vem::get_raw(G_cmin);
    PetscInt dim;
    MatGetSize(raw, &dim, &dim);
    double max_asym = 0.0;
    for (PetscInt i = 0; i < dim; ++i) {
        for (PetscInt j = i + 1; j < dim; ++j) {
            PetscScalar v1, v2;
            MatGetValue(raw, i, j, &v1);
            MatGetValue(raw, j, i, &v2);
            max_asym = std::max(max_asym, std::abs(v1 - v2));
        }
    }
    std::cout << "\nG_cmin 最大不对称误差: " << max_asym
              << " (J^T J 理论上对称，应接近机器精度)\n";

    PetscFinalize();
    return 0;
}
