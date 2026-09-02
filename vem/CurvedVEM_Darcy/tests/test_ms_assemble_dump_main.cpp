/**
 * @file test_ms_assemble_dump_main.cpp
 * @brief 输出 assembleStiffnessMatrix 生成的全局矩阵和右端项
 *
 * 输出到 build/ms_assemble_K.txt 和 build/ms_assemble_F.txt
 */
#include "../solver/MSSolver.h"
#include "../examples/ms_problem.h"
#include "../mesh/straight_mesh.h"

#include <petsc.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

int main(int argc, char* argv[]) {
    PetscInitialize(&argc, &argv, NULL, NULL);

    std::string mesh_file = "mesh/data/square_1x1.msh";
    std::string out_K = "build/ms_assemble_K.txt";
    std::string out_F = "build/ms_assemble_F.txt";
    if (argc > 1) mesh_file = argv[1];
    if (argc > 2) out_K = argv[2];
    if (argc > 3) out_F = argv[3];

    vem::StraightMeshReader mesh;
    if (!mesh.read_mesh(mesh_file)) {
        std::cerr << "无法读取网格: " << mesh_file << std::endl;
        PetscFinalize();
        return 1;
    }

    MaxwellStefan::MSProblem ms_problem;
    ms_problem.set_problem_index(1);
    ms_problem.init_problem();

    const int k = 1;
    MSSolver ms_solver(mesh, ms_problem, 0.01, 1.0, 10, 1e-8, 9, 9, k);
    ms_solver.initialize();

    int n_comp = ms_solver.getNumComponents();
    int total_dof = ms_solver.getTotalDofAll();
    int total_flux = ms_solver.getTotalDofFluxAll();
    int total_conc = ms_solver.getTotalDofConcAll();
    int flux_per_comp = ms_solver.getTotalDofFlux();
    int conc_per_comp = ms_solver.getTotalDofConc();

    const auto& md = *ms_solver.getMeshData();
    std::cout << "网格: " << md.num_cells << " 单元, "
              << md.num_nodes << " 节点, " << md.num_edges << " 边\n";
    std::cout << "k = " << k << ", 组分数 = " << n_comp << "\n";
    std::cout << "通量自由度: " << total_flux << " ("
              << flux_per_comp << "/组分)\n";
    std::cout << "浓度自由度: " << total_conc << " ("
              << conc_per_comp << "/组分)\n";
    std::cout << "总自由度: " << total_dof << "\n";

    // ========== 初值模式组装 ==========
    std::cout << "\n组装初值模式 (is_initial_step=true, is_initial_first_tstep=true)...\n";
    PetscErrorCode ierr = ms_solver.assembleStiffnessMatrix(true, true);
    if (ierr != 0) {
        std::cerr << "组装失败: " << ierr << "\n";
        PetscFinalize();
        return 1;
    }
    std::cout << "✓ 组装完成\n";

    Mat K = vem::get_raw(ms_solver.getSystemStiffnessMatrix());
    Vec F = vem::get_raw(ms_solver.getSystemRhsVec());

    PetscInt n, m;
    MatGetSize(K, &n, &m);

    // ========== 输出矩阵 ==========
    std::cout << "写入矩阵到 " << out_K << " ...\n";
    std::ofstream fK(out_K);
    fK << std::scientific << std::setprecision(12);
    fK << "# 全局刚度矩阵 K (" << n << " x " << m << ")\n";
    fK << "# 格式: row col value (仅非零元，阈值 1e-15)\n";
    fK << "# 自由度布局: 先通量(按组分连续)，后浓度(按组分连续)\n";
    fK << "#   通量范围: [0, " << total_flux << ")\n";
    fK << "#   浓度范围: [" << total_flux << ", " << total_dof << ")\n";

    PetscInt nnz_total = 0;
    const double tol = 1e-15;
    for (PetscInt i = 0; i < n; ++i) {
        PetscInt ncols;
        const PetscInt* cols;
        const PetscScalar* vals;
        MatGetRow(K, i, &ncols, &cols, &vals);
        for (PetscInt j = 0; j < ncols; ++j) {
            double v = PetscRealPart(vals[j]);
            if (std::abs(v) > tol) {
                fK << i << " " << cols[j] << " " << v << "\n";
                nnz_total++;
            }
        }
        MatRestoreRow(K, i, &ncols, &cols, &vals);
    }
    fK.close();
    std::cout << "  非零元数: " << nnz_total << "\n";

    // ========== 输出右端项 ==========
    std::cout << "写入右端项到 " << out_F << " ...\n";
    std::ofstream fF(out_F);
    fF << std::scientific << std::setprecision(12);
    fF << "# 全局右端项向量 F (" << total_dof << ")\n";
    fF << "# 格式: index value\n";
    fF << "# 自由度布局: 先通量(按组分连续)，后浓度(按组分连续)\n";
    fF << "#   通量范围: [0, " << total_flux << ")\n";
    fF << "#   浓度范围: [" << total_flux << ", " << total_dof << ")\n";

    int nonzeros_F = 0;
    for (PetscInt i = 0; i < total_dof; ++i) {
        PetscScalar val;
        VecGetValues(F, 1, &i, &val);
        double v = PetscRealPart(val);
        fF << i << " " << v << "\n";
        if (std::abs(v) > tol) nonzeros_F++;
    }
    fF.close();
    std::cout << "  非零元数: " << nonzeros_F << "\n";

    // ========== 统计信息 ==========
    PetscReal fnorm_K;
    MatNorm(K, NORM_FROBENIUS, &fnorm_K);
    PetscReal norm_F;
    VecNorm(F, NORM_2, &norm_F);

    std::cout << "\n========== 统计信息 ==========\n";
    std::cout << "矩阵 F 范数: " << std::scientific << std::setprecision(6)
              << fnorm_K << "\n";
    std::cout << "右端项 2 范数: " << norm_F << "\n";

    // 各分块的范数
    std::cout << "\n各组分分块 F 范数（通量-通量对角块）:\n";
    for (int c = 0; c < n_comp; ++c) {
        PetscReal bnorm = 0.0;
        int start = c * flux_per_comp;
        int end = start + flux_per_comp;
        for (int i = start; i < end; ++i) {
            PetscInt ncols;
            const PetscInt* cols;
            const PetscScalar* vals;
            MatGetRow(K, i, &ncols, &cols, &vals);
            for (int j = 0; j < ncols; ++j) {
                if (cols[j] >= start && cols[j] < end) {
                    double v = PetscRealPart(vals[j]);
                    bnorm += v * v;
                }
            }
            MatRestoreRow(K, i, &ncols, &cols, &vals);
        }
        std::cout << "  组分 " << c << " K_flux_flux 块: "
                  << std::sqrt(bnorm) << "\n";
    }

    std::cout << "\n浓度块 F 范数:\n";
    for (int c = 0; c < n_comp; ++c) {
        PetscReal bnorm = 0.0;
        int start = total_flux + c * conc_per_comp;
        int end = start + conc_per_comp;
        for (int i = start; i < end; ++i) {
            PetscInt ncols;
            const PetscInt* cols;
            const PetscScalar* vals;
            MatGetRow(K, i, &ncols, &cols, &vals);
            for (int j = 0; j < ncols; ++j) {
                if (cols[j] >= start && cols[j] < end) {
                    double v = PetscRealPart(vals[j]);
                    bnorm += v * v;
                }
            }
            MatRestoreRow(K, i, &ncols, &cols, &vals);
        }
        std::cout << "  组分 " << c << " K_conc_conc 块: "
                  << std::sqrt(bnorm) << "\n";
    }

    std::cout << "\n✓ 完成\n";
    PetscFinalize();
    return 0;
}
