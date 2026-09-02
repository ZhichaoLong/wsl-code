/**
 * @file test_ms_bc_main.cpp
 * @brief 测试 applyNormalFluxBoundaryConditions 边界条件处理
 */
#include "../solver/MSSolver.h"
#include "../examples/ms_problem.h"
#include "../mesh/straight_mesh.h"

#include <petsc.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
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

    MaxwellStefan::MSProblem ms_problem;
    ms_problem.set_problem_index(1);
    ms_problem.init_problem();

    const int k = 1;
    MSSolver ms_solver(mesh, ms_problem, 0.01, 1.0, 10, 1e-8, 9, 9, k);
    ms_solver.initialize();

    int n_comp = ms_solver.getNumComponents();
    int total_dof = ms_solver.getTotalDofAll();
    int total_flux = ms_solver.getTotalDofFluxAll();
    int flux_per_comp = ms_solver.getTotalDofFlux();

    const auto& md = *ms_solver.getMeshData();
    std::cout << "网格: " << md.num_cells << " 单元, "
              << md.num_edges << " 边, "
              << md.num_boundary_edges << " 边界边\n";
    std::cout << "k = " << k << ", 组分数 = " << n_comp << "\n";
    std::cout << "总自由度: " << total_dof
              << " (通量: " << total_flux << ")\n";

    // 组装
    ms_solver.assembleStiffnessMatrix(true, true);
    std::cout << "\n✓ 组装完成\n";

    // 施加边界条件
    AutoPetscMat K_free;
    AutoPetscVec F_free;
    IS free_dofs = PETSC_NULLPTR;
    AutoPetscVec bd_val;
    PetscInt total_dof_ref = 0;

    PetscErrorCode ierr = ms_solver.applyNormalFluxBoundaryConditions(
        K_free, F_free, free_dofs, bd_val, total_dof_ref, 0.0);
    if (ierr != 0) {
        std::cerr << "✗ applyNormalFluxBoundaryConditions 失败: " << ierr << "\n";
        PetscFinalize();
        return 1;
    }
    std::cout << "✓ 边界条件施加成功\n";

    // 检查维度
    PetscInt free_rows, free_cols;
    MatGetSize(vem::get_raw(K_free), &free_rows, &free_cols);
    PetscInt free_size;
    VecGetSize(vem::get_raw(F_free), &free_size);

    std::cout << "\n========== 维度检查 ==========\n";
    std::cout << "总自由度: " << total_dof_ref << "\n";
    std::cout << "自由自由度: " << free_rows
              << " (矩阵 " << free_rows << " x " << free_cols << ")\n";
    std::cout << "右端项自由自由度: " << free_size << "\n";

    // 预期边界通量自由度数
    int bd_edges = md.num_boundary_edges;
    int expected_bd_flux_per_comp = bd_edges * (k + 1);
    int expected_bd_flux = expected_bd_flux_per_comp * n_comp;
    int expected_free_flux = total_flux - expected_bd_flux;
    int expected_free_total = expected_free_flux + ms_solver.getTotalDofConcAll();

    std::cout << "\n边界通量自由度(预期): " << expected_bd_flux
              << " (" << expected_bd_flux_per_comp << "/组分 × " << n_comp << ")\n";
    std::cout << "自由自由度(预期): " << expected_free_total
              << " (通量: " << expected_free_flux
              << " + 浓度: " << ms_solver.getTotalDofConcAll() << ")\n";
    std::cout << "自由自由度(实际): " << free_rows;
    if (free_rows == expected_free_total) {
        std::cout << " ✓\n";
    } else {
        std::cout << " ✗\n";
    }

    // 边界值检查（零通量边界，所以边界值应该都是 0）
    Vec raw_bd = vem::get_raw(bd_val);
    double max_bd_val = 0.0;
    int bd_count = 0;
    for (int comp = 0; comp < n_comp; ++comp) {
        int comp_off = comp * flux_per_comp;
        for (int e = 0; e < md.num_boundary_edges; ++e) {
            int global_edge = md.boundary_edges[e];
            for (int d = 0; d <= k; ++d) {
                PetscInt idx = comp_off + d * md.num_edges + global_edge;
                PetscScalar val;
                VecGetValues(raw_bd, 1, &idx, &val);
                double v = std::abs(PetscRealPart(val));
                if (v > max_bd_val) max_bd_val = v;
                bd_count++;
            }
        }
    }
    std::cout << "\n========== 边界值检查（零通量）==========\n";
    std::cout << "边界自由度总数: " << bd_count << "\n";
    std::cout << "边界值最大绝对值: " << std::scientific << std::setprecision(3)
              << max_bd_val;
    if (max_bd_val < 1e-12) {
        std::cout << " ✓（均为 0）\n";
    } else {
        std::cout << " ✗\n";
    }

    // 自由自由度的矩阵范数
    PetscReal fnorm_free;
    MatNorm(vem::get_raw(K_free), NORM_FROBENIUS, &fnorm_free);
    PetscReal norm_f_free;
    VecNorm(vem::get_raw(F_free), NORM_2, &norm_f_free);

    std::cout << "\n========== 约化系统统计 ==========\n";
    std::cout << "自由矩阵 F 范数: " << std::scientific << std::setprecision(6)
              << fnorm_free << "\n";
    std::cout << "自由右端项 2 范数: " << norm_f_free << "\n";

    // 尝试求解一次（验证系统非奇异）
    AutoPetscVec sol_free = vem::create_vector(free_rows);
    PetscBool converged = PETSC_FALSE;
    ierr = vem::solve_linear_system(K_free, F_free, sol_free, &converged, false);
    std::cout << "\n========== 求解测试 ==========\n";
    if (ierr == 0 && converged) {
        std::cout << "✓ 约化系统求解成功（GMRES 收敛）\n";
    } else {
        std::cout << "✗ 约化系统求解失败 (ierr=" << ierr
                  << ", converged=" << (converged ? "T" : "F") << ")\n";
    }

    if (free_dofs != PETSC_NULLPTR) {
        ISDestroy(&free_dofs);
    }

    std::cout << "\n✓ 边界条件测试完成\n";
    PetscFinalize();
    return 0;
}
