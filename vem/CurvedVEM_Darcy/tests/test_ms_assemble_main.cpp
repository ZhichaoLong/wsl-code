/**
 * @file test_ms_assemble_main.cpp
 * @brief 测试 assembleStiffnessMatrix 全局组装主函数
 *
 * 测试内容：
 *   1. 初值模式组装 (is_initial_step=true, is_initial_first_tstep=true)
 *   2. 矩阵维度、右端项维度正确性
 *   3. 矩阵对称性（对角线块）
 *   4. 鞍点结构（W 块 + W^T 块对应位置取负）
 */
#include "../solver/MSSolver.h"
#include "../examples/ms_problem.h"
#include "../mesh/straight_mesh.h"

#include <petsc.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>

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
    int total_conc = ms_solver.getTotalDofConcAll();

    std::cout << "k = " << k << ", 组分数 = " << n_comp << "\n";
    std::cout << "全局自由度总数: " << total_dof
              << " (通量: " << total_flux
              << ", 浓度: " << total_conc << ")\n";

    // ========== 测试 1: 初值模式组装 ==========
    std::cout << "\n========== 测试 1: 初值模式组装 (step0, picard0) ==========\n";
    PetscErrorCode ierr = ms_solver.assembleStiffnessMatrix(true, true);
    if (ierr != 0) {
        std::cerr << "✗ assembleStiffnessMatrix 返回错误码: " << ierr << "\n";
        PetscFinalize();
        return 1;
    }
    std::cout << "✓ 组装成功\n";

    // 检查矩阵维度
    Mat K = vem::get_raw(ms_solver.getSystemStiffnessMatrix());
    Vec F = vem::get_raw(ms_solver.getSystemRhsVec());

    PetscInt n_rows, n_cols;
    MatGetSize(K, &n_rows, &n_cols);
    PetscInt f_size;
    VecGetSize(F, &f_size);

    bool dim_ok = (n_rows == total_dof && n_cols == total_dof && f_size == total_dof);
    std::cout << "矩阵维度: " << n_rows << " x " << n_cols
              << " (期望 " << total_dof << " x " << total_dof << ") "
              << (dim_ok ? "✓" : "✗") << "\n";
    std::cout << "右端项维度: " << f_size
              << " (期望 " << total_dof << ") "
              << (f_size == total_dof ? "✓" : "✗") << "\n";

    // 计算矩阵的 Frobenius 范数
    PetscReal fnorm_K;
    MatNorm(K, NORM_FROBENIUS, &fnorm_K);
    std::cout << "矩阵 F 范数: " << std::scientific << std::setprecision(6)
              << fnorm_K << "\n";

    PetscReal norm_F;
    VecNorm(F, NORM_2, &norm_F);
    std::cout << "右端项 2 范数: " << std::scientific << std::setprecision(6)
              << norm_F << "\n";

    // ========== 测试 2: 对角块对称性检查（通量-通量块）==========
    std::cout << "\n========== 测试 2: 对角线通量块对称性 ==========\n";
    int flux_per_comp = ms_solver.getTotalDofFlux();
    bool all_sym = true;
    double max_asym = 0.0;

    for (int c = 0; c < n_comp; ++c) {
        int start = c * flux_per_comp;
        int end = start + flux_per_comp;
        double block_asym = 0.0;
        for (int i = start; i < end; ++i) {
            for (int j = i + 1; j < end; ++j) {
                PetscScalar v_ij, v_ji;
                MatGetValues(K, 1, &i, 1, &j, &v_ij);
                MatGetValues(K, 1, &j, 1, &i, &v_ji);
                double diff = std::abs(PetscRealPart(v_ij - v_ji));
                if (diff > block_asym) block_asym = diff;
            }
        }
        std::cout << "  组分 " << c << " 通量块 max|K_ij - K_ji| = "
                  << std::scientific << std::setprecision(3) << block_asym;
        if (block_asym > 1e-10) {
            std::cout << " ✗\n";
            all_sym = false;
        } else {
            std::cout << " ✓\n";
        }
        if (block_asym > max_asym) max_asym = block_asym;
    }
    std::cout << "总体对称性: "
              << (all_sym ? "✓ 所有对角通量块对称" : "✗ 存在不对称") << "\n";

    // ========== 测试 3: 鞍点结构（W 和 -W^T）==========
    // 检查 flux-conc 和 conc-flux 块的转置关系（取负）
    std::cout << "\n========== 测试 3: 鞍点结构 W 与 -W^T 关系 ==========\n";
    int conc_per_comp = ms_solver.getTotalDofConc();
    bool saddle_ok = true;
    double max_saddle_err = 0.0;
    // 只检查第 0 组分（所有组分结构相同）
    {
        int c = 0;
        int flux_start = c * flux_per_comp;
        int conc_start = total_flux + c * conc_per_comp;
        int nf = std::min(flux_per_comp, 20);  // 只抽前 20 个检查
        int nc = std::min(conc_per_comp, 10);
        double err = 0.0;
        for (int i = 0; i < nf; ++i) {
            for (int j = 0; j < nc; ++j) {
                PetscScalar v_flux_conc, v_conc_flux;
                int fi = flux_start + i;
                int ci = conc_start + j;
                MatGetValues(K, 1, &fi, 1, &ci, &v_flux_conc);
                MatGetValues(K, 1, &ci, 1, &fi, &v_conc_flux);
                // K(flux,conc) 应该 = -K(conc,flux)^T 即 v_flux_conc + v_conc_flux = 0
                double diff = std::abs(
                    PetscRealPart(v_flux_conc) + PetscRealPart(v_conc_flux));
                if (diff > err) err = diff;
            }
        }
        std::cout << "  组分 0 max|W_flux_conc + W_conc_flux| = "
                  << std::scientific << std::setprecision(3) << err;
        if (err > 1e-10) {
            std::cout << " ✗\n";
            saddle_ok = false;
        } else {
            std::cout << " ✓\n";
        }
        if (err > max_saddle_err) max_saddle_err = err;
    }

    // ========== 测试 4: 浓度块 H_t 对称性 ==========
    std::cout << "\n========== 测试 4: 浓度块 H_t 对称性 ==========\n";
    bool conc_sym_ok = true;
    for (int c = 0; c < n_comp; ++c) {
        int conc_start = total_flux + c * conc_per_comp;
        int conc_end = conc_start + conc_per_comp;
        double max_diff = 0.0;
        for (int i = conc_start; i < conc_end; ++i) {
            for (int j = i + 1; j < conc_end; ++j) {
                PetscScalar v_ij, v_ji;
                MatGetValues(K, 1, &i, 1, &j, &v_ij);
                MatGetValues(K, 1, &j, 1, &i, &v_ji);
                double diff = std::abs(PetscRealPart(v_ij - v_ji));
                if (diff > max_diff) max_diff = diff;
            }
        }
        std::cout << "  组分 " << c << " 浓度块 max|H_ij - H_ji| = "
                  << std::scientific << std::setprecision(3) << max_diff;
        if (max_diff > 1e-10) {
            std::cout << " ✗\n";
            conc_sym_ok = false;
        } else {
            std::cout << " ✓\n";
        }
    }

    // ========== 测试 5: 再次组装结果一致（幂等性）==========
    std::cout << "\n========== 测试 5: 重复组装一致性 ==========\n";
    // 保存当前矩阵向量的范数
    PetscReal fnorm_before = fnorm_K;
    PetscReal fnorm_F_before = norm_F;

    ierr = ms_solver.assembleStiffnessMatrix(true, true);
    if (ierr != 0) {
        std::cerr << "✗ 第二次组装返回错误码\n";
        PetscFinalize();
        return 1;
    }

    PetscReal fnorm_after;
    MatNorm(K, NORM_FROBENIUS, &fnorm_after);
    PetscReal norm_F_after;
    VecNorm(F, NORM_2, &norm_F_after);

    double k_diff = std::abs(fnorm_before - fnorm_after);
    double f_diff = std::abs(fnorm_F_before - norm_F_after);
    std::cout << "  矩阵 F 范数差: " << std::scientific << std::setprecision(3)
              << k_diff << (k_diff < 1e-12 ? " ✓" : " ✗") << "\n";
    std::cout << "  右端项 2 范数差: " << std::scientific << std::setprecision(3)
              << f_diff << (f_diff < 1e-12 ? " ✓" : " ✗") << "\n";

    // ========== 总结 ==========
    std::cout << "\n========== 测试总结 ==========\n";
    bool all_pass = dim_ok && all_sym && saddle_ok && conc_sym_ok
                    && k_diff < 1e-12 && f_diff < 1e-12;
    std::cout << (all_pass ? "✓ 所有测试通过" : "✗ 存在失败测试") << "\n";

    PetscFinalize();
    return all_pass ? 0 : 1;
}
