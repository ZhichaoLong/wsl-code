#ifndef CURVEDVEM_DARCY_SOLVER_H
#define CURVEDVEM_DARCY_SOLVER_H

#include "HdivMatrix.h"
#include "../examples/darcy_problem.h"

#include <petsc.h>

#include <vector>

using AutoPetscVec = vem::AutoPetscVec;

// Darcy 方程混合虚拟元求解器。
//
// 本类只声明 Darcy 专用局部矩阵、全局组装与求解接口。
// 所有方法的具体实现将在确认相应数学公式后逐步完成。
class DarcySolver : public HdivMatrix {
public:
    // 参数依次为：直边计算网格、Darcy 算例、二维三角形积分点数、
    // 一维 Gauss-Legendre 积分点数、多项式阶数 k。
    DarcySolver(const MeshReader& mesh_reader,
                const Darcy::DarcyProblem& problem,
                int gauss_point_num = 9,
                int gauss_point_num_1d = 9,
                int k = 1);
    ~DarcySolver();

    DarcySolver(const DarcySolver&) = delete;
    DarcySolver& operator=(const DarcySolver&) = delete;

    // ========================================================================
    // 初始化、全局组装与求解
    // ========================================================================
    bool initialize();
    // 完成所有单元局部鞍点矩阵和局部右端的全局累加。
    PetscErrorCode assembleStiffnessMatrix();
    // 组装全局系统、施加法向通量强制边界并求解。
    bool solve();
    // 对已组装系统执行边界降维，使用 GMRES+ILU 求解并恢复完整解。
    PetscErrorCode solveWithDirichletBC();

    // 组装全局系统、施加法向通量强制边界并求解（GPU 版本）。
    bool solveGPU();
    // 对已组装系统执行边界降维，使用 GPU GMRES 求解并恢复完整解。
    PetscErrorCode solveWithDirichletBC_GPU();

    // ========================================================================
    // Darcy 专用单元矩阵接口
    // ========================================================================

    // 系数加权 Gram 矩阵：
    // (G_Kappa)_{alpha,beta}
    //   = integral_E [(K_hat^{-1} m_beta) dot m_alpha] dx_hat。
    AutoPetscMat getMatrixG_Kappa(int mesh_idx);

    // 一致性刚度：K_ac = Pi_poly^T G_Kappa Pi_poly。
    AutoPetscMat getMatrixK_ac(
        const AutoPetscMat& G_Kappa,
        const AutoPetscMat& L2Proj_ploy);

    // 稳定化刚度：
    // K_as = alpha_E (I - Pi_dof)^T (I - Pi_dof)，其中 alpha_E 是
    // 单元平均张量 average(K_hat^{-1}) 的 Frobenius（矩阵 L2）模。
    AutoPetscMat getMatrixK_as(
        int mesh_idx,
        const AutoPetscMat& L2Proj_basis);

    // 单元右端项。速度块暂为零；压力块为
    // -integral_E g_hat m_r，其中 g_hat=det(J)*(g o F)；末尾约束项为零。
    AutoPetscVec getLocalRhs(int mesh_idx);
    // 压力零积分约束向量。压力分量为
    // integral_Ehat m_r det(J) dx_hat；速度分量和末尾乘子分量为零。
    AutoPetscVec getVecLagrange(int mesh_idx);

    // ========================================================================
    // 边界条件
    // ========================================================================

    // 法向通量边界作为 H(div) 速度自由度的本质约束处理。
    PetscErrorCode applyNormalFluxBoundaryConditions(
        AutoPetscMat& reduced_matrix,
        AutoPetscVec& reduced_rhs,
        IS& free_dofs,
        AutoPetscVec& boundary_values,
        PetscInt& total_dof);

    // 压力 Dirichlet 数据通过速度方程边界泛函进入右端项。
    PetscErrorCode assemblePressureBoundaryRhs();

    // ========================================================================
    // 系统矩阵、右端项与解
    // ========================================================================
    const AutoPetscMat& getSystemStiffnessMatrix() const {
        return system_stiffness_mat_;
    }
    AutoPetscMat& getSystemStiffnessMatrix() {
        return system_stiffness_mat_;
    }

    const AutoPetscVec& getSystemRhsVec() const {
        return system_rhs_vec_;
    }
    AutoPetscVec& getSystemRhsVec() {
        return system_rhs_vec_;
    }

    const AutoPetscVec& getSolution() const {
        return solution_;
    }

    const Darcy::DarcyProblem& getProblem() const {
        return *problem_;
    }

    bool isInitialized() const { return initialized_; }
    bool isSolved() const { return solved_; }

    // ========================================================================
    // 后处理与误差
    // ========================================================================
    // 直边计算区域上的 L2 误差：使用映射后的解析解，不额外乘物理 Jacobian。
    double computeL2ErrorPressure(int gauss_point_num = 9);
    double computeL2ErrorFlux(int gauss_point_num = 9);

    double getL2ErrorPressure() const { return l2_error_pressure_; }
    double getL2ErrorFlux() const { return l2_error_flux_; }

private:
    AutoPetscMat system_stiffness_mat_;
    AutoPetscVec system_rhs_vec_;
    AutoPetscVec solution_;

    const Darcy::DarcyProblem* problem_;

    bool initialized_;
    bool solved_;

    double l2_error_pressure_;
    double l2_error_flux_;

    // ========================================================================
    // G_Kappa 与局部速度矩阵
    // ========================================================================

    // 计算 G_Kappa 的单个元素。
    double vemHdiv_gm_gm_Kappa(
        int mesh_idx,
        int ploy_mk1,
        int ploy_mk2);

    AutoPetscMat HdivMatrixG_Kappa(int mesh_idx);
    AutoPetscMat HdivMatrixK_ac(
        const AutoPetscMat& G_Kappa,
        const AutoPetscMat& L2Proj_ploy);
    AutoPetscMat HdivMatrixK_as(
        int mesh_idx,
        const AutoPetscMat& L2Proj_basis);

    // 计算单元稳定化尺度 alpha_E。
    double computeStabilizationScale(int mesh_idx);

    // ========================================================================
    // 耦合块、约束与右端项
    // ========================================================================
    double vemHdiv_mk(int mesh_idx, int ploy_mk);
    double vemHdiv_source_mk(int mesh_idx, int ploy_mk);

    AutoPetscVec HdivVecLagrange(int mesh_idx);
    AutoPetscVec HdivVecRhs(int mesh_idx);

    // ========================================================================
    // 局部到全局映射与方向统一
    // ========================================================================
    std::vector<double> computeAndApplyDirectionAdjustment(
        int elem,
        Mat matrix) const;

    std::vector<PetscInt> getLocalToGlobalDofs(int elem) const;

    void assembleLocalToGlobal(
        int elem,
        const AutoPetscMat& local_matrix);
    void assembleLocalRhsToGlobal(
        int elem,
        const AutoPetscVec& local_rhs);
};

#endif  // CURVEDVEM_DARCY_SOLVER_H
