#ifndef CURVEDVEM_MS_SOLVER_H
#define CURVEDVEM_MS_SOLVER_H

#include "HdivMatrix.h"
#include "../examples/ms_problem.h"

#include <petsc.h>

#include <string>
#include <vector>

using AutoPetscVec = vem::AutoPetscVec;

// ============================================================================
// Maxwell-Stefan 多组分扩散方程混合虚拟元求解器
// 继承 HdivMatrix，所有网格/自由度/高斯点/k 等属性全部来自父类
// ============================================================================

class MSSolver : public HdivMatrix {
public:
    // 参数依次为：直边计算网格、MS 算例、时间步长、终止时间、
    // 皮卡迭代最大步数、皮卡迭代收敛容差、
    // 二维三角形积分点数、一维 Gauss-Legendre 积分点数、多项式阶数 k、
    // save_all_steps：true=保存每个时间步, false=只保存最后一步。
    // data_subfolder：数据子目录名，保存路径为 data/ms_data/{data_subfolder}/
    //                 只存末步时为 data/ms_data/{data_subfolder}_final/
    MSSolver(const MeshReader& mesh_reader,
             const MaxwellStefan::MSProblem& problem,
             double delta_t,
             double final_time,
             int picard_max_iter,
             double picard_tol,
             int gauss_point_num = 9,
             int gauss_point_num_1d = 9,
             int k = 1,
             bool save_all_steps = true,
             const std::string& data_subfolder = "solver_data",
             bool use_bdf2 = false);

    ~MSSolver();

    MSSolver(const MSSolver&) = delete;
    MSSolver& operator=(const MSSolver&) = delete;

    // ========================================================================
    // 初始化、全局组装与求解
    // ========================================================================
    bool initialize();

    // 组装刚度矩阵
    // is_initial_step：是否为当前时间步的第一个皮卡迭代步
    // is_initial_first_tstep：是否为第一个时间步（初值模式计算 A_ij）
    PetscErrorCode assembleStiffnessMatrix(bool is_initial_step,
                                           bool is_initial_first_tstep);

    // 组装全局系统、施加边界条件并求解（单步）
    bool solve();

    // 施加边界条件后求解，返回完整解
    PetscErrorCode solveWithDirichletBC();

    // GPU 版本求解（GMRES + BJACOBI on CUDA）
    PetscErrorCode solveWithDirichletBC_GPU();

    // 时间推进求解（含皮卡迭代）
    // use_gpu: true 则使用 GPU 线性求解器
    PetscErrorCode solveTimeStepping(
        const std::string& folder_name = "solution_ms",
        bool use_gpu = false);

    // ========================================================================
    // 单元矩阵接口
    // ========================================================================

    // c* I 加权 Gram 矩阵：c* * G（线性部分，所有组分相同）
    AutoPetscMat getMatrixG_cmin(int mesh_idx);

    // 初值模式：A_ij * G 矩阵（非线性部分，用初值函数计算 A_ij）
    AutoPetscMat getMatrixAG_init(int mesh_idx, int i, int j);

    // 数值解模式：A_ij * G 矩阵（非线性部分，用多项式系数还原浓度计算 A_ij）
    AutoPetscMat getMatrixAG_poly(int mesh_idx, int i, int j,
                                  const std::vector<double>& u_ploy_coeff);

    // 一致性刚度矩阵：Pi_poly^T * (G_cmin + AG) * Pi_poly
    AutoPetscMat getMatrixK_ac(const AutoPetscMat& G,
                               const AutoPetscMat& L2Proj_ploy);

    // 稳定化刚度矩阵：alpha_E * (I - Pi_dof)^T (I - Pi_dof)
    AutoPetscMat getMatrixK_as(const AutoPetscMat& L2Proj_basis,
                               double coeff = 1.4142135623730951);

    // c* 线性项稳定化系数 alpha_E（常数 c* 加权张量的平均 Frobenius 范数）
    double getStabilizationCoeffCmin(int mesh_idx);

    // 初值模式非线性项稳定化系数 alpha_E（用初值函数计算 A_ij）
    double getStabilizationCoeffInit(int mesh_idx, int i, int j);

    // 数值解模式非线性项稳定化系数 alpha_E（用多项式系数还原浓度计算 A_ij）
    double getStabilizationCoeffPoly(int mesh_idx, int i, int j,
                                     const std::vector<double>& u_ploy_coeff);

    // 单元右端项向量
    // is_initial_first_tstep: 第一个时间步用初值函数，后续用上一时间步解
    AutoPetscVec getLocalRhs(int mesh_idx, bool is_initial_first_tstep);

    // 方向调整向量（测试/调试用）
    std::vector<double> getDirectionVector(int elem) const;

    // 局部-全局自由度映射（测试/调试用）
    std::vector<PetscInt> getLocalToGlobalMap(int elem) const;

    // ========================================================================
    // 边界条件
    // ========================================================================

    // 法向通量边界作为 H(div) 通量自由度的本质约束处理
    // time: 当前时间，用于计算时变边界条件
    PetscErrorCode applyNormalFluxBoundaryConditions(
        AutoPetscMat& reduced_matrix,
        AutoPetscVec& reduced_rhs,
        IS& free_dofs,
        AutoPetscVec& boundary_values,
        PetscInt& total_dof,
        double time = 0.0);

    // ========================================================================
    // 数据访问接口
    // ========================================================================

    // 单个单元的总自由度数（通量+浓度，所有组分）
    int getDofPerElementTotal(int mesh_idx) const {
        return dof_per_element_total_[mesh_idx];
    }
    const std::vector<int>& getDofPerElementTotalVec() const {
        return dof_per_element_total_;
    }

    // 组分数
    int getNumComponents() const { return n_comp_; }

    // 全局自由度总数
    int getTotalDofFluxAll() const { return total_dof_flux_all_; }
    int getTotalDofConcAll() const { return total_dof_conc_all_; }
    int getTotalDofAll() const {
        return total_dof_flux_all_ + total_dof_conc_all_;
    }

    // 系统矩阵/右端/解
    const AutoPetscMat& getSystemStiffnessMatrix() const {
        return system_stiffness_mat_;
    }
    AutoPetscMat& getSystemStiffnessMatrix() {
        return system_stiffness_mat_;
    }
    const AutoPetscVec& getSystemRhsVec() const { return system_rhs_vec_; }
    AutoPetscVec& getSystemRhsVec() { return system_rhs_vec_; }
    const AutoPetscVec& getSolution() const { return solution_; }
    const AutoPetscVec& getSolutionPrevTime() const { return solution_prev_time_; }
    const AutoPetscVec& getSolutionPicard() const { return solution_picard_; }

    const MaxwellStefan::MSProblem& getProblem() const { return *problem_; }

    bool isInitialized() const { return initialized_; }
    bool isSolved() const { return solved_; }
    double getCurrentTime() const { return current_time_; }

    // 获取指定单元指定组分的浓度多项式系数
    // use_prev_time: true=上一时间步解, false=当前迭代解
    std::vector<double> getElementComponentConcentration(
        int element_id, int comp_id, bool use_prev_time = false) const;

    // 获取单个单元所有组分的浓度多项式系数（按组分连续存储）
    std::vector<double> getElementComponentConcentration_allcomp(
        int element_id, bool use_prev_time = false) const;

    // 通用版本：从任意指定解向量中提取单组分浓度多项式系数
    std::vector<double> extractElementConcentration(
        int element_id, int comp_id, const AutoPetscVec& sol) const;

    // 通用版本：从任意指定解向量中提取所有组分浓度多项式系数
    std::vector<double> extractElementConcentration_allcomp(
        int element_id, const AutoPetscVec& sol) const;

    // 保存解向量到文件（subfolder 为 data/ms_data/ 下的子目录名）
    void saveSolutionToFile(double time, const std::string& subfolder) const;

    // ========================================================================
    // 后处理与误差
    // ========================================================================

    // 计算浓度 L2 误差（每个组分一个值）
    std::vector<double> computeConcentrationL2Error(int gauss_point_num = 9);
    // 计算通量 L2 误差（每个组分一个值）
    std::vector<double> computeFluxL2Error(int gauss_point_num = 9);

    // ========================================================================
    // 性能计时（诊断用）
    //   启用后 assembleStiffnessMatrix 累计单元循环内各阶段墙钟耗时，
    //   solveTimeStepping 在每个皮卡迭代后打印明细，便于定位瓶颈。
    // ========================================================================
    struct AssemblyTiming {
        double parent_mats;   // getMatrixD/G/W/H/H_star/B_* 及两个 L2 投影
        double h_curved;      // HdivMatrixH_curved（曲边质量矩阵）
        double g_cmin;        // HdivMatrixG_cmin
        double extract_conc;  // 从解向量提取浓度多项式系数
        double ag;            // HdivMatrixAG_init / HdivMatrixAG_poly（9 次）
        double k_ac;          // HdivMatrixK_ac（9 次 + 对角 3 次 cmin）
        double stab;          // 稳定化系数计算 + HdivMatrixK_as
        double insert_local;  // 子块插入 K_local
        double direction;     // 方向调整
        double rhs;           // HdivVecRhs
        double to_global;     // 局部到全局装配
        double global_final;  // 全局 MatAssemblyBegin/End
        double total;         // assembleStiffnessMatrix 全函数
        int    n_calls;       // 累计调用次数

        AssemblyTiming() { clear(); }
        void clear();
        // 把另一份计时结果累加进来（用于跨多次组装汇总）
        void accumulate(const AssemblyTiming& other);
        // tag 用于标识本次打印的场景（如 "时间步1 皮卡1"）
        void print(const std::string& tag) const;
    };

    // 打开/关闭计时。打开时清零累计器，便于按运行统计
    void setTimingEnabled(bool on) {
        timing_enabled_ = on;
        if (on) timing_accum_.clear();
    }
    bool isTimingEnabled() const { return timing_enabled_; }
    // 最近一次 assembleStiffnessMatrix 的计时
    const AssemblyTiming& getAssemblyTiming() const { return timing_; }
    // 自 setTimingEnabled 起所有 assembleStiffnessMatrix 的累计计时
    const AssemblyTiming& getAssemblyTimingAccum() const {
        return timing_accum_;
    }

private:
    AutoPetscMat system_stiffness_mat_;
    AutoPetscVec system_rhs_vec_;
    AutoPetscVec solution_;                 // 最终收敛解（皮卡迭代收敛后）
    AutoPetscVec solution_prev_time_;       // 上一时间步的解 u^{n-1}（右端项用，皮卡迭代内不变）
    AutoPetscVec solution_prev_prev_time_;  // 上两个时间步的解 u^{n-2}（仅 BDF2 使用）
    AutoPetscVec solution_picard_;          // 当前皮卡迭代步的解（每次迭代更新，用于组装 A(u)）

    const MaxwellStefan::MSProblem* problem_;

    // 组分数
    int n_comp_;

    // 每个单元总自由度数（所有组分）
    std::vector<int> dof_per_element_total_;

    // 全局自由度总数（所有组分）
    int total_dof_flux_all_;
    int total_dof_conc_all_;

    // 时间相关参数
    double delta_t_;
    double final_time_;
    double current_time_;     // 当前时刻（时间推进时更新）
    bool use_bdf2_;           // true=BDF2, false=BDF1(向后欧拉)
    int picard_iter_count_;
    int picard_max_iter_;
    double picard_tol_;

    // 状态标志
    bool initialized_;
    bool solved_;

    // 是否保存每个时间步的解（false 则只保存最后一步）
    bool save_all_steps_;

    // 数据子目录名（data/ms_data/ 下的子目录）
    std::string data_subfolder_;

    // 性能计时
    bool timing_enabled_;
    AssemblyTiming timing_;        // 最近一次组装
    AssemblyTiming timing_accum_;  // 所有组装累计

    // 误差
    std::vector<double> l2_error_concentration_;
    std::vector<double> l2_error_flux_;

    // ========================================================================
    // AG 矩阵与稳定化系数
    // ========================================================================

    // 计算 G_cmin 的单个元素
    double vemHdiv_gm_gm_ploy_cmin(int mesh_idx,
                                   int ploy_mk1, int ploy_mk2);

    // 初值模式：AG 矩阵的单个元素
    double vemHdiv_gm_gm_ploy_AG_init(int mesh_idx, int i, int j,
                                      int ploy_mk1, int ploy_mk2);

    // 数值解模式：AG 矩阵的单个元素
    double vemHdiv_gm_gm_ploy_AG_poly(int mesh_idx, int i, int j,
                                      int ploy_mk1, int ploy_mk2,
                                      const std::vector<double>& u_ploy_coeff);

    // c* 线性项稳定化系数
    double computeStabilizationCoeffCmin(int mesh_idx);

    // 初值模式稳定化系数计算
    double computeStabilizationCoeffInit(int mesh_idx, int i, int j);

    // 数值解模式稳定化系数计算
    double computeStabilizationCoeffPoly(int mesh_idx, int i, int j,
                                         const std::vector<double>& u_ploy_coeff);

    AutoPetscMat HdivMatrixG_cmin(int mesh_idx);
    AutoPetscMat HdivMatrixAG_init(int mesh_idx, int i, int j);
    AutoPetscMat HdivMatrixAG_poly(int mesh_idx, int i, int j,
                                   const std::vector<double>& u_ploy_coeff);
    AutoPetscMat HdivMatrixK_ac(const AutoPetscMat& G,
                                const AutoPetscMat& L2Proj_ploy);
    AutoPetscMat HdivMatrixK_as(const AutoPetscMat& L2Proj_basis,
                                double coeff);

    // 曲边质量矩阵（物理域 L2 内积，含 detJ 积分）
    // 仅用于时间项 H_t = H_curved / Δt
    AutoPetscMat HdivMatrixH_curved(int mesh_idx);

    // ========================================================================
    // 右端项
    // ========================================================================

    // 右端项单个积分值
    // is_initial_first_tstep: 第一个时间步用初值函数，后续用上一时间步解
    double vemHdiv_uh_mk_F(bool is_initial_first_tstep,
                           int mesh_idx, int comp_idx, int ploy_mk);

    AutoPetscVec HdivVecRhs(int mesh_idx, bool is_initial_first_tstep);

    // ========================================================================
    // 局部到全局映射与方向统一
    // ========================================================================
    std::vector<double> computeDirectionVector(int elem) const;
    std::vector<double> computeAndApplyDirectionAdjustment(
        int elem, Mat matrix) const;

    std::vector<PetscInt> getLocalToGlobalDofs(int elem) const;

    void assembleLocalToGlobal(int elem, const AutoPetscMat& local_matrix);
    void assembleLocalRhsToGlobal(int elem, const AutoPetscVec& local_rhs);
};

#endif  // CURVEDVEM_MS_SOLVER_H
