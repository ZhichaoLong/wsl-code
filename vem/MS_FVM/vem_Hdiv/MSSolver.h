#ifndef MS_SOLVER_H
#define MS_SOLVER_H

#include "HdivMatrix.h"
#include "../pde_data.h"
#include "MatrixPetsc.h"
#include <petsc.h>
#include <vector>
#include <memory>

using namespace MatrixPetsc;

// ============================================================================
// Maxwell-Stefan多组分扩散方程虚拟元求解器子类
// 继承 HdivMatrix，所有网格/自由度/高斯点/k 等属性全部来自父类，不重复定义
// ============================================================================

class MSSolver : public HdivMatrix {
public:
    // 构造函数
    // 传入 Maxwell-Stefan 数据 + 父类需要的 mesh_reader + 时间相关参数
    MSSolver(const MeshReader& mesh_reader, const MSPdeData& ms_data,
             double delta_t, double final_time, int picard_max_iter, double picard_tol,
             int gauss_point_num = 9, int k = 1);

    // 析构函数
    ~MSSolver();

    // 禁用拷贝（保持不变）
    MSSolver(const MSSolver&) = delete;
    MSSolver& operator=(const MSSolver&) = delete;

    // ========================================================================
    // 主要求解接口
    // ========================================================================
    bool initialize();
    bool solve();
    //组装刚度矩阵
    //这里需要区分第一个时间步和第一个时间步的第一个皮卡迭代步，
    //因为第一个时间步的第一个皮卡迭代步需要使用初始函数计算的非线性系数，而后续的时间步和皮卡迭代步则使用多项式插值计算的非线性系数
    //右端项第一个时间步是初始值这个是不变的
    PetscErrorCode assembleStiffnessMatrix(bool is_initial_step, bool is_initial_first_tstep);

    // ========================================================================
    // 边界条件处理
    // ========================================================================
    /**
     * 应用狄利克雷边界条件（降阶处理）
     * 只对通量自由度应用边界条件，浓度自由度和拉格朗日乘子保持不变
     * 通过提取子矩阵的方式减少自由度
     * @param K_free 输出：自由自由度的刚度矩阵
     * @param F_free 输出：自由自由度的载荷向量
     * @param free_dofs_is 输出：自由自由度的索引集
     * @param bd_val 输出：边界值向量
     * @param total_dof 输出：原总自由度数
     * @return 错误码
     */
    PetscErrorCode applyDirichletBoundaryConditions(
        AutoPetscMat& K_free,
        AutoPetscVec& F_free,
        IS& free_dofs_is,
        AutoPetscVec& bd_val,
        PetscInt& total_dof);

    // ========================================================================
    // 数据访问接口
    // ========================================================================

    // 获取单个单元的总自由度数（通量+浓度，所有组分）
    int getDofPerElementTotal(int mesh_idx) const {
        return dof_per_element_total_[mesh_idx];
    }

    // 获取所有单元的总自由度数向量
    const std::vector<int>& getDofPerElementTotalVec() const {
        return dof_per_element_total_;
    }

    //读取矩阵
    AutoPetscMat getMatrixG_cmin(int mesh_idx);
    AutoPetscMat getMatrixAG_init(int mesh_idx, int i, int j);
    AutoPetscMat getMatrixAG_poly(int mesh_idx, int i, int j, const std::vector<double>& u_ploy_coeff);

    // 公共接口：计算稳定化系数 coeff - 初始时间步
    double getStabilizationCoeffInit(int mesh_idx, int i, int j);

    // 公共接口：计算稳定化系数 coeff - 后续时间步
    double getStabilizationCoeffPoly(int mesh_idx, int i, int j, const std::vector<double>& u_ploy_coeff);

    // 获取系统刚度矩阵（只读）
    const AutoPetscMat& getSystemStiffnessMatrix() const { return system_stiffness_mat_; }
    // 获取系统右端项向量（只读）
    const AutoPetscVec& getSystemRhsVec() const { return system_rhs_vec_; }
    // 获取系统刚度矩阵（可修改）
    AutoPetscMat& getSystemStiffnessMatrix() { return system_stiffness_mat_; }
    // 获取系统右端项向量（可修改）
    AutoPetscVec& getSystemRhsVec() { return system_rhs_vec_; }
    // 获取数值解
    const AutoPetscVec& getSolution() const { return solution_; }

    /**
     * 获取指定单元和指定组分的浓度多项式系数
     *
     * @param element_id 单元编号
     * @param comp_id 组分编号 (0, 1, ..., n_components-1)
     * @param use_prev_time true=使用上一时间步的解(用于右端项计算), false=使用当前迭代解(用于非线性项)
     * @return std::vector<double> 浓度多项式系数数组
     */
    std::vector<double> getElementComponentConcentration(int element_id, int comp_id, bool use_prev_time = false) const;

    /**
     * @brief 获取单个单元的所有组分的浓度多项式系数
     * @param element_id 单元编号
     * @param use_prev_time true=使用上一时间步的解, false=使用当前迭代解
     * @return std::vector<double> 所有组分的浓度多项式系数数组，按组分顺序连续存储
     */
    std::vector<double> getElementComponentConcentration_allcomp(int element_id, bool use_prev_time = false) const;

    // 获取组分数
    int getNumComponents() const { return n_comp_; }

    // 获取所有组分的全局通量自由度总数
    int getTotalDofFluxAll() const { return total_dof_flux_all_; }

    // 获取所有组分的全局浓度自由度总数
    int getTotalDofConcAll() const { return total_dof_conc_all_; }

    // 获取所有组分的全局自由度总数（通量+浓度）
    int getTotalDofAll() const { return total_dof_flux_all_ + total_dof_conc_all_; }

    // 应用边界条件并求解
    PetscErrorCode solveWithDirichletBC();
    PetscErrorCode solveWithDirichletBC_cuda();
    PetscErrorCode solveWithDirichletBC_cuda_schur();
    PetscErrorCode solveWithDirichletBC_schur();
    // 时间推进求解（含皮卡迭代）
    PetscErrorCode solveTimeStepping(std::string folder_name = "solution_poly_disc");

    // 保存解向量到文件
    void saveSolutionToFile(double time, std::string folder_name) const;

    // 计算粗细网格的浓度L2误差（用于估计收敛阶）
    // 返回值：每个组分一个误差值，共n_comp_个
    std::vector<double> computeConcentrationL2ErrorBetweenGrids(
        int coarse_grid_size, int fine_grid_size, double time,
        int gauss_point_num = 9) const;

    // 计算粗细网格的通量L2误差（投影到多项式空间后计算）
    // 返回值：每个组分一个误差值，共n_comp_个
    std::vector<double> computeFluxL2ErrorBetweenGrids(
        int coarse_grid_size, int fine_grid_size, double time,
        int gauss_point_num = 9) const;

private:
    // 辅助函数：获取指定网格大小的MeshData（通过读取msh文件）
    std::shared_ptr<MeshData> getMeshDataForGridSize(int grid_size) const;

    // 辅助结构体：保存HdivMatrix及其依赖的MeshData生命周期
    struct HdivGridData {
        std::shared_ptr<MeshReader> mesh_reader;
        std::shared_ptr<HdivMatrix> hdiv_mat;
    };

    // 辅助函数：为指定网格创建独立的HdivMatrix并初始化
    HdivGridData createHdivMatrixForGrid(int grid_size) const;

    // 辅助函数：提取指定网格和组分的通量多项式系数（投影到多项式空间）
    std::vector<std::vector<double>> extractFluxPolynomialCoeffs(
        const HdivGridData& grid_data, const std::vector<double>& sol, int comp_idx) const;
    // ========================================================================
    // 只保留子类特有成员变量
    // 所有网格/自由度/k/高斯点 全部继承自 HdivMatrix，不重复写！
    // ========================================================================
    //求解的矩阵和向量函数
    AutoPetscMat system_stiffness_mat_;// PETSc刚度矩阵（智能指针管理)
    AutoPetscVec system_rhs_vec_;    // PETSc右端项向量（智能指针管理)

    // Maxwell-Stefan 方程物理数据
    const MSPdeData* ms_data_;

    // 组分数
    int n_comp_;

    // 每个单元的总自由度数（单个组分自由度 * 组分数）
    std::vector<int> dof_per_element_total_;

    // 所有组分的全局自由度总数
    int total_dof_flux_all_;   // 所有组分的全局通量自由度
    int total_dof_conc_all_;   // 所有组分的全局浓度自由度

    //一些矩阵元素的生成函数
    //计算矩阵G_cmin的内部元素，基函数与基函数的内积
    double vemHdiv_gm_gm_ploy_cmin(int mesh_idx, int ploy_mk1, int ploy_mk2);

    // 初始时间步：计算 A_ij * G 的单个元素
    // 使用 init_A_coeff(i,j,x,y) 计算 A_ij 在高斯点上的值并积分
    double vemHdiv_gm_gm_ploy_AG_init(int mesh_idx, int i, int j, int ploy_mk1, int ploy_mk2);

    // 后续时间步：计算 A_ij * G 的单个元素
    // 使用 poly_A_coeff(i,j,u_ploy_coeff,xD_x,xD_y,hD,x,y) 计算 A_ij 在高斯点上的值并积分
    double vemHdiv_gm_gm_ploy_AG_poly(int mesh_idx, int i, int j, int ploy_mk1, int ploy_mk2,
                                       const std::vector<double>& u_ploy_coeff);
    
    // 计算稳定化系数 coeff 的辅助函数 - 初始时间步
    double computeStabilizationCoeffInit(int mesh_idx, int i, int j);

    // 计算稳定化系数 coeff 的辅助函数 - 后续时间步
    double computeStabilizationCoeffPoly(int mesh_idx, int i, int j, const std::vector<double>& u_ploy_coeff);

    //计算右端项的单个积分值（初始步用初值函数，后续用上一时间步解）
    double vemHdiv_uh_mk_F(bool is_initial_first_tstep, int mesh_idx, int comp_idx, int ploy_mk);

    //一些矩阵的生成
    AutoPetscMat HdivMatrixG_cmin(int mesh_idx);

    // 初始时间步：A_ij * G 矩阵生成
    AutoPetscMat HdivMatrixAG_init(int mesh_idx, int i, int j);

    // 后续时间步：A_ij * G 矩阵生成
    AutoPetscMat HdivMatrixAG_poly(int mesh_idx, int i, int j, const std::vector<double>& u_ploy_coeff);
    //这个函数可以混用，因为只涉及矩阵直接的乘法，不涉及A_ij的计算方式
    AutoPetscMat HdivMatrixK_ac(const AutoPetscMat& G,const AutoPetscMat& L2Proj_ploy);
    //稳定化矩阵
    AutoPetscMat HdivMatrixK_as(const AutoPetscMat& L2Proj_basis, double coeff = sqrt(2.0));
    //右端项向量
    AutoPetscVec HdivVecF(int mesh_idx, bool is_initial_first_tstep);

    


    // 时间相关参数
    double delta_t_;               // 时间步长
    double final_time_;            // 终止时间
    int picard_iter_count_;        // 皮卡迭代当前次数
    int picard_max_iter_;          // 皮卡迭代最大步数
    double picard_tol_;            // 皮卡迭代收敛容差

    // 状态标志
    bool initialized_;
    bool solved_;

    // 数值解（完整自由度）
    AutoPetscVec solution_;

    // 上一个时间步的数值解（用于右端项计算，在皮卡迭代中保持不变）
    AutoPetscVec solution_prev_time_;

    // ========================================================================
    // 辅助函数
    // ========================================================================
    // 计算单元自由度方向调整向量并应用到矩阵（1或-1，用于统一边法向方向）
    std::vector<double> computeAndApplyDirectionAdjustment(int elem, Mat mat) const;
    std::vector<PetscInt> computeGlobalDofIndices(int elem) const;
    // 将局部刚度矩阵组装到全局刚度矩阵
    void assembleLocalToGlobal(int elem, const AutoPetscMat& K_elem);
    // 将局部右端项组装到全局右端项
    void assembleLocalRhsToGlobal(int elem, const AutoPetscVec& F_elem);

};

#endif // MS_SOLVER_H
