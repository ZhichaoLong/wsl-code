#ifndef DARCY_SOLVER_H
#define DARCY_SOLVER_H

#include "HdivMatrix.h"
#include "DarcyData.h"
#include "MatrixPetsc.h"
#include <petsc.h>
#include <vector>
#include <memory>

using namespace MatrixPetsc;

// ============================================================================
// Darcy方程虚拟元求解器子类
// 继承 HdivMatrix，所有网格/自由度/高斯点/k 等属性全部来自父类，不重复定义
// ============================================================================

class DarcySolver : public HdivMatrix {
public:
    // 构造函数
    // 只需要传入 Darcy 数据 + 父类需要的 mesh_reader，完全满足你的需求
    DarcySolver(const MeshReader& mesh_reader, const DarcyPdeData& darcy_data, int gauss_point_num = 9, int k = 1);

    // 析构函数
    ~DarcySolver();

    // 禁用拷贝（保持不变）
    DarcySolver(const DarcySolver&) = delete;
    DarcySolver& operator=(const DarcySolver&) = delete;

    // ========================================================================
    // 主要求解接口
    // ========================================================================
    bool initialize();
    bool solve();
    //组装刚度矩阵
    PetscErrorCode assembleStiffnessMatrix();

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
    AutoPetscVec getVecLagrange(int mesh_idx);
    AutoPetscMat getMatrixK_ac(const AutoPetscMat& G,const AutoPetscMat& L2Proj_ploy);
    AutoPetscMat getMatrixK_as(const AutoPetscMat& L2Proj_basis);

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

    // 应用边界条件并求解
    PetscErrorCode solveWithDirichletBC();
    PetscErrorCode solveWithDirichletBC_GPU();

    // 计算压力的L2误差
    double computeL2ErrorPressure(int gauss_point_num = 9);

    // 计算通量的L2误差（暂时注释）
    // double computeL2ErrorFlux(int gauss_point_num = 9);

    // 获取压力误差
    double getL2ErrorPressure() const { return L2_error_pressure_; }

    // 获取通量误差（暂时注释）
    // double getL2ErrorFlux() const { return L2_error_flux_; }




private:
    // ========================================================================
    // ✅ 只保留子类【特有】成员变量
    // 所有网格/自由度/k/高斯点 全部继承自 HdivMatrix，不重复写！
    // ========================================================================
    //求解的矩阵和向量函数，这里包括拉格朗日乘数的数量
    AutoPetscMat system_stiffness_mat_;// PETSc刚度矩阵（智能指针管理)
    AutoPetscVec system_rhs_vec_;    // PETSc右端项向量（智能指针管理)
    // Darcy 方程物理数据
    const DarcyPdeData* darcy_data_;

    // 状态标志
    bool initialized_;
    bool solved_;

    // 数值解（完整自由度）
    AutoPetscVec solution_;

    // L2误差
    double L2_error_pressure_;
    double L2_error_flux_;

    // ========================================================================
    // 辅助函数
    // ========================================================================
    // 计算单元自由度方向调整向量并应用到矩阵（1或-1，用于统一边法向方向）
    std::vector<double> computeAndApplyDirectionAdjustment(int elem, Mat mat) const;
    // 将局部刚度矩阵组装到全局刚度矩阵
    void assembleLocalToGlobal(int elem, const AutoPetscMat& K_elem);
    // 将局部右端项组装到全局右端项
    void assembleLocalRhsToGlobal(int elem, const AutoPetscVec& F_elem);

    //计算单元矩阵，组装刚度矩阵的部分函数

    //计算单项式mk的积分，用于拉格朗日乘数
    double vemHdiv_mk(int mesh_idx, int ploy_mk1);
    double vemHdiv_f_mk(int mesh_idx, int ploy_mk);


    //计算拉格朗日乘数向量，后续操作就是加在最后一行和最后一列
    AutoPetscVec HdivVecLagrange(int mesh_idx);
    //组装K_ac矩阵
    AutoPetscMat HdivMatrixK_ac(const AutoPetscMat& G,const AutoPetscMat& L2Proj_ploy);
    //组装K_as矩阵，这个矩阵是稳定化项
    AutoPetscMat HdivMatrixK_as(const AutoPetscMat& L2Proj_basis);
    //组装右端项F
    AutoPetscVec HdivVecF(int mesh_idx);



};

#endif // DARCY_SOLVER_H