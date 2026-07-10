#ifndef TPFA1D_SOLVER_NEW_H
#define TPFA1D_SOLVER_NEW_H

#include <petsc.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include "pde_data1d.h"
#include "GaussQuadrature.h"

// 自定义删除器：自动释放PETSc的Mat资源
struct PetscMatDeleterNew {
    void operator()(Mat* mat) const {
        if (mat != nullptr && *mat != PETSC_NULLPTR) {
            MatDestroy(mat);
        }
    }
};

// 自定义删除器：自动释放PETSc的Vec资源
struct PetscVecDeleterNew {
    void operator()(Vec* vec) const {
        if (vec != nullptr && *vec != PETSC_NULLPTR) {
            VecDestroy(vec);
        }
    }
};

using AutoPetscMatNew = std::unique_ptr<Mat, PetscMatDeleterNew>;
using AutoPetscVecNew = std::unique_ptr<Vec, PetscVecDeleterNew>;

/**
 * 一维TPFA (Two-Point Flux Approximation) 有限体积法求解器 (New版本)
 * 多组分Maxwell-Stefan扩散方程
 *
 * 简化版本：
 * - 仅使用单元平均值作为未知量（不使用混合形式）
 * - 使用PETSc的矩阵和向量，用智能指针管理
 * - 自由度分布：[组分0单元0, 组分0单元1, ..., 组分0单元N, 组分1单元0, ...]
 */
class TPFA1DSolverNew
{
private:
    // ========== 网格数据 ==========
    double xmin_;                           // 左边界
    double xmax_;                           // 右边界
    int n_cells_;                           // 单元数量
    int n_nodes_;                           // 节点数量
    double dx_;                             // 单元长度（均匀网格）

    // 节点坐标
    std::vector<double> nodes_;
    // 单元中点坐标
    std::vector<double> cell_centers_;
    // 单元长度
    std::vector<double> cell_lengths_;

    // ========== PDE数据 ==========
    const MSPdeData1D* ms_data_;           // MS方程数据指针
    int gauss_point_num_;                   // 高斯积分点数量

    // ========== 自由度数据 ==========
    int n_dofs_;                            // 总自由度数 (n_comp * n_cells)
    int percomp_dofs_;                      // 每个组分的自由度数 (n_cells)

    // ========== PETSc对象（智能指针管理） ==========
    AutoPetscMatNew system_stiffness_mat_; // 刚度矩阵
    AutoPetscVecNew system_rhs_vec_;       // 右端项向量
    AutoPetscVecNew current_solution_;     // 当前解向量

    // ========== 单元平均值向量 ==========
    std::vector<double> u_avg_;             // 当前时间步 t_n 的单元平均值
    std::vector<double> u_prev_avg_;        // 上一个时间步 t_{n-1} 的单元平均值（用于BDF2）

    // ========== 私有初始化函数 ==========
    void generateUniformMesh();              // 生成均匀网格
    void initDegreesOfFreedom();             // 初始化自由度计算
    void initStiffnessMatrix();              // 初始化PETSc刚度矩阵
    void initUValueAvg();                    // 初始化单元平均值向量
    void initRightHandSide();                // 初始化右端项向量

    // ========== 私有辅助函数 ==========
    double computeEdgeUValue(double u_current, double u_neighbor) const;
    std::vector<std::vector<double>> computeDiffusionCoeffMatrix(bool is_initial, int cell_num, int face_num,
                                                                   const std::vector<double>& u_edge_value_flux_allcomp) const;
    std::vector<std::vector<double>> invertMatrix(const std::vector<std::vector<double>>& mat) const;
    std::vector<double> extractUValueAvgFromVec(Vec solution) const;
    std::vector<double> correctPhysicalSolution(const std::vector<double>& u_avg) const;

public:
    // 构造函数：传入网格范围、网格数、PDE数据和高斯积分点数量
    TPFA1DSolverNew(double xmin, double xmax, int n_cells, const MSPdeData1D* ms_data, int gauss_point_num = 9);

    // 析构函数
    ~TPFA1DSolverNew() = default;

    // 禁止拷贝构造和赋值
    TPFA1DSolverNew(const TPFA1DSolverNew&) = delete;
    TPFA1DSolverNew& operator=(const TPFA1DSolverNew&) = delete;

    // 允许移动构造和赋值
    TPFA1DSolverNew(TPFA1DSolverNew&&) = default;
    TPFA1DSolverNew& operator=(TPFA1DSolverNew&&) = default;

    // ========== 核心接口 ==========

    // 接口：组装刚度矩阵（BDF1/向后欧拉）
    PetscErrorCode assembleStiffnessMatrix(int t_step, double delta_t, bool is_initial);
    PetscErrorCode assembleStiffnessMatrix(int t_step, double delta_t, const std::vector<double>& u_avg, bool is_initial);

    // 接口：组装右端项向量（BDF1/向后欧拉）
    PetscErrorCode assembleRightHandSide(int t_step, double delta_t);
    PetscErrorCode assembleRightHandSide(int t_step, double delta_t, const std::vector<double>& u_prev);

    // 接口：组装刚度矩阵（BDF2 二阶后向差分公式）
    PetscErrorCode assembleStiffnessMatrix_BDF2(int t_step, double delta_t);
    PetscErrorCode assembleStiffnessMatrix_BDF2(int t_step, double delta_t,
                                                  const std::vector<double>& u_prev1,
                                                  const std::vector<double>& u_prev2);

    // 接口：组装右端项向量（BDF2 二阶后向差分公式）
    PetscErrorCode assembleRightHandSide_BDF2(int t_step, double delta_t);
    PetscErrorCode assembleRightHandSide_BDF2(int t_step, double delta_t,
                                               const std::vector<double>& u_prev1,
                                               const std::vector<double>& u_prev2);

    // 接口：求解线性方程组（LU直接求解）
    PetscErrorCode solveLinearSystem(Mat A, Vec b, Vec& x, PetscBool* converged);

    // 接口：求解线性方程组（AMG预处理的GMRES迭代求解）
    PetscErrorCode solveLinearSystem_AMG(Mat A, Vec b, Vec& x, PetscBool* converged);

    // 接口：求解线性方程组（PETSc并行求解器，输入串行矩阵/向量）
    PetscErrorCode solveLinearSystem_MPI(Mat A_seq, Vec b_seq, Vec& x_seq, PetscBool* converged);

    // 接口：皮卡迭代求解非线性方程（BDF1/向后欧拉）
    PetscErrorCode solvePicardIteration(int t_step, double delta_t, int max_iter = 100, double tol = 1e-6);

    // 接口：皮卡迭代求解非线性方程（BDF2 二阶后向差分公式）
    PetscErrorCode solvePicardIteration_BDF2(int t_step, double delta_t, int max_iter = 100, double tol = 1e-6);

    // 接口：皮卡迭代求解非线性方程（MPI并行版本，BDF1/向后欧拉）
    PetscErrorCode solvePicardIteration_MPI(int t_step, double delta_t, int max_iter = 100, double tol = 1e-6);

    // 接口：更新历史时间步解（时间步推进后调用）
    void updateHistorySolutions();

    // 接口：保存解向量到文件
    PetscErrorCode saveSolutionToFile(int t_step, Vec solution, const std::string& directory = "solver_data");

    // 接口：保存最后一个时间步的解向量
    PetscErrorCode saveFinalSolutionToFile(Vec solution, const std::string& directory = "error_data",
                                           const std::string& filename = "");

    // ========== Getter接口 ==========

    Mat getSystemStiffnessMatrix() const {
        return system_stiffness_mat_ ? *(system_stiffness_mat_.get()) : PETSC_NULLPTR;
    }

    Vec getRightHandSide() const {
        return system_rhs_vec_ ? *(system_rhs_vec_.get()) : PETSC_NULLPTR;
    }

    Vec getCurrentSolution() const {
        return current_solution_ ? *(current_solution_.get()) : PETSC_NULLPTR;
    }

    const std::vector<double>& getUValueAvg() const { return u_avg_; }

    const std::vector<double>& getPrevUValueAvg() const { return u_prev_avg_; }

    double getUValueAvg(int cell_idx, int comp_idx) const {
        return u_avg_[comp_idx * n_cells_ + cell_idx];
    }

    int getNumCells() const { return n_cells_; }

    int getNumNodes() const { return n_nodes_; }

    double getXMin() const { return xmin_; }

    double getXMax() const { return xmax_; }

    int getDofsPerComponent() const { return percomp_dofs_; }

    int getTotalDofs() const { return n_dofs_; }

    // 辅助接口：获取自由度索引
    inline int getDofIndex(int comp, int cell) const {
        return comp * percomp_dofs_ + cell;
    }
};

#endif // TPFA1D_SOLVER_NEW_H
