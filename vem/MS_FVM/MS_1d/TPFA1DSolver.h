#ifndef TPFA1D_SOLVER_H
#define TPFA1D_SOLVER_H

#include <petsc.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include "pde_data1d.h"
#include "GaussQuadrature.h"

// 自定义删除器：自动释放PETSc的Mat资源，避免内存泄漏
struct PetscMatDeleter {
    void operator()(Mat* mat) const {
        if (mat != nullptr && *mat != PETSC_NULLPTR) {
            MatDestroy(mat);
        }
    }
};

// 自定义删除器：自动释放PETSc的Vec资源，避免内存泄漏
struct PetscVecDeleter {
    void operator()(Vec* vec) const {
        if (vec != nullptr && *vec != PETSC_NULLPTR) {
            VecDestroy(vec);
        }
    }
};

// 使用智能指针管理PETSc矩阵，自动释放资源
using AutoPetscMat = std::unique_ptr<Mat, PetscMatDeleter>;

// 使用智能指针管理PETSc向量，自动释放资源
using AutoPetscVec = std::unique_ptr<Vec, PetscVecDeleter>;

/**
 * 一维TPFA (Two-Point Flux Approximation) 有限体积法求解器
 * 多组分Maxwell-Stefan扩散方程
 */
class TPFA1DSolver
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
    int dof_avg_;                           // 所有单元的平均自由度数
    int dof_flux_;                          // 所有面的通量自由度数
    int pernon_dof_avg_;                    // 每个组分的单元均值自由度数
    int pernon_dof_flux_;                   // 每个组分的通量自由度数
    int n_faces_;                           // 面的数量（单元数+1）

    // ========== PETSc对象（智能指针管理） ==========
    AutoPetscMat system_stiffness_mat_;    // 刚度矩阵
    AutoPetscVec system_rhs_vec_;          // 右端项向量
    AutoPetscVec current_solution_;        // 当前解向量

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
    double computeUValueAvg(int cell_idx, int comp_idx) const;
    std::vector<double> correctPhysicalSolution(const std::vector<double>& u_avg) const;
    PetscErrorCode applyBoundaryConditions();

public:
    // 构造函数：传入网格范围、网格数、PDE数据和高斯积分点数量
    TPFA1DSolver(double xmin, double xmax, int n_cells, const MSPdeData1D* ms_data, int gauss_point_num = 9);

    // 析构函数
    ~TPFA1DSolver() = default;

    // 禁止拷贝构造和赋值
    TPFA1DSolver(const TPFA1DSolver&) = delete;
    TPFA1DSolver& operator=(const TPFA1DSolver&) = delete;

    // 允许移动构造和赋值
    TPFA1DSolver(TPFA1DSolver&&) = default;
    TPFA1DSolver& operator=(TPFA1DSolver&&) = default;

    // ========== 核心接口 ==========

    // 接口：组装刚度矩阵（BDF1/向后欧拉）
    PetscErrorCode assembleStiffnessMatrix(int t_step, double delta_t) {
        return assembleStiffnessMatrix(t_step, delta_t, this->u_avg_);
    }
    PetscErrorCode assembleStiffnessMatrix(int t_step, double delta_t, const std::vector<double>& u_avg);

    // 接口：组装右端项向量（BDF1/向后欧拉）
    PetscErrorCode assembleRightHandSide(int t_step, double delta_t) {
        return assembleRightHandSide(t_step, delta_t, this->u_avg_);
    }
    PetscErrorCode assembleRightHandSide(int t_step, double delta_t, const std::vector<double>& u_prev);

    // 接口：组装刚度矩阵（BDF2 二阶后向差分公式）
    // BDF2 需要前两个时间步的解 u_prev1 (t_{n}) 和 u_prev2 (t_{n-1})
    PetscErrorCode assembleStiffnessMatrix_BDF2(int t_step, double delta_t) {
        return assembleStiffnessMatrix_BDF2(t_step, delta_t, this->u_avg_, this->u_prev_avg_);
    }
    PetscErrorCode assembleStiffnessMatrix_BDF2(int t_step, double delta_t,
                                                   const std::vector<double>& u_prev1,
                                                   const std::vector<double>& u_prev2);

    // 接口：组装右端项向量（BDF2 二阶后向差分公式）
    // BDF2 需要前两个时间步的解 u_prev1 (t_{n}) 和 u_prev2 (t_{n-1})
    PetscErrorCode assembleRightHandSide_BDF2(int t_step, double delta_t) {
        return assembleRightHandSide_BDF2(t_step, delta_t, this->u_avg_, this->u_prev_avg_);
    }
    PetscErrorCode assembleRightHandSide_BDF2(int t_step, double delta_t,
                                                const std::vector<double>& u_prev1,
                                                const std::vector<double>& u_prev2);

    // 接口：求解线性方程组
    PetscErrorCode solveLinearSystem(Mat A, Vec b, Vec& x, PetscBool* converged);

    // 接口：皮卡迭代求解非线性方程（BDF1/向后欧拉）
    PetscErrorCode solvePicardIteration(int t_step, double delta_t, int max_iter = 100, double tol = 1e-6);

    // 接口：皮卡迭代求解非线性方程（BDF2 二阶后向差分公式）
    // 注意：BDF2 需要前两个时间步的历史解，第一个时间步 (t_step=0) 应用 BDF1
    PetscErrorCode solvePicardIteration_BDF2(int t_step, double delta_t, int max_iter = 100, double tol = 1e-6);

    // 接口：更新历史时间步解（时间步推进后调用，将 u_avg_ 存入 u_prev_avg_）
    void updateHistorySolutions();

    // 接口：从Vec解向量中提取单元平均值部分
    std::vector<double> extractUValueAvgFromVec(Vec solution) const;

    // 接口：保存解向量到文件
    PetscErrorCode saveSolutionToFile(int t_step, Vec solution, const std::string& directory = "solver_data");

    // 接口：保存最后一个时间步的解向量
    PetscErrorCode saveFinalSolutionToFile(Vec solution, const std::string& directory = "error_data",
                                          const std::string& filename = "");

    // ========== Getter接口 ==========

    // 获取刚度矩阵
    Mat getSystemStiffnessMatrix() const {
        return system_stiffness_mat_ ? *(system_stiffness_mat_.get()) : PETSC_NULLPTR;
    }

    // 获取右端项向量
    Vec getRightHandSide() const {
        return system_rhs_vec_ ? *(system_rhs_vec_.get()) : PETSC_NULLPTR;
    }

    // 获取当前解向量
    Vec getCurrentSolution() const {
        return current_solution_ ? *(current_solution_.get()) : PETSC_NULLPTR;
    }

    // 获取当前时间步单元平均值向量
    const std::vector<double>& getUValueAvg() const { return u_avg_; }

    // 获取上一个时间步单元平均值向量（用于BDF2）
    const std::vector<double>& getPrevUValueAvg() const { return u_prev_avg_; }

    // 获取单个单元组分的平均值
    double getUValueAvg(int cell_idx, int comp_idx) const {
        return u_avg_[comp_idx * n_cells_ + cell_idx];
    }

    // 获取单元数量
    int getNumCells() const { return n_cells_; }

    // 获取节点数量
    int getNumNodes() const { return n_nodes_; }

    // 获取网格左边界
    double getXMin() const { return xmin_; }

    // 获取网格右边界
    double getXMax() const { return xmax_; }

    // 获取单元总数自由度
    int getDofPerNonAvg() const { return pernon_dof_avg_; }
};

#endif // TPFA1D_SOLVER_H
