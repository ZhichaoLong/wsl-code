#ifndef MAXWELL_STEFAN_TPFA_SOLVER_H
#define MAXWELL_STEFAN_TPFA_SOLVER_H

#include <petsc.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <memory>  // 智能指针头文件
#include <stdexcept>//错误处理
#include <string>
#include "MeshReader.h"
#include "triangle_mesher.h"
#include "pde_data.h"
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


class MaxwellStefanTPFASolver{
public:
    // 构造函数：绑定网格/方程数据，初始化自由度
    MaxwellStefanTPFASolver(const MeshReader& mesh_reader, const MSPdeData* ms_data, int gauss_point_num = 3)
        : mesh_reader_(&mesh_reader),mesh_data_(&mesh_reader_->getMeshData()), ms_data_(ms_data),
          dof_avg_(0),dof_flux_(0), // 每个单元/边的自由度
          pernon_dof_avg_(0), pernon_dof_flux_(0),   // 全局总自由度
          system_stiffness_mat_(PETSC_NULLPTR),
          system_rhs_vec_(PETSC_NULLPTR) {  // PETSc矩阵和向量初始化为空

        // 空指针检查
        if (mesh_data_ == nullptr) {
            throw std::invalid_argument("Error: MeshData pointer is null!");
        }
        if (ms_data_ == nullptr) {
            throw std::invalid_argument("Error: MSPdeData pointer is null!");
        }

        // 检查PETSc是否初始化
        // 修改第46-48行
        PetscBool is_initialized;
        PetscErrorCode ierr = PetscInitialized(&is_initialized);
        if (ierr != 0 || !is_initialized) {
            throw std::runtime_error("Error: PETSc is not initialized! Call PetscInitialize() first.");
        }


        // 检查组分数合法性
        if (ms_data_->n_components <= 1) {
            throw std::invalid_argument("Error: Number of components must be > 1 (Maxwell-Stefan requires at least 2 components)!");
        }

        // 正确计算自由度
        initDegreesOfFreedom();

        // 初始化PETSc刚度矩阵
        initStiffnessMatrix();

        // 初始化单元平均值向量
        initUValueAvg(gauss_point_num);

        // 初始化右端项向量
        initRightHandSide();
    }

    // 析构函数：智能指针自动释放PETSc矩阵，无需手动销毁
    ~MaxwellStefanTPFASolver() = default;

    // 禁止拷贝构造和赋值（智能指针管理的资源不能简单拷贝）
    MaxwellStefanTPFASolver(const MaxwellStefanTPFASolver&) = delete;
    MaxwellStefanTPFASolver& operator=(const MaxwellStefanTPFASolver&) = delete;

    // 允许移动构造和赋值
    MaxwellStefanTPFASolver(MaxwellStefanTPFASolver&&) = default;
    MaxwellStefanTPFASolver& operator=(MaxwellStefanTPFASolver&&) = default;

    // 核心接口：创建/组装刚度矩阵（成员函数，而非变量）
    PetscErrorCode assembleStiffnessMatrix(int t_step, double delta_t) {
        // 调用带参版本，传入成员向量（自动匹配const引用）
        return assembleStiffnessMatrix(t_step, delta_t, this->u_avg_);
    }
    PetscErrorCode assembleStiffnessMatrix(int t_step, double delta_t, const std::vector<double>& u_avg);

    // 接口：组装右端项向量（Vec格式）
    PetscErrorCode assembleRightHandSide(int t_step, double delta_t) {
        // 调用带参版本，传入成员向量（自动匹配const引用）
        return assembleRightHandSide(t_step, delta_t, this->u_avg_);
    }
    PetscErrorCode assembleRightHandSide(int t_step, double delta_t, const std::vector<double>& u_prev);

    // 接口：求解线性方程组
    PetscErrorCode solveLinearSystem(Mat A, Vec b, Vec& x, PetscBool* converged);

    // 接口：求解线性方程组（PETSc并行求解器，输入串行矩阵/向量）
    PetscErrorCode solveLinearSystem_MPI(Mat A_seq, Vec b_seq, Vec& x_seq, PetscBool* converged);

    // 接口：皮卡迭代求解非线性方程
    PetscErrorCode solvePicardIteration(int t_step, double delta_t, int max_iter = 100, double tol = 1e-6);

    // 接口：从Vec解向量中提取单元平均值部分
    std::vector<double> extractUValueAvgFromVec(Vec solution) const;

    // 接口：保存解向量到文件
    PetscErrorCode saveSolutionToFile(int t_step, Vec solution, const std::string& directory = "solver_data");

    // 接口：保存最后一个时间步的解向量（用于误差计算）
    PetscErrorCode saveFinalSolutionToFile(Vec solution, const std::string& directory = "error_data",
                                          const std::string& filename = "");

    // 接口：获取刚度矩阵（只读） get返回原始指针，也就是mat对象的地址而不是智能指针对象的地址
    Mat getSystemStiffnessMatrix() const {
        return system_stiffness_mat_ ? *(system_stiffness_mat_.get()) : PETSC_NULLPTR;
    }

    // 接口：获取右端项向量（只读）
    Vec getRightHandSide() const {
        return system_rhs_vec_ ? *(system_rhs_vec_.get()) : PETSC_NULLPTR;
    }

    // 接口：获取当前解向量（只读）
    Vec getCurrentSolution() const {
        return current_solution_ ? *(current_solution_.get()) : PETSC_NULLPTR;
    }
    //     // 假设你之后要这样写：
    // PetscErrorCode ierr;

    // // 错误：PETSc 不接受 unique_ptr
    // // ierr = MatSetValues(system_stiffness_mat_, ...);  

    // // 正确：通过 get() 获取原始指针
    // Mat mat = getSystemStiffnessMatrix();
    // ierr = MatSetValues(mat, ...);  // PETSc 可以接受

    // 接口：获取自由度信息
    int getDofPerNonAvg() const { return pernon_dof_avg_; }
    int getDofPerNonFlux() const { return pernon_dof_flux_; }

    // 接口：获取单元平均值向量（只读）
    const std::vector<double>& getUValueAvg() const { return u_avg_; }

    // 接口：获取单个单元组分的平均值
    double getUValueAvg(int element_idx, int comp_idx) const {
        return u_avg_[comp_idx * mesh_data_->total_elements + element_idx];
    }

private:
    const MeshReader* mesh_reader_; // 指向主函数的MeshReader对象
    const MeshData* mesh_data_;   // 网格数据指针（只读）
    const MSPdeData* ms_data_;    // MS方程数据指针（只读）
    int dof_avg_;              // 所有单元的平均自由度数（根据组分数计算）
    int dof_flux_;             // 所有边的自由度数（根据组分数计算）
    int pernon_dof_avg_;          // 每个组分的单元均值自由度数
    int pernon_dof_flux_;          // 每个组分的通量总自由度数
    AutoPetscMat system_stiffness_mat_;// PETSc刚度矩阵（智能指针管理)
    AutoPetscVec system_rhs_vec_;    // PETSc右端项向量（智能指针管理)
    AutoPetscVec current_solution_;  // 当前时间步的解向量（智能指针管理)
    std::vector<double> u_avg_;    // 单元平均值向量，存储格式：[所有单元组分0, 所有单元组分1, ...]

    double computeUValueAvg(int element_idx, int comp_idx, int gauss_point_num) const; // 计算单个单元组分的平均值

    void initDegreesOfFreedom() ;//初始化自由度计算
    void initStiffnessMatrix();//初始化PETSc刚度矩阵
    void initUValueAvg(int gauss_point_num = 9); // 初始化单元平均值向量
    void initRightHandSide(); // 初始化右端项向量

    // 零通量边界处理方法
    PetscErrorCode applyZeroFluxBoundaryCondition();

    // 物理解修正：修正非物理解（负值）并归一化各组分
    std::vector<double> correctPhysicalSolution(const std::vector<double>& u_avg) const;


};















#endif // MAXWELL_STEFAN_TPFA_SOLVER_H