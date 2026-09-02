/*
 * petsc_utils.h
 * PETSc 工具模块 - 矩阵/向量管理、运算、求解
 */

#ifndef CURVEDVEM_PETSC_UTILS_H
#define CURVEDVEM_PETSC_UTILS_H

#include <petsc.h>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cmath>

namespace vem {

// ==========================================
// 1. PETSc 智能指针 - 自动管理资源
// ==========================================

// 删除器 - 释放 Mat
// unique_ptr<T, D> 拥有 T*，销毁时完全交给 deleter 处理，
// 所以除了 MatDestroy 释放 PETSc 对象，还要 delete 释放 new 出来的指针外壳。
struct PetscMatDeleter {
    void operator()(Mat* mat) const {
        if (mat != nullptr) {
            if (*mat != PETSC_NULLPTR) {
                MatDestroy(mat);
            }
            delete mat;
        }
    }
};

// 删除器 - 释放 Vec
struct PetscVecDeleter {
    void operator()(Vec* vec) const {
        if (vec != nullptr) {
            if (*vec != PETSC_NULLPTR) {
                VecDestroy(vec);
            }
            delete vec;
        }
    }
};

// 智能指针类型
using AutoPetscMat = std::unique_ptr<Mat, PetscMatDeleter>;
using AutoPetscVec = std::unique_ptr<Vec, PetscVecDeleter>;

// 获取原始指针
inline Mat get_raw(const AutoPetscMat& mat) {
    return (mat && *mat) ? *mat : PETSC_NULLPTR;
}
inline Vec get_raw(const AutoPetscVec& vec) {
    return (vec && *vec) ? *vec : PETSC_NULLPTR;
}

// ==========================================
// 2. 创建矩阵和向量
// ==========================================

// 创建稀疏矩阵 (AIJ)
AutoPetscMat create_sparse_matrix(PetscInt rows, PetscInt cols, PetscInt nnz_per_row = 10);

// 创建稠密矩阵
AutoPetscMat create_dense_matrix(PetscInt rows, PetscInt cols);
AutoPetscMat create_dense_matrix(PetscInt n);  // 方阵

// 创建向量
AutoPetscVec create_vector(PetscInt n);

// ==========================================
// 3. 小规模矩阵求逆 (高斯消元)
// ==========================================
AutoPetscMat inverse_matrix(const AutoPetscMat& A);

// ==========================================
// 4. 线性方程组求解
// ==========================================

// GMRES 迭代求解 (带 ILU 预条件)
PetscErrorCode solve_linear_system(Mat A, Vec b, Vec x, PetscBool* converged, bool verbose = false);
PetscErrorCode solve_linear_system(const AutoPetscMat& A, const AutoPetscVec& b, const AutoPetscVec& x,
                                    PetscBool* converged, bool verbose = false);

// LU 直接求解
PetscErrorCode solve_linear_system_lu(Mat A, Vec b, Vec x, PetscBool* converged, bool verbose = false);
PetscErrorCode solve_linear_system_lu(const AutoPetscMat& A, const AutoPetscVec& b, const AutoPetscVec& x,
                                        PetscBool* converged, bool verbose = false);

// Schur 补分块求解 (用于鞍点矩阵)
PetscErrorCode solve_linear_system_schur(Mat A, Vec b, Vec x, PetscInt n1, PetscBool* converged, bool verbose = false);
PetscErrorCode solve_linear_system_schur(const AutoPetscMat& A, const AutoPetscVec& b, const AutoPetscVec& x,
                                          PetscInt n1, PetscBool* converged, bool verbose = false);

// ==========================================
// 4.1 GPU 加速线性方程组求解 (CUDA)
// ==========================================

// GPU GMRES 迭代求解 (带 BJACOBI 预条件)
// 无 CUDA 支持时自动回退到 CPU 版本
PetscErrorCode solve_linear_system_cuda(Mat A, Vec b, Vec x, PetscBool* converged, bool verbose = false);
PetscErrorCode solve_linear_system_cuda(const AutoPetscMat& A, const AutoPetscVec& b, const AutoPetscVec& x,
                                          PetscBool* converged, bool verbose = false);

// GPU Schur 补分块求解 (用于鞍点矩阵)
// 外层 FGMRES + FieldSplit/Schur + GAMG 子块预条件
// 无 CUDA 支持时自动回退到 CPU 版本
PetscErrorCode solve_linear_system_cuda_schur(Mat A, Vec b, Vec x, PetscInt n1, PetscBool* converged, bool verbose = false);
PetscErrorCode solve_linear_system_cuda_schur(const AutoPetscMat& A, const AutoPetscVec& b, const AutoPetscVec& x,
                                                PetscInt n1, PetscBool* converged, bool verbose = false);

// ==========================================
// 5. 矩阵运算
// ==========================================

// 矩阵乘法: C = scalar * A * B
AutoPetscMat multiply_matrices(const AutoPetscMat& A, const AutoPetscMat& B, PetscScalar scalar = 1.0);
AutoPetscMat multiply_matrices(Mat A, Mat B, PetscScalar scalar = 1.0);

// 矩阵加法: C = scalar * (A + B)
AutoPetscMat add_matrices(const AutoPetscMat& A, const AutoPetscMat& B, PetscScalar scalar = 1.0);
AutoPetscMat add_matrices(Mat A, Mat B, PetscScalar scalar = 1.0);

// 矩阵缩放: C = scalar * A
AutoPetscMat scale_matrix(const AutoPetscMat& A, PetscScalar scalar);
AutoPetscMat scale_matrix(Mat A, PetscScalar scalar);

// 矩阵转置
AutoPetscMat transpose_matrix(const AutoPetscMat& A);
AutoPetscMat transpose_matrix(Mat A);

// ==========================================
// 6. 分块填充
// ==========================================
void insert_submatrix(Mat M, const AutoPetscMat& A, PetscInt row_start, PetscInt col_start);
void insert_submatrix(Mat M, Mat A, PetscInt row_start, PetscInt col_start);

// 将子矩阵 A 累加到稠密矩阵 M 的 (row_start, col_start) 位置（ADD_VALUES）
void add_submatrix(Mat M, const AutoPetscMat& A, PetscInt row_start, PetscInt col_start);
void add_submatrix(Mat M, Mat A, PetscInt row_start, PetscInt col_start);

// ==========================================
// 7. 调试打印
// ==========================================
void print_matrix(Mat A, const char* name = "Matrix");
void print_matrix(const AutoPetscMat& A, const char* name = "Matrix");
void print_matrix_with_threshold(Mat A, const char* name, double threshold = 1e-16);
void print_matrix_with_threshold(const AutoPetscMat& A, const char* name, double threshold = 1e-16);

void print_vector(Vec v, const char* name = "Vector");
void print_vector(const AutoPetscVec& v, const char* name = "Vector");

}  // namespace vem

#endif  // CURVEDVEM_PETSC_UTILS_H
