#ifndef MATRIX_PETSC_H
#define MATRIX_PETSC_H

#include <petsc.h>
#include <petscdevice.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <vector>

// ==========================================
// MatrixPetsc命名空间 - 提供PETSc矩阵计算工具
// ==========================================
namespace MatrixPetsc {

// ==========================================
// 1. PETSc资源自动管理 - 智能指针
// ==========================================

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

// ==========================================
// 2. 辅助函数 - 创建和管理PETSc对象
// ==========================================

/**
 * 创建一个全0的稀疏矩阵（串行）
 * @param rows 行数
 * @param cols 列数
 * @param nnz_per_row 每行预分配的非零元素个数
 * @return 智能指针管理的矩阵
 */
inline AutoPetscMat createSparseMatrix(PetscInt rows, PetscInt cols, PetscInt nnz_per_row) {
    Mat A;
    PetscErrorCode ierr = MatCreate(PETSC_COMM_SELF, &A);
    if (ierr != 0) throw std::runtime_error("Failed to create matrix");

    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, rows, cols);
    if (ierr != 0) { MatDestroy(&A); throw std::runtime_error("Failed to set matrix sizes"); }

    ierr = MatSetType(A, MATSEQAIJ);
    if (ierr != 0) { MatDestroy(&A); throw std::runtime_error("Failed to set matrix type"); }

    ierr = MatSeqAIJSetPreallocation(A, nnz_per_row, nullptr);
    if (ierr != 0) { MatDestroy(&A); throw std::runtime_error("Failed to preallocate matrix"); }

    // ✅ 加上这一行！直接解决错误
    MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

    ierr = MatSetUp(A);
    if (ierr != 0) { MatDestroy(&A); throw std::runtime_error("Failed to set up matrix"); }

    AutoPetscMat mat_ptr(new Mat);
    *mat_ptr = A;
    return mat_ptr;
}

/**
 * 创建一个全0的稠密矩阵（串行，非方阵）
 * @param rows 行数
 * @param cols 列数
 * @return 智能指针管理的矩阵
 */
inline AutoPetscMat createDenseMatrix(PetscInt rows, PetscInt cols) {
    Mat A;
    PetscErrorCode ierr = MatCreateDense(PETSC_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, rows, cols, nullptr, &A);
    if (ierr != 0) throw std::runtime_error("Failed to create dense matrix");
    ierr = MatSetUp(A);
    if (ierr != 0) { MatDestroy(&A); throw std::runtime_error("Failed to set up matrix"); }
    return AutoPetscMat(new Mat(A));
}

/**
 * 创建一个全0的稠密矩阵（串行，方阵）
 * @param n 矩阵大小
 * @return 智能指针管理的矩阵
 */
inline AutoPetscMat createDenseMatrix(PetscInt n) {
    return createDenseMatrix(n, n);
}

/**
 * 创建一个向量
 * @param n 向量大小
 * @return 智能指针管理的向量
 */
inline AutoPetscVec createVector(PetscInt n) {
    Vec v;
    PetscErrorCode ierr = VecCreate(PETSC_COMM_SELF, &v);
    if (ierr != 0) throw std::runtime_error("Failed to create vector");
    ierr = VecSetSizes(v, PETSC_DECIDE, n);
    if (ierr != 0) { VecDestroy(&v); throw std::runtime_error("Failed to set vector size"); }
    ierr = VecSetUp(v);
    if (ierr != 0) { VecDestroy(&v); throw std::runtime_error("Failed to set up vector"); }
    return AutoPetscVec(new Vec(v));
}

/**
 * 从AutoPetscMat获取原始Mat指针
 */
inline Mat getRawMat(const AutoPetscMat& mat) {
    return (mat && *mat) ? *mat : PETSC_NULLPTR;
}

/**
 * 从AutoPetscVec获取原始Vec指针
 */
inline Vec getRawVec(const AutoPetscVec& vec) {
    return (vec && *vec) ? *vec : PETSC_NULLPTR;
}

// ==========================================
// 3. 小规模矩阵求逆 - 高斯消元法
// ==========================================

/**
 * 计算小规模矩阵的逆（使用高斯消元法）
 * 适用于几百维以下的矩阵
 * @param A 输入矩阵
 * @return A的逆矩阵
 */
inline AutoPetscMat inverseMatrix(const AutoPetscMat& A) {
    Mat matA = getRawMat(A);
    if (matA == PETSC_NULLPTR) throw std::invalid_argument("Null matrix");

    PetscInt n;
    MatGetSize(matA, &n, nullptr);

    // 首先把矩阵转换成一个二维数组，方便操作
    std::vector<std::vector<double>> aug(n, std::vector<double>(2 * n, 0.0));

    // 从PETSc矩阵复制到aug
    for (PetscInt i = 0; i < n; ++i) {
        for (PetscInt j = 0; j < n; ++j) {
            PetscScalar val;
            MatGetValues(matA, 1, &i, 1, &j, &val);
            aug[i][j] = val;
        }
        aug[i][n + i] = 1.0;  // 右边放单位矩阵
    }

    // 高斯消元 - 上三角化
    for (PetscInt k = 0; k < n; ++k) {
        // 找主元
        PetscInt pivot = k;
        for (PetscInt i = k + 1; i < n; ++i) {
            if (std::abs(aug[i][k]) > std::abs(aug[pivot][k])) {
                pivot = i;
            }
        }

        // 交换行
        if (pivot != k) {
            std::swap(aug[k], aug[pivot]);
        }

        // 检查主元是否为0
        if (std::abs(aug[k][k]) < 1e-14) {
            throw std::runtime_error("Matrix is singular, cannot invert");
        }

        // 消去
        for (PetscInt i = k + 1; i < n; ++i) {
            double factor = aug[i][k] / aug[k][k];
            for (PetscInt j = k; j < 2 * n; ++j) {
                aug[i][j] -= factor * aug[k][j];
            }
        }
    }

    // 回代
    for (PetscInt k = n - 1; k >= 0; --k) {
        // 主元归一化
        double div = aug[k][k];
        for (PetscInt j = k; j < 2 * n; ++j) {
            aug[k][j] /= div;
        }

        // 消去上面的行
        for (PetscInt i = 0; i < k; ++i) {
            double factor = aug[i][k];
            for (PetscInt j = k; j < 2 * n; ++j) {
                aug[i][j] -= factor * aug[k][j];
            }
        }
    }

    // 构造逆矩阵
    AutoPetscMat Ainv = createDenseMatrix(n);
    Mat matInv = getRawMat(Ainv);

    for (PetscInt i = 0; i < n; ++i) {
        for (PetscInt j = 0; j < n; ++j) {
            PetscScalar val = aug[i][n + j];
            MatSetValues(matInv, 1, &i, 1, &j, &val, INSERT_VALUES);
        }
    }
    MatAssemblyBegin(matInv, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(matInv, MAT_FINAL_ASSEMBLY);

    return Ainv;
}

// ==========================================
// 4. 线性方程组求解
// ==========================================

/**
 * 求解线性方程组 Ax = b
 * 使用PETSc的KSP求解器，配置为GMRES+ILU，适合中小规模问题
 *
 * @param A 系数矩阵
 * @param b 右端项
 * @param x 解（输出参数）
 * @param converged 输出是否收敛
 * @param verbose 是否打印详细信息
 * @return PetscErrorCode
 */
inline PetscErrorCode solveLinearSystem(Mat A, Vec b, Vec x, PetscBool* converged, bool verbose = false) {
    KSP ksp;
    PC pc;
    KSPConvergedReason reason;
    PetscErrorCode ierr;

    if (verbose) {
        PetscInt m, n, nnz;
        MatInfo info;
        MatGetSize(A, &m, &n);
        MatGetInfo(A, MAT_LOCAL, &info);
        nnz = (PetscInt)info.nz_used;
        PetscPrintf(PETSC_COMM_SELF, "  [CPU] 矩阵规模: %d x %d, 非零元: %d\n",
                    (int)m, (int)n, (int)nnz);
    }

    // 创建并配置KSP求解器
    ierr = KSPCreate(PETSC_COMM_SELF, &ksp);
    if (ierr != 0) return ierr;

    ierr = KSPSetOperators(ksp, A, A);  // 设置系数矩阵
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    // 获取预处理对象并设置为无预处理
    ierr = KSPGetPC(ksp, &pc);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    ierr = PCSetType(pc, PCILU);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    // 配置GMRES迭代法
    ierr = KSPSetType(ksp, KSPFGMRES);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    // 设置GMRES参数
    ierr = KSPGMRESSetRestart(ksp, 100);  // 重启次数
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    // 设置收敛条件
    ierr = KSPSetTolerances(ksp,
                            1e-10,    // 相对残差 tolerance
                            1e-50,   // 绝对残差 tolerance
                            PETSC_DEFAULT,  // divergence tolerance
                            100000000);   // 最大迭代次数
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    // 从命令行读取额外参数
    ierr = KSPSetFromOptions(ksp);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    // 求解线性方程组
    if (verbose) {
        PetscLogDouble t_start, t_end;
        PetscTime(&t_start);

        ierr = KSPSolve(ksp, b, x);

        PetscTime(&t_end);
        PetscPrintf(PETSC_COMM_SELF, "  KSPSolve took: %.3f seconds\n", t_end - t_start);
    } else {
        ierr = KSPSolve(ksp, b, x);
    }

    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    // 检查收敛状态
    ierr = KSPGetConvergedReason(ksp, &reason);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    *converged = (reason >= 0) ? PETSC_TRUE : PETSC_FALSE;

    if (*converged && verbose) {
        PetscInt its;
        KSPGetTotalIterations(ksp, &its);
        PetscPrintf(PETSC_COMM_SELF, "  GMRES求解器成功收敛，迭代次数: %d\n", its);
    } else if (!*converged) {
        PetscPrintf(PETSC_COMM_SELF, "  ERROR: GMRES求解器未收敛，原因: %d\n", reason);
    }

    // 清理资源
    KSPDestroy(&ksp);

    return 0;
}

/**
 * 使用智能指针的线性方程组求解
 */
inline PetscErrorCode solveLinearSystem(const AutoPetscMat& A, const AutoPetscVec& b, const AutoPetscVec& x, PetscBool* converged, bool verbose = false) {
    return solveLinearSystem(getRawMat(A), getRawVec(b), getRawVec(x), converged, verbose);
}

/**
 * CPU 版本 FieldSplit/Schur 补块预条件子求解线性方程组
 * 与 solveLinearSystem_cuda_schur 配置完全相同，仅运行在 CPU 上，用于对比加速效果
 *
 * 适用于鞍点矩阵：
 *   M = [ A    B^T ]    A 是 n1 x n1 (velocity/flux 块)
 *       [ B    0   ]    0 是 n2 x n2 (pressure 块, 大小 = n_total - n1)
 *
 * 外层求解器: FGMRES (restart=200)
 * Schur 补分解类型: UPPER
 * Schur 补预条件: SELFP
 * vel 子块预条件: GAMG
 * pres 子块预条件: Jacobi
 *
 * @param A 系数矩阵
 * @param b 右端项
 * @param x 解（输出参数）
 * @param n1 第一个分块（velocity/flux）的行列数
 * @param converged 输出是否收敛
 * @param verbose 是否打印详细信息
 * @return PetscErrorCode
 */
inline PetscErrorCode solveLinearSystem_schur(Mat A, Vec b, Vec x, PetscInt n1,
                                               PetscBool* converged, bool verbose = false) {
    PetscErrorCode ierr;
    KSP ksp = nullptr;
    PC pc = nullptr;
    KSPConvergedReason reason;
    IS vel_is = nullptr;
    IS pre_is = nullptr;

    if (verbose) {
        PetscPrintf(PETSC_COMM_SELF, "  [CPU-Schur] 开始 CPU + FieldSplit/Schur 求解...\n");
    }

    // 获取矩阵总维度，计算第二个块的大小 n2
    PetscInt n_total;
    ierr = MatGetSize(A, &n_total, nullptr); CHKERRQ(ierr);
    PetscInt n2 = n_total - n1;
    if (n2 <= 0) {
        PetscPrintf(PETSC_COMM_SELF, "  [CPU-Schur] ERROR: n1=%d >= n_total=%d，分块尺寸非法\n",
                    (int)n1, (int)n_total);
        return PETSC_ERR_ARG_WRONG;
    }
    if (verbose) {
        PetscPrintf(PETSC_COMM_SELF, "  [CPU-Schur] 矩阵分块: A(%dx%d) | 0(%dx%d)\n",
                    (int)n1, (int)n1, (int)n2, (int)n2);
    }

    // ===== 1. 构造两个 IS（字段索引）=====
    // vel_is: [0, 1, ..., n1-1]
    // pre_is: [n1, n1+1, ..., n_total-1]
    if (verbose) PetscPrintf(PETSC_COMM_SELF, "  [CPU-Schur] 构造字段索引 IS...\n");

    ierr = ISCreateStride(PETSC_COMM_SELF, n1, 0,  1, &vel_is);
    if (ierr) return ierr;
    ierr = ISCreateStride(PETSC_COMM_SELF, n2, n1, 1, &pre_is);
    if (ierr) { ISDestroy(&vel_is); return ierr; }

    // ===== 2. 创建 KSP + 外层求解器 FGMRES =====
    if (verbose) PetscPrintf(PETSC_COMM_SELF, "  [CPU-Schur] 配置外层 FGMRES + FieldSplit/Schur...\n");

    ierr = KSPCreate(PETSC_COMM_SELF, &ksp);
    if (ierr) goto cleanup;

    ierr = KSPSetOperators(ksp, A, A);
    if (ierr) goto cleanup;

    ierr = KSPSetType(ksp, KSPFGMRES);  // FGMRES 允许内部预条件子自身是迭代法
    if (ierr) goto cleanup;

    // 设置 GMRES 重启次数（与 GPU 版本保持一致）
    ierr = KSPGMRESSetRestart(ksp, 200);
    if (ierr) goto cleanup;

    ierr = KSPSetTolerances(ksp, 1e-10, 1e-12, PETSC_DEFAULT, 1000000);
    if (ierr) goto cleanup;

    // ===== 3. 配置 PCFIELDSPLIT + Schur 补 =====
    ierr = KSPGetPC(ksp, &pc);
    if (ierr) goto cleanup;

    ierr = PCSetType(pc, PCFIELDSPLIT);
    if (ierr) goto cleanup;

    // 设置字段：velocity (A 块), pressure (0 块)
    ierr = PCFieldSplitSetIS(pc, "velocity", vel_is);
    if (ierr) goto cleanup;
    ierr = PCFieldSplitSetIS(pc, "pressure", pre_is);
    if (ierr) goto cleanup;

    // Schur 补分解类型 = UPPER:  M ≈ [A B^T; 0 S]
    ierr = PCFieldSplitSetType(pc, PC_COMPOSITE_SCHUR);
    if (ierr) goto cleanup;
    ierr = PCFieldSplitSetSchurFactType(pc, PC_FIELDSPLIT_SCHUR_FACT_UPPER);
    if (ierr) goto cleanup;
    // Schur 预条件矩阵：A11 = 直接用主矩阵的 A11 块作为 Schur 补近似
    ierr = PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_SELFP, nullptr);
    if (ierr) goto cleanup;

    // ===== 4. 必须 SetUp 后才能拿到子 KSP =====
    ierr = KSPSetFromOptions(ksp);
    if (ierr) goto cleanup;
    ierr = KSPSetUp(ksp);
    if (ierr) goto cleanup;

    // ===== 5. 配置两个子块的 KSP/PC =====
    //   vel 子块: preonly + GAMG
    //   pres 子块: preonly + Jacobi
    {
        KSP* subksp;
        PetscInt n_split;
        ierr = PCFieldSplitGetSubKSP(pc, &n_split, &subksp);
        if (ierr) goto cleanup;
        if (n_split != 2) {
            PetscPrintf(PETSC_COMM_SELF, "  [CPU-Schur] ERROR: 期望 2 个子块, 实际 %d\n",
                        (int)n_split);
            PetscFree(subksp);
            ierr = PETSC_ERR_PLIB;
            goto cleanup;
        }

        // 子块 0: velocity (A 块) - GAMG
        PC sub_pc0;
        ierr = KSPSetType(subksp[0], KSPPREONLY);
        if (ierr) { PetscFree(subksp); goto cleanup; }
        ierr = KSPGetPC(subksp[0], &sub_pc0);
        if (ierr) { PetscFree(subksp); goto cleanup; }
        ierr = PCSetType(sub_pc0, PCGAMG);
        if (ierr) { PetscFree(subksp); goto cleanup; }

        // 子块 1: pressure (Schur 补) - GAMG
        PC sub_pc1;
        ierr = KSPSetType(subksp[1], KSPPREONLY);
        if (ierr) { PetscFree(subksp); goto cleanup; }
        ierr = KSPGetPC(subksp[1], &sub_pc1);
        if (ierr) { PetscFree(subksp); goto cleanup; }
        ierr = PCSetType(sub_pc1, PCGAMG);
        if (ierr) { PetscFree(subksp); goto cleanup; }

        PetscFree(subksp);
    }

    // ===== 6. 求解 =====
    {
        PetscLogDouble t_start = 0, t_end = 0;
        if (verbose) PetscTime(&t_start);

        ierr = KSPSolve(ksp, b, x);

        if (verbose) {
            PetscTime(&t_end);
            PetscPrintf(PETSC_COMM_SELF, "  [CPU-Schur] KSPSolve 耗时: %.3f 秒\n",
                        t_end - t_start);
        }
    }
    if (ierr) goto cleanup;

    // ===== 7. 检查收敛 =====
    ierr = KSPGetConvergedReason(ksp, &reason);
    if (ierr) goto cleanup;
    *converged = (reason >= 0) ? PETSC_TRUE : PETSC_FALSE;

    {
        PetscInt its;
        KSPGetIterationNumber(ksp, &its);
        if (*converged) {
            if (verbose) {
                PetscPrintf(PETSC_COMM_SELF,
                            "  [CPU-Schur] 求解成功，外层 FGMRES 迭代次数: %d\n", (int)its);
            }
        } else {
            PetscPrintf(PETSC_COMM_SELF,
                        "  [CPU-Schur] ERROR: 未收敛, reason=%d, 迭代次数=%d\n",
                        (int)reason, (int)its);
        }
    }

cleanup:
    if (ksp)    KSPDestroy(&ksp);
    if (vel_is) ISDestroy(&vel_is);
    if (pre_is) ISDestroy(&pre_is);
    return ierr;
}

/**
 * 使用智能指针的 CPU + FieldSplit/Schur 求解
 */
inline PetscErrorCode solveLinearSystem_schur(const AutoPetscMat& A, const AutoPetscVec& b,
                                               const AutoPetscVec& x, PetscInt n1,
                                               PetscBool* converged, bool verbose = false) {
    return solveLinearSystem_schur(getRawMat(A), getRawVec(b), getRawVec(x),
                                    n1, converged, verbose);
}

/**
 * 求解线性方程组 Ax = b - 使用纯LU分解（直接求解器）
 * 使用PETSc的PCLU直接求解，无迭代过程，适合中小规模问题
 *
 * @param A 系数矩阵
 * @param b 右端项
 * @param x 解（输出参数）
 * @param converged 输出是否收敛（LU分解总是成功除非矩阵奇异）
 * @param verbose 是否打印详细信息
 * @return PetscErrorCode
 */
inline PetscErrorCode solveLinearSystem_LU(Mat A, Vec b, Vec x, PetscBool* converged, bool verbose = false) {
    KSP ksp;
    PC pc;
    KSPConvergedReason reason;
    PetscErrorCode ierr;

    // 创建并配置KSP求解器
    ierr = KSPCreate(PETSC_COMM_SELF, &ksp);
    if (ierr != 0) return ierr;

    ierr = KSPSetOperators(ksp, A, A);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    // 设置为仅预处理（无迭代）
    ierr = KSPSetType(ksp, KSPPREONLY);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    // 获取预处理对象并设置为LU分解
    ierr = KSPGetPC(ksp, &pc);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    ierr = PCSetType(pc, PCLU);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    // 从命令行读取额外参数
    ierr = KSPSetFromOptions(ksp);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    // 求解线性方程组
    if (verbose) {
        PetscLogDouble t_start, t_end;
        PetscTime(&t_start);

        ierr = KSPSolve(ksp, b, x);

        PetscTime(&t_end);
        PetscPrintf(PETSC_COMM_SELF, "  KSPSolve (LU) took: %.3f seconds\n", t_end - t_start);
    } else {
        ierr = KSPSolve(ksp, b, x);
    }

    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    // 检查收敛状态
    ierr = KSPGetConvergedReason(ksp, &reason);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    *converged = (reason >= 0) ? PETSC_TRUE : PETSC_FALSE;

    if (*converged && verbose) {
        PetscPrintf(PETSC_COMM_SELF, "  LU分解求解成功\n");
    } else if (!*converged) {
        PetscPrintf(PETSC_COMM_SELF, "  ERROR: LU分解求解失败，原因: %d\n", reason);
    }

    // 清理资源
    KSPDestroy(&ksp);

    return 0;
}

/**
 * 使用智能指针的LU分解线性方程组求解
 */
inline PetscErrorCode solveLinearSystem_LU(const AutoPetscMat& A, const AutoPetscVec& b, const AutoPetscVec& x, PetscBool* converged, bool verbose = false) {
    return solveLinearSystem_LU(getRawMat(A), getRawVec(b), getRawVec(x), converged, verbose);
}

// ==========================================
// 4.1 GPU加速线性方程组求解
// ==========================================

/**
 * 求解线性方程组 Ax = b - GPU加速版本
 * 输入CPU上的矩阵和向量，在函数内部转换为GPU存储类型，然后用GPU求解
 *
 * @param A CPU上的系数矩阵
 * @param b CPU上的右端项
 * @param x CPU上的解（输出参数）
 * @param converged 输出是否收敛
 * @param verbose 是否打印详细信息
 * @return PetscErrorCode
 */
inline PetscErrorCode solveLinearSystem_cuda(Mat A, Vec b, Vec x, PetscBool* converged, bool verbose = false) {
    PetscErrorCode ierr;
    KSP ksp;
    PC pc;
    KSPConvergedReason reason;
    Mat A_gpu = nullptr;
    Vec b_gpu = nullptr;
    Vec x_gpu = nullptr;

    if (verbose) {
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA] 开始GPU加速求解...\n");
        PetscInt m, n, nnz;
        MatInfo info;
        MatGetSize(A, &m, &n);
        MatGetInfo(A, MAT_LOCAL, &info);
        nnz = (PetscInt)info.nz_used;
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA] 矩阵规模: %d x %d, 非零元: %d\n",
                    (int)m, (int)n, (int)nnz);
    }

#if defined(PETSC_HAVE_CUDA)
    // 1. 将CPU矩阵转换为GPU矩阵
    if (verbose) {
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA] 转换矩阵到GPU...\n");
    }
    ierr = MatDuplicate(A, MAT_COPY_VALUES, &A_gpu);
    if (ierr != 0) return ierr;
    ierr = MatSetType(A_gpu, MATSEQAIJCUSPARSE);
    if (ierr != 0) { MatDestroy(&A_gpu); return ierr; }

    // 2. 将CPU向量转换为GPU向量
    if (verbose) {
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA] 转换右端项向量到GPU...\n");
    }
    ierr = VecDuplicate(b, &b_gpu);
    if (ierr != 0) { MatDestroy(&A_gpu); return ierr; }
    ierr = VecCopy(b, b_gpu);
    if (ierr != 0) { MatDestroy(&A_gpu); VecDestroy(&b_gpu); return ierr; }
    ierr = VecSetType(b_gpu, VECSEQCUDA);
    if (ierr != 0) { MatDestroy(&A_gpu); VecDestroy(&b_gpu); return ierr; }

    // 3. 创建GPU上的解向量
    if (verbose) {
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA] 创建GPU解向量...\n");
    }
    ierr = VecDuplicate(b_gpu, &x_gpu);
    if (ierr != 0) { MatDestroy(&A_gpu); VecDestroy(&b_gpu); return ierr; }
    ierr = VecSetType(x_gpu, VECSEQCUDA);
    if (ierr != 0) { MatDestroy(&A_gpu); VecDestroy(&b_gpu); VecDestroy(&x_gpu); return ierr; }
#else
    // 如果没有CUDA支持，直接在CPU上求解
    if (verbose) {
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA] PETSc没有CUDA支持，使用CPU求解\n");
    }
    return solveLinearSystem(A, b, x, converged, verbose);
#endif

    // 4. 创建并配置KSP求解器
    if (verbose) {
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA] 配置GMRES求解器...\n");
    }
    ierr = KSPCreate(PETSC_COMM_SELF, &ksp);
    if (ierr != 0) { MatDestroy(&A_gpu); VecDestroy(&b_gpu); VecDestroy(&x_gpu); return ierr; }

    ierr = KSPSetOperators(ksp, A_gpu, A_gpu);
    if (ierr != 0) { KSPDestroy(&ksp); MatDestroy(&A_gpu); VecDestroy(&b_gpu); VecDestroy(&x_gpu); return ierr; }

    // 使用无预条件GMRES
    ierr = KSPGetPC(ksp, &pc);
    if (ierr != 0) { KSPDestroy(&ksp); MatDestroy(&A_gpu); VecDestroy(&b_gpu); VecDestroy(&x_gpu); return ierr; }

    ierr = PCSetType(pc, PCBJACOBI);
    if (ierr != 0) { KSPDestroy(&ksp); MatDestroy(&A_gpu); VecDestroy(&b_gpu); VecDestroy(&x_gpu); return ierr; }

    ierr = KSPSetType(ksp, KSPFGMRES);
    if (ierr != 0) { KSPDestroy(&ksp); MatDestroy(&A_gpu); VecDestroy(&b_gpu); VecDestroy(&x_gpu); return ierr; }

    // 设置GMRES重启次数
    ierr = KSPGMRESSetRestart(ksp, 100);
    if (ierr != 0) { KSPDestroy(&ksp); MatDestroy(&A_gpu); VecDestroy(&b_gpu); VecDestroy(&x_gpu); return ierr; }

    // 设置收敛条件
    ierr = KSPSetTolerances(ksp,
                            1e-10,    // 相对残差 tolerance
                            1e-50,   // 绝对残差 tolerance
                            PETSC_DEFAULT,  // divergence tolerance
                            1000000);   // 最大迭代次数
    if (ierr != 0) { KSPDestroy(&ksp); MatDestroy(&A_gpu); VecDestroy(&b_gpu); VecDestroy(&x_gpu); return ierr; }

    ierr = KSPSetFromOptions(ksp);
    if (ierr != 0) { KSPDestroy(&ksp); MatDestroy(&A_gpu); VecDestroy(&b_gpu); VecDestroy(&x_gpu); return ierr; }

    // 5. 求解线性方程组
    PetscLogDouble t_start = 0, t_end = 0;
    if (verbose) {
        PetscTime(&t_start);
    }

    ierr = KSPSolve(ksp, b_gpu, x_gpu);

    if (verbose) {
        PetscTime(&t_end);
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA] GPU求解完成，耗时: %.3f seconds\n", t_end - t_start);
    }

    if (ierr != 0) { KSPDestroy(&ksp); MatDestroy(&A_gpu); VecDestroy(&b_gpu); VecDestroy(&x_gpu); return ierr; }

    // 6. 将解从GPU拷贝回CPU
    if (verbose) {
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA] 将解从GPU拷贝回CPU...\n");
    }
    ierr = VecCopy(x_gpu, x);
    if (ierr != 0) { KSPDestroy(&ksp); MatDestroy(&A_gpu); VecDestroy(&b_gpu); VecDestroy(&x_gpu); return ierr; }

    // 7. 检查收敛状态
    ierr = KSPGetConvergedReason(ksp, &reason);
    if (ierr != 0) { KSPDestroy(&ksp); MatDestroy(&A_gpu); VecDestroy(&b_gpu); VecDestroy(&x_gpu); return ierr; }

    *converged = (reason >= 0) ? PETSC_TRUE : PETSC_FALSE;

    // 不管是否收敛，都输出迭代次数
    PetscInt its;
    KSPGetTotalIterations(ksp, &its);
    if (*converged) {
        if (verbose) {
            PetscPrintf(PETSC_COMM_SELF, "  [CUDA] GPU GMRES求解成功，迭代次数: %d\n", its);
        }
    } else {
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA] ERROR: GPU GMRES求解未收敛，原因: %d，迭代次数: %d\n", reason, its);
    }

    // 8. 清理资源
    KSPDestroy(&ksp);
    MatDestroy(&A_gpu);
    VecDestroy(&b_gpu);
    VecDestroy(&x_gpu);

    return 0;
}

/**
 * 使用智能指针的GPU加速线性方程组求解
 */
inline PetscErrorCode solveLinearSystem_cuda(const AutoPetscMat& A, const AutoPetscVec& b, const AutoPetscVec& x, PetscBool* converged, bool verbose = false) {
    return solveLinearSystem_cuda(getRawMat(A), getRawVec(b), getRawVec(x), converged, verbose);
}

/**
 * GPU加速 + FieldSplit/Schur 补块预条件子求解线性方程组
 *
 * 适用于鞍点矩阵：
 *   M = [ A    B^T ]    A 是 n1 x n1 (velocity/flux 块)
 *       [ B    0   ]    0 是 n2 x n2 (pressure 块, 大小 = n_total - n1)
 *
 * Schur 补分解类型: UPPER  ( M ≈ [A B^T; 0 S] )
 * Schur 补预条件: PCFieldSplitSchurPreType_A11 (用 A11 块自身近似 S)
 * vel 子块预条件: GAMG (代数多重网格)
 * pres 子块预条件: Jacobi
 *
 * @param A 系数矩阵
 * @param b 右端项
 * @param x 解（输出参数）
 * @param n1 第一个分块（velocity/flux）的行列数
 * @param converged 输出是否收敛
 * @param verbose 是否打印详细信息
 * @return PetscErrorCode
 */
inline PetscErrorCode solveLinearSystem_cuda_schur(Mat A, Vec b, Vec x, PetscInt n1,
                                                    PetscBool* converged, bool verbose = false) {
    PetscErrorCode ierr;
    KSP ksp = nullptr;
    PC pc = nullptr;
    KSPConvergedReason reason;
    Mat A_gpu = nullptr;
    Vec b_gpu = nullptr;
    Vec x_gpu = nullptr;
    IS vel_is = nullptr;
    IS pre_is = nullptr;

    if (verbose) {
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA-Schur] 开始 GPU + FieldSplit/Schur 求解...\n");
    }

    // 获取矩阵总维度，计算第二个块的大小 n2
    PetscInt n_total;
    ierr = MatGetSize(A, &n_total, nullptr); CHKERRQ(ierr);
    PetscInt n2 = n_total - n1;
    if (n2 <= 0) {
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA-Schur] ERROR: n1=%d >= n_total=%d，分块尺寸非法\n",
                    (int)n1, (int)n_total);
        return PETSC_ERR_ARG_WRONG;
    }
    if (verbose) {
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA-Schur] 矩阵分块: A(%dx%d) | 0(%dx%d)\n",
                    (int)n1, (int)n1, (int)n2, (int)n2);
    }

#if defined(PETSC_HAVE_CUDA)
    // ===== 1. 矩阵/向量搬到 GPU =====
    if (verbose) PetscPrintf(PETSC_COMM_SELF, "  [CUDA-Schur] 转换矩阵/向量到 GPU...\n");

    ierr = MatDuplicate(A, MAT_COPY_VALUES, &A_gpu);
    if (ierr) return ierr;
    ierr = MatSetType(A_gpu, MATSEQAIJCUSPARSE);
    if (ierr) { MatDestroy(&A_gpu); return ierr; }

    ierr = VecDuplicate(b, &b_gpu);
    if (ierr) { MatDestroy(&A_gpu); return ierr; }
    ierr = VecCopy(b, b_gpu);
    if (ierr) { MatDestroy(&A_gpu); VecDestroy(&b_gpu); return ierr; }
    ierr = VecSetType(b_gpu, VECSEQCUDA);
    if (ierr) { MatDestroy(&A_gpu); VecDestroy(&b_gpu); return ierr; }

    ierr = VecDuplicate(b_gpu, &x_gpu);
    if (ierr) { MatDestroy(&A_gpu); VecDestroy(&b_gpu); return ierr; }
    ierr = VecSetType(x_gpu, VECSEQCUDA);
    if (ierr) { MatDestroy(&A_gpu); VecDestroy(&b_gpu); VecDestroy(&x_gpu); return ierr; }
#else
    if (verbose) PetscPrintf(PETSC_COMM_SELF, "  [CUDA-Schur] 无 CUDA 支持，退回 CPU 求解\n");
    return solveLinearSystem(A, b, x, converged, verbose);
#endif

    // ===== 2. 构造两个 IS（字段索引）=====
    // vel_is: [0, 1, ..., n1-1]
    // pre_is: [n1, n1+1, ..., n_total-1]
    if (verbose) PetscPrintf(PETSC_COMM_SELF, "  [CUDA-Schur] 构造字段索引 IS...\n");

    ierr = ISCreateStride(PETSC_COMM_SELF, n1, 0,  1, &vel_is);
    if (ierr) { MatDestroy(&A_gpu); VecDestroy(&b_gpu); VecDestroy(&x_gpu); return ierr; }
    ierr = ISCreateStride(PETSC_COMM_SELF, n2, n1, 1, &pre_is);
    if (ierr) { MatDestroy(&A_gpu); VecDestroy(&b_gpu); VecDestroy(&x_gpu);
                ISDestroy(&vel_is); return ierr; }

    // ===== 3. 创建 KSP + 外层求解器 FGMRES =====
    if (verbose) PetscPrintf(PETSC_COMM_SELF, "  [CUDA-Schur] 配置外层 FGMRES + FieldSplit/Schur...\n");

    ierr = KSPCreate(PETSC_COMM_SELF, &ksp);
    if (ierr) goto cleanup;

    ierr = KSPSetOperators(ksp, A_gpu, A_gpu);
    if (ierr) goto cleanup;

    ierr = KSPSetType(ksp, KSPFGMRES);  // FGMRES 允许内部预条件子自身是迭代法
    if (ierr) goto cleanup;

    // 设置 GMRES 重启次数（默认 30 太小，每次重启会丢失 Krylov 子空间信息）
    // 200~500 之间是常见选择，权衡内存（每次重启前需保存 m 个 Krylov 向量）
    ierr = KSPGMRESSetRestart(ksp, 200);
    if (ierr) goto cleanup;

    ierr = KSPSetTolerances(ksp, 1e-10, 1e-12, PETSC_DEFAULT, 1000000);
    if (ierr) goto cleanup;

    // ===== 4. 配置 PCFIELDSPLIT + Schur 补 =====
    ierr = KSPGetPC(ksp, &pc);
    if (ierr) goto cleanup;

    ierr = PCSetType(pc, PCFIELDSPLIT);
    if (ierr) goto cleanup;

    // 设置字段：velocity (A 块), pressure (0 块)
    ierr = PCFieldSplitSetIS(pc, "velocity", vel_is);
    if (ierr) goto cleanup;
    ierr = PCFieldSplitSetIS(pc, "pressure", pre_is);
    if (ierr) goto cleanup;

    // Schur 补分解类型 = UPPER:  M ≈ [A B^T; 0 S]
    ierr = PCFieldSplitSetType(pc, PC_COMPOSITE_SCHUR);
    if (ierr) goto cleanup;
    ierr = PCFieldSplitSetSchurFactType(pc, PC_FIELDSPLIT_SCHUR_FACT_UPPER);
    if (ierr) goto cleanup;
    // Schur 预条件矩阵：A11 = 直接用主矩阵的 A11 块作为 Schur 补近似
    ierr = PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_SELFP, nullptr);
    if (ierr) goto cleanup;

    // ===== 5. 必须 SetUp 后才能拿到子 KSP =====
    ierr = KSPSetFromOptions(ksp);
    if (ierr) goto cleanup;
    ierr = KSPSetUp(ksp);
    if (ierr) goto cleanup;

    // ===== 6. 配置两个子块的 KSP/PC =====
    //   vel 子块: preonly + GAMG
    //   pres 子块: preonly + Jacobi
    {
        KSP* subksp;
        PetscInt n_split;
        ierr = PCFieldSplitGetSubKSP(pc, &n_split, &subksp);
        if (ierr) goto cleanup;
        if (n_split != 2) {
            PetscPrintf(PETSC_COMM_SELF, "  [CUDA-Schur] ERROR: 期望 2 个子块, 实际 %d\n",
                        (int)n_split);
            PetscFree(subksp);
            ierr = PETSC_ERR_PLIB;
            goto cleanup;
        }

        // 子块 0: velocity (A 块) - GAMG
        PC sub_pc0;
        ierr = KSPSetType(subksp[0], KSPPREONLY);
        if (ierr) { PetscFree(subksp); goto cleanup; }
        ierr = KSPGetPC(subksp[0], &sub_pc0);
        if (ierr) { PetscFree(subksp); goto cleanup; }
        ierr = PCSetType(sub_pc0, PCGAMG);
        if (ierr) { PetscFree(subksp); goto cleanup; }

        // 子块 1: pressure (Schur 补) - GAMG
        PC sub_pc1;
        ierr = KSPSetType(subksp[1], KSPPREONLY);
        if (ierr) { PetscFree(subksp); goto cleanup; }
        ierr = KSPGetPC(subksp[1], &sub_pc1);
        if (ierr) { PetscFree(subksp); goto cleanup; }
        ierr = PCSetType(sub_pc1, PCGAMG);
        if (ierr) { PetscFree(subksp); goto cleanup; }

        // 注意：PCFieldSplitGetSubKSP 返回的 subksp 数组由调用者负责 PetscFree
        PetscFree(subksp);
    }

    // ===== 7. 求解 =====
    {
        PetscLogDouble t_start = 0, t_end = 0;
        if (verbose) PetscTime(&t_start);

        ierr = KSPSolve(ksp, b_gpu, x_gpu);

        if (verbose) {
            PetscTime(&t_end);
            PetscPrintf(PETSC_COMM_SELF, "  [CUDA-Schur] KSPSolve 耗时: %.3f 秒\n",
                        t_end - t_start);
        }
    }
    if (ierr) goto cleanup;

    // ===== 8. 拷贝解回 CPU =====
    ierr = VecCopy(x_gpu, x);
    if (ierr) goto cleanup;

    // ===== 9. 检查收敛 =====
    ierr = KSPGetConvergedReason(ksp, &reason);
    if (ierr) goto cleanup;
    *converged = (reason >= 0) ? PETSC_TRUE : PETSC_FALSE;

    {
        PetscInt its;
        KSPGetIterationNumber(ksp, &its);
        if (*converged) {
            if (verbose) {
                PetscPrintf(PETSC_COMM_SELF,
                            "  [CUDA-Schur] 求解成功，外层 FGMRES 迭代次数: %d\n", (int)its);
            }
        } else {
            PetscPrintf(PETSC_COMM_SELF,
                        "  [CUDA-Schur] ERROR: 未收敛, reason=%d, 迭代次数=%d\n",
                        (int)reason, (int)its);
        }
    }

cleanup:
    if (ksp)    KSPDestroy(&ksp);
    if (vel_is) ISDestroy(&vel_is);
    if (pre_is) ISDestroy(&pre_is);
    if (A_gpu)  MatDestroy(&A_gpu);
    if (b_gpu)  VecDestroy(&b_gpu);
    if (x_gpu)  VecDestroy(&x_gpu);
    return ierr;
}

/**
 * 使用智能指针的 GPU + FieldSplit/Schur 求解
 */
inline PetscErrorCode solveLinearSystem_cuda_schur(const AutoPetscMat& A, const AutoPetscVec& b,
                                                    const AutoPetscVec& x, PetscInt n1,
                                                    PetscBool* converged, bool verbose = false) {
    return solveLinearSystem_cuda_schur(getRawMat(A), getRawVec(b), getRawVec(x),
                                         n1, converged, verbose);
}

// ==========================================
// 5. 矩阵运算
// ==========================================

/**
 * 矩阵乘法：计算 C = scalar * A * B
 * 使用PETSc的MatMatMult，返回自动管理内存的AutoPetscMat
 *
 * @param A 输入矩阵A
 * @param B 输入矩阵B
 * @param scalar 缩放系数（默认1.0）
 * @return 结果矩阵C = scalar * A * B
 */
inline AutoPetscMat multiplyMatrices(const AutoPetscMat& A, const AutoPetscMat& B, PetscScalar scalar = 1.0) {
    Mat matA = getRawMat(A);
    Mat matB = getRawMat(B);
    if (matA == PETSC_NULLPTR || matB == PETSC_NULLPTR) {
        throw std::invalid_argument("Null matrix in multiplyMatrices");
    }

    Mat matC;
    PetscErrorCode ierr = MatMatMult(matA, matB, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &matC);
    if (ierr != 0) {
        throw std::runtime_error("MatMatMult failed");
    }

    if (scalar != 1.0) {
        ierr = MatScale(matC, scalar);
        if (ierr != 0) {
            MatDestroy(&matC);
            throw std::runtime_error("MatScale failed");
        }
    }

    return AutoPetscMat(new Mat(matC));
}

/**
 * 矩阵乘法（原始Mat指针版本）：计算 C = scalar * A * B
 *
 * @param A 输入矩阵A
 * @param B 输入矩阵B
 * @param scalar 缩放系数（默认1.0）
 * @return 结果矩阵C = scalar * A * B
 */
inline AutoPetscMat multiplyMatrices(Mat A, Mat B, PetscScalar scalar = 1.0) {
    if (A == PETSC_NULLPTR || B == PETSC_NULLPTR) {
        throw std::invalid_argument("Null matrix in multiplyMatrices");
    }

    Mat matC;
    PetscErrorCode ierr = MatMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &matC);
    if (ierr != 0) {
        throw std::runtime_error("MatMatMult failed");
    }

    if (scalar != 1.0) {
        ierr = MatScale(matC, scalar);
        if (ierr != 0) {
            MatDestroy(&matC);
            throw std::runtime_error("MatScale failed");
        }
    }

    return AutoPetscMat(new Mat(matC));
}

/**
 * 矩阵加法：计算 C = scalar * (A + B)
 * 使用PETSc的MatAXPY，返回自动管理内存的AutoPetscMat
 *
 * @param A 输入矩阵A
 * @param B 输入矩阵B
 * @param scalar 缩放系数（默认1.0）
 * @return 结果矩阵C = scalar * (A + B)
 */
inline AutoPetscMat addMatrices(const AutoPetscMat& A, const AutoPetscMat& B, PetscScalar scalar = 1.0) {
    Mat matA = getRawMat(A);
    Mat matB = getRawMat(B);
    if (matA == PETSC_NULLPTR || matB == PETSC_NULLPTR) {
        throw std::invalid_argument("Null matrix in addMatrices");
    }

    Mat matC;
    PetscErrorCode ierr = MatDuplicate(matA, MAT_COPY_VALUES, &matC);
    if (ierr != 0) {
        throw std::runtime_error("MatDuplicate failed in addMatrices");
    }

    ierr = MatAXPY(matC, 1.0, matB, DIFFERENT_NONZERO_PATTERN);
    if (ierr != 0) {
        MatDestroy(&matC);
        throw std::runtime_error("MatAXPY failed in addMatrices");
    }

    if (scalar != 1.0) {
        ierr = MatScale(matC, scalar);
        if (ierr != 0) {
            MatDestroy(&matC);
            throw std::runtime_error("MatScale failed in addMatrices");
        }
    }

    return AutoPetscMat(new Mat(matC));
}

/**
 * 矩阵加法（原始指针版本）：计算 C = scalar * (A + B)
 *
 * @param A 输入矩阵A
 * @param B 输入矩阵B
 * @param scalar 缩放系数（默认1.0）
 * @return 结果矩阵C = scalar * (A + B)
 */
inline AutoPetscMat addMatrices(Mat A, Mat B, PetscScalar scalar = 1.0) {
    if (A == PETSC_NULLPTR || B == PETSC_NULLPTR) {
        throw std::invalid_argument("Null matrix in addMatrices");
    }

    Mat matC;
    PetscErrorCode ierr = MatDuplicate(A, MAT_COPY_VALUES, &matC);
    if (ierr != 0) {
        throw std::runtime_error("MatDuplicate failed in addMatrices");
    }

    ierr = MatAXPY(matC, 1.0, B, DIFFERENT_NONZERO_PATTERN);
    if (ierr != 0) {
        MatDestroy(&matC);
        throw std::runtime_error("MatAXPY failed in addMatrices");
    }

    if (scalar != 1.0) {
        ierr = MatScale(matC, scalar);
        if (ierr != 0) {
            MatDestroy(&matC);
            throw std::runtime_error("MatScale failed in addMatrices");
        }
    }

    return AutoPetscMat(new Mat(matC));
}

/**
 * 矩阵与常数相乘：计算 C = scalar * A
 * 使用PETSc的MatScale，返回自动管理内存的AutoPetscMat
 *
 * @param A 输入矩阵A
 * @param scalar 缩放系数
 * @return 结果矩阵C = scalar * A
 */
inline AutoPetscMat scaleMatrix(const AutoPetscMat& A, PetscScalar scalar) {
    Mat matA = getRawMat(A);
    if (matA == PETSC_NULLPTR) {
        throw std::invalid_argument("Null matrix in scaleMatrix");
    }

    Mat matC;
    PetscErrorCode ierr = MatDuplicate(matA, MAT_COPY_VALUES, &matC);
    if (ierr != 0) {
        throw std::runtime_error("MatDuplicate failed in scaleMatrix");
    }

    ierr = MatScale(matC, scalar);
    if (ierr != 0) {
        MatDestroy(&matC);
        throw std::runtime_error("MatScale failed in scaleMatrix");
    }

    return AutoPetscMat(new Mat(matC));
}

/**
 * 矩阵与常数相乘（原始指针版本）：计算 C = scalar * A
 *
 * @param A 输入矩阵A
 * @param scalar 缩放系数
 * @return 结果矩阵C = scalar * A
 */
inline AutoPetscMat scaleMatrix(Mat A, PetscScalar scalar) {
    if (A == PETSC_NULLPTR) {
        throw std::invalid_argument("Null matrix in scaleMatrix");
    }

    Mat matC;
    PetscErrorCode ierr = MatDuplicate(A, MAT_COPY_VALUES, &matC);
    if (ierr != 0) {
        throw std::runtime_error("MatDuplicate failed in scaleMatrix");
    }

    ierr = MatScale(matC, scalar);
    if (ierr != 0) {
        MatDestroy(&matC);
        throw std::runtime_error("MatScale failed in scaleMatrix");
    }

    return AutoPetscMat(new Mat(matC));
}

/**
 * 将子矩阵 A 填充到大矩阵 M 的指定位置
 *
 * @param M 大矩阵（将被修改）
 * @param A 子矩阵
 * @param row_start 子矩阵在大矩阵中的起始行
 * @param col_start 子矩阵在大矩阵中的起始列
 */
inline void insertSubMatrix(Mat M, const AutoPetscMat& A, PetscInt row_start, PetscInt col_start) {
    Mat subMat = getRawMat(A);
    if (M == PETSC_NULLPTR || subMat == PETSC_NULLPTR) {
        throw std::invalid_argument("Null matrix in insertSubMatrix");
    }

    PetscInt sub_rows, sub_cols;
    MatGetSize(subMat, &sub_rows, &sub_cols);

    for (PetscInt i = 0; i < sub_rows; ++i) {
        for (PetscInt j = 0; j < sub_cols; ++j) {
            PetscScalar val;
            MatGetValues(subMat, 1, &i, 1, &j, &val);
            PetscInt global_row = row_start + i;
            PetscInt global_col = col_start + j;
            MatSetValues(M, 1, &global_row, 1, &global_col, &val, INSERT_VALUES);
        }
    }
}

// ==========================================
// 6. 其他辅助矩阵操作
// ==========================================

/**
 * 打印矩阵（用于调试），带阈值筛选小量
 */
inline void printMatrixWithThreshold(Mat A, const char* name = "Matrix", double threshold = 1e-16) {
    PetscPrintf(PETSC_COMM_SELF, "\n===== %s =====\n", name);

    PetscInt rows, cols;
    MatGetSize(A, &rows, &cols);

    for (PetscInt i = 0; i < rows; ++i) {
        PetscPrintf(PETSC_COMM_SELF, "row %d:", i);
        for (PetscInt j = 0; j < cols; ++j) {
            PetscScalar val;
            MatGetValues(A, 1, &i, 1, &j, &val);
            if (PetscAbsScalar(val) >= threshold) {
                PetscPrintf(PETSC_COMM_SELF, " (%d, %.6g)", j, val);
            }
        }
        PetscPrintf(PETSC_COMM_SELF, "\n");
    }
}

/**
 * 打印矩阵（用于调试）
 */
inline void printMatrix(Mat A, const char* name = "Matrix") {
    printMatrixWithThreshold(A, name, 1e-16);
}

/**
 * 打印智能指针管理的矩阵（用于调试）
 */
inline void printMatrix(const AutoPetscMat& A, const char* name = "Matrix") {
    Mat mat = getRawMat(A);
    if (mat != PETSC_NULLPTR) {
        printMatrix(mat, name);
    } else {
        PetscPrintf(PETSC_COMM_SELF, "\n===== %s =====\n", name);
        PetscPrintf(PETSC_COMM_SELF, "  [Null matrix]\n");
    }
}

/**
 * 打印向量（用于调试）
 */
inline void printVector(Vec v, const char* name = "Vector") {
    PetscPrintf(PETSC_COMM_SELF, "\n===== %s =====\n", name);
    VecView(v, PETSC_VIEWER_STDOUT_SELF);
}

/**
 * 打印智能指针管理的向量（用于调试）
 */
inline void printVector(const AutoPetscVec& v, const char* name = "Vector") {
    Vec vec = getRawVec(v);
    if (vec != PETSC_NULLPTR) {
        printVector(vec, name);
    } else {
        PetscPrintf(PETSC_COMM_SELF, "\n===== %s =====\n", name);
        PetscPrintf(PETSC_COMM_SELF, "  [Null vector]\n");
    }
}

/**
 * 打印智能指针管理的矩阵（用于调试），带阈值筛选小量
 */
inline void printMatrixWithThreshold(const AutoPetscMat& A, const char* name = "Matrix", double threshold = 1e-16) {
    Mat mat = getRawMat(A);
    if (mat != PETSC_NULLPTR) {
        printMatrixWithThreshold(mat, name, threshold);
    } else {
        PetscPrintf(PETSC_COMM_SELF, "\n===== %s =====\n", name);
        PetscPrintf(PETSC_COMM_SELF, "  [Null matrix]\n");
    }
}

}  // namespace MatrixPetsc

#endif  // MATRIX_PETSC_H
