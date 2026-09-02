/*
 * petsc_utils.cpp
 * PETSc 工具模块实现
 */

#include "petsc_utils.h"

namespace vem {

// ==========================================
// 2. 创建矩阵和向量
// ==========================================

AutoPetscMat create_sparse_matrix(PetscInt rows, PetscInt cols, PetscInt nnz_per_row) {
    Mat A;
    PetscErrorCode ierr = MatCreate(PETSC_COMM_SELF, &A);
    if (ierr != 0) throw std::runtime_error("MatCreate failed");

    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, rows, cols);
    if (ierr != 0) { MatDestroy(&A); throw std::runtime_error("MatSetSizes failed"); }

    ierr = MatSetType(A, MATSEQAIJ);
    if (ierr != 0) { MatDestroy(&A); throw std::runtime_error("MatSetType failed"); }

    ierr = MatSeqAIJSetPreallocation(A, nnz_per_row, nullptr);
    if (ierr != 0) { MatDestroy(&A); throw std::runtime_error("MatSeqAIJSetPreallocation failed"); }

    MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

    ierr = MatSetUp(A);
    if (ierr != 0) { MatDestroy(&A); throw std::runtime_error("MatSetUp failed"); }

    AutoPetscMat mat_ptr(new Mat(A));
    return mat_ptr;
}

AutoPetscMat create_dense_matrix(PetscInt rows, PetscInt cols) {
    Mat A;
    PetscErrorCode ierr = MatCreateDense(PETSC_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, rows, cols, nullptr, &A);
    if (ierr != 0) throw std::runtime_error("MatCreateDense failed");
    ierr = MatSetUp(A);
    if (ierr != 0) { MatDestroy(&A); throw std::runtime_error("MatSetUp failed"); }
    return AutoPetscMat(new Mat(A));
}

AutoPetscMat create_dense_matrix(PetscInt n) {
    return create_dense_matrix(n, n);
}

AutoPetscVec create_vector(PetscInt n) {
    Vec v;
    PetscErrorCode ierr = VecCreate(PETSC_COMM_SELF, &v);
    if (ierr != 0) throw std::runtime_error("VecCreate failed");
    ierr = VecSetSizes(v, PETSC_DECIDE, n);
    if (ierr != 0) { VecDestroy(&v); throw std::runtime_error("VecSetSizes failed"); }
    ierr = VecSetUp(v);
    if (ierr != 0) { VecDestroy(&v); throw std::runtime_error("VecSetUp failed"); }
    return AutoPetscVec(new Vec(v));
}

// ==========================================
// 3. 小规模矩阵求逆 (高斯消元)
// ==========================================

AutoPetscMat inverse_matrix(const AutoPetscMat& A) {
    Mat matA = get_raw(A);
    if (matA == PETSC_NULLPTR) throw std::invalid_argument("Null matrix");

    PetscInt n;
    MatGetSize(matA, &n, nullptr);

    // 拷贝矩阵到二维数组
    std::vector<std::vector<double>> aug(n, std::vector<double>(2 * n, 0.0));
    for (PetscInt i = 0; i < n; ++i) {
        for (PetscInt j = 0; j < n; ++j) {
            PetscScalar val;
            MatGetValues(matA, 1, &i, 1, &j, &val);
            aug[i][j] = val;
        }
        aug[i][n + i] = 1.0;  // 右侧放单位阵
    }

    // 前向消去 (上三角化)
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

        if (std::abs(aug[k][k]) < 1e-14) {
            throw std::runtime_error("Matrix is singular, cannot invert");
        }

        // 消去下方各行
        for (PetscInt i = k + 1; i < n; ++i) {
            double factor = aug[i][k] / aug[k][k];
            for (PetscInt j = k; j < 2 * n; ++j) {
                aug[i][j] -= factor * aug[k][j];
            }
        }
    }

    // 回代
    for (PetscInt k = n - 1; k >= 0; --k) {
        double div = aug[k][k];
        for (PetscInt j = k; j < 2 * n; ++j) {
            aug[k][j] /= div;
        }
        for (PetscInt i = 0; i < k; ++i) {
            double factor = aug[i][k];
            for (PetscInt j = k; j < 2 * n; ++j) {
                aug[i][j] -= factor * aug[k][j];
            }
        }
    }

    // 构造逆矩阵
    AutoPetscMat Ainv = create_dense_matrix(n);
    Mat matInv = get_raw(Ainv);
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

PetscErrorCode solve_linear_system(Mat A, Vec b, Vec x, PetscBool* converged, bool verbose) {
    KSP ksp;
    PC pc;
    KSPConvergedReason reason;
    PetscErrorCode ierr;

    if (verbose) {
        PetscInt m, n;
        MatInfo info;
        MatGetSize(A, &m, &n);
        MatGetInfo(A, MAT_LOCAL, &info);
        PetscPrintf(PETSC_COMM_SELF, "  Solving system: %d x %d\n", (int)m, (int)n);
    }

    ierr = KSPCreate(PETSC_COMM_SELF, &ksp);
    if (ierr != 0) return ierr;

    ierr = KSPSetOperators(ksp, A, A);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    ierr = KSPGetPC(ksp, &pc);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    ierr = PCSetType(pc, PCILU);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    ierr = KSPSetType(ksp, KSPGMRES);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    ierr = KSPGMRESSetRestart(ksp, 100);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    ierr = KSPSetTolerances(ksp, 1e-10, 1e-50, PETSC_DEFAULT, 1000000);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    ierr = KSPSetFromOptions(ksp);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    // 求解
    if (verbose) {
        PetscLogDouble t_start, t_end;
        PetscTime(&t_start);
        ierr = KSPSolve(ksp, b, x);
        PetscTime(&t_end);
        PetscPrintf(PETSC_COMM_SELF, "  Solve time: %.3f sec\n", t_end - t_start);
    } else {
        ierr = KSPSolve(ksp, b, x);
    }

    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    // 检查收敛
    ierr = KSPGetConvergedReason(ksp, &reason);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    *converged = (reason >= 0) ? PETSC_TRUE : PETSC_FALSE;

    if (*converged && verbose) {
        PetscInt its;
        KSPGetTotalIterations(ksp, &its);
        PetscPrintf(PETSC_COMM_SELF, "  Converged in %d iterations\n", (int)its);
    } else if (!*converged) {
        PetscPrintf(PETSC_COMM_SELF, "  Warning: not converged, reason = %d\n", (int)reason);
    }

    KSPDestroy(&ksp);
    return 0;
}

PetscErrorCode solve_linear_system(const AutoPetscMat& A, const AutoPetscVec& b, const AutoPetscVec& x,
                                    PetscBool* converged, bool verbose) {
    return solve_linear_system(get_raw(A), get_raw(b), get_raw(x), converged, verbose);
}

PetscErrorCode solve_linear_system_lu(Mat A, Vec b, Vec x, PetscBool* converged, bool verbose) {
    KSP ksp;
    PC pc;
    KSPConvergedReason reason;
    PetscErrorCode ierr;

    ierr = KSPCreate(PETSC_COMM_SELF, &ksp);
    if (ierr != 0) return ierr;

    ierr = KSPSetOperators(ksp, A, A);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    ierr = KSPSetType(ksp, KSPPREONLY);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    ierr = KSPGetPC(ksp, &pc);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    ierr = PCSetType(pc, PCLU);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    ierr = KSPSetFromOptions(ksp);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    if (verbose) {
        PetscLogDouble t_start, t_end;
        PetscTime(&t_start);
        ierr = KSPSolve(ksp, b, x);
        PetscTime(&t_end);
        PetscPrintf(PETSC_COMM_SELF, "  LU solve time: %.3f sec\n", t_end - t_start);
    } else {
        ierr = KSPSolve(ksp, b, x);
    }

    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    ierr = KSPGetConvergedReason(ksp, &reason);
    if (ierr != 0) { KSPDestroy(&ksp); return ierr; }

    *converged = (reason >= 0) ? PETSC_TRUE : PETSC_FALSE;

    if (*converged && verbose) {
        PetscPrintf(PETSC_COMM_SELF, "  LU solve succeeded\n");
    }

    KSPDestroy(&ksp);
    return 0;
}

PetscErrorCode solve_linear_system_lu(const AutoPetscMat& A, const AutoPetscVec& b, const AutoPetscVec& x,
                                        PetscBool* converged, bool verbose) {
    return solve_linear_system_lu(get_raw(A), get_raw(b), get_raw(x), converged, verbose);
}

PetscErrorCode solve_linear_system_schur(Mat A, Vec b, Vec x, PetscInt n1, PetscBool* converged, bool verbose) {
    PetscErrorCode ierr;
    KSP ksp = nullptr;
    PC pc = nullptr;
    KSPConvergedReason reason;
    IS vel_is = nullptr;
    IS pre_is = nullptr;

    if (verbose) {
        PetscPrintf(PETSC_COMM_SELF, "  Solving saddle point system with Schur complement\n");
    }

    PetscInt n_total;
    ierr = MatGetSize(A, &n_total, nullptr); CHKERRQ(ierr);
    PetscInt n2 = n_total - n1;
    if (n2 <= 0) {
        PetscPrintf(PETSC_COMM_SELF, "  Error: invalid split n1=%d >= n_total=%d\n", (int)n1, (int)n_total);
        return PETSC_ERR_ARG_WRONG;
    }

    if (verbose) {
        PetscPrintf(PETSC_COMM_SELF, "  Block sizes: vel=%d, pres=%d\n", (int)n1, (int)n2);
    }

    ierr = ISCreateStride(PETSC_COMM_SELF, n1, 0, 1, &vel_is);
    if (ierr) return ierr;
    ierr = ISCreateStride(PETSC_COMM_SELF, n2, n1, 1, &pre_is);
    if (ierr) { ISDestroy(&vel_is); return ierr; }

    ierr = KSPCreate(PETSC_COMM_SELF, &ksp);
    if (ierr) goto cleanup;

    ierr = KSPSetOperators(ksp, A, A);
    if (ierr) goto cleanup;

    ierr = KSPSetType(ksp, KSPFGMRES);
    if (ierr) goto cleanup;

    ierr = KSPGMRESSetRestart(ksp, 200);
    if (ierr) goto cleanup;

    ierr = KSPSetTolerances(ksp, 1e-10, 1e-12, PETSC_DEFAULT, 1000000);
    if (ierr) goto cleanup;

    ierr = KSPGetPC(ksp, &pc);
    if (ierr) goto cleanup;

    ierr = PCSetType(pc, PCFIELDSPLIT);
    if (ierr) goto cleanup;

    ierr = PCFieldSplitSetIS(pc, "velocity", vel_is);
    if (ierr) goto cleanup;
    ierr = PCFieldSplitSetIS(pc, "pressure", pre_is);
    if (ierr) goto cleanup;

    ierr = PCFieldSplitSetType(pc, PC_COMPOSITE_SCHUR);
    if (ierr) goto cleanup;
    ierr = PCFieldSplitSetSchurFactType(pc, PC_FIELDSPLIT_SCHUR_FACT_UPPER);
    if (ierr) goto cleanup;
    ierr = PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_SELFP, nullptr);
    if (ierr) goto cleanup;

    ierr = KSPSetFromOptions(ksp);
    if (ierr) goto cleanup;
    ierr = KSPSetUp(ksp);
    if (ierr) goto cleanup;

    // 子块配置
    {
        KSP* subksp;
        PetscInt n_split;
        ierr = PCFieldSplitGetSubKSP(pc, &n_split, &subksp);
        if (ierr) goto cleanup;

        if (n_split == 2) {
            PC sub_pc0, sub_pc1;
            ierr = KSPSetType(subksp[0], KSPPREONLY);
            if (!ierr) ierr = KSPGetPC(subksp[0], &sub_pc0);
            if (!ierr) ierr = PCSetType(sub_pc0, PCGAMG);

            ierr = KSPSetType(subksp[1], KSPPREONLY);
            if (!ierr) ierr = KSPGetPC(subksp[1], &sub_pc1);
            if (!ierr) ierr = PCSetType(sub_pc1, PCGAMG);
        }

        PetscFree(subksp);
    }

    // 求解
    if (verbose) {
        PetscLogDouble t_start, t_end;
        PetscTime(&t_start);
        ierr = KSPSolve(ksp, b, x);
        PetscTime(&t_end);
        PetscPrintf(PETSC_COMM_SELF, "  Schur solve time: %.3f sec\n", t_end - t_start);
    } else {
        ierr = KSPSolve(ksp, b, x);
    }

    if (ierr) goto cleanup;

    ierr = KSPGetConvergedReason(ksp, &reason);
    if (ierr) goto cleanup;
    *converged = (reason >= 0) ? PETSC_TRUE : PETSC_FALSE;

    if (*converged && verbose) {
        PetscInt its;
        KSPGetIterationNumber(ksp, &its);
        PetscPrintf(PETSC_COMM_SELF, "  Converged in %d iterations\n", (int)its);
    }

cleanup:
    if (ksp) KSPDestroy(&ksp);
    if (vel_is) ISDestroy(&vel_is);
    if (pre_is) ISDestroy(&pre_is);
    return ierr;
}

PetscErrorCode solve_linear_system_schur(const AutoPetscMat& A, const AutoPetscVec& b, const AutoPetscVec& x,
                                          PetscInt n1, PetscBool* converged, bool verbose) {
    return solve_linear_system_schur(get_raw(A), get_raw(b), get_raw(x), n1, converged, verbose);
}

// ==========================================
// 4.1 GPU 加速线性方程组求解 (CUDA)
// ==========================================

PetscErrorCode solve_linear_system_cuda(Mat A, Vec b, Vec x, PetscBool* converged, bool verbose) {
    PetscErrorCode ierr;
    KSP ksp = nullptr;
    PC pc = nullptr;
    KSPConvergedReason reason;
    Mat A_gpu = nullptr;
    Vec b_gpu = nullptr;
    Vec x_gpu = nullptr;
    PetscLogDouble t_start = 0, t_end = 0;
    PetscInt its = 0;

    if (verbose) {
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA] Starting GPU-accelerated solve...\n");
        PetscInt m, n;
        MatInfo info;
        MatGetSize(A, &m, &n);
        MatGetInfo(A, MAT_LOCAL, &info);
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA] System size: %d x %d, nnz = %d\n",
                    (int)m, (int)n, (int)info.nz_used);
    }

#if defined(PETSC_HAVE_CUDA)
    // 1. 矩阵搬到 GPU
    if (verbose) PetscPrintf(PETSC_COMM_SELF, "  [CUDA] Moving matrix to GPU...\n");
    ierr = MatDuplicate(A, MAT_COPY_VALUES, &A_gpu);
    if (ierr) return ierr;
    ierr = MatSetType(A_gpu, MATSEQAIJCUSPARSE);
    if (ierr) { MatDestroy(&A_gpu); return ierr; }

    // 2. 右端项搬到 GPU
    if (verbose) PetscPrintf(PETSC_COMM_SELF, "  [CUDA] Moving RHS vector to GPU...\n");
    ierr = VecDuplicate(b, &b_gpu);
    if (ierr) { MatDestroy(&A_gpu); return ierr; }
    ierr = VecCopy(b, b_gpu);
    if (ierr) { MatDestroy(&A_gpu); VecDestroy(&b_gpu); return ierr; }
    ierr = VecSetType(b_gpu, VECSEQCUDA);
    if (ierr) { MatDestroy(&A_gpu); VecDestroy(&b_gpu); return ierr; }

    // 3. 创建 GPU 解向量
    if (verbose) PetscPrintf(PETSC_COMM_SELF, "  [CUDA] Creating GPU solution vector...\n");
    ierr = VecDuplicate(b_gpu, &x_gpu);
    if (ierr) { MatDestroy(&A_gpu); VecDestroy(&b_gpu); return ierr; }
    ierr = VecSetType(x_gpu, VECSEQCUDA);
    if (ierr) { MatDestroy(&A_gpu); VecDestroy(&b_gpu); VecDestroy(&x_gpu); return ierr; }
#else
    // 无 CUDA 支持，回退到 CPU 求解
    if (verbose) {
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA] PETSc not built with CUDA, falling back to CPU solve\n");
    }
    return solve_linear_system(A, b, x, converged, verbose);
#endif

    // 4. 配置 KSP 求解器 (FGMRES + BJACOBI)
    if (verbose) PetscPrintf(PETSC_COMM_SELF, "  [CUDA] Configuring FGMRES solver...\n");
    ierr = KSPCreate(PETSC_COMM_SELF, &ksp);
    if (ierr) goto cuda_cleanup;

    ierr = KSPSetOperators(ksp, A_gpu, A_gpu);
    if (ierr) goto cuda_cleanup;

    ierr = KSPGetPC(ksp, &pc);
    if (ierr) goto cuda_cleanup;

    // Block Jacobi 预条件适合 GPU
    ierr = PCSetType(pc, PCBJACOBI);
    if (ierr) goto cuda_cleanup;

    ierr = KSPSetType(ksp, KSPFGMRES);
    if (ierr) goto cuda_cleanup;

    // restart=50：FGMRES 要存 V(m+1) + Z(m) 约 2m+1 个基向量，
    // 128x128 网格（N=489,984，3.92 MB/向量）下 m=100 要 788 MB、m=50 只要 396 MB，
    // 且每次求解都重新分配释放一遍。实测收敛是「慢但稳」（rtol 1e-10 用 54 个重启周期，
    // 每周期降 0.65），这种情形下加大 m 换不回等比例的迭代数下降，
    // 总正交化开销 ∝ n_iters × m/2 反而变差，故取 50 而非更大。
    // 可用命令行覆盖：-ksp_gmres_restart <m>（KSPSetFromOptions 在本行之后调用）
    ierr = KSPGMRESSetRestart(ksp, 50);
    if (ierr) goto cuda_cleanup;

    ierr = KSPSetTolerances(ksp, 1e-10, 1e-50, PETSC_DEFAULT, 1000000);
    if (ierr) goto cuda_cleanup;

    ierr = KSPSetFromOptions(ksp);
    if (ierr) goto cuda_cleanup;

    // 5. 求解
    if (verbose) PetscTime(&t_start);

    ierr = KSPSolve(ksp, b_gpu, x_gpu);

    if (verbose) {
        PetscTime(&t_end);
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA] KSPSolve time: %.3f sec\n", t_end - t_start);
    }
    if (ierr) goto cuda_cleanup;

    // 6. 解拷回 CPU
    if (verbose) PetscPrintf(PETSC_COMM_SELF, "  [CUDA] Copying solution back to CPU...\n");
    ierr = VecCopy(x_gpu, x);
    if (ierr) goto cuda_cleanup;

    // 7. 检查收敛
    ierr = KSPGetConvergedReason(ksp, &reason);
    if (ierr) goto cuda_cleanup;

    *converged = (reason >= 0) ? PETSC_TRUE : PETSC_FALSE;

    KSPGetTotalIterations(ksp, &its);
    if (*converged) {
        if (verbose) {
            PetscPrintf(PETSC_COMM_SELF, "  [CUDA] Converged in %d iterations\n", (int)its);
        }
    } else {
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA] WARNING: not converged, reason = %d, iterations = %d\n",
                    (int)reason, (int)its);
    }

cuda_cleanup:
    if (ksp)   KSPDestroy(&ksp);
    if (A_gpu) MatDestroy(&A_gpu);
    if (b_gpu) VecDestroy(&b_gpu);
    if (x_gpu) VecDestroy(&x_gpu);
    return ierr;
}

PetscErrorCode solve_linear_system_cuda(const AutoPetscMat& A, const AutoPetscVec& b, const AutoPetscVec& x,
                                          PetscBool* converged, bool verbose) {
    return solve_linear_system_cuda(get_raw(A), get_raw(b), get_raw(x), converged, verbose);
}

// ---- GPU Schur 补分块求解 ----

PetscErrorCode solve_linear_system_cuda_schur(Mat A, Vec b, Vec x, PetscInt n1, PetscBool* converged, bool verbose) {
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
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA-Schur] Starting GPU + FieldSplit/Schur solve...\n");
    }

    PetscInt n_total;
    ierr = MatGetSize(A, &n_total, nullptr); CHKERRQ(ierr);
    PetscInt n2 = n_total - n1;
    if (n2 <= 0) {
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA-Schur] ERROR: n1=%d >= n_total=%d, invalid split\n",
                    (int)n1, (int)n_total);
        return PETSC_ERR_ARG_WRONG;
    }
    if (verbose) {
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA-Schur] Block sizes: vel=%d, pres=%d\n", (int)n1, (int)n2);
    }

#if defined(PETSC_HAVE_CUDA)
    // 1. 矩阵/向量搬到 GPU
    if (verbose) PetscPrintf(PETSC_COMM_SELF, "  [CUDA-Schur] Moving matrix/vectors to GPU...\n");

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
    if (verbose) {
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA-Schur] No CUDA support, falling back to CPU Schur solve\n");
    }
    return solve_linear_system_schur(A, b, x, n1, converged, verbose);
#endif

    // 2. 构造字段索引 IS
    if (verbose) PetscPrintf(PETSC_COMM_SELF, "  [CUDA-Schur] Creating field index sets...\n");

    ierr = ISCreateStride(PETSC_COMM_SELF, n1, 0,  1, &vel_is);
    if (ierr) goto cuda_schur_cleanup;
    ierr = ISCreateStride(PETSC_COMM_SELF, n2, n1, 1, &pre_is);
    if (ierr) goto cuda_schur_cleanup;

    // 3. 配置外层 FGMRES + FieldSplit/Schur
    if (verbose) PetscPrintf(PETSC_COMM_SELF, "  [CUDA-Schur] Configuring FGMRES + FieldSplit/Schur...\n");

    ierr = KSPCreate(PETSC_COMM_SELF, &ksp);
    if (ierr) goto cuda_schur_cleanup;

    ierr = KSPSetOperators(ksp, A_gpu, A_gpu);
    if (ierr) goto cuda_schur_cleanup;

    ierr = KSPSetType(ksp, KSPFGMRES);
    if (ierr) goto cuda_schur_cleanup;

    ierr = KSPGMRESSetRestart(ksp, 200);
    if (ierr) goto cuda_schur_cleanup;

    ierr = KSPSetTolerances(ksp, 1e-10, 1e-12, PETSC_DEFAULT, 1000000);
    if (ierr) goto cuda_schur_cleanup;

    // 4. PCFIELDSPLIT + Schur 补
    ierr = KSPGetPC(ksp, &pc);
    if (ierr) goto cuda_schur_cleanup;

    ierr = PCSetType(pc, PCFIELDSPLIT);
    if (ierr) goto cuda_schur_cleanup;

    ierr = PCFieldSplitSetIS(pc, "velocity", vel_is);
    if (ierr) goto cuda_schur_cleanup;
    ierr = PCFieldSplitSetIS(pc, "pressure", pre_is);
    if (ierr) goto cuda_schur_cleanup;

    // Schur 补: UPPER 分解, SELFP 预条件
    ierr = PCFieldSplitSetType(pc, PC_COMPOSITE_SCHUR);
    if (ierr) goto cuda_schur_cleanup;
    ierr = PCFieldSplitSetSchurFactType(pc, PC_FIELDSPLIT_SCHUR_FACT_UPPER);
    if (ierr) goto cuda_schur_cleanup;
    ierr = PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_SELFP, nullptr);
    if (ierr) goto cuda_schur_cleanup;

    // 5. SetUp 后配置子 KSP
    ierr = KSPSetFromOptions(ksp);
    if (ierr) goto cuda_schur_cleanup;
    ierr = KSPSetUp(ksp);
    if (ierr) goto cuda_schur_cleanup;

    // 6. 子块配置: vel=GAMG, pres=GAMG
    {
        KSP* subksp;
        PetscInt n_split;
        ierr = PCFieldSplitGetSubKSP(pc, &n_split, &subksp);
        if (ierr) goto cuda_schur_cleanup;

        if (n_split == 2) {
            PC sub_pc0, sub_pc1;
            ierr = KSPSetType(subksp[0], KSPPREONLY);
            if (!ierr) ierr = KSPGetPC(subksp[0], &sub_pc0);
            if (!ierr) ierr = PCSetType(sub_pc0, PCGAMG);

            ierr = KSPSetType(subksp[1], KSPPREONLY);
            if (!ierr) ierr = KSPGetPC(subksp[1], &sub_pc1);
            if (!ierr) ierr = PCSetType(sub_pc1, PCGAMG);
        }

        PetscFree(subksp);
        if (ierr) goto cuda_schur_cleanup;
    }

    // 7. 求解
    {
        PetscLogDouble t_start = 0, t_end = 0;
        if (verbose) PetscTime(&t_start);

        ierr = KSPSolve(ksp, b_gpu, x_gpu);

        if (verbose) {
            PetscTime(&t_end);
            PetscPrintf(PETSC_COMM_SELF, "  [CUDA-Schur] KSPSolve time: %.3f sec\n", t_end - t_start);
        }
    }
    if (ierr) goto cuda_schur_cleanup;

    // 8. 解拷回 CPU
    ierr = VecCopy(x_gpu, x);
    if (ierr) goto cuda_schur_cleanup;

    // 9. 检查收敛
    ierr = KSPGetConvergedReason(ksp, &reason);
    if (ierr) goto cuda_schur_cleanup;
    *converged = (reason >= 0) ? PETSC_TRUE : PETSC_FALSE;

    {
        PetscInt its;
        KSPGetIterationNumber(ksp, &its);
        if (*converged) {
            if (verbose) {
                PetscPrintf(PETSC_COMM_SELF, "  [CUDA-Schur] Converged in %d outer FGMRES iterations\n", (int)its);
            }
        } else {
            PetscPrintf(PETSC_COMM_SELF, "  [CUDA-Schur] WARNING: not converged, reason = %d, iterations = %d\n",
                        (int)reason, (int)its);
        }
    }

cuda_schur_cleanup:
    if (ksp)    KSPDestroy(&ksp);
    if (vel_is) ISDestroy(&vel_is);
    if (pre_is) ISDestroy(&pre_is);
    if (A_gpu)  MatDestroy(&A_gpu);
    if (b_gpu)  VecDestroy(&b_gpu);
    if (x_gpu)  VecDestroy(&x_gpu);
    return ierr;
}

PetscErrorCode solve_linear_system_cuda_schur(const AutoPetscMat& A, const AutoPetscVec& b, const AutoPetscVec& x,
                                                PetscInt n1, PetscBool* converged, bool verbose) {
    return solve_linear_system_cuda_schur(get_raw(A), get_raw(b), get_raw(x), n1, converged, verbose);
}

// ==========================================
// 5. 矩阵运算
// ==========================================

AutoPetscMat multiply_matrices(const AutoPetscMat& A, const AutoPetscMat& B, PetscScalar scalar) {
    return multiply_matrices(get_raw(A), get_raw(B), scalar);
}

AutoPetscMat multiply_matrices(Mat A, Mat B, PetscScalar scalar) {
    if (A == PETSC_NULLPTR || B == PETSC_NULLPTR) {
        throw std::invalid_argument("Null matrix in multiply");
    }

    Mat matC;
    PetscErrorCode ierr = MatMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &matC);
    if (ierr != 0) throw std::runtime_error("MatMatMult failed");

    if (scalar != 1.0) {
        ierr = MatScale(matC, scalar);
        if (ierr != 0) {
            MatDestroy(&matC);
            throw std::runtime_error("MatScale failed");
        }
    }

    return AutoPetscMat(new Mat(matC));
}

AutoPetscMat add_matrices(const AutoPetscMat& A, const AutoPetscMat& B, PetscScalar scalar) {
    return add_matrices(get_raw(A), get_raw(B), scalar);
}

AutoPetscMat add_matrices(Mat A, Mat B, PetscScalar scalar) {
    if (A == PETSC_NULLPTR || B == PETSC_NULLPTR) {
        throw std::invalid_argument("Null matrix in add");
    }

    Mat matC;
    PetscErrorCode ierr = MatDuplicate(A, MAT_COPY_VALUES, &matC);
    if (ierr != 0) throw std::runtime_error("MatDuplicate failed");

    ierr = MatAXPY(matC, 1.0, B, DIFFERENT_NONZERO_PATTERN);
    if (ierr != 0) {
        MatDestroy(&matC);
        throw std::runtime_error("MatAXPY failed");
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

AutoPetscMat scale_matrix(const AutoPetscMat& A, PetscScalar scalar) {
    return scale_matrix(get_raw(A), scalar);
}

AutoPetscMat scale_matrix(Mat A, PetscScalar scalar) {
    if (A == PETSC_NULLPTR) {
        throw std::invalid_argument("Null matrix in scale");
    }

    Mat matC;
    PetscErrorCode ierr = MatDuplicate(A, MAT_COPY_VALUES, &matC);
    if (ierr != 0) throw std::runtime_error("MatDuplicate failed");

    ierr = MatScale(matC, scalar);
    if (ierr != 0) {
        MatDestroy(&matC);
        throw std::runtime_error("MatScale failed");
    }

    return AutoPetscMat(new Mat(matC));
}

AutoPetscMat transpose_matrix(const AutoPetscMat& A) {
    return transpose_matrix(get_raw(A));
}

AutoPetscMat transpose_matrix(Mat A) {
    if (A == PETSC_NULLPTR) {
        throw std::invalid_argument("Null matrix in transpose");
    }

    Mat matT;
    PetscErrorCode ierr = MatTranspose(A, MAT_INITIAL_MATRIX, &matT);
    if (ierr != 0) throw std::runtime_error("MatTranspose failed");

    return AutoPetscMat(new Mat(matT));
}

// ==========================================
// 6. 分块填充
// ==========================================

void insert_submatrix(Mat M, const AutoPetscMat& A, PetscInt row_start, PetscInt col_start) {
    insert_submatrix(M, get_raw(A), row_start, col_start);
}

void insert_submatrix(Mat M, Mat A, PetscInt row_start, PetscInt col_start) {
    if (M == PETSC_NULLPTR || A == PETSC_NULLPTR) {
        throw std::invalid_argument("Null matrix in insert_submatrix");
    }

    PetscInt sub_rows, sub_cols;
    MatGetSize(A, &sub_rows, &sub_cols);

    for (PetscInt i = 0; i < sub_rows; ++i) {
        for (PetscInt j = 0; j < sub_cols; ++j) {
            PetscScalar val;
            MatGetValues(A, 1, &i, 1, &j, &val);
            PetscInt global_row = row_start + i;
            PetscInt global_col = col_start + j;
            MatSetValues(M, 1, &global_row, 1, &global_col, &val, INSERT_VALUES);
        }
    }
}

void add_submatrix(Mat M, const AutoPetscMat& A, PetscInt row_start, PetscInt col_start) {
    add_submatrix(M, get_raw(A), row_start, col_start);
}

void add_submatrix(Mat M, Mat A, PetscInt row_start, PetscInt col_start) {
    if (M == PETSC_NULLPTR || A == PETSC_NULLPTR) {
        throw std::invalid_argument("Null matrix in add_submatrix");
    }

    PetscInt sub_rows, sub_cols;
    MatGetSize(A, &sub_rows, &sub_cols);

    for (PetscInt i = 0; i < sub_rows; ++i) {
        for (PetscInt j = 0; j < sub_cols; ++j) {
            PetscScalar val;
            MatGetValues(A, 1, &i, 1, &j, &val);
            if (std::abs(val) < 1e-18) continue;
            PetscInt global_row = row_start + i;
            PetscInt global_col = col_start + j;
            MatSetValues(M, 1, &global_row, 1, &global_col, &val, ADD_VALUES);
        }
    }
}

// ==========================================
// 7. 调试打印
// ==========================================

void print_matrix(Mat A, const char* name) {
    print_matrix_with_threshold(A, name, 1e-16);
}

void print_matrix(const AutoPetscMat& A, const char* name) {
    Mat mat = get_raw(A);
    if (mat != PETSC_NULLPTR) {
        print_matrix(mat, name);
    } else {
        PetscPrintf(PETSC_COMM_SELF, "\n===== %s =====\n", name);
        PetscPrintf(PETSC_COMM_SELF, "  [Null matrix]\n");
    }
}

void print_matrix_with_threshold(Mat A, const char* name, double threshold) {
    PetscPrintf(PETSC_COMM_SELF, "\n===== %s =====\n", name);

    if (A == PETSC_NULLPTR) {
        PetscPrintf(PETSC_COMM_SELF, "  [Null matrix]\n");
        return;
    }

    PetscInt rows, cols;
    MatGetSize(A, &rows, &cols);

    for (PetscInt i = 0; i < rows; ++i) {
        PetscPrintf(PETSC_COMM_SELF, "row %d:", (int)i);
        for (PetscInt j = 0; j < cols; ++j) {
            PetscScalar val;
            MatGetValues(A, 1, &i, 1, &j, &val);
            if (PetscAbsScalar(val) >= threshold) {
                PetscPrintf(PETSC_COMM_SELF, " (%d, %.6g)", (int)j, val);
            }
        }
        PetscPrintf(PETSC_COMM_SELF, "\n");
    }
}

void print_matrix_with_threshold(const AutoPetscMat& A, const char* name, double threshold) {
    Mat mat = get_raw(A);
    if (mat != PETSC_NULLPTR) {
        print_matrix_with_threshold(mat, name, threshold);
    } else {
        PetscPrintf(PETSC_COMM_SELF, "\n===== %s =====\n", name);
        PetscPrintf(PETSC_COMM_SELF, "  [Null matrix]\n");
    }
}

void print_vector(Vec v, const char* name) {
    PetscPrintf(PETSC_COMM_SELF, "\n===== %s =====\n", name);
    VecView(v, PETSC_VIEWER_STDOUT_SELF);
}

void print_vector(const AutoPetscVec& v, const char* name) {
    Vec vec = get_raw(v);
    if (vec != PETSC_NULLPTR) {
        print_vector(vec, name);
    } else {
        PetscPrintf(PETSC_COMM_SELF, "\n===== %s =====\n", name);
        PetscPrintf(PETSC_COMM_SELF, "  [Null vector]\n");
    }
}

}  // namespace vem
