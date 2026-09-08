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

namespace {

// ------------------------------------------------------------------
// CUDA 求解器缓存
//
// 动机：皮卡迭代每步都要解一次同一结构的降阶系统。原实现每次调用都
// MatDuplicate 整个矩阵（64x64 网格下 680 万非零元 ≈ 82 MB）、转成
// MATSEQAIJCUSPARSE、建 KSP、做一遍完整 ILU 符号+数值分解，然后全部销毁。
// 一个算例要重复几百次，几百 MB 级的大块反复 malloc/free 会让 glibc 的
// arena 单调膨胀（表现和内存泄漏一样），最终被 OOM killer 杀掉。
//
// 现在把 A_gpu / b_gpu / x_gpu / ksp 全部缓存下来，只在矩阵结构变化时重建，
// 否则只刷新数值。正确性依据（PETSc 3.25 源码核对）：
//   * MatCopy(A, A_gpu, SAME_NONZERO_PATTERN)：SeqAIJCUSPARSE 未重载
//     ops->copy，因此走 MatCopy_SeqAIJ 快路径，memcpy 值数组后调用
//     MatSeqAIJRestoreArrayWrite，其 CUSPARSE 版把 offloadmask 置为
//     PETSC_OFFLOAD_CPU —— device 副本被作废，下次 GPU 运算重新上传，
//     不会用到旧值。快路径还断言两矩阵 nnz 相等，结构不符会报
//     PETSC_ERR_ARG_INCOMP 硬错而非静默算错。
//   * MatCopy 递增对象 state 但不动 nonzerostate，于是 PCSetUp 以
//     SAME_NONZERO_PATTERN 重做数值分解、复用符号分解，又省一笔。
//   * VecCopy(b, b_gpu) 走 VecCopy_Seq → VecGetArrayWrite/RestoreArrayWrite，
//     CUPM 向量在写访问后同样置 offloadmask = PETSC_OFFLOAD_CPU。
// ------------------------------------------------------------------
struct CudaSolverCache {
    KSP ksp   = nullptr;
    Mat A_gpu = nullptr;
    Vec b_gpu = nullptr;
    Vec x_gpu = nullptr;
    // 结构指纹：阶数 + 非零元个数 + 行指针数组。
    // 只比 (n, nnz) 理论上可能撞车（两种不同结构恰好同阶同 nnz），那样
    // memcpy 会把数值填进错误的槽位且无人报错，所以额外存一份行指针
    // （n+1 个整数，约 1 MB）做强校验。
    PetscInt n   = -1;
    PetscInt nnz = -1;
    std::vector<PetscInt> row_ptr;
    bool finalize_registered = false;
};

CudaSolverCache g_cuda_cache;

PetscErrorCode destroy_cuda_cache() {
    if (g_cuda_cache.ksp)   KSPDestroy(&g_cuda_cache.ksp);
    if (g_cuda_cache.A_gpu) MatDestroy(&g_cuda_cache.A_gpu);
    if (g_cuda_cache.b_gpu) VecDestroy(&g_cuda_cache.b_gpu);
    if (g_cuda_cache.x_gpu) VecDestroy(&g_cuda_cache.x_gpu);
    g_cuda_cache.ksp   = nullptr;
    g_cuda_cache.A_gpu = nullptr;
    g_cuda_cache.b_gpu = nullptr;
    g_cuda_cache.x_gpu = nullptr;
    g_cuda_cache.n   = -1;
    g_cuda_cache.nnz = -1;
    g_cuda_cache.row_ptr.clear();
    g_cuda_cache.row_ptr.shrink_to_fit();
    // PetscFinalize 调用完注册的回调后会清空回调表，
    // 若之后再 PetscInitialize，需要重新注册。
    g_cuda_cache.finalize_registered = false;
    return 0;
}

// 读取矩阵的行指针数组，用于结构指纹比对。
PetscErrorCode get_row_ptr(Mat A, std::vector<PetscInt>& out) {
    PetscInt nrows = 0;
    const PetscInt* ia = nullptr;
    const PetscInt* ja = nullptr;
    PetscBool done = PETSC_FALSE;

    PetscErrorCode ierr =
        MatGetRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &nrows, &ia, &ja, &done);
    if (ierr) return ierr;
    if (!done || ia == nullptr) {
        // 拿不到内部索引就退化为「不比对行指针」，靠 (n, nnz) 加 PETSc 的
        // nnz 断言兜底。
        out.clear();
        MatRestoreRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &nrows, &ia, &ja, &done);
        return 0;
    }
    out.assign(ia, ia + nrows + 1);
    return MatRestoreRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &nrows, &ia, &ja, &done);
}

bool cache_matches(Mat A, PetscInt n, PetscInt nnz) {
    if (g_cuda_cache.ksp == nullptr) return false;
    if (g_cuda_cache.n != n || g_cuda_cache.nnz != nnz) return false;
    if (g_cuda_cache.row_ptr.empty()) return true;  // 无行指针可比，退化判断

    std::vector<PetscInt> current;
    if (get_row_ptr(A, current) != 0) return false;
    return current == g_cuda_cache.row_ptr;
}

}  // namespace

void clear_cuda_solver_cache() {
    destroy_cuda_cache();
}

PetscErrorCode solve_linear_system_cuda(Mat A, Vec b, Vec x, PetscBool* converged, bool verbose) {
    PetscErrorCode ierr;
    KSPConvergedReason reason;
    PetscLogDouble t_start = 0, t_end = 0;
    PetscInt its = 0;
    PetscInt n = 0, ncols = 0, nnz = 0;
    MatInfo info;

    ierr = MatGetSize(A, &n, &ncols);
    if (ierr) return ierr;
    ierr = MatGetInfo(A, MAT_LOCAL, &info);
    if (ierr) return ierr;
    nnz = static_cast<PetscInt>(info.nz_used);

    if (verbose) {
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA] Starting GPU-accelerated solve...\n");
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA] System size: %d x %d, nnz = %d\n",
                    (int)n, (int)ncols, (int)nnz);
    }

#if !defined(PETSC_HAVE_CUDA)
    // 无 CUDA 支持，回退到 CPU 求解
    if (verbose) {
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA] PETSc not built with CUDA, falling back to CPU solve\n");
    }
    return solve_linear_system(A, b, x, converged, verbose);
#else
    if (cache_matches(A, n, nnz)) {
        // ---- 命中缓存：只刷新数值，不重建任何对象 ----
        if (verbose) {
            PetscPrintf(PETSC_COMM_SELF,
                        "  [CUDA] Reusing cached GPU solver (refreshing values only)...\n");
        }
        ierr = MatCopy(A, g_cuda_cache.A_gpu, SAME_NONZERO_PATTERN);
        if (ierr) { destroy_cuda_cache(); return ierr; }

        // 必须重新 KSPSetOperators。否则 KSPSetUp 会在
        // setupstage == KSP_SETUP_NEWRHS 处提前返回、根本不调用 PCSetUp，
        // ILU 预条件就一直冻结在建缓存那一次的矩阵上
        // （PETSc 源码 KSPSetOperators 的注释原文：
        //  "so that next solve call will call PCSetUp() on new matrix"）。
        // 粗网格上皮卡迭代前后矩阵差别很大，冻结的预条件会让 FGMRES
        // 直接发散成 KSP_DIVERGED_DTOL。
        //
        // 这里传的是同一个 Mat 指针，PCSetOperators 只在 Pmat != pc->pmat 时才清
        // matnonzerostate，所以 PCSetUp 仍会判定 SAME_NONZERO_PATTERN，
        // 复用 ILU 的符号分解、只重做数值分解 —— 依然比重建整个 KSP 便宜得多。
        ierr = KSPSetOperators(g_cuda_cache.ksp, g_cuda_cache.A_gpu, g_cuda_cache.A_gpu);
        if (ierr) { destroy_cuda_cache(); return ierr; }
    } else {
        // ---- 未命中（首次调用，或换了网格）：丢弃旧缓存重建 ----
        ierr = destroy_cuda_cache();
        if (ierr) return ierr;

        // 1. 矩阵搬到 GPU
        if (verbose) PetscPrintf(PETSC_COMM_SELF, "  [CUDA] Moving matrix to GPU...\n");
        ierr = MatDuplicate(A, MAT_COPY_VALUES, &g_cuda_cache.A_gpu);
        if (ierr) { destroy_cuda_cache(); return ierr; }
        ierr = MatSetType(g_cuda_cache.A_gpu, MATSEQAIJCUSPARSE);
        if (ierr) { destroy_cuda_cache(); return ierr; }

        // 2. 右端项容器搬到 GPU（数值在下面统一 VecCopy）
        if (verbose) PetscPrintf(PETSC_COMM_SELF, "  [CUDA] Creating GPU vectors...\n");
        ierr = VecDuplicate(b, &g_cuda_cache.b_gpu);
        if (ierr) { destroy_cuda_cache(); return ierr; }
        ierr = VecSetType(g_cuda_cache.b_gpu, VECSEQCUDA);
        if (ierr) { destroy_cuda_cache(); return ierr; }

        // 3. GPU 解向量
        ierr = VecDuplicate(g_cuda_cache.b_gpu, &g_cuda_cache.x_gpu);
        if (ierr) { destroy_cuda_cache(); return ierr; }
        ierr = VecSetType(g_cuda_cache.x_gpu, VECSEQCUDA);
        if (ierr) { destroy_cuda_cache(); return ierr; }

        // 4. 配置 KSP 求解器 (FGMRES + BJACOBI)
        if (verbose) PetscPrintf(PETSC_COMM_SELF, "  [CUDA] Configuring FGMRES solver...\n");
        ierr = KSPCreate(PETSC_COMM_SELF, &g_cuda_cache.ksp);
        if (ierr) { destroy_cuda_cache(); return ierr; }

        ierr = KSPSetOperators(g_cuda_cache.ksp, g_cuda_cache.A_gpu, g_cuda_cache.A_gpu);
        if (ierr) { destroy_cuda_cache(); return ierr; }

        PC pc = nullptr;
        ierr = KSPGetPC(g_cuda_cache.ksp, &pc);
        if (ierr) { destroy_cuda_cache(); return ierr; }

        // Block Jacobi 预条件适合 GPU
        ierr = PCSetType(pc, PCBJACOBI);
        if (ierr) { destroy_cuda_cache(); return ierr; }

        ierr = KSPSetType(g_cuda_cache.ksp, KSPFGMRES);
        if (ierr) { destroy_cuda_cache(); return ierr; }

        // restart=50：FGMRES 要存 V(m+1) + Z(m) 约 2m+1 个基向量，
        // 128x128 网格（N=489,984，3.92 MB/向量）下 m=100 要 788 MB、m=50 只要 396 MB。
        // 实测收敛是「慢但稳」（rtol 1e-10 用 54 个重启周期，每周期降 0.65），
        // 这种情形下加大 m 换不回等比例的迭代数下降，
        // 总正交化开销 ∝ n_iters × m/2 反而变差，故取 50 而非更大。
        // 可用命令行覆盖：-ksp_gmres_restart <m>（KSPSetFromOptions 在本行之后调用）
        ierr = KSPGMRESSetRestart(g_cuda_cache.ksp, 50);
        if (ierr) { destroy_cuda_cache(); return ierr; }

        ierr = KSPSetTolerances(g_cuda_cache.ksp, 1e-10, 1e-50, PETSC_DEFAULT, 1000000);
        if (ierr) { destroy_cuda_cache(); return ierr; }

        ierr = KSPSetFromOptions(g_cuda_cache.ksp);
        if (ierr) { destroy_cuda_cache(); return ierr; }

        // 5. 记录结构指纹
        g_cuda_cache.n   = n;
        g_cuda_cache.nnz = nnz;
        ierr = get_row_ptr(A, g_cuda_cache.row_ptr);
        if (ierr) { destroy_cuda_cache(); return ierr; }

        // 6. 让 PETSc 在 PetscFinalize 时自动释放缓存，
        //    免得主程序漏掉显式清理、被当成 PETSc 对象泄漏。
        if (!g_cuda_cache.finalize_registered) {
            ierr = PetscRegisterFinalize(destroy_cuda_cache);
            if (ierr) { destroy_cuda_cache(); return ierr; }
            g_cuda_cache.finalize_registered = true;
        }
    }

    // ---- 右端项数值搬到 GPU ----
    ierr = VecCopy(b, g_cuda_cache.b_gpu);
    if (ierr) { destroy_cuda_cache(); return ierr; }

    // 解向量清零：复用时上一次的解还留在里面，
    // 而调用方传进来的 x 语义上是零初值（KSPSetInitialGuessNonzero 默认关闭，
    // PETSc 会忽略 x_gpu 的初值，这里清零只为不让陈旧数据造成困惑）。
    ierr = VecSet(g_cuda_cache.x_gpu, 0.0);
    if (ierr) { destroy_cuda_cache(); return ierr; }

    // ---- 求解 ----
    if (verbose) PetscTime(&t_start);

    ierr = KSPSolve(g_cuda_cache.ksp, g_cuda_cache.b_gpu, g_cuda_cache.x_gpu);

    if (verbose) {
        PetscTime(&t_end);
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA] KSPSolve time: %.3f sec\n", t_end - t_start);
    }
    if (ierr) { destroy_cuda_cache(); return ierr; }

    // ---- 解拷回 CPU ----
    if (verbose) PetscPrintf(PETSC_COMM_SELF, "  [CUDA] Copying solution back to CPU...\n");
    ierr = VecCopy(g_cuda_cache.x_gpu, x);
    if (ierr) { destroy_cuda_cache(); return ierr; }

    // ---- 检查收敛 ----
    ierr = KSPGetConvergedReason(g_cuda_cache.ksp, &reason);
    if (ierr) { destroy_cuda_cache(); return ierr; }

    *converged = (reason >= 0) ? PETSC_TRUE : PETSC_FALSE;

    // 注意：KSP 现在跨调用复用，KSPGetTotalIterations 是累计值，
    // 会一路往上涨。本次求解的迭代数要用 KSPGetIterationNumber。
    ierr = KSPGetIterationNumber(g_cuda_cache.ksp, &its);
    if (ierr) { destroy_cuda_cache(); return ierr; }

    if (*converged) {
        if (verbose) {
            PetscPrintf(PETSC_COMM_SELF, "  [CUDA] Converged in %d iterations\n", (int)its);
        }
    } else {
        PetscPrintf(PETSC_COMM_SELF, "  [CUDA] WARNING: not converged, reason = %d, iterations = %d\n",
                    (int)reason, (int)its);
    }

    return 0;
#endif
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
