// ============================================================================
//  matrixSolver_main.cu - PETSc 示例 (简化版)
//
//  创建 10000x10000 的对角占优稀疏矩阵，传输到 GPU，并用 GMRES 求解
//
// ============================================================================

#include <petsc.h>
#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>
#include <petscdevice.h>
#include "setgpu.cuh"
#include <cstdio>

int main(int argc, char** argv)
{
    setGPU(0);
    PetscErrorCode ierr;

#if defined(PETSC_HAVE_CUDA)
    ierr = PetscOptionsSetValue(NULL, "-use_gpu_aware_mpi", "0");
    CHKERRQ(ierr);
#endif

    ierr = PetscInitialize(&argc, &argv, NULL, NULL);
    if (ierr) return ierr;

    // 问题规模
    PetscInt n = 10000;
    PetscPrintf(PETSC_COMM_WORLD, "创建 %dx%d 对角占优矩阵...\n", n, n);

    // 创建稀疏矩阵，每行预留足够空间
    Mat A;
    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, n, n, 60, NULL, &A);
    CHKERRQ(ierr);

    // 填充矩阵
    srand(time(NULL));
    for (PetscInt i = 0; i < n; i++)
    {
        // 每行随机挑选50个非对角位置
        for (PetscInt k = 0; k < 50; k++)
        {
            PetscInt j;
            do
            {
                j = rand() % n;
            } while (j == i); // 避免对角线

            // 生成 -5 到 5 的随机数
            double val = (rand() % 11) - 5.0; // 0-10 -> -5-5
            ierr = MatSetValue(A, i, j, val, INSERT_VALUES);
            CHKERRQ(ierr);
        }

        // 对角线设置为200-250的随机数，保证能收敛
        double diag_val = 200.0 + (rand() % 51); // 200-250
        ierr = MatSetValue(A, i, i, diag_val, INSERT_VALUES);
        CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);

    // 创建右端项 b = 1
    Vec b;
    ierr = VecCreateSeq(PETSC_COMM_SELF, n, &b);
    CHKERRQ(ierr);
    ierr = VecSet(b, 1.0);
    CHKERRQ(ierr);
    ierr = VecAssemblyBegin(b);
    CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b);
    CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD, "✅ CPU 矩阵和向量创建完成！\n");

#if defined(PETSC_HAVE_CUDA)
    PetscPrintf(PETSC_COMM_WORLD, "\n将 CPU 矩阵和向量转换为 GPU 类型...\n");

    ierr = MatSetType(A, MATSEQAIJCUSPARSE);
    CHKERRQ(ierr);

    ierr = VecSetType(b, VECSEQCUDA);
    CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD, "✅ GPU 矩阵和向量创建完成！\n");
    PetscPrintf(PETSC_COMM_WORLD, "  - 矩阵类型: MATSEQAIJCUSPARSE (存储在 GPU 内存)\n");
    PetscPrintf(PETSC_COMM_WORLD, "  - 向量类型: VECSEQCUDA (存储在 GPU 内存)\n");
#else
    PetscPrintf(PETSC_COMM_WORLD, "\nPETSc 没有 CUDA 支持，使用 CPU 求解\n");
#endif

    // 创建解向量 x
    Vec x;
    ierr = VecDuplicate(b, &x);
    CHKERRQ(ierr);
    ierr = VecSet(x, 0.0);
    CHKERRQ(ierr);

    // 创建 KSP 求解器
    KSP ksp;
    ierr = KSPCreate(PETSC_COMM_SELF, &ksp);
    CHKERRQ(ierr);

    ierr = KSPSetOperators(ksp, A, A);
    CHKERRQ(ierr);

    ierr = KSPSetType(ksp, KSPGMRES);
    CHKERRQ(ierr);

    // 设置 GMRES 重启次数
    ierr = KSPGMRESSetRestart(ksp, 50);
    CHKERRQ(ierr);

    // 设置无预条件
    PC pc;
    ierr = KSPGetPC(ksp, &pc);
    CHKERRQ(ierr);
    ierr = PCSetType(pc, PCNONE);
    CHKERRQ(ierr);

    ierr = KSPSetTolerances(ksp, 1e-10, 1e-50, PETSC_DEFAULT, 100000);
    CHKERRQ(ierr);

    // 确保KSP在GPU上运行
    ierr = KSPSetFromOptions(ksp);
    CHKERRQ(ierr);

    // 打印KSP类型和PC类型确认
    const char* ksp_type;
    ierr = KSPGetType(ksp, &ksp_type);
    CHKERRQ(ierr);
    const char* pc_type;
    ierr = PCGetType(pc, &pc_type);
    CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "\n求解器配置:\n");
    PetscPrintf(PETSC_COMM_WORLD, "  - KSP类型: %s\n", ksp_type);
    PetscPrintf(PETSC_COMM_WORLD, "  - PC类型: %s\n", pc_type);

    PetscPrintf(PETSC_COMM_WORLD, "\n开始用 GMRES（无预条件）求解线性方程组...\n");

    // 开始计时
    PetscLogDouble t_start, t_end;
    ierr = PetscTime(&t_start);
    CHKERRQ(ierr);

    ierr = KSPSolve(ksp, b, x);
    CHKERRQ(ierr);

    // 结束计时
    ierr = PetscTime(&t_end);
    CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD, "✅ 求解完成！\n");
    PetscPrintf(PETSC_COMM_WORLD, "  - 求解时间: %.6f 秒\n", t_end - t_start);

    // 输出迭代信息
    KSPConvergedReason reason;
    ierr = KSPGetConvergedReason(ksp, &reason);
    CHKERRQ(ierr);

    PetscInt iter;
    ierr = KSPGetIterationNumber(ksp, &iter);
    CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD, "\n求解信息:\n");
    PetscPrintf(PETSC_COMM_WORLD, "  - 收敛原因: %s\n", KSPConvergedReasons[reason]);
    PetscPrintf(PETSC_COMM_WORLD, "  - 迭代次数: %d\n", iter);

    // 输出解向量的前几个值
    PetscPrintf(PETSC_COMM_WORLD, "\n解向量前10个值:\n");
    const PetscScalar* x_arr;
    ierr = VecGetArrayRead(x, &x_arr);
    CHKERRQ(ierr);
    for (PetscInt i = 0; i < 10 && i < n; i++)
    {
        PetscPrintf(PETSC_COMM_WORLD, "  x[%d] = %g\n", i, x_arr[i]);
    }
    ierr = VecRestoreArrayRead(x, &x_arr);
    CHKERRQ(ierr);

    // 清理
    ierr = KSPDestroy(&ksp);
    CHKERRQ(ierr);
    ierr = VecDestroy(&x);
    CHKERRQ(ierr);
    ierr = MatDestroy(&A);
    CHKERRQ(ierr);
    ierr = VecDestroy(&b);
    CHKERRQ(ierr);

    ierr = PetscFinalize();
    return ierr;
}
