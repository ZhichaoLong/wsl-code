/** PETSc KSP/PC 线性求解与 Vec 代数；全部 Krylov 空间由 PETSc 管理。 */
#include "equation_gpu.cuh"
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>
namespace vgpu {
// MatShell 的 CUDA 访问必须严格配对；异常也归还借用，禁止异常跨越 C 回调。
static PetscErrorCode shell_mult(Mat mat, Vec x, Vec y) {
    MixedGpu* equation = nullptr;
    const PetscScalar* in = nullptr;
    PetscScalar* out = nullptr;
    PetscFunctionBeginUser;
    PetscCall(MatShellGetContext(mat, &equation));
    PetscCall(VecCUDAGetArrayRead(x, &in));
    PetscErrorCode err = VecCUDAGetArrayWrite(y, &out);
    if (err) {
        VecCUDARestoreArrayRead(x, &in);
        PetscFunctionReturn(err);
    }
    try {
        CUDA(cudaDeviceSynchronize());
        equation->apply(in, out);
        CUDA(cudaDeviceSynchronize());
    } catch (...) {
        VecCUDARestoreArrayRead(x, &in);
        VecCUDARestoreArrayWrite(y, &out);
        PetscFunctionReturn(PETSC_ERR_LIB);
    }
    const auto read_error = VecCUDARestoreArrayRead(x, &in);
    const auto write_error = VecCUDARestoreArrayWrite(y, &out);
    // 无论第一个 Restore 是否失败，都尝试归还第二个借用。
    PetscCall(read_error);
    PetscCall(write_error);
    PetscFunctionReturn(PETSC_SUCCESS);
}
// 兼容 matrix-free 模式的近似对角预条件；运算也交由 PETSc Vec 完成。
static PetscErrorCode shell_pc(PC pc, Vec x, Vec y) {
    MixedGpu* equation = nullptr;
    PetscFunctionBeginUser;
    PetscCall(PCShellGetContext(pc, &equation));
    try {
        PetscCall(VecPointwiseDivide(y, x, equation->diagonal_vector().raw()));
    } catch (...) {
        PetscFunctionReturn(PETSC_ERR_LIB);
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}
Krylov::Krylov(int n, bool assembled, std::string pc, int levels, std::string ordering)
    : residual_(n), work_(n), ordered_b_(n), ordered_x_(n), assembled_(assembled), pc_(std::move(pc)),
      ordering_(std::move(ordering)), levels_(levels) {
    PETSC(KSPCreate(PETSC_COMM_SELF, ksp_.put()));
    PETSC(KSPSetType(ksp_.get(), KSPFGMRES));
    PETSC(KSPSetPCSide(ksp_.get(), PC_RIGHT));
    PETSC(KSPSetNormType(ksp_.get(), KSP_NORM_UNPRECONDITIONED));
    PETSC(KSPSetInitialGuessNonzero(ksp_.get(), PETSC_TRUE));
    PETSC(KSPGMRESSetOrthogonalization(ksp_.get(), KSPGMRESClassicalGramSchmidtOrthogonalization));
    PETSC(KSPGMRESSetCGSRefinementType(ksp_.get(), KSP_GMRES_CGS_REFINE_ALWAYS));
    PETSC(KSPSetErrorIfNotConverged(ksp_.get(), PETSC_FALSE));
}
double Krylov::norm(const PetscVector& x) {
    PetscReal value;
    PETSC(VecNorm(x.raw(), NORM_2, &value));
    return value;
}
void Krylov::copy(const PetscVector& x, PetscVector& y) {
    PETSC(VecCopy(x.raw(), y.raw()));
}
double Krylov::difference(const PetscVector& x, const PetscVector& y) {
    PETSC(VecWAXPY(residual_.raw(), -1, y.raw(), x.raw()));
    return norm(residual_);
}
SolveInfo Krylov::solve(MixedGpu& equation, const PetscVector& b, PetscVector& x, double tol, int maxit, int restart) {
    // 默认规则与 SolverOptions 一致；不在数值失败后静默换预条件器。
    if (ordering_ == "auto")
        ordering_ = equation.components() == 1 ? "natural" : "nd";
    Mat op;
    if (assembled_) {
        op = equation.ordered_matrix(ordering_);
    } else {
        if (!shell_.get()) {
            PETSC(MatCreateShell(PETSC_COMM_SELF, equation.size(), equation.size(), equation.size(),
                                 equation.size(), &equation, shell_.put()));
            PETSC(MatShellSetOperation(shell_.get(), MATOP_MULT, (void (*)(void))shell_mult));
            PETSC(MatShellSetVecType(shell_.get(), VECSEQCUDA));
        }
        PETSC(MatShellSetContext(shell_.get(), &equation));
        op = shell_.get();
    }
    const bool changed = operator_ != op || equation_ != &equation;
    PETSC(KSPSetOperators(ksp_.get(), op, op));
    if (changed) {
        // 同维数不同方程可以复用 KSP，但不能复用前一张网格的置换。
        scatter_.reset();
        PC pc;
        PETSC(KSPGetPC(ksp_.get(), &pc));
        if (assembled_) {
            PETSC(PCSetType(pc, pc_.c_str()));
            if (pc_ == "ilu" || pc_ == "lu") {
                // 重复更新 cuSPARSE 因子时曾观察到主机 RSS 持续增长。
                // 显式用 PETSc CPU 因子，避开 GPU ILU0/SpSV 更新路径。
                // 仅设置 MatCUSPARSESetUseCPUSolve 不够：本机 PETSc 的
                // natural + ILU(0) 专用分支不检查该开关。
                // op 仍为 GPU Mat；PETSc 在数值分解时同步其主机镜像，
                // 在 CPU MatSolve 时同步 Vec，因子唯一归 PC/KSP 管理。
                PETSC(PCFactorSetMatSolverType(pc, MATSOLVERPETSC));
                if (pc_ == "ilu")
                    PETSC(PCFactorSetLevels(pc, levels_));
                // 保留原各模式的 shift 设置，避免本次后端切换另改预条件参数。
                if (pc_ == "lu" || levels_ > 0)
                    PETSC(PCFactorSetShiftType(pc, MAT_SHIFT_NONZERO));
                PETSC(PCFactorSetMatOrderingType(pc, MATORDERINGNATURAL));
            }
        } else {
            PETSC(PCSetType(pc, PCSHELL));
            PETSC(PCShellSetContext(pc, &equation));
            PETSC(PCShellSetApply(pc, shell_pc));
        }
        equation_ = &equation;
        operator_ = op; // KSP 持有引用，此处仅借用以识别算子切换。
    }
    PETSC(KSPGMRESSetRestart(ksp_.get(), std::min(restart, equation.size())));
    const double bn = norm(b), threshold = tol * (bn > 0 ? bn : 1);
    const bool permuted = assembled_ && ordering_ != "natural";
    if (permuted && !scatter_.get())
        PETSC(VecScatterCreate(b.raw(), equation.permutation(), ordered_b_.raw(), nullptr, scatter_.put()));
    auto scatter = [&](Vec from, Vec to, ScatterMode mode) {
        PETSC(VecScatterBegin(scatter_.get(), from, to, INSERT_VALUES, mode));
        PETSC(VecScatterEnd(scatter_.get(), from, to, INSERT_VALUES, mode));
    };
    PetscInt total_iterations = 0;
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    double residual = 0, ksp_seconds = 0;
    // 当递推残差已收敛而原顺序 b-Ax 未达标时，最多做两次残差校正。
    // 校正解 A*delta=(b-Ax)，随后 x+=delta；不放宽容差，也不改变原方程。
    for (int refinement = 0; refinement < 3 && total_iterations < maxit; ++refinement) {
        const PetscVector& rhs = refinement ? residual_ : b;
        PetscVector& solution = refinement ? work_ : x;
        if (refinement)
            solution.zero();
        PETSC(KSPSetInitialGuessNonzero(ksp_.get(), refinement ? PETSC_FALSE : PETSC_TRUE));
        PETSC(KSPSetTolerances(ksp_.get(), 0.1 * tol, 0.1 * threshold, PETSC_DEFAULT,
                               maxit - total_iterations));
        if (permuted) {
            scatter(rhs.raw(), ordered_b_.raw(), SCATTER_FORWARD);
            scatter(solution.raw(), ordered_x_.raw(), SCATTER_FORWARD);
        }
        // 仅计 PETSc 求解（含 PC 数值准备），不含前面的 COO 装配与向量置换。
        CUDA(cudaDeviceSynchronize());
        const auto ksp_start = std::chrono::steady_clock::now();
        PETSC(KSPSolve(ksp_.get(), permuted ? ordered_b_.raw() : rhs.raw(),
                      permuted ? ordered_x_.raw() : solution.raw()));
        CUDA(cudaDeviceSynchronize());
        ksp_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - ksp_start).count();
        if (permuted)
            scatter(ordered_x_.raw(), solution.raw(), SCATTER_REVERSE);
        if (refinement)
            PETSC(VecAXPY(x.raw(), 1, solution.raw()));
        PetscInt its;
        PETSC(KSPGetIterationNumber(ksp_.get(), &its));
        PETSC(KSPGetConvergedReason(ksp_.get(), &reason));
        total_iterations += its;
        // 原顺序 MatMult 是最终标准；difference 同时将 b-Ax 保存在 residual_。
        PETSC(MatMult(assembled_ ? equation.assembled_matrix() : op, x.raw(), work_.raw()));
        residual = difference(b, work_) / (bn > 0 ? bn : 1);
        if (reason <= 0 || !std::isfinite(residual))
            break;
        if (residual <= tol)
            return {int(total_iterations), residual, refinement, ksp_seconds};
    }
    std::ostringstream message;
    message << "PETSc KSP failed: reason=" << reason << " iterations=" << total_iterations
            << " true_relative_residual=" << std::scientific << std::setprecision(12) << residual;
    throw std::runtime_error(message.str());
}
// 记录实际对象类型及后端；PCFactorGetMatrix 返回借用句柄，不能 MatDestroy。
void Krylov::describe(std::ostream& out) {
    KSPType kt;
    PCType pt;
    PC pc;
    Mat a;
    MatType mt;
    Vec rhs;
    VecType vt;
    PETSC(KSPGetRhs(ksp_.get(), &rhs));
    PETSC(VecGetType(rhs, &vt));
    PETSC(KSPGetType(ksp_.get(), &kt));
    PETSC(KSPGetPC(ksp_.get(), &pc));
    PETSC(PCGetType(pc, &pt));
    PETSC(KSPGetOperators(ksp_.get(), &a, nullptr));
    PETSC(MatGetType(a, &mt));
    out << "PETSc KSP=" << kt << " PC=" << pt << " Mat=" << mt << " Vec=" << vt;
    if (assembled_) {
        MatInfo info;
        PETSC(MatGetInfo(a, MAT_LOCAL, &info));
        out << " nnz=" << info.nz_used;
        if (pc_ == "ilu" || pc_ == "lu") {
            Mat factor;
            MatSolverType backend;
            MatType factor_type;
            PETSC(PCFactorGetMatrix(pc, &factor));
            PETSC(MatFactorGetSolverType(factor, &backend));
            PETSC(MatGetType(factor, &factor_type));
            // 检查实际因子，不再根据填充级数推测执行设备。
            const bool cpu_factor = std::string(backend) == MATSOLVERPETSC &&
                                    std::string(factor_type) == MATSEQAIJ;
            if (!cpu_factor)
                throw std::runtime_error("Expected PETSc CPU SeqAIJ factor backend");
            out << " factor_backend=" << backend << " global_ordering=" << ordering_
                << " factor_ordering=natural" << " levels=" << levels_
                << " factor_mat=" << factor_type
                << " numeric_factorization=CPU triangular_solve=CPU";
        }
    }
    out << '\n' << std::flush;
}
} // namespace vgpu
