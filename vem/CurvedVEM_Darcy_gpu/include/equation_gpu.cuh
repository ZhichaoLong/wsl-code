/**
 * 混合方程的 GPU 接口：基础父类之上增加 Darcy/MS 算子、边界和后处理。
 * DarcyGpu / MSGpu 选择方程与几何；MixedGpu 共用混合块流程；Krylov 负责线性求解。
 * 所有形参中的 double* 向量均指向 GPU 显存，除明确返回的主机报告外不自动下载。
 */
#pragma once
#include "hdiv_gpu.cuh"
#include "solver_result.h"
namespace vgpu {
// 按值传入 CUDA 核的方程视图，无资源所有权。
struct EquationView {
    // 基础几何、编号和 D/P/W 等矩阵的设备指针。
    BaseView b;
    // 组分数、算例号、全局系统维数；Darcy 含额外压力均值乘子。
    int ns, example, N;
    // 质量块系数：Darcy 为 0，Euler 为 1/dt，BDF2 为 3/(2dt)。
    double rate;
    // AG: 多项式组分 Gram；alpha: 逐单元逐组分稳定化；M: 物理 L² 质量。
    // rhs: 右端；bc: 非齐次边界 lift；diag: 近似预条件对角。
    double *AG, *alpha, *M, *rhs, *bc, *diag;
    // 全局边界自由度为 1，自由未知量为 0。
    int* mask;
    // 积分点原始初值，保留间断函数，不预先用 L² 系数替代。
    const double* initial;
    // 当前时刻每积分点每组分四个数：c、Jx、Jy、f。
    const double* exact;
};
// 继承基础矩阵父类，构造方程因子并提供 matrix-free 操作。
class MixedGpu : public HdivGpu {
    // 方程规模、最大单元通量自由度和当前质量系数，用于分配共享内存。
protected:
    int ns_, example_, N_, maxn_ = 0;
    PetscMat sparse_, ordered_;
    PetscIs permutation_;
    bool ordered_dirty_ = true;
    std::string ordering_;
    void graph_indices(std::vector<PetscInt>&, std::vector<PetscInt>&, std::vector<PetscCount>&);
    Buffer<PetscCount> coo_offsets_;
    PetscVector coo_values_;
    PetscCount coo_entries_ = 0;
    bool sparse_dirty_ = true;
    void create_sparse_graph();
    double rate_ = 0;
    // 持久显存：方程因子、求解右端/边界、积分样本及可选几何矩缓存。
    PetscDense AG_, M_, moments_, tensor_moments_;
    PetscVector alpha_, rhs_, bc_, diag_, initial_, coefficients_, rhs_work_, exact_;
    // 缓存开关与同一时间步的源项采样标记，不改变直接积分数学定义。
    bool use_moments_ = true;
    bool exact_valid_ = false;
    double exact_time_ = 0;
    // 按时间更新源项/真解积分点缓存，Picard 内重复调用时复用。
    void sample_data(double);
    Buffer<int> mask_;
    // 先调用基础父类，再初始化方程缓冲区；只支持一组分 Darcy 或三组分 MS。
public:
    MixedGpu(const vem::StraightMeshReader&, int k, int nq, int ng, const Geometry&, int ns, int example,
             int tile = 4096, int threads = 128);
    EquationView equation(bool include_vectors = true);
    // GPU COO 数值更新；稀疏图只预分配一次，MatSetValuesCOO 汇总共享边。
    Mat assembled_matrix();
    Mat ordered_matrix(const std::string& ordering);
    // 只报告实际已分配的 COO 数值和 CSR 图规模，不把因子大小冒充全部显存。
    PetscCount coo_entries() const { return coo_entries_; }
    double sparse_nonzeros(bool ordered = false) const;
    IS permutation() const { return permutation_.get(); }
    const PetscVector& diagonal_vector() const { return diag_; }
    // 完整全局向量长度。
    int size() const {
        return N_;
    }
    // 当前方程组分数：1 或 3。
    int components() const {
        return ns_;
    }
    // 按 Picard 浓度更新系数和预条件；initial_mode=true 时使用积分点初值。
    void update(const double* guess, bool initial_mode, double rate);
    // 用于切换几何矩收缩/直接积分以做等价对照；显存超预算时自动无缓存。
    void set_moment_cache(bool enabled) {
        use_moments_ = enabled;
    }
    // 只有开关开启且缓存已实际分配时才返回 true。
    bool moment_cache_enabled() const {
        return use_moments_ && moments_.size() > 0;
    }
    // 设备端 y=A*x；constrained=true 应用边界 mask，false 用于计算原始 A*g。
    void apply(const double* x, double* y, bool constrained = true);
    // 结合 Euler/BDF2 历史、源项和边界消元构造右端；历史向量是设备指针。
    void make_rhs(double time, double dt, int step, bool bdf2, const double* prev, const double* prevprev);
    // 积分给定时刻的边界法向矩并更新 mask。
    void boundary_lift(double time);
    // GPU 投影及两级误差/质量归约，返回每组分指标；无真解误差为 NaN。
    ErrorReport errors(const double*, double time);
    // 计算 MS 原始初始质量，供守恒比较；Darcy 返回空数组。
    std::vector<double> initial_mass();
    // 下列只读视图供求解器或数值对照使用，不转移设备内存所有权。
    const PetscVector& rhs() const {
        return rhs_;
    }
    const PetscVector& bc() const {
        return bc_;
    }
    const PetscDense& weights() const {
        return AG_;
    }
    const PetscVector& alpha() const {
        return alpha_;
    }
    const PetscDense& mass() const {
        return M_;
    }
    // 报告方程矩阵因子/缓存显存和各核实际资源占用。
    size_t operator_bytes() const;
    void resources(std::ostream&);
};
// Darcy 入口：一组分、K=I，采用原 Darcy 几何映射。
class DarcyGpu : public MixedGpu {
public:
    DarcyGpu(const vem::StraightMeshReader& r, const Darcy::DarcyProblem& p, int example, int k = 1,
             int nq = 9, int ng = 9, int tile = 4096, int threads = 128)
        : MixedGpu(r, k, nq, ng, darcy_geometry(p), 1, example, tile, threads) {}
};
// Maxwell–Stefan 入口：三组分，采用原 MS 逐单元映射。
class MSGpu : public MixedGpu {
public:
    MSGpu(const vem::StraightMeshReader& r, const MaxwellStefan::MSProblem& p, int example, int k = 1,
          int nq = 9, int ng = 9, int tile = 4096, int threads = 128)
        : MixedGpu(r, k, nq, ng, ms_geometry(p), 3, example, tile, threads) {}
};
// 线性求解收敛信息：实际迭代次数与重新计算的真实相对残差。
struct SolveInfo {
    int iterations;
    double residual;
    int refinements = 0; // 严格真实残差检查触发的校正次数。
    double ksp_seconds = 0; // KSPSolve 墙钟，含 PC 准备与 GPU 同步；校正次数累计。
};
// PETSc KSP 管理全部 Krylov 基与正交化；不再维护自写 Hessenberg/GMRES。
class Krylov {
    PetscKsp ksp_;
    PetscMat shell_;
    PetscVector residual_, work_, ordered_b_, ordered_x_;
    PetscScatter scatter_;
    MixedGpu* equation_ = nullptr;
    Mat operator_ = nullptr; // 借用 KSP 持有的算子，不单独销毁。
    bool assembled_;
    std::string pc_, ordering_;
    int levels_;
public:
    Krylov(int n, bool assembled = true, std::string pc = "ilu", int levels = 0, std::string ordering = "auto");
    SolveInfo solve(MixedGpu&, const PetscVector&, PetscVector&, double tol, int maxit, int restart);
    double norm(const PetscVector&);
    double difference(const PetscVector&, const PetscVector&);
    void copy(const PetscVector&, PetscVector&);
    KSP raw() const { return ksp_.get(); }
    void describe(std::ostream&);
};
} // namespace vgpu
