#include "MSSolver.h"

#include "../lib/polynomial_basis.h"
#include "../mesh/polygon_triangulator.h"
#include "../core/petsc_utils.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>

using Basis = vem::basis::BasisFunctionPloy;
using BasisPoint = vem::basis::Point2D;
using BasisVector = vem::basis::Vector2D;

namespace {

BasisPoint makeBasisPoint(double x, double y) {
    return BasisPoint(x, y);
}

void setMatrixValue(Mat matrix, PetscInt row, PetscInt column, double value) {
    MatSetValue(matrix, row, column, value, INSERT_VALUES);
}

void assembleMatrix(Mat matrix) {
    MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY);
}

// ---- 计时辅助：墙钟秒 ----
using TimerClock = std::chrono::steady_clock;

inline TimerClock::time_point tic() { return TimerClock::now(); }

inline double toc(const TimerClock::time_point& t0) {
    return std::chrono::duration<double>(TimerClock::now() - t0).count();
}

// 打印一行「阶段 耗时 占比」
void printTimingLine(const std::string& name, double secs, double total) {
    double pct = (total > 0.0) ? (100.0 * secs / total) : 0.0;
    std::cout << "      " << std::left << std::setw(16) << name
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(10) << secs << " s"
              << std::setw(8) << std::setprecision(1) << pct << " %"
              << std::endl;
}

}  // namespace

// ============================================================================
// 组装计时
// ============================================================================

void MSSolver::AssemblyTiming::clear() {
    parent_mats = 0.0;
    h_curved = 0.0;
    g_cmin = 0.0;
    extract_conc = 0.0;
    ag = 0.0;
    k_ac = 0.0;
    stab = 0.0;
    insert_local = 0.0;
    direction = 0.0;
    rhs = 0.0;
    to_global = 0.0;
    global_final = 0.0;
    total = 0.0;
    n_calls = 0;
}

void MSSolver::AssemblyTiming::accumulate(const AssemblyTiming& o) {
    parent_mats  += o.parent_mats;
    h_curved     += o.h_curved;
    g_cmin       += o.g_cmin;
    extract_conc += o.extract_conc;
    ag           += o.ag;
    k_ac         += o.k_ac;
    stab         += o.stab;
    insert_local += o.insert_local;
    direction    += o.direction;
    rhs          += o.rhs;
    to_global    += o.to_global;
    global_final += o.global_final;
    total        += o.total;
    n_calls      += o.n_calls;
}

void MSSolver::AssemblyTiming::print(const std::string& tag) const {
    std::cout << "    ---- 组装耗时明细 [" << tag << "] ----" << std::endl;
    printTimingLine("父类单元矩阵", parent_mats, total);
    printTimingLine("H_curved", h_curved, total);
    printTimingLine("G_cmin", g_cmin, total);
    printTimingLine("提取浓度系数", extract_conc, total);
    printTimingLine("AG 矩阵", ag, total);
    printTimingLine("K_ac", k_ac, total);
    printTimingLine("稳定化", stab, total);
    printTimingLine("子块插入", insert_local, total);
    printTimingLine("方向调整", direction, total);
    printTimingLine("右端项", rhs, total);
    printTimingLine("局部到全局", to_global, total);
    printTimingLine("全局最终装配", global_final, total);
    std::cout << "      " << std::left << std::setw(16) << "合计"
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(10) << total << " s" << std::endl;
}

// ============================================================================
// 构造与析构
// ============================================================================

MSSolver::MSSolver(const MeshReader& mesh_reader,
                   const MaxwellStefan::MSProblem& problem,
                   double delta_t,
                   double final_time,
                   int picard_max_iter,
                   double picard_tol,
                   int gauss_point_num,
                   int gauss_point_num_1d,
                   int k,
                   bool save_all_steps,
                   const std::string& data_subfolder,
                   bool use_bdf2)
    : HdivMatrix(mesh_reader, gauss_point_num, gauss_point_num_1d, k),
      problem_(&problem),
      delta_t_(delta_t),
      final_time_(final_time),
      current_time_(0.0),
      use_bdf2_(use_bdf2),
      picard_iter_count_(0),
      picard_max_iter_(picard_max_iter),
      picard_tol_(picard_tol),
      initialized_(false),
      solved_(false),
      save_all_steps_(save_all_steps),
      data_subfolder_(data_subfolder),
      timing_enabled_(false) {
    n_comp_ = problem.get_pde_data().n_components;
}

MSSolver::~MSSolver() = default;

// ============================================================================
// 初始化
// ============================================================================

bool MSSolver::initialize() {
    // 1. 计算每个单元的总自由度：单组分(通量+浓度) * 组分数
    const std::vector<int>& dof_flux_vec = getDofPerElementFluxVec();
    int dof_conc = getDofPerElementConc();
    int num_elem = static_cast<int>(getMeshData()->num_cells);

    dof_per_element_total_.resize(num_elem);
    for (int i = 0; i < num_elem; ++i) {
        int dof_single_comp = dof_flux_vec[i] + dof_conc;
        dof_per_element_total_[i] = dof_single_comp * n_comp_;
    }

    // 2. 计算所有组分的全局自由度：单组分全局自由度 * 组分数
    total_dof_flux_all_ = getTotalDofFlux() * n_comp_;
    total_dof_conc_all_ = getTotalDofConc() * n_comp_;

    PetscInt total_dof = static_cast<PetscInt>(getTotalDofAll());
    if (total_dof <= 0) {
        return false;
    }

    // 3. 预估每行非零元个数，参考 FVM 做法：单元总自由度 * 8
    PetscInt max_dof_total = 0;
    for (int dof_total : dof_per_element_total_) {
        if (dof_total > max_dof_total)
            max_dof_total = dof_total;
    }
    PetscInt nnz_per_row = max_dof_total * 8;
    if (nnz_per_row > total_dof)
        nnz_per_row = total_dof;

    // 4. 创建系统刚度矩阵、右端项、解向量
    system_stiffness_mat_ =
        vem::create_sparse_matrix(total_dof, total_dof, nnz_per_row);
    system_rhs_vec_ = vem::create_vector(total_dof);
    solution_ = vem::create_vector(total_dof);
    solution_prev_time_ = vem::create_vector(total_dof);
    solution_picard_ = vem::create_vector(total_dof);
    // BDF2 用的 u^{n-2} 向量初始化为空（nullptr），
    // 在第 2 个时间步开始时创建并赋值，以区分"未启用"和"已赋值"
    solution_prev_prev_time_.reset();

    // 5. 显式建立全局稀疏矩阵的完整非零结构（值全为零）。
    //    必须一次性把所有单元块的槽位都插进去再 MatAssemblyEnd：
    //    MatAssemblyEnd 会把未使用的预分配空间压缩掉，如果这里只插对角元，
    //    矩阵会被压成每行 1 个非零元，之后第一次真正组装时每个新非零元都要
    //    重新 malloc 并搬移整行数据，开销随规模超线性爆炸
    //    （16x16 网格 k=1 实测：第一次组装 179 s，之后每次仅 0.03 s）。
    //    另外 LU/ILU 及部分 PETSc 预条件器要求每行都存在对角元槽位，
    //    单元块不一定覆盖全部对角元，所以对角线单独再补一遍。
    Mat raw_K = vem::get_raw(system_stiffness_mat_);
    for (PetscInt idx = 0; idx < total_dof; ++idx) {
        PetscScalar zero = 0.0;
        PetscErrorCode ierr = MatSetValue(
            raw_K, idx, idx, zero, INSERT_VALUES);
        if (ierr != 0) {
            throw std::runtime_error(
                "MSSolver::initialize: failed to set diagonal slot");
        }
    }
    {
        const MeshData* mesh_data = getMeshData();
        std::vector<PetscScalar> zero_block;
        for (int elem = 0; elem < mesh_data->num_cells; ++elem) {
            std::vector<PetscInt> global_dofs = getLocalToGlobalDofs(elem);
            PetscInt nd = static_cast<PetscInt>(global_dofs.size());
            zero_block.assign(static_cast<std::size_t>(nd) * nd, 0.0);
            PetscErrorCode ierr = MatSetValues(
                raw_K, nd, global_dofs.data(), nd, global_dofs.data(),
                zero_block.data(), INSERT_VALUES);
            if (ierr != 0) {
                throw std::runtime_error(
                    "MSSolver::initialize: failed to preallocate element block");
            }
        }
    }
    PetscErrorCode ierr = MatAssemblyBegin(raw_K, MAT_FINAL_ASSEMBLY);
    if (ierr != 0) {
        throw std::runtime_error(
            "MSSolver::initialize: MatAssemblyBegin failed");
    }
    ierr = MatAssemblyEnd(raw_K, MAT_FINAL_ASSEMBLY);
    if (ierr != 0) {
        throw std::runtime_error(
            "MSSolver::initialize: MatAssemblyEnd failed");
    }

    initialized_ = true;
    return true;
}

// ============================================================================
// 全局组装
// ============================================================================

PetscErrorCode MSSolver::assembleStiffnessMatrix(bool is_initial_step,
                                                  bool is_initial_first_tstep) {
    PetscErrorCode ierr;
    Mat mat = vem::get_raw(system_stiffness_mat_);
    Vec rhs = vem::get_raw(system_rhs_vec_);

    const bool tm = timing_enabled_;
    TimerClock::time_point t_all = tic();
    TimerClock::time_point t0;
    if (tm) timing_.clear();

    // 清空矩阵和右端项，避免累加之前的迭代结果
    ierr = MatZeroEntries(mat);
    if (ierr != 0) return ierr;
    ierr = VecZeroEntries(rhs);
    if (ierr != 0) return ierr;

    // ========== 获取父类所有属性（通过公共接口）==========
    const MeshData* mesh_data = getMeshData();
    int total_elements = mesh_data->num_cells;
    const std::vector<int>& dof_flux_vec_percomp = getDofPerElementFluxVec();
    int dof_conc_per_elem_percomp = getDofPerElementConc();

    // 单元循环内复用的矩阵变量
    AutoPetscMat matD, matG, matW, matH, matH_t, matH_star;
    AutoPetscMat matB_1, matB_2, matB_grad, matB_curl, matB;
    AutoPetscMat matL2Proj_ploy, matL2Proj_basis;
    AutoPetscMat W_trans, W_trans_neg;

    // 遍历所有单元，组装刚度矩阵
    for (int elem = 0; elem < total_elements; ++elem) {
        int dof_flux_percomp = dof_flux_vec_percomp[elem];

        // ========== 获取父类计算的单元矩阵 ==========
        if (tm) t0 = tic();
        matD = getMatrixD(elem);
        matG = getMatrixG(elem);
        matW = getMatrixW(elem);
        // W 的转置
        W_trans = vem::transpose_matrix(matW);

        matH = getMatrixH(elem);
        if (tm) timing_.parent_mats += toc(t0);

        // H_t = H_curved * alpha_0 / delta_t（时间项质量矩阵，曲边积分含 detJ）
        //   BDF1 (第一个时间步 或 未启用BDF2): alpha_0 = 1
        //   BDF2 (后续时间步且启用BDF2): alpha_0 = 3/2
        if (tm) t0 = tic();
        AutoPetscMat matH_curved = HdivMatrixH_curved(elem);
        if (tm) timing_.h_curved += toc(t0);

        double mass_coeff = 1.0 / delta_t_;
        if (use_bdf2_ && !is_initial_first_tstep) {
            mass_coeff = 1.5 / delta_t_;  // 3 / (2 dt)
        }
        if (tm) t0 = tic();
        matH_t = vem::scale_matrix(matH_curved, mass_coeff);
        matH_star = getMatrixH_star(elem);
        matB_2 = getMatrixB_2(elem);
        matB_1 = getMatrixB_1(matH_star, matH, matW);
        matB_grad = getMatrixB_grad(elem, matH_star, matH, matW);
        matB_curl = getMatrixB_curl(elem);
        matB = getMatrixB(elem, matH_star, matH, matW);
        matL2Proj_ploy = getMatrixL2Proj_ploy(matG, matB);
        matL2Proj_basis = getMatrixL2Proj_basis(matG, matB, matD);

        // W 的转置取负（flux-conc 耦合块）
        W_trans_neg = vem::scale_matrix(W_trans, -1.0);
        if (tm) timing_.parent_mats += toc(t0);

        // ========== 单元局部刚度矩阵大小（所有组分：先通量后浓度）==========
        int total_flux_dof = dof_flux_percomp * n_comp_;
        int total_conc_dof = dof_conc_per_elem_percomp * n_comp_;
        int local_size = total_flux_dof + total_conc_dof;
        AutoPetscMat K_local = vem::create_dense_matrix(local_size, local_size);
        Mat rawK_local = vem::get_raw(K_local);
        MatZeroEntries(rawK_local);

        // ========== 获取本单元的 G_cmin 矩阵 ==========
        if (tm) t0 = tic();
        AutoPetscMat cmin_G = HdivMatrixG_cmin(elem);
        if (tm) timing_.g_cmin += toc(t0);

        // ========== 获取该单元所有组分的浓度多项式系数
        // （非初始步/非首时间步计算一次）==========
        std::vector<double> u_ploy_allcomp;
        if (!(is_initial_step && is_initial_first_tstep)) {
            // 从 solution_picard_ 提取（皮卡迭代用上一步猜测值）
            if (tm) t0 = tic();
            u_ploy_allcomp = getElementComponentConcentration_allcomp(
                elem, false);
            if (tm) timing_.extract_conc += toc(t0);
        }

        // ========== 循环每个组分对 (n,m)，组装通量-通量块 ==========
        for (int n = 0; n < n_comp_; n++) {
            for (int m = 0; m < n_comp_; m++) {
                // 根据情况选择用初始还是多项式系数计算 AG 矩阵
                if (tm) t0 = tic();
                AutoPetscMat A_G;
                if (is_initial_step && is_initial_first_tstep) {
                    A_G = HdivMatrixAG_init(elem, n, m);
                } else {
                    A_G = HdivMatrixAG_poly(elem, n, m, u_ploy_allcomp);
                }
                if (tm) timing_.ag += toc(t0);

                // 计算 AG 对应的 K_ac
                if (tm) t0 = tic();
                AutoPetscMat K_ac = HdivMatrixK_ac(A_G, matL2Proj_ploy);
                if (tm) timing_.k_ac += toc(t0);

                // 对角线块需要加稳定化矩阵和 cmin 项
                if (n == m) {
                    // 1. 稳定化系数：A对角线 + c_min
                    if (tm) t0 = tic();
                    double stab_coeff;
                    if (is_initial_step && is_initial_first_tstep) {
                        stab_coeff = computeStabilizationCoeffInit(elem, n, m);
                    } else {
                        stab_coeff = computeStabilizationCoeffPoly(
                            elem, n, m, u_ploy_allcomp);
                    }
                    // 加上 c* 项的稳定化系数（曲边网格下用积分，不是简单乘面积）
                    stab_coeff += computeStabilizationCoeffCmin(elem);

                    // 2. 稳定化矩阵加到 K_ac
                    AutoPetscMat K_stab = HdivMatrixK_as(
                        matL2Proj_basis, stab_coeff);
                    MatAXPY(vem::get_raw(K_ac), 1.0, vem::get_raw(K_stab),
                            DIFFERENT_NONZERO_PATTERN);
                    MatAssemblyBegin(vem::get_raw(K_ac), MAT_FINAL_ASSEMBLY);
                    MatAssemblyEnd(vem::get_raw(K_ac), MAT_FINAL_ASSEMBLY);
                    if (tm) timing_.stab += toc(t0);

                    // 3. cmin 对应的 K_ac 加到 K_ac
                    if (tm) t0 = tic();
                    AutoPetscMat K_cmin = HdivMatrixK_ac(
                        cmin_G, matL2Proj_ploy);
                    MatAXPY(vem::get_raw(K_ac), 1.0, vem::get_raw(K_cmin),
                            DIFFERENT_NONZERO_PATTERN);
                    MatAssemblyBegin(vem::get_raw(K_ac), MAT_FINAL_ASSEMBLY);
                    MatAssemblyEnd(vem::get_raw(K_ac), MAT_FINAL_ASSEMBLY);
                    if (tm) timing_.k_ac += toc(t0);
                }

                // 插入到 K_local 的对应位置（flux-flux 块）
                if (tm) t0 = tic();
                int row_start = n * dof_flux_percomp;
                int col_start = m * dof_flux_percomp;
                vem::insert_submatrix(rawK_local, K_ac, row_start, col_start);
                if (tm) timing_.insert_local += toc(t0);
            }

            // ========== 插入时间相关的分块矩阵（W, W^T, H_t）==========
            if (tm) t0 = tic();
            int flux_base_n = n * dof_flux_percomp;
            int conc_base_n = total_flux_dof + n * dof_conc_per_elem_percomp;

            // W^T 块：行在通量区，列在浓度区（取负号）
            vem::insert_submatrix(rawK_local, W_trans_neg,
                                  flux_base_n, conc_base_n);

            // W 块：行在浓度区，列在通量区
            vem::insert_submatrix(rawK_local, matW, conc_base_n, flux_base_n);

            // H_t 块：行在浓度区，列在浓度区 (H / delta_t)
            vem::insert_submatrix(rawK_local, matH_t,
                                  conc_base_n, conc_base_n);
            if (tm) timing_.insert_local += toc(t0);
        }

        // ========== 组装完成单元局部矩阵 ==========
        if (tm) t0 = tic();
        MatAssemblyBegin(rawK_local, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(rawK_local, MAT_FINAL_ASSEMBLY);
        if (tm) timing_.insert_local += toc(t0);

        // ========== 计算并应用方向调整 ==========
        if (tm) t0 = tic();
        std::vector<double> local_dof_direction =
            computeAndApplyDirectionAdjustment(elem, rawK_local);
        (void)local_dof_direction;  // 方向向量已在函数内应用到矩阵
        if (tm) timing_.direction += toc(t0);

        // ========== 生成单元右端项 ==========
        if (tm) t0 = tic();
        AutoPetscVec F_elem = HdivVecRhs(elem, is_initial_first_tstep);
        if (tm) timing_.rhs += toc(t0);

        // ========== 组装到全局 ==========
        if (tm) t0 = tic();
        assembleLocalToGlobal(elem, K_local);
        assembleLocalRhsToGlobal(elem, F_elem);
        if (tm) timing_.to_global += toc(t0);
    }

    // ========== 全局矩阵和向量最终装配 ==========
    if (tm) t0 = tic();
    ierr = MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
    if (ierr != 0) return ierr;
    ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);
    if (ierr != 0) return ierr;
    ierr = VecAssemblyBegin(rhs);
    if (ierr != 0) return ierr;
    ierr = VecAssemblyEnd(rhs);
    if (ierr != 0) return ierr;
    if (tm) timing_.global_final += toc(t0);

    if (tm) {
        timing_.total = toc(t_all);
        timing_.n_calls++;
        timing_accum_.accumulate(timing_);
    }

    return 0;
}

// ============================================================================
// 法向通量边界条件（H(div) 本质边界）
//   算法：
//   1. 标记所有边界通量自由度（每个组分的边界边矩自由度）
//   2. 在计算域边界边上积分 boundary_flux_comp · 边单项式，得到边界值
//      （Piola 变换保持法向通量积分，计算域积分等价于物理域积分）
//   3. 右端修正：F = F - K * U_bc
//   4. 提取自由自由度子矩阵和子向量
// ============================================================================

PetscErrorCode MSSolver::applyNormalFluxBoundaryConditions(
    AutoPetscMat& reduced_matrix,
    AutoPetscVec& reduced_rhs,
    IS& free_dofs,
    AutoPetscVec& boundary_values,
    PetscInt& total_dof,
    double time) {
    const MeshData* mesh_data = getMeshData();
    Mat global_matrix = vem::get_raw(system_stiffness_mat_);
    Vec global_rhs = vem::get_raw(system_rhs_vec_);
    if (global_matrix == PETSC_NULLPTR || global_rhs == PETSC_NULLPTR) {
        return PETSC_ERR_ARG_NULL;
    }

    PetscInt matrix_rows = 0, matrix_cols = 0;
    PetscErrorCode ierr = MatGetSize(global_matrix, &matrix_rows, &matrix_cols);
    if (ierr != 0) return ierr;
    ierr = VecGetSize(global_rhs, &total_dof);
    if (ierr != 0) return ierr;
    if (matrix_rows != total_dof || matrix_cols != total_dof) {
        return PETSC_ERR_ARG_SIZ;
    }

    int k = getK();
    int flux_per_comp = getTotalDofFlux();  // 单组分通量自由度
    int num_edges = mesh_data->num_edges;

    const MaxwellStefan::MSPdeData& pde = problem_->get_pde_data();

    // 边界值向量（全尺寸，非边界处为 0）
    boundary_values = vem::create_vector(total_dof);
    Vec raw_bd = vem::get_raw(boundary_values);
    VecSet(raw_bd, 0.0);

    // 边界自由度标记
    std::vector<PetscBool> is_boundary_dof(
        static_cast<std::size_t>(total_dof), PETSC_FALSE);

    Basis edge_basis(k);
    vem::GaussQuadrature quadrature;

    // 遍历所有单元，找到边界边并计算边界通量矩
    // 每条边界边只属于一个单元，按该单元的局部边方向计算，
    // 使边界值与全局组装时的边自由度方向一致。
    for (int elem = 0; elem < mesh_data->num_cells; ++elem) {
        int edge_count = mesh_data->nodes_per_cell[elem];
        int cell_node_start = mesh_data->cell_node_indices[elem];
        int cell_edge_start = mesh_data->cell_edge_indices[elem];

        for (int local_edge = 0; local_edge < edge_count; ++local_edge) {
            int global_edge =
                mesh_data->global_edges[cell_edge_start + local_edge];
            if (mesh_data->edge_occurrence[global_edge] != 1) {
                continue;  // 不是边界边
            }

            // 边的两个端点（计算域坐标）
            int node0 = mesh_data->cell_nodes[cell_node_start + local_edge];
            int node1 = mesh_data->cell_nodes[
                cell_node_start + (local_edge + 1) % edge_count];
            BasisPoint point0(
                mesh_data->node_coords[2 * node0],
                mesh_data->node_coords[2 * node0 + 1]);
            BasisPoint point1(
                mesh_data->node_coords[2 * node1],
                mesh_data->node_coords[2 * node1 + 1]);

            // 计算域边外法向
            Point2D normal = computeEdgeNormal(node0, node1);

            // 计算域边上的高斯积分点
            std::vector<vem::QuadPoint> points =
                quadrature.get_line_2D_points(
                    vem::QuadratureType::GAUSS_LEGENDRE,
                    getGaussPointNum1D(),
                    point0.x, point0.y, point1.x, point1.y);

            // 对每个组分
            for (int comp = 0; comp < n_comp_; ++comp) {
                int comp_offset = comp * flux_per_comp;

                // 对每个多项式次数（degree-major 布局）
                for (int degree = 0; degree <= k; ++degree) {
                    double boundary_moment = 0.0;
                    for (const vem::QuadPoint& point : points) {
                        double xi = point.x[0];
                        double eta = point.x[1];

                        // 计算域法向通量（已含 Piola 变换，
                        // 积分不变性保证无需额外乘 J_e）
                        double flux = pde.boundary_flux_comp(
                            comp, xi, eta, normal.x, normal.y, time);

                        BasisPoint quad_point(xi, eta);
                        double monomial = edge_basis.evalEdgeMonomial(
                            degree, point0, point1, quad_point);

                        boundary_moment += flux * monomial * point.w;
                    }

                    PetscInt global_dof =
                        comp_offset + degree * num_edges + global_edge;
                    PetscScalar val = boundary_moment;
                    ierr = VecSetValue(raw_bd, global_dof,
                                       val, INSERT_VALUES);
                    if (ierr != 0) return ierr;
                    is_boundary_dof[static_cast<std::size_t>(global_dof)] =
                        PETSC_TRUE;
                }
            }
        }
    }

    ierr = VecAssemblyBegin(raw_bd);
    if (ierr != 0) return ierr;
    ierr = VecAssemblyEnd(raw_bd);
    if (ierr != 0) return ierr;

    // 收集自由自由度索引
    std::vector<PetscInt> free_indices;
    free_indices.reserve(static_cast<std::size_t>(total_dof));
    for (PetscInt idx = 0; idx < total_dof; ++idx) {
        if (!is_boundary_dof[static_cast<std::size_t>(idx)]) {
            free_indices.push_back(idx);
        }
    }

    // 右端修正：F = F - K * U_bc
    // （对完整系统做修正，包括浓度方程也受到边界通量的影响）
    Vec mat_times_bd = PETSC_NULLPTR;
    Vec adjusted_rhs = PETSC_NULLPTR;
    ierr = VecDuplicate(global_rhs, &mat_times_bd);
    if (ierr != 0) return ierr;
    ierr = VecDuplicate(global_rhs, &adjusted_rhs);
    if (ierr != 0) {
        VecDestroy(&mat_times_bd);
        return ierr;
    }
    ierr = MatMult(global_matrix, raw_bd, mat_times_bd);
    if (ierr != 0) {
        VecDestroy(&mat_times_bd);
        VecDestroy(&adjusted_rhs);
        return ierr;
    }
    ierr = VecCopy(global_rhs, adjusted_rhs);
    if (ierr != 0) {
        VecDestroy(&mat_times_bd);
        VecDestroy(&adjusted_rhs);
        return ierr;
    }
    ierr = VecAXPY(adjusted_rhs, -1.0, mat_times_bd);
    if (ierr != 0) {
        VecDestroy(&mat_times_bd);
        VecDestroy(&adjusted_rhs);
        return ierr;
    }

    // 创建自由自由度索引集
    ierr = ISCreateGeneral(
        PETSC_COMM_SELF,
        static_cast<PetscInt>(free_indices.size()),
        free_indices.data(), PETSC_COPY_VALUES, &free_dofs);
    if (ierr != 0) {
        VecDestroy(&mat_times_bd);
        VecDestroy(&adjusted_rhs);
        return ierr;
    }

    // 提取自由自由度子矩阵
    Mat reduced_raw = PETSC_NULLPTR;
    ierr = MatCreateSubMatrix(
        global_matrix, free_dofs, free_dofs,
        MAT_INITIAL_MATRIX, &reduced_raw);
    if (ierr != 0) {
        VecDestroy(&mat_times_bd);
        VecDestroy(&adjusted_rhs);
        return ierr;
    }
    reduced_matrix = vem::AutoPetscMat(new Mat(reduced_raw));

    // 提取自由自由度右端项
    Vec rhs_view = PETSC_NULLPTR;
    ierr = VecGetSubVector(adjusted_rhs, free_dofs, &rhs_view);
    if (ierr != 0) {
        VecDestroy(&mat_times_bd);
        VecDestroy(&adjusted_rhs);
        return ierr;
    }
    Vec reduced_rhs_raw = PETSC_NULLPTR;
    ierr = VecDuplicate(rhs_view, &reduced_rhs_raw);
    if (ierr == 0) {
        ierr = VecCopy(rhs_view, reduced_rhs_raw);
    }
    PetscErrorCode restore_err =
        VecRestoreSubVector(adjusted_rhs, free_dofs, &rhs_view);
    VecDestroy(&mat_times_bd);
    VecDestroy(&adjusted_rhs);
    if (ierr != 0) {
        if (reduced_rhs_raw != PETSC_NULLPTR) {
            VecDestroy(&reduced_rhs_raw);
        }
        return ierr;
    }
    if (restore_err != 0) {
        VecDestroy(&reduced_rhs_raw);
        return restore_err;
    }
    reduced_rhs = vem::AutoPetscVec(new Vec(reduced_rhs_raw));

    return 0;
}

// ============================================================================
// 求解（单步）
//   组装 + 边界条件 + 求解一步到位。
//   注意：第一个时间步第一个皮卡迭代使用初值函数，
//   其他情况使用 solution_picard_ 作为非线性项猜测值。
// ============================================================================

bool MSSolver::solve() {
    if (!initialized_) {
        return false;
    }
    // 默认：非初始步、非初始时间步（稳态或后续时间步使用）
    PetscErrorCode error = assembleStiffnessMatrix(false, false);
    if (error != 0) {
        return false;
    }
    return solveWithDirichletBC() == 0;
}

// ============================================================================
// solveWithDirichletBC
//   假定矩阵和右端已组装好（由调用者控制 is_initial_step 标志）。
//   完成：边界降维 → GMRES 求解 → 恢复完整解 → 写入 solution_
// ============================================================================

PetscErrorCode MSSolver::solveWithDirichletBC() {
    solved_ = false;
    solution_.reset();

    AutoPetscMat reduced_matrix;
    AutoPetscVec reduced_rhs;
    AutoPetscVec boundary_values;
    IS free_dofs = PETSC_NULLPTR;
    PetscInt total_dof = 0;

    // 边界条件降维（法向通量本质边界）
    PetscErrorCode ierr = applyNormalFluxBoundaryConditions(
        reduced_matrix, reduced_rhs, free_dofs,
        boundary_values, total_dof, current_time_);
    if (ierr != 0) {
        if (free_dofs != PETSC_NULLPTR) ISDestroy(&free_dofs);
        return ierr;
    }

    // 自由自由度解向量
    PetscInt free_size = 0;
    ierr = VecGetSize(vem::get_raw(reduced_rhs), &free_size);
    if (ierr != 0) {
        ISDestroy(&free_dofs);
        return ierr;
    }
    AutoPetscVec free_solution = vem::create_vector(free_size);
    ierr = VecSet(vem::get_raw(free_solution), 0.0);
    if (ierr != 0) {
        ISDestroy(&free_dofs);
        return ierr;
    }
    ierr = VecAssemblyBegin(vem::get_raw(free_solution));
    if (ierr == 0) ierr = VecAssemblyEnd(vem::get_raw(free_solution));
    if (ierr != 0) {
        ISDestroy(&free_dofs);
        return ierr;
    }

    // GMRES 求解降阶系统
    PetscBool converged = PETSC_FALSE;
    ierr = vem::solve_linear_system(
        reduced_matrix, reduced_rhs, free_solution,
        &converged, true);
    if (ierr != 0 || !converged) {
        ISDestroy(&free_dofs);
        return ierr != 0 ? ierr : PETSC_ERR_NOT_CONVERGED;
    }

    // 恢复完整解：先复制边界值，再填入自由自由度
    Vec solution_raw = PETSC_NULLPTR;
    ierr = VecDuplicate(vem::get_raw(boundary_values), &solution_raw);
    if (ierr != 0) {
        ISDestroy(&free_dofs);
        return ierr;
    }
    solution_ = AutoPetscVec(new Vec(solution_raw));
    ierr = VecCopy(vem::get_raw(boundary_values), vem::get_raw(solution_));
    if (ierr != 0) {
        ISDestroy(&free_dofs);
        solution_.reset();
        return ierr;
    }

    const PetscInt* free_indices = nullptr;
    PetscInt free_index_count = 0;
    ierr = ISGetLocalSize(free_dofs, &free_index_count);
    if (ierr == 0) ierr = ISGetIndices(free_dofs, &free_indices);
    if (ierr != 0) {
        ISDestroy(&free_dofs);
        solution_.reset();
        return ierr;
    }

    const PetscScalar* free_values = nullptr;
    ierr = VecGetArrayRead(vem::get_raw(free_solution), &free_values);
    if (ierr == 0) {
        ierr = VecSetValues(
            vem::get_raw(solution_), free_index_count,
            free_indices, free_values, INSERT_VALUES);
    }
    PetscErrorCode restore_vec_err = VecRestoreArrayRead(
        vem::get_raw(free_solution), &free_values);
    PetscErrorCode restore_idx_err = ISRestoreIndices(free_dofs, &free_indices);
    ISDestroy(&free_dofs);

    if (ierr != 0) { solution_.reset(); return ierr; }
    if (restore_vec_err != 0) { solution_.reset(); return restore_vec_err; }
    if (restore_idx_err != 0) { solution_.reset(); return restore_idx_err; }

    ierr = VecAssemblyBegin(vem::get_raw(solution_));
    if (ierr == 0) ierr = VecAssemblyEnd(vem::get_raw(solution_));
    if (ierr != 0) {
        solution_.reset();
        return ierr;
    }

    PetscInt solution_size = 0;
    ierr = VecGetSize(vem::get_raw(solution_), &solution_size);
    if (ierr != 0 || solution_size != total_dof) {
        solution_.reset();
        return ierr != 0 ? ierr : PETSC_ERR_ARG_SIZ;
    }

    solved_ = true;
    return 0;
}

// ============================================================================
// GPU 版本求解
//   与 solveWithDirichletBC 逻辑相同，仅线性求解器换为 CUDA GMRES + BJACOBI
// ============================================================================

PetscErrorCode MSSolver::solveWithDirichletBC_GPU() {
    solved_ = false;
    solution_.reset();

    AutoPetscMat reduced_matrix;
    AutoPetscVec reduced_rhs;
    AutoPetscVec boundary_values;
    IS free_dofs = PETSC_NULLPTR;
    PetscInt total_dof = 0;

    // 边界条件降维（法向通量本质边界）
    PetscErrorCode ierr = applyNormalFluxBoundaryConditions(
        reduced_matrix, reduced_rhs, free_dofs,
        boundary_values, total_dof, current_time_);
    if (ierr != 0) {
        if (free_dofs != PETSC_NULLPTR) ISDestroy(&free_dofs);
        return ierr;
    }

    // 自由自由度解向量
    PetscInt free_size = 0;
    ierr = VecGetSize(vem::get_raw(reduced_rhs), &free_size);
    if (ierr != 0) {
        ISDestroy(&free_dofs);
        return ierr;
    }
    AutoPetscVec free_solution = vem::create_vector(free_size);
    ierr = VecSet(vem::get_raw(free_solution), 0.0);
    if (ierr != 0) {
        ISDestroy(&free_dofs);
        return ierr;
    }
    ierr = VecAssemblyBegin(vem::get_raw(free_solution));
    if (ierr == 0) ierr = VecAssemblyEnd(vem::get_raw(free_solution));
    if (ierr != 0) {
        ISDestroy(&free_dofs);
        return ierr;
    }

    // GPU GMRES + BJACOBI 求解降阶系统
    PetscBool converged = PETSC_FALSE;
    ierr = vem::solve_linear_system_cuda(
        reduced_matrix, reduced_rhs, free_solution,
        &converged, true);
    if (ierr != 0 || !converged) {
        ISDestroy(&free_dofs);
        return ierr != 0 ? ierr : PETSC_ERR_NOT_CONVERGED;
    }

    // 恢复完整解：先复制边界值，再填入自由自由度
    Vec solution_raw = PETSC_NULLPTR;
    ierr = VecDuplicate(vem::get_raw(boundary_values), &solution_raw);
    if (ierr != 0) {
        ISDestroy(&free_dofs);
        return ierr;
    }
    solution_ = AutoPetscVec(new Vec(solution_raw));
    ierr = VecCopy(vem::get_raw(boundary_values), vem::get_raw(solution_));
    if (ierr != 0) {
        ISDestroy(&free_dofs);
        solution_.reset();
        return ierr;
    }

    const PetscInt* free_indices = nullptr;
    PetscInt free_index_count = 0;
    ierr = ISGetLocalSize(free_dofs, &free_index_count);
    if (ierr == 0) ierr = ISGetIndices(free_dofs, &free_indices);
    if (ierr != 0) {
        ISDestroy(&free_dofs);
        solution_.reset();
        return ierr;
    }

    const PetscScalar* free_values = nullptr;
    ierr = VecGetArrayRead(vem::get_raw(free_solution), &free_values);
    if (ierr == 0) {
        ierr = VecSetValues(
            vem::get_raw(solution_), free_index_count,
            free_indices, free_values, INSERT_VALUES);
    }
    PetscErrorCode restore_vec_err = VecRestoreArrayRead(
        vem::get_raw(free_solution), &free_values);
    PetscErrorCode restore_idx_err = ISRestoreIndices(free_dofs, &free_indices);
    ISDestroy(&free_dofs);

    if (ierr != 0) { solution_.reset(); return ierr; }
    if (restore_vec_err != 0) { solution_.reset(); return restore_vec_err; }
    if (restore_idx_err != 0) { solution_.reset(); return restore_idx_err; }

    ierr = VecAssemblyBegin(vem::get_raw(solution_));
    if (ierr == 0) ierr = VecAssemblyEnd(vem::get_raw(solution_));
    if (ierr != 0) {
        solution_.reset();
        return ierr;
    }

    PetscInt solution_size = 0;
    ierr = VecGetSize(vem::get_raw(solution_), &solution_size);
    if (ierr != 0 || solution_size != total_dof) {
        solution_.reset();
        return ierr != 0 ? ierr : PETSC_ERR_ARG_SIZ;
    }

    solved_ = true;
    return 0;
}

// ============================================================================
// 时间推进（含 Picard 不动点迭代）
//
// 核心逻辑：
//   - 每个时间步内部做 Picard 迭代，线性化非线性项 A(u)
//   - 第 1 个时间步第 1 次 Picard 迭代：用初值真解函数计算 A_ij 和右端时间项
//     （is_initial_step=true, is_initial_first_tstep=true）
//   - 第 1 个时间步第 2 次及以后 Picard 迭代：
//     A_ij 用上一次 Picard 解重构，右端时间项仍用初值函数
//     （is_initial_step=false, is_initial_first_tstep=true）
//   - 第 2 个及以后时间步第 1 次 Picard 迭代：
//     A_ij 用上一时间步解重构，右端时间项用上一时间步解
//     （is_initial_step=false, is_initial_first_tstep=false）
//   - 后续时间步后续 Picard 迭代：A_ij 和右端都用上一次 Picard 解/上一时间步解
//     （is_initial_step=false, is_initial_first_tstep=false）
//
// BDF2 启动策略：
//   - 第 1 个时间步强制使用 BDF1（向后欧拉），质量系数 1/dt，右端只含 u^{n-1}
//   - 第 2 个时间步及以后，若 use_bdf2_=true 则切换为 BDF2：
//       质量系数 3/(2dt)，右端含 (4 u^{n-1} - u^{n-2}) / (2dt)
//
// 向量更新关系：
//   - solution_prev_prev_time_：上两个时间步的解 u^{n-2}（仅 BDF2 使用）
//   - solution_prev_time_：上一时间步收敛解 u^{n-1}（整个时间步内不变）
//   - solution_picard_：当前 Picard 迭代的猜测值（每次迭代更新）
//   - solution_：当前刚求出的新解 u^n
//   - 每步收敛后滚动更新：u^{n-2} ← u^{n-1},  u^{n-1} ← u^n
// ============================================================================

PetscErrorCode MSSolver::solveTimeStepping(const std::string& folder_name,
                                              bool use_gpu) {
    (void)folder_name;  // 保留参数接口，保存路径由 save_all_steps_ 内部控制
    if (!initialized_) {
        return PETSC_ERR_ARG_WRONGSTATE;
    }
    PetscErrorCode ierr;

    current_time_ = 0.0;
    int time_step_count = 0;

    PetscInt vec_size = 0;
    ierr = VecGetSize(vem::get_raw(solution_), &vec_size);
    CHKERRQ(ierr);

    // ========== 时间推进循环 ==========
    while (current_time_ < final_time_ - 1e-12) {
        current_time_ += delta_t_;
        time_step_count++;

        std::cout << "\n========================================" << std::endl;
        std::cout << "时间步 " << time_step_count
                  << ", t = " << current_time_
                  << std::endl;
        std::cout << "========================================" << std::endl;

        // 第 1 个时间步时 solution_prev_time_ 还是初始零向量，
        // 但 is_initial_first_tstep=true 时右端用初值函数，所以不影响。
        // 第 2 个时间步起，solution_prev_time_ 已经存了上一步收敛解，
        // 在本步开始时直接使用。历史解滚动在每步结束后进行。

        // ===== Picard 迭代 =====
        bool converged = false;
        picard_iter_count_ = 0;
        bool is_initial_step;
        bool is_initial_first_tstep;

        // --- 第 1 次 Picard 迭代（直接求解，不做收敛检查）---
        picard_iter_count_++;
        std::cout << "\n  Picard 迭代 " << picard_iter_count_
                  << "/" << picard_max_iter_ << std::endl;

        if (time_step_count == 1) {
            is_initial_first_tstep = true;
            is_initial_step = true;       // 第 1 次迭代用初值真解函数算 A_ij
        } else {
            is_initial_first_tstep = false;
            is_initial_step = false;      // 用上一时间步解算 A_ij
        }

        // 组装（自动根据两个 bool 选择 AG_init / AG_poly、右端方式）
        TimerClock::time_point t_asm = tic();
        ierr = assembleStiffnessMatrix(is_initial_step, is_initial_first_tstep);
        CHKERRQ(ierr);
        double asm_secs = toc(t_asm);

        // 求解
        TimerClock::time_point t_slv = tic();
        if (use_gpu) {
            ierr = solveWithDirichletBC_GPU();
        } else {
            ierr = solveWithDirichletBC();
        }
        CHKERRQ(ierr);
        double slv_secs = toc(t_slv);

        if (timing_enabled_) {
            std::cout << "    [计时] 组装 " << std::fixed
                      << std::setprecision(3) << asm_secs
                      << " s， 求解 " << slv_secs << " s"
                      << "  (AG 模式: "
                      << ((is_initial_step && is_initial_first_tstep)
                              ? "init" : "poly")
                      << ")" << std::endl;
            std::ostringstream tag;
            tag << "时间步" << time_step_count
                << " 皮卡" << picard_iter_count_;
            timing_.print(tag.str());
        }

        // 保存到 solution_picard_ 作为下一次迭代的猜测
        ierr = VecCopy(
            vem::get_raw(solution_),
            vem::get_raw(solution_picard_));
        CHKERRQ(ierr);

        // --- 第 2 次及以后的 Picard 迭代（做收敛检查）---
        while (!converged && picard_iter_count_ < picard_max_iter_) {
            picard_iter_count_++;
            std::cout << "\n  Picard 迭代 " << picard_iter_count_
                      << "/" << picard_max_iter_ << std::endl;

            if (time_step_count == 1) {
                is_initial_first_tstep = true;  // 第一个时间步所有迭代都用初值函数算右端时间项
                is_initial_step = false;        // 从第 2 次迭代起 A_ij 用上一次 Picard 解
            } else {
                is_initial_first_tstep = false;
                is_initial_step = false;
            }

            // 保存旧的 Picard 解用于收敛判断
            AutoPetscVec sol_picard_old = vem::create_vector(vec_size);
            ierr = VecCopy(
                vem::get_raw(solution_picard_),
                vem::get_raw(sol_picard_old));
            CHKERRQ(ierr);

            // 组装（非线性系数用上一次 Picard 迭代的解）
            TimerClock::time_point t_asm2 = tic();
            ierr = assembleStiffnessMatrix(is_initial_step, is_initial_first_tstep);
            CHKERRQ(ierr);
            double asm2_secs = toc(t_asm2);

            // 求解
            TimerClock::time_point t_slv2 = tic();
            if (use_gpu) {
                ierr = solveWithDirichletBC_GPU();
            } else {
                ierr = solveWithDirichletBC();
            }
            CHKERRQ(ierr);
            double slv2_secs = toc(t_slv2);

            if (timing_enabled_) {
                std::cout << "    [计时] 组装 " << std::fixed
                          << std::setprecision(3) << asm2_secs
                          << " s， 求解 " << slv2_secs << " s"
                          << "  (AG 模式: poly)" << std::endl;
                std::ostringstream tag2;
                tag2 << "时间步" << time_step_count
                     << " 皮卡" << picard_iter_count_;
                timing_.print(tag2.str());
            }

            // 计算新旧解差的二范数
            PetscReal norm_diff = 0.0;
            ierr = VecAXPY(vem::get_raw(solution_), -1.0,
                           vem::get_raw(sol_picard_old));
            CHKERRQ(ierr);
            ierr = VecNorm(vem::get_raw(solution_), NORM_2, &norm_diff);
            CHKERRQ(ierr);
            // 恢复 solution_
            ierr = VecAXPY(vem::get_raw(solution_), 1.0,
                           vem::get_raw(sol_picard_old));
            CHKERRQ(ierr);

            std::cout << "    残差: " << std::scientific
                      << norm_diff
                      << " (容差: " << picard_tol_ << ")" << std::endl;

            // 判断收敛
            if (norm_diff < picard_tol_) {
                converged = true;
                std::cout << "    ✓ Picard 收敛 (" << picard_iter_count_
                          << " 步)" << std::endl;
            } else {
                // 未收敛，更新 Picard 猜测值
                ierr = VecCopy(
                    vem::get_raw(solution_),
                    vem::get_raw(solution_picard_));
                CHKERRQ(ierr);
            }
        }

        if (!converged) {
            std::cerr << "  ⚠ 警告：Picard 迭代在 " << picard_max_iter_
                      << " 步内未收敛！" << std::endl;
        }

        // 收敛后，solution_ 和 solution_picard_ 应该相同
        // （最后一次迭代求出的新解就是收敛解，已经在 solution_ 里，
        //  solution_picard_ 是上一次的，所以这里再同步一次）
        if (converged) {
            ierr = VecCopy(
                vem::get_raw(solution_),
                vem::get_raw(solution_picard_));
            CHKERRQ(ierr);
        }

        // 收敛后滚动更新历史解向量：
        //   u^{n-2} ← u^{n-1},   u^{n-1} ← u^n
        // BDF2 第 2 步的 u^{n-2}=u^0 用初值函数（右端里直接处理），
        // 从第 2 步结束后起，solution_prev_prev_time_ 才开始有有效数据。
        if (use_bdf2_ && time_step_count >= 2) {
            // 第 2 步结束时 prev_prev 向量还未创建，先创建
            if (!solution_prev_prev_time_) {
                PetscInt sz = 0;
                VecGetSize(vem::get_raw(solution_prev_time_), &sz);
                solution_prev_prev_time_ = vem::create_vector(sz);
            }
            ierr = VecCopy(
                vem::get_raw(solution_prev_time_),
                vem::get_raw(solution_prev_prev_time_));
            CHKERRQ(ierr);
        }
        // 更新上一时间步解
        ierr = VecCopy(
            vem::get_raw(solution_),
            vem::get_raw(solution_prev_time_));
        CHKERRQ(ierr);

        // 保存当前时刻的解
        if (save_all_steps_) {
            saveSolutionToFile(current_time_, data_subfolder_);
        }
    }

    // 只保存最后一步时，时间推进结束后保存一次到 {data_subfolder_}_final/
    if (!save_all_steps_) {
        saveSolutionToFile(current_time_, data_subfolder_ + "_final");
    }

    std::cout << "\n时间推进完成！共 " << time_step_count << " 个时间步。"
              << std::endl;
    return 0;
}

// ============================================================================
// 单元矩阵接口
// ============================================================================

AutoPetscMat MSSolver::getMatrixG_cmin(int mesh_idx) {
    return HdivMatrixG_cmin(mesh_idx);
}

AutoPetscMat MSSolver::getMatrixAG_init(int mesh_idx, int i, int j) {
    return HdivMatrixAG_init(mesh_idx, i, j);
}

AutoPetscMat MSSolver::getMatrixAG_poly(int mesh_idx, int i, int j,
                                         const std::vector<double>& u_ploy_coeff) {
    return HdivMatrixAG_poly(mesh_idx, i, j, u_ploy_coeff);
}

AutoPetscMat MSSolver::getMatrixK_ac(const AutoPetscMat& G,
                                      const AutoPetscMat& L2Proj_ploy) {
    return HdivMatrixK_ac(G, L2Proj_ploy);
}

AutoPetscMat MSSolver::getMatrixK_as(const AutoPetscMat& L2Proj_basis,
                                      double coeff) {
    return HdivMatrixK_as(L2Proj_basis, coeff);
}

double MSSolver::getStabilizationCoeffCmin(int mesh_idx) {
    return computeStabilizationCoeffCmin(mesh_idx);
}

double MSSolver::getStabilizationCoeffInit(int mesh_idx, int i, int j) {
    return computeStabilizationCoeffInit(mesh_idx, i, j);
}

double MSSolver::getStabilizationCoeffPoly(int mesh_idx, int i, int j,
                                            const std::vector<double>& u_ploy_coeff) {
    return computeStabilizationCoeffPoly(mesh_idx, i, j, u_ploy_coeff);
}

AutoPetscVec MSSolver::getLocalRhs(int mesh_idx, bool is_initial_first_tstep) {
    return HdivVecRhs(mesh_idx, is_initial_first_tstep);
}

// ============================================================================
// G_cmin 矩阵：c* 加权 Gram 矩阵（线性扩散项，各向异性张量形式）
//   G_cmin = integral_hatE g_I^T * (c*/detJ * J^T J) * g_J d hat{x}
// ============================================================================

double MSSolver::vemHdiv_gm_gm_ploy_cmin(int mesh_idx,
                                          int ploy_mk1,
                                          int ploy_mk2) {
    const MeshData* mesh_data = getMeshData();
    if (mesh_idx < 0 || mesh_idx >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }

    Basis basis(getK());
    int dim = basis.getDimHdiv();
    if (ploy_mk1 < 0 || ploy_mk1 >= dim ||
        ploy_mk2 < 0 || ploy_mk2 >= dim) {
        throw std::out_of_range("H(div) polynomial index out of range");
    }

    double cstar = problem_->get_pde_data().c_star;
    const MaxwellStefan::IsoparametricMapping& mapping =
        problem_->get_mapping();

    BasisPoint xD = makeBasisPoint(mesh_data->cell_centroid_x[mesh_idx],
                                   mesh_data->cell_centroid_y[mesh_idx]);
    double hD = mesh_data->cell_diameter[mesh_idx];
    // 单元三角形分割
    vem::PolygonTriangulator triangulator;
    vem::TriangulatedElement element =
        triangulator.triangulateElement(*mesh_data, mesh_idx);
    vem::GaussQuadrature quadrature;

    double result = 0.0;
    for (const vem::Triangle& triangle : element.triangles) {
        //三角形积分
        std::vector<vem::QuadPoint> points = quadrature.get_triangle_points(
            triangle.vertices, getGaussPointNum2D());
        for (const vem::QuadPoint& point : points) {
            double xi = point.x[0];
            double eta = point.x[1];
            BasisPoint x = makeBasisPoint(xi, eta);

            // 几何修正张量 K_xi = (c* / detJ) * J^T * J
            // 按完整 2x2 张量的四个分量存储，不假设对称性
            MaxwellStefan::Tensor2D J = mapping.jacobian(xi, eta);
            //雅可比的行列式
            double detJ = mapping.jacobian_det(xi, eta);
            double s = cstar / detJ;

            // nu = s * J^T * J 的四个分量：
            //   nu_xx = s * (J.xx^2 + J.yx^2)
            //   nu_xy = s * (J.xx * J.xy + J.yx * J.yy)
            //   nu_yx = s * (J.xy * J.xx + J.yy * J.yx)
            //   nu_yy = s * (J.xy^2 + J.yy^2)
            double nu_xx = s * (J.xx * J.xx + J.yx * J.yx);
            double nu_xy = s * (J.xx * J.xy + J.yx * J.yy);
            double nu_yx = s * (J.xy * J.xx + J.yy * J.yx);
            double nu_yy = s * (J.xy * J.xy + J.yy * J.yy);

            BasisVector first = basis.evalHdivBasis(ploy_mk1, xD, hD, x);
            BasisVector second = basis.evalHdivBasis(ploy_mk2, xD, hD, x);

            // g_I^T * nu * g_J 四项展开（非对称张量通用形式）：
            //   = nu_xx * g_Ix * g_Jx
            //   + nu_xy * g_Ix * g_Jy
            //   + nu_yx * g_Iy * g_Jx
            //   + nu_yy * g_Iy * g_Jy
            double val = nu_xx * first.x * second.x
                       + nu_xy * first.x * second.y
                       + nu_yx * first.y * second.x
                       + nu_yy * first.y * second.y;
            result += val * point.w;
        }
    }
    return result;
}

AutoPetscMat MSSolver::HdivMatrixG_cmin(int mesh_idx) {
    const MeshData* mesh_data = getMeshData();
    if (mesh_idx < 0 || mesh_idx >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }

    int dim = Basis(getK()).getDimHdiv();
    AutoPetscMat matrix = vem::create_dense_matrix(dim, dim);
    Mat raw = vem::get_raw(matrix);

    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            double val = vemHdiv_gm_gm_ploy_cmin(mesh_idx, row, col);
            setMatrixValue(raw, row, col, val);
        }
    }

    assembleMatrix(raw);
    return matrix;
}

// ============================================================================
// AG_init 矩阵：初值模式下 A_ij 加权 Gram 矩阵（非线性扩散项，各向异性张量形式）
//   AG_init = integral_hatE g_I^T * (A_ij(xi,eta) / detJ * J^T J) * g_J d hat{x}
//   与 G_cmin 结构相同，只是把常数 c* 换成空间函数 A_ij(xi,eta)
// ============================================================================

double MSSolver::vemHdiv_gm_gm_ploy_AG_init(int mesh_idx,
                                             int i, int j,
                                             int ploy_mk1,
                                             int ploy_mk2) {
    const MeshData* mesh_data = getMeshData();
    if (mesh_idx < 0 || mesh_idx >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }

    Basis basis(getK());
    int dim = basis.getDimHdiv();
    if (ploy_mk1 < 0 || ploy_mk1 >= dim ||
        ploy_mk2 < 0 || ploy_mk2 >= dim) {
        throw std::out_of_range("H(div) polynomial index out of range");
    }

    const MaxwellStefan::MSPdeData& pde = problem_->get_pde_data();
    const MaxwellStefan::IsoparametricMapping& mapping =
        problem_->get_mapping();

    BasisPoint xD = makeBasisPoint(mesh_data->cell_centroid_x[mesh_idx],
                                   mesh_data->cell_centroid_y[mesh_idx]);
    double hD = mesh_data->cell_diameter[mesh_idx];

    vem::PolygonTriangulator triangulator;
    vem::TriangulatedElement element =
        triangulator.triangulateElement(*mesh_data, mesh_idx);
    vem::GaussQuadrature quadrature;

    double result = 0.0;
    for (const vem::Triangle& triangle : element.triangles) {
        std::vector<vem::QuadPoint> points = quadrature.get_triangle_points(
            triangle.vertices, getGaussPointNum2D());
        for (const vem::QuadPoint& point : points) {
            double xi = point.x[0];
            double eta = point.x[1];
            BasisPoint x = makeBasisPoint(xi, eta);

            // A_ij 在计算域上求值（初值模式）
            double Aij = pde.evaluate_A_comp_initial(i, j, xi, eta);

            // 几何修正张量 K_xi = (A_ij / detJ) * J^T * J
            MaxwellStefan::Tensor2D J = mapping.jacobian(xi, eta);
            double detJ = mapping.jacobian_det(xi, eta);
            double s = Aij / detJ;

            // nu = s * J^T * J 的四个分量（完整非对称形式）
            double nu_xx = s * (J.xx * J.xx + J.yx * J.yx);
            double nu_xy = s * (J.xx * J.xy + J.yx * J.yy);
            double nu_yx = s * (J.xy * J.xx + J.yy * J.yx);
            double nu_yy = s * (J.xy * J.xy + J.yy * J.yy);

            BasisVector first = basis.evalHdivBasis(ploy_mk1, xD, hD, x);
            BasisVector second = basis.evalHdivBasis(ploy_mk2, xD, hD, x);

            // g_I^T * nu * g_J 四项展开
            double val = nu_xx * first.x * second.x
                       + nu_xy * first.x * second.y
                       + nu_yx * first.y * second.x
                       + nu_yy * first.y * second.y;
            result += val * point.w;
        }
    }
    return result;
}

AutoPetscMat MSSolver::HdivMatrixAG_init(int mesh_idx, int i, int j) {
    const MeshData* mesh_data = getMeshData();
    if (mesh_idx < 0 || mesh_idx >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }

    int dim = Basis(getK()).getDimHdiv();
    AutoPetscMat matrix = vem::create_dense_matrix(dim, dim);
    Mat raw = vem::get_raw(matrix);

    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            double val = vemHdiv_gm_gm_ploy_AG_init(mesh_idx, i, j, row, col);
            setMatrixValue(raw, row, col, val);
        }
    }

    assembleMatrix(raw);
    return matrix;
}

// ============================================================================
// AG_poly 矩阵：数值解模式下 A_ij 加权 Gram 矩阵（非线性扩散项，各向异性张量形式）
//   AG_poly = integral_hatE g_I^T * (A_ij(u(xi,eta)) / detJ * J^T J) * g_J d hat{x}
//   与 AG_init 结构相同，只是 A_ij 通过浓度多项式系数还原后计算。
//   u_ploy_coeff: 所有组分的浓度单项式系数，按组分连续存储
//                 布局 = [comp0_monom0, ..., comp1_monom0, ...]
// ============================================================================

double MSSolver::vemHdiv_gm_gm_ploy_AG_poly(int mesh_idx,
                                             int i, int j,
                                             int ploy_mk1,
                                             int ploy_mk2,
                                             const std::vector<double>& u_ploy_coeff) {
    const MeshData* mesh_data = getMeshData();
    if (mesh_idx < 0 || mesh_idx >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }

    Basis basis(getK());
    int dim = basis.getDimHdiv();
    if (ploy_mk1 < 0 || ploy_mk1 >= dim ||
        ploy_mk2 < 0 || ploy_mk2 >= dim) {
        throw std::out_of_range("H(div) polynomial index out of range");
    }

    const MaxwellStefan::MSPdeData& pde = problem_->get_pde_data();
    const MaxwellStefan::IsoparametricMapping& mapping =
        problem_->get_mapping();

    double xD_x = mesh_data->cell_centroid_x[mesh_idx];
    double xD_y = mesh_data->cell_centroid_y[mesh_idx];
    BasisPoint xD = makeBasisPoint(xD_x, xD_y);
    double hD = mesh_data->cell_diameter[mesh_idx];

    vem::PolygonTriangulator triangulator;
    vem::TriangulatedElement element =
        triangulator.triangulateElement(*mesh_data, mesh_idx);
    vem::GaussQuadrature quadrature;

    double result = 0.0;
    for (const vem::Triangle& triangle : element.triangles) {
        std::vector<vem::QuadPoint> points = quadrature.get_triangle_points(
            triangle.vertices, getGaussPointNum2D());
        for (const vem::QuadPoint& point : points) {
            double xi = point.x[0];
            double eta = point.x[1];
            BasisPoint x = makeBasisPoint(xi, eta);

            // A_ij 由浓度多项式系数还原求值
            double Aij = pde.evaluate_A_comp_poly(i, j, u_ploy_coeff,
                                                  xD_x, xD_y, hD, xi, eta);

            // 几何修正张量 K_xi = (A_ij / detJ) * J^T * J
            MaxwellStefan::Tensor2D J = mapping.jacobian(xi, eta);
            double detJ = mapping.jacobian_det(xi, eta);
            double s = Aij / detJ;

            // nu = s * J^T * J 的四个分量（完整非对称形式）
            double nu_xx = s * (J.xx * J.xx + J.yx * J.yx);
            double nu_xy = s * (J.xx * J.xy + J.yx * J.yy);
            double nu_yx = s * (J.xy * J.xx + J.yy * J.yx);
            double nu_yy = s * (J.xy * J.xy + J.yy * J.yy);

            BasisVector first = basis.evalHdivBasis(ploy_mk1, xD, hD, x);
            BasisVector second = basis.evalHdivBasis(ploy_mk2, xD, hD, x);

            // g_I^T * nu * g_J 四项展开
            double val = nu_xx * first.x * second.x
                       + nu_xy * first.x * second.y
                       + nu_yx * first.y * second.x
                       + nu_yy * first.y * second.y;
            result += val * point.w;
        }
    }
    return result;
}

AutoPetscMat MSSolver::HdivMatrixAG_poly(int mesh_idx, int i, int j,
                                          const std::vector<double>& u_ploy_coeff) {
    const MeshData* mesh_data = getMeshData();
    if (mesh_idx < 0 || mesh_idx >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }

    int dim = Basis(getK()).getDimHdiv();
    AutoPetscMat matrix = vem::create_dense_matrix(dim, dim);
    Mat raw = vem::get_raw(matrix);

    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            double val = vemHdiv_gm_gm_ploy_AG_poly(mesh_idx, i, j,
                                                    row, col, u_ploy_coeff);
            setMatrixValue(raw, row, col, val);
        }
    }

    assembleMatrix(raw);
    return matrix;
}

// ============================================================================
// 稳定化系数
//
// 参考 Darcy 的 computeStabilizationScale：
//   alpha_E = || < W(x) > ||_F   其中 W 是加权张量，<·> 表示单元平均值
//
// MS 方程的加权张量 W = A_ij(x) * (J^T J / detJ)，含曲边几何修正。
// 线性部分（c* 项）和非线性部分（A_ij 项）分开计算，方便分别检查。
// ============================================================================

double MSSolver::computeStabilizationCoeffCmin(int mesh_idx) {
    const MeshData* mesh_data = getMeshData();
    if (mesh_idx < 0 || mesh_idx >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }

    double cstar = problem_->get_pde_data().c_star;
    const MaxwellStefan::IsoparametricMapping& mapping =
        problem_->get_mapping();

    vem::PolygonTriangulator triangulator;
    vem::TriangulatedElement element =
        triangulator.triangulateElement(*mesh_data, mesh_idx);
    double area = element.total_area;
    if (!std::isfinite(area) || area <= 0.0) {
        throw std::runtime_error("element area must be positive and finite");
    }
    vem::GaussQuadrature quadrature;

    // 积分加权张量 W = c* * J^T J / detJ 的四个分量
    double int_xx = 0.0, int_xy = 0.0, int_yx = 0.0, int_yy = 0.0;
    for (const vem::Triangle& triangle : element.triangles) {
        std::vector<vem::QuadPoint> points = quadrature.get_triangle_points(
            triangle.vertices, getGaussPointNum2D());
        for (const vem::QuadPoint& point : points) {
            double xi = point.x[0];
            double eta = point.x[1];

            MaxwellStefan::Tensor2D J = mapping.jacobian(xi, eta);
            double detJ = mapping.jacobian_det(xi, eta);
            double s = cstar / detJ;

            int_xx += s * (J.xx * J.xx + J.yx * J.yx) * point.w;
            int_xy += s * (J.xx * J.xy + J.yx * J.yy) * point.w;
            int_yx += s * (J.xy * J.xx + J.yy * J.yx) * point.w;
            int_yy += s * (J.xy * J.xy + J.yy * J.yy) * point.w;
        }
    }

    // 平均值的 Frobenius 范数
    double avg_xx = int_xx / area;
    double avg_xy = int_xy / area;
    double avg_yx = int_yx / area;
    double avg_yy = int_yy / area;
    double scale = std::sqrt(avg_xx * avg_xx + avg_xy * avg_xy +
                             avg_yx * avg_yx + avg_yy * avg_yy);
    if (!std::isfinite(scale)) {
        throw std::runtime_error(
            "stabilization coefficient (cmin) must be finite");
    }
    return scale;
}

double MSSolver::computeStabilizationCoeffInit(int mesh_idx, int i, int j) {
    const MeshData* mesh_data = getMeshData();
    if (mesh_idx < 0 || mesh_idx >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }

    const MaxwellStefan::MSPdeData& pde = problem_->get_pde_data();
    const MaxwellStefan::IsoparametricMapping& mapping =
        problem_->get_mapping();

    vem::PolygonTriangulator triangulator;
    vem::TriangulatedElement element =
        triangulator.triangulateElement(*mesh_data, mesh_idx);
    double area = element.total_area;
    if (!std::isfinite(area) || area <= 0.0) {
        throw std::runtime_error("element area must be positive and finite");
    }
    vem::GaussQuadrature quadrature;

    // 积分加权张量 W = A_ij(xi,eta) * J^T J / detJ 的四个分量
    double int_xx = 0.0, int_xy = 0.0, int_yx = 0.0, int_yy = 0.0;
    for (const vem::Triangle& triangle : element.triangles) {
        std::vector<vem::QuadPoint> points = quadrature.get_triangle_points(
            triangle.vertices, getGaussPointNum2D());
        for (const vem::QuadPoint& point : points) {
            double xi = point.x[0];
            double eta = point.x[1];

            double Aij = pde.evaluate_A_comp_initial(i, j, xi, eta);

            MaxwellStefan::Tensor2D J = mapping.jacobian(xi, eta);
            double detJ = mapping.jacobian_det(xi, eta);
            double s = Aij / detJ;

            int_xx += s * (J.xx * J.xx + J.yx * J.yx) * point.w;
            int_xy += s * (J.xx * J.xy + J.yx * J.yy) * point.w;
            int_yx += s * (J.xy * J.xx + J.yy * J.yx) * point.w;
            int_yy += s * (J.xy * J.xy + J.yy * J.yy) * point.w;
        }
    }

    double avg_xx = int_xx / area;
    double avg_xy = int_xy / area;
    double avg_yx = int_yx / area;
    double avg_yy = int_yy / area;
    double scale = std::sqrt(avg_xx * avg_xx + avg_xy * avg_xy +
                             avg_yx * avg_yx + avg_yy * avg_yy);
    if (!std::isfinite(scale)) {
        throw std::runtime_error(
            "stabilization coefficient (init) must be finite");
    }
    return scale;
}

double MSSolver::computeStabilizationCoeffPoly(
    int mesh_idx, int i, int j,
    const std::vector<double>& u_ploy_coeff) {
    const MeshData* mesh_data = getMeshData();
    if (mesh_idx < 0 || mesh_idx >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }

    const MaxwellStefan::MSPdeData& pde = problem_->get_pde_data();
    const MaxwellStefan::IsoparametricMapping& mapping =
        problem_->get_mapping();

    double xD_x = mesh_data->cell_centroid_x[mesh_idx];
    double xD_y = mesh_data->cell_centroid_y[mesh_idx];
    double hD = mesh_data->cell_diameter[mesh_idx];

    vem::PolygonTriangulator triangulator;
    vem::TriangulatedElement element =
        triangulator.triangulateElement(*mesh_data, mesh_idx);
    double area = element.total_area;
    if (!std::isfinite(area) || area <= 0.0) {
        throw std::runtime_error("element area must be positive and finite");
    }
    vem::GaussQuadrature quadrature;

    // 积分加权张量 W = A_ij(u(xi,eta)) * J^T J / detJ 的四个分量
    double int_xx = 0.0, int_xy = 0.0, int_yx = 0.0, int_yy = 0.0;
    for (const vem::Triangle& triangle : element.triangles) {
        std::vector<vem::QuadPoint> points = quadrature.get_triangle_points(
            triangle.vertices, getGaussPointNum2D());
        for (const vem::QuadPoint& point : points) {
            double xi = point.x[0];
            double eta = point.x[1];

            double Aij = pde.evaluate_A_comp_poly(i, j, u_ploy_coeff,
                                                  xD_x, xD_y, hD, xi, eta);

            MaxwellStefan::Tensor2D J = mapping.jacobian(xi, eta);
            double detJ = mapping.jacobian_det(xi, eta);
            double s = Aij / detJ;

            int_xx += s * (J.xx * J.xx + J.yx * J.yx) * point.w;
            int_xy += s * (J.xx * J.xy + J.yx * J.yy) * point.w;
            int_yx += s * (J.xy * J.xx + J.yy * J.yx) * point.w;
            int_yy += s * (J.xy * J.xy + J.yy * J.yy) * point.w;
        }
    }

    double avg_xx = int_xx / area;
    double avg_xy = int_xy / area;
    double avg_yx = int_yx / area;
    double avg_yy = int_yy / area;
    double scale = std::sqrt(avg_xx * avg_xx + avg_xy * avg_xy +
                             avg_yx * avg_yx + avg_yy * avg_yy);
    if (!std::isfinite(scale)) {
        throw std::runtime_error(
            "stabilization coefficient (poly) must be finite");
    }
    return scale;
}

// ============================================================================
// K_ac 一致性矩阵
//   K_ac = Π_poly^T * G * Π_poly
//   G 可以是任意加权 Gram 矩阵（G_cmin / AG_init / AG_poly），
//   传入不同的 G 就得到对应部分的一致性刚度矩阵。
// ============================================================================

AutoPetscMat MSSolver::HdivMatrixK_ac(const AutoPetscMat& G,
                                       const AutoPetscMat& L2Proj_ploy) {
    if (!G || vem::get_raw(G) == PETSC_NULLPTR) {
        throw std::invalid_argument("K_ac: G matrix is null");
    }
    if (!L2Proj_ploy || vem::get_raw(L2Proj_ploy) == PETSC_NULLPTR) {
        throw std::invalid_argument("K_ac: L2Proj_ploy matrix is null");
    }

    // K_ac = Π_poly^T * (G * Π_poly)
    AutoPetscMat temp = vem::multiply_matrices(G, L2Proj_ploy);
    AutoPetscMat L2Proj_T = vem::transpose_matrix(L2Proj_ploy);
    AutoPetscMat K_ac = vem::multiply_matrices(L2Proj_T, temp);
    return K_ac;
}

// ============================================================================
// K_as 稳定化矩阵
//   K_as = alpha_E * (I - Π_basis)^T * (I - Π_basis)
//   L2Proj_basis 是从自由度到单项式基的 L2 投影矩阵（方阵，通量自由度阶数）
//   coeff 是稳定化系数 alpha_E
// ============================================================================

AutoPetscMat MSSolver::HdivMatrixK_as(const AutoPetscMat& L2Proj_basis,
                                       double coeff) {
    if (!L2Proj_basis || vem::get_raw(L2Proj_basis) == PETSC_NULLPTR) {
        throw std::invalid_argument("K_as: L2Proj_basis matrix is null");
    }

    PetscInt rows = 0, cols = 0;
    Mat raw_proj = vem::get_raw(L2Proj_basis);
    PetscErrorCode ierr = MatGetSize(raw_proj, &rows, &cols);
    if (ierr != 0 || rows != cols) {
        throw std::invalid_argument("K_as: L2Proj_basis must be square");
    }

    // I - Π
    AutoPetscMat I_minus_Pi = vem::scale_matrix(L2Proj_basis, -1.0);
    Mat raw = vem::get_raw(I_minus_Pi);
    ierr = MatShift(raw, 1.0);
    if (ierr != 0) {
        throw std::runtime_error("K_as: MatShift failed");
    }
    // MatShift 会改变矩阵结构，需要重新组装
    MatAssemblyBegin(raw, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(raw, MAT_FINAL_ASSEMBLY);

    // (I - Π)^T * (I - Π)
    AutoPetscMat I_minus_Pi_T = vem::transpose_matrix(I_minus_Pi);
    AutoPetscMat K_as = vem::multiply_matrices(I_minus_Pi_T, I_minus_Pi);

    // 乘以稳定化系数 alpha_E
    return vem::scale_matrix(K_as, coeff);
}

// ============================================================================
// 曲边 H 矩阵（物理域质量矩阵）
//   H_ij = ∫_E m_i(x) m_j(x) dx = ∫_hatE m_i(ξ) m_j(ξ) * detJ(ξ) dξdη
//   仅用于时间项 H_t = H_curved / Δt
// ============================================================================

AutoPetscMat MSSolver::HdivMatrixH_curved(int mesh_idx) {
    const MeshData* mesh_data = getMeshData();
    if (mesh_idx < 0 || mesh_idx >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    Basis basis(getK());
    int dim_conc = basis.getDimMonomial(2);
    AutoPetscMat H = vem::create_dense_matrix(dim_conc, dim_conc);
    Mat raw = vem::get_raw(H);
    MatZeroEntries(raw);

    const MaxwellStefan::IsoparametricMapping& mapping =
        problem_->get_mapping();

    BasisPoint xD = makeBasisPoint(mesh_data->cell_centroid_x[mesh_idx],
                                   mesh_data->cell_centroid_y[mesh_idx]);
    double hD = mesh_data->cell_diameter[mesh_idx];

    vem::PolygonTriangulator triangulator;
    vem::TriangulatedElement element =
        triangulator.triangulateElement(*mesh_data, mesh_idx);
    vem::GaussQuadrature quadrature;

    // 预计算每个高斯点的基函数值和 detJ
    for (int i = 0; i < dim_conc; ++i) {
        for (int j = 0; j < dim_conc; ++j) {
            double val = 0.0;
            for (const vem::Triangle& triangle : element.triangles) {
                std::vector<vem::QuadPoint> points =
                    quadrature.get_triangle_points(
                        triangle.vertices, getGaussPointNum2D());
                for (const vem::QuadPoint& point : points) {
                    double xi = point.x[0];
                    double eta = point.x[1];
                    BasisPoint x = makeBasisPoint(xi, eta);
                    double detJ = mapping.jacobian_det(xi, eta);
                    double mi = basis.evalMonomial2D(i, xD, hD, x);
                    double mj = basis.evalMonomial2D(j, xD, hD, x);
                    val += mi * mj * detJ * point.w;
                }
            }
            setMatrixValue(raw, i, j, val);
        }
    }
    assembleMatrix(raw);
    return H;
}

// ============================================================================
// 右端项：vemHdiv_uh_mk_F
//   计算右端项的单个积分值 = ∫_E [f_i(x) + u_prev(x)/Δt] * m_k(x) dx
//   变换到计算域：∫_hatE [f_i_hat(ξ,η) + u_prev(ξ,η)/Δt * detJ] * m_k(ξ,η) dξ dη
//   其中 f_i_hat = detJ * f_i(x(ξ,η)) 是计算域源项（已含 detJ）。
//
//   is_initial_first_tstep = true:  第一个时间步，u_prev 用初值函数
//   is_initial_first_tstep = false: 后续时间步，u_prev 用上一时间步的多项式系数
// ============================================================================

double MSSolver::vemHdiv_uh_mk_F(bool is_initial_first_tstep,
                                  int mesh_idx,
                                  int comp_idx,
                                  int ploy_mk) {
    const MeshData* mesh_data = getMeshData();
    if (mesh_idx < 0 || mesh_idx >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    if (comp_idx < 0 || comp_idx >= n_comp_) {
        throw std::out_of_range("component index out of range");
    }
    Basis basis(getK());
    int dim_conc = basis.getDimMonomial(2);
    if (ploy_mk < 0 || ploy_mk >= dim_conc) {
        throw std::out_of_range("concentration monomial index out of range");
    }

    const MaxwellStefan::MSPdeData& pde = problem_->get_pde_data();
    const MaxwellStefan::IsoparametricMapping& mapping =
        problem_->get_mapping();

    BasisPoint xD = makeBasisPoint(mesh_data->cell_centroid_x[mesh_idx],
                                   mesh_data->cell_centroid_y[mesh_idx]);
    double hD = mesh_data->cell_diameter[mesh_idx];

    vem::PolygonTriangulator triangulator;
    vem::TriangulatedElement element =
        triangulator.triangulateElement(*mesh_data, mesh_idx);
    vem::GaussQuadrature quadrature;

    // 上一时间步浓度的多项式系数（非初始步才需要）
    std::vector<double> u_prev_coeff;
    if (!is_initial_first_tstep) {
        u_prev_coeff = getElementComponentConcentration(
            mesh_idx, comp_idx, true);  // true = solution_prev_time_
    }

    // BDF2 标记（第 2 步及以后的 BDF2 时间步）
    bool use_bdf2_step = use_bdf2_ && !is_initial_first_tstep;

    // BDF2: 提取 u^{n-2} 的多项式系数
    //   - 第 2 步（n=2）：u^{n-2}=u^0 用初值函数（最精确，无需投影）
    //   - 第 3 步及以后：从 solution_prev_prev_time_ 提取
    bool bdf2_prev_prev_from_vector = false;
    std::vector<double> u_prev_prev_coeff;
    if (use_bdf2_step && vem::get_raw(solution_prev_prev_time_) != nullptr) {
        u_prev_prev_coeff = extractElementConcentration(
            mesh_idx, comp_idx, solution_prev_prev_time_);
        bdf2_prev_prev_from_vector = true;
    }

    double result = 0.0;
    for (const vem::Triangle& triangle : element.triangles) {
        std::vector<vem::QuadPoint> points = quadrature.get_triangle_points(
            triangle.vertices, getGaussPointNum2D());
        for (const vem::QuadPoint& point : points) {
            double xi = point.x[0];
            double eta = point.x[1];
            BasisPoint x = makeBasisPoint(xi, eta);
            double detJ = mapping.jacobian_det(xi, eta);

            // 1. 源项（以计算域坐标定义的函数，积分时需乘 detJ）
            //    源项使用当前时刻 current_time_ 求值
            double f_val = pde.source_f_comp(comp_idx, xi, eta, current_time_) * detJ;

            // 2. 时间历史项
            //    BDF1: u^{n-1} / Δt
            //    BDF2: (4 u^{n-1} - u^{n-2}) / (2 Δt)
            double u_val = 0.0;
            if (is_initial_first_tstep) {
                // 初始步：用初值函数（计算域坐标下求值），BDF1 形式
                u_val = pde.initial_concentration_comp(comp_idx, xi, eta);
                u_val = u_val * detJ / delta_t_;
            } else if (use_bdf2_step) {
                // BDF2: (4 u^{n-1} - u^{n-2}) / (2 Δt)
                double u_prev_val = 0.0;
                double u_prev_prev_val = 0.0;
                for (int l = 0; l < dim_conc; ++l) {
                    double basis_val = basis.evalMonomial2D(l, xD, hD, x);
                    u_prev_val += u_prev_coeff[l] * basis_val;
                }
                if (bdf2_prev_prev_from_vector) {
                    // 第 3 步及以后：从多项式系数重构 u^{n-2}
                    for (int l = 0; l < dim_conc; ++l) {
                        double basis_val = basis.evalMonomial2D(l, xD, hD, x);
                        u_prev_prev_val += u_prev_prev_coeff[l] * basis_val;
                    }
                } else {
                    // 第 2 步：u^{n-2} = u^0，直接用初值函数（精确）
                    u_prev_prev_val = pde.initial_concentration_comp(
                        comp_idx, xi, eta);
                }
                u_val = (4.0 * u_prev_val - u_prev_prev_val) * detJ
                      / (2.0 * delta_t_);
            } else {
                // BDF1: u^{n-1} / Δt
                u_val = 0.0;
                for (int l = 0; l < dim_conc; ++l) {
                    double basis_val = basis.evalMonomial2D(l, xD, hD, x);
                    u_val += u_prev_coeff[l] * basis_val;
                }
                u_val = u_val * detJ / delta_t_;
            }

            // 测试函数（单项式）的值
            double basis_mk = basis.evalMonomial2D(ploy_mk, xD, hD, x);

            result += (f_val + u_val) * basis_mk * point.w;
        }
    }
    return result;
}

// ============================================================================
// HdivVecRhs：生成单元右端项向量
//   - 通量部分全部为 0
//   - 浓度部分 = vemHdiv_uh_mk_F 的结果
//   布局（局部向量）：
//     [ 通量部分（全0） | 浓度部分（按组分连续） ]
//     通量内部按组分连续，每个组分内边+泡
// ============================================================================

AutoPetscVec MSSolver::HdivVecRhs(int mesh_idx, bool is_initial_first_tstep) {
    const MeshData* mesh_data = getMeshData();
    if (mesh_idx < 0 || mesh_idx >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }

    int dof_flux_per_comp = getDofPerElementFlux(mesh_idx);
    int dof_conc_per_comp = getDofPerElementConc();
    int total_flux_dof = dof_flux_per_comp * n_comp_;
    int total_conc_dof = dof_conc_per_comp * n_comp_;
    int local_size = total_flux_dof + total_conc_dof;

    AutoPetscVec vecF = vem::create_vector(local_size);
    Vec rawVec = vem::get_raw(vecF);
    VecZeroEntries(rawVec);

    // 通量部分全部为 0，不需要设置

    // 浓度部分
    for (int c = 0; c < n_comp_; ++c) {
        int conc_base = total_flux_dof + c * dof_conc_per_comp;
        for (int mk = 0; mk < dof_conc_per_comp; ++mk) {
            double val = vemHdiv_uh_mk_F(is_initial_first_tstep,
                                          mesh_idx, c, mk);
            PetscInt idx = conc_base + mk;
            VecSetValue(rawVec, idx, val, INSERT_VALUES);
        }
    }

    VecAssemblyBegin(rawVec);
    VecAssemblyEnd(rawVec);
    return vecF;
}

// ============================================================================
// 方向调整向量
//   H(div) 元的边自由度（法向通量矩）需要与全局边方向一致。
//   每个单元按逆时针定义边，相邻单元在公共边上方向相反。
//   规则：偶次矩自由度方向与边方向相关，需要调整符号；
//         奇次矩自由度因 (-1)^l * (-1) = (-1)^{l+1} 抵消，不需调整。
//   边界边只有一个单元，沿用该单元方向，不需调整。
//
//   返回的方向向量长度 = 单元总自由度（通量+浓度，所有组分）。
//   浓度自由度方向恒为 1.0。
// ============================================================================

std::vector<double> MSSolver::getDirectionVector(int elem) const {
    return computeDirectionVector(elem);
}

std::vector<PetscInt> MSSolver::getLocalToGlobalMap(int elem) const {
    return getLocalToGlobalDofs(elem);
}

std::vector<double> MSSolver::computeDirectionVector(int elem) const {
    const MeshData* mesh_data = getMeshData();
    if (elem < 0 || elem >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }

    int k = getK();
    int edge_count = mesh_data->nodes_per_cell[elem];
    int dof_flux_per_comp = getDofPerElementFlux(elem);
    int dof_conc_per_comp = getDofPerElementConc();
    int total_flux_dof = dof_flux_per_comp * n_comp_;
    int total_conc_dof = dof_conc_per_comp * n_comp_;
    int local_size = total_flux_dof + total_conc_dof;

    std::vector<double> directions(local_size, 1.0);

    int cell_node_start = mesh_data->cell_node_indices[elem];
    int cell_edge_start = mesh_data->cell_edge_indices[elem];

    // 先计算单组分通量自由度的方向
    std::vector<double> single_comp_flux_dir(dof_flux_per_comp, 1.0);

    for (int degree = 0; degree <= k; ++degree) {
        for (int local_edge = 0; local_edge < edge_count; ++local_edge) {
            int local_dof = degree * edge_count + local_edge;
            int global_edge =
                mesh_data->global_edges[cell_edge_start + local_edge];

            // 边界边只有一个单元，方向沿用该单元约定，不需调整
            if (mesh_data->edge_occurrence[global_edge] == 1) {
                continue;
            }

            // 内部边：比较局部边端点与全局边端点顺序
            int local_node0 =
                mesh_data->cell_nodes[cell_node_start + local_edge];
            int local_node1 = mesh_data->cell_nodes[
                cell_node_start + (local_edge + 1) % edge_count];
            int global_node0 = mesh_data->edge_endpoints[2 * global_edge];
            int global_node1 = mesh_data->edge_endpoints[2 * global_edge + 1];

            bool same_orientation =
                (local_node0 == global_node0 && local_node1 == global_node1);
            bool reverse_orientation =
                (local_node0 == global_node1 && local_node1 == global_node0);
            if (!same_orientation && !reverse_orientation) {
                throw std::runtime_error(
                    "computeDirectionVector: local edge endpoints "
                    "do not match global edge");
            }

            // 反向边 + 偶次矩 → 方向翻转
            // （法向变号一次，l 次单项式乘 (-1)^l，偶次时整体变号）
            if (reverse_orientation && degree % 2 == 0) {
                single_comp_flux_dir[local_dof] = -1.0;
            }
        }
    }

    // 内部通量自由度（泡函数）方向恒为 1.0，已初始化

    // 复制到所有组分（通量按组分连续排列）
    for (int c = 0; c < n_comp_; ++c) {
        int flux_base = c * dof_flux_per_comp;
        for (int j = 0; j < dof_flux_per_comp; ++j) {
            directions[flux_base + j] = single_comp_flux_dir[j];
        }
    }

    // 浓度自由度方向恒为 1.0，已初始化

    return directions;
}

std::vector<double> MSSolver::computeAndApplyDirectionAdjustment(
    int elem, Mat matrix) const {
    if (matrix == PETSC_NULLPTR) {
        throw std::invalid_argument("local matrix is null");
    }

    std::vector<double> directions = computeDirectionVector(elem);
    PetscInt local_size = static_cast<PetscInt>(directions.size());

    // 取出所有矩阵值
    std::vector<PetscInt> indices(local_size);
    for (PetscInt i = 0; i < local_size; ++i) {
        indices[i] = i;
    }
    std::vector<PetscScalar> values(
        static_cast<std::size_t>(local_size * local_size));
    PetscErrorCode ierr = MatGetValues(
        matrix, local_size, indices.data(),
        local_size, indices.data(), values.data());
    if (ierr != 0) {
        throw std::runtime_error(
            "computeAndApplyDirectionAdjustment: MatGetValues failed");
    }

    // 行列方向相乘
    for (PetscInt i = 0; i < local_size; ++i) {
        double row_dir = directions[i];
        for (PetscInt j = 0; j < local_size; ++j) {
            double col_dir = directions[j];
            PetscInt idx = i * local_size + j;
            values[idx] *= row_dir * col_dir;
        }
    }

    // 写回矩阵
    ierr = MatSetValues(
        matrix, local_size, indices.data(),
        local_size, indices.data(), values.data(), INSERT_VALUES);
    if (ierr != 0) {
        throw std::runtime_error(
            "computeAndApplyDirectionAdjustment: MatSetValues failed");
    }
    MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY);

    return directions;
}

// ============================================================================
// 局部-全局自由度映射
//   布局：先所有组分通量，再所有组分浓度。
//   同类型内按组分连续排列。
// ============================================================================

std::vector<PetscInt> MSSolver::getLocalToGlobalDofs(int elem) const {
    const MeshData* mesh_data = getMeshData();
    if (elem < 0 || elem >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }

    int k = getK();
    int edge_count = mesh_data->nodes_per_cell[elem];
    int dof_flux_per_comp = getDofPerElementFlux(elem);
    int dof_conc_per_comp = getDofPerElementConc();
    int total_flux_dof = dof_flux_per_comp * n_comp_;
    int total_conc_dof = dof_conc_per_comp * n_comp_;
    int local_size = total_flux_dof + total_conc_dof;

    std::vector<PetscInt> global_dofs(local_size, -1);
    int cell_edge_start = mesh_data->cell_edge_indices[elem];
    int internal_dofs = getDimGradMk() + getDimCurlMk();

    int flux_per_comp_global = getTotalDofFlux();   // 单组分全局通量自由度
    int conc_per_comp_global = getTotalDofConc();   // 单组分全局浓度自由度

    for (int c = 0; c < n_comp_; ++c) {
        int comp_flux_base = c * flux_per_comp_global;
        int comp_conc_base = total_dof_flux_all_ + c * conc_per_comp_global;

        // 边通量自由度（按次数 × 边号排列）
        for (int degree = 0; degree <= k; ++degree) {
            for (int local_edge = 0; local_edge < edge_count; ++local_edge) {
                int local_dof = c * dof_flux_per_comp
                              + degree * edge_count + local_edge;
                int global_edge =
                    mesh_data->global_edges[cell_edge_start + local_edge];
                global_dofs[local_dof] =
                    comp_flux_base + degree * mesh_data->num_edges + global_edge;
            }
        }

        // 内部通量自由度（按单元连续）
        int local_internal_start =
            c * dof_flux_per_comp + (k + 1) * edge_count;
        int global_internal_start = comp_flux_base
                                  + mesh_data->num_edges * (k + 1)
                                  + elem * internal_dofs;
        for (int d = 0; d < internal_dofs; ++d) {
            global_dofs[local_internal_start + d] =
                global_internal_start + d;
        }

        // 浓度自由度（按单元连续）
        int local_conc_start = total_flux_dof + c * dof_conc_per_comp;
        int global_conc_start = comp_conc_base + elem * dof_conc_per_comp;
        for (int d = 0; d < dof_conc_per_comp; ++d) {
            global_dofs[local_conc_start + d] = global_conc_start + d;
        }
    }

    return global_dofs;
}

// ============================================================================
// 局部矩阵组装到全局
// ============================================================================

void MSSolver::assembleLocalToGlobal(int elem,
                                      const AutoPetscMat& local_matrix) {
    if (!local_matrix || vem::get_raw(local_matrix) == PETSC_NULLPTR) {
        throw std::invalid_argument("assembleLocalToGlobal: local matrix is null");
    }

    std::vector<PetscInt> global_dofs = getLocalToGlobalDofs(elem);
    PetscInt local_dofs = static_cast<PetscInt>(global_dofs.size());

    // 取出局部矩阵值
    std::vector<PetscInt> indices(local_dofs);
    for (PetscInt i = 0; i < local_dofs; ++i) indices[i] = i;
    std::vector<PetscScalar> values(
        static_cast<std::size_t>(local_dofs * local_dofs));
    Mat raw_local = vem::get_raw(local_matrix);
    PetscErrorCode ierr = MatGetValues(
        raw_local, local_dofs, indices.data(),
        local_dofs, indices.data(), values.data());
    if (ierr != 0) {
        throw std::runtime_error("assembleLocalToGlobal: MatGetValues failed");
    }

    // 加到全局矩阵
    Mat raw_global = vem::get_raw(system_stiffness_mat_);
    ierr = MatSetValues(
        raw_global, local_dofs, global_dofs.data(),
        local_dofs, global_dofs.data(), values.data(), ADD_VALUES);
    if (ierr != 0) {
        throw std::runtime_error("assembleLocalToGlobal: MatSetValues failed");
    }
}

// ============================================================================
// 局部右端项组装到全局
// ============================================================================

void MSSolver::assembleLocalRhsToGlobal(
    int elem, const AutoPetscVec& local_rhs) {
    if (!local_rhs || vem::get_raw(local_rhs) == PETSC_NULLPTR) {
        throw std::invalid_argument(
            "assembleLocalRhsToGlobal: local rhs is null");
    }

    std::vector<PetscInt> global_dofs = getLocalToGlobalDofs(elem);
    PetscInt local_dofs = static_cast<PetscInt>(global_dofs.size());

    std::vector<PetscInt> local_indices(local_dofs);
    for (PetscInt i = 0; i < local_dofs; ++i) local_indices[i] = i;
    std::vector<PetscScalar> values(static_cast<std::size_t>(local_dofs));

    Vec raw_local = vem::get_raw(local_rhs);
    PetscErrorCode ierr = VecGetValues(
        raw_local, local_dofs, local_indices.data(), values.data());
    if (ierr != 0) {
        throw std::runtime_error(
            "assembleLocalRhsToGlobal: VecGetValues failed");
    }

    Vec raw_global = vem::get_raw(system_rhs_vec_);
    ierr = VecSetValues(
        raw_global, local_dofs, global_dofs.data(), values.data(), ADD_VALUES);
    if (ierr != 0) {
        throw std::runtime_error(
            "assembleLocalRhsToGlobal: VecSetValues failed");
    }
}

// ============================================================================
// 浓度多项式系数提取
//
// 全局浓度自由度布局：
//   先所有组分的通量自由度（total_dof_flux_all_ 个），
//   再所有组分的浓度自由度（total_dof_conc_all_ 个）。
//   同类型内按组分连续排列：comp_0 全部浓度 + comp_1 全部浓度 + ...
//   单组分内按单元连续排列：elem_0 的 P_k 系数 + elem_1 的 P_k 系数 + ...
//   每个单元的浓度自由度顺序与 P_k 单项式基顺序一致。
// ============================================================================

std::vector<double> MSSolver::getElementComponentConcentration(
    int element_id, int comp_id, bool use_prev_time) const {
    if (!initialized_) {
        throw std::runtime_error(
            "getElementComponentConcentration: solver not initialized");
    }
    const MeshData* mesh_data = getMeshData();
    if (element_id < 0 || element_id >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    if (comp_id < 0 || comp_id >= n_comp_) {
        throw std::out_of_range("component index out of range");
    }

    int dof_conc = getDofPerElementConc();
    int total_conc_per_comp = getTotalDofConc();

    // 全局偏移：通量块 + 组分偏移 + 单元偏移
    PetscInt global_start = total_dof_flux_all_
                          + comp_id * total_conc_per_comp
                          + element_id * dof_conc;

    const AutoPetscVec& sol = use_prev_time ? solution_prev_time_
                                            : solution_picard_;
    Vec raw_sol = vem::get_raw(sol);

    std::vector<double> coeffs(dof_conc);
    for (int i = 0; i < dof_conc; ++i) {
        PetscScalar val;
        PetscInt idx = global_start + i;
        VecGetValues(raw_sol, 1, &idx, &val);
        coeffs[i] = PetscRealPart(val);
    }
    return coeffs;
}

std::vector<double> MSSolver::getElementComponentConcentration_allcomp(
    int element_id, bool use_prev_time) const {
    if (!initialized_) {
        throw std::runtime_error(
            "getElementComponentConcentration_allcomp: solver not initialized");
    }
    const MeshData* mesh_data = getMeshData();
    if (element_id < 0 || element_id >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }

    int dof_conc = getDofPerElementConc();
    int total_conc_per_comp = getTotalDofConc();

    const AutoPetscVec& sol = use_prev_time ? solution_prev_time_
                                            : solution_picard_;
    Vec raw_sol = vem::get_raw(sol);

    // 按组分连续：[comp0_monom0,...,comp1_monom0,...]
    std::vector<double> coeffs(dof_conc * n_comp_);
    for (int c = 0; c < n_comp_; ++c) {
        PetscInt global_start = total_dof_flux_all_
                              + c * total_conc_per_comp
                              + element_id * dof_conc;
        for (int i = 0; i < dof_conc; ++i) {
            PetscScalar val;
            PetscInt idx = global_start + i;
            VecGetValues(raw_sol, 1, &idx, &val);
            coeffs[c * dof_conc + i] = PetscRealPart(val);
        }
    }
    return coeffs;
}

// ============================================================================
// 通用浓度提取：从任意指定解向量中提取单组分浓度多项式系数
// ============================================================================

std::vector<double> MSSolver::extractElementConcentration(
    int element_id, int comp_id, const AutoPetscVec& sol) const {
    if (!initialized_) {
        throw std::runtime_error(
            "extractElementConcentration: solver not initialized");
    }
    const MeshData* mesh_data = getMeshData();
    if (element_id < 0 || element_id >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    if (comp_id < 0 || comp_id >= n_comp_) {
        throw std::out_of_range("component index out of range");
    }
    Vec raw_sol = vem::get_raw(sol);
    if (raw_sol == nullptr) {
        throw std::invalid_argument("extractElementConcentration: sol is null");
    }

    int dof_conc = getDofPerElementConc();
    int total_conc_per_comp = getTotalDofConc();

    PetscInt global_start = total_dof_flux_all_
                          + comp_id * total_conc_per_comp
                          + element_id * dof_conc;

    std::vector<double> coeffs(dof_conc);
    for (int i = 0; i < dof_conc; ++i) {
        PetscScalar val;
        PetscInt idx = global_start + i;
        VecGetValues(raw_sol, 1, &idx, &val);
        coeffs[i] = PetscRealPart(val);
    }
    return coeffs;
}

// ============================================================================
// 通用浓度提取：从任意指定解向量中提取所有组分浓度多项式系数
// ============================================================================

std::vector<double> MSSolver::extractElementConcentration_allcomp(
    int element_id, const AutoPetscVec& sol) const {
    if (!initialized_) {
        throw std::runtime_error(
            "extractElementConcentration_allcomp: solver not initialized");
    }
    const MeshData* mesh_data = getMeshData();
    if (element_id < 0 || element_id >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    Vec raw_sol = vem::get_raw(sol);
    if (raw_sol == nullptr) {
        throw std::invalid_argument(
            "extractElementConcentration_allcomp: sol is null");
    }

    int dof_conc = getDofPerElementConc();
    int total_conc_per_comp = getTotalDofConc();

    std::vector<double> coeffs(dof_conc * n_comp_);
    for (int c = 0; c < n_comp_; ++c) {
        PetscInt global_start = total_dof_flux_all_
                              + c * total_conc_per_comp
                              + element_id * dof_conc;
        for (int i = 0; i < dof_conc; ++i) {
            PetscScalar val;
            PetscInt idx = global_start + i;
            VecGetValues(raw_sol, 1, &idx, &val);
            coeffs[c * dof_conc + i] = PetscRealPart(val);
        }
    }
    return coeffs;
}

// ============================================================================
// 保存解向量到文件
//   subfolder: 子目录名，如 "solver_data"（每步保存）或 "solver_data_final"（仅最终步）
//   完整路径：data/ms_data/{subfolder}/
//   文件名：{单元总数}_{时刻值}.txt
//   格式：第一行总自由度数，随后每行一个自由度值（科学计数法，12位小数）
// ============================================================================

void MSSolver::saveSolutionToFile(double time,
                                   const std::string& subfolder) const {
    if (!solved_) {
        std::cerr << "saveSolutionToFile: 未解出，跳过保存" << std::endl;
        return;
    }

    const MeshData* mesh_data = getMeshData();
    int total_elem = mesh_data->num_cells;

    // 确保目录存在
    std::string dir = "data/ms_data/" + subfolder + "/";
    std::string mkdir_cmd = "mkdir -p " + dir;
    int ret = system(mkdir_cmd.c_str());
    (void)ret;

    // 构造文件名：{单元总数}_{时刻值}.txt
    char filename[512];
    snprintf(filename, sizeof(filename), "%s%d_%.6f.txt",
             dir.c_str(), total_elem, time);

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    file << std::scientific << std::setprecision(12);

    // 获取解向量数据
    Vec sol = vem::get_raw(solution_);
    PetscInt size = 0;
    VecGetSize(sol, &size);
    const PetscScalar* sol_array = nullptr;
    VecGetArrayRead(sol, &sol_array);

    // 第一行：总自由度数
    file << size << "\n";
    // 每行一个自由度值
    for (PetscInt i = 0; i < size; ++i) {
        file << PetscRealPart(sol_array[i]) << "\n";
    }

    VecRestoreArrayRead(sol, &sol_array);
    file.close();
    std::cout << "  解已保存: " << filename << std::endl;
}

// ============================================================================
// 误差
// ============================================================================

std::vector<double> MSSolver::computeConcentrationL2Error(int gauss_point_num) {
    if (!solved_ || !solution_ || vem::get_raw(solution_) == PETSC_NULLPTR) {
        throw std::runtime_error(
            "concentration L2 error requires a completed numerical solution");
    }

    const MaxwellStefan::MSPdeData& pde = problem_->get_pde_data();
    const MaxwellStefan::IsoparametricMapping& mapping =
        problem_->get_mapping();
    const MeshData* mesh_data = getMeshData();
    Basis basis(getK());
    int num_cells = mesh_data->num_cells;
    int dof_conc = getDofPerElementConc();
    int total_conc_per_comp = getTotalDofConc();

    Vec sol = vem::get_raw(solution_);

    // 第一步：提取所有组分所有单元的浓度多项式系数
    // conc_coeffs[c][elem * dof_conc + m] = 第 c 组分第 elem 单元第 m 个单项式系数
    std::vector<std::vector<double>> conc_coeffs(
        n_comp_, std::vector<double>(num_cells * dof_conc, 0.0));

    for (int c = 0; c < n_comp_; ++c) {
        PetscInt comp_base = total_dof_flux_all_ + c * total_conc_per_comp;
        for (int elem = 0; elem < num_cells; ++elem) {
            PetscInt elem_base = comp_base + elem * dof_conc;
            for (int m = 0; m < dof_conc; ++m) {
                PetscInt idx = elem_base + m;
                PetscScalar val;
                VecGetValues(sol, 1, &idx, &val);
                conc_coeffs[c][elem * dof_conc + m] = PetscRealPart(val);
            }
        }
    }

    // 第二步：逐单元积分误差和真解范数（计算域 L2 范数，不乘 detJ）
    std::vector<double> error_squared(n_comp_, 0.0);
    std::vector<double> exact_squared(n_comp_, 0.0);
    vem::GaussQuadrature quadrature;
    quadrature.get_triangle_reference_points(gauss_point_num);
    vem::PolygonTriangulator triangulator;

    for (int elem = 0; elem < num_cells; ++elem) {
        BasisPoint center(mesh_data->cell_centroid_x[elem],
                          mesh_data->cell_centroid_y[elem]);
        double diameter = mesh_data->cell_diameter[elem];

        vem::TriangulatedElement element =
            triangulator.triangulateElement(*mesh_data, elem);

        for (const vem::Triangle& triangle : element.triangles) {
            std::vector<vem::QuadPoint> points =
                quadrature.get_triangle_points(triangle.vertices,
                                               gauss_point_num);

            for (const vem::QuadPoint& point : points) {
                double xi = point.x[0];
                double eta = point.x[1];
                BasisPoint qp(xi, eta);

                // 先计算所有单项式在该高斯点的值（所有组分共享）
                std::vector<double> monom_vals(dof_conc);
                for (int m = 0; m < dof_conc; ++m) {
                    monom_vals[m] = basis.evalMonomial2D(
                        m, center, diameter, qp);
                }

                // 对每个组分计算数值浓度并与真解比较（计算域 L2）
                for (int c = 0; c < n_comp_; ++c) {
                    double conc_num = 0.0;
                    const double* coeff = &conc_coeffs[c][elem * dof_conc];
                    for (int m = 0; m < dof_conc; ++m) {
                        conc_num += coeff[m] * monom_vals[m];
                    }
                    double conc_exact = pde.exact_concentration_comp(
                        c, xi, eta, current_time_);
                    double diff = conc_num - conc_exact;
                    error_squared[c] += diff * diff * point.w;
                    exact_squared[c] += conc_exact * conc_exact * point.w;
                }
            }
        }
    }

    // 绝对 L2 误差：||c_h - c_exact||
    l2_error_concentration_.resize(n_comp_);
    for (int c = 0; c < n_comp_; ++c) {
        l2_error_concentration_[c] = std::sqrt(error_squared[c]);
    }
    return l2_error_concentration_;
}

std::vector<double> MSSolver::computeFluxL2Error(int gauss_point_num) {
    if (!solved_ || !solution_ || vem::get_raw(solution_) == PETSC_NULLPTR) {
        throw std::runtime_error(
            "flux L2 error requires a completed numerical solution");
    }

    const MaxwellStefan::MSPdeData& pde = problem_->get_pde_data();
    const MeshData* mesh_data = getMeshData();
    Basis basis(getK());
    int polynomial_dim = basis.getDimHdiv();

    std::vector<double> error_squared(n_comp_, 0.0);
    std::vector<double> exact_squared(n_comp_, 0.0);
    vem::GaussQuadrature quadrature;
    quadrature.get_triangle_reference_points(gauss_point_num);
    Vec sol = vem::get_raw(solution_);

    for (int elem = 0; elem < mesh_data->num_cells; ++elem) {
        int dof_flux_per_comp = getDofPerElementFlux(elem);
        int edge_count = mesh_data->nodes_per_cell[elem];
        int edge_dofs = edge_count * (getK() + 1);

        // 获取该单元的全局自由度映射
        std::vector<PetscInt> global_dofs = getLocalToGlobalDofs(elem);

        // 方向调整（单组分通量部分，计算一次，所有组分复用）
        std::vector<double> single_comp_dir(dof_flux_per_comp, 1.0);
        int cell_node_start = mesh_data->cell_node_indices[elem];
        int cell_edge_start = mesh_data->cell_edge_indices[elem];
        for (int degree = 0; degree <= getK(); ++degree) {
            for (int local_edge = 0; local_edge < edge_count; ++local_edge) {
                int local_dof = degree * edge_count + local_edge;
                int global_edge =
                    mesh_data->global_edges[cell_edge_start + local_edge];
                if (mesh_data->edge_occurrence[global_edge] == 1) {
                    continue;  // 边界边不调整
                }
                int local_node0 =
                    mesh_data->cell_nodes[cell_node_start + local_edge];
                int local_node1 = mesh_data->cell_nodes[
                    cell_node_start + (local_edge + 1) % edge_count];
                int global_node0 = mesh_data->edge_endpoints[2 * global_edge];
                int global_node1 =
                    mesh_data->edge_endpoints[2 * global_edge + 1];
                bool reverse = (local_node0 == global_node1 &&
                                local_node1 == global_node0);
                if (reverse && degree % 2 == 0) {
                    single_comp_dir[local_dof] = -1.0;
                }
            }
        }

        // 单组分投影矩阵（所有组分共享）
        AutoPetscMat G = getMatrixG(elem);
        AutoPetscMat W = getMatrixW(elem);
        AutoPetscMat H = getMatrixH(elem);
        AutoPetscMat H_star = getMatrixH_star(elem);
        AutoPetscMat B = getMatrixB(elem, H_star, H, W);
        AutoPetscMat proj = getMatrixL2Proj_ploy(G, B);

        for (int c = 0; c < n_comp_; ++c) {
            // 提取该组分的局部通量 DOF 值并应用方向调整
            std::vector<double> local_flux(dof_flux_per_comp, 0.0);
            int comp_flux_local_base = c * dof_flux_per_comp;
            for (int i = 0; i < dof_flux_per_comp; ++i) {
                PetscInt global_idx = global_dofs[comp_flux_local_base + i];
                PetscScalar val;
                VecGetValues(sol, 1, &global_idx, &val);
                local_flux[i] = single_comp_dir[i] * PetscRealPart(val);
            }

            // 用投影矩阵得到多项式系数
            AutoPetscVec local_flux_vec = vem::create_vector(dof_flux_per_comp);
            for (int i = 0; i < dof_flux_per_comp; ++i) {
                PetscInt idx = i;
                VecSetValue(vem::get_raw(local_flux_vec), idx,
                            local_flux[i], INSERT_VALUES);
            }
            VecAssemblyBegin(vem::get_raw(local_flux_vec));
            VecAssemblyEnd(vem::get_raw(local_flux_vec));

            AutoPetscVec poly_coeffs = vem::create_vector(polynomial_dim);
            PetscErrorCode ierr = MatMult(
                vem::get_raw(proj),
                vem::get_raw(local_flux_vec),
                vem::get_raw(poly_coeffs));
            if (ierr != 0) {
                throw std::runtime_error(
                    "failed to project numerical flux (L2 error)");
            }

            std::vector<double> coeffs(polynomial_dim, 0.0);
            for (int i = 0; i < polynomial_dim; ++i) {
                PetscInt idx = i;
                PetscScalar val;
                VecGetValues(vem::get_raw(poly_coeffs), 1, &idx, &val);
                coeffs[i] = PetscRealPart(val);
            }

            // 在高斯点上求值并积分误差
            BasisPoint center(mesh_data->cell_centroid_x[elem],
                              mesh_data->cell_centroid_y[elem]);
            double diameter = mesh_data->cell_diameter[elem];

            vem::PolygonTriangulator triangulator;
            vem::TriangulatedElement element =
                triangulator.triangulateElement(*mesh_data, elem);

            for (const vem::Triangle& triangle : element.triangles) {
                std::vector<vem::QuadPoint> points =
                    quadrature.get_triangle_points(triangle.vertices,
                                                   gauss_point_num);
                for (const vem::QuadPoint& point : points) {
                    BasisPoint qp(point.x[0], point.x[1]);
                    BasisVector flux_num(0.0, 0.0);
                    for (int i = 0; i < polynomial_dim; ++i) {
                        BasisVector val = basis.evalHdivBasis(
                            i, center, diameter, qp);
                        flux_num.x += coeffs[i] * val.x;
                        flux_num.y += coeffs[i] * val.y;
                    }
                    MaxwellStefan::Vector2D flux_exact =
                        pde.exact_flux_comp(c, point.x[0], point.x[1],
                                            current_time_);
                    double dx = flux_num.x - flux_exact.x;
                    double dy = flux_num.y - flux_exact.y;
                    error_squared[c] += (dx * dx + dy * dy) * point.w;
                    exact_squared[c] += (flux_exact.x * flux_exact.x
                                       + flux_exact.y * flux_exact.y) * point.w;
                }
            }
        }
    }

    // 绝对 L2 误差：||J_h - J_exact||
    l2_error_flux_.resize(n_comp_);
    for (int c = 0; c < n_comp_; ++c) {
        l2_error_flux_[c] = std::sqrt(error_squared[c]);
    }
    return l2_error_flux_;
}
