/**
 * CPU Maxwell–Stefan 四算例初始化及映射实现。
 * 静态几何在 CPU 求值，时间相关积分数据由 GPU 对应函数求值。
 */
#include "ms_problem.h"
#include "../lib/polynomial_basis.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace MaxwellStefan {

// ============================================================================
// 算例管理：初始化
// ============================================================================

bool MSProblem::init_problem() {
    bool ok = false;
    switch (problem_index_) {
        case 1:
            ok = init_problem_1();
            break;
        case 2:
            ok = init_problem_2();
            break;
        case 3:
            ok = init_problem_3();
            break;
        case 4:
            ok = init_problem_4();
            break;
        default:
            std::cerr << "未知算例编号: " << problem_index_ << std::endl;
            return false;
    }

    if (!ok || !mapping_) {
        return ok;
    }

    // 分单元映射必须有网格才能工作（TriBlockSwirlMapping 要靠它建 Q8 节点表）。
    // 漏调 set_mesh 在这里立刻抛异常，而不是留个 nullptr 等到某个 Gauss 点
    // 上崩掉——更坏的情况是静默取到错的雅可比，误差表看着正常却是错的。
    if (mapping_->requires_mesh() && mesh_reader_ == nullptr) {
        throw std::runtime_error(
            "算例 " + std::to_string(problem_index_) +
            " 使用分单元映射，必须在 init_problem() 之前调用 "
            "MSProblem::set_mesh(reader)");
    }

    // 把网格转交给刚构造好的映射对象，作为分单元映射的网格调用基础。
    // set_mesh 对算例 1/2/3 是可选的：未调用时 mesh_reader_ 为 nullptr，
    // 此处直接跳过。那三个映射都是全局连续的解析映射，与单元无关、不读网格，
    // 所以跳过与否对数值结果没有任何影响。
    if (mesh_reader_) {
        mapping_->set_mesh(*mesh_reader_);
    }
    return ok;
}

// ============================================================================
// 算例 1：曲边映射 + 三组分（参数占位，初值/源项等后续补充）
// ============================================================================

bool MSProblem::init_problem_1() {
    // 曲边映射：正弦扰动，默认 eps=0.05
    mapping_.reset(new SinPerturbationMapping(0.05));

    // 组分数
    int n = 3;
    pde_data_.n_components = n;

    // 二元扩散系数倒数 c_ij
    //   c12 = 0.1, c13 = 0.2, c23 = 2.0
    pde_data_.c_ij.assign(n, std::vector<double>(n, 0.0));
    pde_data_.c_ij[0][1] = pde_data_.c_ij[1][0] = 0.1;
    pde_data_.c_ij[0][2] = pde_data_.c_ij[2][0] = 0.2;
    pde_data_.c_ij[1][2] = pde_data_.c_ij[2][1] = 2.0;

    // 计算 c* = min(c_ij)
    double cstar = std::numeric_limits<double>::max();
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (pde_data_.c_ij[i][j] < cstar)
                cstar = pde_data_.c_ij[i][j];
        }
    }
    pde_data_.c_star = cstar;

    // 计算 bar_c_ij = c_ij - c*
    pde_data_.bar_c_ij.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            pde_data_.bar_c_ij[i][j] = pde_data_.c_ij[i][j] - cstar;
        }
    }

    // 计算域上的初始浓度（间断初值，三组分）
    //   c0: 左下块 (ξ∈[0,0.5], η∈[0,0.5]) = 0.8，其余 = 0.1
    //   c1: 左上块 (ξ∈[0,0.5], η∈[0.5,1]) = 0.8，其余 = 0.1
    //   c2: 1 - c0 - c1（右半区域为主），三组分之和恒为 1
    // 内部间断面归属约定（左闭右开、下闭上开）：
    //   ξ=0.5 归右半（低值），η=0.5 归下半（c0 高值，c1 低值）。
    // 外部边界 (ξ=0,1 和 η=0,1) 加 1e-10 容差，
    //   避免逆映射浮点误差导致域外点被判为全低值。
    pde_data_.initial_concentration_comp =
        [](int i, int cell_idx, double xi, double eta) {
        (void)cell_idx;  // 初值直接按计算域坐标分块，与单元无关
        const double tol = 1e-10;

        // c0 高值区：左下块。右边界 ξ=0.5 不包含（归右侧低值），
        //            上边界 η=0.5 包含（归下侧高值）。
        auto ll_high = [&](double x, double y) {
            return x > -tol && x < 0.5 && y > -tol && y <= 0.5;
        };
        // c1 高值区：左上块。右边界 ξ=0.5 不包含（归右侧低值），
        //            下边界 η=0.5 不包含（归下侧低值）。
        auto ul_high = [&](double x, double y) {
            return x > -tol && x < 0.5 && y > 0.5 && y < 1.0 + tol;
        };

        if (i == 0) {
            return ll_high(xi, eta) ? 0.8 : 0.1;
        } else if (i == 1) {
            return ul_high(xi, eta) ? 0.8 : 0.1;
        } else {  // i == 2
            double c0 = ll_high(xi, eta) ? 0.8 : 0.1;
            double c1 = ul_high(xi, eta) ? 0.8 : 0.1;
            return 1.0 - c0 - c1;
        }
    };

    // 物理域上的初始浓度：通过逆映射找 (ξ,η)，再用计算域初值求值
    // 标量点值在映射下不变：c(x(ξ,η)) = ĉ(ξ,η)
    // 逆映射是分单元的，故 cell_idx 需一路透传
    pde_data_.initial_concentration =
        [this](int i, int cell_idx, double x, double y) {
        Point2D comp = invert_mapping(x, y, cell_idx);
        return pde_data_.initial_concentration_comp(i, cell_idx,
                                                   comp.x, comp.y);
    };

    // 源项：物理域和计算域均为 0
    pde_data_.source_f = [](int /*i*/, double /*x*/, double /*y*/, double /*t*/) {
        return 0.0;
    };
    pde_data_.source_f_comp = [](int /*i*/, int /*cell_idx*/,
                                 double /*xi*/, double /*eta*/, double /*t*/) {
        return 0.0;
    };

    // 法向通量边界条件：零通量（所有组分的法向通量均为 0）
    // Piola 变换保持法向通量为零，物理域和计算域都是 0
    pde_data_.boundary_flux = [](int /*i*/, double /*x*/, double /*y*/,
                                 double /*nx*/, double /*ny*/, double /*t*/) {
        return 0.0;
    };
    pde_data_.boundary_flux_comp = [](int /*i*/, int /*cell_idx*/,
                                       double /*xi*/, double /*eta*/,
                                       double /*nx*/, double /*ny*/, double /*t*/) {
        return 0.0;
    };

    // ========== 初值模式：物理域上评估 A_ij（纯非线性部分，不含 c*I）==========
    // 公式与计算域版本完全一致，只是浓度从物理域初值函数获取
    //   非对角元 (i≠j): A_ij = -bar_c_ij * u_i
    //   对角元   (i=j): A_ii = Σ_{k≠i} bar_c_ik * u_k
    pde_data_.evaluate_A_initial =
        [this, n](int i, int j, int cell_idx, double x, double y) {
            if (i < 0 || i >= n || j < 0 || j >= n) {
                throw std::out_of_range("A_ij 索引越界");
            }
            const auto& cij = pde_data_.c_ij;
            double cstar = pde_data_.c_star;
            const auto& init = pde_data_.initial_concentration;

            if (i != j) {
                int i_sym = (i < j) ? i : j;
                int j_sym = (i < j) ? j : i;
                double bar_c = cij[i_sym][j_sym] - cstar;
                double u_i = init(i, cell_idx, x, y);
                return -bar_c * u_i;
            } else {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    if (k == i) continue;
                    int i_sym = (i < k) ? i : k;
                    int k_sym = (i < k) ? k : i;
                    double bar_c = cij[i_sym][k_sym] - cstar;
                    double u_k = init(k, cell_idx, x, y);
                    sum += bar_c * u_k;
                }
                return sum;
            }
        };

    // ========== 初值模式：计算域上评估 A_ij（纯非线性部分，不含 c*I）==========
    // 逻辑完全对齐 MS_FVM 的 init_A_coeff
    //   非对角元 (i≠j): A_ij = -bar_c_ij * u_i
    //   对角元   (i=j): A_ii = Σ_{k≠i} bar_c_ik * u_k
    pde_data_.evaluate_A_comp_initial =
        [this, n](int i, int j, int cell_idx, double xi, double eta) {
            // 步骤1：校验索引合法性
            if (i < 0 || i >= n || j < 0 || j >= n) {
                throw std::out_of_range("A_ij 索引越界");
            }
            const auto& cij = pde_data_.c_ij;
            double cstar = pde_data_.c_star;
            const auto& init = pde_data_.initial_concentration_comp;

            // 步骤2：非对角线元素（i≠j）
            if (i != j) {
                int i_sym = (i < j) ? i : j;
                int j_sym = (i < j) ? j : i;
                double bar_c = cij[i_sym][j_sym] - cstar;
                double u_i = init(i, cell_idx, xi, eta);
                return -bar_c * u_i;
            }
            // 步骤3：对角线元素（i=j）
            else {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    if (k == i) continue;
                    int i_sym = (i < k) ? i : k;
                    int k_sym = (i < k) ? k : i;
                    double bar_c = cij[i_sym][k_sym] - cstar;
                    double u_k = init(k, cell_idx, xi, eta);
                    sum += bar_c * u_k;
                }
                return sum;
            }
        };

    // ========== 数值解模式：计算域上评估 A_ij（纯非线性部分，不含 c*I）==========
    // 完全对齐 MS_FVM 的 poly_A_coeff 逻辑
    //   步骤1：校验索引
    //   步骤2：根据系数向量维数识别 k
    //   步骤3：用单项式基还原各组分在 (ξ,η) 处的浓度值
    //   步骤4：非对角元 A_ij = -bar_c_ij * u_i
    //   步骤5：对角元   A_ii = Σ_{k≠i} bar_c_ik * u_k
    pde_data_.evaluate_A_comp_poly =
        [this, n](int i, int j, const std::vector<double>& u_ploy_coeff,
                  double xD_x, double xD_y, double hD,
                  double xi, double eta) {
            // 步骤1：校验索引合法性
            if (n <= 0) {
                throw std::runtime_error("组分数量无效");
            }
            if (i < 0 || i >= n || j < 0 || j >= n) {
                throw std::out_of_range("A_ij 索引越界");
            }
            const auto& cij = pde_data_.c_ij;
            double cstar = pde_data_.c_star;

            // 步骤2：根据 u_ploy_coeff 的维数自动识别 k
            int total_size = static_cast<int>(u_ploy_coeff.size());
            int dim_poly_per_comp = total_size / n;

            int k = 0;
            while (true) {
                int dim_test = (k + 1) * (k + 2) / 2;
                if (dim_test == dim_poly_per_comp) {
                    break;
                }
                if (dim_test > dim_poly_per_comp || k > 10) {
                    throw std::runtime_error("无法根据 u_ploy_coeff 的维数识别 k");
                }
                k++;
            }

            // 步骤3：根据单项式系数恢复各组分在 (ξ,η) 处的浓度值
            vem::basis::BasisFunctionPloy basis_poly(k);
            vem::basis::Point2D xD(xD_x, xD_y);
            vem::basis::Point2D eval_pt(xi, eta);

            std::vector<double> u_vals(n);
            int dim_poly = dim_poly_per_comp;

            for (int c = 0; c < n; ++c) {
                double val = 0.0;
                for (int m = 0; m < dim_poly; ++m) {
                    double m_val = basis_poly.evalMonomial2D(m, xD, hD, eval_pt);
                    val += u_ploy_coeff[c * dim_poly + m] * m_val;
                }
                u_vals[c] = val;
            }

            // 步骤4：非对角线元素（i≠j）
            if (i != j) {
                int i_sym = (i < j) ? i : j;
                int j_sym = (i < j) ? j : i;
                double bar_c = cij[i_sym][j_sym] - cstar;
                return -bar_c * u_vals[i];
            }
            // 步骤5：对角线元素（i=j）
            else {
                double sum = 0.0;
                for (int k_idx = 0; k_idx < n; ++k_idx) {
                    if (k_idx == i) continue;
                    int i_sym = (i < k_idx) ? i : k_idx;
                    int k_sym = (i < k_idx) ? k_idx : i;
                    double bar_c = cij[i_sym][k_sym] - cstar;
                    sum += bar_c * u_vals[k_idx];
                }
                return sum;
            }
        };

    std::cout << "MS Problem 1 initialized: " << n << " components\n";
    std::cout << "  Mapping: SinPerturbation (eps=0.05)\n";
    std::cout << "  c* = " << pde_data_.c_star << "\n";
    std::cout << "  bar_c_12 = " << pde_data_.bar_c_ij[0][1]
              << ", bar_c_13 = " << pde_data_.bar_c_ij[0][2]
              << ", bar_c_23 = " << pde_data_.bar_c_ij[1][2] << "\n";
    std::cout << "  Initial concentration: discontinuous (computational domain)\n";

    return true;
}

// ============================================================================
// 逆映射：由物理坐标 (x,y) 求 cell_idx 号单元内的计算域坐标 (ξ,η)
// 牛顿迭代求解 x(ξ,η) = x_target, y(ξ,η) = y_target
//
// 映射分单元后，全局逆映射不再唯一定义，必须指明在哪个单元的映射公式下求逆。
// 当前三个映射与单元无关，所以任何 cell_idx 给出相同结果。
// ============================================================================

Point2D MSProblem::invert_mapping(double x, double y, int cell_idx) const {
    if (!mapping_) {
        throw std::runtime_error("MSProblem::invert_mapping: mapping not initialized");
    }

    // 初始猜测：恒等映射 (ξ=x, η=y)
    double xi = x;
    double eta = y;

    const int max_iter = 50;
    const double tol = 1e-14;

    for (int iter = 0; iter < max_iter; ++iter) {
        Point2D phys = mapping_->physical_coords(cell_idx, xi, eta);
        double rx = phys.x - x;
        double ry = phys.y - y;

        if (std::abs(rx) < tol && std::abs(ry) < tol)
            break;

        // 雅可比矩阵 J = [[dxdxi, dxdeta], [dydxi, dydeta]]
        Tensor2D J = mapping_->jacobian(cell_idx, xi, eta);
        double detJ = J.xx * J.yy - J.xy * J.yx;

        if (std::abs(detJ) < 1e-14) {
            throw std::runtime_error("MSProblem::invert_mapping: singular Jacobian");
        }

        // J^{-1} * [-rx, -ry]^T
        double dxi = (J.yy * (-rx) - J.xy * (-ry)) / detJ;
        double deta = (-J.yx * (-rx) + J.xx * (-ry)) / detJ;

        xi += dxi;
        eta += deta;
    }

    return Point2D(xi, eta);
}

// ============================================================================
// 算例 2：制造解算例 — 正弦脉动扩散（带真解，用于误差分析）
//   曲边映射：与算例 1 相同（SinPerturbationMapping, eps=0.05）
//   真解（物理域）：
//     c₁(x,y,t) = 0.25 sin(2πx) sin(8πt) + 0.25
//     c₂(x,y,t) = 0.25 sin(3πy) sin(6πt) + 0.25
//     c₃(x,y,t) = 1 - c₁ - c₂
//   扩散参数：与算例 1 完全相同（c₁₂=0.1, c₁₃=0.2, c₂₃=2.0, c*=0.1）
//   通量本构：J = -A(c)^{-1} ∇c，其中 A(c) = c*I + Ā(c)
//   源项：fᵢ = ∂cᵢ/∂t + ∇·Jᵢ （散度用中心差分近似）
//   边界：法向通量边界（由真解计算）
// ============================================================================

namespace {
// 3×3 矩阵求逆（返回行列式）
inline double mat3_inverse(const double A[3][3], double Ainv[3][3]) {
    double det = A[0][0] * (A[1][1]*A[2][2] - A[1][2]*A[2][1])
               - A[0][1] * (A[1][0]*A[2][2] - A[1][2]*A[2][0])
               + A[0][2] * (A[1][0]*A[2][1] - A[1][1]*A[2][0]);
    if (std::abs(det) < 1e-20) return 0.0;
    double invd = 1.0 / det;
    Ainv[0][0] = (A[1][1]*A[2][2] - A[1][2]*A[2][1]) * invd;
    Ainv[0][1] = (A[0][2]*A[2][1] - A[0][1]*A[2][2]) * invd;
    Ainv[0][2] = (A[0][1]*A[1][2] - A[0][2]*A[1][1]) * invd;
    Ainv[1][0] = (A[1][2]*A[2][0] - A[1][0]*A[2][2]) * invd;
    Ainv[1][1] = (A[0][0]*A[2][2] - A[0][2]*A[2][0]) * invd;
    Ainv[1][2] = (A[0][2]*A[1][0] - A[0][0]*A[1][2]) * invd;
    Ainv[2][0] = (A[1][0]*A[2][1] - A[1][1]*A[2][0]) * invd;
    Ainv[2][1] = (A[0][1]*A[2][0] - A[0][0]*A[2][1]) * invd;
    Ainv[2][2] = (A[0][0]*A[1][1] - A[0][1]*A[1][0]) * invd;
    return det;
}

// 构造 A(c) 矩阵（3 组分）
// 构造完整 A 矩阵（含 cstar 对角元），用于源项、边界条件、初值等需要完整扩散系数的场合
inline void build_ABC(const std::vector<std::vector<double>>& bar_c,
                      double cstar,
                      const double c[3], double A[3][3]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i == j) {
                double s = cstar;
                for (int k = 0; k < 3; ++k) {
                    if (k == i) continue;
                    int is = std::min(i, k);
                    int ks = std::max(i, k);
                    s += bar_c[is][ks] * c[k];
                }
                A[i][j] = s;
            } else {
                int is = std::min(i, j);
                int js = std::max(i, j);
                A[i][j] = -bar_c[is][js] * c[i];
            }
        }
    }
}

// 构造 A_bar 矩阵（仅 bar_c 部分，不含 cstar 对角元）
// 用于非线性迭代 AG_poly：cstar I 由 cmin_G 单独处理
inline void build_ABC_bar(const std::vector<std::vector<double>>& bar_c,
                          const double c[3], double A[3][3]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i == j) {
                double s = 0.0;
                for (int k = 0; k < 3; ++k) {
                    if (k == i) continue;
                    int is = std::min(i, k);
                    int ks = std::max(i, k);
                    s += bar_c[is][ks] * c[k];
                }
                A[i][j] = s;
            } else {
                int is = std::min(i, j);
                int js = std::max(i, j);
                A[i][j] = -bar_c[is][js] * c[i];
            }
        }
    }
}
}  // anonymous namespace

// ============================================================================
// 算例 2 与算例 4 共用的制造解 PDE 数据
//
// 两个算例只有映射不同（算例 2 用全局连续的 SinPerturbationMapping，
// 算例 4 用逐单元 Q8 的 TriBlockSwirlMapping），真解 / 通量 / 源项 /
// 边界条件 / A_ij 一字不差，所以提取出来共用。
//
// 所有物理域函数都是 (x,y) 的解析式；计算域回调只通过
// mapping_->physical_coords / jacobian / jacobian_det 拉回，
// 因此换映射不需要改这里的任何一行。
//
// 调用前 mapping_ 必须已构造（下面的 lambda 捕获 this 并解引用 mapping_）。
// ============================================================================

void MSProblem::setup_manufactured_solution() {
    int n = 3;
    pde_data_.n_components = n;

    // 二元扩散系数倒数（与算例 1 相同）
    pde_data_.c_ij.assign(n, std::vector<double>(n, 0.0));
    pde_data_.c_ij[0][1] = pde_data_.c_ij[1][0] = 0.1;
    pde_data_.c_ij[0][2] = pde_data_.c_ij[2][0] = 0.2;
    pde_data_.c_ij[1][2] = pde_data_.c_ij[2][1] = 2.0;

    // 计算 c* = min(c_ij)
    double cstar = std::numeric_limits<double>::max();
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (pde_data_.c_ij[i][j] < cstar)
                cstar = pde_data_.c_ij[i][j];
    pde_data_.c_star = cstar;

    // bar_c_ij = c_ij - c*
    pde_data_.bar_c_ij.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            pde_data_.bar_c_ij[i][j] = pde_data_.c_ij[i][j] - cstar;

    // 注意：bar_c 引用 pde_data_ 的成员，生命周期一致
    const std::vector<std::vector<double>>& bar_c = pde_data_.bar_c_ij;

    // ========== 精确浓度（物理域） ==========
    pde_data_.exact_concentration = [](int i, double x, double y, double t) {
        if (i == 0) {
            return 0.25 * std::sin(2.0 * M_PI * x)
                        * std::sin(8.0 * M_PI * t) + 0.25;
        } else if (i == 1) {
            return 0.25 * std::sin(3.0 * M_PI * y)
                        * std::sin(6.0 * M_PI * t) + 0.25;
        } else {
            double c0 = 0.25 * std::sin(2.0 * M_PI * x)
                             * std::sin(8.0 * M_PI * t) + 0.25;
            double c1 = 0.25 * std::sin(3.0 * M_PI * y)
                             * std::sin(6.0 * M_PI * t) + 0.25;
            return 1.0 - c0 - c1;
        }
    };

    // ========== 精确浓度（计算域） ==========
    pde_data_.exact_concentration_comp =
        [this](int i, int cell_idx, double xi, double eta, double t) {
            Point2D p = mapping_->physical_coords(cell_idx, xi, eta);
            return pde_data_.exact_concentration(i, p.x, p.y, t);
        };

    // ========== 初值（真解在 t=0 处的值） ==========
    // 物理域版本直接取真解，不经过映射，故 cell_idx 未用到
    pde_data_.initial_concentration =
        [this](int i, int cell_idx, double x, double y) {
        (void)cell_idx;
        return pde_data_.exact_concentration(i, x, y, 0.0);
    };
    pde_data_.initial_concentration_comp =
        [this](int i, int cell_idx, double xi, double eta) {
            return pde_data_.exact_concentration_comp(i, cell_idx, xi, eta, 0.0);
        };

    // ========== 精确通量（物理域） J = -A(c)^{-1} ∇c ==========
    pde_data_.exact_flux =
        [this, &bar_c, cstar](int i, double x, double y, double t)
        -> Vector2D {
            double c[3];
            for (int k = 0; k < 3; ++k)
                c[k] = pde_data_.exact_concentration(k, x, y, t);

            // 物理域梯度 ∇c[k][0] = ∂_x c_k, ∇c[k][1] = ∂_y c_k
            double grad_c[3][2];
            double s8 = std::sin(8.0 * M_PI * t);
            double s6 = std::sin(6.0 * M_PI * t);
            grad_c[0][0] = 0.5 * M_PI * std::cos(2.0*M_PI*x) * s8;
            grad_c[0][1] = 0.0;
            grad_c[1][0] = 0.0;
            grad_c[1][1] = 0.75 * M_PI * std::cos(3.0*M_PI*y) * s6;
            grad_c[2][0] = -grad_c[0][0];
            grad_c[2][1] = -grad_c[1][1];

            double A[3][3];
            build_ABC(bar_c, cstar, c, A);
            double Ainv[3][3];
            mat3_inverse(A, Ainv);

            // J_i = -Σ_k Ainv[i][k] * ∇c_k
            double Jx = 0.0, Jy = 0.0;
            for (int k = 0; k < 3; ++k) {
                Jx += -Ainv[i][k] * grad_c[k][0];
                Jy += -Ainv[i][k] * grad_c[k][1];
            }
            return Vector2D(Jx, Jy);
        };

    // ========== 精确通量（计算域） Ĵ（逆变 Piola 变换） ==========
    //   Ĵ = det(J_fwd) · J_fwd^{-1} · J_phys
    pde_data_.exact_flux_comp =
        [this](int i, int cell_idx, double xi, double eta, double t) -> Vector2D {
            Point2D p = mapping_->physical_coords(cell_idx, xi, eta);
            Vector2D Jp = pde_data_.exact_flux(i, p.x, p.y, t);
            Tensor2D Jfwd = mapping_->jacobian(cell_idx, xi, eta);
            double detJ = mapping_->jacobian_det(cell_idx, xi, eta);
            double inv_det = 1.0 / detJ;
            double tmp_x = (Jfwd.yy * Jp.x - Jfwd.xy * Jp.y) * inv_det;
            double tmp_y = (-Jfwd.yx * Jp.x + Jfwd.xx * Jp.y) * inv_det;
            return Vector2D(tmp_x * detJ, tmp_y * detJ);
        };

    // ========== 源项（物理域） f_i = ∂c_i/∂t + ∇·J_i ==========
    pde_data_.source_f =
        [this](int i, double x, double y, double t) {
            // ∂c_i/∂t 解析形式
            double dc_dt = 0.0;
            if (i == 0) {
                dc_dt = 2.0 * M_PI * std::sin(2.0*M_PI*x)
                                    * std::cos(8.0*M_PI*t);
            } else if (i == 1) {
                dc_dt = 1.5 * M_PI * std::sin(3.0*M_PI*y)
                                    * std::cos(6.0*M_PI*t);
            } else {
                double dc1 = 2.0 * M_PI * std::sin(2.0*M_PI*x)
                                           * std::cos(8.0*M_PI*t);
                double dc2 = 1.5 * M_PI * std::sin(3.0*M_PI*y)
                                           * std::cos(6.0*M_PI*t);
                dc_dt = -dc1 - dc2;
            }

            // ∇·J 中心差分
            const double h = 1e-6;
            Vector2D J_xp = pde_data_.exact_flux(i, x + h, y, t);
            Vector2D J_xm = pde_data_.exact_flux(i, x - h, y, t);
            Vector2D J_yp = pde_data_.exact_flux(i, x, y + h, t);
            Vector2D J_ym = pde_data_.exact_flux(i, x, y - h, t);
            double div_J = (J_xp.x - J_xm.x) / (2.0 * h)
                         + (J_yp.y - J_ym.y) / (2.0 * h);

            return dc_dt + div_J;
        };

    // ========== 源项（计算域） f̂_i = f_i(x(ξ,η), t) ==========
    pde_data_.source_f_comp =
        [this](int i, int cell_idx, double xi, double eta, double t) {
            Point2D p = mapping_->physical_coords(cell_idx, xi, eta);
            return pde_data_.source_f(i, p.x, p.y, t);
        };

    // ========== 法向通量边界（物理域） J_i · n ==========
    pde_data_.boundary_flux =
        [this](int i, double x, double y, double nx, double ny, double t) {
            Vector2D J = pde_data_.exact_flux(i, x, y, t);
            return J.x * nx + J.y * ny;
        };

    // ========== 法向通量边界（计算域） Ĵ_i · n̂ ==========
    pde_data_.boundary_flux_comp =
        [this](int i, int cell_idx, double xi, double eta,
               double nx, double ny, double t) {
            Vector2D Jhat = pde_data_.exact_flux_comp(i, cell_idx, xi, eta, t);
            return Jhat.x * nx + Jhat.y * ny;
        };

    // ========== 初值模式 A_ij（物理域） ==========
    // 注意：返回 A_bar（仅 bar_c 部分，不含 cstar 对角元）
    //       因为 cstar I 由 cmin_G 单独组装
    pde_data_.evaluate_A_initial =
        [this, &bar_c](int i, int j, int cell_idx, double x, double y) {
            double c[3];
            for (int k = 0; k < 3; ++k)
                c[k] = pde_data_.initial_concentration(k, cell_idx, x, y);
            double A[3][3];
            build_ABC_bar(bar_c, c, A);
            return A[i][j];
        };

    // ========== 初值模式 A_ij（计算域） ==========
    // 注意：返回 A_bar（仅 bar_c 部分，不含 cstar 对角元）
    //       因为 cstar I 由 cmin_G 单独组装
    pde_data_.evaluate_A_comp_initial =
        [this, &bar_c](int i, int j, int cell_idx, double xi, double eta) {
            double c[3];
            for (int k = 0; k < 3; ++k)
                c[k] = pde_data_.initial_concentration_comp(k, cell_idx,
                                                            xi, eta);
            double A[3][3];
            build_ABC_bar(bar_c, c, A);
            return A[i][j];
        };

    // ========== 数值解模式 A_ij（计算域，多项式系数还原浓度） ==========
    // 注意：这里返回 A_bar（仅 bar_c 部分，不含 cstar 对角元）
    //       因为 cstar I 由 cmin_G 单独组装
    pde_data_.evaluate_A_comp_poly =
        [this, &bar_c](int i, int j,
                       const std::vector<double>& u_ploy_coeff,
                       double xD_x, double xD_y, double hD,
                       double xi, double eta) {
            const int n = 3;
            int total_size = static_cast<int>(u_ploy_coeff.size());
            int n_monomial = total_size / n;

            // 推 k：dim P_k = (k+1)(k+2)/2 = n_monomial
            int k = 0;
            for (; k < 10; ++k) {
                int d = (k + 1) * (k + 2) / 2;
                if (d == n_monomial) break;
            }

            vem::basis::BasisFunctionPloy basis(k);
            vem::basis::Point2D eval_point(xi, eta);
            vem::basis::Point2D center(xD_x, xD_y);

            double c[3] = {0.0, 0.0, 0.0};
            for (int comp = 0; comp < n; ++comp) {
                double val = 0.0;
                for (int m = 0; m < n_monomial; ++m) {
                    val += u_ploy_coeff[comp * n_monomial + m] *
                        basis.evalMonomial2D(m, center, hD, eval_point);
                }
                c[comp] = val;
            }

            double A[3][3];
            build_ABC_bar(bar_c, c, A);
            return A[i][j];
        };

}

// ============================================================================
// 算例 2：正弦脉动三组分制造解
//   曲边映射：SinPerturbationMapping(eps=0.05)，全局连续的单一解析映射
//   真解：c0 = 0.25 sin(2πx) sin(8πt) + 0.25
//         c1 = 0.25 sin(3πy) sin(6πt) + 0.25
//         c2 = 1 - c0 - c1
//   扩散参数：c01=0.1, c02=0.2, c12=2.0, c* = 0.1
// ============================================================================

bool MSProblem::init_problem_2() {
    mapping_.reset(new SinPerturbationMapping(0.05));
    setup_manufactured_solution();

    std::cout << "MS Problem 2 initialized: 3 components\n";
    std::cout << "  Mapping: SinPerturbation (eps=0.05)\n";
    std::cout << "  c* = " << pde_data_.c_star << "\n";
    std::cout << "  Type: manufactured solution (with exact solution)\n";
    return true;
}

// ============================================================================
// 算例 4：与算例 2 完全相同的制造解 + TriBlockSwirl 逐单元 Q8 曲边映射
//
// 唯一的区别是映射：每个单元一个雅可比，单元内逐点不同，跨单元边界跳变。
// 真解、扩散参数、源项、边界条件与算例 2 逐字相同，所以两者的误差表
// 可以直接横向对比——差异只来自网格扭曲程度。
//
// TriBlockSwirlMapping 必须有网格才能建 Q8 节点表，故本算例要求调用方
// 在 init_problem() **之前**调用 MSProblem::set_mesh()。漏调会在
// init_problem() 里抛异常（见 requires_mesh 校验），不会静默算错。
// ============================================================================

bool MSProblem::init_problem_4() {
    mapping_.reset(new TriBlockSwirlMapping());
    setup_manufactured_solution();

    std::cout << "MS Problem 4 initialized: 3 components\n";
    std::cout << "  Mapping: TriBlockSwirl (逐单元 Q8，真三块 + 全局 swirl)\n";
    std::cout << "  c* = " << pde_data_.c_star << "\n";
    std::cout << "  Type: manufactured solution (with exact solution)\n";
    return true;
}

// ============================================================================
// 算例 3：半圆环三组分扩散（趋于稳态，无制造解）
//   曲边映射：半圆环映射 HalfAnnulusMapping
//     r = ξ + 0.5,  θ = π(η - 0.5)
//     x = r cosθ,  y = r sinθ
//     内半径 0.5，外半径 1.5，右半圆环 θ ∈ [-π/2, π/2]
//   初始条件（计算域，按 ξ 分层）：
//     c0: ξ∈[0,0.25] → 1, 其余 → 0  （内圈纯组分 1）
//     c1: ξ∈[0.75,1] → 1, 其余 → 0  （外圈纯组分 2）
//     c2: 1 - c0 - c1                    （中间纯组分 3）
//   源项：f_i = 0
//   边界条件：零法向通量 J_i · n = 0（所有边界）
//   物理过程：三组分沿径向扩散，趋向稳态均匀分布
// ============================================================================

bool MSProblem::init_problem_3() {
    // 曲边映射：半圆环
    mapping_.reset(new HalfAnnulusMapping());

    int n = 3;
    pde_data_.n_components = n;

    // 二元扩散系数倒数（与算例 1、2 相同）
    pde_data_.c_ij.assign(n, std::vector<double>(n, 0.0));
    pde_data_.c_ij[0][1] = pde_data_.c_ij[1][0] = 0.1;
    pde_data_.c_ij[0][2] = pde_data_.c_ij[2][0] = 0.2;
    pde_data_.c_ij[1][2] = pde_data_.c_ij[2][1] = 2.0;

    // 计算 c* = min(c_ij)
    double cstar = std::numeric_limits<double>::max();
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (pde_data_.c_ij[i][j] < cstar)
                cstar = pde_data_.c_ij[i][j];
    pde_data_.c_star = cstar;

    // 计算 bar_c_ij = c_ij - c*
    pde_data_.bar_c_ij.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            pde_data_.bar_c_ij[i][j] = pde_data_.c_ij[i][j] - cstar;

    // 计算域上的初始浓度（按 ξ 径向分层，与 η 无关）
    //   ξ ∈ [0, 0.25]   → (c0, c1, c2) = (1, 0, 0)  内圈纯组分1
    //   ξ ∈ (0.25, 0.75) → (c0, c1, c2) = (0, 0, 1)  中间纯组分3
    //   ξ ∈ [0.75, 1]   → (c0, c1, c2) = (0, 1, 0)  外圈纯组分2
    // 间断面归属：ξ=0.25 归内圈（高 c0），ξ=0.75 归外圈（高 c1）
    pde_data_.initial_concentration_comp =
        [](int i, int cell_idx, double xi, double eta) {
        (void)cell_idx;  // 初值只按 ξ 径向分层，与单元无关
        (void)eta;
        double c0 = (xi <= 0.25) ? 1.0 : 0.0;
        double c1 = (xi >= 0.75) ? 1.0 : 0.0;
        if (i == 0) return c0;
        if (i == 1) return c1;
        return 1.0 - c0 - c1;
    };

    // 物理域上的初始浓度：通过逆映射找 (ξ,η)（逆映射分单元，需透传 cell_idx）
    pde_data_.initial_concentration =
        [this](int i, int cell_idx, double x, double y) {
        Point2D comp = invert_mapping(x, y, cell_idx);
        return pde_data_.initial_concentration_comp(i, cell_idx,
                                                   comp.x, comp.y);
    };

    // 源项：0
    pde_data_.source_f = [](int /*i*/, double /*x*/, double /*y*/, double /*t*/) {
        return 0.0;
    };
    pde_data_.source_f_comp = [](int /*i*/, int /*cell_idx*/,
                                 double /*xi*/, double /*eta*/, double /*t*/) {
        return 0.0;
    };

    // 边界条件：零法向通量（所有边界）
    pde_data_.boundary_flux = [](int /*i*/, double /*x*/, double /*y*/,
                                 double /*nx*/, double /*ny*/, double /*t*/) {
        return 0.0;
    };
    pde_data_.boundary_flux_comp = [](int /*i*/, int /*cell_idx*/,
                                       double /*xi*/, double /*eta*/,
                                       double /*nx*/, double /*ny*/, double /*t*/) {
        return 0.0;
    };

    // ===== 初值模式：物理域上评估 A_ij =====
    pde_data_.evaluate_A_initial =
        [this, n](int i, int j, int cell_idx, double x, double y) {
            if (i < 0 || i >= n || j < 0 || j >= n) {
                throw std::out_of_range("A_ij 索引越界");
            }
            const auto& cij = pde_data_.c_ij;
            double cstar_val = pde_data_.c_star;
            const auto& init = pde_data_.initial_concentration;

            if (i != j) {
                int is = (i < j) ? i : j;
                int js = (i < j) ? j : i;
                double bar_c = cij[is][js] - cstar_val;
                double u_i = init(i, cell_idx, x, y);
                return -bar_c * u_i;
            } else {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    if (k == i) continue;
                    int is = (i < k) ? i : k;
                    int ks = (i < k) ? k : i;
                    double bar_c = cij[is][ks] - cstar_val;
                    double u_k = init(k, cell_idx, x, y);
                    sum += bar_c * u_k;
                }
                return sum;
            }
        };

    // ===== 初值模式：计算域上评估 A_ij =====
    pde_data_.evaluate_A_comp_initial =
        [this, n](int i, int j, int cell_idx, double xi, double eta) {
            if (i < 0 || i >= n || j < 0 || j >= n) {
                throw std::out_of_range("A_ij 索引越界");
            }
            const auto& cij = pde_data_.c_ij;
            double cstar_val = pde_data_.c_star;
            const auto& init = pde_data_.initial_concentration_comp;

            if (i != j) {
                int is = (i < j) ? i : j;
                int js = (i < j) ? j : i;
                double bar_c = cij[is][js] - cstar_val;
                double u_i = init(i, cell_idx, xi, eta);
                return -bar_c * u_i;
            } else {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    if (k == i) continue;
                    int is = (i < k) ? i : k;
                    int ks = (i < k) ? k : i;
                    double bar_c = cij[is][ks] - cstar_val;
                    double u_k = init(k, cell_idx, xi, eta);
                    sum += bar_c * u_k;
                }
                return sum;
            }
        };

    // ===== 数值解模式：计算域上评估 A_ij（纯非线性部分，不含 c*I）=====
    // 完全对齐算例 1 的 poly_A_coeff 逻辑
    //   步骤1：校验索引
    //   步骤2：根据系数向量维数识别 k
    //   步骤3：用单项式基还原各组分在 (ξ,η) 处的浓度值
    //   步骤4：非对角元 A_ij = -bar_c_ij * u_i
    //   步骤5：对角元   A_ii = Σ_{k≠i} bar_c_ik * u_k
    pde_data_.evaluate_A_comp_poly =
        [this, n](int i, int j, const std::vector<double>& u_ploy_coeff,
                  double xD_x, double xD_y, double hD,
                  double xi, double eta) {
            // 步骤1：校验索引合法性
            if (n <= 0) {
                throw std::runtime_error("组分数量无效");
            }
            if (i < 0 || i >= n || j < 0 || j >= n) {
                throw std::out_of_range("A_ij 索引越界");
            }
            const auto& cij = pde_data_.c_ij;
            double cstar = pde_data_.c_star;

            // 步骤2：根据 u_ploy_coeff 的维数自动识别 k
            int total_size = static_cast<int>(u_ploy_coeff.size());
            int dim_poly_per_comp = total_size / n;

            int k = 0;
            while (true) {
                int dim_test = (k + 1) * (k + 2) / 2;
                if (dim_test == dim_poly_per_comp) {
                    break;
                }
                if (dim_test > dim_poly_per_comp || k > 10) {
                    throw std::runtime_error("无法根据 u_ploy_coeff 的维数识别 k");
                }
                k++;
            }

            // 步骤3：根据单项式系数恢复各组分在 (ξ,η) 处的浓度值
            vem::basis::BasisFunctionPloy basis_poly(k);
            vem::basis::Point2D xD(xD_x, xD_y);
            vem::basis::Point2D eval_pt(xi, eta);

            std::vector<double> u_vals(n);
            int dim_poly = dim_poly_per_comp;

            for (int c = 0; c < n; ++c) {
                double val = 0.0;
                for (int m = 0; m < dim_poly; ++m) {
                    double m_val = basis_poly.evalMonomial2D(m, xD, hD, eval_pt);
                    val += u_ploy_coeff[c * dim_poly + m] * m_val;
                }
                u_vals[c] = val;
            }

            // 步骤4：非对角线元素（i≠j）
            if (i != j) {
                int i_sym = (i < j) ? i : j;
                int j_sym = (i < j) ? j : i;
                double bar_c = cij[i_sym][j_sym] - cstar;
                return -bar_c * u_vals[i];
            }
            // 步骤5：对角线元素（i=j）
            else {
                double sum = 0.0;
                for (int k_idx = 0; k_idx < n; ++k_idx) {
                    if (k_idx == i) continue;
                    int i_sym = (i < k_idx) ? i : k_idx;
                    int k_sym = (i < k_idx) ? k_idx : i;
                    double bar_c = cij[i_sym][k_sym] - cstar;
                    sum += bar_c * u_vals[k_idx];
                }
                return sum;
            }
        };

    std::cout << "MS Problem 3 initialized: 3 components\n";
    std::cout << "  Mapping: HalfAnnulus (r ∈ [0.5, 1.5], θ ∈ [-π/2, π/2])\n";
    std::cout << "  c* = " << cstar << "\n";
    std::cout << "  Type: steady diffusion (no manufactured solution)\n";
    return true;
}

// ============================================================================
// TriBlockSwirlMapping：生成器 G = T∘S
//
// 逐行对应 vem/mesh/TriBlockSwirl/tri_block_swirl_gen.py。
// 符号与设计文档 §4-§6 一致。
// ============================================================================

// 分级函数 g_β(u) = (e^{βu} - 1)/(e^β - 1)，g_β(0)=0, g_β(1)=1
// β=0 时退化为恒等（用 expm1 避免 β 小时的抵消误差）
double TriBlockSwirlMapping::grade(double u, double b) {
    if (std::abs(b) < 1e-12) return u;
    return std::expm1(b * u) / std::expm1(b);
}

// g_β'(u) = β e^{βu}/(e^β - 1)，恒 > 0（β 取任意符号均成立）
double TriBlockSwirlMapping::grade_deriv(double u, double b) {
    if (std::abs(b) < 1e-12) return 1.0;
    return b * std::exp(b * u) / std::expm1(b);
}

// 分界线 x 偏移：smoothstep，s'(0) = s'(1) = 0
// ⇒ 分界线与上下外边界正交相交，交点处不产生尖角
double TriBlockSwirlMapping::s_of(double y) const {
    return p_.s_b + (p_.s_t - p_.s_b) * (3.0 * y * y - 2.0 * y * y * y);
}

// 全局纵向分级：单一光滑函数，不分段。
// 这是「真三块」的关键——左块与右侧两块的纵向坐标来自同一个 v，
// 故分界线上左右节点自动一致，且左块内部 ∂/∂η 处处连续。
double TriBlockSwirlMapping::v_of(double eta) const {
    return grade(eta, p_.bv);
}

// 分界曲线 D(η)：贯穿全高，分开左块与右侧
Point2D TriBlockSwirlMapping::curve_D(double eta) const {
    double y = v_of(eta);
    return Point2D(s_of(y), y);
}

// 水平界面 H(p)：从 T 型节点 D(c) 出发，终止于 (1, y_r)。
// t(1-t) 在两端为 0，故 H(0) = D(c)、H(1) = (1, y_r) 精确成立。
// 块 2 与块 3 使用完全相同的 H 与相同的 bh ⇒ 共享曲边逐点同一（(R2)）。
Point2D TriBlockSwirlMapping::curve_H(double pp) const {
    Point2D d = curve_D(p_.c);
    double t = grade(pp, p_.bh);
    double x = d.x + t * (1.0 - d.x);
    double y = d.y + (p_.y_r - d.y) * t + p_.A_H * t * (1.0 - t);
    return Point2D(x, y);
}

// 块 1（左，ξ ∈ [0, a]）
//   u = 0 ⇒ x = 0（左边界精确为直线）
//   u = 1 ⇒ 恰为分界曲线 D(η)
//   η = 0/1 ⇒ y = 0/1（底顶边界在左块段内也精确为直线）
Point2D TriBlockSwirlMapping::block1(double xi, double eta) const {
    double u = grade(xi / p_.a, p_.b1);
    double y = v_of(eta);
    return Point2D(u * s_of(y), y);
}

namespace {

// Coons（超限插值）面片：四角一致时精确复现四条规定边
// L, R 以 q 为参数；B, T 以 p 为参数
inline Point2D coons_patch(double pp, double qq,
                           const Point2D& L, const Point2D& R,
                           const Point2D& B, const Point2D& T,
                           const Point2D& L0, const Point2D& L1,
                           const Point2D& R0, const Point2D& R1) {
    double x = (1.0 - pp) * L.x + pp * R.x + (1.0 - qq) * B.x + qq * T.x
             - ((1.0 - pp) * (1.0 - qq) * L0.x + pp * (1.0 - qq) * R0.x
                + (1.0 - pp) * qq * L1.x + pp * qq * R1.x);
    double y = (1.0 - pp) * L.y + pp * R.y + (1.0 - qq) * B.y + qq * T.y
             - ((1.0 - pp) * (1.0 - qq) * L0.y + pp * (1.0 - qq) * R0.y
                + (1.0 - pp) * qq * L1.y + pp * qq * R1.y);
    return Point2D(x, y);
}

}  // anonymous namespace

// 块 3（右下）：p = (ξ-a)/(1-a), q = η/c
//   L = D 的下半段, R = R₃, B = 底边, T = H
Point2D TriBlockSwirlMapping::block3(double xi, double eta) const {
    double pp = (xi - p_.a) / (1.0 - p_.a);
    double qq = eta / p_.c;

    Point2D L(curve_D(qq * p_.c));
    Point2D R(1.0, grade(qq, p_.br3) * p_.y_r);
    Point2D B(p_.s_b + grade(pp, p_.bb3) * (1.0 - p_.s_b), 0.0);
    Point2D T(curve_H(pp));

    Point2D L0(curve_D(0.0));
    Point2D L1(curve_D(p_.c));
    Point2D R0(1.0, 0.0);
    Point2D R1(1.0, p_.y_r);
    return coons_patch(pp, qq, L, R, B, T, L0, L1, R0, R1);
}

// 块 2（右上）：p = (ξ-a)/(1-a), q = (η-c)/(1-c)
//   L = D 的上半段, R = R₂, B = H（与块 3 的 T 完全相同）, T = 顶边
Point2D TriBlockSwirlMapping::block2(double xi, double eta) const {
    double pp = (xi - p_.a) / (1.0 - p_.a);
    double qq = (eta - p_.c) / (1.0 - p_.c);

    Point2D L(curve_D(p_.c + qq * (1.0 - p_.c)));
    Point2D R(1.0, p_.y_r + grade(qq, p_.br2) * (1.0 - p_.y_r));
    Point2D B(curve_H(pp));
    Point2D T(p_.s_t + grade(pp, p_.bt2) * (1.0 - p_.s_t), 1.0);

    Point2D L0(curve_D(p_.c));
    Point2D L1(curve_D(1.0));
    Point2D R0(1.0, p_.y_r);
    Point2D R1(1.0, 1.0);
    return coons_patch(pp, qq, L, R, B, T, L0, L1, R0, R1);
}

// 全局 swirl：b = [16 X(1-X) Y(1-Y)]^m 在整条外边界上为 0
// ⇒ T 在边界上逐点恒等 ⇒ 外边界保持直，物理域严格 [0,1]²；
// b > 0 在内部处处成立 ⇒ 不存在「未扭曲的近边界层」。
// 归一化因子 16 使中心处 b = 1，即中心旋转恰为 alpha0。
Point2D TriBlockSwirlMapping::swirl(double X, double Y) const {
    double w = 16.0 * X * (1.0 - X) * Y * (1.0 - Y);
    // X, Y 由 S 映出，理论上落在 [0,1]²，w >= 0。夹一下防止浮点越界后
    // std::pow(负数, 非整数) 产生 NaN——那种错会静默污染整张误差表。
    if (w < 0.0) w = 0.0;
    double b = (p_.m == 1) ? w : std::pow(w, static_cast<double>(p_.m));

    // 旋转支点独立于 T 型节点：与 w 的峰值同点，见 Params::cx/cy 注释。
    Point2D d(p_.cx, p_.cy);
    double al = p_.alpha0 * b;
    double zx = X - d.x;
    double zy = Y - d.y;
    double co = std::cos(al);
    double si = std::sin(al);
    return Point2D(d.x + co * zx - si * zy,
                   d.y + si * zx + co * zy);
}

// G = T∘S。S 按 ξ、η 落在哪一块选公式；界面上两侧公式给出同一结果
// （设计文档 §9.1 实测偏差 2.5e-16 / 0），故分支边界归属不影响结果。
Point2D TriBlockSwirlMapping::generator(double xi, double eta) const {
    Point2D X = (xi <= p_.a) ? block1(xi, eta)
              : ((eta <= p_.c) ? block3(xi, eta) : block2(xi, eta));
    return swirl(X.x, X.y);
}

// ============================================================================
// TriBlockSwirlMapping：Q8 形函数（设计文档 §7.2）
//
// 规范序：角点 0..3 位于 (-1,-1),(1,-1),(1,1),(-1,1)
//         中边点 4..7 位于 (0,-1),(1,0),(0,1),(-1,0)
// ============================================================================

namespace {

// 规范 Q8 局部坐标，与 tri_block_swirl_q8.py 的 CN/MN 一致
const double kQ8a[8] = {-1.0,  1.0, 1.0, -1.0,  0.0, 1.0, 0.0, -1.0};
const double kQ8b[8] = {-1.0, -1.0, 1.0,  1.0, -1.0, 0.0, 1.0,  0.0};

}  // anonymous namespace

void TriBlockSwirlMapping::q8_shape(double a, double b, double N[8]) {
    for (int i = 0; i < 4; ++i) {
        double ai = kQ8a[i], bi = kQ8b[i];
        N[i] = 0.25 * (1.0 + ai * a) * (1.0 + bi * b)
                    * (ai * a + bi * b - 1.0);
    }
    for (int i = 4; i < 8; ++i) {
        double ai = kQ8a[i], bi = kQ8b[i];
        if (ai == 0.0) N[i] = 0.5 * (1.0 - a * a) * (1.0 + bi * b);
        else           N[i] = 0.5 * (1.0 + ai * a) * (1.0 - b * b);
    }
}

void TriBlockSwirlMapping::q8_shape_deriv(double a, double b,
                                          double dNda[8], double dNdb[8]) {
    for (int i = 0; i < 4; ++i) {
        double ai = kQ8a[i], bi = kQ8b[i];
        dNda[i] = 0.25 * ai * (1.0 + bi * b) * (2.0 * ai * a + bi * b);
        dNdb[i] = 0.25 * bi * (1.0 + ai * a) * (ai * a + 2.0 * bi * b);
    }
    for (int i = 4; i < 8; ++i) {
        double ai = kQ8a[i], bi = kQ8b[i];
        if (ai == 0.0) {
            dNda[i] = -a * (1.0 + bi * b);
            dNdb[i] = 0.5 * bi * (1.0 - a * a);
        } else {
            dNda[i] = 0.5 * ai * (1.0 - b * b);
            dNdb[i] = -b * (1.0 + ai * a);
        }
    }
}

// ============================================================================
// TriBlockSwirlMapping：建表
//
// 逐单元从它自己的 4 个参考角点构造 8 个 Q8 节点：
//   角点取 cell_nodes，中边点取相邻两角点的算术平均，
//   对这 8 个参考点各求一次 G。
//
// (R2) 共享边逐点同一按构造成立：相邻单元看到同一对全局节点，
//   (x0+x1)/2 与 (x1+x0)/2 在 IEEE 下逐位相同 ⇒ 共享的 3 个 Q8 节点数值同一。
//
// cell_nodes 保证逆时针，但起始角点在单元间循环变化（实测 square_2x2 是
//   右下→右上→左上→左下），所以这里按每个角点在参考域相对单元中心的实际
//   位置定 (a_i, b_i)，把 8 个节点**重排成规范 Q8 序**。之后雅可比就能用
//   教科书固定公式，起始角点的差异被完全吸收在建表里。
// ============================================================================

void TriBlockSwirlMapping::on_mesh_set() {
    cells_.clear();
    if (mesh_data_ == nullptr) return;

    const vem::StraightMeshData& md = *mesh_data_;
    cells_.resize(md.num_cells);

    for (int e = 0; e < md.num_cells; ++e) {
        int nv = md.nodes_per_cell[e];
        if (nv != 4) {
            throw std::runtime_error(
                "TriBlockSwirlMapping 只支持四边形单元，单元 "
                + std::to_string(e) + " 有 " + std::to_string(nv) + " 个节点");
        }
        int start = md.cell_node_indices[e];

        // 4 个角点的参考坐标
        double cx[4], cy[4];
        for (int i = 0; i < 4; ++i) {
            int nid = md.cell_nodes[start + i];
            cx[i] = md.node_coords[2 * nid];
            cy[i] = md.node_coords[2 * nid + 1];
        }

        // 参考单元的中心与边长（取外接盒；下面会校验它确实是轴对齐矩形）
        double xmin = cx[0], xmax = cx[0], ymin = cy[0], ymax = cy[0];
        for (int i = 1; i < 4; ++i) {
            xmin = std::min(xmin, cx[i]);  xmax = std::max(xmax, cx[i]);
            ymin = std::min(ymin, cy[i]);  ymax = std::max(ymax, cy[i]);
        }
        CellQ8& q = cells_[e];
        q.hx = xmax - xmin;
        q.hy = ymax - ymin;
        q.xi_c  = 0.5 * (xmin + xmax);
        q.eta_c = 0.5 * (ymin + ymax);

        if (q.hx <= 0.0 || q.hy <= 0.0) {
            throw std::runtime_error(
                "TriBlockSwirlMapping：单元 " + std::to_string(e)
                + " 的参考单元退化（边长为 0）");
        }

        // 按参考域几何定每个角点的局部坐标 (a_i, b_i) ∈ {±1}²，
        // 并顺手校验这确实是轴对齐矩形——Q8 的 (a,b) 仿射拉回依赖这一点。
        // 不是矩形就抛，不静默套用一个错的局部坐标。
        const double tol = 1e-10;
        int slot_of_corner[4];
        for (int i = 0; i < 4; ++i) {
            double dx = cx[i] - q.xi_c;
            double dy = cy[i] - q.eta_c;
            if (std::abs(std::abs(dx) - 0.5 * q.hx) > tol ||
                std::abs(std::abs(dy) - 0.5 * q.hy) > tol) {
                throw std::runtime_error(
                    "TriBlockSwirlMapping：单元 " + std::to_string(e)
                    + " 的参考单元不是轴对齐矩形，Q8 局部坐标无法定义");
            }
            double ai = (dx > 0.0) ? 1.0 : -1.0;
            double bi = (dy > 0.0) ? 1.0 : -1.0;
            // 找到 (ai,bi) 在规范序里的槽位
            int slot = -1;
            for (int k = 0; k < 4; ++k) {
                if (kQ8a[k] == ai && kQ8b[k] == bi) { slot = k; break; }
            }
            if (slot < 0) {
                throw std::runtime_error("TriBlockSwirlMapping：角点槽位定位失败");
            }
            slot_of_corner[i] = slot;
        }
        // 四个角点必须占满四个不同槽位（否则说明有重合角点）
        bool seen[4] = {false, false, false, false};
        for (int i = 0; i < 4; ++i) {
            if (seen[slot_of_corner[i]]) {
                throw std::runtime_error(
                    "TriBlockSwirlMapping：单元 " + std::to_string(e)
                    + " 的角点在参考域内重合");
            }
            seen[slot_of_corner[i]] = true;
        }

        // 角点：G 在参考角点处求值，放进规范槽位
        for (int i = 0; i < 4; ++i) {
            Point2D g = generator(cx[i], cy[i]);
            int slot = slot_of_corner[i];
            q.nx[slot] = g.x;
            q.ny[slot] = g.y;
        }

        // 中边点：参考坐标取相邻两角点的算术平均。
        // 局部坐标是两端槽位的中点，据此定规范槽位（4..7）。
        for (int i = 0; i < 4; ++i) {
            int j = (i + 1) % 4;
            double mxi  = 0.5 * (cx[i] + cx[j]);
            double meta = 0.5 * (cy[i] + cy[j]);

            double ma = 0.5 * (kQ8a[slot_of_corner[i]] + kQ8a[slot_of_corner[j]]);
            double mb = 0.5 * (kQ8b[slot_of_corner[i]] + kQ8b[slot_of_corner[j]]);
            int slot = -1;
            for (int k = 4; k < 8; ++k) {
                if (kQ8a[k] == ma && kQ8b[k] == mb) { slot = k; break; }
            }
            if (slot < 0) {
                throw std::runtime_error(
                    "TriBlockSwirlMapping：单元 " + std::to_string(e)
                    + " 的中边点槽位定位失败（角点顺序不是四边形环序）");
            }

            Point2D g = generator(mxi, meta);
            q.nx[slot] = g.x;
            q.ny[slot] = g.y;
        }
    }
}

// ============================================================================
// TriBlockSwirlMapping：6 个映射查询
// ============================================================================

const TriBlockSwirlMapping::CellQ8&
TriBlockSwirlMapping::cell(int cell_idx) const {
    if (cells_.empty()) {
        throw std::runtime_error(
            "TriBlockSwirlMapping 尚未建表：必须先调用 set_mesh()");
    }
    if (cell_idx < 0 || cell_idx >= static_cast<int>(cells_.size())) {
        throw std::out_of_range(
            "TriBlockSwirlMapping：单元编号 " + std::to_string(cell_idx)
            + " 越界（共 " + std::to_string(cells_.size()) + " 个单元）");
    }
    return cells_[cell_idx];
}

Point2D TriBlockSwirlMapping::physical_coords(int cell_idx,
                                              double xi, double eta) const {
    const CellQ8& q = cell(cell_idx);
    double a = 2.0 * (xi - q.xi_c) / q.hx;
    double b = 2.0 * (eta - q.eta_c) / q.hy;
    double N[8];
    q8_shape(a, b, N);
    double x = 0.0, y = 0.0;
    for (int i = 0; i < 8; ++i) {
        x += N[i] * q.nx[i];
        y += N[i] * q.ny[i];
    }
    return Point2D(x, y);
}

// J = [ ∂x/∂ξ  ∂x/∂η ; ∂y/∂ξ  ∂y/∂η ]
// 参考单元边长 hx/hy，到 [-1,1]² 的仿射拉回给出链式因子 2/hx、2/hy。
// 每个单元用自己的 hx/hy，不假设全局均匀步长。
Tensor2D TriBlockSwirlMapping::jacobian(int cell_idx,
                                        double xi, double eta) const {
    const CellQ8& q = cell(cell_idx);
    double a = 2.0 * (xi - q.xi_c) / q.hx;
    double b = 2.0 * (eta - q.eta_c) / q.hy;
    double dNda[8], dNdb[8];
    q8_shape_deriv(a, b, dNda, dNdb);

    double dxda = 0.0, dxdb = 0.0, dyda = 0.0, dydb = 0.0;
    for (int i = 0; i < 8; ++i) {
        dxda += dNda[i] * q.nx[i];
        dxdb += dNdb[i] * q.nx[i];
        dyda += dNda[i] * q.ny[i];
        dydb += dNdb[i] * q.ny[i];
    }
    double sx = 2.0 / q.hx;
    double sy = 2.0 / q.hy;
    return Tensor2D(dxda * sx, dxdb * sy,
                    dyda * sx, dydb * sy);
}

double TriBlockSwirlMapping::jacobian_det(int cell_idx,
                                          double xi, double eta) const {
    Tensor2D J = jacobian(cell_idx, xi, eta);
    return J.xx * J.yy - J.xy * J.yx;
}

Tensor2D TriBlockSwirlMapping::jacobian_inv(int cell_idx,
                                            double xi, double eta) const {
    Tensor2D J = jacobian(cell_idx, xi, eta);
    double det = J.xx * J.yy - J.xy * J.yx;
    if (std::abs(det) < 1e-300) {
        throw std::runtime_error(
            "TriBlockSwirlMapping：单元 " + std::to_string(cell_idx)
            + " 的雅可比奇异");
    }
    double inv = 1.0 / det;
    return Tensor2D(J.yy * inv, -J.xy * inv,
                    -J.yx * inv, J.xx * inv);
}

Tensor2D TriBlockSwirlMapping::jacobian_transpose(int cell_idx,
                                                  double xi,
                                                  double eta) const {
    Tensor2D J = jacobian(cell_idx, xi, eta);
    return Tensor2D(J.xx, J.yx, J.xy, J.yy);
}

Tensor2D TriBlockSwirlMapping::jacobian_transpose_inv(int cell_idx,
                                                      double xi,
                                                      double eta) const {
    Tensor2D Ji = jacobian_inv(cell_idx, xi, eta);
    return Tensor2D(Ji.xx, Ji.yx, Ji.xy, Ji.yy);
}

}  // namespace MaxwellStefan
