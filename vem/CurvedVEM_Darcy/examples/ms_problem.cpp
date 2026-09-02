#include "ms_problem.h"
#include "../lib/polynomial_basis.h"

#include <iostream>
#include <limits>

namespace MaxwellStefan {

// ============================================================================
// 算例管理：初始化
// ============================================================================

bool MSProblem::init_problem() {
    switch (problem_index_) {
        case 1:
            return init_problem_1();
        case 2:
            return init_problem_2();
        case 3:
            return init_problem_3();
        default:
            std::cerr << "未知算例编号: " << problem_index_ << std::endl;
            return false;
    }
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
    pde_data_.initial_concentration_comp = [](int i, double xi, double eta) {
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
    pde_data_.initial_concentration = [this](int i, double x, double y) {
        Point2D comp = invert_mapping(x, y);
        return pde_data_.initial_concentration_comp(i, comp.x, comp.y);
    };

    // 源项：物理域和计算域均为 0
    pde_data_.source_f = [](int /*i*/, double /*x*/, double /*y*/, double /*t*/) {
        return 0.0;
    };
    pde_data_.source_f_comp = [](int /*i*/, double /*xi*/, double /*eta*/, double /*t*/) {
        return 0.0;
    };

    // 法向通量边界条件：零通量（所有组分的法向通量均为 0）
    // Piola 变换保持法向通量为零，物理域和计算域都是 0
    pde_data_.boundary_flux = [](int /*i*/, double /*x*/, double /*y*/,
                                 double /*nx*/, double /*ny*/, double /*t*/) {
        return 0.0;
    };
    pde_data_.boundary_flux_comp = [](int /*i*/, double /*xi*/, double /*eta*/,
                                       double /*nx*/, double /*ny*/, double /*t*/) {
        return 0.0;
    };

    // ========== 初值模式：物理域上评估 A_ij（纯非线性部分，不含 c*I）==========
    // 公式与计算域版本完全一致，只是浓度从物理域初值函数获取
    //   非对角元 (i≠j): A_ij = -bar_c_ij * u_i
    //   对角元   (i=j): A_ii = Σ_{k≠i} bar_c_ik * u_k
    pde_data_.evaluate_A_initial =
        [this, n](int i, int j, double x, double y) {
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
                double u_i = init(i, x, y);
                return -bar_c * u_i;
            } else {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    if (k == i) continue;
                    int i_sym = (i < k) ? i : k;
                    int k_sym = (i < k) ? k : i;
                    double bar_c = cij[i_sym][k_sym] - cstar;
                    double u_k = init(k, x, y);
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
        [this, n](int i, int j, double xi, double eta) {
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
                double u_i = init(i, xi, eta);
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
                    double u_k = init(k, xi, eta);
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
// 逆映射：由物理坐标 (x,y) 求计算域坐标 (ξ,η)
// 牛顿迭代求解 x(ξ,η) = x_target, y(ξ,η) = y_target
// ============================================================================

Point2D MSProblem::invert_mapping(double x, double y) const {
    if (!mapping_) {
        throw std::runtime_error("MSProblem::invert_mapping: mapping not initialized");
    }

    // 初始猜测：恒等映射 (ξ=x, η=y)
    double xi = x;
    double eta = y;

    const int max_iter = 50;
    const double tol = 1e-14;

    for (int iter = 0; iter < max_iter; ++iter) {
        Point2D phys = mapping_->physical_coords(xi, eta);
        double rx = phys.x - x;
        double ry = phys.y - y;

        if (std::abs(rx) < tol && std::abs(ry) < tol)
            break;

        // 雅可比矩阵 J = [[dxdxi, dxdeta], [dydxi, dydeta]]
        Tensor2D J = mapping_->jacobian(xi, eta);
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

bool MSProblem::init_problem_2() {
    // 曲边映射：与算例 1 相同
    mapping_.reset(new SinPerturbationMapping(0.05));

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
        [this](int i, double xi, double eta, double t) {
            Point2D p = mapping_->physical_coords(xi, eta);
            return pde_data_.exact_concentration(i, p.x, p.y, t);
        };

    // ========== 初值（真解在 t=0 处的值） ==========
    pde_data_.initial_concentration = [this](int i, double x, double y) {
        return pde_data_.exact_concentration(i, x, y, 0.0);
    };
    pde_data_.initial_concentration_comp =
        [this](int i, double xi, double eta) {
            return pde_data_.exact_concentration_comp(i, xi, eta, 0.0);
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
        [this](int i, double xi, double eta, double t) -> Vector2D {
            Point2D p = mapping_->physical_coords(xi, eta);
            Vector2D Jp = pde_data_.exact_flux(i, p.x, p.y, t);
            Tensor2D Jfwd = mapping_->jacobian(xi, eta);
            double detJ = mapping_->jacobian_det(xi, eta);
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
        [this](int i, double xi, double eta, double t) {
            Point2D p = mapping_->physical_coords(xi, eta);
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
        [this](int i, double xi, double eta,
               double nx, double ny, double t) {
            Vector2D Jhat = pde_data_.exact_flux_comp(i, xi, eta, t);
            return Jhat.x * nx + Jhat.y * ny;
        };

    // ========== 初值模式 A_ij（物理域） ==========
    // 注意：返回 A_bar（仅 bar_c 部分，不含 cstar 对角元）
    //       因为 cstar I 由 cmin_G 单独组装
    pde_data_.evaluate_A_initial =
        [this, &bar_c](int i, int j, double x, double y) {
            double c[3];
            for (int k = 0; k < 3; ++k)
                c[k] = pde_data_.initial_concentration(k, x, y);
            double A[3][3];
            build_ABC_bar(bar_c, c, A);
            return A[i][j];
        };

    // ========== 初值模式 A_ij（计算域） ==========
    // 注意：返回 A_bar（仅 bar_c 部分，不含 cstar 对角元）
    //       因为 cstar I 由 cmin_G 单独组装
    pde_data_.evaluate_A_comp_initial =
        [this, &bar_c](int i, int j, double xi, double eta) {
            double c[3];
            for (int k = 0; k < 3; ++k)
                c[k] = pde_data_.initial_concentration_comp(k, xi, eta);
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

    std::cout << "MS Problem 2 initialized: 3 components\n";
    std::cout << "  Mapping: SinPerturbation (eps=0.05)\n";
    std::cout << "  c* = " << cstar << "\n";
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
    pde_data_.initial_concentration_comp = [](int i, double xi, double eta) {
        (void)eta;
        double c0 = (xi <= 0.25) ? 1.0 : 0.0;
        double c1 = (xi >= 0.75) ? 1.0 : 0.0;
        if (i == 0) return c0;
        if (i == 1) return c1;
        return 1.0 - c0 - c1;
    };

    // 物理域上的初始浓度：通过逆映射找 (ξ,η)
    pde_data_.initial_concentration = [this](int i, double x, double y) {
        Point2D comp = invert_mapping(x, y);
        return pde_data_.initial_concentration_comp(i, comp.x, comp.y);
    };

    // 源项：0
    pde_data_.source_f = [](int /*i*/, double /*x*/, double /*y*/, double /*t*/) {
        return 0.0;
    };
    pde_data_.source_f_comp = [](int /*i*/, double /*xi*/, double /*eta*/, double /*t*/) {
        return 0.0;
    };

    // 边界条件：零法向通量（所有边界）
    pde_data_.boundary_flux = [](int /*i*/, double /*x*/, double /*y*/,
                                 double /*nx*/, double /*ny*/, double /*t*/) {
        return 0.0;
    };
    pde_data_.boundary_flux_comp = [](int /*i*/, double /*xi*/, double /*eta*/,
                                       double /*nx*/, double /*ny*/, double /*t*/) {
        return 0.0;
    };

    // ===== 初值模式：物理域上评估 A_ij =====
    pde_data_.evaluate_A_initial =
        [this, n](int i, int j, double x, double y) {
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
                double u_i = init(i, x, y);
                return -bar_c * u_i;
            } else {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    if (k == i) continue;
                    int is = (i < k) ? i : k;
                    int ks = (i < k) ? k : i;
                    double bar_c = cij[is][ks] - cstar_val;
                    double u_k = init(k, x, y);
                    sum += bar_c * u_k;
                }
                return sum;
            }
        };

    // ===== 初值模式：计算域上评估 A_ij =====
    pde_data_.evaluate_A_comp_initial =
        [this, n](int i, int j, double xi, double eta) {
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
                double u_i = init(i, xi, eta);
                return -bar_c * u_i;
            } else {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    if (k == i) continue;
                    int is = (i < k) ? i : k;
                    int ks = (i < k) ? k : i;
                    double bar_c = cij[is][ks] - cstar_val;
                    double u_k = init(k, xi, eta);
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

}  // namespace MaxwellStefan
