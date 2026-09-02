#ifndef CURVEDVEM_MS_PROBLEM_H
#define CURVEDVEM_MS_PROBLEM_H

#include <cmath>
#include <vector>
#include <functional>
#include <stdexcept>
#include <memory>
#include <string>

namespace MaxwellStefan {

// ============================================================================
// 基础向量/矩阵类型定义（与 Darcy 保持一致）
// ============================================================================

struct Point2D {
    double x, y;
    Point2D() : x(0.0), y(0.0) {}
    Point2D(double x_, double y_) : x(x_), y(y_) {}
};

struct Vector2D {
    double x, y;
    Vector2D() : x(0.0), y(0.0) {}
    Vector2D(double x_, double y_) : x(x_), y(y_) {}
};

struct Tensor2D {
    double xx, xy, yx, yy;
    Tensor2D() : xx(1.0), xy(0.0), yx(0.0), yy(1.0) {}
    Tensor2D(double xx_, double xy_, double yx_, double yy_)
        : xx(xx_), xy(xy_), yx(yx_), yy(yy_) {}
};

// ============================================================================
// 曲边映射基类
// ============================================================================

class IsoparametricMapping {
public:
    IsoparametricMapping() = default;
    virtual ~IsoparametricMapping() = default;

    virtual Point2D physical_coords(double xi, double eta) const = 0;
    virtual Tensor2D jacobian(double xi, double eta) const = 0;
    virtual double jacobian_det(double xi, double eta) const = 0;
    virtual Tensor2D jacobian_inv(double xi, double eta) const = 0;
    virtual Tensor2D jacobian_transpose(double xi, double eta) const = 0;
    virtual Tensor2D jacobian_transpose_inv(double xi, double eta) const = 0;
};

// ============================================================================
// 正弦扰动的曲边映射（与 Darcy 相同形式，默认 eps=0.05）
//   x = ξ + eps * sin(2π η)
//   y = η + eps * sin(2π ξ)
// ============================================================================

class SinPerturbationMapping : public IsoparametricMapping {
public:
    SinPerturbationMapping(double eps_ = 0.05) : eps(eps_) {}

    Point2D physical_coords(double xi, double eta) const override {
        double x = xi + eps * std::sin(2.0 * M_PI * eta);
        double y = eta + eps * std::sin(2.0 * M_PI * xi);
        return Point2D(x, y);
    }

    Tensor2D jacobian(double xi, double eta) const override {
        double dxdxi = 1.0;
        double dxdeta = eps * 2.0 * M_PI * std::cos(2.0 * M_PI * eta);
        double dydxi = eps * 2.0 * M_PI * std::cos(2.0 * M_PI * xi);
        double dydeta = 1.0;
        return Tensor2D(dxdxi, dxdeta, dydxi, dydeta);
    }

    double jacobian_det(double xi, double eta) const override {
        double dxdxi = 1.0;
        double dxdeta = eps * 2.0 * M_PI * std::cos(2.0 * M_PI * eta);
        double dydxi = eps * 2.0 * M_PI * std::cos(2.0 * M_PI * xi);
        double dydeta = 1.0;
        return dxdxi * dydeta - dxdeta * dydxi;
    }

    Tensor2D jacobian_inv(double xi, double eta) const override {
        double detJ = jacobian_det(xi, eta);
        if (std::abs(detJ) < 1e-14) {
            throw std::runtime_error("Jacobian is singular in MS mapping!");
        }
        Tensor2D J = jacobian(xi, eta);
        double invdet = 1.0 / detJ;
        return Tensor2D(
            J.yy * invdet,  -J.xy * invdet,
            -J.yx * invdet, J.xx * invdet
        );
    }

    Tensor2D jacobian_transpose(double xi, double eta) const override {
        Tensor2D J = jacobian(xi, eta);
        return Tensor2D(J.xx, J.yx, J.xy, J.yy);
    }

    Tensor2D jacobian_transpose_inv(double xi, double eta) const override {
        Tensor2D invJ = jacobian_inv(xi, eta);
        return Tensor2D(invJ.xx, invJ.yx, invJ.xy, invJ.yy);
    }

private:
    double eps;
};

// ============================================================================
// 半圆环映射
//   r = ξ + 0.5,  θ = π(η - 0.5)
//   x = r cosθ,  y = r sinθ
//   物理域：右半圆环，内半径 0.5，外半径 1.5，θ ∈ [-π/2, π/2]
// ============================================================================

class HalfAnnulusMapping : public IsoparametricMapping {
public:
    HalfAnnulusMapping() = default;

    Point2D physical_coords(double xi, double eta) const override {
        double r = xi + 0.5;
        double theta = M_PI * (eta - 0.5);
        double x = r * std::cos(theta);
        double y = r * std::sin(theta);
        return Point2D(x, y);
    }

    Tensor2D jacobian(double xi, double eta) const override {
        double r = xi + 0.5;
        double theta = M_PI * (eta - 0.5);
        double cos_t = std::cos(theta);
        double sin_t = std::sin(theta);
        // ∂x/∂ξ = cosθ,  ∂x/∂η = -r π sinθ
        // ∂y/∂ξ = sinθ,  ∂y/∂η =  r π cosθ
        double dxdxi = cos_t;
        double dxdeta = -r * M_PI * sin_t;
        double dydxi = sin_t;
        double dydeta = r * M_PI * cos_t;
        return Tensor2D(dxdxi, dxdeta, dydxi, dydeta);
    }

    double jacobian_det(double xi, double eta) const override {
        double r = xi + 0.5;
        // detJ = cosθ * r π cosθ - (-r π sinθ) * sinθ = r π (cos²θ + sin²θ) = r π
        return r * M_PI;
    }

    Tensor2D jacobian_inv(double xi, double eta) const override {
        double r = xi + 0.5;
        double theta = M_PI * (eta - 0.5);
        double cos_t = std::cos(theta);
        double sin_t = std::sin(theta);
        double detJ = r * M_PI;
        double invdet = 1.0 / detJ;
        // J = [cosθ,  -rπ sinθ; sinθ,  rπ cosθ]
        // inv(J) = (1/rπ) * [rπ cosθ,  rπ sinθ; -sinθ,  cosθ]
        //         = [cosθ,  sinθ; -sinθ/(rπ),  cosθ/(rπ)]
        return Tensor2D(
            cos_t,              sin_t,
            -sin_t / detJ,      cos_t / detJ
        );
    }

    Tensor2D jacobian_transpose(double xi, double eta) const override {
        Tensor2D J = jacobian(xi, eta);
        return Tensor2D(J.xx, J.yx, J.xy, J.yy);
    }

    Tensor2D jacobian_transpose_inv(double xi, double eta) const override {
        Tensor2D invJ = jacobian_inv(xi, eta);
        return Tensor2D(invJ.xx, invJ.yx, invJ.xy, invJ.yy);
    }
};

// ============================================================================
// 恒等映射（直边网格）
// ============================================================================

class IdentityMapping : public IsoparametricMapping {
public:
    IdentityMapping() = default;

    Point2D physical_coords(double xi, double eta) const override {
        return Point2D(xi, eta);
    }

    Tensor2D jacobian(double xi, double eta) const override {
        (void)xi; (void)eta;
        return Tensor2D(1.0, 0.0, 0.0, 1.0);
    }

    double jacobian_det(double xi, double eta) const override {
        (void)xi; (void)eta;
        return 1.0;
    }

    Tensor2D jacobian_inv(double xi, double eta) const override {
        (void)xi; (void)eta;
        return Tensor2D(1.0, 0.0, 0.0, 1.0);
    }

    Tensor2D jacobian_transpose(double xi, double eta) const override {
        (void)xi; (void)eta;
        return Tensor2D(1.0, 0.0, 0.0, 1.0);
    }

    Tensor2D jacobian_transpose_inv(double xi, double eta) const override {
        (void)xi; (void)eta;
        return Tensor2D(1.0, 0.0, 0.0, 1.0);
    }
};

// ============================================================================
// Maxwell-Stefan 方程 PDE 数据
// ============================================================================

struct MSPdeData {
    int n_components = 3;  // 组分数

    // 二元扩散系数倒数 c_ij
    std::vector<std::vector<double>> c_ij;
    double c_star;                          // 最小扩散系数倒数
    std::vector<std::vector<double>> bar_c_ij;  // c_ij - c_star

    // 计算域上的初始浓度 c_i(ξ, η)
    // 直接用计算域坐标定义，映射到物理域后自然是曲边间断
    std::function<double(int i, double xi, double eta)> initial_concentration_comp;

    // 物理域上的初始浓度 c_i(x, y)
    // 通过逆映射找到对应的 (ξ,η)，再用计算域初值求值
    std::function<double(int i, double x, double y)> initial_concentration;

    // 物理域源项 f_i(x, y, t)（质量守恒右端）
    std::function<double(int i, double x, double y, double t)> source_f;

    // 计算域源项 f̂_i(ξ, η, t) = f_i(x(ξ,η), y(ξ,η), t)
    std::function<double(int i, double xi, double eta, double t)> source_f_comp;

    // 物理域法向通量边界条件：J_i · n(x, y, t)
    std::function<double(int i, double x, double y, double nx, double ny, double t)> boundary_flux;

    // 计算域法向通量边界条件：Ĵ_i · n̂(ξ, η, t)
    // Piola 恒等式：J_i · n ds = Ĵ_i · n̂ dŝ，
    // 因此直接给定计算域法向通量 Ĵ·n̂，供边界积分使用
    std::function<double(int i, double xi, double eta, double nx, double ny, double t)> boundary_flux_comp;

    // ===== 真解（用于误差分析 / 制造解算例） =====

    // 物理域精确浓度 c_i(x, y, t)
    std::function<double(int i, double x, double y, double t)> exact_concentration;

    // 计算域精确浓度 ĉ_i(ξ, η, t) = c_i(x(ξ,η), y(ξ,η), t)
    std::function<double(int i, double xi, double eta, double t)> exact_concentration_comp;

    // 物理域精确通量矢量 J_i(x, y, t)（返回 x,y 两个分量）
    // 由本构关系 J = -Ā(c)^{-1} ∇c 计算
    std::function<Vector2D(int i, double x, double y, double t)> exact_flux;

    // 计算域精确通量矢量 Ĵ_i(ξ, η, t)
    // Piola 变换：Ĵ = J · det(J) · J^{-1} （逆变 Piola）
    std::function<Vector2D(int i, double xi, double eta, double t)> exact_flux_comp;

    // 初值模式：物理域上评估非线性扩散矩阵 A_ij（纯非线性部分，不含 c*I）
    //   A_ii(u) = Σ_{k≠i} bar_c_ik · u_k
    //   A_ij(u) = -bar_c_ji · u_i   (i ≠ j)
    // 浓度 u 从 initial_concentration 获取（内部通过逆映射求值）
    std::function<double(int i, int j, double x, double y)> evaluate_A_initial;

    // 初值模式：计算域上评估非线性扩散矩阵 A_ij（纯非线性部分，不含 c*I）
    //   公式同上，浓度 u 从 initial_concentration_comp 获取
    std::function<double(int i, int j, double xi, double eta)> evaluate_A_comp_initial;

    // 数值解模式：计算域上评估 A_ij（纯非线性部分，不含 c*I）
    // 输入单项式系数向量，先还原各组分浓度值，再计算 A_ij
    //   u_ploy_coeff: 所有组分的单项式系数平铺
    //                布局：[comp0_monom0, ..., comp1_monom0, ...]
    //                长度 = n_components * n_monomial
    //   xD_x, xD_y: 单元质心（缩放基的中心）
    //   hD: 单元特征尺度
    //   xi, eta: 计算点（计算域坐标）
    std::function<double(int i, int j, const std::vector<double>& u_ploy_coeff,
                         double xD_x, double xD_y, double hD,
                         double xi, double eta)> evaluate_A_comp_poly;
};

// ============================================================================
// MS 算例管理类
// ============================================================================

class MSProblem {
public:
    MSProblem() : problem_index_(0) {}
    ~MSProblem() = default;

    // 禁用拷贝
    MSProblem(const MSProblem&) = delete;
    MSProblem& operator=(const MSProblem&) = delete;

    void set_problem_index(int index) { problem_index_ = index; }

    bool init_problem();

    const MSPdeData& get_pde_data() const { return pde_data_; }

    const IsoparametricMapping& get_mapping() const {
        if (!mapping_) {
            throw std::runtime_error("MS problem is not initialized!");
        }
        return *mapping_;
    }

    // 由物理坐标 (x,y) 反求计算域坐标 (ξ,η)
    // 用牛顿迭代求解 x(ξ,η) = x_target, y(ξ,η) = y_target
    Point2D invert_mapping(double x, double y) const;

private:
    int problem_index_;
    MSPdeData pde_data_;
    std::unique_ptr<IsoparametricMapping> mapping_;

    // 初始化具体算例
    bool init_problem_1();
    bool init_problem_2();
    bool init_problem_3();
};

}  // namespace MaxwellStefan

#endif  // CURVEDVEM_MS_PROBLEM_H
