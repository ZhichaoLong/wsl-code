/**
 * CPU Darcy 1/2 算例初始化：曲边/恒等映射与物理解析解。
 * 设备端对应数据在 include/problem_device.cuh 中。
 */
#include "darcy_problem.h"
#include <iostream>

namespace Darcy {

namespace {

Vector2D multiply(const Tensor2D& matrix, const Vector2D& vector) {
    return Vector2D(
        matrix.xx * vector.x + matrix.xy * vector.y,
        matrix.yx * vector.x + matrix.yy * vector.y
    );
}

Tensor2D transform_tensor(const Tensor2D& invJ, double detJ,
                          const Tensor2D& tensor) {
    // detJ * invJ * tensor * invJ^T
    double left_xx = invJ.xx * tensor.xx + invJ.xy * tensor.yx;
    double left_xy = invJ.xx * tensor.xy + invJ.xy * tensor.yy;
    double left_yx = invJ.yx * tensor.xx + invJ.yy * tensor.yx;
    double left_yy = invJ.yx * tensor.xy + invJ.yy * tensor.yy;

    return Tensor2D(
        detJ * (left_xx * invJ.xx + left_xy * invJ.xy),
        detJ * (left_xx * invJ.yx + left_xy * invJ.yy),
        detJ * (left_yx * invJ.xx + left_yy * invJ.xy),
        detJ * (left_yx * invJ.yx + left_yy * invJ.yy)
    );
}

}  // namespace

// ============================================================================
// 直边计算区域上的解析数据
// ============================================================================

double DarcyProblem::computational_solution_p(double xi, double eta) const {
    if (!mapping_ || !pde_data_.solution_p) {
        throw std::runtime_error("Darcy problem is not initialized!");
    }

    Point2D physical_point = mapping_->physical_coords(xi, eta);
    return pde_data_.solution_p(physical_point.x, physical_point.y);
}

Vector2D DarcyProblem::computational_solution_u(double xi, double eta) const {
    if (!mapping_ || !pde_data_.solution_u) {
        throw std::runtime_error("Darcy problem is not initialized!");
    }

    Point2D physical_point = mapping_->physical_coords(xi, eta);
    Vector2D physical_velocity =
        pde_data_.solution_u(physical_point.x, physical_point.y);
    Tensor2D invJ = mapping_->jacobian_inv(xi, eta);
    double detJ = mapping_->jacobian_det(xi, eta);
    Vector2D computational_velocity = multiply(invJ, physical_velocity);
    computational_velocity.x *= detJ;
    computational_velocity.y *= detJ;
    return computational_velocity;
}

double DarcyProblem::computational_source_g(double xi, double eta) const {
    if (!mapping_ || !pde_data_.source_g) {
        throw std::runtime_error("Darcy problem is not initialized!");
    }

    Point2D physical_point = mapping_->physical_coords(xi, eta);
    return mapping_->jacobian_det(xi, eta) *
           pde_data_.source_g(physical_point.x, physical_point.y);
}

Tensor2D DarcyProblem::computational_kappa(double xi, double eta) const {
    if (!mapping_ || !pde_data_.kappa) {
        throw std::runtime_error("Darcy problem is not initialized!");
    }

    Point2D physical_point = mapping_->physical_coords(xi, eta);
    Tensor2D physical_kappa =
        pde_data_.kappa(physical_point.x, physical_point.y);
    Tensor2D invJ = mapping_->jacobian_inv(xi, eta);
    double detJ = mapping_->jacobian_det(xi, eta);
    return transform_tensor(invJ, detJ, physical_kappa);
}

// ============================================================================
// 初始化算例
// ============================================================================

bool DarcyProblem::init_problem() {
    mapping_.reset();
    switch (problem_index_) {
        case 1:
            return init_problem_1();
        case 2:
            return init_problem_2();
        default:
            std::cerr << "错误：未知的算例编号 " << problem_index_ << std::endl;
            return false;
    }
}

// ============================================================================
// 算例1：正弦真解 + 曲边映射
// ============================================================================

bool DarcyProblem::init_problem_1() {
    // 曲边映射：正弦扰动
    mapping_.reset(new SinPerturbationMapping(0.05));

    // 渗透张量 κ = I（单位张量）
    pde_data_.kappa = [](double x, double y) -> Tensor2D {
        (void)x; (void)y; // 标记未使用
        return Tensor2D(1.0, 0.0, 0.0, 1.0);
    };

    // 真解压力 p = sin(πx) * cos(πy)
    pde_data_.solution_p = [](double x, double y) -> double {
        return std::sin(M_PI * x) * std::cos(M_PI * y);
    };

    // 真解速度 u = -κ∇p
    pde_data_.solution_u = [](double x, double y) -> Vector2D {
        double dpdx = M_PI * std::cos(M_PI * x) * std::cos(M_PI * y);
        double dpdy = -M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
        return Vector2D(-dpdx, -dpdy);
    };

    // 散度源项 g = ∇·u = 2π² sin(πx) cos(πy)
    pde_data_.source_g = [](double x, double y) -> double {
        return 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y);
    };

    // Dirichlet 边界条件：用真解
    pde_data_.dirichlet_p = [](double x, double y) -> double {
        return std::sin(M_PI * x) * std::cos(M_PI * y);
    };

    // Neumann 边界条件：u·n
    pde_data_.neumann_un = [](double x, double y, double nx, double ny) -> double {
        (void)x; (void)y; // 暂时不使用
        double dpdx = M_PI * std::cos(M_PI * x) * std::cos(M_PI * y);
        double dpdy = -M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
        // u = -κ∇p, 所以 u·n = -∇p·n
        return - (dpdx * nx + dpdy * ny);
    };

    return true;
}

// ============================================================================
// 算例2：与算例1相同的解析数据，映射扰动 eps=0，严格退化为直边网格
// ============================================================================

bool DarcyProblem::init_problem_2() {
    // 保留同一正弦扰动映射类，只把扰动幅值设为零。这样
    // F(xi,eta)=(xi,eta), J=I, det(J)=1，可用于检查曲边实现的直边退化。
    mapping_.reset(new SinPerturbationMapping(0.0));

    // 渗透张量 kappa = I（与算例1一致）
    pde_data_.kappa = [](double x, double y) -> Tensor2D {
        (void)x;
        (void)y;
        return Tensor2D(1.0, 0.0, 0.0, 1.0);
    };

    // 真解压力 p = sin(pi*x) cos(pi*y)（与算例1一致）
    pde_data_.solution_p = [](double x, double y) -> double {
        return std::sin(M_PI * x) * std::cos(M_PI * y);
    };

    // 真解速度 u = -grad(p)（与算例1一致）
    pde_data_.solution_u = [](double x, double y) -> Vector2D {
        double dpdx = M_PI * std::cos(M_PI * x) * std::cos(M_PI * y);
        double dpdy = -M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
        return Vector2D(-dpdx, -dpdy);
    };

    // 散度源项 g = 2*pi^2 sin(pi*x) cos(pi*y)
    pde_data_.source_g = [](double x, double y) -> double {
        return 2.0 * M_PI * M_PI *
               std::sin(M_PI * x) * std::cos(M_PI * y);
    };

    pde_data_.dirichlet_p = [](double x, double y) -> double {
        return std::sin(M_PI * x) * std::cos(M_PI * y);
    };

    pde_data_.neumann_un = [](
        double x, double y, double nx, double ny) -> double {
        double dpdx = M_PI * std::cos(M_PI * x) * std::cos(M_PI * y);
        double dpdy = -M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
        return -(dpdx * nx + dpdy * ny);
    };

    return true;
}

}  // namespace Darcy
