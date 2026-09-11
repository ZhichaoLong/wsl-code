/**
 * CPU Darcy 算例接口：定义解析数据和几何映射，由 GPU 主机驱动选择编号。
 */
#ifndef CURVEDVEM_DARCY_PROBLEM_H
#define CURVEDVEM_DARCY_PROBLEM_H

#include <cmath>
#include <vector>
#include <functional>
#include <stdexcept>
#include <memory>

namespace Darcy {

// ============================================================================
// 基础向量/矩阵类型定义
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
// 曲边映射基类（可以根据不同算例派生）
// ============================================================================

class IsoparametricMapping {
public:
    IsoparametricMapping() = default;
    virtual ~IsoparametricMapping() = default;

    // 计算物理坐标 x(ξ, η)
    virtual Point2D physical_coords(double xi, double eta) const = 0;

    // 计算雅可比矩阵
    virtual Tensor2D jacobian(double xi, double eta) const = 0;

    // 计算雅可比行列式
    virtual double jacobian_det(double xi, double eta) const = 0;

    // 计算雅可比矩阵的逆
    virtual Tensor2D jacobian_inv(double xi, double eta) const = 0;

    // 计算雅可比矩阵的转置
    virtual Tensor2D jacobian_transpose(double xi, double eta) const = 0;

    // 计算雅可比矩阵的转置的逆
    virtual Tensor2D jacobian_transpose_inv(double xi, double eta) const = 0;
};

// ============================================================================
// 示例1：正弦扰动的曲边映射（来自 md 文档）
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
            throw std::runtime_error("Jacobian is singular!");
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
    double eps;  // 扰动幅值
};

// ============================================================================
// 恒等映射（用于直边网格）
// ============================================================================

class IdentityMapping : public IsoparametricMapping {
public:
    IdentityMapping() = default;

    Point2D physical_coords(double xi, double eta) const override {
        return Point2D(xi, eta);
    }

    Tensor2D jacobian(double xi, double eta) const override {
        (void)xi; (void)eta; // 标记未使用
        return Tensor2D(1.0, 0.0, 0.0, 1.0);
    }

    double jacobian_det(double xi, double eta) const override {
        (void)xi; (void)eta; // 标记未使用
        return 1.0;
    }

    Tensor2D jacobian_inv(double xi, double eta) const override {
        (void)xi; (void)eta; // 标记未使用
        return Tensor2D(1.0, 0.0, 0.0, 1.0);
    }

    Tensor2D jacobian_transpose(double xi, double eta) const override {
        (void)xi; (void)eta; // 标记未使用
        return Tensor2D(1.0, 0.0, 0.0, 1.0);
    }

    Tensor2D jacobian_transpose_inv(double xi, double eta) const override {
        (void)xi; (void)eta; // 标记未使用
        return Tensor2D(1.0, 0.0, 0.0, 1.0);
    }
};

// ============================================================================
// Darcy 方程 PDE 数据
// ============================================================================

struct DarcyPdeData {
    // 渗透张量 κ(x,y) - 2x2 对称张量
    std::function<Tensor2D(double x, double y)> kappa;

    // 压力 Dirichlet 边界条件
    std::function<double(double x, double y)> dirichlet_p;

    // 法向通量 Neumann 边界条件
    std::function<double(double x, double y, double nx, double ny)> neumann_un;

    // 真解压力 p(x,y)
    std::function<double(double x, double y)> solution_p;

    // 真解速度 u(x,y)
    std::function<Vector2D(double x, double y)> solution_u;

    // 散度源项 g(x,y)
    std::function<double(double x, double y)> source_g;
};

// ============================================================================
// Darcy 算例管理类
// ============================================================================

class DarcyProblem {
public:
    DarcyProblem() : problem_index_(0) {}//这是pde类型分类
    ~DarcyProblem() = default;

    // 禁用拷贝
    DarcyProblem(const DarcyProblem&) = delete;
    DarcyProblem& operator=(const DarcyProblem&) = delete;

    // 设置算例编号
    void set_problem_index(int index) { problem_index_ = index; }

    // 初始化算例数据
    bool init_problem();

    // 获取 PDE 数据
    const DarcyPdeData& get_pde_data() const { return pde_data_; }

    // 获取曲边映射
    const IsoparametricMapping& get_mapping() const {
        if (!mapping_) {
            throw std::runtime_error("Darcy problem is not initialized!");
        }
        return *mapping_;
    }

    // 直边计算区域上的真解和系数（由物理区域通过映射拉回）
    double computational_solution_p(double xi, double eta) const;
    Vector2D computational_solution_u(double xi, double eta) const;
    double computational_source_g(double xi, double eta) const;
    Tensor2D computational_kappa(double xi, double eta) const;

private:
    int problem_index_;
    DarcyPdeData pde_data_;
    std::unique_ptr<IsoparametricMapping> mapping_;

    // 初始化具体算例
    bool init_problem_1();  // 正弦真解 + eps=0.05 曲边映射
    bool init_problem_2();  // 同一真解 + eps=0 退化直边映射
};

}  // namespace Darcy

#endif  // CURVEDVEM_DARCY_PROBLEM_H
