/*
 * gauss_quadrature.h
 * 高斯积分模块 - 支持一维线段和二维三角形积分
 * Author: New VEM Architecture
 * Date: 2026-07-14
 */

#ifndef CURVEDVEM_GAUSS_QUADRATURE_H
#define CURVEDVEM_GAUSS_QUADRATURE_H

#include <vector>
#include <stdexcept>
#include <cmath>

namespace vem {

// 积分类型
enum class QuadratureType {
    GAUSS_LEGENDRE,  // 高斯-勒让德积分（内部点）
    GAUSS_LOBATTO    // 高斯-洛巴托积分（包含端点）
};

// 积分点结构体
struct QuadPoint {
    std::vector<double> x;  // 坐标
    double w;               // 权重

    QuadPoint() : w(0.0) {}
    QuadPoint(const std::vector<double>& coords, double weight)
        : x(coords), w(weight) {}
    QuadPoint(double x1, double weight) : x{x1}, w(weight) {}
    QuadPoint(double x1, double x2, double weight) : x{x1, x2}, w(weight) {}
};

class GaussQuadrature {
public:
    GaussQuadrature() = default;
    ~GaussQuadrature() = default;

    // ========== 一维积分 ==========
    // 获取标准区间[-1,1]的积分点
    std::vector<QuadPoint> get_1D_points(QuadratureType type, int order) const;

    // 获取任意区间[a,b]的积分点
    std::vector<QuadPoint> get_1D_points(QuadratureType type, int order,
                                          double a, double b) const;

    // ========== 二维线段积分（平面上的线段） ==========
    std::vector<QuadPoint> get_line_2D_points(QuadratureType type, int order,
                                                double x1, double y1,
                                                double x2, double y2) const;

    // ========== 二维三角形积分 ==========
    // 获取标准三角形的积分点 (标准三角形：(0,0),(1,0),(0,1))
    std::vector<QuadPoint> get_triangle_reference_points(int n_points) const;

    // 获取任意三角形的积分点
    // vertices: [x1,y1,x2,y2,x3,y3]
    std::vector<QuadPoint> get_triangle_points(const std::vector<double>& vertices,
                                                 int n_points) const;

private:
    // 生成标准区间[-1,1]的Gauss-Legendre积分点
    std::vector<QuadPoint> generate_gauss_legendre(int order) const;

    // 生成标准区间[-1,1]的Gauss-Lobatto积分点
    std::vector<QuadPoint> generate_gauss_lobatto(int order) const;

    // 生成标准三角形的积分点
    std::vector<QuadPoint> generate_triangle(int n_points) const;

    // 映射一维积分点到实际区间
    std::vector<QuadPoint> map_1D(const std::vector<QuadPoint>& ref_points,
                                   double a, double b) const;

    // 映射三角形积分点到实际三角形
    std::vector<QuadPoint> map_triangle(const std::vector<QuadPoint>& ref_points,
                                         const std::vector<double>& vertices) const;
};

} // namespace vem

#endif // CURVEDVEM_GAUSS_QUADRATURE_H
