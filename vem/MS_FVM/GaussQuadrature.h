/*
 * GaussQuadrature.h
 * 高斯积分类，支持一维线段(Gauss-Legendre、Gauss-Lobatto)和二维三角形的高斯积分
 * 为三维积分预留接口
 * Author: Generated with Claude Code
 * Date: 2026-03-16
 */

#ifndef GAUSS_QUADRATURE_H
#define GAUSS_QUADRATURE_H

#include <vector>
#include <stdexcept>
#include <cmath>

namespace GaussQ{

// 积分类型枚举
enum class QuadratureType {
    GAUSS_LEGENDRE,   // 高斯-勒让德积分
    GAUSS_LOBATTO,    // 高斯-洛巴托积分
    TRIANGLE          // 三角形高斯积分
};

// 积分维度枚举
enum class QuadratureDimension {
    DIM_1D,  // 一维积分
    DIM_2D,  // 二维积分
    DIM_3D   // 三维积分(预留)
};

// 积分点结构体
struct QuadraturePoint {
    std::vector<double> coordinates;  // 坐标
    double weight;                   // 权重

    QuadraturePoint() : weight(0.0) {}
    QuadraturePoint(const std::vector<double>& coords, double w)
        : coordinates(coords), weight(w) {}
};

class GaussQuadrature {
public:
    // 构造函数
    GaussQuadrature() {}

    // 析构函数
    ~GaussQuadrature() {}

    // 获取一维线段的高斯积分点(支持Gauss-Legendre和Gauss-Lobatto)
    std::vector<QuadraturePoint> getQuadraturePoints1D(
        QuadratureType type, int order,
        double left = -1.0, double right = 1.0) const;

    // 获取二维三角形的高斯积分点
    std::vector<QuadraturePoint> getQuadraturePoints2D(
        const std::vector<double>& vertices, int pointCount) const;

    // 获取二维平面上线段的高斯积分点
    std::vector<QuadraturePoint> getQuadraturePointsLine2D(
        QuadratureType type, double x1, double y1, double x2, double y2, int order) const;

    // 获取三维区域的高斯积分点(预留接口)
    std::vector<QuadraturePoint> getQuadraturePoints3D(
        const std::vector<double>& vertices, int pointCount) const;

    // 获取标准区域的高斯积分点(内部使用)
    std::vector<QuadraturePoint> getReferenceQuadraturePoints(
        QuadratureType type, int order) const;

    // 将标准区域的积分点映射到实际区域(内部使用)
    std::vector<QuadraturePoint> mapToActualDomain(
        const std::vector<QuadraturePoint>& refPoints,
        QuadratureDimension dim,
        const std::vector<double>& domain) const;

private:
    // 生成Gauss-Legendre积分点(标准区间[-1,1])
    std::vector<QuadraturePoint> generateGaussLegendre(int order) const;

    // 生成Gauss-Lobatto积分点(标准区间[-1,1])
    std::vector<QuadraturePoint> generateGaussLobatto(int order) const;

    // 生成三角形高斯积分点(标准三角形)
    std::vector<QuadraturePoint> generateTriangleGauss(int pointCount) const;

    // 将一维标准积分点映射到实际线段
    std::vector<QuadraturePoint> map1DPoints(
        const std::vector<QuadraturePoint>& refPoints,
        double left, double right) const;

    // 将三角形标准积分点映射到实际三角形
    std::vector<QuadraturePoint> mapTrianglePoints(
        const std::vector<QuadraturePoint>& refPoints,
        const std::vector<double>& vertices) const;
};

// 获取一维线段的高斯积分点
inline std::vector<QuadraturePoint> GaussQuadrature::getQuadraturePoints1D(
    QuadratureType type, int order,
    double left, double right) const {//表示只读，不能修改属性

    if (type != QuadratureType::GAUSS_LEGENDRE && type != QuadratureType::GAUSS_LOBATTO) {
        throw std::invalid_argument("1D quadrature only supports GAUSS_LEGENDRE or GAUSS_LOBATTO type");
    }//异常报错，输入无效信息

    auto refPoints = getReferenceQuadraturePoints(type, order);
    return map1DPoints(refPoints, left, right);
}

// 获取二维三角形的高斯积分点
inline std::vector<QuadraturePoint> GaussQuadrature::getQuadraturePoints2D(
    const std::vector<double>& vertices, int pointCount) const {

    if (vertices.size() != 6) {
        throw std::invalid_argument("Triangle vertices must have 6 coordinates (x1,y1,x2,y2,x3,y3)");
    }

    auto refPoints = generateTriangleGauss(pointCount);
    return mapTrianglePoints(refPoints, vertices);
}

// 获取二维平面上线段的高斯积分点
inline std::vector<QuadraturePoint> GaussQuadrature::getQuadraturePointsLine2D(
    QuadratureType type, double x1, double y1, double x2, double y2, int order) const {

    if (type != QuadratureType::GAUSS_LEGENDRE && type != QuadratureType::GAUSS_LOBATTO) {
        throw std::invalid_argument("Line quadrature only supports GAUSS_LEGENDRE or GAUSS_LOBATTO type");
    }

    // 获取一维标准积分点
    auto refPoints = getReferenceQuadraturePoints(type, order);

    // 计算线段长度
    double dx = x2 - x1;
    double dy = y2 - y1;
    double length = std::sqrt(dx * dx + dy * dy);
    double scale = length / 2.0;

    std::vector<QuadraturePoint> mappedPoints;

    // 映射到二维平面上的线段
    for (const auto& refPoint : refPoints) {
        double xi = refPoint.coordinates[0];  // 标准区间[-1,1]中的点
        // 线性映射: x = x1 + (x2 - x1) * (xi + 1) / 2
        double x = x1 + dx * (xi + 1.0) / 2.0;
        double y = y1 + dy * (xi + 1.0) / 2.0;
        double mappedWeight = refPoint.weight * scale;
        mappedPoints.emplace_back(std::vector<double>{x, y}, mappedWeight);
    }

    return mappedPoints;
}

// 获取三维区域的高斯积分点(预留接口)
inline std::vector<QuadraturePoint> GaussQuadrature::getQuadraturePoints3D(
    const std::vector<double>& vertices, int pointCount) const {

    throw std::runtime_error("3D quadrature is not implemented yet");
}

// 获取标准区域的高斯积分点
inline std::vector<QuadraturePoint> GaussQuadrature::getReferenceQuadraturePoints(
    QuadratureType type, int order) const {

    switch (type) {
        case QuadratureType::GAUSS_LEGENDRE:
            return generateGaussLegendre(order);
        case QuadratureType::GAUSS_LOBATTO:
            return generateGaussLobatto(order);
        case QuadratureType::TRIANGLE:
            return generateTriangleGauss(order);
        default:
            throw std::invalid_argument("Unknown quadrature type");
    }
}

// 生成Gauss-Legendre积分点(标准区间[-1,1])
inline std::vector<QuadraturePoint> GaussQuadrature::generateGaussLegendre(int order) const {
    std::vector<QuadraturePoint> points;

    switch (order) {
        case 1: {
            points.emplace_back(std::vector<double>{0.0}, 2.0);
            break;
        }
        case 2: {
            double x = 0.5773502691896257645091488;
            points.emplace_back(std::vector<double>{-x}, 1.0);
            points.emplace_back(std::vector<double>{x}, 1.0);
            break;
        }
        case 3: {
            double x = 0.7745966692414833770358531;
            points.emplace_back(std::vector<double>{-x}, 0.5555555555555556);
            points.emplace_back(std::vector<double>{0.0}, 0.8888888888888889);
            points.emplace_back(std::vector<double>{x}, 0.5555555555555556);
            break;
        }
        case 4: {
            double x1 = 0.8611363115940526;
            double x2 = 0.3399810435848563;
            points.emplace_back(std::vector<double>{-x1}, 0.3478548451374539);
            points.emplace_back(std::vector<double>{-x2}, 0.6521451548625461);
            points.emplace_back(std::vector<double>{x2}, 0.6521451548625461);
            points.emplace_back(std::vector<double>{x1}, 0.3478548451374539);
            break;
        }
        case 5: {
            double x1 = 0.9061798459386640;
            double x2 = 0.5384693101056831;
            points.emplace_back(std::vector<double>{-x1}, 0.2369268850561891);
            points.emplace_back(std::vector<double>{-x2}, 0.4786286704993665);
            points.emplace_back(std::vector<double>{0.0}, 0.5688888888888889);
            points.emplace_back(std::vector<double>{x2}, 0.4786286704993665);
            points.emplace_back(std::vector<double>{x1}, 0.2369268850561891);
            break;
        }
        case 6: {
            double x1 = 0.9324695142031520;
            double x2 = 0.6612093864662645;
            double x3 = 0.2386191860831969;
            points.emplace_back(std::vector<double>{-x1}, 0.1713244923791703);
            points.emplace_back(std::vector<double>{-x2}, 0.3607615730481386);
            points.emplace_back(std::vector<double>{-x3}, 0.4679139345726910);
            points.emplace_back(std::vector<double>{x3}, 0.4679139345726910);
            points.emplace_back(std::vector<double>{x2}, 0.3607615730481386);
            points.emplace_back(std::vector<double>{x1}, 0.1713244923791703);
            break;
        }
        case 7: {
            double x1 = 0.9491079123427585;
            double x2 = 0.7415311855993944;
            double x3 = 0.4058451513773972;
            points.emplace_back(std::vector<double>{-x1}, 0.1294849661688697);
            points.emplace_back(std::vector<double>{-x2}, 0.2797053914892767);
            points.emplace_back(std::vector<double>{-x3}, 0.3818300505051190);
            points.emplace_back(std::vector<double>{0.0}, 0.4179591836734694);
            points.emplace_back(std::vector<double>{x3}, 0.3818300505051190);
            points.emplace_back(std::vector<double>{x2}, 0.2797053914892767);
            points.emplace_back(std::vector<double>{x1}, 0.1294849661688697);
            break;
        }
        case 8: {
            double x1 = 0.9602898564975362;
            double x2 = 0.7966664774136267;
            double x3 = 0.5255324099163290;
            double x4 = 0.1834346424956498;
            points.emplace_back(std::vector<double>{-x1}, 0.1012285362903763);
            points.emplace_back(std::vector<double>{-x2}, 0.2223810344533745);
            points.emplace_back(std::vector<double>{-x3}, 0.3137066458778873);
            points.emplace_back(std::vector<double>{-x4}, 0.3626837833783620);
            points.emplace_back(std::vector<double>{x4}, 0.3626837833783620);
            points.emplace_back(std::vector<double>{x3}, 0.3137066458778873);
            points.emplace_back(std::vector<double>{x2}, 0.2223810344533745);
            points.emplace_back(std::vector<double>{x1}, 0.1012285362903763);
            break;
        }
        case 9: {
            double x1 = 0.9681602395076261;
            double x2 = 0.8360311073266358;
            double x3 = 0.6133714327005904;
            double x4 = 0.3242534234038089;
            points.emplace_back(std::vector<double>{-x1}, 0.0812743883615744);
            points.emplace_back(std::vector<double>{-x2}, 0.1806481606948574);
            points.emplace_back(std::vector<double>{-x3}, 0.2606106964029355);
            points.emplace_back(std::vector<double>{-x4}, 0.3123470770400028);
            points.emplace_back(std::vector<double>{0.0}, 0.3302393550012598);
            points.emplace_back(std::vector<double>{x4}, 0.3123470770400028);
            points.emplace_back(std::vector<double>{x3}, 0.2606106964029355);
            points.emplace_back(std::vector<double>{x2}, 0.1806481606948574);
            points.emplace_back(std::vector<double>{x1}, 0.0812743883615744);
            break;
        }
        case 10: {
            double x1 = 0.9739065285171717;
            double x2 = 0.8650633666889845;
            double x3 = 0.6794095682990244;
            double x4 = 0.4333953941292472;
            double x5 = 0.1488743389816312;
            points.emplace_back(std::vector<double>{-x1}, 0.0666713443086881);
            points.emplace_back(std::vector<double>{-x2}, 0.1494513491505806);
            points.emplace_back(std::vector<double>{-x3}, 0.2190863625159820);
            points.emplace_back(std::vector<double>{-x4}, 0.2692667193099964);
            points.emplace_back(std::vector<double>{-x5}, 0.2955242247147529);
            points.emplace_back(std::vector<double>{x5}, 0.2955242247147529);
            points.emplace_back(std::vector<double>{x4}, 0.2692667193099964);
            points.emplace_back(std::vector<double>{x3}, 0.2190863625159820);
            points.emplace_back(std::vector<double>{x2}, 0.1494513491505806);
            points.emplace_back(std::vector<double>{x1}, 0.0666713443086881);
            break;
        }
        default:
            throw std::invalid_argument("Gauss-Legendre order must be between 1 and 10");
    }

    return points;
}

// 生成Gauss-Lobatto积分点(标准区间[-1,1])
inline std::vector<QuadraturePoint> GaussQuadrature::generateGaussLobatto(int order) const {
    std::vector<QuadraturePoint> points;

    if (order == 1) {  // 2个点
        points.emplace_back(std::vector<double>{-1.0}, 1.0);
        points.emplace_back(std::vector<double>{1.0}, 1.0);
    } else if (order == 2) {  // 3个点
        points.emplace_back(std::vector<double>{-1.0}, 1.0 / 3.0);
        points.emplace_back(std::vector<double>{0.0}, 4.0 / 3.0);
        points.emplace_back(std::vector<double>{1.0}, 1.0 / 3.0);
    } else if (order == 3) {  // 4个点
        double x = 0.447213595499958;
        points.emplace_back(std::vector<double>{-1.0}, 1.0 / 6.0);
        points.emplace_back(std::vector<double>{-x}, 5.0 / 6.0);
        points.emplace_back(std::vector<double>{x}, 5.0 / 6.0);
        points.emplace_back(std::vector<double>{1.0}, 1.0 / 6.0);
    } else if (order == 4) {  // 5个点
        double x = 0.654653670707977;
        points.emplace_back(std::vector<double>{-1.0}, 0.100000000000000);
        points.emplace_back(std::vector<double>{-x}, 0.544444444444444);
        points.emplace_back(std::vector<double>{0.0}, 0.711111111111111);
        points.emplace_back(std::vector<double>{x}, 0.544444444444444);
        points.emplace_back(std::vector<double>{1.0}, 0.100000000000000);
    } else if (order == 5) {  // 6个点
        double x1 = 0.765055323929465;
        double x2 = 0.285231516480645;
        points.emplace_back(std::vector<double>{-1.0}, 0.0666666666666667);
        points.emplace_back(std::vector<double>{-x1}, 0.378474956297847);
        points.emplace_back(std::vector<double>{-x2}, 0.554858377035487);
        points.emplace_back(std::vector<double>{x2}, 0.554858377035487);
        points.emplace_back(std::vector<double>{x1}, 0.378474956297847);
        points.emplace_back(std::vector<double>{1.0}, 0.0666666666666667);
    } else if (order == 6) {  // 7个点
        double x1 = 0.830223896278567;
        double x2 = 0.468848793470714;
        points.emplace_back(std::vector<double>{-1.0}, 0.0476190476190476);
        points.emplace_back(std::vector<double>{-x1}, 0.276826047361566);
        points.emplace_back(std::vector<double>{-x2}, 0.431745381209863);
        points.emplace_back(std::vector<double>{0.0}, 0.487619047619048);
        points.emplace_back(std::vector<double>{x2}, 0.431745381209863);
        points.emplace_back(std::vector<double>{x1}, 0.276826047361566);
        points.emplace_back(std::vector<double>{1.0}, 0.0476190476190476);
    } else if (order == 7) {  // 8个点
        double x1 = 0.871740148509607;
        double x2 = 0.591700181433142;
        double x3 = 0.209299217902479;
        points.emplace_back(std::vector<double>{-1.0}, 0.0357142857142857);
        points.emplace_back(std::vector<double>{-x1}, 0.210704227143506);
        points.emplace_back(std::vector<double>{-x2}, 0.341122692483504);
        points.emplace_back(std::vector<double>{-x3}, 0.412458794658704);
        points.emplace_back(std::vector<double>{x3}, 0.412458794658704);
        points.emplace_back(std::vector<double>{x2}, 0.341122692483504);
        points.emplace_back(std::vector<double>{x1}, 0.210704227143506);
        points.emplace_back(std::vector<double>{1.0}, 0.0357142857142857);
    } else {
        throw std::invalid_argument("Gauss-Lobatto order must be between 1 and 7");
    }

    return points;
}

// 生成三角形高斯积分点(标准三角形)
inline std::vector<QuadraturePoint> GaussQuadrature::generateTriangleGauss(int pointCount) const {
    std::vector<QuadraturePoint> points;

    switch (pointCount) {
        case 3: {
            points.emplace_back(std::vector<double>{0.5, 0.0}, 1.0 / 6.0);
            points.emplace_back(std::vector<double>{0.5, 0.5}, 1.0 / 6.0);
            points.emplace_back(std::vector<double>{0.0, 0.5}, 1.0 / 6.0);
            break;
        }
        case 4: {
            double s = 1.0 / std::sqrt(3);
            points.emplace_back(std::vector<double>{(1 + s)/2, (1 - s)*(1 + s)/4}, (1 - s)/8);
            points.emplace_back(std::vector<double>{(1 + s)/2, (1 - s)*(1 - s)/4}, (1 - s)/8);
            points.emplace_back(std::vector<double>{(1 - s)/2, (1 + s)*(1 + s)/4}, (1 + s)/8);
            points.emplace_back(std::vector<double>{(1 - s)/2, (1 + s)*(1 - s)/4}, (1 + s)/8);
            break;
        }
        case 9: {
            double s = std::sqrt(3.0 / 5.0);
            points.emplace_back(std::vector<double>{0.5, 0.25}, 64.0 / 81.0 * 1.0 / 8);
            points.emplace_back(std::vector<double>{(1 + s)/2, (1 - s)*(1 + s)/4}, 100.0 / 324.0 * (1 - s)/8);
            points.emplace_back(std::vector<double>{(1 + s)/2, (1 - s)*(1 - s)/4}, 100.0 / 324.0 * (1 - s)/8);
            points.emplace_back(std::vector<double>{(1 - s)/2, (1 + s)*(1 + s)/4}, 100.0 / 324.0 * (1 + s)/8);
            points.emplace_back(std::vector<double>{(1 - s)/2, (1 + s)*(1 - s)/4}, 100.0 / 324.0 * (1 + s)/8);
            points.emplace_back(std::vector<double>{0.5, (1 + s)/4}, 40.0 / 81.0 * 1.0 / 8);
            points.emplace_back(std::vector<double>{0.5, (1 - s)/4}, 40.0 / 81.0 * 1.0 / 8);
            points.emplace_back(std::vector<double>{(1 + s)/2, (1 - s)/4}, 40.0 / 81.0 * (1 - s)/8);
            points.emplace_back(std::vector<double>{(1 - s)/2, (1 + s)/4}, 40.0 / 81.0 * (1 + s)/8);
            break;
        }
        case 12: {
            // 12点高斯积分的坐标点 (面积坐标转换为标准三角形坐标)
            points.emplace_back(std::vector<double>{0.5611400349004341, 0.2194299825497830}, 0.1713331241529811 * 0.5);
            points.emplace_back(std::vector<double>{0.2194299825497830, 0.2194299825497830}, 0.1713331241529811 * 0.5);
            points.emplace_back(std::vector<double>{0.2194299825497830, 0.5611400349004341}, 0.1713331241529811 * 0.5);
            points.emplace_back(std::vector<double>{0.0397240717755698, 0.4801379641122151}, 0.0807310895930309 * 0.5);
            points.emplace_back(std::vector<double>{0.4801379641122151, 0.4801379641122151}, 0.0807310895930309 * 0.5);
            points.emplace_back(std::vector<double>{0.4801379641122151, 0.0397240717755698}, 0.0807310895930309 * 0.5);
            points.emplace_back(std::vector<double>{0.0193717243612408, 0.1416190159239682}, 0.0406345597936607 * 0.5);
            points.emplace_back(std::vector<double>{0.1416190159239682, 0.8390092597147911}, 0.0406345597936607 * 0.5);
            points.emplace_back(std::vector<double>{0.8390092597147911, 0.0193717243612408}, 0.0406345597936607 * 0.5);
            points.emplace_back(std::vector<double>{0.1416190159239682, 0.0193717243612408}, 0.0406345597936607 * 0.5);
            points.emplace_back(std::vector<double>{0.0193717243612408, 0.8390092597147911}, 0.0406345597936607 * 0.5);
            points.emplace_back(std::vector<double>{0.8390092597147911, 0.1416190159239682}, 0.0406345597936607 * 0.5);
            break;
        }
        default:
            throw std::invalid_argument("Triangle quadrature point count must be 3, 4, 9, or 12");
    }

    return points;
}

// 将一维标准积分点映射到实际线段
inline std::vector<QuadraturePoint> GaussQuadrature::map1DPoints(
    const std::vector<QuadraturePoint>& refPoints,
    double left, double right) const {

    std::vector<QuadraturePoint> mappedPoints;
    double length = right - left;
    double center = (left + right) / 2.0;
    double scale = length / 2.0;

    for (const auto& refPoint : refPoints) {
        double mappedX = center + refPoint.coordinates[0] * scale;
        double mappedWeight = refPoint.weight * scale;
        mappedPoints.emplace_back(std::vector<double>{mappedX}, mappedWeight);
    }

    return mappedPoints;
}

// 将三角形标准积分点映射到实际三角形
inline std::vector<QuadraturePoint> GaussQuadrature::mapTrianglePoints(
    const std::vector<QuadraturePoint>& refPoints,
    const std::vector<double>& vertices) const {

    std::vector<QuadraturePoint> mappedPoints;

    // 三角形三个顶点
    double x1 = vertices[0], y1 = vertices[1];
    double x2 = vertices[2], y2 = vertices[3];
    double x3 = vertices[4], y3 = vertices[5];

    // 计算雅可比行列式（三角形的雅可比行列式是该值）
    double J = std::fabs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

    for (const auto& refPoint : refPoints) {
        double xi = refPoint.coordinates[0];
        double eta = refPoint.coordinates[1];

        // 将标准三角形坐标转换到实际三角形坐标
        double x = x1 + (x2 - x1) * xi + (x3 - x1) * eta;
        double y = y1 + (y2 - y1) * xi + (y3 - y1) * eta;

        double mappedWeight = refPoint.weight * J;
        mappedPoints.emplace_back(std::vector<double>{x, y}, mappedWeight);
    }

    return mappedPoints;
}

}  // namespace VEM

#endif  // GAUSS_QUADRATURE_H
