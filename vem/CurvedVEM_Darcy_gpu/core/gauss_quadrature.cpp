/**
 * CPU 积分规则实现：生成固定节点和权重；GPU 基础矩阵积分使用上传后的数据。
 */
/*
 * gauss_quadrature.cpp
 * 高斯积分模块实现
 */

#include "gauss_quadrature.h"

namespace vem {

// ========== 一维积分 - 标准区间 ==========
std::vector<QuadPoint> GaussQuadrature::get_1D_points(QuadratureType type, int order) const {
    switch (type) {
        case QuadratureType::GAUSS_LEGENDRE:
            return generate_gauss_legendre(order);
        case QuadratureType::GAUSS_LOBATTO:
            return generate_gauss_lobatto(order);
        default:
            throw std::invalid_argument("Unknown quadrature type");
    }
}

// ========== 一维积分 - 任意区间 ==========
std::vector<QuadPoint> GaussQuadrature::get_1D_points(QuadratureType type, int order,
                                                       double a, double b) const {
    auto ref_points = get_1D_points(type, order);
    return map_1D(ref_points, a, b);
}

// ========== 二维线段积分 ==========
std::vector<QuadPoint> GaussQuadrature::get_line_2D_points(QuadratureType type, int order,
                                                             double x1, double y1,
                                                             double x2, double y2) const {
    auto ref_points = get_1D_points(type, order);

    double dx = x2 - x1;
    double dy = y2 - y1;
    double length = std::sqrt(dx * dx + dy * dy);
    double scale = length / 2.0;

    std::vector<QuadPoint> result;
    for (const auto& p : ref_points) {
        double xi = p.x[0];
        double x = x1 + dx * (xi + 1.0) / 2.0;
        double y = y1 + dy * (xi + 1.0) / 2.0;
        double w = p.w * scale;
        result.emplace_back(std::vector<double>{x, y}, w);
    }

    return result;
}

// ========== 三角形积分 - 标准三角形 ==========
std::vector<QuadPoint> GaussQuadrature::get_triangle_reference_points(int n_points) const {
    return generate_triangle(n_points);
}

// ========== 三角形积分 - 任意三角形 ==========
std::vector<QuadPoint> GaussQuadrature::get_triangle_points(const std::vector<double>& vertices,
                                                              int n_points) const {
    if (vertices.size() != 6) {
        throw std::invalid_argument("Triangle vertices needs 6 values: [x1,y1,x2,y2,x3,y3]");
    }
    auto ref_points = generate_triangle(n_points);
    return map_triangle(ref_points, vertices);
}

// ========== 生成 Gauss-Legendre 积分点 ==========
std::vector<QuadPoint> GaussQuadrature::generate_gauss_legendre(int order) const {
    std::vector<QuadPoint> points;

    switch (order) {
        case 1: {
            points.emplace_back(0.0, 2.0);
            break;
        }
        case 2: {
            double x = 0.5773502691896257645091488;
            points.emplace_back(-x, 1.0);
            points.emplace_back(x, 1.0);
            break;
        }
        case 3: {
            double x = 0.7745966692414833770358531;
            points.emplace_back(-x, 0.5555555555555556);
            points.emplace_back(0.0, 0.8888888888888889);
            points.emplace_back(x, 0.5555555555555556);
            break;
        }
        case 4: {
            double x1 = 0.8611363115940526;
            double x2 = 0.3399810435848563;
            points.emplace_back(-x1, 0.3478548451374539);
            points.emplace_back(-x2, 0.6521451548625461);
            points.emplace_back(x2, 0.6521451548625461);
            points.emplace_back(x1, 0.3478548451374539);
            break;
        }
        case 5: {
            double x1 = 0.9061798459386640;
            double x2 = 0.5384693101056831;
            points.emplace_back(-x1, 0.2369268850561891);
            points.emplace_back(-x2, 0.4786286704993665);
            points.emplace_back(0.0, 0.5688888888888889);
            points.emplace_back(x2, 0.4786286704993665);
            points.emplace_back(x1, 0.2369268850561891);
            break;
        }
        case 6: {
            double x1 = 0.9324695142031520;
            double x2 = 0.6612093864662645;
            double x3 = 0.2386191860831969;
            points.emplace_back(-x1, 0.1713244923791703);
            points.emplace_back(-x2, 0.3607615730481386);
            points.emplace_back(-x3, 0.4679139345726910);
            points.emplace_back(x3, 0.4679139345726910);
            points.emplace_back(x2, 0.3607615730481386);
            points.emplace_back(x1, 0.1713244923791703);
            break;
        }
        case 7: {
            double x1 = 0.9491079123427585;
            double x2 = 0.7415311855993944;
            double x3 = 0.4058451513773972;
            points.emplace_back(-x1, 0.1294849661688697);
            points.emplace_back(-x2, 0.2797053914892767);
            points.emplace_back(-x3, 0.3818300505051190);
            points.emplace_back(0.0, 0.4179591836734694);
            points.emplace_back(x3, 0.3818300505051190);
            points.emplace_back(x2, 0.2797053914892767);
            points.emplace_back(x1, 0.1294849661688697);
            break;
        }
        case 8: {
            double x1 = 0.9602898564975362;
            double x2 = 0.7966664774136267;
            double x3 = 0.5255324099163290;
            double x4 = 0.1834346424956498;
            points.emplace_back(-x1, 0.1012285362903763);
            points.emplace_back(-x2, 0.2223810344533745);
            points.emplace_back(-x3, 0.3137066458778873);
            points.emplace_back(-x4, 0.3626837833783620);
            points.emplace_back(x4, 0.3626837833783620);
            points.emplace_back(x3, 0.3137066458778873);
            points.emplace_back(x2, 0.2223810344533745);
            points.emplace_back(x1, 0.1012285362903763);
            break;
        }
        case 9: {
            double x1 = 0.9681602395076261;
            double x2 = 0.8360311073266358;
            double x3 = 0.6133714327005904;
            double x4 = 0.3242534234038089;
            points.emplace_back(-x1, 0.0812743883615744);
            points.emplace_back(-x2, 0.1806481606948574);
            points.emplace_back(-x3, 0.2606106964029355);
            points.emplace_back(-x4, 0.3123470770400028);
            points.emplace_back(0.0, 0.3302393550012598);
            points.emplace_back(x4, 0.3123470770400028);
            points.emplace_back(x3, 0.2606106964029355);
            points.emplace_back(x2, 0.1806481606948574);
            points.emplace_back(x1, 0.0812743883615744);
            break;
        }
        case 10: {
            double x1 = 0.9739065285171717;
            double x2 = 0.8650633666889845;
            double x3 = 0.6794095682990244;
            double x4 = 0.4333953941292472;
            double x5 = 0.1488743389816312;
            points.emplace_back(-x1, 0.0666713443086881);
            points.emplace_back(-x2, 0.1494513491505806);
            points.emplace_back(-x3, 0.2190863625159820);
            points.emplace_back(-x4, 0.2692667193099964);
            points.emplace_back(-x5, 0.2955242247147529);
            points.emplace_back(x5, 0.2955242247147529);
            points.emplace_back(x4, 0.2692667193099964);
            points.emplace_back(x3, 0.2190863625159820);
            points.emplace_back(x2, 0.1494513491505806);
            points.emplace_back(x1, 0.0666713443086881);
            break;
        }
        default:
            throw std::invalid_argument("Gauss-Legendre order must be between 1 and 10");
    }

    return points;
}

// ========== 生成 Gauss-Lobatto 积分点 ==========
std::vector<QuadPoint> GaussQuadrature::generate_gauss_lobatto(int order) const {
    std::vector<QuadPoint> points;

    if (order == 1) {  // 2个点
        points.emplace_back(-1.0, 1.0);
        points.emplace_back(1.0, 1.0);
    } else if (order == 2) {  // 3个点
        points.emplace_back(-1.0, 1.0 / 3.0);
        points.emplace_back(0.0, 4.0 / 3.0);
        points.emplace_back(1.0, 1.0 / 3.0);
    } else if (order == 3) {  // 4个点
        double x = 0.447213595499958;
        points.emplace_back(-1.0, 1.0 / 6.0);
        points.emplace_back(-x, 5.0 / 6.0);
        points.emplace_back(x, 5.0 / 6.0);
        points.emplace_back(1.0, 1.0 / 6.0);
    } else if (order == 4) {  // 5个点
        double x = 0.654653670707977;
        points.emplace_back(-1.0, 0.1);
        points.emplace_back(-x, 0.544444444444444);
        points.emplace_back(0.0, 0.711111111111111);
        points.emplace_back(x, 0.544444444444444);
        points.emplace_back(1.0, 0.1);
    } else if (order == 5) {  // 6个点
        double x1 = 0.765055323929465;
        double x2 = 0.285231516480645;
        points.emplace_back(-1.0, 0.0666666666666667);
        points.emplace_back(-x1, 0.378474956297847);
        points.emplace_back(-x2, 0.554858377035487);
        points.emplace_back(x2, 0.554858377035487);
        points.emplace_back(x1, 0.378474956297847);
        points.emplace_back(1.0, 0.0666666666666667);
    } else if (order == 6) {  // 7个点
        double x1 = 0.830223896278567;
        double x2 = 0.468848793470714;
        points.emplace_back(-1.0, 0.0476190476190476);
        points.emplace_back(-x1, 0.276826047361566);
        points.emplace_back(-x2, 0.431745381209863);
        points.emplace_back(0.0, 0.487619047619048);
        points.emplace_back(x2, 0.431745381209863);
        points.emplace_back(x1, 0.276826047361566);
        points.emplace_back(1.0, 0.0476190476190476);
    } else if (order == 7) {  // 8个点
        double x1 = 0.871740148509607;
        double x2 = 0.591700181433142;
        double x3 = 0.209299217902479;
        points.emplace_back(-1.0, 0.0357142857142857);
        points.emplace_back(-x1, 0.210704227143506);
        points.emplace_back(-x2, 0.341122692483504);
        points.emplace_back(-x3, 0.412458794658704);
        points.emplace_back(x3, 0.412458794658704);
        points.emplace_back(x2, 0.341122692483504);
        points.emplace_back(x1, 0.210704227143506);
        points.emplace_back(1.0, 0.0357142857142857);
    } else {
        throw std::invalid_argument("Gauss-Lobatto order must be between 1 and 7");
    }

    return points;
}

// ========== 生成三角形积分点 ==========
std::vector<QuadPoint> GaussQuadrature::generate_triangle(int n_points) const {
    std::vector<QuadPoint> points;

    switch (n_points) {
        case 1: {
            points.emplace_back(1.0/3.0, 1.0/3.0, 0.5);
            break;
        }
        case 3: {
            points.emplace_back(0.5, 0.0, 1.0 / 6.0);
            points.emplace_back(0.5, 0.5, 1.0 / 6.0);
            points.emplace_back(0.0, 0.5, 1.0 / 6.0);
            break;
        }
        case 4: {
            double s = 1.0 / std::sqrt(3.0);
            points.emplace_back((1 + s)/2, (1 - s)*(1 + s)/4, (1 - s)/8.0);
            points.emplace_back((1 + s)/2, (1 - s)*(1 - s)/4, (1 - s)/8.0);
            points.emplace_back((1 - s)/2, (1 + s)*(1 + s)/4, (1 + s)/8.0);
            points.emplace_back((1 - s)/2, (1 + s)*(1 - s)/4, (1 + s)/8.0);
            break;
        }
        case 7: {
            // 7点三角形积分
            points.emplace_back(1.0/3.0, 1.0/3.0, 0.1125);
            double a = 0.797426985353087;
            double b = 0.101286507323456;
            points.emplace_back(a, b, 0.0629695902724135);
            points.emplace_back(b, a, 0.0629695902724135);
            points.emplace_back(b, b, 0.0629695902724135);
            double c = 0.059715871789770;
            double d = 0.470142064105115;
            points.emplace_back(c, d, 0.0661970763907399);
            points.emplace_back(d, c, 0.0661970763907399);
            points.emplace_back(d, d, 0.0661970763907399);
            break;
        }
        case 9: {
            double s = std::sqrt(3.0 / 5.0);
            points.emplace_back(0.5, 0.25, 64.0 / 81.0 * 1.0 / 8.0);
            points.emplace_back((1 + s)/2, (1 - s)*(1 + s)/4, 100.0 / 324.0 * (1 - s)/8.0);
            points.emplace_back((1 + s)/2, (1 - s)*(1 - s)/4, 100.0 / 324.0 * (1 - s)/8.0);
            points.emplace_back((1 - s)/2, (1 + s)*(1 + s)/4, 100.0 / 324.0 * (1 + s)/8.0);
            points.emplace_back((1 - s)/2, (1 + s)*(1 - s)/4, 100.0 / 324.0 * (1 + s)/8.0);
            points.emplace_back(0.5, (1 + s)/4, 40.0 / 81.0 * 1.0 / 8.0);
            points.emplace_back(0.5, (1 - s)/4, 40.0 / 81.0 * 1.0 / 8.0);
            points.emplace_back((1 + s)/2, (1 - s)/4, 40.0 / 81.0 * (1 - s)/8.0);
            points.emplace_back((1 - s)/2, (1 + s)/4, 40.0 / 81.0 * (1 + s)/8.0);
            break;
        }
        case 12: {
            points.emplace_back(0.5611400349004341, 0.2194299825497830, 0.1713331241529811 * 0.5);
            points.emplace_back(0.2194299825497830, 0.2194299825497830, 0.1713331241529811 * 0.5);
            points.emplace_back(0.2194299825497830, 0.5611400349004341, 0.1713331241529811 * 0.5);
            points.emplace_back(0.0397240717755698, 0.4801379641122151, 0.0807310895930309 * 0.5);
            points.emplace_back(0.4801379641122151, 0.4801379641122151, 0.0807310895930309 * 0.5);
            points.emplace_back(0.4801379641122151, 0.0397240717755698, 0.0807310895930309 * 0.5);
            points.emplace_back(0.0193717243612408, 0.1416190159239682, 0.0406345597936607 * 0.5);
            points.emplace_back(0.1416190159239682, 0.8390092597147911, 0.0406345597936607 * 0.5);
            points.emplace_back(0.8390092597147911, 0.0193717243612408, 0.0406345597936607 * 0.5);
            points.emplace_back(0.1416190159239682, 0.0193717243612408, 0.0406345597936607 * 0.5);
            points.emplace_back(0.0193717243612408, 0.8390092597147911, 0.0406345597936607 * 0.5);
            points.emplace_back(0.8390092597147911, 0.1416190159239682, 0.0406345597936607 * 0.5);
            break;
        }
        default:
            throw std::invalid_argument("Triangle quadrature supports: 1, 3, 4, 7, 9, 12 points");
    }

    return points;
}

// ========== 映射一维积分点 ==========
std::vector<QuadPoint> GaussQuadrature::map_1D(const std::vector<QuadPoint>& ref_points,
                                                double a, double b) const {
    std::vector<QuadPoint> result;
    double length = b - a;
    double center = (a + b) / 2.0;
    double scale = length / 2.0;

    for (const auto& p : ref_points) {
        double x = center + p.x[0] * scale;
        double w = p.w * scale;
        result.emplace_back(std::vector<double>{x}, w);
    }

    return result;
}

// ========== 映射三角形积分点 ==========
std::vector<QuadPoint> GaussQuadrature::map_triangle(const std::vector<QuadPoint>& ref_points,
                                                      const std::vector<double>& vertices) const {
    std::vector<QuadPoint> result;

    double x1 = vertices[0], y1 = vertices[1];
    double x2 = vertices[2], y2 = vertices[3];
    double x3 = vertices[4], y3 = vertices[5];

    // 雅可比行列式（三角形面积的2倍）
    double J = std::fabs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

    for (const auto& p : ref_points) {
        double xi = p.x[0];
        double eta = p.x[1];
        double x = x1 + (x2 - x1) * xi + (x3 - x1) * eta;
        double y = y1 + (y2 - y1) * xi + (y3 - y1) * eta;
        double w = p.w * J;
        result.emplace_back(std::vector<double>{x, y}, w);
    }

    return result;
}

} // namespace vem
