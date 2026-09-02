#include "polynomial_basis.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vem {
namespace basis {

namespace {

double scaledCoordinate(double value, double center, double scale) {
    return (value - center) / scale;
}

}  // namespace

BasisFunctionPloy::BasisFunctionPloy(int k)
    : k_(k),
      dim_monomial_1d_(0),
      dim_monomial_2d_(0),
      dim_grad_space_(0),
      dim_complement_space_(0),
      dim_hdiv_(0) {
    if (k < 0 || k > 4) {
        throw std::invalid_argument("polynomial degree k must be between 0 and 4");
    }

    dim_monomial_1d_ = dimPk(k_, 1);
    dim_monomial_2d_ = dimPk(k_, 2);
    initMultiIndex1D();
    initMultiIndex2D();

    dim_grad_space_ = dimPk(k_ + 1, 2) - 1;
    dim_complement_space_ = k_ == 0 ? 0 : dimPk(k_ - 1, 2);
    dim_hdiv_ = dim_grad_space_ + dim_complement_space_;
}

long BasisFunctionPloy::comb(int n, int k) {
    if (n < 0 || k < 0 || k > n) {
        throw std::invalid_argument("comb requires n >= k >= 0");
    }

    k = std::min(k, n - k);
    long result = 1;
    for (int i = 1; i <= k; ++i) {
        if (result > std::numeric_limits<long>::max() / (n - k + i)) {
            throw std::overflow_error("comb result exceeds long range");
        }
        result = result * (n - k + i) / i;
    }
    return result;
}

int BasisFunctionPloy::dimPk(int k, int d) {
    if (k < 0) {
        throw std::invalid_argument("polynomial degree must be non-negative");
    }
    if (d != 1 && d != 2) {
        throw std::invalid_argument("polynomial dimension must be 1 or 2");
    }

    long dimension = comb(k + d, d);
    if (dimension > std::numeric_limits<int>::max()) {
        throw std::overflow_error("polynomial-space dimension exceeds int range");
    }
    return static_cast<int>(dimension);
}

int BasisFunctionPloy::getDimMonomial(int d) const {
    if (d == 1) {
        return dim_monomial_1d_;
    }
    if (d == 2) {
        return dim_monomial_2d_;
    }
    throw std::invalid_argument("monomial dimension must be 1 or 2");
}

const std::vector<std::vector<int>>&
BasisFunctionPloy::getMultiIndex(int d) const {
    if (d == 1) {
        return mi_1d_;
    }
    if (d == 2) {
        return mi_2d_;
    }
    throw std::invalid_argument("multi-index dimension must be 1 or 2");
}

double BasisFunctionPloy::evalMonomial2D(
    int nk, const Point2D& xD, double hD, const Point2D& x) const {
    validateMonomialIndex(nk, 2);
    validateScale(hD);

    const std::vector<int>& alpha = mi_2d_[nk];
    double xm = scaledCoordinate(x.x, xD.x, hD);
    double ym = scaledCoordinate(x.y, xD.y, hD);
    return std::pow(xm, alpha[0]) * std::pow(ym, alpha[1]);
}

double BasisFunctionPloy::evalMonomialDeriv2D(
    int nk, const Point2D& xD, double hD, const Point2D& x,
    int d_num) const {
    validateMonomialIndex(nk, 2);
    validateScale(hD);
    validateDirection(d_num);

    const std::vector<int>& alpha = mi_2d_[nk];
    double xm = scaledCoordinate(x.x, xD.x, hD);
    double ym = scaledCoordinate(x.y, xD.y, hD);

    if (d_num == 0) {
        if (alpha[0] == 0) {
            return 0.0;
        }
        return alpha[0] * std::pow(xm, alpha[0] - 1) *
               std::pow(ym, alpha[1]) / hD;
    }

    if (alpha[1] == 0) {
        return 0.0;
    }
    return alpha[1] * std::pow(xm, alpha[0]) *
           std::pow(ym, alpha[1] - 1) / hD;
}

double BasisFunctionPloy::evalMonomialSecondDeriv2D(
    int nk, const Point2D& xD, double hD, const Point2D& x,
    int d_num) const {
    validateMonomialIndex(nk, 2);
    validateScale(hD);
    validateDirection(d_num);

    const std::vector<int>& alpha = mi_2d_[nk];
    double xm = scaledCoordinate(x.x, xD.x, hD);
    double ym = scaledCoordinate(x.y, xD.y, hD);

    if (d_num == 0) {
        if (alpha[0] < 2) {
            return 0.0;
        }
        return alpha[0] * (alpha[0] - 1) *
               std::pow(xm, alpha[0] - 2) * std::pow(ym, alpha[1]) /
               (hD * hD);
    }

    if (alpha[1] < 2) {
        return 0.0;
    }
    return alpha[1] * (alpha[1] - 1) * std::pow(xm, alpha[0]) *
           std::pow(ym, alpha[1] - 2) / (hD * hD);
}

double BasisFunctionPloy::evalMonomial1D(
    int nk, const Point1D& xD, double hD, const Point1D& x) const {
    validateMonomialIndex(nk, 1);
    validateScale(hD);
    return std::pow(scaledCoordinate(x.x, xD.x, hD), mi_1d_[nk][0]);
}

void BasisFunctionPloy::getEdgeBasisCoeffs(
    int nk, const Point2D& xD, double hD, const Point2D& p1,
    const Point2D& p2, std::vector<double>& coefficients) const {
    validateMonomialIndex(nk, 2);
    validateScale(hD);

    double edge_x = p2.x - p1.x;
    double edge_y = p2.y - p1.y;
    double edge_length = std::sqrt(edge_x * edge_x + edge_y * edge_y);
    if (!(edge_length > 0.0) || !std::isfinite(edge_length)) {
        throw std::invalid_argument("edge endpoints must define a nonzero finite edge");
    }

    const std::vector<int>& alpha = mi_2d_[nk];
    int total_degree = alpha[0] + alpha[1];
    coefficients.assign(k_ + 1, 0.0);

    double midpoint_x = 0.5 * (p1.x + p2.x);
    double midpoint_y = 0.5 * (p1.y + p2.y);
    double tangent_x = edge_x / edge_length;
    double tangent_y = edge_y / edge_length;
    double offset_x = midpoint_x - xD.x;
    double offset_y = midpoint_y - xD.y;

    for (int i = 0; i <= alpha[0]; ++i) {
        for (int j = 0; j <= alpha[1]; ++j) {
            int edge_degree = i + j;
            double coefficient =
                comb(alpha[0], i) * comb(alpha[1], j) *
                std::pow(offset_x, alpha[0] - i) *
                std::pow(offset_y, alpha[1] - j) *
                std::pow(tangent_x, i) * std::pow(tangent_y, j);
            coefficients[edge_degree] += coefficient;
        }
    }

    for (int degree = 0; degree <= total_degree; ++degree) {
        coefficients[degree] *=
            std::pow(edge_length, degree) / std::pow(hD, total_degree);
    }
}

double BasisFunctionPloy::evalEdgeMonomial(
    int nk, const Point2D& x0, const Point2D& x1, const Point2D& x) const {
    if (nk < 0) {
        throw std::out_of_range("edge monomial degree must be non-negative");
    }

    double edge_x = x1.x - x0.x;
    double edge_y = x1.y - x0.y;
    double edge_length = std::sqrt(edge_x * edge_x + edge_y * edge_y);
    if (!(edge_length > 0.0) || !std::isfinite(edge_length)) {
        throw std::invalid_argument("edge endpoints must define a nonzero finite edge");
    }

    double midpoint_x = 0.5 * (x0.x + x1.x);
    double midpoint_y = 0.5 * (x0.y + x1.y);
    double tangent_x = edge_x / edge_length;
    double tangent_y = edge_y / edge_length;
    double coordinate = ((x.x - midpoint_x) * tangent_x +
                         (x.y - midpoint_y) * tangent_y) /
                        edge_length;
    return std::pow(coordinate, nk);
}

int BasisFunctionPloy::getDimHdiv() const {
    return dim_hdiv_;
}

int BasisFunctionPloy::getDimGradSpace() const {
    return dim_grad_space_;
}

int BasisFunctionPloy::getDimComplementSpace() const {
    return dim_complement_space_;
}

bool BasisFunctionPloy::isInGradSpace(int nk) const {
    validateHdivIndex(nk);
    return nk < dim_grad_space_;
}

Vector2D BasisFunctionPloy::evalHdivBasis(
    int nk, const Point2D& xD, double hD, const Point2D& x) const {
    validateHdivIndex(nk);
    validateScale(hD);

    double xm = scaledCoordinate(x.x, xD.x, hD);
    double ym = scaledCoordinate(x.y, xD.y, hD);

    if (nk < dim_grad_space_) {
        int x_degree = 0;
        int y_degree = 0;
        // P_{k+1} 中第 0 个指标是常数；梯度基从第 1 个开始。
        multiIndex2D(nk + 1, x_degree, y_degree);

        double value_x = 0.0;
        double value_y = 0.0;
        if (x_degree > 0) {
            value_x = x_degree * std::pow(xm, x_degree - 1) *
                      std::pow(ym, y_degree) / hD;
        }
        if (y_degree > 0) {
            value_y = y_degree * std::pow(xm, x_degree) *
                      std::pow(ym, y_degree - 1) / hD;
        }
        return Vector2D(value_x, value_y);
    }

    int complement_index = nk - dim_grad_space_;
    int degree = 1;
    while (complement_index >= degree) {
        complement_index -= degree;
        ++degree;
    }

    int y_degree = complement_index;
    int x_degree = degree - y_degree;
    double factor = static_cast<double>(y_degree + 1) / x_degree;

    // 与旧工程的补空间完全一致：
    // (-(b+1)/a * xhat^(a-1) yhat^(b+1), xhat^a yhat^b)。
    return Vector2D(
        -factor * std::pow(xm, x_degree - 1) *
            std::pow(ym, y_degree + 1),
        std::pow(xm, x_degree) * std::pow(ym, y_degree)
    );
}

void BasisFunctionPloy::initMultiIndex1D() {
    mi_1d_.resize(dim_monomial_1d_);
    for (int index = 0; index < dim_monomial_1d_; ++index) {
        mi_1d_[index] = std::vector<int>(1, index);
    }
}

void BasisFunctionPloy::initMultiIndex2D() {
    mi_2d_.resize(dim_monomial_2d_);
    for (int index = 0; index < dim_monomial_2d_; ++index) {
        int x_degree = 0;
        int y_degree = 0;
        multiIndex2D(index, x_degree, y_degree);
        mi_2d_[index] = {x_degree, y_degree};
    }
}

void BasisFunctionPloy::multiIndex2D(
    int index, int& x_degree, int& y_degree) {
    if (index < 0) {
        throw std::out_of_range("multi-index must be non-negative");
    }

    int degree = 0;
    int preceding = 0;
    while (index >= preceding + degree + 1) {
        preceding += degree + 1;
        ++degree;
    }

    int offset = index - preceding;
    x_degree = degree - offset;
    y_degree = offset;
}

void BasisFunctionPloy::validateScale(double hD) {
    if (!(hD > 0.0) || !std::isfinite(hD)) {
        throw std::invalid_argument("element scale hD must be positive and finite");
    }
}

void BasisFunctionPloy::validateDirection(int d_num) {
    if (d_num != 0 && d_num != 1) {
        throw std::out_of_range("derivative direction must be 0 or 1");
    }
}

void BasisFunctionPloy::validateMonomialIndex(int nk, int d) const {
    int dimension = getDimMonomial(d);
    if (nk < 0 || nk >= dimension) {
        throw std::out_of_range("monomial basis index out of range");
    }
}

void BasisFunctionPloy::validateHdivIndex(int nk) const {
    if (nk < 0 || nk >= dim_hdiv_) {
        throw std::out_of_range("H(div) basis index out of range");
    }
}

}  // namespace basis
}  // namespace vem
