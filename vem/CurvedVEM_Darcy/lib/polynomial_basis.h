#ifndef CURVEDVEM_POLYNOMIAL_BASIS_H
#define CURVEDVEM_POLYNOMIAL_BASIS_H

#include <vector>

namespace vem {
namespace basis {

// ============================================================================
// 一维点（标量坐标）
// ============================================================================
struct Point1D {
    double x;

    Point1D() : x(0.0) {}
    explicit Point1D(double x_) : x(x_) {}
};

// ============================================================================
// 二维点（标量坐标）
// ============================================================================
struct Point2D {
    double x, y;

    Point2D() : x(0.0), y(0.0) {}
    Point2D(double x_, double y_) : x(x_), y(y_) {}
};

// ============================================================================
// 二维向量
// ============================================================================
struct Vector2D {
    double x, y;

    Vector2D() : x(0.0), y(0.0) {}
    Vector2D(double x_, double y_) : x(x_), y(y_) {}
};

// ============================================================================
// 多项式基函数类
//
// 功能：在参考单元上构造和评估各类多项式空间的基函数，
//       包括标量单项式基、H(div) 向量多项式基。
//
// 坐标约定：
//   - 所有基函数都使用「单元局部缩放坐标」评估，即
//       x_hat = (x - xD) / hD,   y_hat = (y - yD) / hD
//     其中 (xD, yD) 是单元质心，hD 是单元特征尺度。
//   - 这样处理可以避免大单元上的单项式系数溢出，提高数值稳定性。
//
// 支持的多项式空间：
//   1. P_k    — 二维标量 k 次多项式空间，单项式基 m_α(x,y) = x^α y^β
//   2. P_k^1D — 一维标量 k 次多项式空间
//   3. G_k    — H(div) 梯度子空间：{∇ p | p ∈ P_{k+1}}，维数 dim(P_{k+1}) - 1
//   4. C_k    — H(div) 旋度补空间：散度为零的向量多项式，维数 dim(P_{k-1})
//   5. H(div) 向量多项式空间：G_k ⊕ C_k（直和）
// ============================================================================
class BasisFunctionPloy {
public:
    // 构造函数：指定最高多项式次数 k（支持 0 ~ 4）
    explicit BasisFunctionPloy(int k);
    ~BasisFunctionPloy() = default;

    // ----- 组合数与空间维数 -----

    /// 计算组合数 C(n, k) = n! / (k! * (n-k)!)
    static long comb(int n, int k);

    /// 计算 d 维 k 次多项式空间维数：dim P_k^d = C(k+d, d)
    static int dimPk(int k, int d);

    // ----- 标量单项式基（P_k） -----

    /// 返回 d 维（1 或 2）k 次单项式基的个数 = dim P_k^d
    int getDimMonomial(int d) const;

    /// 返回 d 维单项式基的多重指标列表
    ///   每个元素为 [x_degree, y_degree]（二维）或 [degree]（一维）
    const std::vector<std::vector<int>>& getMultiIndex(int d) const;

    /// 评估第 nk 个二维标量单项式在点 x 处的值
    ///   m_nk(x) = ((x-xD)/hD)^α * ((y-yD)/hD)^β
    double evalMonomial2D(int nk, const Point2D& xD, double hD,
                          const Point2D& x) const;

    /// 评估第 nk 个二维单项式对 d_num 方向的一阶偏导
    ///   d_num = 0 → ∂/∂x，d_num = 1 → ∂/∂y
    double evalMonomialDeriv2D(int nk, const Point2D& xD, double hD,
                               const Point2D& x, int d_num) const;

    /// 评估第 nk 个二维单项式对 d_num 方向的二阶偏导
    ///   d_num = 0 → ∂²/∂x²，d_num = 1 → ∂²/∂y²
    double evalMonomialSecondDeriv2D(int nk, const Point2D& xD, double hD,
                                     const Point2D& x, int d_num) const;

    /// 评估第 nk 个一维标量单项式在点 x 处的值
    ///   m_nk(x) = ((x-xD)/hD)^nk
    double evalMonomial1D(int nk, const Point1D& xD, double hD,
                          const Point1D& x) const;

    // ----- 边上的单项式 -----

    /// 将二维单项式 m_nk 限制在给定边上，展开为边局部坐标的一维多项式系数
    ///   输出 coefficients 的长度为 k+1，coefficients[i] 是 t^i 的系数
    ///   边参数 t ∈ [-0.5, 0.5]，以边中点为原点、以边方向为局部坐标
    void getEdgeBasisCoeffs(int nk, const Point2D& xD, double hD,
                            const Point2D& point1, const Point2D& point2,
                            std::vector<double>& coefficients) const;

    /// 评估边上的 nk 次单项式在点 x 处的值（局部坐标归一化到 [-0.5,0.5]）
    double evalEdgeMonomial(int nk, const Point2D& x0, const Point2D& x1,
                            const Point2D& x) const;

    // ----- H(div) 向量多项式基 -----

    /// 返回 H(div) 向量多项式空间的维数 = dim G_k + dim C_k
    int getDimHdiv() const;

    /// 返回梯度子空间 G_k = {∇ p | p ∈ P_{k+1}} 的维数 = dim(P_{k+1}) - 1
    int getDimGradSpace() const;

    /// 返回旋度补空间 C_k 的维数 = dim(P_{k-1})（k=0 时为 0）
    int getDimComplementSpace() const;

    /// 判断第 nk 个 H(div) 基函数是否属于梯度子空间 G_k
    ///   前 dimGradSpace 个基为梯度型，后面的为补空间型
    bool isInGradSpace(int nk) const;

    /// 评估第 nk 个 H(div) 向量多项式基函数在点 x 处的值
    ///   - 前 dimGradSpace 个：∇ p_{nk+1}（P_{k+1} 去掉常数后的梯度）
    ///   - 后 dimComplementSpace 个：补空间基，散度为零
    Vector2D evalHdivBasis(int nk, const Point2D& xD, double hD,
                           const Point2D& x) const;

private:
    int k_;  // 多项式最高次数

    // 标量单项式基
    int dim_monomial_1d_;                       // 一维 P_k 维数
    int dim_monomial_2d_;                       // 二维 P_k 维数
    std::vector<std::vector<int>> mi_1d_;       // 一维多重指标
    std::vector<std::vector<int>> mi_2d_;       // 二维多重指标

    // H(div) 向量多项式基
    int dim_grad_space_;       // 梯度子空间 G_k 维数
    int dim_complement_space_; // 旋度补空间 C_k 维数
    int dim_hdiv_;             // H(div) 空间维数 = dim_grad + dim_complement

    // ----- 初始化 -----
    void initMultiIndex1D();
    void initMultiIndex2D();

    /// 将标量单项式的序号 index 转换为二维多重指标 (x_degree, y_degree)
    static void multiIndex2D(int index, int& x_degree, int& y_degree);

    // ----- 参数校验 -----
    static void validateScale(double hD);
    static void validateDirection(int d_num);
    void validateMonomialIndex(int nk, int d) const;
    void validateHdivIndex(int nk) const;
};

}  // namespace basis
}  // namespace vem

#endif  // CURVEDVEM_POLYNOMIAL_BASIS_H
