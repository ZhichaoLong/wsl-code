#ifndef BASIS_FUNCTION_PLOY_H
#define BASIS_FUNCTION_PLOY_H

#include <vector>
#include <cmath>
#include <stdexcept>

namespace vemhdiv {

/**
 * 二维点结构体 - 简洁C++风格
 */
struct Point2D {
    double x;
    double y;

    Point2D() : x(0.0), y(0.0) {}
    Point2D(double x_, double y_) : x(x_), y(y_) {}
};

/**
 * 一维点结构体
 */
struct Point1D {
    double x;

    Point1D() : x(0.0) {}
    explicit Point1D(double x_) : x(x_) {}
};

/**
 * 二维向量结构体 - 用于Hdiv基函数返回值
 */
struct Vector2D {
    double x;
    double y;

    Vector2D() : x(0.0), y(0.0) {}
    Vector2D(double x_, double y_) : x(x_), y(y_) {}
};

/**
 * 基函数类 - 提供多项式基函数和Hdiv空间基函数
 *
 * 该类包含两种基函数空间：
 * 1. 缩放单项式基函数空间 - k次多项式空间，以缩放单项式为基
 * 2. Hdiv空间基函数 - 由导数空间 + 补空间组成的k次多项式向量空间
 */
class BasisFunctionPloy {
public:
    /**
     * 构造函数
     * @param k 多项式最高次数（支持0,1,2,3）
     */
    explicit BasisFunctionPloy(int k);

    /// 析构函数
    ~BasisFunctionPloy() = default;

    // ==================== 工具函数 ====================

    /**
     * 计算组合数 C(n, k)
     * @param n 总数
     * @param k 选取数
     * @return 组合数结果
     */
    static long comb(int n, int k);

    /**
     * 计算d维空间中k次多项式空间的维数
     * @param k 多项式次数
     * @param d 空间维度
     * @return 多项式空间维数
     */
    static int dimPk(int k, int d);

    // ==================== 缩放单项式基函数相关 ====================

    /**
     * 获取k次多项式空间的维数
     * @param d 空间维度（1或2）
     * @return 多项式空间维数
     */
    int getDimMonomial(int d) const;

    /**
     * 获取多重指标数组
     * @param d 空间维度（1或2）
     * @return 多重指标数组，每个元素为(alpha1, alpha2,...)
     */
    const std::vector<std::vector<int>>& getMultiIndex(int d) const;

    /**
     * 计算二维缩放单项式基函数值
     * @param nk 基函数索引（对应多重指标数组中的索引）
     * @param xD 单元质心坐标
     * @param hD 单元直径
     * @param x 计算点坐标
     * @return 基函数值
     */
    double evalMonomial2D(int nk, const Point2D& xD, double hD, const Point2D& x) const;

    /**
     * 计算二维缩放单项式基函数的一阶偏导数
     * @param nk 基函数索引
     * @param xD 单元质心坐标
     * @param hD 单元直径
     * @param x 计算点坐标
     * @param d_num 求导方向（0表示x方向，1表示y方向）
     * @return 导数值
     */
    double evalMonomialDeriv2D(int nk, const Point2D& xD, double hD, const Point2D& x, int d_num) const;

    /**
     * 计算二维缩放单项式基函数的二阶偏导数
     * @param nk 基函数索引
     * @param xD 单元质心坐标
     * @param hD 单元直径
     * @param x 计算点坐标
     * @param d_num 求导方向（0表示x方向，1表示y方向）
     * @return 二阶导数值
     */
    double evalMonomialSecondDeriv2D(int nk, const Point2D& xD, double hD, const Point2D& x, int d_num) const;

    /**
     * 计算一维缩放单项式基函数值
     * @param nk 基函数索引
     * @param xD 单元中心坐标
     * @param hD 单元长度
     * @param x 计算点坐标
     * @return 基函数值
     */
    double evalMonomial1D(int nk, const Point1D& xD, double hD, const Point1D& x) const;

    /**
     * 计算二维边上的一维单项式映射到二维的系数
     *
     * 将二维单项式 ((x-xD)/hD)^alpha * ((y-yD)/hD)^beta 在边上表示为
     * 一维单项式 sum_{k=0}^{alpha+beta} coefficients[k] * ((s-se)/he)^k
     *
     * @param nk 二维基函数索引
     * @param xD 单元质心坐标
     * @param hD 单元直径
     * @param point1 边的第一个端点
     * @param point2 边的第二个端点
     * @param coefficients 输出系数数组
     */
    void getEdgeBasisCoeffs(int nk, const Point2D& xD, double hD,
                           const Point2D& point1, const Point2D& point2,
                           std::vector<double>& coefficients) const;

    /**
     * 计算一维边界单项式（映射到二维空间）
     * @param nk 一维单项式次数
     * @param x0 边起点坐标
     * @param x1 边终点坐标
     * @param x 二维计算点坐标
     * @return 单项式值 ((x-xe)·te/he)^nk
     */
    double evalEdgeMonomial(int nk, const Point2D& x0, const Point2D& x1, const Point2D& x) const;

    // ==================== Hdiv空间基函数相关 ====================

    /**
     * 获取Hdiv空间的维数（k次多项式向量空间）
     * @return Hdiv空间维数
     */
    int getDimHdiv() const;

    /**
     * 获取导数空间维数（原C代码vemdiv_dim_grad_k）
     * @return 导数空间维数
     */
    int getDimGradSpace() const;

    /**
     * 获取补空间维数（原C代码vemdiv_dim_grad_k_c）
     * @return 补空间维数
     */
    int getDimComplementSpace() const;

    /**
     * 判断一个基函数索引是否属于导数空间
     * @param nk 基函数索引
     * @return true表示属于导数空间，false表示属于补空间
     */
    bool isInGradSpace(int nk) const;

    /**
     * 计算Hdiv基函数值（向量值）
     *
     * Hdiv基函数分为两部分：
     * 1. 导数空间基函数 - 来自 (k+1) 次缩放单项式的梯度
     *    注意：添加了正确的系数 1/hD
     * 2. 补空间基函数 - 旋转变换后的多项式
     *
     * @param nk 基函数索引（0 <= nk < dimHdiv）
     * @param xD 单元质心坐标
     * @param hD 单元直径
     * @param x 计算点坐标
     * @return 向量值结果
     */
    Vector2D evalHdivBasis(int nk, const Point2D& xD, double hD, const Point2D& x) const;

private:
    int k_; ///< 多项式最高次数

    // ==================== 缩放单项式基函数数据 ====================
    int dim_monomial_1d_;           ///< 一维k次多项式空间维数
    int dim_monomial_2d_;           ///< 二维k次多项式空间维数
    std::vector<std::vector<int>> mi_1d_;  ///< 一维多重指标数组
    std::vector<std::vector<int>> mi_2d_;  ///< 二维多重指标数组

    // ==================== Hdiv空间基函数数据 ====================
    int dim_grad_space_;            ///< 导数空间维数
    int dim_complement_space_;      ///< 补空间维数
    int dim_hdiv_;                  ///< Hdiv空间总维数

    /// 初始化一维多重指标数组
    void initMultiIndex1D();

    /// 初始化二维多重指标数组
    void initMultiIndex2D();

    /**
     * 通过全局索引获取二维多重指标
     * @param j 全局索引（从1开始）
     * @param m 输出x方向次数
     * @param n 输出y方向次数
     */
    static void multiIndex2D(int j, int& m, int& n);
};

} // namespace vemhdiv

#endif
