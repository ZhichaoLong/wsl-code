/**
 * 设备端算例数据：与 examples 中的 CPU 方程数据保持同一数学定义。
 * 当前支持 Darcy 1/2 和三组分 MS 1/2/3/4；几何映射已在 CPU 打包进 Sample。
 * 这些函数在积分点由 CUDA 核调用，不能将 CPU std::function 直接传入设备。
 */
#pragma once
#include "device.cuh"
namespace vgpu {
// 统一双精度 π 常量，制造解、通量和时间导数共用。
constexpr double pi = 3.1415926535897932384626433832795;
// 原 MS 数据的修正组分阻力系数；0.1 的对角正则项由调用方单独加入。
__device__ inline double bar(int i, int j) {
    if (i == j)
        return 0;
    if (i + j == 1)
        return 0;
    return i + j == 2 ? 0.1 : 1.9;
}
// MS 浓度耦合矩阵：非对角 -bar_ij*c_i，对角 Σ_(l≠i)bar_il*c_l。
__device__ inline double aij(int i, int j, const double* c) {
    if (i != j)
        return -bar(i, j) * c[i];
    double a = 0;
    for (int l = 0; l < 3; ++l)
        if (l != i)
            a += bar(i, l) * c[l];
    return a;
}
// 制造解三组分浓度，c₂=1-c₀-c₁，保证总浓度恒为 1。
__device__ inline void exact_c(double x, double y, double t, double* c) {
    c[0] = 0.25 + 0.25 * sin(2 * pi * x) * sin(8 * pi * t);
    c[1] = 0.25 + 0.25 * sin(3 * pi * y) * sin(6 * pi * t);
    c[2] = 1 - c[0] - c[1];
}
// 间断初值在积分点直接取值；不先投影，从而保留原 CPU 初始历史项语义。
__device__ inline void initial(int example, const Sample& s, double* c) {
    if (example == 1) {
        c[0] = s.x > -1e-10 && s.x < 0.5 && s.y > -1e-10 && s.y <= 0.5 ? 0.8 : 0.1;
        c[1] = s.x > -1e-10 && s.x < 0.5 && s.y > 0.5 && s.y < 1 + 1e-10 ? 0.8 : 0.1;
        c[2] = 1 - c[0] - c[1];
    } else if (example == 3) {
        c[0] = s.x <= 0.25 ? 1 : 0;
        c[1] = s.x >= 0.75 ? 1 : 0;
        c[2] = 1 - c[0] - c[1];
    } else
        exact_c(s.X, s.Y, 0, c);
}
// 由 (A(c)+0.1I)J=-∇c 求物理通量；每个线程只使用固定 3×3 小数组。
__device__ inline void flux(double x, double y, double t, double* fx, double* fy) {
    double c[3], A[9], inv[9];
    exact_c(x, y, t, c);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            A[3 * i + j] = aij(i, j, c) + (i == j ? 0.1 : 0);
    double det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) +
                 A[2] * (A[3] * A[7] - A[4] * A[6]);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            int r = (j + 1) % 3, s = (j + 2) % 3, u = (i + 1) % 3, v = (i + 2) % 3;
            inv[i * 3 + j] = (A[r * 3 + u] * A[s * 3 + v] - A[r * 3 + v] * A[s * 3 + u]) / det;
        }
    double gx = 0.5 * pi * cos(2 * pi * x) * sin(8 * pi * t),
           gy = 0.75 * pi * cos(3 * pi * y) * sin(6 * pi * t);
    for (int i = 0; i < 3; ++i) {
        fx[i] = -(inv[3 * i] - inv[3 * i + 2]) * gx;
        fy[i] = -(inv[3 * i + 1] - inv[3 * i + 2]) * gy;
    }
}
// 统一返回浓度/压力、物理通量和源项 f；MS 1/3 没有解析时间解。
__device__ inline void data(int ns, int example, const Sample& s, double t, double* c, double* fx, double* fy,
                            double* f) {
    if (ns == 1) {
        c[0] = sin(pi * s.X) * cos(pi * s.Y);
        fx[0] = -pi * cos(pi * s.X) * cos(pi * s.Y);
        fy[0] = pi * sin(pi * s.X) * sin(pi * s.Y);
        f[0] = 2 * pi * pi * c[0];
        return;
    }
    if (example == 1 || example == 3) {
        initial(example, s, c);
        for (int i = 0; i < 3; ++i)
            fx[i] = fy[i] = f[i] = 0;
        return;
    }
    exact_c(s.X, s.Y, t, c);
    flux(s.X, s.Y, t, fx, fy);
    double xp[3], xm[3], yp[3], ym[3], dummy[3];
    // 制造解源项采用与原算例一致的中心差分通量散度，步长为 1e-6。
    double h = 1e-6;
    flux(s.X + h, s.Y, t, xp, dummy);
    flux(s.X - h, s.Y, t, xm, dummy);
    flux(s.X, s.Y + h, t, dummy, yp);
    flux(s.X, s.Y - h, t, dummy, ym);
    double dt[3] = {2 * pi * sin(2 * pi * s.X) * cos(8 * pi * t),
                    1.5 * pi * sin(3 * pi * s.Y) * cos(6 * pi * t), 0};
    dt[2] = -dt[0] - dt[1];
    for (int i = 0; i < 3; ++i)
        f[i] = dt[i] + (xp[i] - xm[i] + yp[i] - ym[i]) / (2 * h);
}
} // namespace vgpu
