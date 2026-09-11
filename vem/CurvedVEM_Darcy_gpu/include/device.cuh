/**
 * CUDA 公共设施：错误检查、显存所有权、扁平设备视图及局部多项式工具。
 * CPU 代码也可包含本文件；设备数学函数仅在 __CUDACC__ 下编译。
 * GPU 小矩阵求逆由整个 block 协作使用共享内存，避免每线程保存稠密矩阵。
 */
#pragma once
#include <cuda_runtime.h>
#include "petsc_gpu.h"
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <ostream>
namespace vgpu {
// 将 CUDA 返回码转为带调用位置的异常；主机驱动统一记录失败。
inline void check(cudaError_t e, const char* where) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
}
#define CUDA(call) ::vgpu::check((call), #call)
// RAII 显存缓冲区：只允许移动，禁止复制；析构自动释放设备内存。
template <class T> class Buffer {
    T* p_ = nullptr;
    size_t n_ = 0;

public:
    Buffer() = default;
    // 分配 n 个 T 元素，内容未初始化；需要全零时显式调用 zero。
    explicit Buffer(size_t n) {
        resize(n);
    }
    // 不在析构函数中抛异常，保证异常退出时仍可释放其他资源。
    ~Buffer() {
        if (p_)
            cudaFree(p_);
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    // 移动时转移设备指针，并将原对象置空。
    Buffer(Buffer&& b) noexcept : p_(b.p_), n_(b.n_) {
        b.p_ = nullptr;
        b.n_ = 0;
    }
    // 移动赋值先释放当前对象原有显存，再接管对方指针。
    Buffer& operator=(Buffer&& b) noexcept {
        if (this != &b) {
            if (p_)
                cudaFree(p_);
            p_ = b.p_;
            n_ = b.n_;
            b.p_ = nullptr;
            b.n_ = 0;
        }
        return *this;
    }
    // 容量改变时重新分配，不保留旧内容；n=0 可用于释放诊断缓存。
    void resize(size_t n) {
        if (n == n_)
            return;
        T* p = nullptr;
        if (n)
            CUDA(cudaMalloc(&p, n * sizeof(T)));
        if (p_) {
            const auto error = cudaFree(p_);
            if (error != cudaSuccess) {
                // 旧缓冲区释放失败时，新分配也必须回收。
                if (p) cudaFree(p);
                CUDA(error);
            }
        }
        p_ = p;
        n_ = n;
    }
    // CPU vector→GPU，必要时重分配大小。
    void upload(const std::vector<T>& a) {
        resize(a.size());
        if (n_)
            CUDA(cudaMemcpy(p_, a.data(), n_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    // GPU→CPU 并返回主机 vector；仅在验证或最终保存等场景使用。
    std::vector<T> download() const {
        std::vector<T> a(n_);
        if (n_)
            CUDA(cudaMemcpy(a.data(), p_, n_ * sizeof(T), cudaMemcpyDeviceToHost));
        return a;
    }
    // 将已分配缓冲区逐字节置零。
    void zero() {
        if (n_)
            CUDA(cudaMemset(p_, 0, n_ * sizeof(T)));
    }
    // 返回裸设备指针；调用者不得自行释放。
    T* data() {
        return p_;
    }
    const T* data() const {
        return p_;
    }
    // 元素个数，不是字节数。
    size_t size() const {
        return n_;
    }
};
// 单元记录（所有偏移以元素个数计，非字节）：
// ne/n: 边数/局部通量自由度；vo/nv: 体积分偏移/点数；eo/lo: 边点/局部编号偏移。
// dd/gg/ww/hh/hs: D(B,P)/G/W/H/H* 偏移；cx,cy,h,area: 计算域尺度。
struct Cell {
    int ne, n, vo, nv, eo, lo, dd, gg, ww, hh, hs;
    double cx, cy, h, area;
};
// 积分点：x/y/w 为计算域坐标/权重，X/Y 为物理坐标；j00..j11 和 det 为映射 J。
struct Sample {
    double x, y, w, X, Y, j00, j01, j10, j11, det;
};
// 边积分点另存归一化边参数 t∈[-1/2,1/2] 与计算域单元外法向 nx/ny。
struct EdgeSample {
    Sample s;
    double t, nx, ny;
};
// k: 阶次；q: dim(Pk)；p: 2q；grad: dim(P(k+1))-1；internal: 内部通量数。
// nc/nf/np/ng: 单元数/单组分全局通量数/L² 数/边积分点数。
struct Sizes {
    int k, q, p, grad, internal, nc, nf, np, ng;
};
// 无所有权的设备基础视图；几何与 map/sign 只读，矩阵缓冲区由基础构造核写入。
struct BaseView {
    Sizes s;
    const Cell* cells;
    const Sample* vol;
    const EdgeSample* edge;
    const int* map;
    const double* sign;
    double *D, *G, *W, *H, *Hs, *B, *P, *Hi;
};
// 二维完整多项式空间 Pk 的维数 (k+1)(k+2)/2。
__host__ __device__ inline int dim(int k) {
    return (k + 1) * (k + 2) / 2;
}
#ifdef __CUDACC__
// 查询已编译核的寄存器、本地内存、共享内存及每 SM 可驻留 block 上限。
template <class F>
void kernel_resources(std::ostream& out, const char* name, F kernel, int threads, size_t shared) {
    cudaFuncAttributes a;
    CUDA(cudaFuncGetAttributes(&a, kernel));
    int blocks;
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, shared));
    out << name << " registers_per_thread=" << a.numRegs << " local_bytes_per_thread=" << a.localSizeBytes
        << " shared_bytes_per_block=" << a.sharedSizeBytes + shared
        << " active_blocks_per_SM_limit=" << blocks << "\n";
}
// 小非负整数幂，避免通用 pow 的开销与不必要舍入差异。
__device__ inline double power(double x, int n) {
    double r = 1;
    for (int j = 0; j < n; ++j)
        r *= x;
    return r;
}
// 总次数递增，同次内 y 次数递增：1,x,y,x²,xy,y²,...。
__device__ inline void exponent(int i, int& a, int& b) {
    int d = 0;
    while (i > d) {
        i -= d + 1;
        ++d;
    }
    b = i;
    a = d - i;
}
// 以单元质心为中心、直径 h 为尺度的单项式。
__device__ inline double mono(int i, const Cell& c, double x, double y) {
    int a, b;
    exponent(i, a, b);
    return power((x - c.cx) / c.h, a) * power((y - c.cy) / c.h, b);
}
// 向量基先排列非恒定 P(k+1) 的梯度，再排列原混合元的补空间基。
// 梯度基含 1/h 尺度；基函数次序必须与 D/B 的自由度定义一致。
__device__ inline double2 basis(int i, const Cell& c, int k, double x, double y) {
    double X = (x - c.cx) / c.h, Y = (y - c.cy) / c.h;
    int a, b, ng = dim(k + 1) - 1;
    if (i < ng) {
        exponent(i + 1, a, b);
        return make_double2(a ? a * power(X, a - 1) * power(Y, b) / c.h : 0,
                            b ? b * power(X, a) * power(Y, b - 1) / c.h : 0);
    }
    int z = i - ng, d = 1;
    while (z >= d) {
        z -= d;
        ++d;
    }
    b = z;
    a = d - b;
    return make_double2(-double(b + 1) / a * power(X, a - 1) * power(Y, b + 1), power(X, a) * power(Y, b));
}
// 二项式系数，用于把二维单项式精确展开为边参数 t 的多项式。
__device__ inline double binom(int a, int b) {
    double r = 1;
    for (int j = 1; j <= b; ++j)
        r *= double(a - j + 1) / j;
    return r;
}
// 计算单项式限制到直边后的 t^r 系数；从两个积分点恢复边中点与方向。
__device__ inline double edge_coeff(int index, int r, const Cell& c, const EdgeSample* e, int ng) {
    // Recover midpoint and endpoint difference from quadrature positions and normalized t.
    const EdgeSample &u = e[0], &v = e[ng - 1];
    double dx = (v.s.x - u.s.x) / (v.t - u.t), dy = (v.s.y - u.s.y) / (v.t - u.t);
    double X = (u.s.x - u.t * dx - c.cx) / c.h, Y = (u.s.y - u.t * dy - c.cy) / c.h;
    dx /= c.h;
    dy /= c.h;
    int a, b;
    exponent(index, a, b);
    double sum = 0;
    for (int i = 0; i <= a; ++i) {
        int j = r - i;
        if (j >= 0 && j <= b)
            sum +=
                binom(a, i) * binom(b, j) * power(X, a - i) * power(Y, b - j) * power(dx, i) * power(dy, j);
    }
    return sum;
}
// 归一化边矩 ∫[-1/2,1/2] t^r dt；奇数为 0，偶数为 (1/2)^r/(r+1)。
__device__ inline double moment(int r) {
    return r % 2 ? 0 : power(0.5, r) / (r + 1);
}
// Collective block inversion. Small shared-memory matrix, partial pivoting, no per-thread dense arrays.
// 共享内存 Gauss–Jordan 求逆，带部分选主元；a 被覆盖，inv 输出行主序逆矩阵。
// 所有线程必须参与：每一步同步选主元、换行、归一化及消元，状态记录首个坏单元。
__device__ inline void inverse(double* a, double* inv, int n, int* status, int cid) {
    for (int z = threadIdx.x; z < n * n; z += blockDim.x)
        inv[z] = (z / n == z % n);
    __syncthreads();
    for (int k = 0; k < n; ++k) {
        __shared__ int pivot;
        __shared__ double diag;
        __shared__ double scale;
        if (threadIdx.x == 0) {
            pivot = k;
            scale = 0;
            for (int i = k; i < n; ++i) {
                scale = fmax(scale, fabs(a[i * n + k]));
                if (fabs(a[i * n + k]) > fabs(a[pivot * n + k]))
                    pivot = i;
            }
            diag = a[pivot * n + k];
            if (!isfinite(diag) || fabs(diag) < 1e-28)
                atomicCAS(status, 0, cid + 1);
        }
        __syncthreads();
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            if (pivot != k) {
                double t = a[k * n + j];
                a[k * n + j] = a[pivot * n + j];
                a[pivot * n + j] = t;
                t = inv[k * n + j];
                inv[k * n + j] = inv[pivot * n + j];
                inv[pivot * n + j] = t;
            }
        }
        __syncthreads();
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            a[k * n + j] /= diag;
            inv[k * n + j] /= diag;
        }
        __syncthreads();
        // One thread owns a row; eliminates using a scalar saved before the row is changed.
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (i != k) {
                double f = a[i * n + k];
                for (int j = 0; j < n; ++j) {
                    a[i * n + j] -= f * a[k * n + j];
                    inv[i * n + j] -= f * inv[k * n + j];
                }
            }
        __syncthreads();
    }
}
#endif
} // namespace vgpu
