/**
 * Darcy / Maxwell–Stefan 方程层的 GPU 核与主机调度。
 * 继承 HdivGpu 的 D/P/W 等基础数据，只构造多项式加权 Gram、稳定化和质量因子。
 * 每单元按因子应用混合块，不保存完整单元刚度矩阵，也不组装全局 CSR。
 * 通量共享边通过方向 gather/scatter 与 atomicAdd 汇总；L² 自由度由本单元独占。
 */
#include "equation_gpu.cuh"
#include "problem_device.cuh"
#include <limits>
namespace vgpu {
// 全局 L² 索引：先跳过所有组分通量块，再按组分、单元、单项式排列。
__device__ inline int pressure(const EquationView& v, int id, int s, int j) {
    return v.ns * v.b.s.nf + s * v.b.s.np + id * v.b.s.q + j;
}
// 每 block 一个单元，线程遍历体积分点，直接保存原始浓度初值。
__global__ void init_samples(BaseView b, int ns, int example, double* init) {
    int id = blockIdx.x;
    Cell c = b.cells[id];
    for (int i = threadIdx.x; i < c.nv; i += blockDim.x) {
        double u[3];
        initial(example, b.vol[c.vo + i], u);
        for (int s = 0; s < ns; ++s)
            init[(c.vo + i) * ns + s] = u[s];
    }
}
// 缓存当前时刻的数据；每积分点每组分依次存 c、Jx、Jy、f。
__global__ void sample_exact(EquationView v, double time, double* out) {
    int id = blockIdx.x;
    Cell c = v.b.cells[id];
    for (int l = threadIdx.x; l < c.nv; l += blockDim.x) {
        double conc[3], fx[3], fy[3], f[3];
        data(v.ns, v.example, v.b.vol[c.vo + l], time, conc, fx, fy, f);
        for (int i = 0; i < v.ns; ++i) {
            int offset = ((c.vo + l) * v.ns + i) * 4;
            out[offset] = conc[i];
            out[offset + 1] = fx[i];
            out[offset + 2] = fy[i];
            out[offset + 3] = f[i];
        }
    }
}
// 积分点浓度来自初值或 Picard 的 L² 系数，进而构造组分耦合系数。
__global__ void coefficients(EquationView v, const double* guess, bool init, double* coeff) {
    int id = blockIdx.x;
    Cell c = v.b.cells[id];
    int ns = v.ns;
    for (int l = threadIdx.x; l < c.nv; l += blockDim.x) {
        Sample t = v.b.vol[c.vo + l];
        double u[3] = {0, 0, 0};
        for (int i = 0; i < ns; ++i) {
            if (init)
                u[i] = v.initial[(c.vo + l) * ns + i];
            else
                for (int j = 0; j < v.b.s.q; ++j)
                    u[i] += guess[pressure(v, id, i, j)] * mono(j, c, t.x, t.y);
        }
        for (int i = 0; i < ns; ++i)
            for (int j = 0; j < ns; ++j)
                coeff[(c.vo + l) * ns * ns + i * ns + j] = ns == 1 ? 1 : aij(i, j, u);
    }
}
// Piola 变换下 K=I 的通量度量：T=JᵀJ/det(J)，按 2×2 行主序保存。
__device__ inline void tensor(const Sample& t, double* a) {
    a[0] = (t.j00 * t.j00 + t.j10 * t.j10) / t.det;
    a[1] = (t.j00 * t.j01 + t.j10 * t.j11) / t.det;
    a[2] = a[1];
    a[3] = (t.j01 * t.j01 + t.j11 * t.j11) / t.det;
}
// Static geometric moments. MS coefficients are exactly linear in concentration.
// The cache is in polynomial space, not a local flux stiffness matrix.
// 预积分 C_(r,a,b)=∫m_r g_aᵀTg_b，T_r=∫m_r T，并构造物理质量矩阵。
// 每 block 一个单元，线程分担多项式矩条目；缓存只依赖固定几何。
__global__ void preintegrate_moments(EquationView v, const int* ids, double* C, double* T) {
    int id = ids[blockIdx.x];
    Cell c = v.b.cells[id];
    int p = v.b.s.p, q = v.b.s.q;
    for (int z = threadIdx.x; z < q * p * p; z += blockDim.x) {
        int r = z / (p * p), a = (z % (p * p)) / p, b = z % p;
        double sum = 0;
        for (int l = 0; l < c.nv; ++l) {
            Sample t = v.b.vol[c.vo + l];
            double metric[4];
            tensor(t, metric);
            auto ga = basis(a, c, v.b.s.k, t.x, t.y), gb = basis(b, c, v.b.s.k, t.x, t.y);
            sum +=
                t.w * mono(r, c, t.x, t.y) *
                (ga.x * (metric[0] * gb.x + metric[1] * gb.y) + ga.y * (metric[2] * gb.x + metric[3] * gb.y));
        }
        C[c.gg * q + z] = sum;
    }
    for (int z = threadIdx.x; z < q * 4; z += blockDim.x) {
        double sum = 0;
        for (int l = 0; l < c.nv; ++l) {
            Sample t = v.b.vol[c.vo + l];
            double metric[4];
            tensor(t, metric);
            sum += t.w * mono(z / 4, c, t.x, t.y) * metric[z % 4];
        }
        T[id * q * 4 + z] = sum;
    }
    for (int z = threadIdx.x; z < q * q; z += blockDim.x) {
        double sum = 0;
        for (int l = 0; l < c.nv; ++l) {
            Sample t = v.b.vol[c.vo + l];
            sum += t.w * t.det * mono(z / q, c, t.x, t.y) * mono(z % q, c, t.x, t.y);
        }
        v.M[c.hh + z] = sum;
    }
}
// 从当前 L² 系数提取 A_ij(c) 的第 r 个单项式系数（不含常数正则项）。
__device__ inline double concentration_coefficient(EquationView v, int id, int i, int j, int r,
                                                   const double* state) {
    if (i != j)
        return -bar(i, j) * state[pressure(v, id, i, r)];
    double a = 0;
    for (int l = 0; l < 3; ++l)
        if (l != i)
            a += bar(i, l) * state[pressure(v, id, l, r)];
    return a;
}
// MS 系数对浓度线性，可与固定几何矩精确收缩；仅对角组分设置稳定化。
__global__ void contract_moments(EquationView v, const int* ids, const double* state, const double* C,
                                 const double* T) {
    int id = ids[blockIdx.x];
    Cell c = v.b.cells[id];
    int p = v.b.s.p, q = v.b.s.q;
    for (int z = threadIdx.x; z < 9 * p * p; z += blockDim.x) {
        int ij = z / (p * p), ab = z % (p * p), i = ij / 3, j = ij % 3;
        double sum = i == j ? 0.1 * C[c.gg * q + ab] : 0;
        for (int r = 0; r < q; ++r)
            sum += concentration_coefficient(v, id, i, j, r, state) * C[c.gg * q + r * p * p + ab];
        v.AG[c.gg * 9 + z] = sum;
    }
    for (int i = threadIdx.x; i < 3; i += blockDim.x) {
        double na = 0, nb = 0;
        for (int j = 0; j < 4; ++j) {
            double a = 0;
            for (int r = 0; r < q; ++r)
                a += concentration_coefficient(v, id, i, i, r, state) * T[id * q * 4 + r * 4 + j];
            double b = 0.1 * T[id * q * 4 + j];
            na += a * a;
            nb += b * b;
        }
        v.alpha[id * 3 + i] = (sqrt(na) + sqrt(nb)) / c.area;
    }
}
// 直接积分路径：构造 AG_ij、稳定化 alpha_i 与 M；初值或无缓存时使用。
__global__ void integrate_equation(EquationView v, const int* ids, const double* coeff) {
    int id = ids[blockIdx.x];
    Cell c = v.b.cells[id];
    int p = v.b.s.p, q = v.b.s.q, ns = v.ns;
    for (int z = threadIdx.x; z < ns * ns * p * p; z += blockDim.x) {
        int ij = z / (p * p), i = (z % (p * p)) / p, j = z % p;
        double sum = 0;
        for (int l = 0; l < c.nv; ++l) {
            Sample t = v.b.vol[c.vo + l];
            double T[4];
            tensor(t, T);
            auto a = basis(i, c, v.b.s.k, t.x, t.y), b = basis(j, c, v.b.s.k, t.x, t.y);
            double coef = coeff[(c.vo + l) * ns * ns + ij];
            if (ns > 1 && ij / ns == ij % ns)
                coef += 0.1;
            sum += t.w * coef * (a.x * (T[0] * b.x + T[1] * b.y) + a.y * (T[2] * b.x + T[3] * b.y));
        }
        v.AG[c.gg * ns * ns + z] = sum;
    }
    for (int i = threadIdx.x; i < ns; i += blockDim.x) {
        double a[4] = {}, b[4] = {};
        for (int l = 0; l < c.nv; ++l) {
            Sample t = v.b.vol[c.vo + l];
            double T[4];
            tensor(t, T);
            double cf = coeff[(c.vo + l) * ns * ns + i * ns + i];
            for (int j = 0; j < 4; ++j) {
                a[j] += t.w * cf * T[j];
                if (ns > 1)
                    b[j] += 0.1 * t.w * T[j];
            }
        }
        double na = 0, nb = 0;
        for (int j = 0; j < 4; ++j) {
            na += a[j] * a[j];
            nb += b[j] * b[j];
        }
        v.alpha[id * ns + i] = (sqrt(na) + sqrt(nb)) / c.area;
    }
    for (int z = threadIdx.x; z < q * q; z += blockDim.x) {
        double sum = 0;
        for (int l = 0; l < c.nv; ++l) {
            Sample t = v.b.vol[c.vo + l];
            sum += t.w * t.det * mono(z / q, c, t.x, t.y) * mono(z % q, c, t.x, t.y);
        }
        v.M[c.hh + z] = sum;
    }
}
// 一个 block 应用一个单元的混合算子；先 gather 局部方向通量，再投影与 scatter。
// 通量块为 PᵀAGP + alpha*(I-DP)ᵀ(I-DP)，耦合块为 -Wᵀ/W。
__global__ void element_apply(EquationView v, const int* ids, const double* x, double* y, bool constrained) {
    int id = ids[blockIdx.x];
    Cell c = v.b.cells[id];
    int ns = v.ns, n = c.n, p = v.b.s.p, q = v.b.s.q;
    extern __shared__ double sh[];
    // 共享内存依次存 u、Pu、r=u-DPu、Dᵀr、AG(Pu)，共 ns*(2n+3p) 个 double。
    double* u = sh;
    double* z = u + ns * n;
    double* r = z + ns * p;
    double* dt = r + ns * n;
    double* wz = dt + ns * p;
    for (int j = threadIdx.x; j < ns * n; j += blockDim.x) {
        int s = j / n, i = j % n, g = s * v.b.s.nf + v.b.map[c.lo + i];
        // 边界 mask 对应 Q：受约束应用时先清零输入边界通量，自由通量乘方向符号。
        u[j] = (constrained && v.mask[g]) ? 0 : v.b.sign[c.lo + i] * x[g];
    }
    __syncthreads();
    for (int j = threadIdx.x; j < ns * p; j += blockDim.x) {
        int s = j / p, a = j % p;
        double sum = 0;
        for (int i = 0; i < n; ++i)
            sum += v.b.P[c.dd + a * n + i] * u[s * n + i];
        z[j] = sum;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < ns * n; j += blockDim.x) {
        int s = j / n, i = j % n;
        double sum = u[j];
        for (int a = 0; a < p; ++a)
            sum -= v.b.D[c.dd + i * p + a] * z[s * p + a];
        r[j] = sum;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < ns * p; j += blockDim.x) {
        int s = j / p, a = j % p;
        double d = 0, w = 0;
        for (int i = 0; i < n; ++i)
            d += v.b.D[c.dd + i * p + a] * r[s * n + i];
        dt[j] = d;
        for (int t = 0; t < ns; ++t)
            for (int b = 0; b < p; ++b)
                w += v.AG[c.gg * ns * ns + (s * ns + t) * p * p + a * p + b] * z[t * p + b];
        wz[j] = w;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < ns * n; j += blockDim.x) {
        int s = j / n, i = j % n, g = s * v.b.s.nf + v.b.map[c.lo + i];
        if (constrained && v.mask[g])
            continue;
        double a = v.alpha[id * ns + s], sum = a * r[j];
        for (int b = 0; b < p; ++b)
            sum += v.b.P[c.dd + b * n + i] * (wz[s * p + b] - a * dt[s * p + b]);
        for (int b = 0; b < q; ++b)
            sum -= v.b.W[c.ww + b * n + i] * x[pressure(v, id, s, b)];
        // scatter 再乘相同方向符号；相邻单元同时写共享边，必须原子累加。
        atomicAdd(y + g, v.b.sign[c.lo + i] * sum);
    }
    // L² 行为 W*u + rate*M*c；Darcy 另加压力均值拉格朗日乘子。
    for (int j = threadIdx.x; j < ns * q; j += blockDim.x) {
        int s = j / q, a = j % q, g = pressure(v, id, s, a);
        double sum = 0;
        for (int i = 0; i < n; ++i)
            sum += v.b.W[c.ww + a * n + i] * u[s * n + i];
        for (int b = 0; b < q; ++b)
            sum += v.rate * v.M[c.hh + a * q + b] * x[pressure(v, id, s, b)];
        if (ns == 1)
            sum += v.M[c.hh + a * q] * x[v.N - 1];
        y[g] = sum;
    }
    if (ns == 1 && threadIdx.x == 0) {
        double sum = 0;
        for (int a = 0; a < q; ++a)
            sum += v.M[c.hh + a * q] * x[pressure(v, id, 0, a)];
        atomicAdd(y + v.N - 1, sum);
    }
}
// 补上边界单位行，使完整受约束算子为 Q*A*Q + I-Q。
__global__ void finish_apply(int N, const int* mask, const double* x, double* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && mask[i])
        y[i] = x[i];
}
// 近似预条件：通量对角 + 压力的 rate*M 与 W*diag(A)⁻¹*Wᵀ 对角。
// 只生成向量，不装配 Schur 矩阵；共享内存 d 保存本单元通量对角。
__global__ void diagonal(EquationView v, const int* ids) {
    int id = ids[blockIdx.x];
    Cell c = v.b.cells[id];
    int p = v.b.s.p, q = v.b.s.q, n = c.n, ns = v.ns;
    extern __shared__ double d[];
    for (int z = threadIdx.x; z < ns * n; z += blockDim.x) {
        int s = z / n, i = z % n;
        double a = 0, stab = 0;
        for (int r = 0; r < p; ++r)
            for (int t = 0; t < p; ++t)
                a += v.b.P[c.dd + r * n + i] * v.AG[c.gg * ns * ns + (s * ns + s) * p * p + r * p + t] *
                     v.b.P[c.dd + t * n + i];
        for (int j = 0; j < n; ++j) {
            double r = (j == i);
            for (int b = 0; b < p; ++b)
                r -= v.b.D[c.dd + j * p + b] * v.b.P[c.dd + b * n + i];
            stab += r * r;
        }
        d[z] = fmax(fabs(a + v.alpha[id * ns + s] * stab), 1e-20);
        atomicAdd(v.diag + s * v.b.s.nf + v.b.map[c.lo + i], d[z]);
    }
    __syncthreads();
    for (int z = threadIdx.x; z < ns * q; z += blockDim.x) {
        int s = z / q, j = z % q;
        double sum = fabs(v.rate * v.M[c.hh + j * q + j]);
        for (int i = 0; i < n; ++i) {
            double w = v.b.W[c.ww + j * n + i];
            sum += w * w / d[s * n + i];
        }
        v.diag[pressure(v, id, s, j)] = fmax(sum, 1e-20);
        if (ns == 1)
            atomicAdd(v.diag + v.N - 1, v.M[c.hh + j * q] * v.M[c.hh + j * q] / fmax(sum, 1e-20));
    }
}
// 物理通量先通过 det(J)J⁻¹ 拉回计算域，再积分外法向边矩形成边界 lift g。
__global__ void boundary_kernel(EquationView v, double time, const int* boundary) {
    int id = blockIdx.x;
    Cell c = v.b.cells[id];
    int ns = v.ns;
    for (int z = threadIdx.x; z < ns * (v.b.s.k + 1) * c.ne; z += blockDim.x) {
        int s = z / ((v.b.s.k + 1) * c.ne), i = z % ((v.b.s.k + 1) * c.ne);
        if (!boundary[c.lo + i])
            continue;
        int g = s * v.b.s.nf + v.b.map[c.lo + i], edge = i % c.ne, r = i / c.ne;
        double sum = 0;
        for (int l = 0; l < v.b.s.ng; ++l) {
            EdgeSample t = v.b.edge[c.eo + edge * v.b.s.ng + l];
            double cx[3], fx[3], fy[3], f[3];
            data(ns, v.example, t.s, time, cx, fx, fy, f);
            double hx = t.s.j11 * fx[s] - t.s.j01 * fy[s], hy = -t.s.j10 * fx[s] + t.s.j00 * fy[s];
            sum += t.s.w * (hx * t.nx + hy * t.ny) * power(t.t, r);
        }
        v.bc[g] = sum;
        v.mask[g] = 1;
    }
}
// L² 右端为 ∫(源项+时间历史)*m_j*det(J)。
// Euler 历史 cⁿ/dt；BDF2 为 (2cⁿ-0.5cⁿ⁻¹)/dt，含 c⁰ 时直接读取积分点初值。
__global__ void rhs_kernel(EquationView v, double /*time*/, double dt, int step, bool bdf, const double* prev,
                           const double* prevprev) {
    int id = blockIdx.x;
    Cell c = v.b.cells[id];
    int q = v.b.s.q, ns = v.ns;
    for (int z = threadIdx.x; z < ns * q; z += blockDim.x) {
        int s = z / q, j = z % q;
        double sum = 0;
        for (int l = 0; l < c.nv; ++l) {
            Sample t = v.b.vol[c.vo + l];
            double source = v.exact[((c.vo + l) * ns + s) * 4 + 3];
            double history = 0;
            if (ns > 1) {
                double old = 0, older = 0;
                if (step == 1)
                    old = v.initial[(c.vo + l) * ns + s];
                else
                    for (int a = 0; a < q; ++a)
                        old += prev[pressure(v, id, s, a)] * mono(a, c, t.x, t.y);
                if (bdf && step > 1) {
                    if (step == 2)
                        older = v.initial[(c.vo + l) * ns + s];
                    else
                        for (int a = 0; a < q; ++a)
                            older += prevprev[pressure(v, id, s, a)] * mono(a, c, t.x, t.y);
                    history = (2 * old - 0.5 * older) / dt;
                } else
                    history = old / dt;
            }
            sum += t.w * t.det * (source + history) * mono(j, c, t.x, t.y);
        }
        v.rhs[pressure(v, id, s, j)] = sum;
    }
}
// 非齐次通量边界：自由行写 b-A*g，边界行直接写 g。
__global__ void lift_rhs(int N, const int* mask, const double* bc, const double* ab, double* rhs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        rhs[i] = mask[i] ? bc[i] : rhs[i] - ab[i];
}
// 只修正预条件对角数据；实际点除由 PETSc VecPointwiseDivide 执行。
__global__ void fix_diagonal(int n,const int* mask,double* d) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) d[i]=mask[i]?1:fmax(fabs(d[i]),1e-20);
}
// 每 block 一个单元：先算投影通量与 H⁻¹Wu 散度，再并行积分。
// 六项依次为计算域标量/通量误差平方、物理域标量/通量/散度误差平方和质量。
__global__ void error_kernel(EquationView v, const double* x, double time, double* values) {
    int id = blockIdx.x;
    Cell c = v.b.cells[id];
    int ns = v.ns, p = v.b.s.p, q = v.b.s.q, n = c.n;
    extern __shared__ double sh[];
    double* z = sh;
    double* div = z + ns * p;
    for (int a = threadIdx.x; a < ns * p; a += blockDim.x) {
        int s = a / p, b = a % p;
        double sum = 0;
        for (int i = 0; i < n; ++i)
            sum += v.b.P[c.dd + b * n + i] * v.b.sign[c.lo + i] * x[s * v.b.s.nf + v.b.map[c.lo + i]];
        z[a] = sum;
    }
    for (int a = threadIdx.x; a < ns * q; a += blockDim.x) {
        int s = a / q, b = a % q;
        double sum = 0;
        for (int r = 0; r < q; ++r)
            for (int i = 0; i < n; ++i)
                sum += v.b.Hi[c.hh + b * q + r] * v.b.W[c.ww + r * n + i] * v.b.sign[c.lo + i] *
                       x[s * v.b.s.nf + v.b.map[c.lo + i]];
        div[a] = sum;
    }
    __syncthreads();
    for (int s = 0; s < ns; ++s) {
        double es = 0, ef = 0, eps = 0, epf = 0, ed = 0, mass = 0;
        for (int l = threadIdx.x; l < c.nv; l += blockDim.x) {
            Sample t = v.b.vol[c.vo + l];
            const double* reference = v.exact + ((c.vo + l) * ns + s) * 4;
            double cn = 0, ux = 0, uy = 0, dn = 0;
            for (int a = 0; a < q; ++a) {
                double m = mono(a, c, t.x, t.y);
                cn += x[pressure(v, id, s, a)] * m;
                dn += div[s * q + a] * m;
            }
            for (int a = 0; a < p; ++a) {
                auto b = basis(a, c, v.b.s.k, t.x, t.y);
                ux += z[s * p + a] * b.x;
                uy += z[s * p + a] * b.y;
            }
            double dc = cn - reference[0], dx = ux - (t.j11 * reference[1] - t.j01 * reference[2]),
                   dy = uy - (-t.j10 * reference[1] + t.j00 * reference[2]);
            es += t.w * dc * dc;
            eps += t.w * t.det * dc * dc;
            ef += t.w * (dx * dx + dy * dy);
            dx = (t.j00 * ux + t.j01 * uy) / t.det - reference[1];
            dy = (t.j10 * ux + t.j11 * uy) / t.det - reference[2];
            epf += t.w * t.det * (dx * dx + dy * dy);
            double dct = 0;
            if (ns > 1 && (v.example == 2 || v.example == 4)) {
                double a = 2 * pi * sin(2 * pi * t.X) * cos(8 * pi * time),
                       b = 1.5 * pi * sin(3 * pi * t.Y) * cos(6 * pi * time);
                dct = s == 0 ? a : s == 1 ? b : -a - b;
            }
            double dd = dn / t.det - (reference[3] - dct);
            ed += t.w * t.det * dd * dd;
            mass += t.w * t.det * cn;
        }
        double a[6] = {es, ef, eps, epf, ed, mass};
        // 每组分复用共享内存进行六路树形归约，线程数必须为 2 的幂。
        double* reduce = div + ns * q;
        for (int j = 0; j < 6; ++j)
            reduce[j * blockDim.x + threadIdx.x] = a[j];
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride /= 2) {
            if (threadIdx.x < stride)
                for (int j = 0; j < 6; ++j)
                    reduce[j * blockDim.x + threadIdx.x] += reduce[j * blockDim.x + threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            for (int j = 0; j < 6; ++j)
                values[(id * ns + s) * 6 + j] = reduce[j * blockDim.x];
        __syncthreads();
    }
}
// 第二级归约：每个 block 汇总一个组分的一项指标；误差开平方，质量不变。
__global__ void reduce_errors(const double* values, int nc, int ns, double* out) {
    int z = blockIdx.x;
    extern __shared__ double s[];
    double a = 0;
    for (int e = threadIdx.x; e < nc; e += blockDim.x)
        a += values[(e * ns + z / 6) * 6 + z % 6];
    s[threadIdx.x] = a;
    __syncthreads();
    for (int d = blockDim.x / 2; d; d /= 2) {
        if (threadIdx.x < d)
            s[threadIdx.x] += s[threadIdx.x + d];
        __syncthreads();
    }
    if (!threadIdx.x)
        out[z] = z % 6 == 5 ? s[0] : sqrt(fmax(s[0], 0.0));
}
// 积分物理初始质量，供无源零通量的 MS 1/3 验证各组分守恒。
__global__ void initial_mass_kernel(EquationView v, double* values) {
    int id = blockIdx.x;
    Cell c = v.b.cells[id];
    for (int s = threadIdx.x; s < v.ns; s += blockDim.x) {
        double m = 0;
        for (int l = 0; l < c.nv; ++l) {
            Sample t = v.b.vol[c.vo + l];
            m += t.w * t.det * v.initial[(c.vo + l) * v.ns + s];
        }
        values[(id * v.ns + s) * 6 + 5] = m;
    }
}
namespace {
// 限制设备端固定大小组分数组的适用范围：Darcy 一组分或 MS 三组分。
int checked_components(int ns) {
    if (ns != 1 && ns != 3)
        throw std::invalid_argument("only Darcy (1) and MS (3) components are supported");
    return ns;
}
} // namespace
// 父类先建立基础矩阵；本层配置方程因子、初值、边界与工作向量。
MixedGpu::MixedGpu(const vem::StraightMeshReader& r, int k, int nq, int ng, const Geometry& g, int ns,
                   int example, int tile, int threads)
    : HdivGpu(r, k, nq, ng, g, tile, threads), ns_(checked_components(ns)), example_(example),
      N_(ns_ * (host_.s.nf + host_.s.np) + (ns_ == 1)) {
    if (example < 1 || example > (ns_ == 1 ? 2 : 4))
        throw std::invalid_argument("invalid equation example");
    for (const auto& c : host_.cells)
        maxn_ = std::max(maxn_, c.n);
    AG_.resize(host_.ngram * ns * ns);
    alpha_.resize(host_.s.nc * ns);
    M_.resize(host_.nh);
    rhs_.resize(N_);
    bc_.resize(N_);
    mask_.resize(N_);
    diag_.resize(N_);
    initial_.resize(host_.vol.size() * ns);
    coefficients_.resize(host_.vol.size() * ns * ns);
    exact_.resize(host_.vol.size() * ns * 4);
    rhs_work_.resize(N_);
    bc_.zero();
    mask_.zero();
    init_samples<<<host_.s.nc, threads_>>>(view(), ns_, example_, initial_.data());
    CUDA(cudaGetLastError());
    size_t cm = host_.ngram * host_.s.q, tm = size_t(host_.s.nc) * host_.s.q * 4;
    // 几何矩缓存上限为 256 MiB，超过预算自动回到直接积分，数学公式不变。
    if (ns_ == 3 && (cm + tm) * sizeof(double) <= 256ull * 1024 * 1024) {
        moments_.resize(cm);
        tensor_moments_.resize(tm);
        for (auto& b : buckets_)
            for (size_t off = 0; off < b.size(); off += tile_) {
                int count = std::min(size_t(tile_), b.size() - off);
                preintegrate_moments<<<count, threads_>>>(equation(), b.data() + off, moments_.data(),
                                                          tensor_moments_.data());
                CUDA(cudaGetLastError());
            }
    }
    CUDA(cudaDeviceSynchronize());
}
// 汇集基础视图和方程数据裸指针；核函数按值接收，不持有资源所有权。
EquationView MixedGpu::equation(bool include_vectors) {
    return {view(),    ns_,         example_,   N_,           rate_,        AG_.data(),      alpha_.data(),
            M_.data(), include_vectors ? rhs_.data() : nullptr, include_vectors ? bc_.data() : nullptr,
            include_vectors ? diag_.data() : nullptr, mask_.data(), initial_.data(), exact_.data()};
}
// 每次 Picard 更新当前浓度对应的 AG/alpha 和预条件对角；rate 为时间质量系数。
void MixedGpu::update(const double* guess, bool init, double rate) {
    if (!init && !guess)
        throw std::invalid_argument("missing Picard state");
    if (!std::isfinite(rate) || rate < 0)
        throw std::invalid_argument("invalid mass coefficient");
    rate_ = rate;
    sparse_dirty_ = true;
    ordered_dirty_ = true;
    double* cf = coefficients_.data();
    // 清零之后重新借用，不能在 Restore 之后继续沿用先前的 diag 指针。
    diag_.zero();
    auto v = equation();
    bool cached = ns_ == 3 && !init && moment_cache_enabled();
    if (!cached) {
        coefficients<<<host_.s.nc, threads_>>>(v, guess, init, cf);
        CUDA(cudaGetLastError());
    }
    for (auto& b : buckets_)
        for (size_t s = 0; s < b.size(); s += tile_) {
            int n = std::min(size_t(tile_), b.size() - s);
            if (cached)
                contract_moments<<<n, threads_>>>(v, b.data() + s, guess, moments_.data(),
                                                  tensor_moments_.data());
            else
                integrate_equation<<<n, threads_>>>(v, b.data() + s, cf);
            CUDA(cudaGetLastError());
            diagonal<<<n, threads_, ns_ * maxn_ * sizeof(double)>>>(v, b.data() + s);
            CUDA(cudaGetLastError());
        }
    CUDA(cudaDeviceSynchronize());
}
// 矩阵无关 A*x 调度；清空输出，再按边数桶分片应用单元算子。
// 输入输出不能别名，否则 scatter 会覆盖尚未读取的数据。
void MixedGpu::apply(const double* x, double* y, bool constrained) {
    if (!x || !y || x == y)
        throw std::invalid_argument("apply requires distinct nonnull input/output vectors");
    CUDA(cudaMemset(y, 0, N_ * sizeof(double)));
    auto v = equation(false);
    size_t shared = ns_ * (2 * maxn_ + 3 * host_.s.p) * sizeof(double);
    if (shared > 48 * 1024)
        throw std::runtime_error("element exceeds shared-memory budget");
    for (auto& b : buckets_)
        for (size_t s = 0; s < b.size(); s += tile_) {
            int n = std::min(size_t(tile_), b.size() - s);
            element_apply<<<n, threads_, shared>>>(v, b.data() + s, x, y, constrained);
            CUDA(cudaGetLastError());
        }
    if (constrained) {
        finish_apply<<<(N_ + 255) / 256, 256>>>(N_, mask_.data(), x, y);
        CUDA(cudaGetLastError());
    }
}
// 重算指定时刻的非齐次边界值与 mask，覆盖上一次时间步数据。
void MixedGpu::boundary_lift(double time) {
    sparse_dirty_ = true;
    ordered_dirty_ = true;
    bc_.zero();
    mask_.zero();
    boundary_kernel<<<host_.s.nc, threads_>>>(equation(), time, boundary_.data());
    CUDA(cudaGetLastError());
    fix_diagonal<<<(N_+255)/256,256>>>(N_,mask_.data(),diag_.data());
    CUDA(cudaGetLastError());
}
// 先积分历史与源项，再用未加约束的 A*g 完成边界消元。
void MixedGpu::make_rhs(double time, double dt, int step, bool bdf, const double* prev,
                        const double* prevprev) {
    if (ns_ > 1 && (!(dt > 0) || step < 1 || (step > 1 && !prev) || (bdf && step > 2 && !prevprev)))
        throw std::invalid_argument("invalid time history");
    rhs_.zero();
    sample_data(time);
    boundary_lift(time);
    rhs_kernel<<<host_.s.nc, threads_>>>(equation(), time, dt, step, bdf, prev, prevprev);
    CUDA(cudaGetLastError());
    double* ab = rhs_work_.data();
    apply(bc_.data(), ab, false);
    lift_rhs<<<(N_ + 255) / 256, 256>>>(N_, mask_.data(), bc_.data(), ab, rhs_.data());
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
}
// GPU 两级归约后仅下载每组分六个数；MS 1/3 的真解误差标记 NaN。
ErrorReport MixedGpu::errors(const double* x, double time) {
    sample_data(time);
    PetscVector per(host_.s.nc * ns_ * 6), sum(ns_ * 6);
    error_kernel<<<host_.s.nc, threads_, (ns_ * (host_.s.p + host_.s.q) + 6 * threads_) * sizeof(double)>>>(
        equation(), x, time, per.data());
    CUDA(cudaGetLastError());
    reduce_errors<<<ns_ * 6, 256, 256 * sizeof(double)>>>(per.data(), host_.s.nc, ns_, sum.data());
    CUDA(cudaGetLastError());
    auto a = sum.download();
    ErrorReport r;
    for (int s = 0; s < ns_; ++s) {
        bool exact = ns_ == 1 || example_ == 2 || example_ == 4;
        double nan = std::numeric_limits<double>::quiet_NaN();
        r.scalar_comp.push_back(exact ? a[s * 6] : nan);
        r.flux_comp.push_back(exact ? a[s * 6 + 1] : nan);
        r.scalar_phys.push_back(exact ? a[s * 6 + 2] : nan);
        r.flux_phys.push_back(exact ? a[s * 6 + 3] : nan);
        r.div_phys.push_back(exact ? a[s * 6 + 4] : nan);
        r.mass.push_back(a[s * 6 + 5]);
    }
    return r;
}
// 仅 MS 存在浓度初始质量；Darcy 返回空数组。
std::vector<double> MixedGpu::initial_mass() {
    if (ns_ == 1)
        return {};
    PetscVector per(host_.s.nc * ns_ * 6), sum(ns_ * 6);
    per.zero();
    initial_mass_kernel<<<host_.s.nc, threads_>>>(equation(), per.data());
    CUDA(cudaGetLastError());
    reduce_errors<<<ns_ * 6, 256, 256 * sizeof(double)>>>(per.data(), host_.s.nc, ns_, sum.data());
    CUDA(cudaGetLastError());
    auto a = sum.download();
    std::vector<double> out;
    for (int i = 0; i < ns_; ++i)
        out.push_back(a[i * 6 + 5]);
    return out;
}
// 同一时间步的 Picard 共用源项/真解缓存，只有时间变化才重新采样。
void MixedGpu::sample_data(double time) {
    if (!std::isfinite(time))
        throw std::invalid_argument("nonfinite time");
    if (!exact_valid_ || time != exact_time_) {
        sample_exact<<<host_.s.nc, threads_>>>(equation(), time, exact_.data());
        CUDA(cudaGetLastError());
        exact_time_ = time;
        exact_valid_ = true;
    }
}
// 只统计 AG、稳定化、质量和几何矩缓存，不含 Krylov 向量。
size_t MixedGpu::operator_bytes() const {
    return (AG_.size() + alpha_.size() + M_.size() + moments_.size() + tensor_moments_.size()) *
           sizeof(double);
}
// 报告实际编译的核资源使用量和预积分缓存大小，供选择线程数时参考。
void MixedGpu::resources(std::ostream& out) {
    basis_resources(out);
    out << "MS_moment_cache_bytes=" << (moments_.size() + tensor_moments_.size()) * sizeof(double) << "\n";
    kernel_resources(out, "contract_moments", contract_moments, threads_, 0);
    kernel_resources(out, "sample_exact", sample_exact, threads_, 0);
    kernel_resources(out, "integrate_equation", integrate_equation, threads_, 0);
    kernel_resources(out, "element_apply", element_apply, threads_,
                     ns_ * (2 * maxn_ + 3 * host_.s.p) * sizeof(double));
    kernel_resources(out, "error_kernel", error_kernel, threads_,
                     (ns_ * (host_.s.p + host_.s.q) + 6 * threads_) * sizeof(double));
}
// 每单元完整混合矩阵，直接从 P/AG/D/W/M 计算，不做全局逐项 MatSetValues。
// COO 的重复坐标由 PETSc 在 GPU 求和；边界用 Q*A*Q+I-Q，与 MatShell 完全一致。
__device__ int local_global(EquationView v, Cell c, int id, int a) {
    if (a < v.ns*c.n) return (a/c.n)*v.b.s.nf + v.b.map[c.lo+a%c.n];
    a -= v.ns*c.n;
    if (a < v.ns*v.b.s.q) return pressure(v,id,a/v.b.s.q,a%v.b.s.q);
    return v.N-1;
}
__global__ void coo_matrix(EquationView v, const int* ids, const PetscCount* offsets, double* values) {
    int id=ids[blockIdx.x]; Cell c=v.b.cells[id];
    int n=c.n, p=v.b.s.p, q=v.b.s.q, nf=v.ns*n, np=v.ns*q, d=nf+np+(v.ns==1);
    for (int z=threadIdx.x; z<d*d; z+=blockDim.x) {
        int i=z/d,j=z%d,gi=local_global(v,c,id,i),gj=local_global(v,c,id,j);
        double value=0;
        if (!v.mask[gi] && !v.mask[gj]) {
            if (i<nf && j<nf) {
                int s=i/n,t=j/n,a=i%n,b=j%n;
                for (int r=0;r<p;++r) for (int l=0;l<p;++l)
                    value += v.b.P[c.dd+r*n+a]*v.AG[c.gg*v.ns*v.ns+(s*v.ns+t)*p*p+r*p+l]*v.b.P[c.dd+l*n+b];
                if (s==t) {
                    double stab=0;
                    for(int l=0;l<n;++l) {
                        double ra=(l==a),rb=(l==b);
                        for(int r=0;r<p;++r) {
                            ra-=v.b.D[c.dd+l*p+r]*v.b.P[c.dd+r*n+a];
                            rb-=v.b.D[c.dd+l*p+r]*v.b.P[c.dd+r*n+b];
                        }
                        stab+=ra*rb;
                    }
                    value+=v.alpha[id*v.ns+s]*stab;
                }
                value*=v.b.sign[c.lo+a]*v.b.sign[c.lo+b];
            } else if (i<nf && j<nf+np) {
                int s=i/n,a=i%n,t=(j-nf)/q,b=(j-nf)%q;
                if(s==t) value=-v.b.W[c.ww+b*n+a]*v.b.sign[c.lo+a];
            } else if (i<nf+np && i>=nf && j<nf) {
                int s=(i-nf)/q,a=(i-nf)%q,t=j/n,b=j%n;
                if(s==t) value=v.b.W[c.ww+a*n+b]*v.b.sign[c.lo+b];
            } else if (i>=nf && i<nf+np && j>=nf && j<nf+np) {
                int s=(i-nf)/q,a=(i-nf)%q,t=(j-nf)/q,b=(j-nf)%q;
                if(s==t) value=v.rate*v.M[c.hh+a*q+b];
            } else if(v.ns==1) {
                if(i==d-1 && j>=nf && j<nf+np) value=v.M[c.hh+(j-nf)*q];
                if(j==d-1 && i>=nf && i<nf+np) value=v.M[c.hh+(i-nf)*q];
            }
        }
        values[offsets[id]+z]=value;
    }
}
__global__ void coo_boundary(int n, const int* mask, double* values) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) values[i]=mask[i]?1:0;
}
// CPU 只生成/排序稀疏图；矩阵数值不下载到 CPU。
void MixedGpu::graph_indices(std::vector<PetscInt>& rows, std::vector<PetscInt>& cols,
                             std::vector<PetscCount>& offsets) {
    offsets.assign(host_.s.nc+1,0);
    for(int id=0;id<host_.s.nc;++id) {
        PetscCount d=ns_*(host_.cells[id].n+host_.s.q)+(ns_==1);
        offsets[id+1]=offsets[id]+d*d;
    }
    coo_entries_=offsets.back()+N_;
    rows.resize(coo_entries_); cols.resize(coo_entries_);
    for(int id=0;id<host_.s.nc;++id) {
        const auto& c=host_.cells[id]; std::vector<PetscInt> dofs;
        for(int s=0;s<ns_;++s) for(int i=0;i<c.n;++i) dofs.push_back(s*host_.s.nf+host_.map[c.lo+i]);
        for(int s=0;s<ns_;++s) for(int j=0;j<host_.s.q;++j) dofs.push_back(ns_*host_.s.nf+s*host_.s.np+id*host_.s.q+j);
        if(ns_==1) dofs.push_back(N_-1);
        for(size_t i=0;i<dofs.size();++i) for(size_t j=0;j<dofs.size();++j) {
            auto z=offsets[id]+i*dofs.size()+j; rows[z]=dofs[i]; cols[z]=dofs[j];
        }
    }
    for(int i=0;i<N_;++i) rows[offsets.back()+i]=cols[offsets.back()+i]=i;
}
void MixedGpu::create_sparse_graph() {
    std::vector<PetscCount> offsets;
    std::vector<PetscInt> rows,cols;
    graph_indices(rows,cols,offsets);
    size_t free,total; CUDA(cudaMemGetInfo(&free,&total));
    if(size_t(coo_entries_)*sizeof(double)>free/3) throw std::runtime_error("COO storage exceeds GPU budget; use matrix-free or smaller mesh/order");
    PETSC(MatCreate(PETSC_COMM_SELF,sparse_.put()));
    PETSC(MatSetSizes(sparse_.get(),N_,N_,N_,N_));
    PETSC(MatSetType(sparse_.get(),MATSEQAIJCUSPARSE));
    PETSC(MatSetPreallocationCOO(sparse_.get(),coo_entries_,rows.data(),cols.data()));
    coo_offsets_.upload(offsets); coo_values_.resize(coo_entries_);
}
double MixedGpu::sparse_nonzeros(bool ordered) const {
    Mat matrix = ordered ? ordered_.get() : sparse_.get();
    if (!matrix) return 0;
    MatInfo info;
    PETSC(MatGetInfo(matrix, MAT_LOCAL, &info));
    return info.nz_used;
}
// A_ordered = R A R^T。只改变 COO 的图索引，复用原顺序的 GPU 数值数组。
Mat MixedGpu::ordered_matrix(const std::string& ordering) {
    Mat original=assembled_matrix();
    if(ordering=="natural") return original;
    if(!ordered_.get()) {
        PetscIs col;
        PETSC(MatGetOrdering(original,ordering.c_str(),permutation_.put(),col.put()));
        PetscBool equal; PETSC(ISEqual(permutation_.get(),col.get(),&equal));
        if(!equal) throw std::runtime_error("需要对称行列排序");
        const PetscInt* indices=nullptr;
        std::vector<PetscInt> inverse(N_);
        PETSC(ISGetIndices(permutation_.get(),&indices));
        for(int i=0;i<N_;++i)inverse[indices[i]]=i;
        PETSC(ISRestoreIndices(permutation_.get(),&indices));
        std::vector<PetscCount> offsets;std::vector<PetscInt> rows,cols;
        graph_indices(rows,cols,offsets);
        for(PetscCount i=0;i<coo_entries_;++i) {rows[i]=inverse[rows[i]];cols[i]=inverse[cols[i]];}
        PETSC(MatCreate(PETSC_COMM_SELF,ordered_.put()));
        PETSC(MatSetSizes(ordered_.get(),N_,N_,N_,N_));
        PETSC(MatSetType(ordered_.get(),MATSEQAIJCUSPARSE));
        PETSC(MatSetPreallocationCOO(ordered_.get(),coo_entries_,rows.data(),cols.data()));
        ordering_=ordering;
    }
    if(ordering!=ordering_)throw std::logic_error("一个方程对象的排序在初始化后不可更改");
    if(ordered_dirty_) {
        PETSC(MatSetValuesCOO(ordered_.get(),coo_values_.data(),INSERT_VALUES));
        coo_values_.restore(); ordered_dirty_=false;
    }
    return ordered_.get();
}
Mat MixedGpu::assembled_matrix() {
    if(!sparse_.get()) create_sparse_graph();
    if(sparse_dirty_) {
        auto v=equation();
        for(auto& bucket:buckets_) for(size_t off=0;off<bucket.size();off+=tile_) {
            int count=std::min(size_t(tile_),bucket.size()-off);
            coo_matrix<<<count,threads_>>>(v,bucket.data()+off,coo_offsets_.data(),coo_values_.data());
            CUDA(cudaGetLastError());
        }
        coo_boundary<<<(N_+255)/256,256>>>(N_,mask_.data(),coo_values_.data()+coo_entries_-N_);
        CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
        // 设备 COO 指针有效期覆盖本调用，PETSc 不取得其所有权。
        PETSC(MatSetValuesCOO(sparse_.get(),coo_values_.data(),INSERT_VALUES));
        coo_values_.restore();
        sparse_dirty_=false;
    }
    return sparse_.get();
}
} // namespace vgpu
