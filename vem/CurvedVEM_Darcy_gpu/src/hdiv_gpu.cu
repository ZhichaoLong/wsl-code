/**
 * H(div) 基础矩阵与投影的 GPU 实现，不包含 Darcy/MS 系数或时间格式。
 * 同边数单元分桶，每个 CUDA block 负责一个单元，线程跨矩阵条目循环。
 * D、G、H、H*、W 的积分写入显存；投影构造的小矩阵求逆使用块共享内存。
 */
#include "hdiv_gpu.cuh"
namespace vgpu {
// 基础积分核：p=dim([P_k]^2)，q=dim(P_k)，n=本单元 H(div) 自由度数。
// D 为 n×p，G 为 p×p，H 为 q×q，Hs 为 grad×q，W 为 q×n，均按行存储。
__global__ void integrate_base(BaseView v, const int* ids) {
    int id = ids[blockIdx.x];
    Cell c = v.cells[id];
    int p = v.s.p, q = v.s.q, k = v.s.k, n = c.n;
    // G_ab = ∫ g_a·g_b：向量多项式 Gram 矩阵，使用计算域积分权重。
    for (int z = threadIdx.x; z < p * p; z += blockDim.x) {
        int i = z / p, j = z % p;
        double sum = 0;
        for (int l = 0; l < c.nv; ++l) {
            Sample t = v.vol[c.vo + l];
            auto a = basis(i, c, k, t.x, t.y), b = basis(j, c, k, t.x, t.y);
            sum += t.w * (a.x * b.x + a.y * b.y);
        }
        v.G[c.gg + z] = sum;
    }
    // H_ab = ∫ m_a m_b：散度多项式质量矩阵。
    for (int z = threadIdx.x; z < q * q; z += blockDim.x) {
        int i = z / q, j = z % q;
        double sum = 0;
        for (int l = 0; l < c.nv; ++l) {
            Sample t = v.vol[c.vo + l];
            sum += t.w * mono(i, c, t.x, t.y) * mono(j, c, t.x, t.y);
        }
        v.H[c.hh + z] = sum;
    }
    // H*_ab = ∫ m_(a+1) m_b：梯度基对应的分部积分体项。
    for (int z = threadIdx.x; z < v.s.grad * q; z += blockDim.x) {
        int i = z / q, j = z % q;
        double sum = 0;
        for (int l = 0; l < c.nv; ++l) {
            Sample t = v.vol[c.vo + l];
            sum += t.w * mono(i + 1, c, t.x, t.y) * mono(j, c, t.x, t.y);
        }
        v.Hs[c.hs + z] = sum;
    }
    // D_ij = dof_i(g_j)：先按矩次数排列边法向矩，再排列内部梯度/补空间矩。
    for (int z = threadIdx.x; z < n * p; z += blockDim.x) {
        int i = z / p, j = z % p;
        double sum = 0;
        if (i < (k + 1) * c.ne) {
            int e = i % c.ne, r = i / c.ne;
            for (int l = 0; l < v.s.ng; ++l) {
                EdgeSample t = v.edge[c.eo + e * v.s.ng + l];
                auto a = basis(j, c, k, t.s.x, t.s.y);
                sum += t.s.w * (a.x * t.nx + a.y * t.ny) * power(t.t, r);
            }
        } else {
            int a = i - (k + 1) * c.ne;
            if (a >= q - 1)
                a = a - (q - 1) + v.s.grad;
            for (int l = 0; l < c.nv; ++l) {
                Sample t = v.vol[c.vo + l];
                auto b = basis(j, c, k, t.x, t.y), d = basis(a, c, k, t.x, t.y);
                sum += t.w * (b.x * d.x + b.y * d.y);
            }
        }
        v.D[c.dd + z] = sum;
    }
    // W_ij = ∫ m_i div(phi_j)：边迹系数减去内部梯度矩，来自分部积分。
    for (int z = threadIdx.x; z < q * n; z += blockDim.x) {
        int i = z / n, j = z % n;
        double w = 0;
        if (j < (k + 1) * c.ne)
            w = edge_coeff(i, j / c.ne, c, v.edge + c.eo + (j % c.ne) * v.s.ng, v.s.ng);
        else if (j - (k + 1) * c.ne == i - 1 && i > 0)
            w = -1;
        v.W[c.ww + z] = w;
    }
}
// 投影核：先算 H⁻¹ 与归一化边质量矩阵逆，再构造 B，最后 P=G⁻¹B。
// 整个 block 共同调用 inverse；包含同步的路径不能仅由部分线程进入。
__global__ void projection(BaseView v, const int* ids, int* status) {
    int id = ids[blockIdx.x];
    Cell c = v.cells[id];
    int p = v.s.p, q = v.s.q, n = c.n, k = v.s.k, rk = k + 1;
    extern __shared__ double sh[];
    // 共享内存划分为两个 p×p 区域，复用来保存待求逆矩阵与逆矩阵。
    double *a = sh, *iv = sh + p * p;
    for (int z = threadIdx.x; z < q * q; z += blockDim.x)
        a[z] = v.H[c.hh + z];
    __syncthreads();
    inverse(a, iv, q, status, id);
    for (int z = threadIdx.x; z < q * q; z += blockDim.x)
        v.Hi[c.hh + z] = iv[z];
    __syncthreads();
    // Edge mass normalized to unit length. Length cancels between H*_e and H_e.
    for (int z = threadIdx.x; z < rk * rk; z += blockDim.x)
        a[z] = moment(z / rk + z % rk);
    __syncthreads();
    inverse(a, iv, rk, status, id);
    // 梯度行通过分部积分构造；补空间行由对应内部自由度直接给出。
    for (int z = threadIdx.x; z < p * n; z += blockDim.x) {
        int i = z / n, j = z % n;
        double b = 0;
        if (i < v.s.grad) {
            for (int r = 0; r < q; ++r) {
                double div = 0;
                for (int s = 0; s < q; ++s)
                    div += v.Hi[c.hh + r * q + s] * v.W[c.ww + s * n + j];
                b -= v.Hs[c.hs + i * q + r] * div;
            }
            if (j < rk * c.ne) {
                int degree = j / c.ne;
                const EdgeSample* e = v.edge + c.eo + (j % c.ne) * v.s.ng;
                for (int s = 0; s <= k + 1; ++s) {
                    double coef = edge_coeff(i + 1, s, c, e, v.s.ng);
                    for (int t = 0; t < rk; ++t)
                        b += coef * moment(s + t) * iv[t * rk + degree];
                }
            }
        } else if (j == rk * c.ne + (q - 1) + (i - v.s.grad))
            b = 1;
        v.B[c.dd + z] = b;
    }
    __syncthreads();
    for (int z = threadIdx.x; z < p * p; z += blockDim.x)
        a[z] = v.G[c.gg + z];
    __syncthreads();
    inverse(a, iv, p, status, id);
    for (int z = threadIdx.x; z < p * n; z += blockDim.x) {
        int i = z / n, j = z % n;
        double sum = 0;
        for (int t = 0; t < p; ++t)
            sum += iv[i * p + t] * v.B[c.dd + t * n + j];
        v.P[c.dd + z] = sum;
    }
}
// CPU 网格打包完成后一次上传几何、局部编号、方向符号和单元分桶。
// 父类构造基础矩阵，方程子类再分配系数、质量块和求解数据。
HdivGpu::HdivGpu(const vem::StraightMeshReader& m, int k, int nq, int ng, const Geometry& g, int tile,
                 int threads)
    : host_(m, k, nq, ng, g), tile_(tile), threads_(threads) {
    if (tile < 1 || threads < 32 || threads > 256 || (threads & (threads - 1)))
        throw std::invalid_argument("tile > 0 and threads in {32,64,128,256} required");
    cells_.upload(host_.cells);
    vol_.upload(host_.vol);
    edge_.upload(host_.edge);
    map_.upload(host_.map);
    sign_.upload(host_.sign);
    boundary_.upload(host_.boundary);
    status_.resize(1);
    D_.resize(host_.nd);
    B_.resize(host_.nd);
    P_.resize(host_.nd);
    G_.resize(host_.ngram);
    W_.resize(host_.nw);
    H_.resize(host_.nh);
    Hi_.resize(host_.nh);
    Hs_.resize(host_.nhs);
    for (auto& b : host_.buckets) {
        buckets_.emplace_back();
        buckets_.back().upload(b.second);
    }
    build_basis();
}
// 生成只含尺寸与裸设备指针的轻量视图，按值传给 CUDA 核；不转移所有权。
BaseView HdivGpu::view() {
    return {host_.s,   cells_.data(), vol_.data(), edge_.data(), map_.data(), sign_.data(), D_.data(),
            G_.data(), W_.data(),     H_.data(),   Hs_.data(),   B_.data(),   P_.data(),    Hi_.data()};
}
// 按桶分片 launch；每片最多 tile_ 个 block，片内每个 block 对应一个单元。
// 末尾同步并检查求逆状态；奇异单元编号由设备端报告给 CPU。
void HdivGpu::build_basis() {
    G_.resize(host_.ngram);
    H_.resize(host_.nh);
    Hs_.resize(host_.nhs);
    B_.resize(host_.nd);
    status_.zero();
    auto v = view();
    for (auto& b : buckets_)
        for (size_t start = 0; start < b.size(); start += tile_) {
            int count = std::min(size_t(tile_), b.size() - start);
            integrate_base<<<count, threads_>>>(v, b.data() + start);
            CUDA(cudaGetLastError());
            projection<<<count, threads_, 2 * v.s.p * v.s.p * sizeof(double)>>>(v, b.data() + start,
                                                                                status_.data());
            CUDA(cudaGetLastError());
        }
    CUDA(cudaDeviceSynchronize());
    int s = status_.download()[0];
    if (s)
        throw std::runtime_error("singular basis matrix in cell " + std::to_string(s - 1));
}
// 统计当前仍分配的基础矩阵显存字节数，不含几何与求解器向量。
size_t HdivGpu::basis_bytes() const {
    return sizeof(double) *
           (D_.size() + G_.size() + W_.size() + H_.size() + Hs_.size() + B_.size() + P_.size() + Hi_.size());
}
// 查询编译后的核寄存器、共享内存与可驻留 block 数，写入资源日志。
void HdivGpu::basis_resources(std::ostream& out) {
    kernel_resources(out, "integrate_base", integrate_base, threads_, 0);
    kernel_resources(out, "projection", projection, threads_, 2 * host_.s.p * host_.s.p * sizeof(double));
}
// 生产迭代保留 D/P/W/H⁻¹；释放只用于基础构造和对照的 G/H/H*/B。
void HdivGpu::compact_basis() {
    CUDA(cudaDeviceSynchronize());
    G_.resize(0);
    H_.resize(0);
    Hs_.resize(0);
    B_.resize(0);
}
} // namespace vgpu
