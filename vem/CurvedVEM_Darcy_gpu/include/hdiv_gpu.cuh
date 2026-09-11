/**
 * H(div) 基础计算父类及 CPU→GPU 网格打包接口。
 * 数学基础矩阵属于 HdivGpu，方程系数与时间离散属于派生的 MixedGpu。
 * 基础数据按单元连续存储，并沿用 CPU 项目的局部自由度顺序和边法向约定。
 */
#pragma once
#include "device.cuh"
#include "../mesh/straight_mesh.h"
#include "../examples/ms_problem.h"
#include <functional>
#include <map>
#include <ostream>
namespace vgpu {
// CPU 几何回调：输入单元编号，填充积分点的物理坐标、Jacobian 与行列式。
using Geometry = std::function<void(int, Sample&)>;
// 主机端扁平网格；矩阵条目尚未计算，仅保存积分几何、偏移与方向映射。
struct HostMesh {
    // 多项式维数、单元数、全局通量/L² 自由度数等公共尺寸。
    Sizes s{};
    // 各单元的矩阵/积分偏移与几何尺度。
    std::vector<Cell> cells;
    // 所有单元的体积分点（计算域与物理域）。
    std::vector<Sample> vol;
    // 按单元局部逆时针边序排列的边积分点。
    std::vector<EdgeSample> edge;
    // map 为局部→全局通量编号；boundary 标记边界自由度。
    std::vector<int> map, boundary;
    // 内部边第 r 阶矩方向因子，反向时为 (-1)^(r+1)；边界与内部矩为 +1。
    std::vector<double> sign;
    // 键为边数，值为该尺寸桶内的单元编号。
    std::map<int, std::vector<int>> buckets;
    // nd: D/B/P 总条目；ngram: G；nw: W；nh: H；nhs: H*。
    size_t nd = 0, ngram = 0, nw = 0, nh = 0, nhs = 0;
    // 仅使用 CPU 读取器及映射建立上述数组，不启动 GPU 基础矩阵核。
    HostMesh(const vem::StraightMeshReader&, int k, int nq, int ng, const Geometry&);
};
// 基础计算父类：拥有几何显存、基础矩阵显存和分桶；派生类复用其设备视图。
class HdivGpu {
    // CPU 镜像与 GPU 几何缓冲区；Buffer 管理各自显存生命周期。
protected:
    HostMesh host_;
    Buffer<Cell> cells_;
    Buffer<Sample> vol_;
    Buffer<EdgeSample> edge_;
    // 局部编号、边界标记、求逆失败状态（0 正常，非零为单元编号+1）。
    Buffer<int> map_, boundary_, status_;
    // 各矩阵均按行存储：D(n×p)、B/P(p×n)、G(p×p)、W(q×n)、H/Hi(q×q)。
    PetscVector sign_;
    PetscDense D_, G_, W_, H_, Hs_, B_, P_, Hi_;
    std::vector<Buffer<int>> buckets_;
    // tile_ 为单次 launch 的单元上限，threads_ 为每单元 block 的线程数。
    int tile_, threads_;
    // 构造时上传几何并计算基础矩阵；禁止复制，避免设备指针重复释放。
public:
    HdivGpu(const vem::StraightMeshReader&, int k, int nq, int ng, const Geometry&, int tile = 4096,
            int threads = 128);
    virtual ~HdivGpu() = default;
    HdivGpu(const HdivGpu&) = delete;
    HdivGpu& operator=(const HdivGpu&) = delete;
    // 重建基础矩阵，可用于初始化、回归对照与基础构造计时。
    void build_basis();
    // 释放 G/B/H/H* 临时缓存，保留正式算子依赖的 D/P/W/H⁻¹。
    void compact_basis();
    // 返回无所有权的设备指针集合，生命周期不能超过本对象。
    BaseView view();
    // 只读 CPU 网格元数据，用于调度和结果说明。
    const HostMesh& host() const {
        return host_;
    }
    // 下列只读访问器用于矩阵对照；compact_basis 后 G/H/B 对应 Buffer 为空。
    const PetscDense& D() const {
        return D_;
    }
    const PetscDense& G() const {
        return G_;
    }
    const PetscDense& W() const {
        return W_;
    }
    const PetscDense& H() const {
        return H_;
    }
    const PetscDense& B() const {
        return B_;
    }
    const PetscDense& P() const {
        return P_;
    }
    // 当前基础矩阵显存字节数及基础核资源报告。
    size_t basis_bytes() const;
    void basis_resources(std::ostream&);
};
// 将原方程对象的几何接口适配成统一 CPU 打包回调。
Geometry ms_geometry(const MaxwellStefan::MSProblem&);
Geometry darcy_geometry(const Darcy::DarcyProblem&);
} // namespace vgpu
