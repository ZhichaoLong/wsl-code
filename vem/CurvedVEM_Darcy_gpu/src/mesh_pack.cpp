/**
 * 沿用原 CPU 网格的节点、逆时针单元边序与全局边编号，打包 GPU 所需静态数据。
 * CPU 完成三角剖分、积分点、映射/Jacobian 及自由度映射；GPU 再计算基础矩阵。
 * 内部边反向时第 r 阶法向矩乘 (-1)^(r+1)，边界仍沿用原程序外法向约定。
 */
#include "hdiv_gpu.cuh"
#include "../mesh/polygon_triangulator.h"
#include "../core/gauss_quadrature.h"
#include <limits>
namespace vgpu {
namespace {
// Degree-4 extension: 6x6 Duffy rule integrates the degree-9 cross moments.
// 生成三角形体积分点；36 点规则通过 6×6 Duffy 变换支持 k=4 交叉矩。
std::vector<vem::QuadPoint> volume_points(const vem::GaussQuadrature& quad, const std::vector<double>& v,
                                          int nq) {
    if (nq != 36)
        return quad.get_triangle_points(v, nq);
    auto line = quad.get_1D_points(vem::QuadratureType::GAUSS_LEGENDRE, 6, 0, 1);
    std::vector<vem::QuadPoint> out;
    double det = std::abs((v[2] - v[0]) * (v[5] - v[1]) - (v[4] - v[0]) * (v[3] - v[1]));
    for (const auto& a : line)
        for (const auto& b : line) {
            double u = a.x[0], w = (1 - u) * b.x[0];
            out.emplace_back(v[0] + u * (v[2] - v[0]) + w * (v[4] - v[0]),
                             v[1] + u * (v[3] - v[1]) + w * (v[5] - v[1]), a.w * b.w * (1 - u) * det);
        }
    return out;
}
} // namespace
// 遍历 CPU 单元，构造扁平数组与每单元偏移，不修改原始网格对象。
HostMesh::HostMesh(const vem::StraightMeshReader& reader, int k, int nq, int ng, const Geometry& geometry) {
    if (k < 0 || k > 4)
        throw std::invalid_argument("GPU degree must be 0..4");
    if (k == 3 && nq != 12 && nq != 36)
        throw std::invalid_argument("k=3 requires triangle rule 12 or 36");
    if (k == 4 && nq != 36)
        throw std::invalid_argument("k=4 requires triangle rule 36");
    if (k >= 1 && nq == 1)
        throw std::invalid_argument("triangle rule 1 under-integrates k>=1");
    if (k >= 2 && (nq == 3 || nq == 4))
        throw std::invalid_argument("triangle rule under-integrates k>=2");
    if (ng < k + 2)
        throw std::invalid_argument("edge rule needs at least k+2 points");
    const auto& m = reader.get_mesh_data();
    if (m.num_cells <= 0)
        throw std::invalid_argument("empty mesh");
    // 全局通量：各阶边矩块在前、单元内部矩在后；L² 每单元 q 个系数。
    s.k = k;
    s.q = dim(k);
    s.p = 2 * s.q;
    s.grad = dim(k + 1) - 1;
    s.internal = s.q - 1 + k * (k + 1) / 2;
    s.nc = m.num_cells;
    s.nf = m.num_edges * (k + 1) + m.num_cells * s.internal;
    s.np = s.q * m.num_cells;
    s.ng = ng;
    vem::GaussQuadrature quad;
    vem::PolygonTriangulator tri;
    for (int e = 0; e < s.nc; ++e) {
        Cell c{};
        c.ne = m.nodes_per_cell[e];
        c.n = (k + 1) * c.ne + s.internal;
        c.cx = m.cell_centroid_x[e];
        c.cy = m.cell_centroid_y[e];
        c.h = m.cell_diameter[e];
        // 保存本单元在体积分、边积分、局部映射和各矩阵连续数组中的起始位置。
        c.vo = vol.size();
        c.eo = edge.size();
        c.lo = map.size();
        c.dd = nd;
        c.gg = ngram;
        c.ww = nw;
        c.hh = nh;
        c.hs = nhs;
        auto tr = tri.triangulateElement(m, e);
        c.area = tr.total_area;
        if (!(c.h > 0 && c.area > 0))
            throw std::invalid_argument("degenerate element");
        // 每个积分点同时保存计算域坐标与物理域映射，避免时间步中重复几何计算。
        for (const auto& t : tr.triangles)
            for (const auto& q : volume_points(quad, t.vertices, nq)) {
                Sample a{};
                a.x = q.x[0];
                a.y = q.x[1];
                a.w = q.w;
                geometry(e, a);
                if (!(a.det > 0) || !std::isfinite(a.det))
                    throw std::runtime_error("nonpositive Jacobian in cell " + std::to_string(e));
                vol.push_back(a);
            }
        c.nv = vol.size() - c.vo;
        // 边参数 t∈[-1/2,1/2] 随本单元逆时针边方向递增；外法向为 (dy,-dx)/L。
        for (int j = 0; j < c.ne; ++j) {
            int a = m.cell_nodes[m.cell_node_indices[e] + j],
                b = m.cell_nodes[m.cell_node_indices[e] + (j + 1) % c.ne];
            double x = m.node_coords[2 * a], y = m.node_coords[2 * a + 1], dx = m.node_coords[2 * b] - x,
                   dy = m.node_coords[2 * b + 1] - y, L = std::hypot(dx, dy);
            for (const auto& q :
                 quad.get_line_2D_points(vem::QuadratureType::GAUSS_LEGENDRE, ng, x, y, x + dx, y + dy)) {
                EdgeSample z{};
                z.s.x = q.x[0];
                z.s.y = q.x[1];
                z.s.w = q.w;
                geometry(e, z.s);
                if (!(z.s.det > 0))
                    throw std::runtime_error("invalid boundary Jacobian");
                z.t = ((z.s.x - x) * dx + (z.s.y - y) * dy) / (L * L) - 0.5;
                z.nx = dy / L;
                z.ny = -dx / L;
                edge.push_back(z);
            }
        }
        // 同一内部边反向会同时改变法向和边参数：n→-n、t^r→(-1)^r t^r。
        // 因此偶数 r 变号、奇数 r 不变；边界自由度保持原 CPU 外法向约定。
        for (int r = 0; r <= k; ++r)
            for (int j = 0; j < c.ne; ++j) {
                int ed = m.global_edges[m.cell_edge_indices[e] + j];
                int a = m.cell_nodes[m.cell_node_indices[e] + j];
                bool reverse = a != m.edge_endpoints[2 * ed];
                map.push_back(r * m.num_edges + ed);
                sign.push_back(m.edge_occurrence[ed] == 2 && reverse && r % 2 == 0 ? -1 : 1);
                boundary.push_back(m.edge_occurrence[ed] == 1);
            }
        // 内部自由度只属于一个单元，不需要方向调整，也不需要共享边累加。
        for (int j = 0; j < s.internal; ++j) {
            map.push_back(m.num_edges * (k + 1) + e * s.internal + j);
            sign.push_back(1);
            boundary.push_back(0);
        }
        nd += c.n * s.p;
        ngram += s.p * s.p;
        nw += s.q * c.n;
        nh += s.q * s.q;
        nhs += s.grad * s.q;
        if (nd > size_t(std::numeric_limits<int>::max()) ||
            edge.size() > size_t(std::numeric_limits<int>::max()))
            throw std::overflow_error("mesh exceeds 32-bit device offset capacity");
        cells.push_back(c);
        // 按边数分桶，让同一 launch 内单元矩阵尺寸一致，减少分支和资源浪费。
        buckets[c.ne].push_back(e);
    }
}
// 适配 MS 的逐单元映射接口，支持正弦、半圆环及 TriBlockSwirl Q8。
Geometry ms_geometry(const MaxwellStefan::MSProblem& p) {
    return [&p](int c, Sample& s) {
        auto x = p.get_mapping().physical_coords(c, s.x, s.y);
        auto j = p.get_mapping().jacobian(c, s.x, s.y);
        s.X = x.x;
        s.Y = x.y;
        s.j00 = j.xx;
        s.j01 = j.xy;
        s.j10 = j.yx;
        s.j11 = j.yy;
        s.det = p.get_mapping().jacobian_det(c, s.x, s.y);
    };
}
// 适配 Darcy 的全局映射接口，填充相同的 GPU 静态几何结构。
Geometry darcy_geometry(const Darcy::DarcyProblem& p) {
    return [&p](int, Sample& s) {
        auto x = p.get_mapping().physical_coords(s.x, s.y);
        auto j = p.get_mapping().jacobian(s.x, s.y);
        s.X = x.x;
        s.Y = x.y;
        s.j00 = j.xx;
        s.j01 = j.xy;
        s.j10 = j.yx;
        s.j11 = j.yy;
        s.det = p.get_mapping().jacobian_det(s.x, s.y);
    };
}
} // namespace vgpu
