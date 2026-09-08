/**
 * @file test_vtk_mesh_main.cpp
 * @brief 测试 StraightMeshReader::read_mesh_vtk 读取悬点网格，并检查
 *        get_mesh_data() 返回的 StraightMeshData 是否与这种混合多边形网格兼容
 *
 * 用最小的悬点网格 mesh/data_refined/square_2x2_refined.vtk：
 *   40 个单元 = 32 个四边形 + 8 个五边形，55 个节点。
 *
 * 兼容性的关键在于 StraightMeshData 的三块派生数据是否仍然正确：
 *   1. 单元属性（质心/面积/直径）—— compute_cell_properties 对 nodes_per_cell != 4
 *      走的是另一条分支，必须单独验
 *   2. 边拓扑（global_edges / edge_endpoints / edge_occurrence）—— 悬点若没被
 *      粗单元认领，会出现「只出现 1 次却不在区域边界上」的半条边
 *   3. 边界信息（boundary_nodes / boundary_edges）—— 加密区把下/上边界切碎了，
 *      boundary_edges 里出现 -1 就说明相邻边界节点之间并没有真实的边
 */

#include "../mesh/straight_mesh.h"
#include "../mesh/polygon_triangulator.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_failed = 0;
int g_passed = 0;

void check(bool ok, const std::string& name, const std::string& detail = "") {
    if (ok) {
        ++g_passed;
        std::cout << "  [PASS] " << name;
    } else {
        ++g_failed;
        std::cout << "  [FAIL] " << name;
    }
    if (!detail.empty()) std::cout << "  (" << detail << ")";
    std::cout << std::endl;
}

std::string num(double v) {
    std::ostringstream os;
    os << std::setprecision(10) << v;
    return os.str();
}

std::string num(int v) {
    std::ostringstream os;
    os << v;
    return os.str();
}

// 多边形有符号面积（鞋带公式）
double signed_area(const std::vector<double>& x, const std::vector<double>& y) {
    double a = 0.0;
    int n = static_cast<int>(x.size());
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        a += x[i] * y[j] - x[j] * y[i];
    }
    return 0.5 * a;
}

// 多边形的真实（面积加权）质心，用于和 cell_centroid_* 对照
void true_centroid(const std::vector<double>& x, const std::vector<double>& y,
                   double& cx, double& cy) {
    double a = signed_area(x, y);
    double sx = 0.0, sy = 0.0;
    int n = static_cast<int>(x.size());
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        double cross = x[i] * y[j] - x[j] * y[i];
        sx += (x[i] + x[j]) * cross;
        sy += (y[i] + y[j]) * cross;
    }
    cx = sx / (6.0 * a);
    cy = sy / (6.0 * a);
}

void cell_xy(const vem::StraightMeshData& m, int e,
             std::vector<double>& x, std::vector<double>& y) {
    int start = m.cell_node_indices[e];
    int nv = m.nodes_per_cell[e];
    x.assign(nv, 0.0);
    y.assign(nv, 0.0);
    for (int i = 0; i < nv; ++i) {
        int nid = m.cell_nodes[start + i];
        x[i] = m.node_coords[2 * nid];
        y[i] = m.node_coords[2 * nid + 1];
    }
}

}  // namespace

int main(int argc, char** argv) {
    // 默认用最小的悬点网格；可从命令行传入其他网格做规模验证
    const std::string vtk_file =
        (argc > 1) ? argv[1] : "mesh/data_refined/square_2x2_refined.vtk";
    const bool is_default = (argc <= 1);
    const double tol = 1e-9;

    std::cout << "================================================\n"
              << " read_mesh_vtk + StraightMeshData 兼容性测试\n"
              << " 网格: " << vtk_file << "\n"
              << "================================================\n" << std::endl;

    vem::StraightMeshReader reader;
    if (!reader.read_mesh_vtk(vtk_file)) {
        std::cerr << "读取失败，测试终止。请先在项目根目录运行 build/tools/refine_mesh" << std::endl;
        return 1;
    }
    const vem::StraightMeshData& m = reader.get_mesh_data();

    // ---------------------------------------------------------------- 1
    std::cout << "\n[1] 基本计数" << std::endl;
    std::cout << "        节点 " << m.num_nodes << ", 单元 " << m.num_cells
              << ", 边 " << m.num_edges << std::endl;
    if (is_default) {
        check(m.num_nodes == 55, "节点数 == 55", "实际 " + num(m.num_nodes));
        check(m.num_cells == 40, "单元数 == 40", "实际 " + num(m.num_cells));
    }
    check(static_cast<int>(m.cell_nodes.size()) == m.cell_node_indices.back(),
          "cell_nodes 长度与 cell_node_indices 末项一致");
    check(static_cast<int>(m.cell_node_indices.size()) == m.num_cells + 1,
          "cell_node_indices 长度 == num_cells + 1");

    // ---------------------------------------------------------------- 2
    std::cout << "\n[2] 单元边数分布（悬点是否真的让粗单元升格成五边形）" << std::endl;
    std::map<int, int> nv_count;
    for (int e = 0; e < m.num_cells; ++e) nv_count[m.nodes_per_cell[e]]++;
    for (std::map<int, int>::const_iterator it = nv_count.begin(); it != nv_count.end(); ++it)
        std::cout << "        " << it->first << " 边形: " << it->second << " 个" << std::endl;
    if (is_default) {
        check(nv_count.size() == 2 && nv_count[4] == 32 && nv_count[5] == 8,
              "32 个四边形 + 8 个五边形，无其他边数");
    } else {
        check(nv_count.size() == 2 && nv_count.count(4) && nv_count.count(5),
              "只含四边形与五边形");
    }

    // ---------------------------------------------------------------- 3
    std::cout << "\n[3] 单元几何属性（compute_cell_properties 的非四边形分支）" << std::endl;
    double area_sum = 0.0;
    int neg_area = 0, bad_diam = 0;
    for (int e = 0; e < m.num_cells; ++e) {
        std::vector<double> x, y;
        cell_xy(m, e, x, y);
        if (signed_area(x, y) <= 0.0) ++neg_area;
        area_sum += m.cell_area[e];
        if (m.cell_diameter[e] <= 0.0) ++bad_diam;
    }
    check(neg_area == 0, "所有单元顶点逆时针", "逆序单元 " + num(neg_area));
    check(std::fabs(area_sum - 1.0) < tol, "单元面积之和 == 区域面积 1.0",
          "实际 " + num(area_sum));
    check(bad_diam == 0, "所有单元直径 > 0");

    // 质心：四边形走 bbox 中点分支，非四边形走顶点平均分支。
    // 顶点平均对五边形并不是真实质心，这里把偏差量出来。
    double max_dev_quad = 0.0, max_dev_penta = 0.0;
    int worst_penta = -1;
    for (int e = 0; e < m.num_cells; ++e) {
        std::vector<double> x, y;
        cell_xy(m, e, x, y);
        double cx, cy;
        true_centroid(x, y, cx, cy);
        double dev = std::max(std::fabs(cx - m.cell_centroid_x[e]),
                              std::fabs(cy - m.cell_centroid_y[e]));
        if (m.nodes_per_cell[e] == 4) {
            max_dev_quad = std::max(max_dev_quad, dev);
        } else if (dev > max_dev_penta) {
            max_dev_penta = dev;
            worst_penta = e;
        }
    }
    check(max_dev_quad < tol, "四边形质心 == 真实质心", "最大偏差 " + num(max_dev_quad));
    check(max_dev_penta < tol, "五边形质心 == 真实质心", "最大偏差 " + num(max_dev_penta));
    if (max_dev_penta >= tol && worst_penta >= 0) {
        std::vector<double> x, y;
        cell_xy(m, worst_penta, x, y);
        double cx, cy;
        true_centroid(x, y, cx, cy);
        std::cout << "         └ 单元 " << worst_penta
                  << " 存储质心 (" << num(m.cell_centroid_x[worst_penta]) << ", "
                  << num(m.cell_centroid_y[worst_penta]) << ")"
                  << "  真实质心 (" << num(cx) << ", " << num(cy) << ")"
                  << "  单元直径 " << num(m.cell_diameter[worst_penta]) << std::endl;
    }

    // ---------------------------------------------------------------- 4
    std::cout << "\n[4] 边拓扑" << std::endl;
    check(static_cast<int>(m.edge_endpoints.size()) == 2 * m.num_edges,
          "edge_endpoints 长度 == 2 * num_edges");
    check(static_cast<int>(m.edge_occurrence.size()) == m.num_edges,
          "edge_occurrence 长度 == num_edges");
    check(static_cast<int>(m.global_edges.size()) == m.cell_edge_indices.back(),
          "global_edges 长度与 cell_edge_indices 末项一致");

    // 欧拉公式 V - E + F = 2（F 含无界面）
    int euler = m.num_nodes - m.num_edges + (m.num_cells + 1);
    check(euler == 2, "欧拉公式 V - E + F == 2",
          "V=" + num(m.num_nodes) + " E=" + num(m.num_edges) +
          " F=" + num(m.num_cells + 1) + " -> " + num(euler));

    // 这一条是悬点是否处理正确的判据：只出现 1 次的边必须整条躺在区域边界上。
    // 若某个悬点只被细单元一侧认领、粗单元没升格，半条边就会「只用 1 次却在内部」。
    int occ_bad = 0, once_interior = 0, once_total = 0;
    const double btol = 1e-8;
    for (int j = 0; j < m.num_edges; ++j) {
        int occ = m.edge_occurrence[j];
        if (occ != 1 && occ != 2) { ++occ_bad; continue; }
        if (occ != 1) continue;
        ++once_total;
        int a = m.edge_endpoints[2 * j], b = m.edge_endpoints[2 * j + 1];
        double xa = m.node_coords[2 * a], ya = m.node_coords[2 * a + 1];
        double xb = m.node_coords[2 * b], yb = m.node_coords[2 * b + 1];
        bool on_bdry =
            (std::fabs(xa - m.x_min) < btol && std::fabs(xb - m.x_min) < btol) ||
            (std::fabs(xa - m.x_max) < btol && std::fabs(xb - m.x_max) < btol) ||
            (std::fabs(ya - m.y_min) < btol && std::fabs(yb - m.y_min) < btol) ||
            (std::fabs(ya - m.y_max) < btol && std::fabs(yb - m.y_max) < btol);
        if (!on_bdry) ++once_interior;
    }
    check(occ_bad == 0, "edge_occurrence 只有 1 或 2", "异常 " + num(occ_bad));
    check(once_interior == 0, "只出现 1 次的边全部躺在区域边界上（悬点判据）",
          "内部半边 " + num(once_interior) + " / 单次边 " + num(once_total));

    // ---------------------------------------------------------------- 5
    std::cout << "\n[5] 边界信息" << std::endl;
    int minus_one = 0;
    for (size_t i = 0; i < m.boundary_edges.size(); ++i)
        if (m.boundary_edges[i] < 0) ++minus_one;
    check(minus_one == 0, "boundary_edges 中没有 -1（相邻边界节点之间都有真实的边）",
          "缺失 " + num(minus_one));
    check(m.num_boundary_edges == once_total,
          "num_boundary_edges == 只出现 1 次的边数",
          num(m.num_boundary_edges) + " vs " + num(once_total));
    check(!m.boundary_nodes.empty() && m.boundary_nodes.front() == m.boundary_nodes.back(),
          "boundary_nodes 首尾闭合");
    check(m.num_boundary_nodes == static_cast<int>(m.boundary_nodes.size()),
          "num_boundary_nodes 与数组长度一致");
    check(m.bnode_indices.size() == 5 && m.bedge_indices.size() == 5,
          "下/右/上/左 四段索引齐全");
    std::cout << "        边界节点 " << m.num_boundary_nodes
              << "（含闭合重复）, 边界边 " << m.num_boundary_edges << std::endl;

    // 加密区把下/上边界切碎了，这两条边的节点数应当比左/右多
    int n_bottom = m.bnode_indices[1] - m.bnode_indices[0];
    int n_right  = m.bnode_indices[2] - m.bnode_indices[1];
    int n_top    = m.bnode_indices[3] - m.bnode_indices[2];
    int n_left   = m.bnode_indices[4] - 1 - m.bnode_indices[3];
    std::cout << "        下 " << n_bottom << " / 右 " << n_right
              << " / 上 " << n_top << " / 左 " << n_left << " 个节点" << std::endl;
    check(n_bottom > n_left && n_top > n_right,
          "下/上边界节点多于左/右（加密条带竖直穿过上下边界）");

    // ---------------------------------------------------------------- 6
    std::cout << "\n[6] 单元-边邻接查询" << std::endl;
    int adj_bad = 0, bdry_pair = 0;
    for (int j = 0; j < m.num_edges; ++j) {
        std::pair<int, int> pr = reader.find_adjacent_cells_by_edge(j);
        if (pr.first < 0 || pr.second < 0) { ++adj_bad; continue; }
        if (m.edge_occurrence[j] == 1) {
            if (pr.first != pr.second) ++adj_bad; else ++bdry_pair;
        } else {
            if (pr.first == pr.second) ++adj_bad;
        }
        // 返回的单元必须真的含有这条边
        for (int k = 0; k < 2; ++k) {
            int c = (k == 0) ? pr.first : pr.second;
            int s = m.cell_edge_indices[c], n = m.nodes_per_cell[c];
            bool found = false;
            for (int t = 0; t < n; ++t)
                if (m.global_edges[s + t] == j) { found = true; break; }
            if (!found) ++adj_bad;
        }
    }
    check(adj_bad == 0, "所有边的相邻单元查询自洽", "异常 " + num(adj_bad));
    check(bdry_pair == m.num_boundary_edges,
          "边界边返回同一单元的条数 == num_boundary_edges",
          num(bdry_pair) + " vs " + num(m.num_boundary_edges));

    // ---------------------------------------------------------------- 7
    // 悬点网格独有的风险：五边形有三个**共线**顶点（悬点及其两侧端点）。
    // 求解器所有单元积分都走 PolygonTriangulator 的耳切法，而耳切法对共线顶点
    // 用 cross <= 1e-8 直接判非耳；若某个单元找不到耳会 throw，必须实测。
    std::cout << "\n[7] 多边形三角剖分（求解器单元积分的前置步骤）" << std::endl;
    {
        vem::PolygonTriangulator tri;
        int throw_count = 0, ntri_bad = 0, area_bad = 0, degenerate = 0;
        double max_area_err = 0.0, min_tri_area = 1e300;
        for (int e = 0; e < m.num_cells; ++e) {
            try {
                vem::TriangulatedElement te = tri.triangulateElement(m, e);
                if (static_cast<int>(te.triangles.size()) != m.nodes_per_cell[e] - 2)
                    ++ntri_bad;
                double err = std::fabs(te.total_area - m.cell_area[e]);
                max_area_err = std::max(max_area_err, err);
                if (err > 1e-12) ++area_bad;
                for (size_t t = 0; t < te.triangles.size(); ++t) {
                    if (te.triangles[t].area <= 0.0) ++degenerate;
                    min_tri_area = std::min(min_tri_area, te.triangles[t].area);
                }
            } catch (const std::exception& ex) {
                if (throw_count == 0)
                    std::cout << "         └ 单元 " << e << " 抛异常: " << ex.what() << std::endl;
                ++throw_count;
            }
        }
        check(throw_count == 0, "所有单元都能完成耳切剖分（含共线悬点的五边形）",
              "失败 " + num(throw_count));
        check(ntri_bad == 0, "三角形个数 == 顶点数 - 2", "异常 " + num(ntri_bad));
        check(degenerate == 0, "无零面积三角形", "异常 " + num(degenerate));
        check(area_bad == 0, "剖分面积之和 == cell_area", "最大偏差 " + num(max_area_err));
        std::cout << "        最小三角形面积 " << num(min_tri_area) << std::endl;
    }

    // ---------------------------------------------------------------- 汇总
    std::cout << "\n================================================" << std::endl;
    std::cout << " 通过 " << g_passed << " 项，失败 " << g_failed << " 项" << std::endl;
    std::cout << "================================================" << std::endl;
    return g_failed == 0 ? 0 : 1;
}
