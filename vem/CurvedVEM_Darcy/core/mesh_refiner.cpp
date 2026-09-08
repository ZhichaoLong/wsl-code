#include "mesh_refiner.h"
#include "../mesh/straight_mesh.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>

namespace vem {

namespace {

inline double node_x(const std::vector<double>& c, int i) { return c[2 * i]; }
inline double node_y(const std::vector<double>& c, int i) { return c[2 * i + 1]; }

// 边键：小端在前。相邻两个单元看到同一条边时得到同一个键，
// 中点因此只被创建一次 —— 这正是悬点在两侧严格重合的保证。
typedef std::pair<int, int> EdgeKey;
inline EdgeKey make_edge_key(int a, int b) {
    return a < b ? EdgeKey(a, b) : EdgeKey(b, a);
}

// 单元是否整体落在 x 条带内。
// 用「全部节点」而非质心判据：条带边界落在格线上时两者等价，
// 不落在格线上时全节点判据不会切出半个单元。
bool cell_all_in_strip(const StraightMeshData& in, int e,
                       double x_lo, double x_hi, double tol) {
    int start = in.cell_node_indices[e];
    int nv = in.nodes_per_cell[e];
    for (int i = 0; i < nv; ++i) {
        double x = node_x(in.node_coords, in.cell_nodes[start + i]);
        if (x < x_lo - tol || x > x_hi + tol) return false;
    }
    return true;
}

// 多边形有向面积（shoelace），逆时针为正。
double signed_area(const PolyMeshData& m, int e) {
    int start = m.cell_offsets[e];
    int nv = m.nodes_of_cell(e);
    double a = 0.0;
    for (int i = 0; i < nv; ++i) {
        int p = m.cell_nodes[start + i];
        int q = m.cell_nodes[start + (i + 1) % nv];
        a += node_x(m.node_coords, p) * node_y(m.node_coords, q)
           - node_x(m.node_coords, q) * node_y(m.node_coords, p);
    }
    return 0.5 * a;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// refine_x_strip
// ---------------------------------------------------------------------------

bool refine_x_strip(const StraightMeshData& in,
                    double x_lo, double x_hi,
                    PolyMeshData& out,
                    RefineStats& stats,
                    double tol) {
    out.clear();
    stats = RefineStats();
    stats.num_cells_in = in.num_cells;

    // ---- 1. 标记条带内单元；同时拒绝非四边形输入 ----
    std::vector<char> in_strip(in.num_cells, 0);
    for (int e = 0; e < in.num_cells; ++e) {
        if (in.nodes_per_cell[e] != 4) return false;
        if (cell_all_in_strip(in, e, x_lo, x_hi, tol)) {
            in_strip[e] = 1;
            ++stats.num_refined;
        }
    }

    // ---- 2. 原始节点原样搬过去，编号不变 ----
    // 必须先于新节点，否则后面记下的中点编号会被整体顶掉。
    out.node_coords = in.node_coords;
    out.num_nodes   = in.num_nodes;

    // ---- 3. 为所有加密单元的边建中点，跨单元去重 ----
    // 先一次性建完，条带外单元才能在第 4 步查到自己边上有没有悬点。
    std::map<EdgeKey, int> midpoint;
    for (int e = 0; e < in.num_cells; ++e) {
        if (!in_strip[e]) continue;
        int start = in.cell_node_indices[e];
        for (int i = 0; i < 4; ++i) {
            int a = in.cell_nodes[start + i];
            int b = in.cell_nodes[start + (i + 1) % 4];
            EdgeKey key = make_edge_key(a, b);
            if (midpoint.count(key)) continue;
            midpoint[key] = out.num_nodes++;
            out.node_coords.push_back(
                0.5 * (node_x(in.node_coords, a) + node_x(in.node_coords, b)));
            out.node_coords.push_back(
                0.5 * (node_y(in.node_coords, a) + node_y(in.node_coords, b)));
        }
    }

    // ---- 4. 生成输出单元 ----
    out.cell_offsets.push_back(0);
    std::vector<int> poly;

    for (int e = 0; e < in.num_cells; ++e) {
        int start = in.cell_node_indices[e];
        const int n0 = in.cell_nodes[start + 0];
        const int n1 = in.cell_nodes[start + 1];
        const int n2 = in.cell_nodes[start + 2];
        const int n3 = in.cell_nodes[start + 3];

        if (in_strip[e]) {
            // 加密单元：4 个边中点 + 1 个形心，切成 4 个子四边形
            int m01 = midpoint[make_edge_key(n0, n1)];
            int m12 = midpoint[make_edge_key(n1, n2)];
            int m23 = midpoint[make_edge_key(n2, n3)];
            int m30 = midpoint[make_edge_key(n3, n0)];

            int c = out.num_nodes++;
            out.node_coords.push_back(0.25 * (node_x(in.node_coords, n0)
                                            + node_x(in.node_coords, n1)
                                            + node_x(in.node_coords, n2)
                                            + node_x(in.node_coords, n3)));
            out.node_coords.push_back(0.25 * (node_y(in.node_coords, n0)
                                            + node_y(in.node_coords, n1)
                                            + node_y(in.node_coords, n2)
                                            + node_y(in.node_coords, n3)));

            // 四个子单元，均保持逆时针
            const int sub[4][4] = {
                { n0, m01,   c, m30 },
                { m01, n1, m12,   c },
                {   c, m12, n2, m23 },
                { m30,  c, m23,  n3 }
            };
            for (int q = 0; q < 4; ++q) {
                for (int v = 0; v < 4; ++v)
                    out.cell_nodes.push_back(sub[q][v]);
                out.cell_offsets.push_back(out.cell_offsets.back() + 4);
                ++out.num_cells;
                ++stats.num_quads;
            }
            if (4 > stats.max_poly) stats.max_poly = 4;
        } else {
            // 条带外单元：逐边查悬点，查到就顺序插入 —— 四边形升格为多边形
            poly.clear();
            const int corner[4] = { n0, n1, n2, n3 };
            for (int i = 0; i < 4; ++i) {
                poly.push_back(corner[i]);
                std::map<EdgeKey, int>::const_iterator it =
                    midpoint.find(make_edge_key(corner[i], corner[(i + 1) % 4]));
                if (it != midpoint.end()) {
                    poly.push_back(it->second);
                    ++stats.num_hanging;
                }
            }

            int nv = static_cast<int>(poly.size());
            for (int i = 0; i < nv; ++i)
                out.cell_nodes.push_back(poly[i]);
            out.cell_offsets.push_back(out.cell_offsets.back() + nv);
            ++out.num_cells;

            if      (nv == 4) ++stats.num_quads;
            else if (nv == 5) ++stats.num_pentagons;
            else              ++stats.num_other;
            if (nv > stats.max_poly) stats.max_poly = nv;
        }
    }

    stats.num_cells_out = out.num_cells;
    return true;
}

// ---------------------------------------------------------------------------
// 输出
// ---------------------------------------------------------------------------

bool write_vtk(const PolyMeshData& mesh, const std::string& filename,
               const std::string& title) {
    std::ofstream f(filename.c_str());
    if (!f.is_open()) return false;
    f << std::setprecision(17);

    f << "# vtk DataFile Version 3.0\n";
    f << title << "\n";
    f << "ASCII\n";
    f << "DATASET UNSTRUCTURED_GRID\n\n";

    f << "POINTS " << mesh.num_nodes << " double\n";
    for (int i = 0; i < mesh.num_nodes; ++i)
        f << mesh.node_coords[2 * i] << " "
          << mesh.node_coords[2 * i + 1] << " 0\n";

    // CELLS 的第二个数是「整数总个数」= 每个单元的 (1 + 顶点数) 之和
    int total = 0;
    for (int e = 0; e < mesh.num_cells; ++e)
        total += 1 + mesh.nodes_of_cell(e);

    f << "\nCELLS " << mesh.num_cells << " " << total << "\n";
    for (int e = 0; e < mesh.num_cells; ++e) {
        int start = mesh.cell_offsets[e];
        int nv = mesh.nodes_of_cell(e);
        f << nv;
        for (int i = 0; i < nv; ++i) f << " " << mesh.cell_nodes[start + i];
        f << "\n";
    }

    // 四边形也写成 VTK_POLYGON(7)，读取端不必分支
    f << "\nCELL_TYPES " << mesh.num_cells << "\n";
    for (int e = 0; e < mesh.num_cells; ++e) f << "7\n";

    // 顶点数作为 cell data，方便在 ParaView 里一眼看出五边形在哪
    f << "\nCELL_DATA " << mesh.num_cells << "\n";
    f << "SCALARS num_vertices int 1\n";
    f << "LOOKUP_TABLE default\n";
    for (int e = 0; e < mesh.num_cells; ++e)
        f << mesh.nodes_of_cell(e) << "\n";

    return f.good();
}

bool write_poly(const PolyMeshData& mesh, const std::string& filename) {
    std::ofstream f(filename.c_str());
    if (!f.is_open()) return false;
    f << std::setprecision(17);

    f << "NODES " << mesh.num_nodes << "\n";
    for (int i = 0; i < mesh.num_nodes; ++i)
        f << mesh.node_coords[2 * i] << " "
          << mesh.node_coords[2 * i + 1] << "\n";

    f << "CELLS " << mesh.num_cells << "\n";
    for (int e = 0; e < mesh.num_cells; ++e) {
        int start = mesh.cell_offsets[e];
        int nv = mesh.nodes_of_cell(e);
        f << nv;
        for (int i = 0; i < nv; ++i) f << " " << mesh.cell_nodes[start + i];
        f << "\n";
    }

    return f.good();
}

// ---------------------------------------------------------------------------
// validate_poly_mesh
// ---------------------------------------------------------------------------

int validate_poly_mesh(const PolyMeshData& mesh,
                       std::vector<std::string>& messages,
                       double tol) {
    messages.clear();
    std::ostringstream os;

    // --- A. 结构自洽 ---
    if (mesh.node_coords.size() != static_cast<size_t>(2 * mesh.num_nodes)) {
        os << "node_coords 长度 " << mesh.node_coords.size()
           << " != 2 * num_nodes " << 2 * mesh.num_nodes;
        messages.push_back(os.str()); os.str("");
    }
    if (mesh.cell_offsets.size() != static_cast<size_t>(mesh.num_cells + 1)) {
        os << "cell_offsets 长度 " << mesh.cell_offsets.size()
           << " != num_cells + 1 = " << mesh.num_cells + 1;
        messages.push_back(os.str()); os.str("");
        return static_cast<int>(messages.size());   // 后续检查依赖 offsets
    }
    if (!mesh.cell_offsets.empty()
        && mesh.cell_nodes.size() != static_cast<size_t>(mesh.cell_offsets.back())) {
        os << "cell_nodes 长度 " << mesh.cell_nodes.size()
           << " != cell_offsets.back() " << mesh.cell_offsets.back();
        messages.push_back(os.str()); os.str("");
        return static_cast<int>(messages.size());
    }

    // --- B. 逐单元：索引范围、重复顶点、逆时针、面积非退化 ---
    double total_area = 0.0;
    for (int e = 0; e < mesh.num_cells; ++e) {
        int start = mesh.cell_offsets[e];
        int nv = mesh.nodes_of_cell(e);

        if (nv < 3) {
            os << "单元 " << e << " 只有 " << nv << " 个顶点";
            messages.push_back(os.str()); os.str("");
            continue;
        }

        bool bad_index = false;
        std::set<int> seen;
        for (int i = 0; i < nv; ++i) {
            int n = mesh.cell_nodes[start + i];
            if (n < 0 || n >= mesh.num_nodes) {
                os << "单元 " << e << " 的顶点 " << n << " 越界 [0, "
                   << mesh.num_nodes - 1 << "]";
                messages.push_back(os.str()); os.str("");
                bad_index = true;
                break;
            }
            if (!seen.insert(n).second) {
                os << "单元 " << e << " 顶点 " << n << " 重复";
                messages.push_back(os.str()); os.str("");
                bad_index = true;
                break;
            }
        }
        if (bad_index) continue;

        double a = signed_area(mesh, e);
        if (a <= tol) {
            os << "单元 " << e << " 有向面积 " << a
               << " 非正（顶点未按逆时针排列或单元退化）";
            messages.push_back(os.str()); os.str("");
        }
        total_area += std::fabs(a);
    }

    // --- C. 边的流形性 —— 悬点处理是否正确，全看这一条 ---
    // 若某个悬点只被细单元一侧认领、粗单元没升格成多边形，
    // 半条边就会只出现 1 次却又不在区域边界上，这里立刻抓到。
    double xmin = 0, xmax = 0, ymin = 0, ymax = 0;
    if (mesh.num_nodes > 0) {
        xmin = xmax = mesh.node_coords[0];
        ymin = ymax = mesh.node_coords[1];
        for (int i = 1; i < mesh.num_nodes; ++i) {
            double x = mesh.node_coords[2 * i];
            double y = mesh.node_coords[2 * i + 1];
            if (x < xmin) xmin = x;
            if (x > xmax) xmax = x;
            if (y < ymin) ymin = y;
            if (y > ymax) ymax = y;
        }
    }

    std::map<std::pair<int, int>, int> edge_count;
    for (int e = 0; e < mesh.num_cells; ++e) {
        int start = mesh.cell_offsets[e];
        int nv = mesh.nodes_of_cell(e);
        if (nv < 3) continue;
        for (int i = 0; i < nv; ++i) {
            int a = mesh.cell_nodes[start + i];
            int b = mesh.cell_nodes[start + (i + 1) % nv];
            if (a < 0 || b < 0 || a >= mesh.num_nodes || b >= mesh.num_nodes)
                continue;
            ++edge_count[a < b ? std::make_pair(a, b) : std::make_pair(b, a)];
        }
    }

    int n_boundary = 0, n_interior = 0, n_bad = 0;
    for (std::map<std::pair<int, int>, int>::const_iterator it = edge_count.begin();
         it != edge_count.end(); ++it) {
        int cnt = it->second;
        if (cnt == 2) { ++n_interior; continue; }
        if (cnt > 2) {
            if (n_bad < 10) {
                os << "边 (" << it->first.first << ", " << it->first.second
                   << ") 被 " << cnt << " 个单元共享（应为 1 或 2）";
                messages.push_back(os.str()); os.str("");
            }
            ++n_bad;
            continue;
        }
        // cnt == 1：必须整条落在区域外边界上
        int a = it->first.first, b = it->first.second;
        double ax = mesh.node_coords[2*a],     ay = mesh.node_coords[2*a + 1];
        double bx = mesh.node_coords[2*b],     by = mesh.node_coords[2*b + 1];
        bool on_bdry =
            (std::fabs(ax - xmin) < tol && std::fabs(bx - xmin) < tol) ||
            (std::fabs(ax - xmax) < tol && std::fabs(bx - xmax) < tol) ||
            (std::fabs(ay - ymin) < tol && std::fabs(by - ymin) < tol) ||
            (std::fabs(ay - ymax) < tol && std::fabs(by - ymax) < tol);
        if (!on_bdry) {
            if (n_bad < 10) {
                os << "内部边 (" << a << ", " << b << ") 只被 1 个单元使用"
                   << " —— 悬点未被相邻粗单元认领";
                messages.push_back(os.str()); os.str("");
            }
            ++n_bad;
        } else {
            ++n_boundary;
        }
    }
    if (n_bad > 10) {
        os << "……另有 " << (n_bad - 10) << " 条问题边未列出";
        messages.push_back(os.str()); os.str("");
    }
    (void)n_interior;
    (void)n_boundary;

    // --- D. 面积守恒：单元面积之和应等于外接矩形面积 ---
    // 覆盖了「漏单元」「单元重叠」两类错误，是 B、C 之外的独立一票。
    double rect_area = (xmax - xmin) * (ymax - ymin);
    if (std::fabs(total_area - rect_area) > 1e-9 * (rect_area > 0 ? rect_area : 1.0)) {
        os << "单元面积之和 " << total_area << " != 外接矩形面积 " << rect_area
           << "（差 " << (total_area - rect_area) << "）";
        messages.push_back(os.str()); os.str("");
    }

    // --- E. 孤立节点 ---
    std::vector<char> used(mesh.num_nodes, 0);
    for (size_t i = 0; i < mesh.cell_nodes.size(); ++i) {
        int n = mesh.cell_nodes[i];
        if (n >= 0 && n < mesh.num_nodes) used[n] = 1;
    }
    int n_orphan = 0;
    for (int i = 0; i < mesh.num_nodes; ++i) if (!used[i]) ++n_orphan;
    if (n_orphan > 0) {
        os << "有 " << n_orphan << " 个节点未被任何单元引用";
        messages.push_back(os.str()); os.str("");
    }

    return static_cast<int>(messages.size());
}

}  // namespace vem
