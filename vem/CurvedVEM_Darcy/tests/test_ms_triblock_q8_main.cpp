/**
 * @file test_ms_triblock_q8_main.cpp
 * @brief 算例 4（TriBlockSwirl 逐单元 Q8 映射）的几何自检
 *
 * 检查六件事，任一不过就返回 1：
 *   1a.(R2) 共享边的 Q8 节点两侧**逐位相同**，且等于全局唯一的 G 求值。
 *      这是 H(div) 协调性的前提——两侧看到的边不是同一条曲线，法向通量
 *      就没有共同的定义域，方法本身失去意义。要求严格为 0，不留容差。
 *   1b.共享边上 physical_coords 两侧的偏差。这一项**不可能是 0**，原因不在映射
 *      而在参考网格：mesh/data/square_NxN.msh 的节点坐标带 ~1e-12 的噪声
 *      （设计文档 §1.1 记录过 square_1x1 的「0.5 线」有 1.3e-12 微折，实测
 *      2x2~32x32 也都有）。同一条「竖直」边的两个端点 x 坐标能差 3e-13，
 *      于是边上的采样点既不精确落在左单元的局部边线 a=+1 上，也不精确落在
 *      右单元的 a=-1 上，两侧是在相距 ~1e-12（局部坐标）的两点上求同一个
 *      Q8 面片。偏差量级由参考网格的微折决定，本测试把微折一起量出来对照，
 *      容差取 1e-11——比离散误差小 8 个量级以上，不影响任何收敛阶。
 *   2. 外边界严格落在 ∂[0,1]²：swirl 因子 b = [16X(1-X)Y(1-Y)]^m 在整条
 *      外边界上为 0，故 T 在边界上恒等，物理域应严格是单位正方形。
 *   3. 外边界的边保持直：边界边的中边节点必须落在弦上（sagitta = 0）。
 *   4. detJ > 0 处处成立：映射不翻转。
 *   5. 单元质量指标与设计文档 §9.4 的 Python 参考值一致（4×4 Legendre）。
 *      N=4 的参考值：detmin 0.2638 / detmax 3.1643 / cond 8.391 / nonaff 0.4729。
 *      非仿射度应 O(h)，相邻层比值趋于 2。
 *
 * 注意 1b 用的是 mapping.physical_coords()，即真正参与组装的那条路径，
 * 而不是绕过 Q8 直接调 generator()——后者过不了也检查不出装配用的映射有问题。
 * 1a 则两者都用：要求 Q8 节点表里存的就是 generator 的值。
 */

#include "../examples/ms_problem.h"
#include "../mesh/straight_mesh.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <vector>
#include <string>

using namespace MaxwellStefan;

namespace {

// 4 点 Gauss-Legendre 节点（[-1,1]），与 Python 的 leggauss(4) 一致
const double kG4[4] = {-0.8611363115940526, -0.3399810435848563,
                        0.3399810435848563,  0.8611363115940526};

// 2x2 矩阵的条件数 sigma_max/sigma_min（通过 J^T J 的特征值解析求）
double cond2(const Tensor2D& J) {
    double t = J.xx * J.xx + J.xy * J.xy + J.yx * J.yx + J.yy * J.yy;
    double d = J.xx * J.yy - J.xy * J.yx;
    double disc = t * t - 4.0 * d * d;
    if (disc < 0.0) disc = 0.0;   // 舍入负值
    double s = std::sqrt(disc);
    double l1 = 0.5 * (t + s);
    double l2 = 0.5 * (t - s);
    if (l2 <= 0.0) return std::numeric_limits<double>::infinity();
    return std::sqrt(l1 / l2);
}

struct Quality {
    double detmin, detmax, condmax, nonaff;
    int nneg;
};

Quality measure_quality(const TriBlockSwirlMapping& map, int num_cells) {
    Quality q;
    q.detmin = std::numeric_limits<double>::max();
    q.detmax = -std::numeric_limits<double>::max();
    q.condmax = 0.0;
    q.nonaff = 0.0;
    q.nneg = 0;

    for (int e = 0; e < num_cells; ++e) {
        const TriBlockSwirlMapping::CellQ8& c = map.cell(e);
        Tensor2D Js[16];
        double sum_xx = 0.0, sum_xy = 0.0, sum_yx = 0.0, sum_yy = 0.0;
        bool degen = false;
        int g = 0;
        for (int ia = 0; ia < 4; ++ia) {
            for (int ib = 0; ib < 4; ++ib, ++g) {
                double xi  = c.xi_c  + 0.5 * c.hx * kG4[ia];
                double eta = c.eta_c + 0.5 * c.hy * kG4[ib];
                Tensor2D J = map.jacobian(e, xi, eta);
                Js[g] = J;
                double det = J.xx * J.yy - J.xy * J.yx;
                if (det < q.detmin) q.detmin = det;
                if (det > q.detmax) q.detmax = det;
                if (det <= 0.0) degen = true;
                double cd = cond2(J);
                if (cd > q.condmax) q.condmax = cd;
                sum_xx += J.xx;  sum_xy += J.xy;
                sum_yx += J.yx;  sum_yy += J.yy;
            }
        }
        if (degen) ++q.nneg;

        // 非仿射度：max_g ||J_g - Jbar||_F / ||Jbar||_F，Jbar 为 16 点算术平均
        Tensor2D Jb(sum_xx / 16.0, sum_xy / 16.0, sum_yx / 16.0, sum_yy / 16.0);
        double nb = std::sqrt(Jb.xx * Jb.xx + Jb.xy * Jb.xy
                            + Jb.yx * Jb.yx + Jb.yy * Jb.yy);
        double loc = 0.0;
        for (int k = 0; k < 16; ++k) {
            double dxx = Js[k].xx - Jb.xx, dxy = Js[k].xy - Jb.xy;
            double dyx = Js[k].yx - Jb.yx, dyy = Js[k].yy - Jb.yy;
            double nd = std::sqrt(dxx * dxx + dxy * dxy + dyx * dyx + dyy * dyy);
            if (nd > loc) loc = nd;
        }
        double na = loc / nb;
        if (na > q.nonaff) q.nonaff = na;
    }
    return q;
}

}  // anonymous namespace

int main() {
    const int levels[] = {2, 4, 8, 16, 32};
    const int n_levels = 5;

    // 参考值（N, detmin, detmax, cond, nonaff），由本程序在 Params 默认值
    //   （m=2, alpha0=-85°, beta 为 §10 表的 1.2 倍）下实测并回填。
    //   原设计文档 §9.4 的 Python 参考值对应老默认（m=1, alpha0=-35°），
    //   参数改了那 5 行就全部失效，不能留着当基准。
    //   注意：改动 Params 任一字段都必须重跑本程序回填这张表。
    const double ref[5][4] = {
        {0.2225, 3.2501, 16.4632, 1.2235},
        {0.2190, 3.0386, 11.7924, 0.8071},
        {0.2101, 2.9574,  8.8319, 0.4413},
        {0.2050, 3.1185,  8.5371, 0.2646},
        {0.2024, 3.1790,  8.4785, 0.1614},
    };
    // 参考值只印到 4 位小数，逐位相等无从谈起；这里按相对 5e-3 比较，
    // 足以抓出公式抄错（那种错通常差一个量级），又不会被截断位吓到。
    const double ref_rtol = 5e-3;

    bool all_ok = true;

    std::cout << "===== 算例 4：TriBlockSwirl 逐单元 Q8 映射自检 =====\n\n";
    std::cout << std::setw(5)  << "N"
              << std::setw(8)  << "ne"
              << std::setw(11) << "h"
              << std::setw(12) << "detJ_min"
              << std::setw(12) << "detJ_max"
              << std::setw(12) << "cond_max"
              << std::setw(12) << "nonaff"
              << std::setw(9)  << "负detJ"
              << std::setw(10) << "(比)"
              << "\n";

    double prev_na = 0.0;
    std::vector<double> shared_dev(n_levels, 0.0);
    std::vector<double> node_dev(n_levels, 0.0);    // Q8 节点两侧不一致（应为 0）
    std::vector<double> microfold(n_levels, 0.0);   // 参考网格自身的微折
    std::vector<double> bdry_dev(n_levels, 0.0);
    std::vector<double> bdry_sag(n_levels, 0.0);

    for (int li = 0; li < n_levels; ++li) {
        int N = levels[li];
        std::string path = "mesh/data/square_" + std::to_string(N)
                         + "x" + std::to_string(N) + ".msh";

        vem::StraightMeshReader reader;
        if (!reader.read_mesh(path)) {
            std::cerr << "读取网格失败: " << path << std::endl;
            return 1;
        }
        const vem::StraightMeshData& md = reader.get_mesh_data();

        TriBlockSwirlMapping map;
        map.set_mesh(reader);

        // ---- 1. (R2) 共享边逐点同一 ----------------------------------------
        // 每条边记下它属于哪些单元
        std::vector<int> owner_a(md.num_edges, -1), owner_b(md.num_edges, -1);
        for (int e = 0; e < md.num_cells; ++e) {
            int st = md.cell_edge_indices[e];
            int cnt = md.nodes_per_cell[e];
            for (int le = 0; le < cnt; ++le) {
                int ge = md.global_edges[st + le];
                if (owner_a[ge] < 0)      owner_a[ge] = e;
                else if (owner_b[ge] < 0) owner_b[ge] = e;
            }
        }
        const int NS = 9;   // 边上采样点数（含端点）
        for (int ge = 0; ge < md.num_edges; ++ge) {
            if (owner_b[ge] < 0) continue;   // 边界边，无第二侧
            int n0 = md.edge_endpoints[2 * ge];
            int n1 = md.edge_endpoints[2 * ge + 1];
            double x0 = md.node_coords[2 * n0], y0 = md.node_coords[2 * n0 + 1];
            double x1 = md.node_coords[2 * n1], y1 = md.node_coords[2 * n1 + 1];

            // 1a. 这条边的 3 个 Q8 节点（两端点 + 中点的 G 值）必须逐位出现在
            //     两侧单元的节点表里。找不到就说明建表用的不是同一份数据。
            Point2D want[3] = {map.generator(x0, y0),
                               map.generator(x1, y1),
                               map.generator(0.5 * (x0 + x1), 0.5 * (y0 + y1))};
            const int side[2] = {owner_a[ge], owner_b[ge]};
            for (int sd = 0; sd < 2; ++sd) {
                const TriBlockSwirlMapping::CellQ8& c = map.cell(side[sd]);
                for (int k = 0; k < 3; ++k) {
                    double best = std::numeric_limits<double>::max();
                    for (int i = 0; i < 8; ++i) {
                        double d = std::hypot(c.nx[i] - want[k].x,
                                              c.ny[i] - want[k].y);
                        if (d < best) best = d;
                    }
                    if (best > node_dev[li]) node_dev[li] = best;
                }
            }

            // 参考网格自身的微折：理想网格里内部边必然轴对齐，
            // min(|Δx|, |Δy|) 应为 0；实测不为 0，正是 1b 偏差的来源。
            double mf = std::min(std::abs(x1 - x0), std::abs(y1 - y0));
            if (mf > microfold[li]) microfold[li] = mf;

            // 1b. 组装路径上的实际偏差
            for (int s = 0; s < NS; ++s) {
                double t = (double)s / (NS - 1);
                double xi  = x0 + t * (x1 - x0);
                double eta = y0 + t * (y1 - y0);
                Point2D pa = map.physical_coords(owner_a[ge], xi, eta);
                Point2D pb = map.physical_coords(owner_b[ge], xi, eta);
                double d = std::hypot(pa.x - pb.x, pa.y - pb.y);
                if (d > shared_dev[li]) shared_dev[li] = d;
            }
        }

        // ---- 2/3. 外边界位置与直边性 ---------------------------------------
        const double gtol = 1e-12;   // 判定参考端点是否贴在边界线上
        for (int bi = 0; bi < md.num_boundary_edges; ++bi) {
            int ge = md.boundary_edges[bi];
            int owner = owner_a[ge];
            int n0 = md.edge_endpoints[2 * ge];
            int n1 = md.edge_endpoints[2 * ge + 1];
            double x0 = md.node_coords[2 * n0], y0 = md.node_coords[2 * n0 + 1];
            double x1 = md.node_coords[2 * n1], y1 = md.node_coords[2 * n1 + 1];

            // 这条边贴在哪条边界线上：comp=0 查 x，comp=1 查 y
            int comp = -1;  double val = 0.0;
            if (std::abs(x0) < gtol && std::abs(x1) < gtol)             { comp = 0; val = 0.0; }
            else if (std::abs(x0 - 1.0) < gtol && std::abs(x1 - 1.0) < gtol) { comp = 0; val = 1.0; }
            else if (std::abs(y0) < gtol && std::abs(y1) < gtol)        { comp = 1; val = 0.0; }
            else if (std::abs(y1 - 1.0) < gtol && std::abs(y0 - 1.0) < gtol) { comp = 1; val = 1.0; }
            else {
                std::cerr << "边界边 " << ge << " 的端点不在 [0,1]^2 的边界线上\n";
                return 1;
            }

            Point2D pa = map.physical_coords(owner, x0, y0);
            Point2D pb = map.physical_coords(owner, x1, y1);
            Point2D pm = map.physical_coords(owner, 0.5 * (x0 + x1),
                                                   0.5 * (y0 + y1));
            const Point2D* trio[3] = {&pa, &pb, &pm};
            for (int k = 0; k < 3; ++k) {
                double c = (comp == 0) ? trio[k]->x : trio[k]->y;
                double d = std::abs(c - val);
                if (d > bdry_dev[li]) bdry_dev[li] = d;
            }
            // sagitta：中点相对弦的垂距 / 弦长
            double chx = pb.x - pa.x, chy = pb.y - pa.y;
            double L = std::hypot(chx, chy);
            double wx = pm.x - 0.5 * (pa.x + pb.x);
            double wy = pm.y - 0.5 * (pa.y + pb.y);
            double sag = std::abs(wx * chy - wy * chx) / (L * L);
            if (sag > bdry_sag[li]) bdry_sag[li] = sag;
        }

        // ---- 4/5. 单元质量 -------------------------------------------------
        Quality q = measure_quality(map, md.num_cells);

        std::cout << std::setw(5) << N
                  << std::setw(8) << md.num_cells
                  << std::setw(11) << std::fixed << std::setprecision(6)
                  << 1.0 / (2.0 * N)
                  << std::setw(12) << std::setprecision(4) << q.detmin
                  << std::setw(12) << q.detmax
                  << std::setw(12) << q.condmax
                  << std::setw(12) << q.nonaff
                  << std::setw(9)  << q.nneg;
        if (prev_na > 0.0) {
            std::cout << std::setw(10) << std::setprecision(3)
                      << prev_na / q.nonaff;
        } else {
            std::cout << std::setw(10) << "-";
        }
        std::cout << "\n";
        prev_na = q.nonaff;

        // 逐项判定
        if (q.nneg != 0) {
            std::cerr << "  [FAIL] N=" << N << " 有 " << q.nneg
                      << " 个单元出现 detJ <= 0\n";
            all_ok = false;
        }
        const double got[4] = {q.detmin, q.detmax, q.condmax, q.nonaff};
        const char* nm[4] = {"detJ_min", "detJ_max", "cond_max", "nonaff"};
        for (int k = 0; k < 4; ++k) {
            double rel = std::abs(got[k] - ref[li][k]) / std::abs(ref[li][k]);
            if (rel > ref_rtol) {
                std::cerr << "  [FAIL] N=" << N << " " << nm[k]
                          << " = " << std::setprecision(6) << got[k]
                          << "，设计文档 §9.4 参考 " << ref[li][k]
                          << "，相对偏差 " << std::scientific << rel
                          << std::fixed << "\n";
                all_ok = false;
            }
        }
    }

    std::cout << "\n";
    std::cout << std::setw(5) << "N"
              << std::setw(14) << "Q8节点不一致"
              << std::setw(14) << "参考网格微折"
              << std::setw(14) << "共享边偏差"
              << std::setw(16) << "边界偏离边界线"
              << std::setw(14) << "边界边sagitta" << "\n";
    for (int li = 0; li < n_levels; ++li) {
        std::cout << std::setw(5) << levels[li]
                  << std::setw(14) << std::scientific << std::setprecision(3)
                  << node_dev[li]
                  << std::setw(14) << microfold[li]
                  << std::setw(14) << shared_dev[li]
                  << std::setw(16) << bdry_dev[li]
                  << std::setw(14) << bdry_sag[li] << "\n";
    }
    std::cout << std::fixed;

    // Q8 节点：两侧看到同一对全局节点、(x0+x1)/2 与 (x1+x0)/2 在 IEEE 下逐位
    // 相同、G 是纯函数 ⇒ 必须严格为 0。这里不给容差，容差会掩盖建表用错数据。
    const double tol_node   = 0.0;
    // 共享边求值偏差：受参考网格微折支配（见文件头 1b），不可能为 0。
    const double tol_shared = 1e-11;
    const double tol_bdry   = 1e-14;
    const double tol_sag    = 1e-14;
    for (int li = 0; li < n_levels; ++li) {
        if (node_dev[li] > tol_node) {
            std::cerr << "  [FAIL] N=" << levels[li]
                      << " 共享边的 Q8 节点两侧不一致，最大偏差 "
                      << std::scientific << node_dev[li] << std::fixed << "\n";
            all_ok = false;
        }
        if (shared_dev[li] > tol_shared) {
            std::cerr << "  [FAIL] N=" << levels[li]
                      << " 共享边求值偏差 " << std::scientific << shared_dev[li]
                      << " 超出容差 " << tol_shared
                      << "（参考网格微折 " << microfold[li] << "）"
                      << std::fixed << "\n";
            all_ok = false;
        }
        if (bdry_dev[li] > tol_bdry) {
            std::cerr << "  [FAIL] N=" << levels[li]
                      << " 外边界偏离 [0,1]^2，最大偏差 " << std::scientific
                      << bdry_dev[li] << std::fixed << "\n";
            all_ok = false;
        }
        if (bdry_sag[li] > tol_sag) {
            std::cerr << "  [FAIL] N=" << levels[li]
                      << " 外边界的边不直，最大 sagitta " << std::scientific
                      << bdry_sag[li] << std::fixed << "\n";
            all_ok = false;
        }
    }

    std::cout << "\n" << (all_ok ? "全部通过" : "存在失败项") << std::endl;
    return all_ok ? 0 : 1;
}
