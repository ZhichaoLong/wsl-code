/**
 * @file export_triblock_mesh.cpp
 * @brief 导出算例 4（TriBlockSwirl 逐单元 Q8 映射）的几何数据，供 Python 作图核对
 *
 * 导出三样东西，分别回答三个不同的问题：
 *   1. q8_cells_N{N}.txt   —— 每个单元的边界曲线，用 physical_coords() 逐单元求值。
 *      这是**真正参与组装的那条路径**。相邻单元的曲线必须严密贴合、无缝无叠，
 *      画出来才是一张合法网格；有缝就是 (R2) 破了。
 *   2. exact_lines_N{N}.txt —— 精确生成器 G 沿参考网格线的密采样曲线。
 *      Q8 只是 G 的二次插值，两者的差就是插值误差。把 1 叠在 2 上，
 *      能看出 Q8 有没有把曲边抓住，而不是只看单元自己好不好看。
 *   3. detj_N{N}.txt       —— 每个单元的 detJ（质心处 / 单元内极值），用于上色，
 *      直观展示「一个单元一个雅可比」以及 detJ 跨界面的跳变。
 *
 * 用法: ./build/tools/export_triblock_mesh [N ...]      默认 N = 4 8
 *       square_NxN.msh 给出 2N × 2N 个单元
 */

#include "../examples/ms_problem.h"
#include "../mesh/straight_mesh.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

using namespace MaxwellStefan;

namespace {

const char* kOutDir = "data/triblock";

// 4 点 Gauss-Legendre，用于取单元内 detJ 极值（与自检测试同一套点）
const double kG4[4] = {-0.8611363115940526, -0.3399810435848563,
                        0.3399810435848563,  0.8611363115940526};

void ensure_dir() {
    mkdir("data", 0755);          // 已存在则 EEXIST，忽略
    mkdir(kOutDir, 0755);
}

bool export_level(int N) {
    std::string mesh_file = "mesh/data/square_" + std::to_string(N)
                          + "x" + std::to_string(N) + ".msh";
    vem::StraightMeshReader reader;
    if (!reader.read_mesh(mesh_file)) {
        std::cerr << "读取网格失败: " << mesh_file << std::endl;
        return false;
    }
    const vem::StraightMeshData& md = reader.get_mesh_data();

    TriBlockSwirlMapping map;
    map.set_mesh(reader);

    const std::string tag = std::to_string(N);

    // ---- 1. Q8 单元边界（组装路径）------------------------------------------
    // 每条边采 17 点。Q8 沿边是二次曲线，17 点足够画光滑，也足够暴露缝隙。
    const int NS = 17;
    {
        std::string fn = std::string(kOutDir) + "/q8_cells_N" + tag + ".txt";
        FILE* f = std::fopen(fn.c_str(), "w");
        if (!f) { std::cerr << "无法写入 " << fn << std::endl; return false; }
        std::fprintf(f, "# 每行一个单元的闭合边界折线: cell_idx x0 y0 x1 y1 ...\n");
        std::fprintf(f, "# 由 physical_coords(cell_idx, xi, eta) 逐单元求值\n");
        for (int e = 0; e < md.num_cells; ++e) {
            int st = md.cell_node_indices[e];
            int nv = md.nodes_per_cell[e];
            std::fprintf(f, "%d", e);
            for (int lv = 0; lv < nv; ++lv) {
                int a = md.cell_nodes[st + lv];
                int b = md.cell_nodes[st + (lv + 1) % nv];
                double xa = md.node_coords[2 * a], ya = md.node_coords[2 * a + 1];
                double xb = md.node_coords[2 * b], yb = md.node_coords[2 * b + 1];
                // 末点留给下一条边的首点，避免重复
                for (int s = 0; s < NS - 1; ++s) {
                    double t = (double)s / (NS - 1);
                    Point2D p = map.physical_coords(e, xa + t * (xb - xa),
                                                       ya + t * (yb - ya));
                    std::fprintf(f, " %.17g %.17g", p.x, p.y);
                }
            }
            std::fprintf(f, "\n");
        }
        std::fclose(f);
    }

    // ---- 2. 精确生成器沿参考网格线 -----------------------------------------
    // n = 2N 个单元 ⇒ 网格线 ξ = i/n, η = j/n，各密采 401 点
    {
        int n = 2 * N;
        const int ND = 401;
        std::string fn = std::string(kOutDir) + "/exact_lines_N" + tag + ".txt";
        FILE* f = std::fopen(fn.c_str(), "w");
        if (!f) { std::cerr << "无法写入 " << fn << std::endl; return false; }
        std::fprintf(f, "# 每行一条精确曲线（生成器 G 沿参考网格线）: tag x0 y0 x1 y1 ...\n");
        for (int i = 0; i <= n; ++i) {
            double xi = (double)i / n;
            std::fprintf(f, "XI");
            for (int s = 0; s < ND; ++s) {
                double eta = (double)s / (ND - 1);
                Point2D p = map.generator(xi, eta);
                std::fprintf(f, " %.17g %.17g", p.x, p.y);
            }
            std::fprintf(f, "\n");
        }
        for (int j = 0; j <= n; ++j) {
            double eta = (double)j / n;
            std::fprintf(f, "ETA");
            for (int s = 0; s < ND; ++s) {
                double xi = (double)s / (ND - 1);
                Point2D p = map.generator(xi, eta);
                std::fprintf(f, " %.17g %.17g", p.x, p.y);
            }
            std::fprintf(f, "\n");
        }
        std::fclose(f);
    }

    // ---- 3. 逐单元 detJ ----------------------------------------------------
    {
        std::string fn = std::string(kOutDir) + "/detj_N" + tag + ".txt";
        FILE* f = std::fopen(fn.c_str(), "w");
        if (!f) { std::cerr << "无法写入 " << fn << std::endl; return false; }
        std::fprintf(f, "# cell_idx det_center det_min det_max\n");
        for (int e = 0; e < md.num_cells; ++e) {
            const TriBlockSwirlMapping::CellQ8& c = map.cell(e);
            double dc = map.jacobian_det(e, c.xi_c, c.eta_c);
            double dmin = dc, dmax = dc;
            for (int ia = 0; ia < 4; ++ia) {
                for (int ib = 0; ib < 4; ++ib) {
                    double d = map.jacobian_det(
                        e, c.xi_c + 0.5 * c.hx * kG4[ia],
                           c.eta_c + 0.5 * c.hy * kG4[ib]);
                    if (d < dmin) dmin = d;
                    if (d > dmax) dmax = d;
                }
            }
            std::fprintf(f, "%d %.17g %.17g %.17g\n", e, dc, dmin, dmax);
        }
        std::fclose(f);
    }

    std::cout << "  N=" << N << " (" << md.num_cells << " 单元) 已导出\n";
    return true;
}

}  // anonymous namespace

int main(int argc, char** argv) {
    std::vector<int> levels;
    for (int i = 1; i < argc; ++i) levels.push_back(std::atoi(argv[i]));
    if (levels.empty()) { levels.push_back(4); levels.push_back(8); }

    ensure_dir();
    std::cout << "导出 TriBlockSwirl 几何数据到 " << kOutDir << "/\n";
    for (size_t i = 0; i < levels.size(); ++i) {
        if (!export_level(levels[i])) return 1;
    }

    // 三块结构的界面曲线：竖直分界线 D（贯穿全高）与水平界面 H（只在 ξ≥a）。
    // 单独导出是为了在图上标出「哪里应该出现 detJ 跳变」，
    // 不然一张扭曲网格图看不出三块结构在哪。
    {
        vem::StraightMeshReader reader;
        if (!reader.read_mesh("mesh/data/square_4x4.msh")) return 1;
        TriBlockSwirlMapping map;
        map.set_mesh(reader);
        TriBlockSwirlMapping::Params p;   // 与映射默认参数一致

        std::string fn = std::string(kOutDir) + "/interfaces.txt";
        FILE* f = std::fopen(fn.c_str(), "w");
        if (!f) return 1;
        std::fprintf(f, "# 每行一条界面曲线: tag x0 y0 x1 y1 ...\n");
        const int ND = 801;
        std::fprintf(f, "D");
        for (int s = 0; s < ND; ++s) {
            Point2D q = map.generator(p.a, (double)s / (ND - 1));
            std::fprintf(f, " %.17g %.17g", q.x, q.y);
        }
        std::fprintf(f, "\nH");
        for (int s = 0; s < ND; ++s) {
            double xi = p.a + (1.0 - p.a) * (double)s / (ND - 1);
            Point2D q = map.generator(xi, p.c);
            std::fprintf(f, " %.17g %.17g", q.x, q.y);
        }
        std::fprintf(f, "\n");
        std::fclose(f);
        std::cout << "  界面曲线已导出\n";
    }

    std::cout << "完成\n";
    return 0;
}
