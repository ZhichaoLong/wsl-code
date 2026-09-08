/**
 * @file triblock_sweep.cpp
 * @brief 扫 TriBlockSwirl 的扭曲参数，量出「还能再夸张多少」的硬边界
 *
 * 想回答的问题：把扭曲程度往上推，什么时候网格就废了？
 *
 * 判废的标准只有一条硬的：**detJ 处处 > 0**。
 * 不是为了「可微」，而是 detJ <= 0 意味着这个单元被翻了面——四边形映过去
 * 自交、相邻单元互相叠上去，那已经不是一张合法网格了，跟求不求导无关。
 * 而且 MSSolver.cpp 里 14 处直接用了 jacobian()/jacobian_det()（Piola 变换、
 * 面积元），detJ 变号会静默污染整张误差表，不会报错。
 *
 * 所以本工具对每组参数量两层：
 *   1. 生成器 G 本身的 detJ（中心差分，全局 201×201）。这一层与网格无关，
 *      是「这个曲边域自己合不合法」——G 一旦不单射，任何网格都救不回来。
 *   2. 每个单元 Q8 映射的 detJ / 条件数 / 非仿射度（N=4,8,16）。
 *      采样用 9×9 均匀点（含边界），比 4 点 Gauss 更容易抓住靠角点的折叠；
 *      §9.4 参考表用的是 4 点 Gauss，两者数值不会完全一致，这是有意的。
 *
 * 另外把每组参数在 N=8 的单元边界折线导出来，供 plot_triblock_sweep.py 画图，
 * 光看数字看不出「夸张」长什么样。
 *
 * 用法: ./build/tools/triblock_sweep [tag ...]     不给 tag 则跑全部
 */

#include "../examples/ms_problem.h"
#include "../mesh/straight_mesh.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

using namespace MaxwellStefan;

namespace {

const char* kOutDir = "data/triblock/sweep";

void ensure_dir() {
    mkdir("data", 0755);
    mkdir("data/triblock", 0755);
    mkdir(kOutDir, 0755);
}

double deg(double d) { return d * M_PI / 180.0; }

// ---------------------------------------------------------------------------
// 候选参数组。分两类：
//   单旋钮组（alpha / grade / shear / bulge）——为了看清哪个旋钮最有效；
//   组合梯度组（L1..L4）——为了找出综合起来的极限。
// L4 是故意推过头的，预期 detJ 变号；留着它是为了把边界标出来，
// 不是候选方案。
// ---------------------------------------------------------------------------
struct Cand {
    const char* tag;
    const char* note;
    TriBlockSwirlMapping::Params p;
};

// 统一缩放全部分级系数：分级只改单元疏密，g' 恒 > 0，单独加不会折叠，
// 只会把 detJ_max/detJ_min 拉开。
TriBlockSwirlMapping::Params scale_grade(TriBlockSwirlMapping::Params p,
                                         double k) {
    p.bv *= k; p.b1 *= k; p.bb3 *= k; p.bt2 *= k;
    p.br3 *= k; p.br2 *= k; p.bh *= k;
    return p;
}

std::vector<Cand> build_cands() {
    std::vector<Cand> v;
    TriBlockSwirlMapping::Params d;   // 现行默认（§10 参数表）

    {   Cand c; c.tag = "L0";    c.note = "现行默认";           c.p = d; v.push_back(c); }

    // --- 单旋钮：中心旋转角 ---
    for (int i = 0; i < 4; ++i) {
        const double as[4] = {-55.0, -75.0, -95.0, -115.0};
        const char* tg[4] = {"a55", "a75", "a95", "a115"};
        Cand c; c.tag = tg[i]; c.note = "只改 alpha0";
        c.p = d; c.p.alpha0 = deg(as[i]);
        v.push_back(c);
    }

    // --- 单旋钮：分级强度（单元疏密对比） ---
    for (int i = 0; i < 3; ++i) {
        const double ks[3] = {2.0, 3.0, 4.0};
        const char* tg[3] = {"g2", "g3", "g4"};
        Cand c; c.tag = tg[i]; c.note = "只把全部 beta 放大";
        c.p = scale_grade(d, ks[i]);
        v.push_back(c);
    }

    // --- 单旋钮：竖直分界线倾斜量 s_t - s_b ---
    {   Cand c; c.tag = "s50"; c.note = "只改 s_b/s_t";
        c.p = d; c.p.s_b = 0.22; c.p.s_t = 0.72; v.push_back(c); }
    {   Cand c; c.tag = "s70"; c.note = "只改 s_b/s_t";
        c.p = d; c.p.s_b = 0.12; c.p.s_t = 0.82; v.push_back(c); }

    // --- 单旋钮：水平界面鼓包 + 右端高度 ---
    {   Cand c; c.tag = "h_mid"; c.note = "只改 A_H/y_r";
        c.p = d; c.p.A_H = 0.22; c.p.y_r = 0.78; v.push_back(c); }
    {   Cand c; c.tag = "h_big"; c.note = "只改 A_H/y_r";
        c.p = d; c.p.A_H = 0.32; c.p.y_r = 0.88; v.push_back(c); }

    // --- 组合梯度 ---
    {   Cand c; c.tag = "L1"; c.note = "组合：中度";
        c.p = scale_grade(d, 1.5);
        c.p.alpha0 = deg(-55.0);
        c.p.s_b = 0.28; c.p.s_t = 0.72;
        c.p.A_H = 0.18; c.p.y_r = 0.72;
        v.push_back(c); }
    {   Cand c; c.tag = "L2"; c.note = "组合：重度";
        c.p = scale_grade(d, 2.0);
        c.p.alpha0 = deg(-75.0);
        c.p.s_b = 0.22; c.p.s_t = 0.78;
        c.p.A_H = 0.24; c.p.y_r = 0.80;
        v.push_back(c); }
    {   Cand c; c.tag = "L3"; c.note = "组合：极限候选";
        c.p = scale_grade(d, 2.5);
        c.p.alpha0 = deg(-95.0);
        c.p.s_b = 0.16; c.p.s_t = 0.84;
        c.p.A_H = 0.30; c.p.y_r = 0.86;
        v.push_back(c); }
    {   Cand c; c.tag = "L4"; c.note = "组合：故意推过头";
        c.p = scale_grade(d, 3.0);
        c.p.alpha0 = deg(-125.0);
        c.p.s_b = 0.10; c.p.s_t = 0.90;
        c.p.A_H = 0.38; c.p.y_r = 0.92;
        v.push_back(c); }

    return v;
}

// ---------------------------------------------------------------------------
// 第 1 层：生成器 G 自己的 detJ（中心差分）
// 与网格无关。G 的 detJ 变号 ⇒ 曲边域自己就折叠了。
// 差分跨块界面时给的是两侧的混合值——这里只当极小值探测器用，够了。
// ---------------------------------------------------------------------------
struct GenQ { double detmin; double detmax; int nneg; };

GenQ gen_quality(const TriBlockSwirlMapping& m, int ng) {
    GenQ q;
    q.detmin = std::numeric_limits<double>::max();
    q.detmax = -std::numeric_limits<double>::max();
    q.nneg = 0;
    const double h = 1e-5;
    for (int i = 0; i <= ng; ++i) {
        double xi = h + (1.0 - 2.0 * h) * (double)i / ng;
        for (int j = 0; j <= ng; ++j) {
            double eta = h + (1.0 - 2.0 * h) * (double)j / ng;
            Point2D px = m.generator(xi + h, eta);
            Point2D mx = m.generator(xi - h, eta);
            Point2D py = m.generator(xi, eta + h);
            Point2D my = m.generator(xi, eta - h);
            double xxi = (px.x - mx.x) / (2 * h), yxi = (px.y - mx.y) / (2 * h);
            double xet = (py.x - my.x) / (2 * h), yet = (py.y - my.y) / (2 * h);
            double det = xxi * yet - xet * yxi;
            if (det < q.detmin) q.detmin = det;
            if (det > q.detmax) q.detmax = det;
            if (det <= 0.0) ++q.nneg;
        }
    }
    return q;
}

// ---------------------------------------------------------------------------
// 第 2 层：每单元 Q8 映射的质量
// ---------------------------------------------------------------------------
double cond2(const Tensor2D& J) {
    double a = J.xx, b = J.xy, c = J.yx, dd = J.yy;
    double t = a * a + b * b + c * c + dd * dd;
    double det = a * dd - b * c;
    double s = std::sqrt(std::max(0.0, t * t - 4.0 * det * det));
    double l1 = 0.5 * (t + s), l2 = 0.5 * (t - s);
    if (l2 <= 0.0) return std::numeric_limits<double>::infinity();
    return std::sqrt(l1 / l2);
}

struct CellQ {
    double detmin, detmax, condmax, nonaff;
    int nbad;      // detJ <= 0 的单元数
};

CellQ cell_quality(const TriBlockSwirlMapping& m, int nc, int ns) {
    CellQ q;
    q.detmin = std::numeric_limits<double>::max();
    q.detmax = -std::numeric_limits<double>::max();
    q.condmax = 0.0;
    q.nonaff = 0.0;
    q.nbad = 0;

    std::vector<Tensor2D> Js;
    for (int e = 0; e < nc; ++e) {
        const TriBlockSwirlMapping::CellQ8& c = m.cell(e);
        Js.clear();
        double sxx = 0, sxy = 0, syx = 0, syy = 0;
        bool bad = false;
        for (int ia = 0; ia <= ns; ++ia) {
            double a = -1.0 + 2.0 * (double)ia / ns;
            for (int ib = 0; ib <= ns; ++ib) {
                double b = -1.0 + 2.0 * (double)ib / ns;
                double xi  = c.xi_c  + 0.5 * c.hx * a;
                double eta = c.eta_c + 0.5 * c.hy * b;
                Tensor2D J = m.jacobian(e, xi, eta);
                Js.push_back(J);
                double det = J.xx * J.yy - J.xy * J.yx;
                if (det < q.detmin) q.detmin = det;
                if (det > q.detmax) q.detmax = det;
                if (det <= 0.0) bad = true;
                double cd = cond2(J);
                if (cd > q.condmax) q.condmax = cd;
                sxx += J.xx; sxy += J.xy; syx += J.yx; syy += J.yy;
            }
        }
        if (bad) ++q.nbad;

        double n = (double)Js.size();
        Tensor2D Jb(sxx / n, sxy / n, syx / n, syy / n);
        double nb = std::sqrt(Jb.xx * Jb.xx + Jb.xy * Jb.xy
                            + Jb.yx * Jb.yx + Jb.yy * Jb.yy);
        double loc = 0.0;
        for (size_t k = 0; k < Js.size(); ++k) {
            double dxx = Js[k].xx - Jb.xx, dxy = Js[k].xy - Jb.xy;
            double dyx = Js[k].yx - Jb.yx, dyy = Js[k].yy - Jb.yy;
            double nd = std::sqrt(dxx * dxx + dxy * dxy + dyx * dyx + dyy * dyy);
            if (nd > loc) loc = nd;
        }
        if (nb > 0.0 && loc / nb > q.nonaff) q.nonaff = loc / nb;
    }
    return q;
}

// ---------------------------------------------------------------------------
// 共享边重合性：直接量相邻单元共享边上的采样点距离。
// 改参数不该破 (R2)，量一下确认。这里不重算映射，只调 physical_coords。
// 简化做法：用「同一条参考边两侧各自求值」——遍历所有单元的 4 条参考边，
// 按参考端点做键配对，比较物理采样点。
// ---------------------------------------------------------------------------
struct EdgeRec {
    long key;                 // 量化后的参考边键
    std::vector<Point2D> pts;
};

long qkey(double v) { return (long)std::floor(v * 1e9 + 0.5); }

double shared_edge_dev(const TriBlockSwirlMapping& m, int nc) {
    const int NP = 9;
    std::vector<EdgeRec> recs;
    for (int e = 0; e < nc; ++e) {
        const TriBlockSwirlMapping::CellQ8& c = m.cell(e);
        double x0 = c.xi_c - 0.5 * c.hx, x1 = c.xi_c + 0.5 * c.hx;
        double y0 = c.eta_c - 0.5 * c.hy, y1 = c.eta_c + 0.5 * c.hy;
        // 4 条参考边：下、右、上、左，统一按 (小端 -> 大端) 定向，
        // 这样配对时不必再考虑方向
        for (int s = 0; s < 4; ++s) {
            double ax, ay, bx, by;
            if (s == 0) { ax = x0; ay = y0; bx = x1; by = y0; }
            else if (s == 1) { ax = x1; ay = y0; bx = x1; by = y1; }
            else if (s == 2) { ax = x0; ay = y1; bx = x1; by = y1; }
            else { ax = x0; ay = y0; bx = x0; by = y1; }
            EdgeRec r;
            r.key = ((qkey(ax) * 1000003L + qkey(ay)) * 1000003L
                     + qkey(bx)) * 1000003L + qkey(by);
            for (int k = 0; k < NP; ++k) {
                double t = (double)k / (NP - 1);
                r.pts.push_back(m.physical_coords(e, ax + t * (bx - ax),
                                                  ay + t * (by - ay)));
            }
            recs.push_back(r);
        }
    }
    double worst = 0.0;
    for (size_t i = 0; i < recs.size(); ++i) {
        for (size_t j = i + 1; j < recs.size(); ++j) {
            if (recs[i].key != recs[j].key) continue;
            for (int k = 0; k < NP; ++k) {
                double dx = recs[i].pts[k].x - recs[j].pts[k].x;
                double dy = recs[i].pts[k].y - recs[j].pts[k].y;
                double d = std::sqrt(dx * dx + dy * dy);
                if (d > worst) worst = d;
            }
        }
    }
    return worst;
}

// 导出单元边界折线，格式与 q8_cells_N*.txt 一致，方便复用画图脚本
bool export_cells(const TriBlockSwirlMapping& m, int nc, const std::string& fn) {
    FILE* f = std::fopen(fn.c_str(), "w");
    if (!f) return false;
    std::fprintf(f, "# 每行一个单元的闭合边界折线: cell_idx x0 y0 x1 y1 ...\n");
    std::fprintf(f, "# 来自 physical_coords()，即真正参与组装的那条路径\n");
    const int NPE = 17;
    for (int e = 0; e < nc; ++e) {
        const TriBlockSwirlMapping::CellQ8& c = m.cell(e);
        double x0 = c.xi_c - 0.5 * c.hx, x1 = c.xi_c + 0.5 * c.hx;
        double y0 = c.eta_c - 0.5 * c.hy, y1 = c.eta_c + 0.5 * c.hy;
        const double cx[4] = {x0, x1, x1, x0};
        const double cy[4] = {y0, y0, y1, y1};
        std::fprintf(f, "%d", e);
        for (int s = 0; s < 4; ++s) {
            int s2 = (s + 1) % 4;
            for (int k = 0; k < NPE - 1; ++k) {   // 去掉重复端点
                double t = (double)k / (NPE - 1);
                Point2D q = m.physical_coords(e, cx[s] + t * (cx[s2] - cx[s]),
                                              cy[s] + t * (cy[s2] - cy[s]));
                std::fprintf(f, " %.17g %.17g", q.x, q.y);
            }
        }
        std::fprintf(f, "\n");
    }
    std::fclose(f);
    return true;
}

}  // namespace

// -p 模式：直接从命令行拼一组参数，省得为了细扫一个旋钮反复改代码重编。
//   ./triblock_sweep -p mytag alpha0=-45 gscale=2 s_b=0.12 A_H=0.3 m=2
//   alpha0 以**度**为单位；gscale 是全部 beta 的统一放大倍数。
bool parse_cli(int argc, char** argv, std::vector<Cand>& out,
               std::vector<std::string>& tag_store) {
    TriBlockSwirlMapping::Params p;
    double gscale = 1.0;
    std::string tag = "cli";
    for (int i = 2; i < argc; ++i) {
        std::string s = argv[i];
        size_t eq = s.find('=');
        if (eq == std::string::npos) { tag = s; continue; }
        std::string k = s.substr(0, eq);
        double v = std::atof(s.c_str() + eq + 1);
        if      (k == "alpha0") p.alpha0 = deg(v);
        else if (k == "gscale") gscale = v;
        else if (k == "a")   p.a   = v;
        else if (k == "c")   p.c   = v;
        else if (k == "s_b") p.s_b = v;
        else if (k == "s_t") p.s_t = v;
        else if (k == "bv")  p.bv  = v;
        else if (k == "b1")  p.b1  = v;
        else if (k == "bb3") p.bb3 = v;
        else if (k == "bt2") p.bt2 = v;
        else if (k == "br3") p.br3 = v;
        else if (k == "br2") p.br2 = v;
        else if (k == "bh")  p.bh  = v;
        else if (k == "y_r") p.y_r = v;
        else if (k == "A_H") p.A_H = v;
        else if (k == "m")   p.m   = (int)v;
        else if (k == "cx")  p.cx  = v;
        else if (k == "cy")  p.cy  = v;
        else { std::cerr << "未知参数: " << k << std::endl; return false; }
    }
    if (gscale != 1.0) p = scale_grade(p, gscale);
    tag_store.push_back(tag);
    Cand c;
    c.tag = tag_store.back().c_str();
    c.note = "-p";
    c.p = p;
    out.push_back(c);
    return true;
}

int main(int argc, char** argv) {
    std::vector<std::string> want;
    std::vector<Cand> cands;
    std::vector<std::string> tag_store;
    tag_store.reserve(64);   // 防止 c_str() 因扩容失效

    if (argc >= 2 && std::string(argv[1]) == "-p") {
        if (!parse_cli(argc, argv, cands, tag_store)) return 1;
    } else {
        for (int i = 1; i < argc; ++i) want.push_back(argv[i]);
        cands = build_cands();
    }

    ensure_dir();
    const int levels[4] = {2, 4, 8, 16};   // N=2 必须在内：q8 测试查它，且它是最容易翻面的一层

    // 网格只读一次，三个层各一份
    vem::StraightMeshReader readers[4];
    for (int i = 0; i < 4; ++i) {
        std::string fn = "mesh/data/square_" + std::to_string(levels[i])
                       + "x" + std::to_string(levels[i]) + ".msh";
        if (!readers[i].read_mesh(fn)) {
            std::cerr << "读取网格失败: " << fn << std::endl;
            return 1;
        }
    }

    std::cout << "\nTriBlockSwirl 扭曲参数扫描\n"
              << "判废标准：detJ <= 0（单元翻面 ⇒ 不是合法网格）\n"
              << "G 层 = 生成器自身 detJ（201x201 中心差分，与网格无关）\n"
              << "Q8 层 = 逐单元 Q8 detJ / 条件数 / 非仿射度（9x9 均匀采样，含边界）\n\n";

    std::cout << std::left << std::setw(8) << "tag"
              << std::setw(20) << "改动"
              << std::setw(9) << "alpha0"
              << std::right
              << std::setw(11) << "G_detmin"
              << std::setw(8) << "G_neg"
              << std::setw(4) << "N"
              << std::setw(11) << "detmin"
              << std::setw(10) << "detmax"
              << std::setw(9) << "det比"
              << std::setw(10) << "cond"
              << std::setw(9) << "nonaff"
              << std::setw(7) << "坏元"
              << std::setw(11) << "共享边"
              << "\n";
    std::cout << std::string(136, '-') << "\n";

    for (size_t ci = 0; ci < cands.size(); ++ci) {
        const Cand& cd = cands[ci];
        if (!want.empty()) {
            bool hit = false;
            for (size_t k = 0; k < want.size(); ++k)
                if (want[k] == cd.tag) hit = true;
            if (!hit) continue;
        }

        // G 层：与网格无关，任取一个 mapping 实例即可（不必 set_mesh，
        // generator() 不依赖节点表）
        TriBlockSwirlMapping g(cd.p);
        GenQ gq = gen_quality(g, 200);

        for (int li = 0; li < 4; ++li) {
            TriBlockSwirlMapping m(cd.p);
            m.set_mesh(readers[li]);
            int nc = m.num_cells();
            CellQ q = cell_quality(m, nc, 8);
            double dev = (levels[li] == 4) ? shared_edge_dev(m, nc) : -1.0;

            std::cout << std::left
                      << std::setw(8) << (li == 0 ? cd.tag : "")
                      << std::setw(20) << (li == 0 ? cd.note : "")
                      << std::setw(9);
            if (li == 0) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.0f",
                              cd.p.alpha0 * 180.0 / M_PI);
                std::cout << buf;
            } else {
                std::cout << "";
            }
            std::cout << std::right << std::scientific << std::setprecision(3);
            if (li == 0) std::cout << std::setw(11) << gq.detmin
                                   << std::setw(8) << gq.nneg;
            else         std::cout << std::setw(11) << "" << std::setw(8) << "";
            std::cout << std::setw(4) << levels[li]
                      << std::setw(11) << q.detmin
                      << std::setw(10) << q.detmax;
            std::cout << std::fixed << std::setprecision(1)
                      << std::setw(9) << (q.detmin > 0 ? q.detmax / q.detmin : -1.0)
                      << std::setw(10) << q.condmax
                      << std::setprecision(3) << std::setw(9) << q.nonaff
                      << std::setw(7) << q.nbad;
            if (dev >= 0.0) std::cout << std::scientific << std::setprecision(2)
                                      << std::setw(11) << dev;
            else            std::cout << std::setw(11) << "";
            std::cout << "\n";

            if (levels[li] == 8) {
                std::string fn = std::string(kOutDir) + "/cells_"
                               + cd.tag + "_N8.txt";
                if (!export_cells(m, nc, fn)) {
                    std::cerr << "写文件失败: " << fn << std::endl;
                    return 1;
                }
            }
        }
        std::cout << std::string(136, '-') << "\n";
    }

    std::cout << "\n单元边界折线已导出到 " << kOutDir << "/cells_<tag>_N8.txt\n";
    return 0;
}
