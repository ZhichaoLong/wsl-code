/**
 * @file check_ms_conservation.cpp
 * @brief 对已保存的 MS 解做守恒性 / 渐近性检查，不重新求解
 *
 * 动机：算例 3（半圆环三组分扩散）**没有制造解**，算不出收敛阶，
 * 所以"悬点网格上算得对不对"必须换判据。用三条 Maxwell-Stefan 本身
 * 必须满足的性质，它们都不需要参考解：
 *
 *   A. 组分和恒等于 1        —— MS 的代数约束 Σ_i c_i = 1，逐点成立
 *   B. 各组分总质量守恒       —— 零通量边界 + 无源项，M_i(t) = M_i(0)
 *   C. 趋于均匀稳态          —— t 足够大时 c_i → M_i(0)/|Ω|，方差 → 0
 *
 * 三个量都在**物理域**上算：积分点由计算域三角剖分生成，
 * 再乘 detJ 换到物理域，即 ∫_Ω f dx = Σ_E ∫_E f·detJ dξdη。
 *
 * 同一套判据可以套在悬点网格（.vtk）和普通四边形网格（.msh）上，
 * 两边对比即可判断悬点是否引入了额外误差。
 *
 * 用法：
 *   ./build/tools/check_ms_conservation <mesh_file> <data_dir> [problem_index] [k]
 * 例：
 *   ./build/tools/check_ms_conservation mesh/data_refined/square_8x8_refined.vtk \
 *       data/ms_data/ms_half_annulus_hanging_8x8 3 1
 */

#include "../solver/MSSolver.h"
#include "../examples/ms_problem.h"
#include "../mesh/straight_mesh.h"
#include "../mesh/polygon_triangulator.h"
#include "../lib/polynomial_basis.h"
#include "../core/gauss_quadrature.h"

#include <petsc.h>
#include <dirent.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<double> readSolutionVector(const std::string& filename, int expected_size) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("无法打开解文件: " + filename);
    int size = 0;
    file >> size;
    if (size != expected_size) {
        std::ostringstream oss;
        oss << "解向量大小不匹配: 文件 " << size << "，期望 " << expected_size;
        throw std::runtime_error(oss.str());
    }
    std::vector<double> sol(size);
    for (int i = 0; i < size; ++i) file >> sol[i];
    return sol;
}

std::vector<std::pair<double, std::string>> listSolutionFiles(const std::string& dir,
                                                              int num_cells) {
    std::vector<std::pair<double, std::string>> result;
    std::string prefix = std::to_string(num_cells) + "_";
    DIR* dp = opendir(dir.c_str());
    if (!dp) throw std::runtime_error("无法打开目录: " + dir);
    struct dirent* entry;
    while ((entry = readdir(dp)) != NULL) {
        std::string name = entry->d_name;
        if (name.find(prefix) != 0) continue;
        if (name.size() < 4 || name.rfind(".txt") != name.size() - 4) continue;
        std::string time_str = name.substr(prefix.size(), name.size() - prefix.size() - 4);
        try {
            result.push_back(std::make_pair(std::stod(time_str), dir + "/" + name));
        } catch (...) { continue; }
    }
    closedir(dp);
    std::sort(result.begin(), result.end());
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    PetscErrorCode err = PetscInitialize(&argc, &argv, NULL, NULL);
    if (err != 0) return static_cast<int>(err);

    try {
        if (argc < 3) {
            std::cerr << "用法: " << argv[0]
                      << " <mesh_file(.vtk|.msh)> <data_dir> [problem_index=3] [k=1]\n";
            PetscFinalize();
            return 1;
        }
        const std::string mesh_file = argv[1];
        const std::string data_dir = argv[2];
        const int problem_index = (argc > 3) ? std::atoi(argv[3]) : 3;
        const int k = (argc > 4) ? std::atoi(argv[4]) : 1;
        const int n_quad_tri = 12;   // 三角形积分点数，取够高以免求积误差混进判据

        // ========== 读网格：按扩展名自动选 reader ==========
        vem::StraightMeshReader reader;
        bool is_vtk = (mesh_file.size() > 4 &&
                       mesh_file.compare(mesh_file.size() - 4, 4, ".vtk") == 0);
        bool ok = is_vtk ? reader.read_mesh_vtk(mesh_file) : reader.read_mesh(mesh_file);
        if (!ok) throw std::runtime_error("无法读取网格: " + mesh_file);
        const vem::StraightMeshData& mesh = reader.get_mesh_data();

        int n_penta = 0;
        for (int e = 0; e < mesh.num_cells; ++e)
            if (mesh.nodes_per_cell[e] != 4) ++n_penta;

        // ========== 建算例与求解器（只为拿到自由度布局和映射，不求解） ==========
        MaxwellStefan::MSProblem problem;
        problem.set_problem_index(problem_index);
        if (!problem.init_problem()) throw std::runtime_error("无法初始化 MS 算例");

        MSSolver solver(reader, problem, 0.001, 0.001, 50, 1e-6, 9, 9, k, false, "tmp");
        if (!solver.initialize()) throw std::runtime_error("求解器初始化失败");

        const int n_comp = solver.getNumComponents();
        const int dof_conc_elem = solver.getDofPerElementConc();
        const int total_conc_per_comp = solver.getTotalDofConc();
        const int total_flux_all = solver.getTotalDofFluxAll();
        const int total_dof_all = solver.getTotalDofAll();

        const MaxwellStefan::IsoparametricMapping& mapping = problem.get_mapping();

        std::cout << "\n================================================================\n"
                  << " MS 守恒性检查   算例 " << problem_index << ",  k = " << k << "\n"
                  << " 网格 : " << mesh_file << "\n"
                  << "        " << mesh.num_cells << " 单元（其中 " << n_penta
                  << " 个非四边形）, " << mesh.num_nodes << " 节点\n"
                  << " 数据 : " << data_dir << "\n"
                  << "================================================================\n";

        // ========== 预计算积分点：物理域权重 w·detJ 与单项式值 ==========
        // 这些量只依赖网格与映射，与时刻无关，算一次反复用。
        vem::basis::BasisFunctionPloy basis(k);
        vem::PolygonTriangulator triangulator;
        vem::GaussQuadrature quadrature;

        struct QP {
            int elem;
            double w_phys;                 // 物理域积分权重 = w_comp * detJ
            std::vector<double> mvals;     // 单项式在该点的值
        };
        std::vector<QP> qps;
        double area_phys = 0.0;

        for (int elem = 0; elem < mesh.num_cells; ++elem) {
            const vem::basis::Point2D center(mesh.cell_centroid_x[elem],
                                             mesh.cell_centroid_y[elem]);
            const double diameter = mesh.cell_diameter[elem];
            vem::TriangulatedElement te = triangulator.triangulateElement(mesh, elem);

            for (size_t t = 0; t < te.triangles.size(); ++t) {
                std::vector<vem::QuadPoint> pts =
                    quadrature.get_triangle_points(te.triangles[t].vertices, n_quad_tri);
                for (size_t p = 0; p < pts.size(); ++p) {
                    const double xi = pts[p].x[0];
                    const double eta = pts[p].x[1];
                    const double detJ = mapping.jacobian_det(elem, xi, eta);

                    QP q;
                    q.elem = elem;
                    q.w_phys = pts[p].w * detJ;
                    q.mvals.resize(dof_conc_elem);
                    const vem::basis::Point2D cp(xi, eta);
                    for (int m = 0; m < dof_conc_elem; ++m)
                        q.mvals[m] = basis.evalMonomial2D(m, center, diameter, cp);
                    area_phys += q.w_phys;
                    qps.push_back(q);
                }
            }
        }

        // 半圆环解析面积 = π/2 * (R² - r²) = π/2 * (1.5² - 0.5²) = π
        std::cout << "\n 物理域面积（数值积分）= " << std::setprecision(12) << area_phys;
        if (problem_index == 3)
            std::cout << "   解析值 π = " << M_PI
                      << "   相对偏差 " << std::scientific << std::setprecision(3)
                      << std::fabs(area_phys - M_PI) / M_PI;
        std::cout << std::defaultfloat << "\n";

        // ========== 逐时刻计算 ==========
        std::vector<std::pair<double, std::string>> files =
            listSolutionFiles(data_dir, mesh.num_cells);
        if (files.empty()) throw std::runtime_error("目录中没有匹配的解文件: " + data_dir);

        std::cout << "\n 共 " << files.size() << " 个时刻\n\n";
        std::cout << std::left << std::setw(10) << "  t"
                  << std::setw(15) << "M_0" << std::setw(15) << "M_1"
                  << std::setw(15) << "M_2"
                  << std::setw(13) << "max|Σc-1|"
                  << std::setw(12) << "min c_i"
                  << std::setw(12) << "非均匀度" << "\n";
        std::cout << std::string(92, '-') << "\n";

        std::vector<double> M0_ref;
        double max_sum_dev_all = 0.0, min_c_all = 1e300, max_mass_drift = 0.0;

        for (size_t fi = 0; fi < files.size(); ++fi) {
            std::vector<double> sol = readSolutionVector(files[fi].second, total_dof_all);

            std::vector<double> M(n_comp, 0.0);
            std::vector<double> M2(n_comp, 0.0);   // ∫c² ，用于算方差
            double max_sum_dev = 0.0, min_c = 1e300;

            for (size_t q = 0; q < qps.size(); ++q) {
                const QP& qp = qps[q];
                double csum = 0.0;
                for (int c = 0; c < n_comp; ++c) {
                    double val = 0.0;
                    const int base = total_flux_all + c * total_conc_per_comp
                                   + qp.elem * dof_conc_elem;
                    for (int m = 0; m < dof_conc_elem; ++m)
                        val += sol[base + m] * qp.mvals[m];
                    M[c]  += qp.w_phys * val;
                    M2[c] += qp.w_phys * val * val;
                    csum += val;
                    min_c = std::min(min_c, val);
                }
                max_sum_dev = std::max(max_sum_dev, std::fabs(csum - 1.0));
            }

            if (fi == 0) M0_ref = M;

            // 非均匀度：所有组分中 sqrt(∫(c-c̄)²/|Ω|) 的最大值，稳态时应 → 0
            double nonuniform = 0.0;
            for (int c = 0; c < n_comp; ++c) {
                double mean = M[c] / area_phys;
                double var = M2[c] / area_phys - mean * mean;
                nonuniform = std::max(nonuniform, std::sqrt(std::max(var, 0.0)));
            }

            for (int c = 0; c < n_comp; ++c)
                max_mass_drift = std::max(max_mass_drift, std::fabs(M[c] - M0_ref[c]));
            max_sum_dev_all = std::max(max_sum_dev_all, max_sum_dev);
            min_c_all = std::min(min_c_all, min_c);

            // 只打印首、末与等间隔若干行，避免刷屏
            bool show = (fi == 0) || (fi + 1 == files.size()) ||
                        (files.size() <= 20) || (fi % (files.size() / 10) == 0);
            if (show) {
                std::cout << std::left << "  " << std::setw(8) << std::fixed
                          << std::setprecision(4) << files[fi].first;
                std::cout << std::setprecision(8);
                for (int c = 0; c < n_comp; ++c)
                    std::cout << std::setw(15) << M[c];
                std::cout << std::scientific << std::setprecision(2)
                          << std::setw(13) << max_sum_dev
                          << std::setw(12) << min_c
                          << std::setw(12) << nonuniform << "\n";
                std::cout << std::defaultfloat;
            }
        }

        // ========== 汇总 ==========
        std::cout << std::string(92, '-') << "\n\n";
        std::cout << std::scientific << std::setprecision(4);
        std::cout << " [A] 组分和约束    max|Σc_i - 1|  = " << max_sum_dev_all << "\n";
        std::cout << " [B] 质量守恒      max|M_i(t)-M_i(0)| = " << max_mass_drift
                  << "   （相对 " << max_mass_drift / std::max(area_phys, 1e-30) << "）\n";
        std::cout << " [C] 全程最小浓度  min c_i        = " << min_c_all
                  << (min_c_all < -1e-6 ? "   ← 出现明显负值" : "") << "\n";
        if (problem_index == 3) {
            std::cout << "\n 稳态理论值（M_i(0)/|Ω|）: ";
            for (size_t c = 0; c < M0_ref.size(); ++c)
                std::cout << std::fixed << std::setprecision(6)
                          << M0_ref[c] / area_phys << "  ";
            std::cout << "\n";
        }
        std::cout << "\n";

    } catch (const std::exception& e) {
        std::cerr << "\n 错误: " << e.what() << "\n\n";
        PetscFinalize();
        return 1;
    }

    PetscFinalize();
    return 0;
}
