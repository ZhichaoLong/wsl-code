/**
 * 批量导出 MS 浓度场绘图数据（曲边物理域）
 * 读取 data/ms_data/solver_data/ 下所有时刻的解向量，
 * 在每个单元内采样浓度，通过曲边映射得到物理域坐标。
 *
 * 输出：
 *   - triangles.csv          三角拓扑（所有时刻共享）
 *   - curved_edges.csv       曲边网格线（所有时刻共享）
 *   - concentration_tXXX.XXXXX.csv   每个时刻的点坐标 + 3组分浓度
 *
 * 另外也导出初值时刻 t=0 的浓度（用初值函数直接计算）。
 */

#include "../solver/MSSolver.h"

#include <petsc.h>

#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int pointIndex(const std::map<std::pair<int, int>, int>& indices,
               int i, int j) {
    auto found = indices.find(std::make_pair(i, j));
    if (found == indices.end()) {
        throw std::runtime_error("细分三角形点编号不存在");
    }
    return found->second;
}

/**
 * 从 txt 文件读取解向量
 * 格式：第一行总自由度数，随后每行一个值
 */
std::vector<double> readSolutionVector(const std::string& filename,
                                        int expected_size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开解文件: " + filename);
    }
    int size = 0;
    file >> size;
    if (size != expected_size) {
        std::ostringstream oss;
        oss << "解向量大小不匹配: 文件 " << size
            << "，期望 " << expected_size;
        throw std::runtime_error(oss.str());
    }
    std::vector<double> sol(size);
    for (int i = 0; i < size; ++i) {
        file >> sol[i];
    }
    return sol;
}

// 列出目录中所有匹配 "{num_cells}_*.txt" 的文件，返回按时刻排序的路径列表
std::vector<std::pair<double, std::string>> listSolutionFiles(
    const std::string& dir, int num_cells) {
    std::vector<std::pair<double, std::string>> result;
    std::string prefix = std::to_string(num_cells) + "_";
    DIR* dp = opendir(dir.c_str());
    if (!dp) {
        throw std::runtime_error("无法打开目录: " + dir);
    }
    struct dirent* entry;
    while ((entry = readdir(dp)) != NULL) {
        std::string name = entry->d_name;
        if (name.find(prefix) != 0) continue;
        if (name.rfind(".txt") != name.size() - 4) continue;
        // 提取时刻值
        std::string time_str = name.substr(prefix.size(),
                                           name.size() - prefix.size() - 4);
        try {
            double t = std::stod(time_str);
            result.push_back(std::make_pair(t, dir + "/" + name));
        } catch (...) {
            continue;
        }
    }
    closedir(dp);
    std::sort(result.begin(), result.end());
    return result;
}

void writeCurvedEdges(std::ofstream& output,
                      const vem::StraightMeshData& mesh,
                      const MaxwellStefan::IsoparametricMapping& mapping,
                      int samples_per_edge) {
    output << "edge,point,x,y\n";
    output << std::setprecision(17);
    for (int edge = 0; edge < mesh.num_edges; ++edge) {
        const int node0 = mesh.edge_endpoints[2 * edge];
        const int node1 = mesh.edge_endpoints[2 * edge + 1];
        const double xi0 = mesh.node_coords[2 * node0];
        const double eta0 = mesh.node_coords[2 * node0 + 1];
        const double xi1 = mesh.node_coords[2 * node1];
        const double eta1 = mesh.node_coords[2 * node1 + 1];
        for (int point = 0; point <= samples_per_edge; ++point) {
            const double t = static_cast<double>(point) / samples_per_edge;
            const double xi = (1.0 - t) * xi0 + t * xi1;
            const double eta = (1.0 - t) * eta0 + t * eta1;
            const MaxwellStefan::Point2D physical =
                mapping.physical_coords(xi, eta);
            output << edge << ',' << point << ','
                   << physical.x << ',' << physical.y << '\n';
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    PetscErrorCode error = PetscInitialize(&argc, &argv, NULL, NULL);
    if (error != 0) {
        return static_cast<int>(error);
    }

    int status = 0;
    try {
        // 参数
        const std::string mesh_file = argc >= 2
            ? argv[1] : "mesh/data/square_4x4.msh";
        const std::string sol_dir = argc >= 3
            ? argv[2] : "data/ms_data/solver_data";
        const std::string out_dir = argc >= 4
            ? argv[3] : "data/ms_concentration_plot";
        const int problem_index = argc >= 5
            ? std::stoi(argv[4]) : 1;
        const int subdivisions = 24;
        const int k = 1;
        const int gauss_point_num = 9;
        const int gauss_point_num_1d = 9;

        // 读取网格
        vem::StraightMeshReader reader;
        if (!reader.read_mesh(mesh_file)) {
            throw std::runtime_error("无法读取网格: " + mesh_file);
        }
        const auto& mesh = reader.get_mesh_data();

        // 初始化算例
        MaxwellStefan::MSProblem problem;
        problem.set_problem_index(problem_index);
        if (!problem.init_problem()) {
            throw std::runtime_error("无法初始化 MS 算例");
        }

        // 用 MSSolver 获取自由度信息
        MSSolver solver(reader, problem,
                        0.001, 0.1,
                        50, 1e-10,
                        gauss_point_num, gauss_point_num_1d,
                        k, false);
        if (!solver.initialize()) {
            throw std::runtime_error("求解器初始化失败");
        }

        int total_dof = solver.getTotalDofAll();
        int total_flux_all = solver.getTotalDofFluxAll();
        int n_comp = problem.get_pde_data().n_components;
        int dof_conc_per_elem = solver.getDofPerElementConc();
        int total_conc_per_comp = solver.getTotalDofConc();

        // 列出所有解文件
        auto sol_files = listSolutionFiles(sol_dir, mesh.num_cells);
        std::cout << "找到 " << sol_files.size() << " 个时刻的解文件\n";

        // 创建输出目录
        std::string mkdir_cmd = "mkdir -p " + out_dir;
        int ret = system(mkdir_cmd.c_str());
        (void)ret;

        const MaxwellStefan::IsoparametricMapping& mapping =
            problem.get_mapping();
        vem::basis::BasisFunctionPloy basis(k);
        vem::PolygonTriangulator triangulator;
        const auto& pde = problem.get_pde_data();

        // ========== 第一步：预计算采样点（所有时刻共享拓扑与坐标） ==========
        // 存储每个采样点的物理坐标、以及每个单元的多项式求值索引
        struct SamplePoint {
            int cell;
            double x;       // 物理域 x
            double y;       // 物理域 y
            double xi;      // 计算域 xi
            double eta;     // 计算域 eta
        };
        std::vector<SamplePoint> sample_points;
        std::vector<std::tuple<int, int, int>> triangles_list;  // 三角形拓扑

        int base_triangle_id = 0;
        int next_triangle_id = 0;

        // 为每个单元预计算单项式在采样点的值（这样后续每个时刻只做点积）
        // monomial_vals[point_id][m] = 第 m 个单项式在该点的值
        std::vector<std::vector<double>> monomial_vals;

        for (int elem = 0; elem < mesh.num_cells; ++elem) {
            const vem::basis::Point2D center(
                mesh.cell_centroid_x[elem], mesh.cell_centroid_y[elem]);
            const double diameter = mesh.cell_diameter[elem];
            const vem::TriangulatedElement triangulated =
                triangulator.triangulateElement(mesh, elem);

            for (const vem::Triangle& triangle : triangulated.triangles) {
                const double xi0 = triangle.vertices[0];
                const double eta0 = triangle.vertices[1];
                const double xi1 = triangle.vertices[2];
                const double eta1 = triangle.vertices[3];
                const double xi2 = triangle.vertices[4];
                const double eta2 = triangle.vertices[5];
                std::map<std::pair<int, int>, int> indices;

                int point_base = static_cast<int>(sample_points.size());

                // 采样点
                for (int i = 0; i <= subdivisions; ++i) {
                    for (int j = 0; j <= subdivisions - i; ++j) {
                        const double a =
                            static_cast<double>(i) / subdivisions;
                        const double b =
                            static_cast<double>(j) / subdivisions;
                        const double bc = 1.0 - a - b;
                        const double xi = bc * xi0 + a * xi1 + b * xi2;
                        const double eta = bc * eta0 + a * eta1 + b * eta2;
                        const vem::basis::Point2D comp_point(xi, eta);

                        const MaxwellStefan::Point2D physical =
                            mapping.physical_coords(xi, eta);

                        SamplePoint sp;
                        sp.cell = elem;
                        sp.x = physical.x;
                        sp.y = physical.y;
                        sp.xi = xi;
                        sp.eta = eta;
                        sample_points.push_back(sp);

                        // 计算所有单项式在该点的值
                        std::vector<double> mvals(dof_conc_per_elem);
                        for (int m = 0; m < dof_conc_per_elem; ++m) {
                            mvals[m] = basis.evalMonomial2D(
                                m, center, diameter, comp_point);
                        }
                        monomial_vals.push_back(mvals);

                        const int local_idx =
                            static_cast<int>(sample_points.size())
                            - 1 - point_base;
                        indices[std::make_pair(i, j)] = point_base + local_idx;
                    }
                }

                // 三角形拓扑
                for (int i = 0; i < subdivisions; ++i) {
                    for (int j = 0; j < subdivisions - i; ++j) {
                        const int p00 = pointIndex(indices, i, j);
                        const int p10 = pointIndex(indices, i + 1, j);
                        const int p01 = pointIndex(indices, i, j + 1);
                        triangles_list.push_back(
                            std::make_tuple(p00, p10, p01));
                        next_triangle_id++;
                        if (i + j <= subdivisions - 2) {
                            const int p11 = pointIndex(
                                indices, i + 1, j + 1);
                            triangles_list.push_back(
                                std::make_tuple(p10, p11, p01));
                            next_triangle_id++;
                        }
                    }
                }
                ++base_triangle_id;
            }
        }

        int n_points = static_cast<int>(sample_points.size());
        std::cout << "采样点数: " << n_points
                  << "，三角形数: " << triangles_list.size() << "\n";

        // ========== 第二步：写拓扑文件（只写一次） ==========
        {
            std::ofstream tri_file(out_dir + "/triangles.csv");
            tri_file << "triangle_id,cell,p0,p1,p2\n";
            for (size_t ti = 0; ti < triangles_list.size(); ++ti) {
                int p0 = std::get<0>(triangles_list[ti]);
                int p1 = std::get<1>(triangles_list[ti]);
                int p2 = std::get<2>(triangles_list[ti]);
                tri_file << ti << ',' << sample_points[p0].cell
                         << ',' << p0 << ',' << p1 << ',' << p2 << '\n';
            }
        }

        // 曲边文件（只写一次）
        {
            std::ofstream edge_file(out_dir + "/curved_edges.csv");
            writeCurvedEdges(edge_file, mesh, mapping, 60);
        }

        // ========== 第三步：初值时刻 t=0（用初值函数计算） ==========
        {
            std::ostringstream fname;
            fname << out_dir << "/concentration_t0.000000.csv";
            std::ofstream file(fname.str());
            file << std::setprecision(17);
            file << "point_id,cell,x,y,c0,c1,c2\n";
            for (int p = 0; p < n_points; ++p) {
                const auto& sp = sample_points[p];
                file << p << ',' << sp.cell << ','
                     << sp.x << ',' << sp.y;
                for (int c = 0; c < n_comp; ++c) {
                    double val = pde.initial_concentration_comp(
                        c, sp.xi, sp.eta);
                    file << ',' << val;
                }
                file << '\n';
            }
            std::cout << "  t=0.000000 (initial) → " << fname.str() << "\n";
        }

        // ========== 第四步：所有时间步 ==========
        for (const auto& entry : sol_files) {
            double time = entry.first;
            const std::string& sol_file = entry.second;

            // 读取解向量
            std::vector<double> solution = readSolutionVector(sol_file,
                                                              total_dof);

            // 计算每个单元的浓度多项式系数
            // 为了高效，按点计算：每个点用自己单元的系数做点积
            // 先按单元提取系数
            std::vector<std::vector<double>> elem_conc(
                mesh.num_cells,
                std::vector<double>(n_comp * dof_conc_per_elem, 0.0));
            for (int elem = 0; elem < mesh.num_cells; ++elem) {
                for (int c = 0; c < n_comp; ++c) {
                    for (int m = 0; m < dof_conc_per_elem; ++m) {
                        int global_idx = total_flux_all
                                       + c * total_conc_per_comp
                                       + elem * dof_conc_per_elem
                                       + m;
                        elem_conc[elem][c * dof_conc_per_elem + m] =
                            solution[global_idx];
                    }
                }
            }

            // 输出文件
            std::ostringstream time_ss;
            time_ss << std::fixed << std::setprecision(6) << time;
            std::string time_str = time_ss.str();
            std::string fname = out_dir + "/concentration_t" + time_str + ".csv";
            std::ofstream file(fname);
            file << std::setprecision(17);
            file << "point_id,cell,x,y,c0,c1,c2\n";

            for (int p = 0; p < n_points; ++p) {
                const auto& sp = sample_points[p];
                const auto& coeffs = elem_conc[sp.cell];
                const auto& mvals = monomial_vals[p];

                file << p << ',' << sp.cell << ','
                     << sp.x << ',' << sp.y;
                for (int c = 0; c < n_comp; ++c) {
                    double val = 0.0;
                    for (int m = 0; m < dof_conc_per_elem; ++m) {
                        val += coeffs[c * dof_conc_per_elem + m]
                             * mvals[m];
                    }
                    file << ',' << val;
                }
                file << '\n';
            }
            std::cout << "  t=" << time_str << " → " << fname << "\n";
        }

        std::cout << "\n全部导出完成！共 " << (sol_files.size() + 1)
                  << " 个时刻（含初值）\n";
        std::cout << "输出目录: " << out_dir << "\n";

    } catch (const std::exception& exception) {
        std::cerr << "\n导出浓度绘图数据失败: "
                  << exception.what() << "\n\n";
        status = 1;
    }

    PetscFinalize();
    return status;
}
