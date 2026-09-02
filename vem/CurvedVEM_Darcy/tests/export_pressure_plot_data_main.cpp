#include "../solver/DarcySolver.h"

#include <petsc.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct GridPoint {
    int i;
    int j;
    double xi;
    double eta;
};

int pointIndex(const std::map<std::pair<int, int>, int>& indices,
               int i, int j) {
    auto found = indices.find(std::make_pair(i, j));
    if (found == indices.end()) {
        throw std::runtime_error("细分三角形点编号不存在");
    }
    return found->second;
}

void writeCurvedEdges(std::ofstream& output,
                      const vem::StraightMeshData& mesh,
                      const Darcy::IsoparametricMapping& mapping,
                      int samples_per_edge) {
    output << "edge,point,x,y\n";
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
            const Darcy::Point2D physical = mapping.physical_coords(xi, eta);
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
        const std::string mesh_file = argc >= 2
            ? argv[1]
            : "mesh/data/square_1x1.msh";
        const std::string point_file = argc >= 3
            ? argv[2]
            : "data/pressure_plot/pressure_points.csv";
        const std::string triangle_file = argc >= 4
            ? argv[3]
            : "data/pressure_plot/pressure_triangles.csv";
        const std::string edge_file = argc >= 5
            ? argv[4]
            : "data/pressure_plot/curved_edges.csv";
        const int subdivisions = 24;

        vem::StraightMeshReader reader;
        if (!reader.read_mesh(mesh_file)) {
            throw std::runtime_error("无法读取网格: " + mesh_file);
        }

        Darcy::DarcyProblem problem;
        problem.set_problem_index(1);
        if (!problem.init_problem()) {
            throw std::runtime_error("无法初始化算例1");
        }

        DarcySolver solver(reader, problem, 9, 9, 1);
        if (!solver.solve()) {
            throw std::runtime_error("Darcy 方程求解失败");
        }

        std::ofstream points(point_file);
        std::ofstream triangles(triangle_file);
        std::ofstream edges(edge_file);
        if (!points || !triangles || !edges) {
            throw std::runtime_error("无法创建压力绘图数据文件");
        }
        points << std::setprecision(17);
        triangles << std::setprecision(17);
        edges << std::setprecision(17);
        points << "point_id,cell,base_triangle,xi,eta,x,y,p_exact,p_numerical\n";
        triangles << "triangle_id,cell,p0,p1,p2\n";

        const auto& mesh = reader.get_mesh_data();
        const Darcy::IsoparametricMapping& mapping = problem.get_mapping();
        vem::basis::BasisFunctionPloy basis(solver.getK());
        vem::PolygonTriangulator triangulator;
        const int pressure_dofs = solver.getDofPerElementConc();
        int next_point_id = 0;
        int next_triangle_id = 0;
        int base_triangle_id = 0;

        for (int elem = 0; elem < mesh.num_cells; ++elem) {
            std::vector<double> pressure_coefficients(pressure_dofs, 0.0);
            for (int index = 0; index < pressure_dofs; ++index) {
                const PetscInt global_index =
                    solver.getTotalDofFlux() + elem * pressure_dofs + index;
                PetscScalar value = 0.0;
                VecGetValues(vem::get_raw(solver.getSolution()),
                             1, &global_index, &value);
                pressure_coefficients[index] = PetscRealPart(value);
            }

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

                // 包含三条边界的规则重心网格。每个原始三角形独立保存，
                // 从而不在压力间断的单元边界两侧共享数值节点。
                for (int i = 0; i <= subdivisions; ++i) {
                    for (int j = 0; j <= subdivisions - i; ++j) {
                        const double a =
                            static_cast<double>(i) / subdivisions;
                        const double b =
                            static_cast<double>(j) / subdivisions;
                        const double c = 1.0 - a - b;
                        const double xi = c * xi0 + a * xi1 + b * xi2;
                        const double eta = c * eta0 + a * eta1 + b * eta2;
                        const vem::basis::Point2D computational_point(xi, eta);
                        double pressure_numerical = 0.0;
                        for (int index = 0; index < pressure_dofs; ++index) {
                            pressure_numerical += pressure_coefficients[index] *
                                basis.evalMonomial2D(
                                    index, center, diameter,
                                    computational_point);
                        }
                        const Darcy::Point2D physical =
                            mapping.physical_coords(xi, eta);
                        const double pressure_exact =
                            problem.get_pde_data().solution_p(
                                physical.x, physical.y);
                        const int point_id = next_point_id++;
                        indices[std::make_pair(i, j)] = point_id;
                        points << point_id << ',' << elem << ','
                               << base_triangle_id << ',' << xi << ',' << eta
                               << ',' << physical.x << ',' << physical.y << ','
                               << pressure_exact << ',' << pressure_numerical
                               << '\n';
                    }
                }

                // 显式保存细分三角形连接关系，禁止绘图端重新剖分。
                for (int i = 0; i < subdivisions; ++i) {
                    for (int j = 0; j < subdivisions - i; ++j) {
                        const int p00 = pointIndex(indices, i, j);
                        const int p10 = pointIndex(indices, i + 1, j);
                        const int p01 = pointIndex(indices, i, j + 1);
                        triangles << next_triangle_id++ << ',' << elem << ','
                                  << p00 << ',' << p10 << ',' << p01 << '\n';
                        if (i + j <= subdivisions - 2) {
                            const int p11 = pointIndex(indices, i + 1, j + 1);
                            triangles << next_triangle_id++ << ',' << elem << ','
                                      << p10 << ',' << p11 << ',' << p01 << '\n';
                        }
                    }
                }
                ++base_triangle_id;
            }
        }

        writeCurvedEdges(edges, mesh, mapping, 60);
        std::cout << "压力点数据: " << point_file << '\n';
        std::cout << "显式三角形拓扑: " << triangle_file << '\n';
        std::cout << "曲边网格数据: " << edge_file << '\n';
    } catch (const std::exception& exception) {
        std::cerr << "导出压力绘图数据失败: "
                  << exception.what() << std::endl;
        status = 1;
    }

    PetscFinalize();
    return status;
}
