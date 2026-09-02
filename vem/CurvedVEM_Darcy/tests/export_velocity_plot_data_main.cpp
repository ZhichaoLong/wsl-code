#include "../solver/DarcySolver.h"

#include <petsc.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct SamplePoint {
    double xi;
    double eta;
};

std::vector<SamplePoint> sampleTriangle(const vem::Triangle& triangle,
                                        int subdivisions) {
    std::vector<SamplePoint> points;
    const double x0 = triangle.vertices[0];
    const double y0 = triangle.vertices[1];
    const double x1 = triangle.vertices[2];
    const double y1 = triangle.vertices[3];
    const double x2 = triangle.vertices[4];
    const double y2 = triangle.vertices[5];

    // 使用严格内部的重心采样，避免相邻单元边上的重复值影响绘图。
    for (int i = 0; i < subdivisions; ++i) {
        for (int j = 0; j < subdivisions - i; ++j) {
            const double a = (i + 1.0 / 3.0) / subdivisions;
            const double b = (j + 1.0 / 3.0) / subdivisions;
            if (a + b >= 1.0) {
                continue;
            }
            const double c = 1.0 - a - b;
            points.push_back({c * x0 + a * x1 + b * x2,
                              c * y0 + a * y1 + b * y2});
        }
    }
    return points;
}

std::vector<PetscInt> localFluxGlobalDofs(const DarcySolver& solver,
                                           int elem) {
    const auto* mesh = solver.getMeshData();
    const int k = solver.getK();
    const int edge_count = mesh->nodes_per_cell[elem];
    const int edge_dofs = edge_count * (k + 1);
    const int internal_dofs = solver.getDimGradMk() + solver.getDimCurlMk();
    const int flux_dofs = solver.getDofPerElementFlux(elem);
    std::vector<PetscInt> result(flux_dofs, -1);

    const int edge_start = mesh->cell_edge_indices[elem];
    for (int degree = 0; degree <= k; ++degree) {
        for (int local_edge = 0; local_edge < edge_count; ++local_edge) {
            const int global_edge = mesh->global_edges[edge_start + local_edge];
            result[degree * edge_count + local_edge] =
                degree * mesh->num_edges + global_edge;
        }
    }
    const int internal_start =
        mesh->num_edges * (k + 1) + elem * internal_dofs;
    for (int index = 0; index < internal_dofs; ++index) {
        result[edge_dofs + index] = internal_start + index;
    }
    return result;
}

std::vector<double> localDirection(const DarcySolver& solver, int elem) {
    const auto* mesh = solver.getMeshData();
    const int k = solver.getK();
    const int edge_count = mesh->nodes_per_cell[elem];
    const int flux_dofs = solver.getDofPerElementFlux(elem);
    const int edge_start = mesh->cell_edge_indices[elem];
    const int node_start = mesh->cell_node_indices[elem];
    std::vector<double> direction(flux_dofs, 1.0);

    for (int degree = 0; degree <= k; ++degree) {
        for (int local_edge = 0; local_edge < edge_count; ++local_edge) {
            const int global_edge = mesh->global_edges[edge_start + local_edge];
            if (mesh->edge_occurrence[global_edge] != 2) {
                continue;
            }
            const int local_node0 = mesh->cell_nodes[node_start + local_edge];
            const int local_node1 = mesh->cell_nodes[
                node_start + (local_edge + 1) % edge_count];
            const int global_node0 = mesh->edge_endpoints[2 * global_edge];
            const int global_node1 = mesh->edge_endpoints[2 * global_edge + 1];
            const bool reversed = local_node0 == global_node1 &&
                                  local_node1 == global_node0;
            if (reversed && degree % 2 == 0) {
                direction[degree * edge_count + local_edge] = -1.0;
            }
        }
    }
    return direction;
}

std::vector<double> projectedFluxCoefficients(DarcySolver& solver, int elem) {
    const int flux_dofs = solver.getDofPerElementFlux(elem);
    std::vector<PetscInt> global_dofs = localFluxGlobalDofs(solver, elem);
    std::vector<double> direction = localDirection(solver, elem);

    AutoPetscVec local_flux = vem::create_vector(flux_dofs);
    for (int index = 0; index < flux_dofs; ++index) {
        PetscScalar global_value = 0.0;
        VecGetValues(vem::get_raw(solver.getSolution()),
                     1, &global_dofs[index], &global_value);
        PetscScalar local_value =
            direction[index] * PetscRealPart(global_value);
        PetscInt local_index = index;
        VecSetValue(vem::get_raw(local_flux), local_index,
                    local_value, INSERT_VALUES);
    }
    VecAssemblyBegin(vem::get_raw(local_flux));
    VecAssemblyEnd(vem::get_raw(local_flux));

    AutoPetscMat G = solver.getMatrixG(elem);
    AutoPetscMat W = solver.getMatrixW(elem);
    AutoPetscMat H = solver.getMatrixH(elem);
    AutoPetscMat H_star = solver.getMatrixH_star(elem);
    AutoPetscMat B = solver.getMatrixB(elem, H_star, H, W);
    AutoPetscMat projection = solver.getMatrixL2Proj_ploy(G, B);

    PetscInt rows = 0;
    PetscInt columns = 0;
    MatGetSize(vem::get_raw(projection), &rows, &columns);
    AutoPetscVec coefficients = vem::create_vector(rows);
    MatMult(vem::get_raw(projection), vem::get_raw(local_flux),
            vem::get_raw(coefficients));

    std::vector<double> result(rows, 0.0);
    for (PetscInt index = 0; index < rows; ++index) {
        PetscScalar value = 0.0;
        VecGetValues(vem::get_raw(coefficients), 1, &index, &value);
        result[index] = PetscRealPart(value);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    PetscErrorCode error = PetscInitialize(&argc, &argv, NULL, NULL);
    if (error != 0) {
        return static_cast<int>(error);
    }

    int status = 0;
    try {
        const std::string output_file =
            argc >= 2 ? argv[1] : "data/velocity_plot/velocity_samples.csv";
        const int subdivisions = 12;

        vem::StraightMeshReader reader;
        if (!reader.read_mesh("/home/lzccode/code/vem/gmesh_py/mesh_data/orthogonal_1x1.msh")) {
            throw std::runtime_error("无法读取 orthogonal_1x1.msh");
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

        std::ofstream output(output_file);
        if (!output) {
            throw std::runtime_error("无法创建速度采样文件: " + output_file);
        }
        output << std::setprecision(17);
        output << "cell,xi,eta,x,y,u_exact_x,u_exact_y,u_num_x,u_num_y\n";

        const auto& mesh = reader.get_mesh_data();
        const Darcy::IsoparametricMapping& mapping = problem.get_mapping();
        vem::basis::BasisFunctionPloy basis(solver.getK());
        vem::PolygonTriangulator triangulator;

        for (int elem = 0; elem < mesh.num_cells; ++elem) {
            std::vector<double> coefficients =
                projectedFluxCoefficients(solver, elem);
            vem::basis::Point2D center(mesh.cell_centroid_x[elem],
                                       mesh.cell_centroid_y[elem]);
            const double diameter = mesh.cell_diameter[elem];
            vem::TriangulatedElement triangulated =
                triangulator.triangulateElement(mesh, elem);

            for (const vem::Triangle& triangle : triangulated.triangles) {
                for (const SamplePoint& sample :
                     sampleTriangle(triangle, subdivisions)) {
                    vem::basis::Point2D computational_point(
                        sample.xi, sample.eta);
                    vem::basis::Vector2D numerical_hat(0.0, 0.0);
                    for (std::size_t index = 0;
                         index < coefficients.size(); ++index) {
                        vem::basis::Vector2D value = basis.evalHdivBasis(
                            static_cast<int>(index), center, diameter,
                            computational_point);
                        numerical_hat.x += coefficients[index] * value.x;
                        numerical_hat.y += coefficients[index] * value.y;
                    }

                    const Darcy::Tensor2D jacobian =
                        mapping.jacobian(sample.xi, sample.eta);
                    const double detJ =
                        mapping.jacobian_det(sample.xi, sample.eta);
                    const double numerical_x =
                        (jacobian.xx * numerical_hat.x +
                         jacobian.xy * numerical_hat.y) / detJ;
                    const double numerical_y =
                        (jacobian.yx * numerical_hat.x +
                         jacobian.yy * numerical_hat.y) / detJ;

                    const Darcy::Point2D physical =
                        mapping.physical_coords(sample.xi, sample.eta);
                    const Darcy::Vector2D exact =
                        problem.get_pde_data().solution_u(
                            physical.x, physical.y);
                    output << elem << ',' << sample.xi << ',' << sample.eta
                           << ',' << physical.x << ',' << physical.y << ','
                           << exact.x << ',' << exact.y << ','
                           << numerical_x << ',' << numerical_y << '\n';
                }
            }
        }
        std::cout << "速度绘图数据已写入: " << output_file << std::endl;
    } catch (const std::exception& exception) {
        std::cerr << "导出速度绘图数据失败: " << exception.what() << std::endl;
        status = 1;
    }

    PetscFinalize();
    return status;
}
