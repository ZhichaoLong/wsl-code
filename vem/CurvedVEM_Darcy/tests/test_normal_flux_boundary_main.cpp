#include "../solver/DarcySolver.h"

#include <petsc.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

bool nearlyEqual(double first, double second, double tolerance) {
    return std::fabs(first - second) <= tolerance *
        (1.0 + std::max(std::fabs(first), std::fabs(second)));
}

}  // namespace

int main(int argc, char** argv) {
    PetscErrorCode error = PetscInitialize(&argc, &argv, NULL, NULL);
    if (error != 0) {
        return static_cast<int>(error);
    }

    bool passed = true;
    IS free_dofs = PETSC_NULLPTR;
    try {
        vem::StraightMeshReader reader;
        if (!reader.read_mesh("/home/lzccode/code/vem/gmesh_py/mesh_data/orthogonal_1x1.msh")) {
            throw std::runtime_error("无法读取边界条件测试网格");
        }

        Darcy::DarcyProblem problem;
        problem.set_problem_index(1);
        if (!problem.init_problem()) {
            throw std::runtime_error("无法初始化算例1");
        }

        DarcySolver solver(reader, problem, 9, 9, 1);
        error = solver.assembleStiffnessMatrix();
        if (error != 0) {
            throw std::runtime_error("全局系统组装失败");
        }

        AutoPetscMat reduced_matrix;
        AutoPetscVec reduced_rhs;
        AutoPetscVec boundary_values;
        PetscInt total_dof = 0;
        error = solver.applyNormalFluxBoundaryConditions(
            reduced_matrix, reduced_rhs, free_dofs,
            boundary_values, total_dof);
        if (error != 0) {
            throw std::runtime_error("法向通量边界条件处理失败");
        }

        const auto& mesh = reader.get_mesh_data();
        PetscInt expected_boundary_dofs =
            mesh.num_boundary_edges * (solver.getK() + 1);
        PetscInt free_size = 0;
        ISGetLocalSize(free_dofs, &free_size);
        PetscInt reduced_rows = 0;
        PetscInt reduced_cols = 0;
        MatGetSize(vem::get_raw(reduced_matrix),
                   &reduced_rows, &reduced_cols);
        PetscInt reduced_rhs_size = 0;
        VecGetSize(vem::get_raw(reduced_rhs), &reduced_rhs_size);

        passed &= total_dof == solver.getTotalDofFlux() +
                                   solver.getTotalDofConc() + 1;
        passed &= free_size == total_dof - expected_boundary_dofs;
        passed &= reduced_rows == free_size && reduced_cols == free_size;
        passed &= reduced_rhs_size == free_size;

        // 独立检查一个边界自由度：找到第一个边界边所属单元，按该单元
        // 的逆时针局部方向重新计算 0 次法向通量矩。
        int selected_global_edge = -1;
        int selected_node0 = -1;
        int selected_node1 = -1;
        for (int elem = 0; elem < mesh.num_cells && selected_global_edge < 0;
             ++elem) {
            int edge_count = mesh.nodes_per_cell[elem];
            int node_start = mesh.cell_node_indices[elem];
            int edge_start = mesh.cell_edge_indices[elem];
            for (int local_edge = 0; local_edge < edge_count; ++local_edge) {
                int global_edge = mesh.global_edges[edge_start + local_edge];
                if (mesh.edge_occurrence[global_edge] == 1) {
                    selected_global_edge = global_edge;
                    selected_node0 = mesh.cell_nodes[node_start + local_edge];
                    selected_node1 = mesh.cell_nodes[
                        node_start + (local_edge + 1) % edge_count];
                    break;
                }
            }
        }
        if (selected_global_edge < 0) {
            throw std::runtime_error("没有找到边界边");
        }

        Point2D normal = solver.computeEdgeNormal(
            selected_node0, selected_node1);
        double x0 = mesh.node_coords[2 * selected_node0];
        double y0 = mesh.node_coords[2 * selected_node0 + 1];
        double x1 = mesh.node_coords[2 * selected_node1];
        double y1 = mesh.node_coords[2 * selected_node1 + 1];
        vem::GaussQuadrature quadrature;
        auto points = quadrature.get_line_2D_points(
            vem::QuadratureType::GAUSS_LEGENDRE, 9,
            x0, y0, x1, y1);
        double expected_moment = 0.0;
        for (const auto& point : points) {
            Darcy::Vector2D velocity = problem.computational_solution_u(
                point.x[0], point.x[1]);
            expected_moment +=
                (velocity.x * normal.x + velocity.y * normal.y) * point.w;
        }
        PetscInt boundary_index = selected_global_edge;
        PetscScalar actual_moment = 0.0;
        VecGetValues(vem::get_raw(boundary_values), 1,
                     &boundary_index, &actual_moment);
        passed &= nearlyEqual(
            PetscRealPart(actual_moment), expected_moment, 1e-12);

        // 内部边自由度和非速度自由度在边界值向量中应保持零。
        for (int edge = 0; edge < mesh.num_edges; ++edge) {
            if (mesh.edge_occurrence[edge] != 2) {
                continue;
            }
            for (int degree = 0; degree <= solver.getK(); ++degree) {
                PetscInt index = degree * mesh.num_edges + edge;
                PetscScalar value = 0.0;
                VecGetValues(vem::get_raw(boundary_values), 1,
                             &index, &value);
                passed &= std::fabs(PetscRealPart(value)) < 1e-14;
            }
        }
        for (PetscInt index = solver.getTotalDofFlux();
             index < total_dof; ++index) {
            PetscScalar value = 0.0;
            VecGetValues(vem::get_raw(boundary_values), 1,
                         &index, &value);
            passed &= std::fabs(PetscRealPart(value)) < 1e-14;
        }

        std::cout << "法向通量边界自由度数: "
                  << expected_boundary_dofs << '\n';
        std::cout << "原系统维数: " << total_dof
                  << ", 降维系统维数: " << reduced_rows << '\n';
        std::cout << "边界边 " << selected_global_edge
                  << " 的 0 次通量矩: 实际="
                  << PetscRealPart(actual_moment)
                  << ", 独立积分=" << expected_moment << '\n';
    } catch (const std::exception& exception) {
        std::cerr << "边界条件测试异常: " << exception.what() << std::endl;
        passed = false;
    }

    if (free_dofs != PETSC_NULLPTR) {
        ISDestroy(&free_dofs);
    }
    PetscFinalize();
    if (!passed) {
        std::cerr << "法向通量边界条件测试失败" << std::endl;
        return 1;
    }
    std::cout << "法向通量边界条件测试通过 ✓" << std::endl;
    return 0;
}
