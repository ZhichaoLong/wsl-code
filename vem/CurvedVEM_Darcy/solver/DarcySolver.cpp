#include "DarcySolver.h"

#include <cmath>
#include <stdexcept>

namespace {

using Basis = vem::basis::BasisFunctionPloy;
using BasisPoint = vem::basis::Point2D;
using BasisVector = vem::basis::Vector2D;

Darcy::Tensor2D inverseTensor(const Darcy::Tensor2D& tensor) {
    double determinant = tensor.xx * tensor.yy - tensor.xy * tensor.yx;
    if (!std::isfinite(determinant) || std::fabs(determinant) < 1e-14) {
        throw std::runtime_error(
            "computational permeability tensor is singular");
    }

    double inverse_determinant = 1.0 / determinant;
    return Darcy::Tensor2D(
        tensor.yy * inverse_determinant,
        -tensor.xy * inverse_determinant,
        -tensor.yx * inverse_determinant,
        tensor.xx * inverse_determinant);
}

BasisVector multiply(const Darcy::Tensor2D& tensor,
                     const BasisVector& vector) {
    return BasisVector(
        tensor.xx * vector.x + tensor.xy * vector.y,
        tensor.yx * vector.x + tensor.yy * vector.y);
}

void setMatrixValue(Mat matrix, PetscInt row, PetscInt column, double value) {
    PetscScalar scalar = value;
    PetscErrorCode error = MatSetValues(
        matrix, 1, &row, 1, &column, &scalar, INSERT_VALUES);
    if (error != 0) {
        throw std::runtime_error("MatSetValues failed while assembling G_Kappa");
    }
}

void assembleMatrix(Mat matrix) {
    PetscErrorCode error = MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY);
    if (error == 0) {
        error = MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY);
    }
    if (error != 0) {
        throw std::runtime_error("PETSc matrix assembly failed");
    }
}

}  // namespace

DarcySolver::DarcySolver(const MeshReader& mesh_reader,
                         const Darcy::DarcyProblem& problem,
                         int gauss_point_num,
                         int gauss_point_num_1d,
                         int k)
    : HdivMatrix(mesh_reader, gauss_point_num, gauss_point_num_1d, k),
      problem_(&problem),
      initialized_(false),
      solved_(false),
      l2_error_pressure_(0.0),
      l2_error_flux_(0.0) {
    // get_mapping() 会在算例尚未初始化时抛出明确异常。
    problem_->get_mapping();
    if (!initialize()) {
        throw std::runtime_error("DarcySolver initialization failed");
    }
}

DarcySolver::~DarcySolver() = default;

bool DarcySolver::initialize() {
    PetscInt total_dof =
        static_cast<PetscInt>(getTotalDofFlux() + getTotalDofConc() + 1);
    if (total_dof <= 0) {
        return false;
    }

    const std::vector<int>& flux_dofs = getDofPerElementFluxVec();
    PetscInt max_local_dof = 1;
    for (int flux_dof : flux_dofs) {
        PetscInt local_dof = static_cast<PetscInt>(
            flux_dof + getDofPerElementConc() + 1);
        if (local_dof > max_local_dof) {
            max_local_dof = local_dof;
        }
    }

    PetscInt nnz_per_row = max_local_dof * 6;
    if (nnz_per_row > total_dof) {
        nnz_per_row = total_dof;
    }

    system_stiffness_mat_ =
        vem::create_sparse_matrix(total_dof, total_dof, nnz_per_row);
    system_rhs_vec_ = vem::create_vector(total_dof);

    // 显式建立全局稀疏矩阵的全部对角线结构。即使当前值为零，
    // LU/ILU 及部分 PETSc 预条件器也要求每一行存在对角元槽位。
    Mat raw_system_matrix = vem::get_raw(system_stiffness_mat_);
    for (PetscInt index = 0; index < total_dof; ++index) {
        PetscScalar zero = 0.0;
        PetscErrorCode diagonal_error = MatSetValue(
            raw_system_matrix, index, index, zero, INSERT_VALUES);
        if (diagonal_error != 0) {
            throw std::runtime_error(
                "failed to initialize system matrix diagonal");
        }
    }
    PetscErrorCode error = MatAssemblyBegin(
        raw_system_matrix, MAT_FINAL_ASSEMBLY);
    if (error == 0) {
        error = MatAssemblyEnd(raw_system_matrix, MAT_FINAL_ASSEMBLY);
    }
    if (error != 0) {
        throw std::runtime_error(
            "system stiffness matrix diagonal assembly failed");
    }

    // 只清零数值，保留上面建立的对角线非零结构。
    error = MatZeroEntries(raw_system_matrix);
    if (error != 0) {
        throw std::runtime_error("MatZeroEntries failed during initialization");
    }
    error = MatAssemblyBegin(
        vem::get_raw(system_stiffness_mat_), MAT_FINAL_ASSEMBLY);
    if (error == 0) {
        error = MatAssemblyEnd(
            vem::get_raw(system_stiffness_mat_), MAT_FINAL_ASSEMBLY);
    }
    if (error != 0) {
        throw std::runtime_error(
            "system stiffness matrix assembly failed during initialization");
    }
    error = VecSet(vem::get_raw(system_rhs_vec_), 0.0);
    if (error != 0) {
        throw std::runtime_error("VecSet failed during initialization");
    }
    error = VecAssemblyBegin(vem::get_raw(system_rhs_vec_));
    if (error == 0) {
        error = VecAssemblyEnd(vem::get_raw(system_rhs_vec_));
    }
    if (error != 0) {
        throw std::runtime_error("right-hand-side vector assembly failed");
    }

    solution_.reset();
    initialized_ = true;
    solved_ = false;
    l2_error_pressure_ = 0.0;
    l2_error_flux_ = 0.0;
    return true;
}

AutoPetscMat DarcySolver::getMatrixG_Kappa(int mesh_idx) {
    return HdivMatrixG_Kappa(mesh_idx);
}

AutoPetscMat DarcySolver::getMatrixK_ac(
    const AutoPetscMat& G_Kappa,
    const AutoPetscMat& L2Proj_ploy) {
    return HdivMatrixK_ac(G_Kappa, L2Proj_ploy);
}

AutoPetscMat DarcySolver::getMatrixK_as(
    int mesh_idx,
    const AutoPetscMat& L2Proj_basis) {
    return HdivMatrixK_as(mesh_idx, L2Proj_basis);
}

AutoPetscVec DarcySolver::getLocalRhs(int mesh_idx) {
    return HdivVecRhs(mesh_idx);
}

AutoPetscVec DarcySolver::getVecLagrange(int mesh_idx) {
    return HdivVecLagrange(mesh_idx);
}

std::vector<PetscInt> DarcySolver::getLocalToGlobalDofs(int elem) const {
    const MeshData* mesh_data = getMeshData();
    if (elem < 0 || elem >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }

    int k = getK();
    int edge_count = mesh_data->nodes_per_cell[elem];
    int edge_dofs = edge_count * (k + 1);
    int flux_dofs = getDofPerElementFlux(elem);
    int pressure_dofs = getDofPerElementConc();
    int local_dofs = flux_dofs + pressure_dofs + 1;
    int internal_dofs = getDimGradMk() + getDimCurlMk();

    std::vector<PetscInt> global_dofs(local_dofs, -1);
    int cell_edge_start = mesh_data->cell_edge_indices[elem];

    // 边自由度按“多项式次数块 × 局部边号”排列。
    for (int degree = 0; degree <= k; ++degree) {
        for (int local_edge = 0; local_edge < edge_count; ++local_edge) {
            int global_edge =
                mesh_data->global_edges[cell_edge_start + local_edge];
            int local_index = degree * edge_count + local_edge;
            global_dofs[local_index] =
                degree * mesh_data->num_edges + global_edge;
        }
    }

    // 单元内部速度自由度按单元连续编号。
    int local_index = edge_dofs;
    int internal_start =
        mesh_data->num_edges * (k + 1) + elem * internal_dofs;
    for (int index = 0; index < internal_dofs; ++index) {
        global_dofs[local_index++] = internal_start + index;
    }

    // 压力自由度位于全部速度自由度之后，按单元连续编号。
    int pressure_start = getTotalDofFlux() + elem * pressure_dofs;
    for (int index = 0; index < pressure_dofs; ++index) {
        global_dofs[local_index++] = pressure_start + index;
    }

    // 所有单元共享同一个全局压力零均值 Lagrange 乘子。
    global_dofs[local_index] = getTotalDofFlux() + getTotalDofConc();
    return global_dofs;
}

std::vector<double> DarcySolver::computeAndApplyDirectionAdjustment(
    int elem,
    Mat matrix) const {
    const MeshData* mesh_data = getMeshData();
    if (elem < 0 || elem >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    if (matrix == PETSC_NULLPTR) {
        throw std::invalid_argument("local matrix is null");
    }

    int k = getK();
    int edge_count = mesh_data->nodes_per_cell[elem];
    int flux_dofs = getDofPerElementFlux(elem);
    int local_dofs = flux_dofs + getDofPerElementConc() + 1;
    int cell_node_start = mesh_data->cell_node_indices[elem];
    int cell_edge_start = mesh_data->cell_edge_indices[elem];

    PetscInt rows = 0;
    PetscInt columns = 0;
    PetscErrorCode error = MatGetSize(matrix, &rows, &columns);
    if (error != 0 || rows != local_dofs || columns != local_dofs) {
        throw std::invalid_argument(
            "local matrix dimension does not match element dofs");
    }

    std::vector<double> directions(local_dofs, 1.0);
    for (int degree = 0; degree <= k; ++degree) {
        for (int local_edge = 0; local_edge < edge_count; ++local_edge) {
            int local_dof = degree * edge_count + local_edge;
            int global_edge =
                mesh_data->global_edges[cell_edge_start + local_edge];

            // 边界边只属于一个单元，沿用该单元的外法向约定。
            if (mesh_data->edge_occurrence[global_edge] == 1) {
                continue;
            }

            int local_node0 =
                mesh_data->cell_nodes[cell_node_start + local_edge];
            int local_node1 = mesh_data->cell_nodes[
                cell_node_start + (local_edge + 1) % edge_count];
            int global_node0 = mesh_data->edge_endpoints[2 * global_edge];
            int global_node1 = mesh_data->edge_endpoints[2 * global_edge + 1];

            bool same_orientation =
                local_node0 == global_node0 && local_node1 == global_node1;
            bool reverse_orientation =
                local_node0 == global_node1 && local_node1 == global_node0;
            if (!same_orientation && !reverse_orientation) {
                throw std::runtime_error(
                    "local edge endpoints do not match global edge");
            }

            // 反向边上：法向改变一次符号，l 次边单项式再产生 (-1)^l，
            // 因此偶次矩改变符号、奇次矩保持不变。
            if (reverse_orientation && degree % 2 == 0) {
                directions[local_dof] = -1.0;
            }
        }
    }

    std::vector<PetscInt> indices(local_dofs);
    for (int index = 0; index < local_dofs; ++index) {
        indices[index] = index;
    }
    std::vector<PetscScalar> values(
        static_cast<std::size_t>(local_dofs * local_dofs));
    error = MatGetValues(
        matrix, local_dofs, indices.data(),
        local_dofs, indices.data(), values.data());
    if (error != 0) {
        throw std::runtime_error(
            "MatGetValues failed during direction adjustment");
    }

    for (int row = 0; row < local_dofs; ++row) {
        for (int column = 0; column < local_dofs; ++column) {
            values[static_cast<std::size_t>(row * local_dofs + column)] *=
                directions[row] * directions[column];
        }
    }
    error = MatSetValues(
        matrix, local_dofs, indices.data(),
        local_dofs, indices.data(), values.data(), INSERT_VALUES);
    if (error != 0) {
        throw std::runtime_error(
            "MatSetValues failed during direction adjustment");
    }
    assembleMatrix(matrix);
    return directions;
}

void DarcySolver::assembleLocalToGlobal(
    int elem,
    const AutoPetscMat& local_matrix) {
    if (!local_matrix || vem::get_raw(local_matrix) == PETSC_NULLPTR) {
        throw std::invalid_argument("local matrix is null");
    }

    std::vector<PetscInt> global_dofs = getLocalToGlobalDofs(elem);
    int local_dofs = static_cast<int>(global_dofs.size());
    PetscInt rows = 0;
    PetscInt columns = 0;
    PetscErrorCode error = MatGetSize(
        vem::get_raw(local_matrix), &rows, &columns);
    if (error != 0 || rows != local_dofs || columns != local_dofs) {
        throw std::invalid_argument(
            "local matrix dimension does not match local-to-global map");
    }

    std::vector<PetscInt> local_indices(local_dofs);
    for (int index = 0; index < local_dofs; ++index) {
        local_indices[index] = index;
    }
    std::vector<PetscScalar> values(
        static_cast<std::size_t>(local_dofs * local_dofs));
    error = MatGetValues(
        vem::get_raw(local_matrix), local_dofs, local_indices.data(),
        local_dofs, local_indices.data(), values.data());
    if (error != 0) {
        throw std::runtime_error(
            "MatGetValues failed during local-to-global assembly");
    }

    error = MatSetValues(
        vem::get_raw(system_stiffness_mat_),
        local_dofs, global_dofs.data(),
        local_dofs, global_dofs.data(), values.data(), ADD_VALUES);
    if (error != 0) {
        throw std::runtime_error(
            "MatSetValues failed during global matrix assembly");
    }
}

void DarcySolver::assembleLocalRhsToGlobal(
    int elem,
    const AutoPetscVec& local_rhs) {
    if (!local_rhs || vem::get_raw(local_rhs) == PETSC_NULLPTR) {
        throw std::invalid_argument("local RHS is null");
    }

    std::vector<PetscInt> global_dofs = getLocalToGlobalDofs(elem);
    int local_dofs = static_cast<int>(global_dofs.size());
    PetscInt vector_size = 0;
    PetscErrorCode error = VecGetSize(vem::get_raw(local_rhs), &vector_size);
    if (error != 0 || vector_size != local_dofs) {
        throw std::invalid_argument(
            "local RHS dimension does not match local-to-global map");
    }

    std::vector<PetscInt> local_indices(local_dofs);
    for (int index = 0; index < local_dofs; ++index) {
        local_indices[index] = index;
    }
    std::vector<PetscScalar> values(local_dofs);
    error = VecGetValues(
        vem::get_raw(local_rhs), local_dofs,
        local_indices.data(), values.data());
    if (error != 0) {
        throw std::runtime_error(
            "VecGetValues failed during local RHS assembly");
    }

    error = VecSetValues(
        vem::get_raw(system_rhs_vec_), local_dofs,
        global_dofs.data(), values.data(), ADD_VALUES);
    if (error != 0) {
        throw std::runtime_error(
            "VecSetValues failed during global RHS assembly");
    }
}

PetscErrorCode DarcySolver::assembleStiffnessMatrix() {
    Mat global_matrix = vem::get_raw(system_stiffness_mat_);
    Vec global_rhs = vem::get_raw(system_rhs_vec_);
    PetscErrorCode error = MatZeroEntries(global_matrix);
    if (error != 0) {
        return error;
    }
    error = VecSet(global_rhs, 0.0);
    if (error != 0) {
        return error;
    }

    try {
        const MeshData* mesh_data = getMeshData();
        for (int elem = 0; elem < mesh_data->num_cells; ++elem) {
            int flux_dofs = getDofPerElementFlux(elem);
            int pressure_dofs = getDofPerElementConc();
            int local_dofs = flux_dofs + pressure_dofs + 1;

            AutoPetscMat D = getMatrixD(elem);
            AutoPetscMat G = getMatrixG(elem);
            AutoPetscMat W = getMatrixW(elem);
            AutoPetscMat H = getMatrixH(elem);
            AutoPetscMat H_star = getMatrixH_star(elem);
            AutoPetscMat B = getMatrixB(elem, H_star, H, W);
            AutoPetscMat projection_poly = getMatrixL2Proj_ploy(G, B);
            AutoPetscMat projection_dof = getMatrixL2Proj_basis(G, B, D);
            AutoPetscMat G_Kappa = getMatrixG_Kappa(elem);
            AutoPetscMat K_ac = getMatrixK_ac(G_Kappa, projection_poly);
            AutoPetscMat K_as = getMatrixK_as(elem, projection_dof);
            AutoPetscMat velocity_matrix = vem::add_matrices(K_ac, K_as);
            AutoPetscMat W_transpose = vem::transpose_matrix(W);
            AutoPetscVec lagrange = getVecLagrange(elem);
            AutoPetscVec local_rhs = getLocalRhs(elem);

            AutoPetscMat local_matrix =
                vem::create_dense_matrix(local_dofs, local_dofs);
            Mat raw_local = vem::get_raw(local_matrix);
            error = MatZeroEntries(raw_local);
            if (error != 0) {
                return error;
            }

            // [ K_ac+K_as  -W^T  0 ]
            // [ W           0    L ]
            // [ 0           L^T  0 ]
            vem::insert_submatrix(raw_local, velocity_matrix, 0, 0);
            AutoPetscMat negative_W_transpose = vem::scale_matrix(W_transpose, -1.0);
            vem::insert_submatrix(
                raw_local, negative_W_transpose, 0, flux_dofs);
            vem::insert_submatrix(raw_local, W, flux_dofs, 0);

            int lagrange_index = flux_dofs + pressure_dofs;
            for (int pressure = 0; pressure < pressure_dofs; ++pressure) {
                PetscInt vector_index = flux_dofs + pressure;
                PetscScalar value = 0.0;
                error = VecGetValues(
                    vem::get_raw(lagrange), 1, &vector_index, &value);
                if (error != 0) {
                    return error;
                }
                error = MatSetValue(
                    raw_local, vector_index, lagrange_index,
                    value, INSERT_VALUES);
                if (error != 0) {
                    return error;
                }
                error = MatSetValue(
                    raw_local, lagrange_index, vector_index,
                    value, INSERT_VALUES);
                if (error != 0) {
                    return error;
                }
            }
            assembleMatrix(raw_local);

            computeAndApplyDirectionAdjustment(elem, raw_local);
            assembleLocalToGlobal(elem, local_matrix);
            assembleLocalRhsToGlobal(elem, local_rhs);
        }
    } catch (const std::exception&) {
        return PETSC_ERR_LIB;
    }

    error = MatAssemblyBegin(global_matrix, MAT_FINAL_ASSEMBLY);
    if (error != 0) {
        return error;
    }
    error = MatAssemblyEnd(global_matrix, MAT_FINAL_ASSEMBLY);
    if (error != 0) {
        return error;
    }
    error = VecAssemblyBegin(global_rhs);
    if (error != 0) {
        return error;
    }
    error = VecAssemblyEnd(global_rhs);
    return error;
}

PetscErrorCode DarcySolver::applyNormalFluxBoundaryConditions(
    AutoPetscMat& reduced_matrix,
    AutoPetscVec& reduced_rhs,
    IS& free_dofs,
    AutoPetscVec& boundary_values,
    PetscInt& total_dof) {
    const MeshData* mesh_data = getMeshData();
    Mat global_matrix = vem::get_raw(system_stiffness_mat_);
    Vec global_rhs = vem::get_raw(system_rhs_vec_);
    if (global_matrix == PETSC_NULLPTR || global_rhs == PETSC_NULLPTR) {
        return PETSC_ERR_ARG_NULL;
    }

    PetscInt matrix_rows = 0;
    PetscInt matrix_columns = 0;
    PetscCall(MatGetSize(global_matrix, &matrix_rows, &matrix_columns));
    PetscCall(VecGetSize(global_rhs, &total_dof));
    if (matrix_rows != total_dof || matrix_columns != total_dof) {
        return PETSC_ERR_ARG_SIZ;
    }

    boundary_values = vem::create_vector(total_dof);
    Vec raw_boundary_values = vem::get_raw(boundary_values);
    PetscCall(VecSet(raw_boundary_values, 0.0));

    std::vector<PetscBool> is_boundary_dof(
        static_cast<std::size_t>(total_dof), PETSC_FALSE);
    Basis edge_basis(getK());
    vem::GaussQuadrature quadrature;

    // 每条边界边只属于一个单元。按该单元逆时针局部边方向计算
    // 计算区域外法向和边单项式，使边界值与全局组装时保留的边界
    // 局部自由度方向完全一致。
    for (int elem = 0; elem < mesh_data->num_cells; ++elem) {
        int edge_count = mesh_data->nodes_per_cell[elem];
        int cell_node_start = mesh_data->cell_node_indices[elem];
        int cell_edge_start = mesh_data->cell_edge_indices[elem];

        for (int local_edge = 0; local_edge < edge_count; ++local_edge) {
            int global_edge =
                mesh_data->global_edges[cell_edge_start + local_edge];
            if (mesh_data->edge_occurrence[global_edge] != 1) {
                continue;
            }

            int node0 =
                mesh_data->cell_nodes[cell_node_start + local_edge];
            int node1 = mesh_data->cell_nodes[
                cell_node_start + (local_edge + 1) % edge_count];
            BasisPoint point0(
                mesh_data->node_coords[2 * node0],
                mesh_data->node_coords[2 * node0 + 1]);
            BasisPoint point1(
                mesh_data->node_coords[2 * node1],
                mesh_data->node_coords[2 * node1 + 1]);
            Point2D normal = computeEdgeNormal(node0, node1);

            std::vector<vem::QuadPoint> points =
                quadrature.get_line_2D_points(
                    vem::QuadratureType::GAUSS_LEGENDRE,
                    getGaussPointNum1D(),
                    point0.x, point0.y, point1.x, point1.y);

            for (int degree = 0; degree <= getK(); ++degree) {
                double boundary_moment = 0.0;
                for (const vem::QuadPoint& point : points) {
                    double xi = point.x[0];
                    double eta = point.x[1];
                    Darcy::Vector2D velocity_hat =
                        problem_->computational_solution_u(xi, eta);
                    double normal_flux =
                        velocity_hat.x * normal.x +
                        velocity_hat.y * normal.y;
                    BasisPoint quadrature_point(xi, eta);
                    double monomial = edge_basis.evalEdgeMonomial(
                        degree, point0, point1, quadrature_point);

                    // computational_solution_u 已经执行逆 Piola 变换，
                    // 法向通量守恒关系中不再额外乘物理曲边 Jacobian。
                    boundary_moment +=
                        normal_flux * monomial * point.w;
                }

                PetscInt global_dof =
                    degree * mesh_data->num_edges + global_edge;
                PetscScalar value = boundary_moment;
                PetscCall(VecSetValue(
                    raw_boundary_values, global_dof,
                    value, INSERT_VALUES));
                is_boundary_dof[static_cast<std::size_t>(global_dof)] =
                    PETSC_TRUE;
            }
        }
    }

    PetscCall(VecAssemblyBegin(raw_boundary_values));
    PetscCall(VecAssemblyEnd(raw_boundary_values));

    std::vector<PetscInt> free_indices;
    free_indices.reserve(static_cast<std::size_t>(total_dof));
    for (PetscInt index = 0; index < total_dof; ++index) {
        if (!is_boundary_dof[static_cast<std::size_t>(index)]) {
            free_indices.push_back(index);
        }
    }

    // 完整右端修正：F_free 来自 F - K * U_boundary。
    Vec matrix_times_boundary = PETSC_NULLPTR;
    Vec adjusted_rhs = PETSC_NULLPTR;
    PetscCall(VecDuplicate(global_rhs, &matrix_times_boundary));
    PetscCall(VecDuplicate(global_rhs, &adjusted_rhs));
    PetscCall(MatMult(
        global_matrix, raw_boundary_values,
        matrix_times_boundary));
    PetscCall(VecCopy(global_rhs, adjusted_rhs));
    PetscCall(VecAXPY(adjusted_rhs, -1.0, matrix_times_boundary));

    PetscCall(ISCreateGeneral(
        PETSC_COMM_SELF,
        static_cast<PetscInt>(free_indices.size()),
        free_indices.data(), PETSC_COPY_VALUES, &free_dofs));

    Mat reduced_raw = PETSC_NULLPTR;
    PetscErrorCode error = MatCreateSubMatrix(
        global_matrix, free_dofs, free_dofs,
        MAT_INITIAL_MATRIX, &reduced_raw);
    if (error != 0) {
        VecDestroy(&matrix_times_boundary);
        VecDestroy(&adjusted_rhs);
        return error;
    }
    reduced_matrix = AutoPetscMat(new Mat(reduced_raw));

    Vec rhs_view = PETSC_NULLPTR;
    error = VecGetSubVector(adjusted_rhs, free_dofs, &rhs_view);
    if (error != 0) {
        VecDestroy(&matrix_times_boundary);
        VecDestroy(&adjusted_rhs);
        return error;
    }
    Vec reduced_rhs_raw = PETSC_NULLPTR;
    error = VecDuplicate(rhs_view, &reduced_rhs_raw);
    if (error == 0) {
        error = VecCopy(rhs_view, reduced_rhs_raw);
    }
    PetscErrorCode restore_error =
        VecRestoreSubVector(adjusted_rhs, free_dofs, &rhs_view);
    VecDestroy(&matrix_times_boundary);
    VecDestroy(&adjusted_rhs);
    if (error != 0) {
        if (reduced_rhs_raw != PETSC_NULLPTR) {
            VecDestroy(&reduced_rhs_raw);
        }
        return error;
    }
    if (restore_error != 0) {
        VecDestroy(&reduced_rhs_raw);
        return restore_error;
    }
    reduced_rhs = AutoPetscVec(new Vec(reduced_rhs_raw));
    return 0;
}

PetscErrorCode DarcySolver::solveWithDirichletBC() {
    solved_ = false;
    solution_.reset();

    AutoPetscMat reduced_matrix;
    AutoPetscVec reduced_rhs;
    AutoPetscVec boundary_values;
    IS free_dofs = PETSC_NULLPTR;
    PetscInt total_dof = 0;

    PetscErrorCode error = applyNormalFluxBoundaryConditions(
        reduced_matrix, reduced_rhs, free_dofs,
        boundary_values, total_dof);
    if (error != 0) {
        if (free_dofs != PETSC_NULLPTR) {
            ISDestroy(&free_dofs);
        }
        return error;
    }

    PetscInt free_size = 0;
    error = VecGetSize(vem::get_raw(reduced_rhs), &free_size);
    if (error != 0) {
        ISDestroy(&free_dofs);
        return error;
    }

    AutoPetscVec free_solution = vem::create_vector(free_size);
    error = VecSet(vem::get_raw(free_solution), 0.0);
    if (error != 0) {
        ISDestroy(&free_dofs);
        return error;
    }
    error = VecAssemblyBegin(vem::get_raw(free_solution));
    if (error == 0) {
        error = VecAssemblyEnd(vem::get_raw(free_solution));
    }
    if (error != 0) {
        ISDestroy(&free_dofs);
        return error;
    }

    PetscBool converged = PETSC_FALSE;
    error = vem::solve_linear_system(
        reduced_matrix, reduced_rhs, free_solution,
        &converged, true);
    if (error != 0 || !converged) {
        ISDestroy(&free_dofs);
        return error != 0 ? error : PETSC_ERR_NOT_CONVERGED;
    }

    Vec solution_raw = PETSC_NULLPTR;
    error = VecDuplicate(vem::get_raw(boundary_values), &solution_raw);
    if (error != 0) {
        ISDestroy(&free_dofs);
        return error;
    }
    solution_ = AutoPetscVec(new Vec(solution_raw));
    error = VecCopy(
        vem::get_raw(boundary_values), vem::get_raw(solution_));
    if (error != 0) {
        ISDestroy(&free_dofs);
        solution_.reset();
        return error;
    }

    const PetscInt* free_indices = nullptr;
    PetscInt free_index_count = 0;
    error = ISGetLocalSize(free_dofs, &free_index_count);
    if (error == 0) {
        error = ISGetIndices(free_dofs, &free_indices);
    }
    if (error != 0) {
        ISDestroy(&free_dofs);
        solution_.reset();
        return error;
    }

    const PetscScalar* free_values = nullptr;
    error = VecGetArrayRead(
        vem::get_raw(free_solution), &free_values);
    if (error == 0) {
        error = VecSetValues(
            vem::get_raw(solution_), free_index_count,
            free_indices, free_values, INSERT_VALUES);
    }
    PetscErrorCode restore_vector_error = VecRestoreArrayRead(
        vem::get_raw(free_solution), &free_values);
    PetscErrorCode restore_indices_error =
        ISRestoreIndices(free_dofs, &free_indices);
    ISDestroy(&free_dofs);

    if (error != 0) {
        solution_.reset();
        return error;
    }
    if (restore_vector_error != 0) {
        solution_.reset();
        return restore_vector_error;
    }
    if (restore_indices_error != 0) {
        solution_.reset();
        return restore_indices_error;
    }

    error = VecAssemblyBegin(vem::get_raw(solution_));
    if (error == 0) {
        error = VecAssemblyEnd(vem::get_raw(solution_));
    }
    if (error != 0) {
        solution_.reset();
        return error;
    }

    PetscInt solution_size = 0;
    error = VecGetSize(vem::get_raw(solution_), &solution_size);
    if (error != 0 || solution_size != total_dof) {
        solution_.reset();
        return error != 0 ? error : PETSC_ERR_ARG_SIZ;
    }

    solved_ = true;
    return 0;
}

bool DarcySolver::solve() {
    if (!initialized_) {
        return false;
    }
    PetscErrorCode error = assembleStiffnessMatrix();
    if (error != 0) {
        return false;
    }
    return solveWithDirichletBC() == 0;
}

// ========================================================================
// GPU 版本求解
// ========================================================================

PetscErrorCode DarcySolver::solveWithDirichletBC_GPU() {
    solved_ = false;
    solution_.reset();

    AutoPetscMat reduced_matrix;
    AutoPetscVec reduced_rhs;
    AutoPetscVec boundary_values;
    IS free_dofs = PETSC_NULLPTR;
    PetscInt total_dof = 0;

    PetscErrorCode error = applyNormalFluxBoundaryConditions(
        reduced_matrix, reduced_rhs, free_dofs,
        boundary_values, total_dof);
    if (error != 0) {
        if (free_dofs != PETSC_NULLPTR) {
            ISDestroy(&free_dofs);
        }
        return error;
    }

    PetscInt free_size = 0;
    error = VecGetSize(vem::get_raw(reduced_rhs), &free_size);
    if (error != 0) {
        ISDestroy(&free_dofs);
        return error;
    }

    AutoPetscVec free_solution = vem::create_vector(free_size);
    error = VecSet(vem::get_raw(free_solution), 0.0);
    if (error != 0) {
        ISDestroy(&free_dofs);
        return error;
    }
    error = VecAssemblyBegin(vem::get_raw(free_solution));
    if (error == 0) {
        error = VecAssemblyEnd(vem::get_raw(free_solution));
    }
    if (error != 0) {
        ISDestroy(&free_dofs);
        return error;
    }

    // 使用 GPU GMRES + BJACOBI 求解
    PetscBool converged = PETSC_FALSE;
    error = vem::solve_linear_system_cuda(
        reduced_matrix, reduced_rhs, free_solution,
        &converged, true);
    if (error != 0 || !converged) {
        ISDestroy(&free_dofs);
        return error != 0 ? error : PETSC_ERR_NOT_CONVERGED;
    }

    Vec solution_raw = PETSC_NULLPTR;
    error = VecDuplicate(vem::get_raw(boundary_values), &solution_raw);
    if (error != 0) {
        ISDestroy(&free_dofs);
        return error;
    }
    solution_ = AutoPetscVec(new Vec(solution_raw));
    error = VecCopy(
        vem::get_raw(boundary_values), vem::get_raw(solution_));
    if (error != 0) {
        ISDestroy(&free_dofs);
        solution_.reset();
        return error;
    }

    const PetscInt* free_indices = nullptr;
    PetscInt free_index_count = 0;
    error = ISGetLocalSize(free_dofs, &free_index_count);
    if (error == 0) {
        error = ISGetIndices(free_dofs, &free_indices);
    }
    if (error != 0) {
        ISDestroy(&free_dofs);
        solution_.reset();
        return error;
    }

    const PetscScalar* free_values = nullptr;
    error = VecGetArrayRead(
        vem::get_raw(free_solution), &free_values);
    if (error == 0) {
        error = VecSetValues(
            vem::get_raw(solution_), free_index_count,
            free_indices, free_values, INSERT_VALUES);
    }
    PetscErrorCode restore_vector_error = VecRestoreArrayRead(
        vem::get_raw(free_solution), &free_values);
    PetscErrorCode restore_indices_error =
        ISRestoreIndices(free_dofs, &free_indices);
    ISDestroy(&free_dofs);

    if (error != 0) {
        solution_.reset();
        return error;
    }
    if (restore_vector_error != 0) {
        solution_.reset();
        return restore_vector_error;
    }
    if (restore_indices_error != 0) {
        solution_.reset();
        return restore_indices_error;
    }

    error = VecAssemblyBegin(vem::get_raw(solution_));
    if (error == 0) {
        error = VecAssemblyEnd(vem::get_raw(solution_));
    }
    if (error != 0) {
        solution_.reset();
        return error;
    }

    PetscInt solution_size = 0;
    error = VecGetSize(vem::get_raw(solution_), &solution_size);
    if (error != 0 || solution_size != total_dof) {
        solution_.reset();
        return error != 0 ? error : PETSC_ERR_ARG_SIZ;
    }

    solved_ = true;
    return 0;
}

bool DarcySolver::solveGPU() {
    if (!initialized_) {
        return false;
    }
    PetscErrorCode error = assembleStiffnessMatrix();
    if (error != 0) {
        return false;
    }
    return solveWithDirichletBC_GPU() == 0;
}

double DarcySolver::computeL2ErrorPressure(int gauss_point_num) {
    if (!solved_ || !solution_ || vem::get_raw(solution_) == PETSC_NULLPTR) {
        throw std::runtime_error(
            "pressure error requires a completed numerical solution");
    }

    vem::GaussQuadrature quadrature;
    quadrature.get_triangle_reference_points(gauss_point_num);
    const MeshData* mesh_data = getMeshData();
    Basis basis(getK());
    int pressure_dofs = getDofPerElementConc();
    double error_squared = 0.0;

    for (int elem = 0; elem < mesh_data->num_cells; ++elem) {
        BasisPoint center(mesh_data->cell_centroid_x[elem],
                          mesh_data->cell_centroid_y[elem]);
        double diameter = mesh_data->cell_diameter[elem];
        std::vector<double> coefficients(pressure_dofs, 0.0);
        for (int index = 0; index < pressure_dofs; ++index) {
            PetscInt global_index =
                getTotalDofFlux() + elem * pressure_dofs + index;
            PetscScalar value = 0.0;
            PetscErrorCode error = VecGetValues(
                vem::get_raw(solution_), 1, &global_index, &value);
            if (error != 0) {
                throw std::runtime_error(
                    "failed to read pressure solution coefficient");
            }
            coefficients[index] = PetscRealPart(value);
        }

        vem::PolygonTriangulator triangulator;
        vem::TriangulatedElement element =
            triangulator.triangulateElement(*mesh_data, elem);
        for (const vem::Triangle& triangle : element.triangles) {
            std::vector<vem::QuadPoint> points = quadrature.get_triangle_points(
                triangle.vertices, gauss_point_num);
            for (const vem::QuadPoint& point : points) {
                BasisPoint quadrature_point(point.x[0], point.x[1]);
                double pressure_numerical = 0.0;
                for (int index = 0; index < pressure_dofs; ++index) {
                    pressure_numerical += coefficients[index] *
                        basis.evalMonomial2D(
                            index, center, diameter, quadrature_point);
                }
                double pressure_exact = problem_->computational_solution_p(
                    point.x[0], point.x[1]);
                double difference = pressure_numerical - pressure_exact;
                error_squared += difference * difference * point.w;
            }
        }
    }

    l2_error_pressure_ = std::sqrt(error_squared);
    return l2_error_pressure_;
}

double DarcySolver::computeL2ErrorFlux(int gauss_point_num) {
    if (!solved_ || !solution_ || vem::get_raw(solution_) == PETSC_NULLPTR) {
        throw std::runtime_error(
            "flux error requires a completed numerical solution");
    }

    vem::GaussQuadrature quadrature;
    quadrature.get_triangle_reference_points(gauss_point_num);
    const MeshData* mesh_data = getMeshData();
    Basis basis(getK());
    int polynomial_dimension = basis.getDimHdiv();
    double error_squared = 0.0;

    for (int elem = 0; elem < mesh_data->num_cells; ++elem) {
        int flux_dofs = getDofPerElementFlux(elem);
        int edge_count = mesh_data->nodes_per_cell[elem];
        int edge_dofs = edge_count * (getK() + 1);
        std::vector<double> local_flux(flux_dofs, 0.0);
        std::vector<PetscInt> global_dofs = getLocalToGlobalDofs(elem);
        int cell_node_start = mesh_data->cell_node_indices[elem];
        int cell_edge_start = mesh_data->cell_edge_indices[elem];

        for (int local_index = 0; local_index < flux_dofs; ++local_index) {
            PetscInt global_index = global_dofs[local_index];
            PetscScalar value = 0.0;
            PetscErrorCode error = VecGetValues(
                vem::get_raw(solution_), 1, &global_index, &value);
            if (error != 0) {
                throw std::runtime_error(
                    "failed to read flux solution coefficient");
            }

            double direction = 1.0;
            if (local_index < edge_dofs) {
                int degree = local_index / edge_count;
                int local_edge = local_index % edge_count;
                int global_edge =
                    mesh_data->global_edges[cell_edge_start + local_edge];
                if (mesh_data->edge_occurrence[global_edge] == 2) {
                    int local_node0 = mesh_data->cell_nodes[
                        cell_node_start + local_edge];
                    int local_node1 = mesh_data->cell_nodes[
                        cell_node_start + (local_edge + 1) % edge_count];
                    int global_node0 = mesh_data->edge_endpoints[2 * global_edge];
                    int global_node1 =
                        mesh_data->edge_endpoints[2 * global_edge + 1];
                    bool reverse = local_node0 == global_node1 &&
                                   local_node1 == global_node0;
                    if (reverse && degree % 2 == 0) {
                        direction = -1.0;
                    }
                }
            }
            local_flux[local_index] = direction * PetscRealPart(value);
        }

        AutoPetscMat G = getMatrixG(elem);
        AutoPetscMat W = getMatrixW(elem);
        AutoPetscMat H = getMatrixH(elem);
        AutoPetscMat H_star = getMatrixH_star(elem);
        AutoPetscMat B = getMatrixB(elem, H_star, H, W);
        AutoPetscMat projection = getMatrixL2Proj_ploy(G, B);

        AutoPetscVec local_flux_vector = vem::create_vector(flux_dofs);
        for (int index = 0; index < flux_dofs; ++index) {
            PetscInt petsc_index = index;
            PetscScalar value = local_flux[index];
            PetscErrorCode error = VecSetValue(
                vem::get_raw(local_flux_vector), petsc_index,
                value, INSERT_VALUES);
            if (error != 0) {
                throw std::runtime_error("failed to set local flux dof");
            }
        }
        VecAssemblyBegin(vem::get_raw(local_flux_vector));
        VecAssemblyEnd(vem::get_raw(local_flux_vector));

        AutoPetscVec polynomial_coefficients =
            vem::create_vector(polynomial_dimension);
        PetscErrorCode error = MatMult(
            vem::get_raw(projection),
            vem::get_raw(local_flux_vector),
            vem::get_raw(polynomial_coefficients));
        if (error != 0) {
            throw std::runtime_error("failed to project numerical flux");
        }

        std::vector<double> coefficients(polynomial_dimension, 0.0);
        for (int index = 0; index < polynomial_dimension; ++index) {
            PetscInt petsc_index = index;
            PetscScalar value = 0.0;
            error = VecGetValues(
                vem::get_raw(polynomial_coefficients), 1,
                &petsc_index, &value);
            if (error != 0) {
                throw std::runtime_error(
                    "failed to read projected flux coefficient");
            }
            coefficients[index] = PetscRealPart(value);
        }

        BasisPoint center(mesh_data->cell_centroid_x[elem],
                          mesh_data->cell_centroid_y[elem]);
        double diameter = mesh_data->cell_diameter[elem];
        vem::PolygonTriangulator triangulator;
        vem::TriangulatedElement element =
            triangulator.triangulateElement(*mesh_data, elem);
        for (const vem::Triangle& triangle : element.triangles) {
            std::vector<vem::QuadPoint> points = quadrature.get_triangle_points(
                triangle.vertices, gauss_point_num);
            for (const vem::QuadPoint& point : points) {
                BasisPoint quadrature_point(point.x[0], point.x[1]);
                BasisVector flux_numerical(0.0, 0.0);
                for (int index = 0; index < polynomial_dimension; ++index) {
                    BasisVector value = basis.evalHdivBasis(
                        index, center, diameter, quadrature_point);
                    flux_numerical.x += coefficients[index] * value.x;
                    flux_numerical.y += coefficients[index] * value.y;
                }
                Darcy::Vector2D flux_exact =
                    problem_->computational_solution_u(
                        point.x[0], point.x[1]);
                double difference_x = flux_numerical.x - flux_exact.x;
                double difference_y = flux_numerical.y - flux_exact.y;
                error_squared += (difference_x * difference_x +
                                  difference_y * difference_y) * point.w;
            }
        }
    }

    l2_error_flux_ = std::sqrt(error_squared);
    return l2_error_flux_;
}

double DarcySolver::vemHdiv_gm_gm_Kappa(int mesh_idx,
                                         int ploy_mk1,
                                         int ploy_mk2) {
    const MeshData* mesh_data = getMeshData();
    if (mesh_idx < 0 || mesh_idx >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }

    Basis basis(getK());
    int dimension = basis.getDimHdiv();
    if (ploy_mk1 < 0 || ploy_mk1 >= dimension ||
        ploy_mk2 < 0 || ploy_mk2 >= dimension) {
        throw std::out_of_range("H(div) polynomial index out of range");
    }

    BasisPoint center(mesh_data->cell_centroid_x[mesh_idx],
                      mesh_data->cell_centroid_y[mesh_idx]);
    double diameter = mesh_data->cell_diameter[mesh_idx];

    vem::PolygonTriangulator triangulator;
    vem::TriangulatedElement element =
        triangulator.triangulateElement(*mesh_data, mesh_idx);
    vem::GaussQuadrature quadrature;

    double result = 0.0;
    for (const vem::Triangle& triangle : element.triangles) {
        std::vector<vem::QuadPoint> points = quadrature.get_triangle_points(
            triangle.vertices, getGaussPointNum2D());
        for (const vem::QuadPoint& point : points) {
            double xi = point.x[0];
            double eta = point.x[1];
            BasisPoint quadrature_point(xi, eta);

            BasisVector first = basis.evalHdivBasis(
                ploy_mk1, center, diameter, quadrature_point);
            BasisVector second = basis.evalHdivBasis(
                ploy_mk2, center, diameter, quadrature_point);

            Darcy::Tensor2D kappa_hat =
                problem_->computational_kappa(xi, eta);
            Darcy::Tensor2D inverse_kappa_hat = inverseTensor(kappa_hat);
            BasisVector weighted_second = multiply(inverse_kappa_hat, second);

            result += (first.x * weighted_second.x +
                       first.y * weighted_second.y) * point.w;
        }
    }
    return result;
}

double DarcySolver::computeStabilizationScale(int mesh_idx) {
    const MeshData* mesh_data = getMeshData();
    if (mesh_idx < 0 || mesh_idx >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }

    vem::PolygonTriangulator triangulator;
    vem::TriangulatedElement element =
        triangulator.triangulateElement(*mesh_data, mesh_idx);
    double area = element.total_area;
    if (!std::isfinite(area) || area <= 0.0) {
        throw std::runtime_error("element area must be positive and finite");
    }
    vem::GaussQuadrature quadrature;

    Darcy::Tensor2D integral(0.0, 0.0, 0.0, 0.0);
    for (const vem::Triangle& triangle : element.triangles) {
        std::vector<vem::QuadPoint> points = quadrature.get_triangle_points(
            triangle.vertices, getGaussPointNum2D());
        for (const vem::QuadPoint& point : points) {
            Darcy::Tensor2D kappa_hat =
                problem_->computational_kappa(point.x[0], point.x[1]);
            Darcy::Tensor2D inverse_kappa_hat = inverseTensor(kappa_hat);
            integral.xx += inverse_kappa_hat.xx * point.w;
            integral.xy += inverse_kappa_hat.xy * point.w;
            integral.yx += inverse_kappa_hat.yx * point.w;
            integral.yy += inverse_kappa_hat.yy * point.w;
        }
    }

    Darcy::Tensor2D average(
        integral.xx / area,
        integral.xy / area,
        integral.yx / area,
        integral.yy / area);
    double scale = std::sqrt(
        average.xx * average.xx + average.xy * average.xy +
        average.yx * average.yx + average.yy * average.yy);
    if (!std::isfinite(scale) || scale <= 0.0) {
        throw std::runtime_error(
            "stabilization scale must be positive and finite");
    }
    return scale;
}

AutoPetscMat DarcySolver::HdivMatrixK_as(
    int mesh_idx,
    const AutoPetscMat& L2Proj_basis) {
    if (!L2Proj_basis ||
        vem::get_raw(L2Proj_basis) == PETSC_NULLPTR) {
        throw std::invalid_argument(
            "L2Proj_basis must be a valid matrix");
    }

    PetscInt rows = 0;
    PetscInt columns = 0;
    PetscErrorCode error = MatGetSize(
        vem::get_raw(L2Proj_basis), &rows, &columns);
    if (error != 0) {
        throw std::runtime_error(
            "MatGetSize failed while assembling K_as");
    }
    if (rows != columns) {
        throw std::invalid_argument(
            "L2Proj_basis must be square");
    }
    if (rows != getDofPerElementFlux(mesh_idx)) {
        throw std::invalid_argument(
            "L2Proj_basis dimension does not match element flux dofs");
    }

    AutoPetscMat identity_minus_projection =
        vem::scale_matrix(L2Proj_basis, -1.0);
    Mat raw_complement = vem::get_raw(identity_minus_projection);
    error = MatShift(raw_complement, 1.0);
    if (error != 0) {
        throw std::runtime_error("MatShift failed while assembling K_as");
    }
    assembleMatrix(raw_complement);

    AutoPetscMat complement_transpose =
        vem::transpose_matrix(identity_minus_projection);
    AutoPetscMat stabilization = vem::multiply_matrices(
        complement_transpose, identity_minus_projection);
    return vem::scale_matrix(
        stabilization, computeStabilizationScale(mesh_idx));
}

double DarcySolver::vemHdiv_source_mk(int mesh_idx, int ploy_mk) {
    const MeshData* mesh_data = getMeshData();
    if (mesh_idx < 0 || mesh_idx >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }

    Basis basis(getK());
    int pressure_dimension = basis.getDimMonomial(2);
    if (ploy_mk < 0 || ploy_mk >= pressure_dimension) {
        throw std::out_of_range("pressure polynomial index out of range");
    }

    BasisPoint center(mesh_data->cell_centroid_x[mesh_idx],
                      mesh_data->cell_centroid_y[mesh_idx]);
    double diameter = mesh_data->cell_diameter[mesh_idx];
    vem::PolygonTriangulator triangulator;
    vem::TriangulatedElement element =
        triangulator.triangulateElement(*mesh_data, mesh_idx);
    vem::GaussQuadrature quadrature;

    double result = 0.0;
    for (const vem::Triangle& triangle : element.triangles) {
        std::vector<vem::QuadPoint> points = quadrature.get_triangle_points(
            triangle.vertices, getGaussPointNum2D());
        for (const vem::QuadPoint& point : points) {
            BasisPoint quadrature_point(point.x[0], point.x[1]);
            // computational_source_g 已经包含 det(J)*(g o F)，这里不能
            // 再额外乘一次物理映射 Jacobian。
            double source_hat = problem_->computational_source_g(
                point.x[0], point.x[1]);
            double pressure_basis = basis.evalMonomial2D(
                ploy_mk, center, diameter, quadrature_point);
            result += source_hat * pressure_basis * point.w;
        }
    }
    return result;
}

AutoPetscVec DarcySolver::HdivVecRhs(int mesh_idx) {
    const MeshData* mesh_data = getMeshData();
    if (mesh_idx < 0 || mesh_idx >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }

    int flux_dofs = getDofPerElementFlux(mesh_idx);
    int pressure_dofs = getDofPerElementConc();
    int local_dofs = flux_dofs + pressure_dofs + 1;
    AutoPetscVec rhs = vem::create_vector(local_dofs);
    Vec raw_rhs = vem::get_raw(rhs);

    PetscErrorCode error = VecSet(raw_rhs, 0.0);
    if (error != 0) {
        throw std::runtime_error("VecSet failed while assembling local RHS");
    }
    for (int pressure_index = 0;
         pressure_index < pressure_dofs;
         ++pressure_index) {
        PetscInt vector_index = flux_dofs + pressure_index;
        PetscScalar value = vemHdiv_source_mk(
            mesh_idx, pressure_index);
        error = VecSetValue(
            raw_rhs, vector_index, value, INSERT_VALUES);
        if (error != 0) {
            throw std::runtime_error(
                "VecSetValue failed while assembling local RHS");
        }
    }

    error = VecAssemblyBegin(raw_rhs);
    if (error == 0) {
        error = VecAssemblyEnd(raw_rhs);
    }
    if (error != 0) {
        throw std::runtime_error("local RHS vector assembly failed");
    }
    return rhs;
}

double DarcySolver::vemHdiv_mk(int mesh_idx, int ploy_mk) {
    const MeshData* mesh_data = getMeshData();
    if (mesh_idx < 0 || mesh_idx >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }

    Basis basis(getK());
    int pressure_dimension = basis.getDimMonomial(2);
    if (ploy_mk < 0 || ploy_mk >= pressure_dimension) {
        throw std::out_of_range("pressure polynomial index out of range");
    }

    BasisPoint center(mesh_data->cell_centroid_x[mesh_idx],
                      mesh_data->cell_centroid_y[mesh_idx]);
    double diameter = mesh_data->cell_diameter[mesh_idx];
    vem::PolygonTriangulator triangulator;
    vem::TriangulatedElement element =
        triangulator.triangulateElement(*mesh_data, mesh_idx);
    vem::GaussQuadrature quadrature;

    double result = 0.0;
    for (const vem::Triangle& triangle : element.triangles) {
        std::vector<vem::QuadPoint> points = quadrature.get_triangle_points(
            triangle.vertices, getGaussPointNum2D());
        for (const vem::QuadPoint& point : points) {
            BasisPoint quadrature_point(point.x[0], point.x[1]);
            result += basis.evalMonomial2D(
                ploy_mk, center, diameter, quadrature_point) * point.w;
        }
    }
    return result;
}

AutoPetscVec DarcySolver::HdivVecLagrange(int mesh_idx) {
    const MeshData* mesh_data = getMeshData();
    if (mesh_idx < 0 || mesh_idx >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }

    int flux_dofs = getDofPerElementFlux(mesh_idx);
    int pressure_dofs = getDofPerElementConc();
    int local_dofs = flux_dofs + pressure_dofs + 1;
    AutoPetscVec lagrange = vem::create_vector(local_dofs);
    Vec raw_lagrange = vem::get_raw(lagrange);

    PetscErrorCode error = VecSet(raw_lagrange, 0.0);
    if (error != 0) {
        throw std::runtime_error(
            "VecSet failed while assembling Lagrange vector");
    }

    // 压力零积分约束：∫_E p dx = 0。
    // 拉回计算区域后为 ∫_Ẽ p̂ det(J) dẼ = 0。
    // 因此 Lagrange 向量在压力自由度位置的值为：
    //   L_r = ∫_Ẽ m_r det(J) dẼ。
    // 速度块和末尾 Lagrange 乘子位置为零。
    Basis basis(getK());
    BasisPoint center(mesh_data->cell_centroid_x[mesh_idx],
                      mesh_data->cell_centroid_y[mesh_idx]);
    double diameter = mesh_data->cell_diameter[mesh_idx];
    vem::PolygonTriangulator triangulator;
    vem::TriangulatedElement element =
        triangulator.triangulateElement(*mesh_data, mesh_idx);
    vem::GaussQuadrature quadrature;

    for (int pressure_index = 0;
         pressure_index < pressure_dofs;
         ++pressure_index) {
        double value = 0.0;
        for (const vem::Triangle& triangle : element.triangles) {
            std::vector<vem::QuadPoint> points =
                quadrature.get_triangle_points(
                    triangle.vertices, getGaussPointNum2D());
            for (const vem::QuadPoint& point : points) {
                double xi = point.x[0];
                double eta = point.x[1];
                BasisPoint quadrature_point(xi, eta);
                double detJ = getProblem().get_mapping().jacobian_det(
                    xi, eta);
                value += basis.evalMonomial2D(
                    pressure_index, center, diameter,
                    quadrature_point) * detJ * point.w;
            }
        }

        PetscInt vector_index = flux_dofs + pressure_index;
        error = VecSetValue(
            raw_lagrange, vector_index, value, INSERT_VALUES);
        if (error != 0) {
            throw std::runtime_error(
                "VecSetValue failed while assembling Lagrange vector");
        }
    }

    error = VecAssemblyBegin(raw_lagrange);
    if (error == 0) {
        error = VecAssemblyEnd(raw_lagrange);
    }
    if (error != 0) {
        throw std::runtime_error(
            "Lagrange vector assembly failed");
    }
    return lagrange;
}

AutoPetscMat DarcySolver::HdivMatrixG_Kappa(int mesh_idx) {
    const MeshData* mesh_data = getMeshData();
    if (mesh_idx < 0 || mesh_idx >= mesh_data->num_cells) {
        throw std::out_of_range("element index out of range");
    }

    int dimension = Basis(getK()).getDimHdiv();
    AutoPetscMat matrix = vem::create_dense_matrix(dimension, dimension);
    Mat raw_matrix = vem::get_raw(matrix);

    for (int row = 0; row < dimension; ++row) {
        for (int column = 0; column < dimension; ++column) {
            double value = vemHdiv_gm_gm_Kappa(
                mesh_idx, row, column);
            setMatrixValue(raw_matrix, row, column, value);
        }
    }

    assembleMatrix(raw_matrix);
    return matrix;
}

AutoPetscMat DarcySolver::HdivMatrixK_ac(
    const AutoPetscMat& G_Kappa,
    const AutoPetscMat& L2Proj_ploy) {
    if (!G_Kappa || vem::get_raw(G_Kappa) == PETSC_NULLPTR ||
        !L2Proj_ploy || vem::get_raw(L2Proj_ploy) == PETSC_NULLPTR) {
        throw std::invalid_argument(
            "G_Kappa and L2Proj_ploy must be valid matrices");
    }

    PetscInt gram_rows = 0;
    PetscInt gram_cols = 0;
    PetscInt projection_rows = 0;
    PetscInt projection_cols = 0;
    PetscErrorCode error =
        MatGetSize(vem::get_raw(G_Kappa), &gram_rows, &gram_cols);
    if (error == 0) {
        error = MatGetSize(vem::get_raw(L2Proj_ploy),
                           &projection_rows, &projection_cols);
    }
    if (error != 0) {
        throw std::runtime_error("MatGetSize failed while assembling K_ac");
    }
    if (gram_rows != gram_cols || gram_cols != projection_rows) {
        throw std::invalid_argument(
            "matrix dimensions do not satisfy Pi^T G_Kappa Pi");
    }

    AutoPetscMat weighted_projection =
        vem::multiply_matrices(G_Kappa, L2Proj_ploy);
    AutoPetscMat projection_transpose =
        vem::transpose_matrix(L2Proj_ploy);
    return vem::multiply_matrices(
        projection_transpose, weighted_projection);
}
