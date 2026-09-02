#include "HdivMatrix.h"

#include <cmath>

namespace {

using Basis = vem::basis::BasisFunctionPloy;
using BasisPoint = vem::basis::Point2D;
using BasisVector = vem::basis::Vector2D;

BasisPoint makeBasisPoint(double x, double y) {
    return BasisPoint(x, y);
}

void assemble(Mat matrix) {
    MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY);
}

void setValue(Mat matrix, PetscInt row, PetscInt col, double value,
              InsertMode mode = INSERT_VALUES) {
    PetscScalar scalar = value;
    MatSetValues(matrix, 1, &row, 1, &col, &scalar, mode);
}

}  // namespace

std::vector<int> HdivMatrix::getElementNodes(int mesh_idx) const {
    if (mesh_idx < 0 || mesh_idx >= mesh_data_->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    int start_idx = mesh_data_->cell_node_indices[mesh_idx];
    int num_nodes = mesh_data_->nodes_per_cell[mesh_idx];
    return std::vector<int>(mesh_data_->cell_nodes.begin() + start_idx,
                            mesh_data_->cell_nodes.begin() + start_idx + num_nodes);
}

Point2D HdivMatrix::computeEdgeNormal(int node1, int node2) const {
    if (node1 < 0 || node1 >= mesh_data_->num_nodes ||
        node2 < 0 || node2 >= mesh_data_->num_nodes) {
        throw std::out_of_range("edge node index out of range");
    }

    double dx = mesh_data_->node_coords[2 * node2] -
                mesh_data_->node_coords[2 * node1];
    double dy = mesh_data_->node_coords[2 * node2 + 1] -
                mesh_data_->node_coords[2 * node1 + 1];
    double length = std::sqrt(dx * dx + dy * dy);
    if (!(length > 1e-14)) {
        throw std::invalid_argument("edge endpoints define a degenerate edge");
    }

    dx /= length;
    dy /= length;
    return Point2D(dy, -dx);
}

void HdivMatrix::initDegrees() {
    if (k_ < 0 || k_ > 4) {
        throw std::invalid_argument("polynomial degree k must be between 0 and 4");
    }
    if (k_ == 3 && gauss_point_num_ != 12) {
        throw std::invalid_argument(
            "polynomial degree k=3 requires 12 triangle quadrature points");
    }
    // 二维参数表示三角形积分点数，一维参数表示 Gauss-Legendre 点数。
    vem::GaussQuadrature quadrature;
    quadrature.get_triangle_reference_points(gauss_point_num_);
    quadrature.get_1D_points(vem::QuadratureType::GAUSS_LEGENDRE,
                             gauss_point_num_1d_);

    Basis basis(k_);
    dim_grad_mk_ = Basis::dimPk(k_, 2) - 1;
    dim_curl_mk_ = Basis::dimPk(k_, 2) - (k_ + 1);
    int internal_dof = dim_grad_mk_ + dim_curl_mk_;

    dof_per_element_flux_.clear();
    dof_per_element_flux_.reserve(mesh_data_->num_cells);
    for (int cell = 0; cell < mesh_data_->num_cells; ++cell) {
        int num_edges = mesh_data_->nodes_per_cell[cell];
        dof_per_element_flux_.push_back((k_ + 1) * num_edges + internal_dof);
    }

    dof_per_element_conc_ = Basis::dimPk(k_, 2);
    total_dof_flux_ = mesh_data_->num_edges * (k_ + 1) +
                      mesh_data_->num_cells * internal_dof;
    total_dof_conc_ = dof_per_element_conc_ * mesh_data_->num_cells;
}

double HdivMatrix::Delta_function(int i, int j) {
    return i == j ? 1.0 : 0.0;
}

double HdivMatrix::vemHdiv_dof_g_value(int mesh_idx, int local_dof_idx,
                                        int ploy_mk) {
    if (mesh_idx < 0 || mesh_idx >= mesh_data_->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    int num_edges = mesh_data_->nodes_per_cell[mesh_idx];
    int edge_dof = (k_ + 1) * num_edges;
    int dof_count = dof_per_element_flux_[mesh_idx];
    if (local_dof_idx < 0 || local_dof_idx >= dof_count) {
        throw std::out_of_range("local flux dof index out of range");
    }

    std::vector<int> element_nodes = getElementNodes(mesh_idx);
    BasisPoint xD = makeBasisPoint(mesh_data_->cell_centroid_x[mesh_idx],
                                   mesh_data_->cell_centroid_y[mesh_idx]);
    double hD = mesh_data_->cell_diameter[mesh_idx];
    Basis basis(k_);
    vem::GaussQuadrature quadrature;

    double result = 0.0;
    if (local_dof_idx < edge_dof) {
        int edge_index = local_dof_idx % num_edges;
        int node1 = element_nodes[edge_index];
        int node2 = element_nodes[(edge_index + 1) % num_edges];
        int ploy_exp = local_dof_idx / num_edges;
        Point2D normal = computeEdgeNormal(node1, node2);
        double x1 = mesh_data_->node_coords[2 * node1];
        double y1 = mesh_data_->node_coords[2 * node1 + 1];
        double x2 = mesh_data_->node_coords[2 * node2];
        double y2 = mesh_data_->node_coords[2 * node2 + 1];
        BasisPoint point1 = makeBasisPoint(x1, y1);
        BasisPoint point2 = makeBasisPoint(x2, y2);

        std::vector<vem::QuadPoint> points = quadrature.get_line_2D_points(
            vem::QuadratureType::GAUSS_LEGENDRE, gauss_point_num_1d_,
            x1, y1, x2, y2);
        for (const vem::QuadPoint& point : points) {
            BasisPoint x = makeBasisPoint(point.x[0], point.x[1]);
            BasisVector hdiv = basis.evalHdivBasis(ploy_mk, xD, hD, x);
            double edge_value = basis.evalEdgeMonomial(ploy_exp, point1, point2, x);
            result += point.w * (hdiv.x * normal.x + hdiv.y * normal.y) *
                      edge_value;
        }
        return result;
    }

    vem::PolygonTriangulator triangulator;
    vem::TriangulatedElement element =
        triangulator.triangulateElement(*mesh_data_, mesh_idx);

    if (local_dof_idx < edge_dof + dim_grad_mk_) {
        int grad_dof = local_dof_idx - edge_dof;
        for (const vem::Triangle& triangle : element.triangles) {
            std::vector<vem::QuadPoint> points = quadrature.get_triangle_points(
                triangle.vertices, gauss_point_num_);
            for (const vem::QuadPoint& point : points) {
                BasisPoint x = makeBasisPoint(point.x[0], point.x[1]);
                BasisVector value = basis.evalHdivBasis(ploy_mk, xD, hD, x);
                BasisVector moment = basis.evalHdivBasis(grad_dof, xD, hD, x);
                result += (value.x * moment.x + value.y * moment.y) * point.w;
            }
        }
        return result;
    }

    int curl_dof = local_dof_idx - edge_dof - dim_grad_mk_;
    int complement_index = curl_dof + basis.getDimGradSpace();
    for (const vem::Triangle& triangle : element.triangles) {
        std::vector<vem::QuadPoint> points = quadrature.get_triangle_points(
            triangle.vertices, gauss_point_num_);
        for (const vem::QuadPoint& point : points) {
            BasisPoint x = makeBasisPoint(point.x[0], point.x[1]);
            BasisVector value = basis.evalHdivBasis(ploy_mk, xD, hD, x);
            BasisVector moment = basis.evalHdivBasis(complement_index, xD, hD, x);
            result += (value.x * moment.x + value.y * moment.y) * point.w;
        }
    }
    return result;
}

double HdivMatrix::vemHdiv_gm_gm_ploy(int mesh_idx, int ploy_mk1,
                                       int ploy_mk2) {
    if (mesh_idx < 0 || mesh_idx >= mesh_data_->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    Basis basis(k_);
    BasisPoint xD = makeBasisPoint(mesh_data_->cell_centroid_x[mesh_idx],
                                   mesh_data_->cell_centroid_y[mesh_idx]);
    double hD = mesh_data_->cell_diameter[mesh_idx];
    vem::GaussQuadrature quadrature;
    vem::PolygonTriangulator triangulator;
    vem::TriangulatedElement element =
        triangulator.triangulateElement(*mesh_data_, mesh_idx);

    double result = 0.0;
    for (const vem::Triangle& triangle : element.triangles) {
        std::vector<vem::QuadPoint> points = quadrature.get_triangle_points(
            triangle.vertices, gauss_point_num_);
        for (const vem::QuadPoint& point : points) {
            BasisPoint x = makeBasisPoint(point.x[0], point.x[1]);
            BasisVector first = basis.evalHdivBasis(ploy_mk1, xD, hD, x);
            BasisVector second = basis.evalHdivBasis(ploy_mk2, xD, hD, x);
            result += (first.x * second.x + first.y * second.y) * point.w;
        }
    }
    return result;
}

double HdivMatrix::vemHdiv_phi_grad_mk(int mesh_idx, int ploy_mk,
                                        int local_dof_idx) {
    if (mesh_idx < 0 || mesh_idx >= mesh_data_->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    int edge_dof = (k_ + 1) * mesh_data_->nodes_per_cell[mesh_idx];
    int grad_dof = local_dof_idx - edge_dof;
    return grad_dof == ploy_mk - 1 && grad_dof >= 0 ? -1.0 : 0.0;
}

double HdivMatrix::vemHdiv_phi_curl_mk(int mesh_idx, int ploy_gk,
                                        int local_dof_idx) {
    if (mesh_idx < 0 || mesh_idx >= mesh_data_->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    int edge_dof = (k_ + 1) * mesh_data_->nodes_per_cell[mesh_idx];
    int curl_dof = local_dof_idx - edge_dof - dim_grad_mk_;
    return curl_dof == ploy_gk && curl_dof >= 0 ? 1.0 : 0.0;
}

double HdivMatrix::vemHdiv_phi_normal_mk(int mesh_idx, int ploy_mk,
                                          int local_dof_idx) {
    if (mesh_idx < 0 || mesh_idx >= mesh_data_->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    int num_edges = mesh_data_->nodes_per_cell[mesh_idx];
    int edge_idx = local_dof_idx % num_edges;
    std::vector<int> nodes = getElementNodes(mesh_idx);
    int node0 = nodes[edge_idx];
    int node1 = nodes[(edge_idx + 1) % num_edges];

    BasisPoint xD = makeBasisPoint(mesh_data_->cell_centroid_x[mesh_idx],
                                   mesh_data_->cell_centroid_y[mesh_idx]);
    BasisPoint x0 = makeBasisPoint(mesh_data_->node_coords[2 * node0],
                                   mesh_data_->node_coords[2 * node0 + 1]);
    BasisPoint x1 = makeBasisPoint(mesh_data_->node_coords[2 * node1],
                                   mesh_data_->node_coords[2 * node1 + 1]);
    Basis basis(k_);
    std::vector<double> coefficients;
    basis.getEdgeBasisCoeffs(ploy_mk, xD, mesh_data_->cell_diameter[mesh_idx],
                             x0, x1, coefficients);

    int dof_type = local_dof_idx / num_edges;
    return coefficients[dof_type];
}

double HdivMatrix::vemHdiv_mk_mk(int mesh_idx, int ploy_mk1,
                                  int ploy_mk2) {
    if (mesh_idx < 0 || mesh_idx >= mesh_data_->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    Basis basis(k_ + 1);
    BasisPoint xD = makeBasisPoint(mesh_data_->cell_centroid_x[mesh_idx],
                                   mesh_data_->cell_centroid_y[mesh_idx]);
    double hD = mesh_data_->cell_diameter[mesh_idx];
    vem::GaussQuadrature quadrature;
    vem::PolygonTriangulator triangulator;
    vem::TriangulatedElement element =
        triangulator.triangulateElement(*mesh_data_, mesh_idx);

    double result = 0.0;
    for (const vem::Triangle& triangle : element.triangles) {
        std::vector<vem::QuadPoint> points = quadrature.get_triangle_points(
            triangle.vertices, gauss_point_num_);
        for (const vem::QuadPoint& point : points) {
            BasisPoint x = makeBasisPoint(point.x[0], point.x[1]);
            result += basis.evalMonomial2D(ploy_mk1, xD, hD, x) *
                      basis.evalMonomial2D(ploy_mk2, xD, hD, x) * point.w;
        }
    }
    return result;
}

double HdivMatrix::vemHdiv_phi_normal_mk_edge(int mesh_idx, int ploy_mk,
                                               int local_dof_idx) {
    if (mesh_idx < 0 || mesh_idx >= mesh_data_->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    int num_edges = mesh_data_->nodes_per_cell[mesh_idx];
    return Delta_function(local_dof_idx % num_edges, ploy_mk);
}

double HdivMatrix::vemHdiv_mk_mk_edge(int mesh_idx, int ploy_mk1,
                                       int ploy_mk2, int edge_idx) {
    if (mesh_idx < 0 || mesh_idx >= mesh_data_->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    if (edge_idx < 0 || edge_idx >= mesh_data_->nodes_per_cell[mesh_idx]) {
        throw std::out_of_range("local edge index out of range");
    }
    int num_edges = mesh_data_->nodes_per_cell[mesh_idx];
    std::vector<int> nodes = getElementNodes(mesh_idx);
    int node0 = nodes[edge_idx];
    int node1 = nodes[(edge_idx + 1) % num_edges];
    BasisPoint x0 = makeBasisPoint(mesh_data_->node_coords[2 * node0],
                                   mesh_data_->node_coords[2 * node0 + 1]);
    BasisPoint x1 = makeBasisPoint(mesh_data_->node_coords[2 * node1],
                                   mesh_data_->node_coords[2 * node1 + 1]);
    Basis basis(k_);
    vem::GaussQuadrature quadrature;
    std::vector<vem::QuadPoint> points = quadrature.get_line_2D_points(
        vem::QuadratureType::GAUSS_LEGENDRE, gauss_point_num_1d_,
        x0.x, x0.y, x1.x, x1.y);

    double result = 0.0;
    for (const vem::QuadPoint& point : points) {
        BasisPoint x = makeBasisPoint(point.x[0], point.x[1]);
        result += basis.evalEdgeMonomial(ploy_mk1, x0, x1, x) *
                  basis.evalEdgeMonomial(ploy_mk2, x0, x1, x) * point.w;
    }
    return result;
}

double HdivMatrix::vemHdiv_mk1D_mk2D(int mesh_idx, int ploy_mk1,
                                       int ploy_mk2, int edge_idx) {
    if (mesh_idx < 0 || mesh_idx >= mesh_data_->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    if (edge_idx < 0 || edge_idx >= mesh_data_->nodes_per_cell[mesh_idx]) {
        throw std::out_of_range("local edge index out of range");
    }
    int num_edges = mesh_data_->nodes_per_cell[mesh_idx];
    std::vector<int> nodes = getElementNodes(mesh_idx);
    int node0 = nodes[edge_idx];
    int node1 = nodes[(edge_idx + 1) % num_edges];
    BasisPoint xD = makeBasisPoint(mesh_data_->cell_centroid_x[mesh_idx],
                                   mesh_data_->cell_centroid_y[mesh_idx]);
    BasisPoint x0 = makeBasisPoint(mesh_data_->node_coords[2 * node0],
                                   mesh_data_->node_coords[2 * node0 + 1]);
    BasisPoint x1 = makeBasisPoint(mesh_data_->node_coords[2 * node1],
                                   mesh_data_->node_coords[2 * node1 + 1]);
    Basis edge_basis(k_);
    Basis volume_basis(k_ + 1);
    vem::GaussQuadrature quadrature;
    std::vector<vem::QuadPoint> points = quadrature.get_line_2D_points(
        vem::QuadratureType::GAUSS_LEGENDRE, gauss_point_num_1d_,
        x0.x, x0.y, x1.x, x1.y);

    double result = 0.0;
    for (const vem::QuadPoint& point : points) {
        BasisPoint x = makeBasisPoint(point.x[0], point.x[1]);
        result += volume_basis.evalMonomial2D(
                      ploy_mk2 + 1, xD, mesh_data_->cell_diameter[mesh_idx], x) *
                  edge_basis.evalEdgeMonomial(ploy_mk1, x0, x1, x) * point.w;
    }
    return result;
}

AutoPetscMat HdivMatrix::getMatrixD(int mesh_idx) { return HdivMatrixD(mesh_idx); }
AutoPetscMat HdivMatrix::getMatrixG(int mesh_idx) { return HdivMatrixG(mesh_idx); }
AutoPetscMat HdivMatrix::getMatrixW(int mesh_idx) { return HdivMatrixW(mesh_idx); }
AutoPetscMat HdivMatrix::getMatrixH(int mesh_idx) { return HdivMatrixH(mesh_idx); }
AutoPetscMat HdivMatrix::getMatrixH_star(int mesh_idx) {
    return HdivMatrixH_star(mesh_idx);
}
AutoPetscMat HdivMatrix::getMatrixH_edge(int mesh_idx, int edge_idx) {
    return HdivMatrixH_edge(mesh_idx, edge_idx);
}
AutoPetscMat HdivMatrix::getMatrixW_edge(int mesh_idx) {
    return HdivMatrixW_edge(mesh_idx);
}
AutoPetscMat HdivMatrix::getMatrixH_edge_star(int mesh_idx, int edge_idx) {
    return HdivMatrixH_edge_star(mesh_idx, edge_idx);
}
AutoPetscMat HdivMatrix::getMatrixB_2(int mesh_idx) { return HdivMatrixB_2(mesh_idx); }
AutoPetscMat HdivMatrix::getMatrixB_1(const AutoPetscMat& H_star,
                                      const AutoPetscMat& H,
                                      const AutoPetscMat& W) {
    return HdivMatrixB_1(H_star, H, W);
}
AutoPetscMat HdivMatrix::getMatrixB_grad(int mesh_idx,
                                         const AutoPetscMat& H_star,
                                         const AutoPetscMat& H,
                                         const AutoPetscMat& W) {
    return HdivMatrixB_grad(mesh_idx, H_star, H, W);
}
AutoPetscMat HdivMatrix::getMatrixB_curl(int mesh_idx) {
    return HdivMatrixB_curl(mesh_idx);
}
AutoPetscMat HdivMatrix::getMatrixB(int mesh_idx, const AutoPetscMat& H_star,
                                     const AutoPetscMat& H,
                                     const AutoPetscMat& W) {
    return HdivMatrixB(mesh_idx, H_star, H, W);
}
AutoPetscMat HdivMatrix::getMatrixL2Proj_ploy(const AutoPetscMat& G,
                                               const AutoPetscMat& B) {
    return HdivMatrixL2Proj_ploy(G, B);
}
AutoPetscMat HdivMatrix::getMatrixL2Proj_basis(const AutoPetscMat& G,
                                                const AutoPetscMat& B,
                                                const AutoPetscMat& D) {
    return HdivMatrixL2Proj_basis(G, B, D);
}

AutoPetscMat HdivMatrix::HdivMatrixD(int mesh_idx) {
    if (mesh_idx < 0 || mesh_idx >= mesh_data_->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    Basis basis(k_);
    int rows = dof_per_element_flux_[mesh_idx];
    int cols = basis.getDimHdiv();
    AutoPetscMat matrix = vem::create_dense_matrix(rows, cols);
    Mat raw = vem::get_raw(matrix);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            setValue(raw, row, col, vemHdiv_dof_g_value(mesh_idx, row, col));
        }
    }
    assemble(raw);
    return matrix;
}

AutoPetscMat HdivMatrix::HdivMatrixG(int mesh_idx) {
    if (mesh_idx < 0 || mesh_idx >= mesh_data_->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    int dimension = Basis(k_).getDimHdiv();
    AutoPetscMat matrix = vem::create_dense_matrix(dimension, dimension);
    Mat raw = vem::get_raw(matrix);
    for (int row = 0; row < dimension; ++row) {
        for (int col = 0; col < dimension; ++col) {
            setValue(raw, row, col, vemHdiv_gm_gm_ploy(mesh_idx, row, col));
        }
    }
    assemble(raw);
    return matrix;
}

AutoPetscMat HdivMatrix::HdivMatrixW(int mesh_idx) {
    if (mesh_idx < 0 || mesh_idx >= mesh_data_->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    int num_edges = mesh_data_->nodes_per_cell[mesh_idx];
    int edge_dof = (k_ + 1) * num_edges;
    int rows = Basis(k_).getDimMonomial(2);
    int cols = dof_per_element_flux_[mesh_idx];
    AutoPetscMat matrix = vem::create_dense_matrix(rows, cols);
    Mat raw = vem::get_raw(matrix);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            double value = vemHdiv_phi_grad_mk(mesh_idx, row, col);
            if (col < edge_dof) {
                value += vemHdiv_phi_normal_mk(mesh_idx, row, col);
            }
            setValue(raw, row, col, value);
        }
    }
    assemble(raw);
    return matrix;
}

AutoPetscMat HdivMatrix::HdivMatrixH(int mesh_idx) {
    if (mesh_idx < 0 || mesh_idx >= mesh_data_->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    int dimension = Basis(k_).getDimMonomial(2);
    AutoPetscMat matrix = vem::create_dense_matrix(dimension, dimension);
    Mat raw = vem::get_raw(matrix);
    for (int row = 0; row < dimension; ++row) {
        for (int col = 0; col < dimension; ++col) {
            setValue(raw, row, col, vemHdiv_mk_mk(mesh_idx, row, col));
        }
    }
    assemble(raw);
    return matrix;
}

AutoPetscMat HdivMatrix::HdivMatrixH_star(int mesh_idx) {
    if (mesh_idx < 0 || mesh_idx >= mesh_data_->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    Basis basis(k_);
    int rows = basis.getDimGradSpace();
    int cols = basis.getDimMonomial(2);
    AutoPetscMat matrix = vem::create_dense_matrix(rows, cols);
    Mat raw = vem::get_raw(matrix);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            setValue(raw, row, col, vemHdiv_mk_mk(mesh_idx, row + 1, col));
        }
    }
    assemble(raw);
    return matrix;
}

AutoPetscMat HdivMatrix::HdivMatrixH_edge(int mesh_idx, int edge_idx) {
    if (mesh_idx < 0 || mesh_idx >= mesh_data_->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    if (edge_idx < 0 || edge_idx >= mesh_data_->nodes_per_cell[mesh_idx]) {
        throw std::out_of_range("local edge index out of range");
    }
    int dimension = Basis(k_).getDimMonomial(1);
    AutoPetscMat matrix = vem::create_dense_matrix(dimension, dimension);
    Mat raw = vem::get_raw(matrix);
    for (int row = 0; row < dimension; ++row) {
        for (int col = 0; col < dimension; ++col) {
            setValue(raw, row, col,
                     vemHdiv_mk_mk_edge(mesh_idx, row, col, edge_idx));
        }
    }
    assemble(raw);
    return matrix;
}

AutoPetscMat HdivMatrix::HdivMatrixW_edge(int mesh_idx) {
    if (mesh_idx < 0 || mesh_idx >= mesh_data_->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    int dimension = k_ + 1;
    AutoPetscMat matrix = vem::create_dense_matrix(dimension, dimension);
    Mat raw = vem::get_raw(matrix);
    for (int row = 0; row < dimension; ++row) {
        for (int col = 0; col < dimension; ++col) {
            setValue(raw, row, col,
                     vemHdiv_phi_normal_mk_edge(mesh_idx, row, col));
        }
    }
    assemble(raw);
    return matrix;
}

AutoPetscMat HdivMatrix::HdivMatrixH_edge_star(int mesh_idx, int edge_idx) {
    if (mesh_idx < 0 || mesh_idx >= mesh_data_->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    if (edge_idx < 0 || edge_idx >= mesh_data_->nodes_per_cell[mesh_idx]) {
        throw std::out_of_range("local edge index out of range");
    }
    int rows = Basis(k_).getDimGradSpace();
    int cols = k_ + 1;
    AutoPetscMat matrix = vem::create_dense_matrix(rows, cols);
    Mat raw = vem::get_raw(matrix);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            setValue(raw, row, col,
                     vemHdiv_mk1D_mk2D(mesh_idx, col, row, edge_idx));
        }
    }
    assemble(raw);
    return matrix;
}

AutoPetscMat HdivMatrix::HdivMatrixB_2(int mesh_idx) {
    if (mesh_idx < 0 || mesh_idx >= mesh_data_->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    int num_edges = mesh_data_->nodes_per_cell[mesh_idx];
    int rows = Basis(k_).getDimGradSpace();
    int cols = dof_per_element_flux_[mesh_idx];
    AutoPetscMat matrix = vem::create_dense_matrix(rows, cols);
    Mat raw = vem::get_raw(matrix);

    for (int edge = 0; edge < num_edges; ++edge) {
        AutoPetscMat H_edge = HdivMatrixH_edge(mesh_idx, edge);
        AutoPetscMat H_edge_star = HdivMatrixH_edge_star(mesh_idx, edge);
        AutoPetscMat W_edge = HdivMatrixW_edge(mesh_idx);
        AutoPetscMat H_edge_inv = vem::inverse_matrix(H_edge);
        AutoPetscMat temporary = vem::multiply_matrices(H_edge_inv, W_edge);
        AutoPetscMat result = vem::multiply_matrices(H_edge_star, temporary);
        Mat result_raw = vem::get_raw(result);

        for (int edge_col = 0; edge_col < k_ + 1; ++edge_col) {
            int global_col = edge_col * num_edges + edge;
            for (int row = 0; row < rows; ++row) {
                PetscScalar value = 0.0;
                PetscInt petsc_row = row;
                PetscInt petsc_col = edge_col;
                MatGetValues(result_raw, 1, &petsc_row, 1, &petsc_col, &value);
                setValue(raw, row, global_col, PetscRealPart(value), ADD_VALUES);
            }
        }
    }
    assemble(raw);
    return matrix;
}

AutoPetscMat HdivMatrix::HdivMatrixB_1(const AutoPetscMat& H_star,
                                        const AutoPetscMat& H,
                                        const AutoPetscMat& W) {
    AutoPetscMat H_inv = vem::inverse_matrix(H);
    AutoPetscMat temporary = vem::multiply_matrices(H_inv, W);
    return vem::multiply_matrices(H_star, temporary, -1.0);
}

AutoPetscMat HdivMatrix::HdivMatrixB_grad(int mesh_idx,
                                           const AutoPetscMat& H_star,
                                           const AutoPetscMat& H,
                                           const AutoPetscMat& W) {
    AutoPetscMat B_1 = HdivMatrixB_1(H_star, H, W);
    AutoPetscMat B_2 = HdivMatrixB_2(mesh_idx);
    return vem::add_matrices(B_1, B_2);
}

AutoPetscMat HdivMatrix::HdivMatrixB_curl(int mesh_idx) {
    if (mesh_idx < 0 || mesh_idx >= mesh_data_->num_cells) {
        throw std::out_of_range("element index out of range");
    }
    int rows = dim_curl_mk_;
    int cols = dof_per_element_flux_[mesh_idx];
    AutoPetscMat matrix = vem::create_dense_matrix(rows, cols);
    Mat raw = vem::get_raw(matrix);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            setValue(raw, row, col,
                     vemHdiv_phi_curl_mk(mesh_idx, row, col));
        }
    }
    assemble(raw);
    return matrix;
}

AutoPetscMat HdivMatrix::HdivMatrixB(int mesh_idx,
                                      const AutoPetscMat& H_star,
                                      const AutoPetscMat& H,
                                      const AutoPetscMat& W) {
    AutoPetscMat B_grad = HdivMatrixB_grad(mesh_idx, H_star, H, W);
    AutoPetscMat B_curl = HdivMatrixB_curl(mesh_idx);
    PetscInt grad_rows = 0;
    PetscInt grad_cols = 0;
    PetscInt curl_rows = 0;
    PetscInt curl_cols = 0;
    MatGetSize(vem::get_raw(B_grad), &grad_rows, &grad_cols);
    MatGetSize(vem::get_raw(B_curl), &curl_rows, &curl_cols);
    if (curl_cols != grad_cols) {
        throw std::runtime_error("B_grad and B_curl column counts differ");
    }

    AutoPetscMat matrix = vem::create_dense_matrix(grad_rows + curl_rows,
                                                    grad_cols);
    Mat raw = vem::get_raw(matrix);
    vem::insert_submatrix(raw, B_grad, 0, 0);
    if (curl_rows > 0) {
        vem::insert_submatrix(raw, B_curl, grad_rows, 0);
    }
    assemble(raw);
    return matrix;
}

AutoPetscMat HdivMatrix::HdivMatrixL2Proj_ploy(const AutoPetscMat& G,
                                                const AutoPetscMat& B) {
    AutoPetscMat G_inv = vem::inverse_matrix(G);
    return vem::multiply_matrices(G_inv, B);
}

AutoPetscMat HdivMatrix::HdivMatrixL2Proj_basis(const AutoPetscMat& G,
                                                 const AutoPetscMat& B,
                                                 const AutoPetscMat& D) {
    AutoPetscMat projection = HdivMatrixL2Proj_ploy(G, B);
    return vem::multiply_matrices(D, projection);
}
