#ifndef CURVEDVEM_HDIV_MATRIX_H
#define CURVEDVEM_HDIV_MATRIX_H

#include <petsc.h>

#include <stdexcept>
#include <vector>

#include "../core/gauss_quadrature.h"
#include "../core/petsc_utils.h"
#include "../lib/polynomial_basis.h"
#include "../mesh/polygon_triangulator.h"
#include "../mesh/straight_mesh.h"

// 保留旧 HdivMatrix 的类型名称和接口形式，仅将底层类型适配到新架构。
using AutoPetscMat = vem::AutoPetscMat;
using MeshData = vem::StraightMeshData;
using MeshReader = vem::StraightMeshReader;
using Point2D = vem::Point2D;

class HdivMatrix {
public:
    // 参数依次为：直边计算网格、二维三角形积分点数、
    // 一维 Gauss-Legendre 积分点数、多项式阶数 k。
    HdivMatrix(const MeshReader& mesh_reader, int gauss_point_num = 9,
               int gauss_point_num_1d = 9, int k = 1)
        : mesh_reader_(&mesh_reader), mesh_data_(&mesh_reader_->get_mesh_data()),
          gauss_point_num_(gauss_point_num),
          gauss_point_num_1d_(gauss_point_num_1d), k_(k) {
        if (mesh_data_ == nullptr) {
            throw std::invalid_argument("Error: MeshData pointer is null!");
        }
        initDegrees();
    }

    AutoPetscMat getMatrixD(int mesh_idx);
    AutoPetscMat getMatrixG(int mesh_idx);
    AutoPetscMat getMatrixW(int mesh_idx);
    AutoPetscMat getMatrixH(int mesh_idx);
    AutoPetscMat getMatrixH_star(int mesh_idx);
    AutoPetscMat getMatrixH_edge(int mesh_idx, int edge_idx);
    AutoPetscMat getMatrixW_edge(int mesh_idx);
    AutoPetscMat getMatrixH_edge_star(int mesh_idx, int edge_idx);
    AutoPetscMat getMatrixB_2(int mesh_idx);
    AutoPetscMat getMatrixB_1(const AutoPetscMat& H_star, const AutoPetscMat& H,
                              const AutoPetscMat& W);
    AutoPetscMat getMatrixB_grad(int mesh_idx, const AutoPetscMat& H_star,
                                 const AutoPetscMat& H, const AutoPetscMat& W);
    AutoPetscMat getMatrixB_curl(int mesh_idx);
    AutoPetscMat getMatrixB(int mesh_idx, const AutoPetscMat& H_star,
                            const AutoPetscMat& H, const AutoPetscMat& W);
    AutoPetscMat getMatrixL2Proj_ploy(const AutoPetscMat& G, const AutoPetscMat& B);
    AutoPetscMat getMatrixL2Proj_basis(const AutoPetscMat& G, const AutoPetscMat& B,
                                       const AutoPetscMat& D);

    const MeshReader* getMeshReader() const { return mesh_reader_; }
    const MeshData* getMeshData() const { return mesh_data_; }
    int getGaussPointNum() const { return gauss_point_num_; }
    int getGaussPointNum2D() const { return gauss_point_num_; }
    int getGaussPointNum1D() const { return gauss_point_num_1d_; }
    int getK() const { return k_; }
    int getDofPerElementFlux(int mesh_idx) const {
        return dof_per_element_flux_[mesh_idx];
    }
    const std::vector<int>& getDofPerElementFluxVec() const {
        return dof_per_element_flux_;
    }
    int getDofPerElementConc() const { return dof_per_element_conc_; }
    int getTotalDofFlux() const { return total_dof_flux_; }
    int getTotalDofConc() const { return total_dof_conc_; }
    int getTotalDof() const { return total_dof_flux_ + total_dof_conc_; }
    int getDimGradMk() const { return dim_grad_mk_; }
    int getDimCurlMk() const { return dim_curl_mk_; }

    std::vector<int> getElementNodes(int mesh_idx) const;
    Point2D computeEdgeNormal(int node1, int node2) const;
    double Delta_function(int i, int j);

private:
    const MeshReader* mesh_reader_;
    const MeshData* mesh_data_;
    int gauss_point_num_;
    int gauss_point_num_1d_;
    int k_;
    std::vector<int> dof_per_element_flux_;
    int dof_per_element_conc_;
    int total_dof_flux_;
    int total_dof_conc_;
    int dim_grad_mk_;
    int dim_curl_mk_;

    void initDegrees();

    double vemHdiv_dof_g_value(int mesh_idx, int local_dof_idx, int ploy_mk);
    double vemHdiv_gm_gm_ploy(int mesh_idx, int ploy_mk1, int ploy_mk2);
    double vemHdiv_phi_grad_mk(int mesh_idx, int ploy_mk, int local_dof_idx);
    double vemHdiv_phi_curl_mk(int mesh_idx, int ploy_gk, int local_dof_idx);
    double vemHdiv_phi_normal_mk(int mesh_idx, int ploy_mk, int local_dof_idx);
    double vemHdiv_mk_mk(int mesh_idx, int ploy_mk1, int ploy_mk2);
    double vemHdiv_phi_normal_mk_edge(int mesh_idx, int ploy_mk,
                                      int local_dof_idx);
    double vemHdiv_mk_mk_edge(int mesh_idx, int ploy_mk1, int ploy_mk2,
                              int edge_idx);
    double vemHdiv_mk1D_mk2D(int mesh_idx, int ploy_mk1, int ploy_mk2,
                              int edge_idx);

    AutoPetscMat HdivMatrixD(int mesh_idx);
    AutoPetscMat HdivMatrixG(int mesh_idx);
    AutoPetscMat HdivMatrixW(int mesh_idx);
    AutoPetscMat HdivMatrixH(int mesh_idx);
    AutoPetscMat HdivMatrixH_star(int mesh_idx);
    AutoPetscMat HdivMatrixH_edge(int mesh_idx, int edge_idx);
    AutoPetscMat HdivMatrixW_edge(int mesh_idx);
    AutoPetscMat HdivMatrixH_edge_star(int mesh_idx, int edge_idx);
    AutoPetscMat HdivMatrixB_2(int mesh_idx);
    AutoPetscMat HdivMatrixB_1(const AutoPetscMat& H_star, const AutoPetscMat& H,
                               const AutoPetscMat& W);
    AutoPetscMat HdivMatrixB_grad(int mesh_idx, const AutoPetscMat& H_star,
                                  const AutoPetscMat& H, const AutoPetscMat& W);
    AutoPetscMat HdivMatrixB_curl(int mesh_idx);
    AutoPetscMat HdivMatrixB(int mesh_idx, const AutoPetscMat& H_star,
                             const AutoPetscMat& H, const AutoPetscMat& W);
    AutoPetscMat HdivMatrixL2Proj_ploy(const AutoPetscMat& G,
                                       const AutoPetscMat& B);
    AutoPetscMat HdivMatrixL2Proj_basis(const AutoPetscMat& G,
                                        const AutoPetscMat& B,
                                        const AutoPetscMat& D);
};

#endif  // CURVEDVEM_HDIV_MATRIX_H
