#ifndef HdivMatrix_H
#define HdivMatrix_H

#include <petsc.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <memory>  // 智能指针头文件
#include <stdexcept>//错误处理
#include <string>
#include "MatrixPetsc.h"
#include "../GaussQuadrature.h"
#include "../MeshReader.h"
#include "../triangle_mesher.h"
#include "../pde_data.h"
// 注意：包含 BasisFunctionPloy.h 时，里面的类型在 vemhdiv 命名空间中
#include "BasisFunctionPloy.h"

using namespace MatrixPetsc;

class HdivMatrix {
public:
    //构造函数
    HdivMatrix(const MeshReader& mesh_reader, int gauss_point_num = 9, int k = 1)
        : mesh_reader_(&mesh_reader), mesh_data_(&mesh_reader_->getMeshData()),
          gauss_point_num_(gauss_point_num), k_(k) {
        // 空指针检查
        if (mesh_data_ == nullptr) {
            throw std::invalid_argument("Error: MeshData pointer is null!");
        }
    initDegrees(); // 初始化自由度计算

    }

    // 获取单元的D矩阵（多项式基函数在自由度上的取值）
    AutoPetscMat getMatrixD(int mesh_idx);
    AutoPetscMat getMatrixG(int mesh_idx);
    AutoPetscMat getMatrixW(int mesh_idx);
    AutoPetscMat getMatrixH(int mesh_idx);
    AutoPetscMat getMatrixH_star(int mesh_idx);
    AutoPetscMat getMatrixH_edge(int mesh_idx, int edge_idx); //获取边界上的矩阵
    AutoPetscMat getMatrixW_edge(int mesh_idx);
    AutoPetscMat getMatrixH_edge_star(int mesh_idx, int edge_idx); //获取边界上的矩阵
    AutoPetscMat getMatrixB_2(int mesh_idx);
    AutoPetscMat getMatrixB_1(const AutoPetscMat& H_star, const AutoPetscMat& H, const AutoPetscMat& W);
    AutoPetscMat getMatrixB_grad(int mesh_idx, const AutoPetscMat& H_star, const AutoPetscMat& H, const AutoPetscMat& W);
    AutoPetscMat getMatrixB_curl(int mesh_idx);
    AutoPetscMat getMatrixB(int mesh_idx, const AutoPetscMat& H_star, const AutoPetscMat& H, const AutoPetscMat& W);
    AutoPetscMat getMatrixL2Proj_ploy(const AutoPetscMat& G, const AutoPetscMat& B);
    AutoPetscMat getMatrixL2Proj_basis(const AutoPetscMat& G, const AutoPetscMat& B, const AutoPetscMat& D);

    // ========================================================================
    // 数据访问接口
    // ========================================================================

    // 获取网格读取器
    const MeshReader* getMeshReader() const { return mesh_reader_; }

    // 获取网格数据
    const MeshData* getMeshData() const { return mesh_data_; }

    // 获取高斯积分点数量
    int getGaussPointNum() const { return gauss_point_num_; }

    // 获取多项式阶数k
    int getK() const { return k_; }

    // 获取单个单元的通量自由度数
    int getDofPerElementFlux(int mesh_idx) const {
        return dof_per_element_flux_[mesh_idx];
    }

    // 获取所有单元的通量自由度数向量
    const std::vector<int>& getDofPerElementFluxVec() const {
        return dof_per_element_flux_;
    }

    // 获取单个单元的浓度自由度数
    int getDofPerElementConc() const { return dof_per_element_conc_; }

    // 获取全局通量自由度总数
    int getTotalDofFlux() const { return total_dof_flux_; }

    // 获取全局浓度自由度总数
    int getTotalDofConc() const { return total_dof_conc_; }

    // 获取全局自由度总数（通量+浓度）
    int getTotalDof() const { return total_dof_flux_ + total_dof_conc_; }

    // 获取梯度矩空间维数
    int getDimGradMk() const { return dim_grad_mk_; }

    // 获取旋度矩空间维数
    int getDimCurlMk() const { return dim_curl_mk_; }

    // ========================================================================
    // 辅助函数
    // ========================================================================

    // 获取单元节点编号
    std::vector<int> getElementNodes(int mesh_idx) const;

    // 计算边的外法向量，输入参数是逆时针输入边的端点编号，输出是边的外法向量（单位向量）
    Point2D computeEdgeNormal(int node1, int node2) const;

    // 计算Delta函数，输入参数是两个局部自由度编号，输出是Delta函数的值（0或1）
    double Delta_function(int i, int j);

private:
    const MeshReader* mesh_reader_; // 指向主函数的MeshReader对象
    const MeshData* mesh_data_;   // 网格数据指针（只读）
    int gauss_point_num_;    // 高斯积分点数量
    int k_;                 // H(div)空间的多项式最高次数
    //自由度的计算，单个组分的局部自由度
    std::vector<int> dof_per_element_flux_; // Hdiv空间的变量即通量自由度，每个单元的数量，根据k计算，这和单元的边有关，应该是个向量
    int dof_per_element_conc_;  // 浓度的自由度，每个单元的自由度数量，也是根据k计算，其实就是k次多项式的维数
    //单个组分的浓度总自由度，所有浓度直接乘以浓度数量就行
    int total_dof_flux_; // 全局通量自由度总数
    int total_dof_conc_; // 全局浓度自由度总数

    //多项式空间维数
    int dim_grad_mk_;//这是自由度中梯度矩的数目
    int dim_curl_mk_;//这是自由度中旋度矩的数目


    //初始化函数
    void initDegrees(); // 初始化自由度计算

    //计算二维多项式基函数的自由度取值，这里输入二维多项式的编号，总共2*nk个基函数组成的空间
    double vemHdiv_dof_g_value(int mesh_idx, int local_dof_idx, int ploy_mk);
    //计算矩阵G的内部元素，基函数与基函数的内积
    double vemHdiv_gm_gm_ploy(int mesh_idx, int ploy_mk1, int ploy_mk2);
    //计算负的基函数phi与mk的梯度的内积，记住这里的mk是缩放单项式
    double vemHdiv_phi_grad_mk(int mesh_idx, int ploy_mk, int local_dof_idx);

    //计算基函数phi与curl旋度空间基函数的内积
    double vemHdiv_phi_curl_mk(int mesh_idx, int ploy_gk, int local_dof_idx);

    //基函数的法向与单项式mk的内积，记住这里的mk是缩放单项式，只有这里只有边界上的积分
    double vemHdiv_phi_normal_mk(int mesh_idx, int ploy_mk, int local_dof_idx);
    //计算单项式mk与nk之间的内积
    double vemHdiv_mk_mk(int mesh_idx, int ploy_mk1, int ploy_mk2);

    //计算phi的法向量与mk的内积，这个是用自由度计算的边界积分
    double vemHdiv_phi_normal_mk_edge(int mesh_idx, int ploy_mk, int local_dof_idx);

    //计算边界e上的单项式乘积的积分
    double vemHdiv_mk_mk_edge(int mesh_idx, int ploy_mk1, int ploy_mk2, int edge_idx);
    //计算边界上的一维多项式和mk+1的内积
    double vemHdiv_mk1D_mk2D(int mesh_idx, int ploy_mk1, int ploy_mk2, int edge_idx);


    //组装Hdiv矩阵，单元的子矩阵
    //组装空间的基函数的自由度取值
    AutoPetscMat HdivMatrixD(int mesh_idx);
    AutoPetscMat HdivMatrixG(int mesh_idx);
    AutoPetscMat HdivMatrixW(int mesh_idx);
    AutoPetscMat HdivMatrixH(int mesh_idx);
    AutoPetscMat HdivMatrixH_star(int mesh_idx);
    AutoPetscMat HdivMatrixH_edge(int mesh_idx, int edge_idx); //组装边界上的矩阵
    AutoPetscMat HdivMatrixW_edge(int mesh_idx);
    AutoPetscMat HdivMatrixH_edge_star(int mesh_idx, int edge_idx);
    AutoPetscMat HdivMatrixB_2(int mesh_idx);
    AutoPetscMat HdivMatrixB_1(const AutoPetscMat& H_star, const AutoPetscMat& H, const AutoPetscMat& W);
    AutoPetscMat HdivMatrixB_grad(int mesh_idx, const AutoPetscMat& H_star, const AutoPetscMat& H, const AutoPetscMat& W);
    AutoPetscMat HdivMatrixB_curl(int mesh_idx);
    AutoPetscMat HdivMatrixB(int mesh_idx, const AutoPetscMat& H_star, const AutoPetscMat& H, const AutoPetscMat& W);//组装矩阵B，并且已经验证BD=G
    AutoPetscMat HdivMatrixL2Proj_ploy(const AutoPetscMat& G, const AutoPetscMat& B); //组装多项式空间的L2投影矩阵
    AutoPetscMat HdivMatrixL2Proj_basis(const AutoPetscMat& G, const AutoPetscMat& B, const AutoPetscMat& D); //组装基函数空间的L2投影矩阵
};



#endif // HdivMatrix_H
