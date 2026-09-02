#include "../solver/HdivMatrix.h"

#include <petsc.h>

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void printMatrix(const std::string& name, const AutoPetscMat& matrix) {
    Mat raw = vem::get_raw(matrix);
    PetscInt rows = 0;
    PetscInt cols = 0;
    MatGetSize(raw, &rows, &cols);

    std::cout << "\n" << name << " (" << rows << " x " << cols << ")\n";
    std::cout << std::scientific << std::setprecision(10);
    for (PetscInt row = 0; row < rows; ++row) {
        std::cout << "  [";
        for (PetscInt col = 0; col < cols; ++col) {
            PetscScalar value = 0.0;
            MatGetValues(raw, 1, &row, 1, &col, &value);
            if (col > 0) {
                std::cout << ' ';
            }
            std::cout << std::setw(18) << PetscRealPart(value);
        }
        std::cout << " ]\n";
    }
    std::cout << std::defaultfloat;
}

void printIntegerVector(const std::string& name,
                        const std::vector<int>& values) {
    std::cout << name << ": [";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index > 0) {
            std::cout << ", ";
        }
        std::cout << values[index];
    }
    std::cout << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
    PetscErrorCode error = PetscInitialize(&argc, &argv, NULL, NULL);
    if (error != 0) {
        return static_cast<int>(error);
    }

    int status = 0;
    try {
        const std::string mesh_file = "/home/lzccode/code/vem/gmesh_py/mesh_data/orthogonal_1x1.msh";
        const int quadrature_points_2d = 9;
        const int quadrature_points_1d = 9;
        const int k = 1;
        const int cell = 0;

        std::cout << "============================================================\n"
                  << "  新版 HdivMatrix 属性、方法和矩阵输出\n"
                  << "============================================================\n"
                  << "网格文件: " << mesh_file << '\n'
                  << "多项式阶数 k: " << k << '\n'
                  << "二维三角形积分点数: " << quadrature_points_2d << '\n'
                  << "一维边积分点数: " << quadrature_points_1d << '\n'
                  << "输出单元编号: " << cell << '\n';

        vem::StraightMeshReader reader;
        if (!reader.read_mesh(mesh_file)) {
            throw std::runtime_error("无法读取正交网格: " + mesh_file);
        }

        const vem::StraightMeshData& mesh = reader.get_mesh_data();
        if (cell < 0 || cell >= mesh.num_cells) {
            throw std::out_of_range("输出单元编号超出网格范围");
        }

        HdivMatrix hdiv(reader, quadrature_points_2d,
                         quadrature_points_1d, k);

        std::cout << "\n-------------------- 网格属性 --------------------\n";
        std::cout << "节点总数: " << mesh.num_nodes << '\n';
        std::cout << "单元总数: " << mesh.num_cells << '\n';
        std::cout << "全局边总数: " << mesh.num_edges << '\n';
        std::cout << "单元 " << cell << " 的节点数/边数: "
                  << mesh.nodes_per_cell[cell] << '\n';
        std::cout << "单元质心: (" << mesh.cell_centroid_x[cell] << ", "
                  << mesh.cell_centroid_y[cell] << ")\n";
        std::cout << "单元面积: " << mesh.cell_area[cell] << '\n';
        std::cout << "单元直径: " << mesh.cell_diameter[cell] << '\n';

        std::cout << "\n---------------- HdivMatrix 属性 ----------------\n";
        std::cout << "getMeshReader() 是否指向当前读取器: "
                  << (hdiv.getMeshReader() == &reader ? "是" : "否") << '\n';
        std::cout << "getMeshData() 是否指向当前网格数据: "
                  << (hdiv.getMeshData() == &mesh ? "是" : "否") << '\n';
        std::cout << "getGaussPointNum(): " << hdiv.getGaussPointNum()
                  << "（兼容接口，表示二维积分点数）\n";
        std::cout << "getGaussPointNum2D(): " << hdiv.getGaussPointNum2D()
                  << '\n';
        std::cout << "getGaussPointNum1D(): " << hdiv.getGaussPointNum1D()
                  << '\n';
        std::cout << "getK(): " << hdiv.getK() << '\n';
        std::cout << "getDofPerElementFlux(" << cell << "): "
                  << hdiv.getDofPerElementFlux(cell) << '\n';
        printIntegerVector("getDofPerElementFluxVec()",
                           hdiv.getDofPerElementFluxVec());
        std::cout << "getDofPerElementConc(): "
                  << hdiv.getDofPerElementConc() << '\n';
        std::cout << "getTotalDofFlux(): " << hdiv.getTotalDofFlux() << '\n';
        std::cout << "getTotalDofConc(): " << hdiv.getTotalDofConc() << '\n';
        std::cout << "getTotalDof(): " << hdiv.getTotalDof() << '\n';
        std::cout << "getDimGradMk(): " << hdiv.getDimGradMk() << '\n';
        std::cout << "getDimCurlMk(): " << hdiv.getDimCurlMk() << '\n';
        std::cout << "Delta_function(0,0): " << hdiv.Delta_function(0, 0)
                  << '\n';
        std::cout << "Delta_function(0,1): " << hdiv.Delta_function(0, 1)
                  << '\n';

        std::cout << "\n--------------- 单元节点与外法向 ---------------\n";
        std::vector<int> nodes = hdiv.getElementNodes(cell);
        printIntegerVector("getElementNodes()", nodes);
        for (std::size_t edge = 0; edge < nodes.size(); ++edge) {
            int node0 = nodes[edge];
            int node1 = nodes[(edge + 1) % nodes.size()];
            Point2D normal = hdiv.computeEdgeNormal(node0, node1);
            std::cout << "局部边 " << edge << " (节点 " << node0 << " -> "
                      << node1 << ") 外法向: (" << normal.x << ", "
                      << normal.y << ")\n";
        }

        AutoPetscMat D = hdiv.getMatrixD(cell);
        AutoPetscMat G = hdiv.getMatrixG(cell);
        AutoPetscMat W = hdiv.getMatrixW(cell);
        AutoPetscMat H = hdiv.getMatrixH(cell);
        AutoPetscMat H_star = hdiv.getMatrixH_star(cell);

        std::cout << "\n------------------- 基础矩阵 --------------------\n";
        printMatrix("D", D);
        printMatrix("G", G);
        printMatrix("W", W);
        printMatrix("H", H);
        printMatrix("H_star", H_star);

        std::cout << "\n------------------- 边上矩阵 --------------------\n";
        for (std::size_t edge = 0; edge < nodes.size(); ++edge) {
            const int edge_index = static_cast<int>(edge);
            AutoPetscMat H_edge = hdiv.getMatrixH_edge(cell, edge_index);
            AutoPetscMat W_edge = hdiv.getMatrixW_edge(cell);
            AutoPetscMat H_edge_star =
                hdiv.getMatrixH_edge_star(cell, edge_index);
            printMatrix("H_edge[" + std::to_string(edge) + "]", H_edge);
            printMatrix("W_edge[" + std::to_string(edge) + "]", W_edge);
            printMatrix("H_edge_star[" + std::to_string(edge) + "]",
                        H_edge_star);
        }

        AutoPetscMat B_1 = hdiv.getMatrixB_1(H_star, H, W);
        AutoPetscMat B_2 = hdiv.getMatrixB_2(cell);
        AutoPetscMat B_grad = hdiv.getMatrixB_grad(cell, H_star, H, W);
        AutoPetscMat B_curl = hdiv.getMatrixB_curl(cell);
        AutoPetscMat B = hdiv.getMatrixB(cell, H_star, H, W);
        AutoPetscMat projection_poly = hdiv.getMatrixL2Proj_ploy(G, B);
        AutoPetscMat projection_basis =
            hdiv.getMatrixL2Proj_basis(G, B, D);
        AutoPetscMat BD = vem::multiply_matrices(B, D);

        std::cout << "\n--------------- 投影与组合矩阵 -----------------\n";
        printMatrix("B_1", B_1);
        printMatrix("B_2", B_2);
        printMatrix("B_grad", B_grad);
        printMatrix("B_curl", B_curl);
        printMatrix("B", B);
        printMatrix("L2Proj_ploy = G^{-1}B", projection_poly);
        printMatrix("L2Proj_basis = DG^{-1}B", projection_basis);
        printMatrix("BD = B * D", BD);
        printMatrix("G（用于直接检查 BD 是否等于 G）", G);

        std::cout << "\n============================================================\n"
                  << "  新版 HdivMatrix 全部属性、方法和矩阵输出完成 ✓\n"
                  << "============================================================\n";
    } catch (const std::exception& exception) {
        std::cerr << "新版 HdivMatrix 输出测试失败: " << exception.what()
                  << std::endl;
        status = 1;
    }

    PetscFinalize();
    return status;
}
