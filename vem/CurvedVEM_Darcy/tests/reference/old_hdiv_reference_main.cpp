#include "HdivMatrix.h"
#include "MeshReader.h"
#include "MatrixPetsc.h"

#include <petsc.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void writeMatrix(std::ofstream& output, const std::string& name,
                 const MatrixPetsc::AutoPetscMat& matrix) {
    Mat raw = MatrixPetsc::getRawMat(matrix);
    PetscInt rows = 0;
    PetscInt cols = 0;
    MatGetSize(raw, &rows, &cols);
    output << "MATRIX " << name << ' ' << rows << ' ' << cols << '\n';
    for (PetscInt row = 0; row < rows; ++row) {
        for (PetscInt col = 0; col < cols; ++col) {
            PetscScalar value = 0.0;
            MatGetValues(raw, 1, &row, 1, &col, &value);
            output << PetscRealPart(value) << '\n';
        }
    }
}

void writeCase(std::ofstream& output, const std::string& mesh_name,
               const std::string& mesh_path, int k, int cell) {
    MeshReader reader;
    if (!reader.generateCompleteMesh(mesh_path)) {
        throw std::runtime_error("旧网格读取器无法读取: " + mesh_path);
    }

    const MeshData& mesh = reader.getMeshData();
    if (cell < 0 || cell >= mesh.total_elements) {
        throw std::out_of_range("参考单元编号越界");
    }

    HdivMatrix hdiv(reader, 9, k);
    std::vector<int> nodes = hdiv.getElementNodes(cell);
    output << "CASE " << mesh_name << ' ' << k << ' ' << cell << ' ' << 9 << '\n';
    output << "META " << hdiv.getDofPerElementFlux(cell) << ' '
           << hdiv.getDofPerElementConc() << ' ' << hdiv.getTotalDofFlux() << ' '
           << hdiv.getTotalDofConc() << ' ' << hdiv.getTotalDof() << ' '
           << hdiv.getDimGradMk() << ' ' << hdiv.getDimCurlMk() << ' '
           << nodes.size();
    for (int node : nodes) {
        output << ' ' << node;
    }
    output << '\n';

    for (std::size_t edge = 0; edge < nodes.size(); ++edge) {
        Point2D normal = hdiv.computeEdgeNormal(
            nodes[edge], nodes[(edge + 1) % nodes.size()]);
        output << "NORMAL " << edge << ' ' << normal.x << ' ' << normal.y << '\n';
    }

    MatrixPetsc::AutoPetscMat D = hdiv.getMatrixD(cell);
    MatrixPetsc::AutoPetscMat G = hdiv.getMatrixG(cell);
    MatrixPetsc::AutoPetscMat W = hdiv.getMatrixW(cell);
    MatrixPetsc::AutoPetscMat H = hdiv.getMatrixH(cell);
    MatrixPetsc::AutoPetscMat H_star = hdiv.getMatrixH_star(cell);
    MatrixPetsc::AutoPetscMat B_1 = hdiv.getMatrixB_1(H_star, H, W);
    MatrixPetsc::AutoPetscMat B_2 = hdiv.getMatrixB_2(cell);
    MatrixPetsc::AutoPetscMat B_grad = hdiv.getMatrixB_grad(cell, H_star, H, W);
    MatrixPetsc::AutoPetscMat B_curl = hdiv.getMatrixB_curl(cell);
    MatrixPetsc::AutoPetscMat B = hdiv.getMatrixB(cell, H_star, H, W);
    MatrixPetsc::AutoPetscMat projection_poly = hdiv.getMatrixL2Proj_ploy(G, B);
    MatrixPetsc::AutoPetscMat projection_dof =
        hdiv.getMatrixL2Proj_basis(G, B, D);

    writeMatrix(output, "D", D);
    writeMatrix(output, "G", G);
    writeMatrix(output, "W", W);
    writeMatrix(output, "H", H);
    writeMatrix(output, "H_star", H_star);
    for (std::size_t edge = 0; edge < nodes.size(); ++edge) {
        MatrixPetsc::AutoPetscMat H_edge =
            hdiv.getMatrixH_edge(cell, static_cast<int>(edge));
        MatrixPetsc::AutoPetscMat W_edge = hdiv.getMatrixW_edge(cell);
        MatrixPetsc::AutoPetscMat H_edge_star =
            hdiv.getMatrixH_edge_star(cell, static_cast<int>(edge));
        writeMatrix(output, "H_edge_" + std::to_string(edge), H_edge);
        writeMatrix(output, "W_edge_" + std::to_string(edge), W_edge);
        writeMatrix(output, "H_edge_star_" + std::to_string(edge), H_edge_star);
    }
    writeMatrix(output, "B_1", B_1);
    writeMatrix(output, "B_2", B_2);
    writeMatrix(output, "B_grad", B_grad);
    writeMatrix(output, "B_curl", B_curl);
    writeMatrix(output, "B", B);
    writeMatrix(output, "L2Proj_ploy", projection_poly);
    writeMatrix(output, "L2Proj_basis", projection_dof);
    output << "ENDCASE\n";
}

}  // namespace

int main(int argc, char** argv) {
    PetscErrorCode error = PetscInitialize(&argc, &argv, NULL, NULL);
    if (error != 0) {
        return static_cast<int>(error);
    }

    int status = 0;
    try {
        if (argc != 3) {
            throw std::invalid_argument(
                "用法: old_hdiv_reference <输出文件> <新项目网格目录>");
        }
        std::ofstream output(argv[1]);
        if (!output) {
            throw std::runtime_error("无法创建旧实现参考数据文件");
        }
        output << std::setprecision(std::numeric_limits<double>::max_digits10);
        output << "HDIV_REFERENCE 1\n";

        const std::vector<std::string> meshes = {
            "square_2x2.msh", "orthogonal_2x2.msh"};
        for (const std::string& mesh : meshes) {
            std::string path = std::string(argv[2]) + "/" + mesh;
            for (int k = 1; k <= 2; ++k) {
                for (int cell = 0; cell <= 1; ++cell) {
                    writeCase(output, mesh, path, k, cell);
                }
            }
        }
        output << "END\n";
    } catch (const std::exception& exception) {
        std::cerr << "生成旧 HdivMatrix 参考数据失败: "
                  << exception.what() << std::endl;
        status = 1;
    }

    PetscFinalize();
    return status;
}
