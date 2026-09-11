/**
 * CPU 多边形三角剖分接口：用于构造多边形单元的体积分区域。
 */
#ifndef POLYGON_TRIANGULATOR_H
#define POLYGON_TRIANGULATOR_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include "straight_mesh.h"
#include "curved_mesh.h"

namespace vem {

// 三角形结构体 - 用于表示剖分后的三角形
struct Triangle {
    // 三角形三个顶点的全局编号
    std::vector<int> nodes;
    // 三个顶点坐标 [x0,y0, x1,y1, x2,y2]
    std::vector<double> vertices;
    // 三角形面积
    double area = 0.0;

    // 构造函数
    Triangle() : nodes(3, 0), vertices(6, 0.0) {}

    Triangle(int n0, int n1, int n2,
             double x0, double y0, double x1, double y1, double x2, double y2)
        : nodes({n0, n1, n2}), vertices({x0, y0, x1, y1, x2, y2}) {
        area = calculateArea();
    }

    // 计算当前三角形面积
    double calculateArea() const {
        return 0.5 * std::fabs((vertices[2] - vertices[0]) * (vertices[5] - vertices[1]) -
                              (vertices[4] - vertices[0]) * (vertices[3] - vertices[1]));
    }
};

// 单个单元的三角剖分结果
struct TriangulatedElement {
    // 剖分得到的所有三角形
    std::vector<Triangle> triangles;
    // 该单元的总面积（所有三角形面积之和）
    double total_area = 0.0;
    // 清空数据
    void clear() {
        triangles.clear();
        total_area = 0.0;
    }
};

// 整个网格的三角剖分结果
struct TriangulatedMesh {
    // 每个单元的剖分结果
    std::vector<TriangulatedElement> element_triangulations;
    // 清空数据
    void clear() {
        element_triangulations.clear();
    }
};

// 多边形三角剖分类
class PolygonTriangulator {
public:
    PolygonTriangulator() = default;
    ~PolygonTriangulator() = default;

    // ========== 对直边网格的操作 ==========

    // 对直边网格的指定单元进行三角剖分
    TriangulatedElement triangulateElement(const StraightMeshData& meshData, int element_id);

    // 对整个直边网格进行三角剖分
    TriangulatedMesh triangulateMesh(const StraightMeshData& meshData);

    // ========== 对曲边网格的操作 ==========

    // 对曲边网格的指定单元进行三角剖分
    TriangulatedElement triangulateElement(const CurvedMeshData& meshData, int element_id);

    // 对整个曲边网格进行三角剖分
    TriangulatedMesh triangulateMesh(const CurvedMeshData& meshData);

private:
    // ========== 内部辅助函数 ==========

    // 验证单元数据合法性
    bool validateElement(int num_nodes, const std::vector<int>& cell_nodes,
                        const std::vector<int>& cell_node_indices,
                        const std::vector<int>& nodes_per_cell,
                        int num_cells, int element_id) const;

    // 判断点是否在三角形内
    bool isPointInsideTriangle(const std::vector<double>& points,
                              int p_idx, int a_idx, int b_idx, int c_idx) const;

    // 判断指定顶点是否为"耳朵"
    bool isEar(int i, const std::vector<double>& points, int num_points) const;

    // 耳切法核心实现
    TriangulatedElement earClipping(const std::vector<int>& vertex_indices,
                                    const std::vector<double>& points);
};

}  // namespace vem

#endif  // POLYGON_TRIANGULATOR_H
