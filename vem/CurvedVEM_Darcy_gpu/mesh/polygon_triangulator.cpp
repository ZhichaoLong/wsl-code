/**
 * CPU 多边形剖分实现：处理原网格多边形并提供三角形面积及顶点。
 */
#include "polygon_triangulator.h"

namespace vem {

// ========== 验证单元数据合法性 ==========
bool PolygonTriangulator::validateElement(int num_nodes,
                                          const std::vector<int>& cell_nodes,
                                          const std::vector<int>& cell_node_indices,
                                          const std::vector<int>& nodes_per_cell,
                                          int num_cells,
                                          int element_id) const {
    if (element_id < 0 || element_id >= num_cells) {
        throw std::out_of_range("Element ID " + std::to_string(element_id) +
                               " out of range (0~" + std::to_string(num_cells - 1) + ")");
    }

    if (static_cast<size_t>(element_id) >= cell_node_indices.size()) {
        throw std::runtime_error("No node start index for element " + std::to_string(element_id));
    }

    if (static_cast<size_t>(element_id) >= nodes_per_cell.size()) {
        throw std::runtime_error("No vertex count for element " + std::to_string(element_id));
    }

    int num_vertices = nodes_per_cell[element_id];
    if (num_vertices < 3) {
        throw std::invalid_argument("Element " + std::to_string(element_id) +
                                  " has invalid vertex count: " + std::to_string(num_vertices));
    }

    int node_start_idx = cell_node_indices[element_id];
    if (node_start_idx < 0 ||
        static_cast<size_t>(node_start_idx + num_vertices) > cell_nodes.size()) {
        throw std::runtime_error("Element " + std::to_string(element_id) +
                                 " node indices are out of range");
    }

    for (int i = 0; i < num_vertices; ++i) {
        int node_id = cell_nodes[node_start_idx + i];
        if (node_id < 0 || node_id >= num_nodes) {
            throw std::runtime_error("Element " + std::to_string(element_id) +
                                     " contains invalid node ID " + std::to_string(node_id));
        }
    }

    return true;
}

// ========== 判断点是否在三角形内 ==========
bool PolygonTriangulator::isPointInsideTriangle(const std::vector<double>& points,
                                                int p_idx, int a_idx, int b_idx, int c_idx) const {
    double px = points[2 * p_idx], py = points[2 * p_idx + 1];
    double ax = points[2 * a_idx], ay = points[2 * a_idx + 1];
    double bx = points[2 * b_idx], by = points[2 * b_idx + 1];
    double cx = points[2 * c_idx], cy = points[2 * c_idx + 1];

    double area_total = 0.5 * std::fabs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay));
    double area1 = 0.5 * std::fabs((ax - px) * (by - py) - (bx - px) * (ay - py));
    double area2 = 0.5 * std::fabs((bx - px) * (cy - py) - (cx - px) * (by - py));
    double area3 = 0.5 * std::fabs((cx - px) * (ay - py) - (ax - px) * (cy - py));

    return std::fabs((area1 + area2 + area3) - area_total) < 1e-6;
}

// ========== 判断指定顶点是否为"耳朵" ==========
bool PolygonTriangulator::isEar(int i, const std::vector<double>& points, int num_points) const {
    int prev = (i - 1 + num_points) % num_points;
    int next = (i + 1) % num_points;

    double xi = points[2 * i], yi = points[2 * i + 1];
    double x_prev = points[2 * prev], y_prev = points[2 * prev + 1];
    double x_next = points[2 * next], y_next = points[2 * next + 1];

    double cross = (x_next - xi) * (y_prev - yi) - (x_prev - xi) * (y_next - yi);
    if (cross <= 1e-8) {
        return false;
    }

    for (int j = 0; j < num_points; ++j) {
        if (j == i || j == prev || j == next) {
            continue;
        }
        if (isPointInsideTriangle(points, j, prev, i, next)) {
            return false;
        }
    }

    return true;
}

// ========== 耳切法核心实现 ==========
TriangulatedElement PolygonTriangulator::earClipping(const std::vector<int>& vertex_indices,
                                                      const std::vector<double>& points) {
    TriangulatedElement result;
    result.total_area = 0.0;

    int num_vertices = static_cast<int>(vertex_indices.size());
    if (num_vertices < 3) {
        return result;
    }

    int remaining_points = num_vertices;
    std::vector<int> temp_indices = vertex_indices;
    std::vector<double> temp_points = points;

    while (remaining_points > 3) {
        bool found_ear = false;
        for (int i = 0; i < remaining_points; ++i) {
            if (isEar(i, temp_points, remaining_points)) {
                found_ear = true;
                int prev = (i - 1 + remaining_points) % remaining_points;
                int next = (i + 1) % remaining_points;

                // 创建三角形
                result.triangles.emplace_back(
                    temp_indices[prev], temp_indices[i], temp_indices[next],
                    temp_points[2 * prev], temp_points[2 * prev + 1],
                    temp_points[2 * i], temp_points[2 * i + 1],
                    temp_points[2 * next], temp_points[2 * next + 1]
                );
                result.total_area += result.triangles.back().area;

                // 移除耳朵点
                temp_points.erase(temp_points.begin() + 2 * i, temp_points.begin() + 2 * (i + 1));
                temp_indices.erase(temp_indices.begin() + i);
                remaining_points--;
                break;
            }
        }
        if (!found_ear) {
            throw std::runtime_error("No valid ear found for polygon");
        }
    }

    // 添加最后一个三角形
    result.triangles.emplace_back(
        temp_indices[0], temp_indices[1], temp_indices[2],
        temp_points[0], temp_points[1],
        temp_points[2], temp_points[3],
        temp_points[4], temp_points[5]
    );
    result.total_area += result.triangles.back().area;

    return result;
}

// ========== 对直边网格的指定单元进行三角剖分 ==========
TriangulatedElement PolygonTriangulator::triangulateElement(const StraightMeshData& meshData,
                                                           int element_id) {
    if (!validateElement(meshData.num_nodes, meshData.cell_nodes,
                        meshData.cell_node_indices, meshData.nodes_per_cell,
                        meshData.num_cells, element_id)) {
        return TriangulatedElement();
    }

    int num_vertices = meshData.nodes_per_cell[element_id];
    int node_start_idx = meshData.cell_node_indices[element_id];

    std::vector<int> vertex_indices;
    std::vector<double> points;
    points.reserve(2 * num_vertices);

    for (int i = 0; i < num_vertices; ++i) {
        int global_node_id = meshData.cell_nodes[static_cast<size_t>(node_start_idx) + i];
        vertex_indices.push_back(global_node_id);
        double x = meshData.node_coords[2 * global_node_id];
        double y = meshData.node_coords[2 * global_node_id + 1];
        points.push_back(x);
        points.push_back(y);
    }

    return earClipping(vertex_indices, points);
}

// ========== 对整个直边网格进行三角剖分 ==========
TriangulatedMesh PolygonTriangulator::triangulateMesh(const StraightMeshData& meshData) {
    TriangulatedMesh result;
    result.element_triangulations.reserve(meshData.num_cells);

    for (int i = 0; i < meshData.num_cells; ++i) {
        result.element_triangulations.push_back(triangulateElement(meshData, i));
    }

    return result;
}

// ========== 对曲边网格的指定单元进行三角剖分 ==========
TriangulatedElement PolygonTriangulator::triangulateElement(const CurvedMeshData& meshData,
                                                           int element_id) {
    if (!validateElement(meshData.num_nodes, meshData.cell_nodes,
                        meshData.cell_node_indices, meshData.nodes_per_cell,
                        meshData.num_cells, element_id)) {
        return TriangulatedElement();
    }

    int num_vertices = meshData.nodes_per_cell[element_id];
    int node_start_idx = meshData.cell_node_indices[element_id];

    std::vector<int> vertex_indices;
    std::vector<double> points;
    points.reserve(2 * num_vertices);

    for (int i = 0; i < num_vertices; ++i) {
        int global_node_id = meshData.cell_nodes[static_cast<size_t>(node_start_idx) + i];
        vertex_indices.push_back(global_node_id);
        double x = meshData.node_coords[2 * global_node_id];
        double y = meshData.node_coords[2 * global_node_id + 1];
        points.push_back(x);
        points.push_back(y);
    }

    return earClipping(vertex_indices, points);
}

// ========== 对整个曲边网格进行三角剖分 ==========
TriangulatedMesh PolygonTriangulator::triangulateMesh(const CurvedMeshData& meshData) {
    TriangulatedMesh result;
    result.element_triangulations.reserve(meshData.num_cells);

    for (int i = 0; i < meshData.num_cells; ++i) {
        result.element_triangulations.push_back(triangulateElement(meshData, i));
    }

    return result;
}

}  // namespace vem
