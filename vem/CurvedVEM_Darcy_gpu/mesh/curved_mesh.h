/**
 * CPU 曲边网格数据接口：保存原工程的几何/曲边表示，保留既有注释与类型。
 */
#ifndef CURVED_MESH_H
#define CURVED_MESH_H

#include <vector>
#include <string>
#include <utility>
#include <memory>
#include "../examples/darcy_problem.h"

namespace vem {

// ============================================================================
// 曲边网格数据结构（简化版）
// ============================================================================

struct CurvedMeshData {
    // 节点坐标（物理域中）
    std::vector<double> node_coords;  // [x0, y0, x1, y1, ...]
    int num_nodes = 0;

    // 单元信息
    std::vector<int> cell_nodes;      // 单元的节点全局编号（逆时针）
    std::vector<int> cell_node_indices;  // 每个单元在 cell_nodes 中的起始索引
    std::vector<int> nodes_per_cell;     // 每个单元的节点数
    int num_cells = 0;

    // 边信息
    std::vector<int> cell_edge_indices;  // 每个单元在 global_edges 中的起始索引
    std::vector<int> global_edges;       // 单元的边全局编号（逆时针）
    std::vector<int> edge_endpoints;     // 边的端点 [n0, n1, n2, n3, ...]
    std::vector<int> edge_occurrence;    // 边出现次数（1=边界，2=内部）
    int num_edges = 0;

    // 边界信息
    std::vector<int> boundary_nodes;     // 逆时针边界节点（含首尾闭合）
    std::vector<int> bnode_indices;      // 边界点索引 [下, 右, 上, 左, 结束]
    std::vector<int> boundary_edges;     // 逆时针边界边（全局编号）
    std::vector<int> bedge_indices;      // 边界边索引 [下, 右, 上, 左, 结束]
    int num_boundary_nodes = 0;
    int num_boundary_edges = 0;

    // 网格边界范围（物理域中）
    double x_min = 0.0, x_max = 0.0;
    double y_min = 0.0, y_max = 0.0;

    // 清空数据
    void clear() {
        node_coords.clear();
        num_nodes = 0;
        cell_nodes.clear();
        cell_node_indices.clear();
        nodes_per_cell.clear();
        num_cells = 0;
        cell_edge_indices.clear();
        global_edges.clear();
        edge_endpoints.clear();
        edge_occurrence.clear();
        num_edges = 0;
        boundary_nodes.clear();
        bnode_indices.clear();
        boundary_edges.clear();
        bedge_indices.clear();
        num_boundary_nodes = 0;
        num_boundary_edges = 0;
        x_min = x_max = y_min = y_max = 0.0;
    }
};

// ============================================================================
// 曲边网格生成器
// ============================================================================

class CurvedMeshGenerator {
public:
    CurvedMeshGenerator() = default;
    ~CurvedMeshGenerator() = default;

    // 从直边网格和映射生成曲边网格
    // straight_mesh_data: 直边网格（其坐标被视为参考坐标）
    // mapping: 曲边映射（参考坐标 -> 物理坐标）
    bool generate_from_straight(
        const std::vector<double>& straight_node_coords,
        int straight_num_nodes,
        const std::vector<int>& straight_cell_nodes,
        const std::vector<int>& straight_cell_node_indices,
        const std::vector<int>& straight_nodes_per_cell,
        int straight_num_cells,
        const std::vector<int>& straight_cell_edge_indices,
        const std::vector<int>& straight_global_edges,
        const std::vector<int>& straight_edge_endpoints,
        const std::vector<int>& straight_edge_occurrence,
        int straight_num_edges,
        const std::vector<int>& straight_boundary_nodes,
        const std::vector<int>& straight_bnode_indices,
        const std::vector<int>& straight_boundary_edges,
        const std::vector<int>& straight_bedge_indices,
        int straight_num_boundary_nodes,
        int straight_num_boundary_edges,
        const Darcy::IsoparametricMapping& mapping
    );

    // 简化版本：直接从直边网格数据结构生成
    bool generate_from_straight(
        const CurvedMeshData& straight_mesh,  // 直边网格的拓扑可以复用
        const Darcy::IsoparametricMapping& mapping
    );

    // 获取生成的曲边网格数据
    const CurvedMeshData& get_mesh_data() const { return mesh_data_; }

private:
    CurvedMeshData mesh_data_;

    // 计算网格边界范围
    void compute_mesh_bounds();
};

}  // namespace vem

#endif  // CURVED_MESH_H
