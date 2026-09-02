#include "curved_mesh.h"
#include <algorithm>
#include <cmath>

namespace vem {

// ============================================================================
// 从直边网格和映射生成曲边网格
// ============================================================================

bool CurvedMeshGenerator::generate_from_straight(
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
) {
    // 清空旧数据
    mesh_data_.clear();

    // 1. 复制拓扑信息（节点数、单元数、边数等）
    mesh_data_.num_nodes = straight_num_nodes;
    mesh_data_.num_cells = straight_num_cells;
    mesh_data_.num_edges = straight_num_edges;
    mesh_data_.num_boundary_nodes = straight_num_boundary_nodes;
    mesh_data_.num_boundary_edges = straight_num_boundary_edges;

    // 2. 复制单元拓扑
    mesh_data_.cell_nodes = straight_cell_nodes;
    mesh_data_.cell_node_indices = straight_cell_node_indices;
    mesh_data_.nodes_per_cell = straight_nodes_per_cell;

    // 3. 复制边拓扑
    mesh_data_.cell_edge_indices = straight_cell_edge_indices;
    mesh_data_.global_edges = straight_global_edges;
    mesh_data_.edge_endpoints = straight_edge_endpoints;
    mesh_data_.edge_occurrence = straight_edge_occurrence;

    // 4. 复制边界拓扑
    mesh_data_.boundary_nodes = straight_boundary_nodes;
    mesh_data_.bnode_indices = straight_bnode_indices;
    mesh_data_.boundary_edges = straight_boundary_edges;
    mesh_data_.bedge_indices = straight_bedge_indices;

    // 5. 通过映射计算曲边网格的节点坐标（核心步骤）
    mesh_data_.node_coords.resize(2 * mesh_data_.num_nodes);
    for (int i = 0; i < mesh_data_.num_nodes; ++i) {
        // 直边网格的坐标作为参考坐标 (xi, eta)
        double xi = straight_node_coords[2 * i];
        double eta = straight_node_coords[2 * i + 1];

        // 通过映射计算物理坐标 (x, y)
        Darcy::Point2D physical = mapping.physical_coords(xi, eta);

        mesh_data_.node_coords[2 * i] = physical.x;
        mesh_data_.node_coords[2 * i + 1] = physical.y;
    }

    // 6. 计算曲边网格的边界范围
    compute_mesh_bounds();

    return true;
}

// ============================================================================
// 简化版本：直接从数据结构生成
// ============================================================================

bool CurvedMeshGenerator::generate_from_straight(
    const CurvedMeshData& straight_mesh,
    const Darcy::IsoparametricMapping& mapping
) {
    return generate_from_straight(
        straight_mesh.node_coords,
        straight_mesh.num_nodes,
        straight_mesh.cell_nodes,
        straight_mesh.cell_node_indices,
        straight_mesh.nodes_per_cell,
        straight_mesh.num_cells,
        straight_mesh.cell_edge_indices,
        straight_mesh.global_edges,
        straight_mesh.edge_endpoints,
        straight_mesh.edge_occurrence,
        straight_mesh.num_edges,
        straight_mesh.boundary_nodes,
        straight_mesh.bnode_indices,
        straight_mesh.boundary_edges,
        straight_mesh.bedge_indices,
        straight_mesh.num_boundary_nodes,
        straight_mesh.num_boundary_edges,
        mapping
    );
}

// ============================================================================
// 计算网格边界范围
// ============================================================================

void CurvedMeshGenerator::compute_mesh_bounds() {
    if (mesh_data_.node_coords.empty() || mesh_data_.num_nodes == 0) {
        return;
    }

    // 初始化边界
    mesh_data_.x_min = mesh_data_.node_coords[0];
    mesh_data_.x_max = mesh_data_.node_coords[0];
    mesh_data_.y_min = mesh_data_.node_coords[1];
    mesh_data_.y_max = mesh_data_.node_coords[1];

    // 遍历所有节点找边界
    for (int i = 0; i < mesh_data_.num_nodes; ++i) {
        double x = mesh_data_.node_coords[2 * i];
        double y = mesh_data_.node_coords[2 * i + 1];

        if (x < mesh_data_.x_min) mesh_data_.x_min = x;
        if (x > mesh_data_.x_max) mesh_data_.x_max = x;
        if (y < mesh_data_.y_min) mesh_data_.y_min = y;
        if (y > mesh_data_.y_max) mesh_data_.y_max = y;
    }
}

}  // namespace vem
