/**
 * CPU 原网格数据与读取接口：保存逆时针单元节点、全局边及边界信息。
 * GPU 打包直接沿用这些编号，不改变网格文件格式。
 */
#ifndef STRAIGHT_MESH_H
#define STRAIGHT_MESH_H

#include <vector>
#include <string>
#include <utility>
#include "curved_mesh.h"

namespace vem {

// 2D点结构
struct Point2D {
    double x, y;
    Point2D(double x = 0.0, double y = 0.0) : x(x), y(y) {}
};

// 直边网格数据结构
struct StraightMeshData {
    // 节点坐标
    std::vector<double> node_coords;  // [x0, y0, x1, y1, ..., xn-1, yn-1]
    int num_nodes = 0;

    // 单元信息
    std::vector<int> cell_nodes;      // 单元的节点全局编号（按逆时针排列）
    std::vector<int> cell_node_indices;  // 每个单元在cell_nodes中的起始索引
    std::vector<int> nodes_per_cell;     // 每个单元的节点数
    std::vector<double> cell_centroid_x; // 单元质心x坐标
    std::vector<double> cell_centroid_y; // 单元质心y坐标
    std::vector<double> cell_area;       // 单元面积
    std::vector<double> cell_diameter;   // 单元直径
    int num_cells = 0;

    // 边信息
    std::vector<int> cell_edge_indices;   // 每个单元在global_edges中的起始索引
    std::vector<int> global_edges;        // 单元的边全局编号（按逆时针排列）
    std::vector<int> edge_endpoints;      // 边的端点列表（[小端点, 大端点]，每条边占2位）
    std::vector<int> edge_occurrence;     // 边的出现次数（1表示边界边，2表示内部边）
    int num_edges = 0;

    // 边界信息
    std::vector<int> boundary_nodes;      // 逆时针排序的边界节点（含首尾闭合）
    std::vector<int> bnode_indices;       // 边界节点索引（下/右/上/左边界起始索引）
    std::vector<int> boundary_edges;      // 逆时针排序的边界边（全局编号）
    std::vector<int> bedge_indices;       // 边界边索引（下/右/上/左边界起始索引）
    int num_boundary_nodes = 0;
    int num_boundary_edges = 0;

    // 网格边界范围
    double x_min = 0.0, x_max = 0.0;
    double y_min = 0.0, y_max = 0.0;

    // 清空数据
    void clear() {
        node_coords.clear();
        num_nodes = 0;
        cell_nodes.clear();
        cell_node_indices.clear();
        nodes_per_cell.clear();
        cell_centroid_x.clear();
        cell_centroid_y.clear();
        cell_area.clear();
        cell_diameter.clear();
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

// 直边网格读取器
class StraightMeshReader {
public:
    StraightMeshReader() = default;
    ~StraightMeshReader() = default;

    // 读取Gmsh格式网格文件
    bool read_mesh(const std::string& filename);

    // 读取legacy VTK格式网格文件（core/mesh_refiner 生成的悬点网格）
    //
    // 与 read_mesh 的区别只在解析层：VTK 的 CELLS 段每行自带顶点数，
    // 天然支持变边数多边形，所以悬点网格（四边形 + 五边形混合）能直接读进来。
    // 解析完之后走的是与 read_mesh 完全相同的后处理链：
    //   compute_cell_properties -> generate_element_edges -> compute_boundary_info
    // 因此 get_mesh_data() 拿到的 StraightMeshData 语义与 msh 路径一致。
    //
    // 支持的 CELL_TYPES：5(三角形) / 7(多边形) / 9(四边形)，三者顶点表语义相同。
    bool read_mesh_vtk(const std::string& filename);

    // 获取网格数据
    const StraightMeshData& get_mesh_data() const { return mesh_data_; }

    // 查找单元局部边对应的相邻单元
    // 如果是边界边，返回当前单元ID
    int find_adjacent_cell(int cell_id, int local_edge_id) const;

    // 输入边的全局编号，输出相邻的两个单元
    // 如果是边界边，两个单元相同
    std::pair<int, int> find_adjacent_cells_by_edge(int global_edge_id) const;

    // 转换为曲边网格数据结构（只复制拓扑，坐标仍是直边的）
    void to_curved_mesh_data(CurvedMeshData& out) const;

private:
    StraightMeshData mesh_data_;

    // 读取节点坐标
    bool read_coords(const std::string& filename);

    // 读取单元信息
    bool read_elements(const std::string& filename);

    // 解析 legacy VTK 的 POINTS / CELLS / CELL_TYPES 三段
    bool read_vtk_nodes_and_cells(const std::string& filename);

    // 计算单元属性（质心、面积、直径）
    bool compute_cell_properties();

    // 生成单元边拓扑信息
    bool generate_element_edges();

    // 计算边界信息
    bool compute_boundary_info();

    // 辅助函数
    int get_nodes_count_by_element_type(int element_type);
    void reorder_polygon_to_ccw(std::vector<int>& nodes, const std::vector<double>& global_node_coords);
    double polygon_signed_area(const std::vector<double>& x, const std::vector<double>& y);
    void compute_mesh_bounds();
    int edge_hash_key(int a, int b) const;

    // 验证输入参数
    bool validate_input(int cell_id, int local_edge_id) const;
};

} // namespace vem

#endif // STRAIGHT_MESH_H
