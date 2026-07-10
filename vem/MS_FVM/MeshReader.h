#ifndef MESH_READER_H
#define MESH_READER_H
#include <sstream>   // 必须包含：std::istringstream的定义
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <algorithm> //std::reverse

// 定义三维坐标结构体（推荐）
struct Point2D {
    double x;
    double y;

    // 可选：构造函数，方便初始化
    Point2D(double x_val = 0.0, double y_val = 0.0)
        : x(x_val), y(y_val) {}
};

struct MeshData{
    //全局节点存储坐标向量
    std::vector<double> global_node_coords; // 存储所有节点的坐标，格式为 [x0, y0, x1, y1, ..., xN, yN]
    // 单元相关
    std::vector<int> global_nodes;          // 单元的顶点全局编号（按逆时针排序）
    std::vector<int> node_indices;          // 每个单元在global_nodes中的起始索引
    std::vector<int> nodes_per_element;     // 每个单元的顶点数（动态，数组）
    std::vector<int> num_edge;              // 每个单元的边数（等于顶点数，数组）
    std::vector<double> centroid_x;         // 单元质心x坐标
    std::vector<double> centroid_y;         // 单元质心y坐标
    std::vector<double> area;               // 单元面积
    std::vector<double> diameter;           // 单元最长直径
    int total_elements = 0;                // 单元总数
    int total_nodes = 0;                    // 总顶点数（无内点）

    // 边相关
    std::vector<int> elem_edge_indices;     // 每个单元在global_edges中的起始索引
    std::vector<int> global_edges;          // 单元的边全局编号（按逆时针排序）
    std::vector<int> edge_endpoints;        // 边的端点列表（[小端点, 大端点]，每个边占2位）
    int total_edges = 0;                    // 全局边总数（去重后）
    std::vector<int> edge_occurrence;       // 边的出现次数（用于判断边界边：出现1次的是边界边）

    // 边界点相关
    std::vector<int> boundary_nodes;        // 逆时针排序的边界点（含首尾闭合）
    std::vector<int> bnode_indices;         // 边界点索引（下/右/上/左边界起始索引）
    int total_boundary_nodes = 0;           // 边界点总数

    // 边界边相关
    std::vector<int> boundary_edges;        // 逆时针排序的边界边（全局编号）
    std::vector<int> bedge_indices;         // 边界边索引（下/右/上/左边界起始索引）
    int total_boundary_edges = 0;           // 边界边总数

    // 网格边界范围（自动计算）
    double x_min = 0.0, x_max = 0.0;
    double y_min = 0.0, y_max = 0.0;

    // 清空数据
    void clear() {
        global_node_coords.clear();
        global_nodes.clear();
        node_indices.clear();
        nodes_per_element.clear();
        num_edge.clear();
        centroid_x.clear();
        centroid_y.clear();
        area.clear();
        diameter.clear();
        total_elements = 0;
        total_nodes = 0;

        elem_edge_indices.clear();
        global_edges.clear();
        edge_endpoints.clear();
        total_edges = 0;
        edge_occurrence.clear();

        boundary_nodes.clear();
        bnode_indices.clear();
        total_boundary_nodes = 0;

        boundary_edges.clear();
        bedge_indices.clear();
        total_boundary_edges = 0;

        x_min = x_max = y_min = y_max = 0.0;
    }
};
// 哈希表节点（用于边去重）
struct HashNode {
    int key;                // 边的哈希键（由两个端点生成）
    int edge_id;            // 边的全局编号
    HashNode* next = nullptr;  // 链表下一个节点
    
    HashNode(int k, int id) : key(k), edge_id(id), next(nullptr) {}
};

// 哈希表结构
struct HashTable {
    std::vector<HashNode*> buckets;     // 桶数组
    int size = 0;                       // 桶数量
    
    HashTable(int s) : size(s) {
        buckets.resize(s, nullptr);
    }
    
    ~HashTable() {
        for (auto node : buckets) {
            while (node) {
                HashNode* temp = node;
                node = node->next;
                delete temp;
            }
        }
    }
};

// 边界边结构体（用于排序）
struct BoundaryEdge {
    int edge_id = 0;            // 边全局编号
    int node1 = 0, node2 = 0;   // 端点
    double x1 = 0.0, y1 = 0.0;  // 端点1坐标
    double x2 = 0.0, y2 = 0.0;  // 端点2坐标
    double mid_x = 0.0, mid_y = 0.0;  // 边中点坐标
};

//开始写网格数据读取的类
class MeshReader {
public:
    //构造函数
    MeshReader() = default;
    //析构函数
    ~MeshReader() = default;
    // 新增：读取Gmsh 4.1格式的.msh文件中的节点坐标（C++版本）
    // 返回值：true表示读取成功，false失败（替代C语言的NULL返回）
    bool readCoords(const std::string& filename);
    // 读取并填充网格单元数据（补充单元、索引、边界等信息）
    bool readElements(const std::string& filename);
    // 新增：计算单元属性（质心、面积、直径）
    bool computeElementProperties();
    //生成单元边的拓扑信息
    bool generateElementEdges();
    // 新增：获取网格数据（供外部访问读取后的坐标）
    const MeshData& getMeshData() const { return meshData; }
    // 计算边界信息（主函数）
    bool computeBoundaryInfo();
    // 一行代码生成完整网格数据
    bool generateCompleteMesh(const std::string& filename);
    //查找单元局部边对应的相邻单元,输入参数是单元编号和单元的局部编号0 1 2 3这样，输出相邻的单元，如果没有相邻单元就输出自己。
    int findAdjacentElement(int element_id, int local_edge_id) const;//加了const才能用指针类访问使用
    // 输入边的全局编号，输出相邻的两个单元。如果是边界边，两个单元相同
    std::pair<int, int> findAdjacentElementsByEdge(int global_edge_id) const;
private:
    MeshData meshData; //网格数据
    // 私有辅助函数
    // 根据Gmsh单元类型获取顶点数
    int getNodesCountByElementType(int element_type);
    // 将多边形节点逆时针排序
    void reorderPolygonToCCW(std::vector<int>& nodes, const std::vector<double>& global_node_coords);
    // 计算多边形有向面积（用于判断顺逆时针）
    double polygonSignedArea(const std::vector<double>& x, const std::vector<double>& y);
    // 计算网格边界范围
    void computeMeshBounds();
    // 边哈希相关私有辅助函数（替代C的static函数）
    int edgeHashKey(int a, int b) const;                  // 生成边的哈希键
    int hashTableFind(HashTable& table, int a, int b) const; // 查找边的全局编号
    void hashTableInsert(HashTable& table, int a, int b, int edge_id); // 插入边到哈希表
    // 判断节点是否在网格边界上
    bool isBoundaryNode(int node_id, double tol = 1e-6) const;
     // 获取边界边对应的两个端点（保持逆时针顺序）
    std::pair<int, int> getBoundaryEdgeEndpoints(int boundary_edge_idx) const;
    //检查输入参数合法性
    bool validateInput(int element_id, int local_edge_id) const;
};

#endif // MESH_READER_H