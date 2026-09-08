#ifndef CURVEDVEM_MESH_REFINER_H
#define CURVEDVEM_MESH_REFINER_H

#include <string>
#include <vector>

namespace vem {

class StraightMeshReader;
struct StraightMeshData;

// ============================================================================
// 悬点网格生成
//
// 背景：虚拟元法适配任意多边形单元，所以「悬点」不需要特殊的约束方程去处理——
//   一个四边形单元若某条边上多出一个悬点，直接把它当五边形单元求解即可，
//   悬点自动升格为该单元的一个顶点。本模块就是把这个想法落成网格文件。
//
// 做法：对输入的结构化四边形网格，把落在指定 x 条带内的单元四分加密
//   （1 -> 4），条带外的单元保持不变。加密区与非加密区的交界上，
//   粗单元那一侧就出现悬点：交界边被细单元分成两半，中点即悬点。
//   于是该粗单元从四边形变成五边形。
//
// 为什么不写回 .msh：Gmsh 格式没有通用多边形单元类型（type 2 = 三角形，
//   type 3 = 四边形，再往上只有高阶单元，没有 n 边形），五边形无法表达。
//   故改写 VTK（VTK_POLYGON 支持任意顶点数）与一个极简文本格式。
//
// 本模块只负责生成网格文件，不参与任何求解流程，也不依赖 PETSc。
// ============================================================================

// 多边形网格：单元节点数可变，这是与 StraightMeshData 的本质区别。
struct PolyMeshData {
    // 节点坐标，[x0, y0, x1, y1, ...]
    std::vector<double> node_coords;
    int num_nodes = 0;

    // 单元节点全局编号，逆时针。变长，故需要 offsets 定位。
    std::vector<int> cell_nodes;
    std::vector<int> cell_offsets;    // 长度 num_cells + 1，第 e 个单元占
                                      // [cell_offsets[e], cell_offsets[e+1])
    int num_cells = 0;

    int nodes_of_cell(int e) const {
        return cell_offsets[e + 1] - cell_offsets[e];
    }

    void clear() {
        node_coords.clear();
        num_nodes = 0;
        cell_nodes.clear();
        cell_offsets.clear();
        num_cells = 0;
    }
};

// 加密结果的统计信息，供调用方打印与自检。
struct RefineStats {
    int num_cells_in  = 0;   // 输入单元数
    int num_cells_out = 0;   // 输出单元数
    int num_refined   = 0;   // 被四分的单元数
    int num_hanging   = 0;   // 悬点总数
    int num_quads     = 0;   // 输出中四边形数
    int num_pentagons = 0;   // 输出中五边形数
    int num_other     = 0;   // 输出中其他边数的多边形数（正常应为 0）
    int max_poly      = 0;   // 输出中最大单元边数
};

// 对 x 落在 [x_lo, x_hi] 条带内的单元做一次四分加密，生成带悬点的多边形网格。
//
// 判据是「单元的**所有**节点 x 都在条带内」而非质心在条带内：条带边界
//   x = x_lo / x_hi 恰好落在格线上时，两种判据结果相同；不落在格线上时，
//   全节点判据不会切出半个单元，行为更可预期。
//
// tol 是坐标比较容差。**必须给**：mesh/data/*.msh 里 0.25 实际存的是
//   0.2499999999993471，精确比较会把本该加密的一整列单元漏掉。
//
// in 必须是四边形网格；遇到非四边形单元返回 false。
bool refine_x_strip(const StraightMeshData& in,
                    double x_lo, double x_hi,
                    PolyMeshData& out,
                    RefineStats& stats,
                    double tol = 1e-8);

// 写 legacy VTK（ASCII，UNSTRUCTURED_GRID，全部单元用 VTK_POLYGON = 7）。
// 四边形也写成 4 顶点多边形，避免混用 VTK_QUAD 造成读取端分支。
bool write_vtk(const PolyMeshData& mesh, const std::string& filename,
               const std::string& title = "CurvedVEM hanging-node mesh");

// 写极简文本格式，供后续自己写 reader 用：
//   NODES <n>
//   <x> <y>            × n
//   CELLS <m>
//   <k> <n0> ... <nk-1> × m
bool write_poly(const PolyMeshData& mesh, const std::string& filename);

// 自检：验证输出网格的拓扑一致性。返回问题条数，0 表示全部通过。
// 检查项见实现文件。messages 收集人类可读的问题描述。
int validate_poly_mesh(const PolyMeshData& mesh,
                       std::vector<std::string>& messages,
                       double tol = 1e-8);

}  // namespace vem

#endif  // CURVEDVEM_MESH_REFINER_H
