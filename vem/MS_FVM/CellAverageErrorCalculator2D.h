#ifndef CELL_AVERAGE_ERROR_CALCULATOR_2D_H
#define CELL_AVERAGE_ERROR_CALCULATOR_2D_H

#include <vector>
#include <string>
#include <utility>

/**
 * @brief 简单的二维网格单元数据结构
 */
struct CellData2D {
    int id;                 // 单元编号
    int n_nodes;            // 顶点数
    std::vector<double> vertices;  // 顶点坐标 [x0,y0,x1,y1,...]
    double centroid_x;      // 质心x坐标
    double centroid_y;      // 质心y坐标
    double area;            // 单元面积
};

/**
 * @brief 简单的二维网格数据结构
 */
struct SimpleMeshData2D {
    int nx;                 // x方向单元数
    int ny;                 // y方向单元数
    int total_elements;     // 总单元数
    double h;               // 单元尺寸
    std::vector<CellData2D> cells;  // 所有单元数据

    void clear() {
        nx = ny = total_elements = 0;
        h = 0.0;
        cells.clear();
    }
};

/**
 * @brief 二维单元平均误差计算类
 *
 * 实现逻辑：
 * - 使用最密网格（128x128）作为参考解
 * - 对于每个粗网格单元，其单元中心恰好落在细网格的四个单元的公共顶点上
 * - 在该公共顶点位置，取相邻四个细网格单元的平均值作为参考值
 * - 计算粗网格数值解与参考解之间的 L2 误差
 *
 * 使用方法：
 * 1. 以最细网格（128x128）作为参考真解构造对象
 * 2. 调用 computeErrorForAllGrids() 计算所有网格的误差和收敛阶
 */
class CellAverageErrorCalculator2D {
public:
    /**
     * @brief 构造函数
     * @param fine_mesh_file 最细网格文件路径（用作真解）
     * @param fine_solution_file 最细网格解文件路径
     * @param n_components 组分数（默认为3）
     */
    CellAverageErrorCalculator2D(const std::string& fine_mesh_file,
                                  const std::string& fine_solution_file,
                                  int n_components = 3);

    ~CellAverageErrorCalculator2D() = default;

    // 禁止拷贝
    CellAverageErrorCalculator2D(const CellAverageErrorCalculator2D&) = delete;
    CellAverageErrorCalculator2D& operator=(const CellAverageErrorCalculator2D&) = delete;

    /**
     * @brief 从简化格式文件读取网格数据
     * @param filename 网格文件路径
     * @param mesh_data 输出的网格数据
     * @return 成功返回true
     */
    static bool readSimpleMesh(const std::string& filename, SimpleMeshData2D& mesh_data);

    /**
     * @brief 从文件加载解数据
     * @param filename 解文件路径
     * @param n_elements 单元总数
     * @param n_components 组分数
     * @return 解向量，格式：[所有单元组分0, 所有单元组分1, ...]
     */
    static std::vector<double> loadSolutionFromFile(const std::string& filename,
                                                      int n_elements,
                                                      int n_components);

    /**
     * @brief 找到四个相邻的细网格单元（它们共享一个顶点，该顶点是粗网格单元的中心）
     * @param x 粗网格单元中心的x坐标
     * @param y 粗网格单元中心的y坐标
     * @return 四个相邻细网格单元的索引
     *
     * 说明：对于结构化正方形网格，粗网格单元的中心 (x,y) 恰好是
     * 四个细网格单元的公共顶点。这四个细网格单元分别位于：
     * - 左下：单元质心在 (x - h_fine/2, y - h_fine/2)
     * - 右下：单元质心在 (x + h_fine/2, y - h_fine/2)
     * - 右上：单元质心在 (x + h_fine/2, y + h_fine/2)
     * - 左上：单元质心在 (x - h_fine/2, y + h_fine/2)
     * 其中 h_fine 是细网格的单元尺寸
     */
    std::vector<int> findFourNeighboringFineCells(double x, double y) const;

    /**
     * @brief 在给定的点（粗网格单元中心）获取细网格的参考值
     * @param x 点的x坐标
     * @param y 点的y坐标
     * @param comp 组分索引
     * @return 四个相邻细网格单元的平均值
     */
    double getFineReferenceValueAt(double x, double y, int comp) const;

    /**
     * @brief 计算单个粗网格相对于细网格真解的L2误差
     * @param coarse_mesh_file 粗网格文件
     * @param coarse_solution_file 粗网格解文件
     * @return 每个组分的L2误差
     */
    std::vector<double> computeL2Error(const std::string& coarse_mesh_file,
                                        const std::string& coarse_solution_file) const;

    /**
     * @brief 计算整个收敛阶（对一系列网格）
     * @param mesh_files 各级粗网格文件列表，从粗到细排列（不包含最细网格）
     * @param solution_files 对应各级粗网格的解文件列表
     * @param output_file 输出误差和收敛阶的文件路径
     * @return 每个网格每个组分的误差，最后一行是收敛阶
     */
    std::vector<std::vector<double>> computeConvergenceOrder(
        const std::vector<std::string>& mesh_files,
        const std::vector<std::string>& solution_files,
        const std::string& output_file) const;

    /**
     * @brief 为所有预设的网格计算误差和收敛阶
     * @param data_dir 数据文件夹路径（默认为 "error_data_new_0.0001"）
     * @param output_file 输出文件路径
     */
    void computeErrorForAllGrids(const std::string& data_dir = "error_data_new_0.0001",
                                  const std::string& output_file = "convergence_results.txt") const;

private:
    // 细网格（真解）数据
    SimpleMeshData2D fine_mesh_;          // 细网格数据
    std::vector<double> fine_solution_;    // 细网格解（用作参考真解）
    int n_components_;                      // 组分数
};

#endif // CELL_AVERAGE_ERROR_CALCULATOR_2D_H
