#ifndef VEM_FVM_ERROR_H
#define VEM_FVM_ERROR_H

#include <vector>
#include <string>
#include "CellAverageErrorCalculator2D.h"

/**
 * @brief VEM 与 FVM 细网格参考解之间的误差计算类
 *
 * 实现逻辑：
 * - 使用 FVM 最密网格（128x128）作为参考解
 * - 读取 VEM 计算的单元中心值
 * - 对于每个 VEM 单元中心，取相邻四个 FVM 细网格单元的平均值作为参考值
 * - 计算 L2 误差
 */
class VemFvmErrorCalculator {
public:
    /**
     * @brief 构造函数
     * @param fine_mesh_file FVM 最细网格文件
     * @param fine_solution_file FVM 最细网格解文件
     * @param n_components 组分数（默认为3）
     */
    VemFvmErrorCalculator(const std::string& fine_mesh_file,
                         const std::string& fine_solution_file,
                         int n_components = 3);

    ~VemFvmErrorCalculator() = default;

    // 禁止拷贝
    VemFvmErrorCalculator(const VemFvmErrorCalculator&) = delete;
    VemFvmErrorCalculator& operator=(const VemFvmErrorCalculator&) = delete;

    /**
     * @brief 从 VEM 输出文件读取单元中心数据
     * @param vem_data_file VEM 数据文件 (如 vem_data/vem_test_4.txt)
     * @param cell_centroid_x 输出单元中心 x 坐标
     * @param cell_centroid_y 输出单元中心 y 坐标
     * @param cell_values 输出每个组分的单元中心值，格式 [n_cells x n_components]
     * @param h 输出网格尺寸
     * @return 成功返回 true
     */
    bool readVemCentroidData(const std::string& vem_data_file,
                           std::vector<double>& cell_centroid_x,
                           std::vector<double>& cell_centroid_y,
                           std::vector<std::vector<double>>& cell_values,
                           double& h) const;

    /**
     * @brief 计算 VEM 相对于 FVM 细网格的 L2 误差
     * @param vem_data_file VEM 数据文件
     * @return 每个组分的 L2 误差
     */
    std::vector<double> computeL2Error(const std::string& vem_data_file) const;

    /**
     * @brief 计算整个收敛阶（对一系列 VEM 网格）
     * @param vem_data_files VEM 数据文件列表（从粗到细）
     * @param output_file 输出文件
     * @return 误差和收敛阶
     */
    std::vector<std::vector<double>> computeConvergenceOrder(
        const std::vector<std::string>& vem_data_files,
        const std::string& output_file) const;

private:
    CellAverageErrorCalculator2D fvm_error_calc_;  // 使用已有的 FVM 误差计算器
    int n_components_;
};

#endif
