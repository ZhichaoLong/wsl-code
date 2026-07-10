#ifndef PDE_DATA1D_H
#define PDE_DATA1D_H

#include <cmath>
#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <functional>

// ============================================================================
// 函数指针类型定义（一维版本）
// ============================================================================

using ScalarFunc1D = std::function<double(int num, double x)>;
using BoundaryFunc1D = std::function<double(int num, int boundary_id, double x)>;
using MatrixCoeffFunc1D = std::function<double(int i, int j, double x)>;
using MatrixCoeffFunc1D_mid = std::function<double(int i, int j, const std::vector<double>& u_avg)>;
// 新增：A对u的雅可比函数类型，返回 dA_ij/du_k
using MatrixJacFunc1D = std::function<double(int i, int j, int k, const std::vector<double>& u_avg)>;

// ============================================================================
// 一维Maxwell-Stefan方程数据结构体
// ============================================================================
struct MSPdeData1D {
    int n_components = 0;                           // 组分数
    std::vector<double> diff_coeff;                 // 扩散系数，一维存储 D_12, D_13, D_23...
    double diff_coeff_min = 0.0;                    // 扩散系数最小值
    double diff_coeff_max = 0.0;                    // 扩散系数最大值
    ScalarFunc1D init_func = nullptr;               // 初始函数
    BoundaryFunc1D boundary_func = nullptr;         // 边界函数
    MatrixCoeffFunc1D init_A_coeff = nullptr;       // 初值扩散非线性系数矩阵
    // 注意：mid_A_coeff需要长度为n_components的向量输入，代表每个组分的单元均值
    MatrixCoeffFunc1D_mid mid_A_coeff = nullptr;    // 单元均值计算的非线性系数矩阵
    MatrixJacFunc1D jac_A_coeff = nullptr;          // 新增：A对u的雅可比 dA_ij/du_k
};

// ============================================================================
// 一维PDE数据管理类
// ============================================================================

class PdeData1D {
public:
    PdeData1D() : pde_type_(0), pde_index_(0), ms_data_(nullptr) {}
    ~PdeData1D() = default;
    // 禁用拷贝
    PdeData1D(const PdeData1D&) = delete;
    PdeData1D& operator=(const PdeData1D&) = delete;

    // 设置PDE类型和编号，目前0-MS方程
    void setPdeType(int type) { pde_type_ = type; }
    // 该pde类型下的第几个算例
    void setPdeIndex(int index) { pde_index_ = index; }

    // 初始化数据
    bool initPdeData();

    // 获取MSPDE数据（只读）
    const MSPdeData1D* getMSPdeData() const {
        return ms_data_.get();
    }

private:
    // 私有成员变量
    int pde_type_;          // PDE类型
    int pde_index_;         // 算例编号
    std::unique_ptr<MSPdeData1D> ms_data_;  // 智能指针管理MS方程数据

    // 具体初始化方法
    bool initMSPdeData();
};

#endif // PDE_DATA1D_H
