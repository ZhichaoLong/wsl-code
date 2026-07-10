#ifndef PDE_DATA_H
#define PDE_DATA_H

#include <cmath>
#include <iostream>
#include <vector>
#include <memory>  // 智能指针头文件
#include <stdexcept>//错误处理
#include <string>
#include <functional>  // 必须加这个头文件，可以捕获函数指针和lambda表达式
// ============================================================================
// 函数指针类型定义
// ============================================================================

using ScalarFunc = std::function<double(int num, double x, double y)>;
using VectorFunc = std::function<double*(int num, double x, double y)>;
using BoundaryFunc = std::function<double(int num, int boundary_id, double x, double y)>;
using MatrixCoeffFunc = std::function<double(int i, int j, double x, double y)>;
using MatrixCoeffFunc_mid = std::function<double(int i, int j, const std::vector<double>& u_avg)>;
// poly_A_coeff: 输入i,j,多项式系数向量, 质心, 直径, 计算点, 返回A_ij
using MatrixCoeffFunc_ploy = std::function<double(int i, int j, const std::vector<double>& u_ploy_coeff,
                                                   double xD_x, double xD_y, double hD, double x, double y)>;



// ============================================================================
// MaxwellStefan方程数据结构体
// ============================================================================
struct MSPdeData{
    int n_components = 0;//组分数
    std::vector<double> diff_coeff;//扩散系数，一维存储
    double diff_coeff_min = 0.0;//扩散系数最小值
    double diff_coeff_max = 0.0;//扩散系数最大值
    ScalarFunc init_func = nullptr;//初始函数
    BoundaryFunc boundary_func = nullptr;//边界函数
    MatrixCoeffFunc init_A_coeff = nullptr; // 扩散非线性系数函数，这是初值函数的计算方式
    //此处注意，这里的mid_A_coeff需要一个长度为n_components的向量输入，代表每个组分的单元均值，这样才能计算出正确的非线性系数值
    MatrixCoeffFunc_mid mid_A_coeff = nullptr; // 扩散非线性系数函数，这是中值函数的计算方式
    MatrixCoeffFunc_ploy poly_A_coeff = nullptr; // 扩散非线性系数函数，这是多项式插值计算方式
};

// ============================================================================
// PDE数据管理类
// ============================================================================

class PdeData {
public:
    PdeData() : pde_type_(0), pde_index_(0), ms_data_(nullptr) {}
    ~PdeData() = default;
    // 禁用拷贝（避免智能指针浅拷贝，推荐做法）
    PdeData(const PdeData&) = delete;
    PdeData& operator=(const PdeData&) = delete;

    // 设置PDE类型和编号，目前0-MS方程
    void setPdeType(int type) { pde_type_ = type; }
    //该pde类型下的第几个算例
    void setPdeIndex(int index) { pde_index_ = index; }
    // 算例PDE结构体，这里存储该算例的结构体数据

    // 初始化数据
    bool initPdeData();

    // 获取MSPDE数据（只读，返回裸指针保证兼容性）
    const MSPdeData* getMSPdeData() const {
        return ms_data_.get();
    }

private:
    //私有成员变量
    // PDE类型和编号
    int pde_type_;
    int pde_index_;
    //pde数据结构体指针
    std::unique_ptr<MSPdeData> ms_data_;  // 智能指针替代裸指针，自动管理内存
    // 具体初始化方法 - 用户自己实现
    bool initMSPdeData();
};

#endif // PDE_DATA_H
