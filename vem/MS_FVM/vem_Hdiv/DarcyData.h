#ifndef DARCY_DATA_H
#define DARCY_DATA_H

#include <cmath>
#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <functional>

// ============================================================================
// 函数类型定义（放在命名空间中避免与 pde_data.h 冲突）
// ============================================================================

namespace darcy {
    using ScalarFunc = std::function<double(double x, double y)>;
    using VectorFunc = std::function<std::vector<double>(double x, double y)>;
    using TensorFunc = std::function<double(int i, int j, double x, double y)>;
    using BoundaryFunc = std::function<double(int boundary_id, double x, double y)>;
}

// ============================================================================
// Darcy方程数据结构体
// ============================================================================

struct DarcyPdeData {
    // 扩散张量 κ(x,y) - 2x2对称张量
    darcy::TensorFunc kappa = nullptr;

    // 对流向量 b(x,y) - 2维向量
    darcy::VectorFunc b_coeff = nullptr;

    // 反应系数 γ(x,y) - 标量
    darcy::ScalarFunc gamma_func = nullptr;

    // Dirichlet边界条件
    darcy::BoundaryFunc dirichlet_bc = nullptr;

    // 真解 p(x,y) - 标量
    darcy::ScalarFunc solution_p = nullptr;

    // 真解通量 u = -κ∇p + b p - 2维向量
    darcy::VectorFunc solution_u = nullptr;

    // 右端源项 f(x,y) - 标量
    darcy::ScalarFunc source_func = nullptr;
};

// ============================================================================
// Darcy数据管理类
// ============================================================================

class DarcyData {
public:
    DarcyData() : pde_index_(0), darcy_data_(nullptr) {}
    ~DarcyData() = default;

    // 禁用拷贝
    DarcyData(const DarcyData&) = delete;
    DarcyData& operator=(const DarcyData&) = delete;

    // 设置算例编号
    void setPdeIndex(int index) { pde_index_ = index; }

    // 初始化数据
    bool initPdeData();

    // 获取Darcy数据（只读）
    const DarcyPdeData* getDarcyPdeData() const {
        return darcy_data_.get();
    }

private:
    // 算例编号
    int pde_index_;

    // Darcy数据智能指针
    std::unique_ptr<DarcyPdeData> darcy_data_;

    // 初始化第i个算例
    bool initDarcyPdeData(int index);
};

// ============================================================================
// Darcy方程实现（内联函数，方便头文件直接使用）
// ============================================================================

inline bool DarcyData::initPdeData() {
    darcy_data_.reset();
    return initDarcyPdeData(pde_index_);
}

inline bool DarcyData::initDarcyPdeData(int index) {
    darcy_data_ = std::make_unique<DarcyPdeData>();
    if (!darcy_data_) {
        std::cerr << "错误：内存分配失败" << std::endl;
        return false;
    }

    switch(index) {
        case 1: {
            // 算例1：非对称张量，有对流和反应项
            darcy_data_->kappa = [](int i, int j, double x, double y) -> double {
                if (i == 0 && j == 0) return y*y + 1.0;
                if (i == 0 && j == 1) return -x*y;
                if (i == 1 && j == 0) return -x*y;
                if (i == 1 && j == 1) return x*x + 1.0;
                return 0.0;
            };

            darcy_data_->b_coeff = [](double x, double y) -> std::vector<double> {
                return {x, y};
            };

            darcy_data_->gamma_func = [](double x, double y) -> double {
                return x*x + y*y*y;
            };

            darcy_data_->dirichlet_bc = [](int boundary_id, double x, double y) -> double {
                return 0.0;
            };

            darcy_data_->solution_p = [](double x, double y) -> double {
                return x*x*y + std::sin(2*M_PI*x)*std::sin(2*M_PI*y) + 2.0;
            };

            darcy_data_->solution_u = [](double x, double y) -> std::vector<double> {
                // u = -κ∇p + b p
                double p = x*x*y + std::sin(2*M_PI*x)*std::sin(2*M_PI*y) + 2.0;
                double dp_dx = 2*x*y + 2*M_PI*std::cos(2*M_PI*x)*std::sin(2*M_PI*y);
                double dp_dy = x*x + 2*M_PI*std::sin(2*M_PI*x)*std::cos(2*M_PI*y);

                double k11 = y*y + 1.0;
                double k12 = -x*y;
                double k21 = -x*y;
                double k22 = x*x + 1.0;

                double kappa_nabla_p_x = k11 * dp_dx + k12 * dp_dy;
                double kappa_nabla_p_y = k21 * dp_dx + k22 * dp_dy;

                double b1 = x, b2 = y;
                double bp_x = b1 * p;
                double bp_y = b2 * p;

                return {-kappa_nabla_p_x + bp_x, -kappa_nabla_p_y + bp_y};
            };

            darcy_data_->source_func = [](double x, double y) -> double {
                // p = x²y + sin(2πx)sin(2πy) + 2
                double p = x*x*y + std::sin(2*M_PI*x)*std::sin(2*M_PI*y) + 2.0;
                double dp_dx = 2*x*y + 2*M_PI*std::cos(2*M_PI*x)*std::sin(2*M_PI*y);
                double dp_dy = x*x + 2*M_PI*std::sin(2*M_PI*x)*std::cos(2*M_PI*y);

                double d2p_dx2 = 2*y - 4*M_PI*M_PI*std::sin(2*M_PI*x)*std::sin(2*M_PI*y);
                double d2p_dy2 = -4*M_PI*M_PI*std::sin(2*M_PI*x)*std::sin(2*M_PI*y);
                double d2p_dxdy = 2*x + 4*M_PI*M_PI*std::cos(2*M_PI*x)*std::cos(2*M_PI*y);

                double k11 = y*y + 1.0, k12 = -x*y;
                double k21 = -x*y, k22 = x*x + 1.0;

                double dk11_dy = 2*y;
                double dk12_dx = -y, dk12_dy = -x;
                double dk21_dx = -y, dk21_dy = -x;
                double dk22_dx = 2*x;

                double div_kappa_nabla_p =
                    dk11_dy * dp_dx + k11 * d2p_dx2 +
                    dk12_dy * dp_dy + k12 * d2p_dxdy +
                    dk21_dx * dp_dx + k21 * d2p_dxdy +
                    dk22_dx * dp_dy + k22 * d2p_dy2;

                double b1 = x, b2 = y;
                double db1_dx = 1.0, db2_dy = 1.0;
                double div_bp = db1_dx * p + b1 * dp_dx + db2_dy * p + b2 * dp_dy;

                double divergence = -div_kappa_nabla_p + div_bp;

                double gamma = x*x + y*y*y;
                double gamma_p = gamma * p;

                return divergence + gamma_p;
            };
            break;
        }
        case 2: {
            // 算例2：简单泊松方程
            darcy_data_->kappa = [](int i, int j, double x, double y) -> double {
                return (i == j) ? 1.0 : 0.0;
            };

            darcy_data_->b_coeff = [](double x, double y) -> std::vector<double> {
                return {0.0, 0.0};
            };

            darcy_data_->gamma_func = [](double x, double y) -> double {
                return 0.0;
            };

            darcy_data_->dirichlet_bc = [](int boundary_id, double x, double y) -> double {
                return 0.0;
            };

            darcy_data_->solution_p = [](double x, double y) -> double {
                return std::sin(M_PI*x)*std::sin(M_PI*y);
            };

            darcy_data_->solution_u = [](double x, double y) -> std::vector<double> {
                double dp_dx = M_PI*std::cos(M_PI*x)*std::sin(M_PI*y);
                double dp_dy = M_PI*std::sin(M_PI*x)*std::cos(M_PI*y);
                return {-dp_dx, -dp_dy};
            };

            darcy_data_->source_func = [](double x, double y) -> double {
                return 2*M_PI*M_PI*std::sin(M_PI*x)*std::sin(M_PI*y);
            };
            break;
        }
        case 3: {
            // 算例3：余弦真解
            darcy_data_->kappa = [](int i, int j, double x, double y) -> double {
                return (i == j) ? 1.0 : 0.0;
            };

            darcy_data_->b_coeff = [](double x, double y) -> std::vector<double> {
                return {0.0, 0.0};
            };

            darcy_data_->gamma_func = [](double x, double y) -> double {
                return 0.0;
            };

            darcy_data_->dirichlet_bc = [](int boundary_id, double x, double y) -> double {
                return 0.0;
            };

            darcy_data_->solution_p = [](double x, double y) -> double {
                return std::cos(M_PI*x)*std::cos(M_PI*y);
            };

            darcy_data_->solution_u = [](double x, double y) -> std::vector<double> {
                double dp_dx = M_PI*std::sin(M_PI*x)*std::cos(M_PI*y);
                double dp_dy = M_PI*std::cos(M_PI*x)*std::sin(M_PI*y);
                return {-dp_dx, -dp_dy};
            };

            darcy_data_->source_func = [](double x, double y) -> double {
                return 2*M_PI*M_PI*std::cos(M_PI*x)*std::cos(M_PI*y);
            };
            break;
        }
        case 4: {
            // 算例4：sin-cos混合真解
            darcy_data_->kappa = [](int i, int j, double x, double y) -> double {
                return (i == j) ? 1.0 : 0.0;
            };

            darcy_data_->b_coeff = [](double x, double y) -> std::vector<double> {
                return {0.0, 0.0};
            };

            darcy_data_->gamma_func = [](double x, double y) -> double {
                return 0.0;
            };

            darcy_data_->dirichlet_bc = [](int boundary_id, double x, double y) -> double {
                return 0.0;
            };

            darcy_data_->solution_p = [](double x, double y) -> double {
                return std::sin(M_PI*x)*std::cos(M_PI*y);
            };

            darcy_data_->solution_u = [](double x, double y) -> std::vector<double> {
                double dp_dx = M_PI*std::cos(M_PI*x)*std::cos(M_PI*y);
                double dp_dy = -M_PI*std::sin(M_PI*x)*std::sin(M_PI*y);
                return {-dp_dx, -dp_dy};
            };

            darcy_data_->source_func = [](double x, double y) -> double {
                return 2*M_PI*M_PI*std::sin(M_PI*x)*std::cos(M_PI*y);
            };
            break;
        }
        default:
            std::cerr << "错误：未知的算例编号 " << index << std::endl;
            darcy_data_.reset();
            return false;
    }

    return true;
}

#endif // DARCY_DATA_H
