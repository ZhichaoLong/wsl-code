/**
 * @file test_ms_initial_main.cpp
 * @brief 测试 MS 物理域初值函数与计算域初值的一致性
 *
 * 对 [0,1]×[0,1] 内的采样点 (ξ,η)：
 *   1. 用计算域初值函数直接求值：c_comp = initial_concentration_comp(i, ξ, η)
 *   2. 用正映射得到物理坐标：(x,y) = mapping.physical_coords(ξ, η)
 *   3. 用物理域初值函数求值：c_phys = initial_concentration(i, x, y)
 *   4. 对比 c_comp 与 c_phys，应该相等（标量点值映射不变性）
 */

#include "../examples/ms_problem.h"

#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    using namespace MaxwellStefan;

    MSProblem problem;
    problem.set_problem_index(1);
    if (!problem.init_problem()) {
        std::cerr << "MSProblem 初始化失败！" << std::endl;
        return 1;
    }

    const auto& pde = problem.get_pde_data();
    const auto& mapping = problem.get_mapping();

    std::cout << "===== 测试：物理域初值 vs 计算域初值 =====\n";
    std::cout << "采样网格：11 x 11（等间隔）\n\n";

    const int N = 11;
    double max_err = 0.0;
    int total = 0;
    int fail = 0;

    for (int i = 0; i < pde.n_components; ++i) {
        std::cout << "--- 组分 " << i << " ---\n";
        for (int ix = 0; ix < N; ++ix) {
            for (int iy = 0; iy < N; ++iy) {
                double xi = (double)ix / (N - 1);
                double eta = (double)iy / (N - 1);

                // 计算域初值
                double c_comp = pde.initial_concentration_comp(i, xi, eta);

                // 正映射到物理域
                Point2D phys = mapping.physical_coords(xi, eta);

                // 物理域初值（内部通过逆映射反求 ξ,η）
                double c_phys = pde.initial_concentration(i, phys.x, phys.y);

                double err = std::abs(c_comp - c_phys);
                max_err = std::max(max_err, err);
                total++;

                if (err > 1e-10) {
                    fail++;
                    if (fail <= 10) {
                        std::cout << "  ❌ 不匹配: (ξ,η)=(" << xi << "," << eta
                                  << ") (x,y)=(" << std::setprecision(6)
                                  << phys.x << "," << phys.y << ")"
                                  << " c_comp=" << c_comp
                                  << " c_phys=" << c_phys
                                  << " err=" << err << "\n";
                    }
                }
            }
        }
    }

    std::cout << "\n===== 结果 =====\n";
    std::cout << "总采样点数: " << total << "\n";
    std::cout << "最大误差: " << max_err << "\n";
    std::cout << "不匹配点数: " << fail << "\n";

    if (max_err < 1e-10) {
        std::cout << "✅ 测试通过：物理域初值与计算域初值一致\n";
        return 0;
    } else {
        std::cout << "❌ 测试失败\n";
        return 1;
    }
}
