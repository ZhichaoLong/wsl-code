/**
 * @file test_ms_A_initial_main.cpp
 * @brief 测试物理域 A_ij 与计算域 A_ij 在对应点上的一致性
 *
 * 对 [0,1]×[0,1] 内的采样点 (ξ,η)：
 *   1. 计算域 A_ij = evaluate_A_comp_initial(i, j, ξ, η)
 *   2. 正映射得到物理坐标 (x,y) = physical_coords(ξ, η)
 *   3. 物理域 A_ij = evaluate_A_initial(i, j, x, y)
 *   4. 对比两者，应该一致（A 只依赖浓度点值，标量映射不变）
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
    int n = pde.n_components;

    std::cout << "===== 测试：物理域 A_ij vs 计算域 A_ij =====\n";
    std::cout << "采样网格：11 x 11（等间隔，跳过间断面附近）\n\n";

    const int N = 11;
    double max_err = 0.0;
    int total = 0;
    int fail = 0;

    const double disc_tol = 0.08;  // 跳过间断面 0.5 附近的点

    for (int ix = 0; ix < N; ++ix) {
        for (int iy = 0; iy < N; ++iy) {
            double xi = (double)ix / (N - 1);
            double eta = (double)iy / (N - 1);

            // 跳过间断面附近（避免逆映射浮点误差导致跳变）
            if (std::abs(xi - 0.5) < disc_tol || std::abs(eta - 0.5) < disc_tol)
                continue;

            // 本测试不加载网格，只在 [0,1]^2 上均匀采样，故单元编号统一传 0。
            // 当前三种映射（Sin/HalfAnnulus/Identity）都与单元无关，传 0 与传
            // 任何编号等价；将来引入分单元映射后，本测试需改成按单元逐个采样。
            const int cell_idx = 0;
            Point2D phys = mapping.physical_coords(cell_idx, xi, eta);

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    double A_comp =
                        pde.evaluate_A_comp_initial(i, j, cell_idx, xi, eta);
                    double A_phys =
                        pde.evaluate_A_initial(i, j, cell_idx, phys.x, phys.y);
                    double err = std::abs(A_comp - A_phys);

                    max_err = std::max(max_err, err);
                    total++;

                    if (err > 1e-9) {
                        fail++;
                        if (fail <= 10) {
                            std::cout << "  ❌ A(" << i << "," << j
                                      << ") 不匹配: (ξ,η)=(" << xi << "," << eta
                                      << ") (x,y)=(" << std::setprecision(6)
                                      << phys.x << "," << phys.y << ")"
                                      << " A_comp=" << A_comp
                                      << " A_phys=" << A_phys
                                      << " err=" << err << "\n";
                        }
                    }
                }
            }
        }
    }

    std::cout << "\n===== 结果 =====\n";
    std::cout << "总矩阵元采样数: " << total << "\n";
    std::cout << "最大误差: " << max_err << "\n";
    std::cout << "不匹配数: " << fail << "\n";

    if (max_err < 1e-9) {
        std::cout << "✅ 测试通过：物理域 A_ij 与计算域 A_ij 一致\n";
        return 0;
    } else {
        std::cout << "❌ 测试失败\n";
        return 1;
    }
}
