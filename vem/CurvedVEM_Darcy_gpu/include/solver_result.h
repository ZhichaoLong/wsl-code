/** 求解结果的 CPU 小型结构：主函数直接取误差和耗时，不解析输出文件。 */
#pragma once
#include <string>
#include <vector>
#include <iosfwd>

namespace vgpu {
// comp 为计算域；phys 为物理域。每个数组按组分排列，无真解时误差为 NaN。
struct ErrorReport {
    std::vector<double> scalar_comp, flux_comp, scalar_phys, flux_phys, div_phys, mass;
};

struct CaseResult {
    int cells = 0, dofs = 0, steps = 0, linear_iterations = 0;
    double time = 0, residual = 0;
    double setup_seconds = 0;     // 网格读取（若内部读取）与 GPU 算子准备。
    double iteration_seconds = 0; // 所有时间步的系数更新、右端、Picard 和 FGMRES。
    double ksp_seconds = 0; // 全部 KSPSolve（含预条件），是 iteration_seconds 的子项。
    double linear_solve_seconds = 0; // 全部 Krylov::solve，另外含 COO/排序/置换/真残差。
    double error_seconds = 0;     // GPU 误差/质量归约及小型报告下载。
    double solution_seconds = 0;  // 最终解下载与写文件；禁用保存时为 0。
    ErrorReport errors;
    std::vector<double> initial_mass;
    std::string failure; // 失败原因；非空时不得参与收敛阶计算。
};
// 打印各组分计算域/物理域误差、质量与分阶段墙钟耗时。
void print_case_result(std::ostream& out, const CaseResult& result);
} // namespace vgpu
