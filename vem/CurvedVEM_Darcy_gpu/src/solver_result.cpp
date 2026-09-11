/** 单网格结果的终端与日志输出；不依赖 CUDA，也不从 JSON 重新读取结果。 */
#include "solver_result.h"
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <ostream>

namespace vgpu {
void print_case_result(std::ostream& out, const CaseResult& r) {
    const auto flags = out.flags();
    const auto precision = out.precision();
    out << "\n最终结果：cells=" << r.cells << " dofs=" << r.dofs << " T=" << std::scientific
        << std::setprecision(6) << r.time << "，真实相对残差=" << r.residual << '\n';
    const auto& e = r.errors;
    for (size_t i = 0; i < e.mass.size(); ++i) {
        out << "  组分 " << i;
        if (std::isfinite(e.scalar_comp[i])) {
            out << "  计算域 L2: 标量=" << e.scalar_comp[i] << " 通量=" << e.flux_comp[i]
                << "\n          物理域 L2: 标量=" << e.scalar_phys[i] << " 通量=" << e.flux_phys[i]
                << " 散度=" << e.div_phys[i];
        } else {
            out << "  无解析解：不报告解析误差与收敛阶";
        }
        out << "\n          物理质量=" << e.mass[i];
        if (i < r.initial_mass.size())
            out << " 初始质量=" << r.initial_mass[i] << " 漂移=" << e.mass[i] - r.initial_mass[i];
        out << '\n';
    }
    out << std::fixed << std::setprecision(4) << "  准备=" << r.setup_seconds
        << " s；迭代=" << r.iteration_seconds << " s；误差计算=" << r.error_seconds
        << " s；平均每步迭代=" << r.iteration_seconds / std::max(1, r.steps)
        << " s；PETSc KSP=" << r.ksp_seconds << " s；线性求解流程=" << r.linear_solve_seconds
        << " s；解下载/保存=" << r.solution_seconds << " s；线性迭代总数=" << r.linear_iterations << '\n'
        << std::flush;
    out.flags(flags);
    out.precision(precision);
}
} // namespace vgpu
