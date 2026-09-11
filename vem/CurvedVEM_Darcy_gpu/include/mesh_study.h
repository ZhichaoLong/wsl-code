/**
 * 单个算例的网格实验工具：网格尺度、与空间阶匹配的时间网格、累计收敛表。
 * main 显式配置算例/网格序列/输出；本模块不决定要运行哪些不同的方程算例。
 */
#pragma once
#include "solver_driver.h"
#include <limits>

namespace vgpu {
enum class MeshScale { Equivalent, MaxDiameter };   // sqrt(计算域面积/N) 或最大单元直径。
enum class ErrorDomain { Computational, Physical }; // 收敛表使用同一种误差范数。
enum class TimePolicy { Balanced, FixedTarget };    // dt目标=C*h^((k+1)/p)，或固定目标步长。

struct TimeOptions {
    double final_time = 0.1; // 所有网格统一的误差比较时刻 T。
    TimePolicy policy = TimePolicy::Balanced;
    double coefficient = 0.1; // 平衡时间误差时的 C。
    double fixed_dt = 0.001;  // FixedTarget 模式使用，不随 h 变化。
    int max_steps = 1000000;  // 超过上限明确失败，不静默增大 dt。
    bool smooth_steps = true; // 步数向上选 2^a*5^b；关闭则只使用 ceil。
};
struct TimeGrid {
    double target_dt = 0, dt = 0;
    int steps = 1;
};
struct MeshEntry {
    std::string label, file, output; // 显示名、原 CPU 网格文件、该网格独立输出目录。
};
struct StudyOptions {
    std::string title;
    std::vector<MeshEntry> meshes;
    MeshScale scale = MeshScale::Equivalent;
    ErrorDomain error_domain = ErrorDomain::Computational; // 与原 MS 的 L2 接口一致。
    TimeOptions time;
    std::string table_file = "output/convergence.csv";    // 每层完成即刷新。
    std::string terminal_file = "output/convergence.txt"; // 累计结果表及失败说明。
};
struct MeshMeasure {
    int cells = 0;
    double area = 0, h_equivalent = 0, h_max = 0;
};
struct MeshResult {
    std::string label;
    double h = 0, h_max = 0, wall_seconds = 0;
    TimeGrid time_grid;
    CaseResult solve;
    std::vector<double> scalar_order, flux_order; // 首层/无真解/不可比较时为 NaN。
};

/** BDF2 p=2（至少两步），Euler p=1；dt=T/N 保证同一终止时间，不加半步。 */
TimeGrid make_time_grid(double h, int k, bool bdf2, const TimeOptions& options);
/** 从实际网格测尺度；不依据文件名推断单元数，非均匀网格建议用最大直径。 */
MeshMeasure measure_mesh(const vem::StraightMeshReader& reader);
/** log(e_old/e_new)/log(h_old/h_new)，保留负阶和零阶；无效输入返回 NaN。 */
double observed_order(double old_h, double new_h, double old_error, double new_error);
/** 一个方程算例的逐网格实验；每层完成立即输出累计表，失败保留此前结果并停止。 */
int run_mesh_study(const SolverOptions& base, const StudyOptions& study);
} // namespace vgpu
