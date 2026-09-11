/**
 * GPU 混合虚拟元求解的主机接口。
 * main 负责选择算例和参数；run_case 负责初始化、时间推进及结果保存。
 * 主函数用普通 C++ 编译，并包含 PETSc/CUDA 的主机端类型与会话管理。
 */
#pragma once
#include "solver_result.h"
#include "petsc_gpu.h"
#include <string>
#include <vector>

namespace vem {
class StraightMeshReader;
}

namespace vgpu {

// 命令行帮助以正常栈展开结束，让 PetscSession 执行 Finalize。
struct HelpRequested {};

/** 输出目录内的文件名，可在 main 中逐项修改；五个名称必须互不相同。 */
struct OutputFiles {
    std::string log = "run.log";               // 资源信息、Picard 与 FGMRES 迭代日志。
    std::string history = "history.csv";       // 每步每组分的误差、残差和质量。
    std::string solution = "solution.csv";     // 最终全局自由度解。
    std::string report = "report.json";        // 最终收敛状态、误差和耗时。
    std::string parameters = "parameters.txt"; // 实际生效的全部运行参数。
};

/** 单个算例的完整配置。Darcy 为稳态；时间设置作用于 Maxwell–Stefan。 */
struct SolverOptions {
    // PETSc 算子/预条件，可在 main 修改；稀疏 GPU 装配默认开启。
    bool assembled = true;
    std::string petsc_pc = "ilu";
    std::string petsc_ordering = "auto"; // Darcy 用 natural 保留消元顺序，MS 用 ND 提高三角求解并行度。
    int ilu_levels = 0; // ILU/LU 因子用 PETSc CPU 后端，避开 cuSPARSE 更新路径；MatMult 仍在 GPU。

    std::string equation = "ms";                   // "darcy" 或 "ms"。
    int example = 2;                               // Darcy: 1/2；MS: 1/2/3/4。
    std::string mesh = "mesh/data/square_2x2.msh"; // 原 CPU 读取器支持的 msh/vtk。
    std::string output = "output/run";             // 本算例输出目录。
    OutputFiles files;                             // 目录内文件名称。

    int k = 1;         // H(div)–L² 离散阶次，支持 0..4。
    int nq = 0;        // 每三角形积分点数；0 自动选 9/12/36（k≤2/k=3/k=4）。
    int ng = 9;        // 每条边的 Gauss 点数，k+2 ≤ ng ≤ 10。
    double dt = 0.001; // MS 时间步长，终止时间为 dt*steps。
    int steps = 3;     // MS 时间步数；Darcy 始终只解一次。
    bool bdf2 = true;  // true: BDF2（首步 Euler）；false: 全部使用 Euler。

    int picard = 50;           // 每时间步最大 Picard 迭代次数。
    double picard_tol = 1e-9;  // 非线性迭代收敛容差。
    int maxit = 4000;          // 每次 FGMRES 求解的最大迭代次数。
    int restart = 80;          // FGMRES 重启长度，影响 Krylov 显存用量。
    double linear_tol = 1e-11; // 线性系统真实相对残差容差。

    int threads = 128;           // 每个 CUDA block 的线程数：32/64/128/256。
    int tile = 4096;             // 一个分桶每次 launch 最多处理的单元数。
    int benchmark = 0;           // 性能计时重复次数；0 表示关闭。
    bool benchmark_only = false; // true 只测算子性能；false 仍完整求解 PDE。
    bool save_solution = true;   // 收敛实验可关闭最终自由度下载/写盘，仍保存误差报告。
    int error_interval = 1;      // 每几步计算误差/质量；0 只在最终步计算，最终步始终计算。
    int print_interval = 1;      // 终端时间步进度间隔；0 仅首/末步。每步迭代日志仍落盘。
};

/** 检查参数、解析 nq=0；失败抛出异常，不创建输出或访问 GPU。 */
SolverOptions validated_options(SolverOptions options);

/** 命令行兼容接口，由 main/vem_gpu_cli_main.cpp 使用；普通 main 不调用它。 */
SolverOptions parse_options(int argc, char** argv, bool all_examples = false);

/** 完整求解一个算例并保存结果；成功返回 0，失败返回 1。 */
// prepared_mesh 可复用主函数已读取的网格；调用期间必须保持对象存活。
int run_case(const SolverOptions& options, CaseResult* result = nullptr,
             const vem::StraightMeshReader* prepared_mesh = nullptr);

/** 按 main 给出的顺序、独立参数和目录运行；失败后继续，汇总写入 summary_file。 */
int run_examples(const std::vector<SolverOptions>& cases, const std::string& summary_file);

/** 保留旧的统一参数六算例接口；需要逐例配置时使用 run_examples。 */
int run_all_examples(const SolverOptions& options);
} // namespace vgpu
