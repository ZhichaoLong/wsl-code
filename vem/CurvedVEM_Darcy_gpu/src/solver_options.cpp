/**
 * 求解参数：公共校验与可选的命令行兼容接口。
 * 正式算例在 main 中直接配置；这里不读取环境变量或暗中覆盖 main 的设置。
 */
#include "solver_driver.h"
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>

namespace vgpu {
// 仅兼容入口展示命令行帮助；普通主函数的参数说明直接写在源码中。
static void usage(bool all_examples) {
    if (all_examples)
        std::cout << "Run all six examples: Darcy 1/2 and MS 1/2/3/4.\n";
    std::cout << "Usage: " << (all_examples ? "vem_gpu_all" : "vem_gpu_cli");
    if (!all_examples)
        std::cout << " [--equation ms|darcy] [--example N]";
    std::cout << " [--mesh path.msh|path.vtk]\n"
              << "  --k 0..4 --nq 9|12|36 --ng 2..10 --dt .001 --steps 3 --scheme euler|bdf2\n"
              << "  --picard 50 --picard-tol 1e-9 --linear-tol 1e-11 --maxit 4000 --restart 80\n"
              << "  --operator assembled|shell --petsc-pc ilu|lu|jacobi|gamg --ilu-levels 0\n"
              << "  --petsc-ordering auto|natural|rcm|nd\n"
              << "  --threads 128 --tile 4096 --output "
              << (all_examples ? "output/all_examples" : "output/run") << '\n';
    if (!all_examples)
        std::cout << "  --benchmark 10 --benchmark-only\n";
    std::cout << "Mesh generation and static geometry stay on CPU. Assembly, Krylov vectors,\n"
              << "matrix-free application and error integration run on a single CUDA GPU.\n";
}
// 将旧命令行参数转换为同一个配置对象，最终统一调用公共校验。
SolverOptions parse_options(int argc, char** argv, bool all_examples) {
    SolverOptions o;
    if (all_examples)
        o.output = "output/all_examples";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help") {
            usage(all_examples);
            throw HelpRequested{};
        }
        if (all_examples &&
            (a == "--equation" || a == "--example" || a == "--benchmark" || a == "--benchmark-only"))
            throw std::invalid_argument(a + " is only available in the single-case vem_gpu_cli entry");
        if (a == "--benchmark-only") {
            o.benchmark_only = true;
            continue;
        }
        if (i + 1 >= argc)
            throw std::invalid_argument("missing value for " + a);
        std::string v = argv[++i];
        if (a == "--equation")
            o.equation = v;
        else if (a == "--petsc-pc")
            o.petsc_pc = v;
        else if (a == "--petsc-ordering")
            o.petsc_ordering = v;
        else if (a == "--ilu-levels")
            o.ilu_levels = std::stoi(v);
        else if (a == "--operator") {
            if (v != "assembled" && v != "shell") throw std::invalid_argument("operator must be assembled or shell");
            o.assembled = v == "assembled";
        }
        else if (a == "--mesh")
            o.mesh = v;
        else if (a == "--output")
            o.output = v;
        else if (a == "--example")
            o.example = std::stoi(v);
        else if (a == "--k")
            o.k = std::stoi(v);
        else if (a == "--nq") {
            o.nq = std::stoi(v);
        } else if (a == "--ng")
            o.ng = std::stoi(v);
        else if (a == "--steps")
            o.steps = std::stoi(v);
        else if (a == "--picard")
            o.picard = std::stoi(v);
        else if (a == "--maxit")
            o.maxit = std::stoi(v);
        else if (a == "--restart")
            o.restart = std::stoi(v);
        else if (a == "--threads")
            o.threads = std::stoi(v);
        else if (a == "--tile")
            o.tile = std::stoi(v);
        else if (a == "--benchmark")
            o.benchmark = std::stoi(v);
        else if (a == "--dt")
            o.dt = std::stod(v);
        else if (a == "--linear-tol")
            o.linear_tol = std::stod(v);
        else if (a == "--picard-tol")
            o.picard_tol = std::stod(v);
        else if (a == "--scheme") {
            if (v != "euler" && v != "bdf2")
                throw std::invalid_argument("scheme must be euler or bdf2");
            o.bdf2 = v == "bdf2";
        } else
            throw std::invalid_argument("unknown option " + a);
    }
    return validated_options(o);
}

// 返回已解析自动积分点数的副本；在创建输出和分配 GPU 内存前拒绝无效配置。
SolverOptions validated_options(SolverOptions o) {
    // Darcy 的零压力质量块/均值乘子需要通量先消元；MS 的正时间质量块使用 ND。
    if (o.petsc_ordering == "auto")
        o.petsc_ordering = o.equation == "darcy" ? "natural" : "nd";
    if (o.petsc_ordering != "natural" && o.petsc_ordering != "rcm" && o.petsc_ordering != "nd")
        throw std::invalid_argument("PETSc ordering must be natural/rcm/nd");
    if (o.ilu_levels < 0 || (o.petsc_pc != "ilu" && o.petsc_pc != "lu" && o.petsc_pc != "jacobi" && o.petsc_pc != "gamg"))
        throw std::invalid_argument("PETSc PC must be ilu/lu/jacobi/gamg, levels >= 0");
    if (o.equation != "ms" && o.equation != "darcy")
        throw std::invalid_argument("equation must be ms or darcy");
    if (o.example < 1 || o.example > (o.equation == "ms" ? 4 : 2))
        throw std::invalid_argument("invalid example");
    if (o.k < 0 || o.k > 4)
        throw std::invalid_argument("polynomial degree k must be 0..4");
    if (o.nq == 0)
        o.nq = o.k == 4 ? 36 : o.k == 3 ? 12 : 9;
    // 保留原 CPU 规则在低阶下的可用选项；9/12/36 是推荐的默认规则。
    if (o.nq != 1 && o.nq != 3 && o.nq != 4 && o.nq != 7 && o.nq != 9 && o.nq != 12 && o.nq != 36)
        throw std::invalid_argument("triangle quadrature must be 1, 3, 4, 7, 9, 12 or 36");
    if ((o.k >= 1 && o.nq == 1) || (o.k >= 2 && (o.nq == 3 || o.nq == 4)))
        throw std::invalid_argument("triangle quadrature under-integrates this degree");
    if ((o.k == 3 && o.nq < 12) || (o.k == 4 && o.nq != 36))
        throw std::invalid_argument("insufficient triangle quadrature for this degree");
    if (o.ng < o.k + 2 || o.ng > 10)
        throw std::invalid_argument("edge quadrature must be between k+2 and 10");
    if (o.tile < 1 || (o.threads != 32 && o.threads != 64 && o.threads != 128 && o.threads != 256))
        throw std::invalid_argument("tile must be positive; threads must be 32, 64, 128 or 256");
    if (o.steps < 1 || o.picard < 1 || o.maxit < 1 || o.restart < 1 || !std::isfinite(o.dt) || o.dt <= 0 ||
        !std::isfinite(o.dt * o.steps) || !(o.picard_tol > 0 && o.picard_tol < 1) ||
        !(o.linear_tol > 0 && o.linear_tol < 1) || o.benchmark < 0)
        throw std::invalid_argument("invalid iteration/time parameters");
    if (o.error_interval < 0 || o.print_interval < 0)
        throw std::invalid_argument("diagnostic/print intervals must be nonnegative");
    if (o.mesh.empty() || o.output.empty())
        throw std::invalid_argument("mesh and output paths must not be empty");
    // 文件名只表示 output 目录下的文件；检查重名，避免报告覆盖解或日志。
    std::set<std::string> names;
    for (const auto& name :
         {o.files.log, o.files.history, o.files.solution, o.files.report, o.files.parameters}) {
        const std::filesystem::path path(name);
        if (name.empty() || path.has_parent_path() || name == "." || name == ".." ||
            !names.insert(name).second)
            throw std::invalid_argument("output file names must be distinct plain file names");
    }
    if (o.benchmark_only && o.benchmark == 0)
        o.benchmark = 10;
    return o;
}
} // namespace vgpu
