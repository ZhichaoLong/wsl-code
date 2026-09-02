/**
 * CurvedVEM MS - BDF2 空间收敛阶主程序入口（浓度 + 通量）
 * 功能：Maxwell-Stefan 制造解算例（problem_index = 2），
 *       用 BDF2 时间格式（第一步 BDF1 启动），时间步长随网格加密而减小，
 *       在多套网格上计算浓度与通量 L2 误差及空间收敛阶。
 *
 * 时间步长取法：dt = 0.1 * h^((k+1)/2)
 *       BDF2 的时间误差是 O(dt^2)，k 阶元的空间误差是 O(h^(k+1))，令两者同阶
 *       dt^2 ~ h^(k+1) 即得 dt ~ h^((k+1)/2)。指数随 k 变化：
 *         k=1 -> dt = 0.1*h^1.0     k=2 -> dt = 0.1*h^1.5
 *       这样时间误差既不会在细网格上变成误差地板掩盖空间收敛阶，
 *       也不会像取更高指数那样白烧几十倍的时间步。
 *       （BDF1 的对应取法是 dt = 0.1 * h^(k+1)，本程序不使用。）
 *
 * 注意：比较不同网格的误差必须在同一个终止时间 T，所以 dt 不能直接取公式值，
 *       而是取「不小于 ceil(T / dt_target) 的最小 2^a*5^b 形式步数」，dt = T / n_steps。
 *       这样 dt <= 公式值（时间误差只会比设计值更小，不影响空间阶的测量），
 *       且 dt 一定是有限小数、尾数只由 2 和 5 组成（5e-3 / 2.5e-3 / 1.25e-3 / 5e-4）。
 *       步数下界取 2，是为了保证 BDF2 本身被真正用到（第 1 步是 BDF1 启动）。
 *
 * 配置：k=1, T=1e-1, 网格 2x2~128x128（meshes = {1,2,4,8,16,32,64}，square_nxn 给 2n×2n 单元）
 *       GPU 求解，不保存任何数据文件，仅输出误差表与收敛阶
 *       每套网格算完即打印一次累计误差表，避免后续网格失败时丢掉已有结果
 */

#include "../solver/MSSolver.h"

#include <petsc.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Result {
    std::string label;
    int num_cells;
    double h;
    double dt;
    int n_steps;
    std::vector<double> conc_err;
    std::vector<double> conc_order;
    std::vector<double> flux_err;
    std::vector<double> flux_order;
};

// 按 dt = 0.1 * h^((k+1)/2) 定步长，再调整为能整除 T 的值
struct TimeGrid {
    double dt;
    int n_steps;
};

// dt = 0.1 * h^p 的指数 p = (k+1)/2。
// 由来：BDF2 的时间误差是 O(dt^2)，k 阶元的空间误差是 O(h^(k+1))，
// 令两者同阶 dt^2 ~ h^(k+1) 即得 dt ~ h^((k+1)/2)。
// 这样时间误差既不会在细网格上变成误差地板掩盖空间收敛阶，
// 也不会像取更高指数那样白烧几十倍的时间步。
//   k=1 -> p=1.0    k=2 -> p=1.5    k=3 -> p=2.0
double dtExponent(int k) {
    return 0.5 * static_cast<double>(k + 1);
}

// 允许的步数：形如 2^a * 5^b 的整数（2,4,5,8,10,16,20,25,32,40,50,...）。
// 这样 dt = T / n_steps 一定是有限小数、尾数只由 2 和 5 组成
//（如 5e-3、2.5e-3、1.25e-3、5e-4），不会出现 5.263e-4 这种难看的值。
const std::vector<int>& niceStepCounts() {
    static const std::vector<int> counts = [] {
        std::vector<int> v;
        for (long long p2 = 1; p2 <= 100000; p2 *= 2)
            for (long long p5 = 1; p2 * p5 <= 100000; p5 *= 5)
                v.push_back(static_cast<int>(p2 * p5));
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        return v;
    }();
    return counts;
}

TimeGrid makeTimeGrid(double h, double final_time, int k) {
    double dt_target = 0.1 * std::pow(h, dtExponent(k));
    // 需要的最少步数（向上取整，保证 dt <= 公式值）
    int n_min = static_cast<int>(std::ceil(final_time / dt_target));
    if (n_min < 2)
        n_min = 2;  // 保证 BDF2 至少被用到一步

    // 取不小于 n_min 的最小「漂亮」步数
    const std::vector<int>& counts = niceStepCounts();
    int n_steps = counts.back();
    for (int c : counts) {
        if (c >= n_min) {
            n_steps = c;
            break;
        }
    }

    TimeGrid tg;
    tg.n_steps = n_steps;
    tg.dt = final_time / static_cast<double>(n_steps);
    return tg;
}

Result runCase(int nx, double final_time, int k,
               int gauss, int picard_max, double picard_tol,
               int n_comp, bool use_gpu) {
    std::string mesh_file = "mesh/data/square_" + std::to_string(nx)
                          + "x" + std::to_string(nx) + ".msh";
    int n = 2 * nx;  // square_nxn.msh 实际为 2n × 2n 个曲边单元
    std::string label = std::to_string(n) + "x" + std::to_string(n);
    double h = 1.0 / static_cast<double>(n);

    TimeGrid tg = makeTimeGrid(h, final_time, k);

    vem::StraightMeshReader reader;
    if (!reader.read_mesh(mesh_file))
        throw std::runtime_error("无法读取网格: " + mesh_file);

    MaxwellStefan::MSProblem problem;
    problem.set_problem_index(2);
    if (!problem.init_problem())
        throw std::runtime_error("无法初始化 MS 算例");

    // 终止时间直接传 T，不要再加半步容差。
    // MSSolver 的推进循环判据本身已自带浮点容差：
    //     while (current_time_ < final_time_ - 1e-12) { current_time_ += delta_t_; ... }
    // 再加 0.5*dt 会让 t = T 时判据仍成立，多走一步、停在 T + dt。
    // 而 dt 随网格变化，各套网格就会停在不同时刻（0.15 / 0.125 / ... / 0.10078），
    // 误差不再可比，computeOrders 算出的空间阶失效。
    double t_end = final_time;

    // 不保存任何数据；最后一个参数 true = 启用 BDF2（第一步自动用 BDF1 启动）
    MSSolver solver(reader, problem, tg.dt, t_end,
                    picard_max, picard_tol,
                    gauss, gauss, k,
                    false,    // 不保存
                    "",
                    true);    // use_bdf2
    if (!solver.initialize())
        throw std::runtime_error("求解器初始化失败: " + label);

    PetscErrorCode ierr = solver.solveTimeStepping("ms_bdf2", use_gpu);
    if (ierr != 0)
        throw std::runtime_error("时间推进失败: " + label);

    Result r;
    r.label = label;
    r.num_cells = n * n;
    r.h = h;
    r.dt = tg.dt;
    r.n_steps = tg.n_steps;
    r.conc_err = solver.computeConcentrationL2Error(gauss);
    r.flux_err = solver.computeFluxL2Error(gauss);
    r.conc_order.assign(n_comp, 0.0);
    r.flux_order.assign(n_comp, 0.0);
    return r;
}

void computeOrders(std::vector<Result>& results) {
    for (size_t i = 1; i < results.size(); ++i) {
        double log_h = std::log(results[i-1].h / results[i].h);
        for (size_t c = 0; c < results[i].conc_err.size(); ++c)
            results[i].conc_order[c] =
                std::log(results[i-1].conc_err[c] / results[i].conc_err[c]) / log_h;
        for (size_t c = 0; c < results[i].flux_err.size(); ++c)
            results[i].flux_order[c] =
                std::log(results[i-1].flux_err[c] / results[i].flux_err[c]) / log_h;
    }
}

void printTimeGridTable(const std::vector<int>& meshes, double final_time,
                        int k) {
    std::cout << "  时间步长取法：dt = 0.1 * h^" << std::fixed
              << std::setprecision(2) << dtExponent(k) << "  （=(k+1)/2），"
              << "调整为整除 T = " << std::scientific << std::setprecision(1)
              << final_time << "\n\n";
    std::cout << std::setw(10) << "网格"
              << std::setw(12) << "h"
              << std::setw(16) << "dt 公式值"
              << std::setw(8) << "步数"
              << std::setw(16) << "dt 实际值" << "\n";
    std::cout << std::string(62, '-') << "\n";
    for (int nx : meshes) {
        int n = 2 * nx;
        double h = 1.0 / static_cast<double>(n);
        TimeGrid tg = makeTimeGrid(h, final_time, k);
        std::cout << std::setw(10) << (std::to_string(n) + "x" + std::to_string(n))
                  << std::setw(12) << std::fixed << std::setprecision(5) << h
                  << std::setw(16) << std::scientific << std::setprecision(4)
                  << (0.1 * std::pow(h, dtExponent(k)))
                  << std::setw(8) << tg.n_steps
                  << std::setw(16) << std::scientific << std::setprecision(4)
                  << tg.dt << "\n";
    }
    std::cout << "\n";
}

void printTable(const std::string& title,
                const std::vector<Result>& results,
                int n_comp, bool is_flux) {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "  " << title << "\n";
    std::cout << "============================================================\n\n";

    std::string sym = is_flux ? "J" : "c";
    std::cout << std::setw(10) << "网格"
              << std::setw(10) << "单元数"
              << std::setw(10) << "h"
              << std::setw(12) << "dt"
              << std::setw(8) << "步数";
    for (int c = 0; c < n_comp; ++c)
        std::cout << std::setw(16) << (sym + std::to_string(c) + " 误差");
    for (int c = 0; c < n_comp; ++c)
        std::cout << std::setw(10) << (sym + std::to_string(c) + " 阶");
    std::cout << "\n";
    std::cout << std::string(50 + n_comp * 26, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::setw(10) << r.label
                  << std::setw(10) << r.num_cells
                  << std::setw(10) << std::fixed
                  << std::setprecision(4) << r.h
                  << std::setw(12) << std::scientific
                  << std::setprecision(2) << r.dt
                  << std::setw(8) << r.n_steps;
        const auto& err = is_flux ? r.flux_err : r.conc_err;
        const auto& ord = is_flux ? r.flux_order : r.conc_order;
        for (int c = 0; c < n_comp; ++c)
            std::cout << std::setw(16) << std::scientific
                      << std::setprecision(4) << err[c];
        for (int c = 0; c < n_comp; ++c) {
            if (ord[c] == 0.0)
                std::cout << std::setw(10) << "  -  ";
            else
                std::cout << std::setw(10) << std::fixed
                          << std::setprecision(2) << ord[c];
        }
        std::cout << "\n";
    }
    std::cout << "\n============================================================\n";
}

}  // namespace

int main(int argc, char** argv) {
#if defined(PETSC_HAVE_CUDA)
    setenv("CUDA_VISIBLE_DEVICES", "0", 1);
    PetscOptionsSetValue(NULL, "-use_gpu_aware_mpi", "0");
    PetscOptionsSetValue(NULL, "-device_select", "0");
#endif

    PetscErrorCode error = PetscInitialize(&argc, &argv, NULL, NULL);
    if (error != 0) return static_cast<int>(error);

    int status = 0;
    try {
        const int k = 1;
        const int gauss = 9;
        const int n_comp = 3;
        const int picard_max = 50;
        const double picard_tol = 1e-6;
        const bool use_gpu = true;

        // 所有网格共用的终止时间（误差必须在同一时刻比较）
        const double final_time = 1e-1;

        // 测试网格序列
        std::vector<int> meshes = {1, 2, 4, 8, 16, 32, 64};

        std::cout << "\n";
        std::cout << "============================================================\n";
        std::cout << "  MS BDF2 空间收敛阶测试（制造解，曲边 H(div) 混合虚元）\n";
        std::cout << "  时间格式：BDF2（第 1 步 BDF1 启动）\n";
        std::cout << "  k = " << k
                  << "   T = " << std::scientific << std::setprecision(1)
                  << final_time << "\n";
        std::cout << "  GPU = " << (use_gpu ? "on" : "off")
                  << "   误差类型：绝对 L2\n";
        std::cout << "============================================================\n\n";

        printTimeGridTable(meshes, final_time, k);

        std::string suffix = "（k=" + std::to_string(k)
                           + "，曲边，dt=0.1h^"
                           + std::to_string(dtExponent(k)).substr(0, 4) + "）";

        std::vector<Result> results;
        for (size_t i = 0; i < meshes.size(); ++i) {
            int nx = meshes[i];
            std::cerr << "[" << (i + 1) << "/" << meshes.size()
                      << "]  网格 " << 2 * nx << "x" << 2 * nx << " ... ";
            std::cerr.flush();

            std::clock_t t0 = std::clock();
            Result r = runCase(nx, final_time, k, gauss,
                               picard_max, picard_tol, n_comp, use_gpu);
            double secs = double(std::clock() - t0) / CLOCKS_PER_SEC;
            results.push_back(r);

            std::cerr << "done (" << r.num_cells << " cells, "
                      << r.n_steps << " steps, dt="
                      << std::scientific << std::setprecision(2) << r.dt << ", "
                      << std::fixed << std::setprecision(1) << secs << "s, "
                      << "c0=" << std::scientific << std::setprecision(2)
                      << r.conc_err[0]
                      << ", J0=" << std::scientific << std::setprecision(2)
                      << r.flux_err[0] << ")\n";
            std::cerr.flush();

            // 每套网格算完立刻输出「截至当前」的完整误差表并刷新 stdout。
            // 这样即使后面某套网格 OOM / 崩溃，已完成的结果也不会丢。
            // 最后一次迭代打印的就是最终完整结果，循环外不再重复打印。
            computeOrders(results);
            std::cout << "\n>>>>>> 已完成 " << (i + 1) << "/" << meshes.size()
                      << " 套网格，以下为累计结果 <<<<<<\n";
            printTable("BDF2 浓度 L2 误差与空间收敛阶" + suffix,
                       results, n_comp, false);
            printTable("BDF2 通量 L2 误差与空间收敛阶" + suffix,
                       results, n_comp, true);
            std::cout << std::endl;  // endl 刷新缓冲，保证已重定向到文件的内容落盘
        }

    } catch (const std::exception& e) {
        std::cerr << "\n  错误：" << e.what() << "\n\n";
        status = 1;
    }

    PetscFinalize();
    return status;
}
