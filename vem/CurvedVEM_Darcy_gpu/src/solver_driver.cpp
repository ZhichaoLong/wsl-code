/**
 * 单算例的公共求解流程：CPU 读取网格，GPU 构造算子、迭代与误差积分。
 * 所有配置来自 main 传入的 SolverOptions；这里不解析命令行。
 * Darcy 求解一次；MS 使用 Euler/BDF2 时间推进，每步进行 Picard 迭代。
 */
#include "equation_gpu.cuh"
#include "solver_driver.h"
#include "runtime_metrics.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <memory>
#include <cmath>
namespace vgpu {
namespace {
// 保存已通过校验且自动积分点数已解析的配置，便于从输出目录复现运行。
void write_parameters(std::ostream& out, const SolverOptions& o) {
    out << std::setprecision(17) << std::boolalpha;
    out << "petsc_ksp=fgmres\n";
    out << "petsc_operator=" << (o.assembled ? "aijcusparse" : "shell")
        << "\npetsc_pc=" << o.petsc_pc << "\npetsc_ordering=" << o.petsc_ordering << "\nilu_levels=" << o.ilu_levels << "\n";
    out << "equation=" << o.equation << "\nexample=" << o.example << "\nmesh=" << std::quoted(o.mesh)
        << "\noutput=" << std::quoted(o.output) << "\nk=" << o.k << "\nnq=" << o.nq << "\nng=" << o.ng
        << "\ndt=" << o.dt << "\nsteps=" << o.steps << "\nbdf2=" << o.bdf2
        << "\nsolve_steps=" << (o.equation == "darcy" ? 1 : o.steps)
        << "\ntarget_time=" << (o.equation == "darcy" ? 0 : o.steps * o.dt) << "\npicard=" << o.picard
        << "\npicard_tol=" << o.picard_tol << "\nmaxit=" << o.maxit << "\nrestart=" << o.restart
        << "\nlinear_tol=" << o.linear_tol << "\nthreads=" << o.threads << "\ntile=" << o.tile
        << "\nbenchmark=" << o.benchmark << "\nbenchmark_only=" << o.benchmark_only
        << "\nlog=" << std::quoted(o.files.log) << "\nhistory=" << std::quoted(o.files.history)
        << "\nsolution=" << std::quoted(o.files.solution) << "\nreport=" << std::quoted(o.files.report)
        << "\nparameters=" << std::quoted(o.files.parameters) << "\nsave_solution=" << o.save_solution
        << "\nerror_interval=" << o.error_interval << "\nprint_interval=" << o.print_interval << '\n';
}
using Clock = std::chrono::steady_clock;
// 主机端墙钟耗时，包含该阶段的调度与等待。
double seconds(Clock::time_point a) {
    return std::chrono::duration<double>(Clock::now() - a).count();
}
// 无解析解的误差为 NaN，JSON 使用 null 表示，避免输出伪误差或无效 JSON。
void json_array(std::ostream& f, const std::vector<double>& a) {
    f << '[';
    for (size_t i = 0; i < a.size(); ++i) {
        if (i)
            f << ',';
        if (std::isfinite(a[i]))
            f << a[i];
        else
            f << "null";
    }
    f << ']';
}
// CUDA event 的资源所有权；离开计时作用域时自动释放。
struct Event {
    cudaEvent_t e;
    Event() {
        CUDA(cudaEventCreate(&e));
    }
    ~Event() {
        cudaEventDestroy(e);
    }
};
// 先预热再记录同一默认流上的重复执行平均耗时（毫秒）。
template <class F> double timed(int repeats, F f) {
    f();
    CUDA(cudaDeviceSynchronize());
    Event a, b;
    CUDA(cudaEventRecord(a.e));
    for (int i = 0; i < repeats; ++i)
        f();
    CUDA(cudaEventRecord(b.e));
    CUDA(cudaEventSynchronize(b.e));
    float ms;
    CUDA(cudaEventElapsedTime(&ms, a.e, b.e));
    return ms / repeats;
}

} // namespace

// 执行一个算例并管理其资源生命周期；异常转换为非零退出状态。
int run_case(const SolverOptions& options, CaseResult* result, const vem::StraightMeshReader* prepared_mesh) {
    if (result)
        *result = CaseResult{};
    SolverOptions o = options;
    std::string result_directory;
    try {
        o = validated_options(o);
        std::filesystem::create_directories(o.output);
        result_directory = o.output;
        std::ofstream log(o.output + "/" + o.files.log), metrics(o.output + "/" + o.files.history);
        if (!log || !metrics)
            throw std::runtime_error("cannot open output files");
        {
            std::ofstream status(o.output + "/" + o.files.report);
            status << "{\"status\":\"running\"}\n";
        }
        // 在 GPU 初始化前记录配置；即使驱动或后续求解失败也能定位当次参数。
        std::ofstream parameters(o.output + "/" + o.files.parameters);
        write_parameters(parameters, o);
        if (!parameters)
            throw std::runtime_error("cannot write parameter snapshot");
        parameters.close();
        write_parameters(log, o);
        std::cout << "算例=" << o.equation << " " << o.example << " 网格=" << o.mesh << " k=" << o.k
                  << " 积分点=" << o.nq << "/" << o.ng << " 输出=" << o.output << " 报告=" << o.files.report
                  << std::endl;
        log << std::setprecision(16);
        metrics << std::setprecision(16);
        metrics << "step,time,picard,linear_iterations,residual,component,scalar_comp,flux_comp,scalar_phys,"
                   "flux_phys,div_phys,mass\n";
        // GPU 初始化；几何、网格和算例对象仍由 CPU 构造。
        cudaDeviceProp prop;
        CUDA(cudaGetDeviceProperties(&prop, 0));
        CUDA(cudaSetDevice(0));
        CUDA(cudaFree(nullptr));
        auto start = Clock::now();
        // 收敛实验在 CPU 读取网格并测量 h 后复用同一个 reader，避免重复读大网格。
        vem::StraightMeshReader owned_mesh;
        if (!prepared_mesh) {
            bool ok = o.mesh.size() >= 4 && o.mesh.substr(o.mesh.size() - 4) == ".vtk"
                          ? owned_mesh.read_mesh_vtk(o.mesh)
                          : owned_mesh.read_mesh(o.mesh);
            if (!ok)
                throw std::runtime_error("mesh load failed: " + o.mesh);
        }
        const auto& mesh = prepared_mesh ? *prepared_mesh : owned_mesh;
        std::unique_ptr<MixedGpu> A;
        Darcy::DarcyProblem dp;
        MaxwellStefan::MSProblem mp;
        if (o.equation == "darcy") {
            dp.set_problem_index(o.example);
            if (!dp.init_problem())
                throw std::runtime_error("problem initialization failed");
            A = std::make_unique<DarcyGpu>(mesh, dp, o.example, o.k, o.nq, o.ng, o.tile, o.threads);
        } else {
            mp.set_mesh(mesh);
            mp.set_problem_index(o.example);
            if (!mp.init_problem())
                throw std::runtime_error("problem initialization failed");
            A = std::make_unique<MSGpu>(mesh, mp, o.example, o.k, o.nq, o.ng, o.tile, o.threads);
        }
        double setup_seconds = seconds(start);
        A->resources(log);
        int n = A->size();
        log << "GPU=" << prop.name << " SM=" << prop.multiProcessorCount << " memory=" << prop.totalGlobalMem
            << " registers_per_SM=" << prop.regsPerMultiprocessor
            << " shared_per_SM=" << prop.sharedMemPerMultiprocessor << "\n";
        log << "cells=" << A->host().s.nc << " dofs=" << n << " k=" << o.k << " threads=" << o.threads
            << " tile=" << o.tile << " basis_bytes=" << A->basis_bytes()
            << " operator_bytes=" << A->operator_bytes() << " setup_seconds=" << setup_seconds << "\n";
        std::cout << "GPU=" << prop.name << " cells=" << A->host().s.nc << " dofs=" << n
                  << " setup_seconds=" << setup_seconds << std::endl;
        // x 为当前解，guess 为 Picard 状态；prev/older 保存两个时间层的 GPU 向量。
        PetscVector x(n), guess(n), prev(n), older(n), work(n);
        x.zero();
        guess.zero();
        prev.zero();
        older.zero();
        Krylov solver(n, o.assembled, o.petsc_pc, o.ilu_levels, o.petsc_ordering);
        auto mass0 = A->initial_mass();
        double basis_ms = 0, update_ms = 0, picard_ms = 0, apply_ms = 0;
        // CUDA event 测量预热后的基础构造、系数更新和 matrix-free 算子耗时。
        if (o.benchmark) {
            basis_ms = timed(o.benchmark, [&]() { A->build_basis(); });
            update_ms =
                timed(o.benchmark, [&]() { A->update(nullptr, true, o.equation == "ms" ? 1 / o.dt : 0); });
            picard_ms = timed(o.benchmark,
                              [&]() { A->update(guess.data(), false, o.equation == "ms" ? 1 / o.dt : 0); });
            A->update(nullptr, true, o.equation == "ms" ? 1 / o.dt : 0);
            A->boundary_lift(.001);
            CUDA(cudaMemset(x.data(), 0, n * sizeof(double)));
            solver.copy(A->bc(), x);
            apply_ms = timed(o.benchmark, [&]() { A->apply(x.data(), work.data()); });
            log << "basis_ms=" << basis_ms << " update_including_diagonal_ms=" << update_ms
                << " picard_update_ms=" << picard_ms << " apply_ms=" << apply_ms << "\n";
            std::cout << "basis_ms=" << basis_ms << " update_ms=" << update_ms
                      << " picard_update_ms=" << picard_ms << " apply_ms=" << apply_ms << std::endl;
        }
        // 求解仅需 D/P/W/H⁻¹ 与方程因子，释放 G/B/H/H* 诊断缓存。
        A->compact_basis();
        log << "resident_basis_bytes=" << A->basis_bytes() << "\n";
        ErrorReport errors;
        double iteration_seconds = 0, error_seconds = 0, solution_seconds = 0;
        double ksp_seconds = 0, linear_solve_seconds = 0;
        const auto prepared_memory = sample_memory();
        log << "memory phase=prepared rss_bytes=" << prepared_memory.rss_bytes
            << " gpu_used_bytes=" << prepared_memory.gpu_used_bytes
            << " gpu_free_bytes=" << prepared_memory.gpu_free_bytes << '\n' << std::flush;
        double residual = 0;
        int total_linear = 0;
        double final_time = 0;
        auto solve_start = Clock::now();
        if (!o.benchmark_only) {
            int steps = o.equation == "darcy" ? 1 : o.steps;
            // BDF2 第二步起质量系数为 3/(2dt)；首步采用 Euler 的 1/dt。
            for (int step = 1; step <= steps; ++step) {
                const auto step_start = Clock::now();
                double time = o.equation == "darcy" ? 0 : step * o.dt;
                double rate = o.equation == "darcy" ? 0 : (o.bdf2 && step > 1 ? 1.5 : 1) / o.dt;
                bool accepted = false;
                int linear = 0, used = 0;
                double step_ksp_seconds = 0, step_linear_seconds = 0;
                solver.copy(prev, guess);
                solver.copy(prev, x);
                // 固定浓度更新 MS 通量块，再解一次混合线性系统。
                for (int it = 0; it < (o.equation == "darcy" ? 1 : o.picard); ++it) {
                    A->update(guess.data(), step == 1 && it == 0, rate);
                    A->make_rhs(time, o.dt, step, o.bdf2, prev.data(), older.data());
                    const auto linear_start = Clock::now();
                    auto si = solver.solve(*A, A->rhs(), x, o.linear_tol, o.maxit, o.restart);
                    step_linear_seconds += seconds(linear_start);
                    step_ksp_seconds += si.ksp_seconds;
                    if (step == 1 && it == 0) { solver.describe(log); solver.describe(std::cout); }
                    linear += si.iterations;
                    residual = si.residual;
                    double diff =
                        solver.difference(x, guess) / std::max(1., solver.norm(x));
                    solver.copy(x, guess);
                    used = it + 1;
                    log << "step=" << step << " picard=" << used << " linear=" << si.iterations
                        << " ksp_seconds=" << si.ksp_seconds
                        << " residual=" << residual << " refinements=" << si.refinements << " relative_change=" << diff << "\n";
                    if (o.equation == "darcy" || diff <= o.picard_tol) {
                        // 以最终浓度重新计算非线性真实残差，避免只凭迭代差接受解。
                        if (o.equation == "ms") {
                            A->update(x.data(), false, rate);
                            A->make_rhs(time, o.dt, step, o.bdf2, prev.data(), older.data());
                            A->apply(x.data(), work.data());
                            residual = solver.difference(work, A->rhs()) /
                                       std::max(solver.norm(A->rhs()), 1e-30);
                        }
                        if (residual <= std::max(o.picard_tol, 5 * o.linear_tol)) {
                            accepted = true;
                            break;
                        }
                    }
                }
                if (!accepted)
                    throw std::runtime_error("Picard failed at step " + std::to_string(step));
                total_linear += linear;
                final_time = time;
                CUDA(cudaDeviceSynchronize());
                const double step_seconds = seconds(step_start);
                iteration_seconds += step_seconds;
                ksp_seconds += step_ksp_seconds;
                linear_solve_seconds += step_linear_seconds;
                // 长时间运行可只在选定时间步计算误差，最终步始终进行完整后处理。
                const bool diagnostics =
                    step == steps || (o.error_interval > 0 && step % o.error_interval == 0);
                if (diagnostics) {
                    const auto error_start = Clock::now();
                    errors = A->errors(x.data(), time);
                    error_seconds += seconds(error_start);
                    for (int s = 0; s < A->components(); ++s)
                        metrics << step << ',' << time << ',' << used << ',' << linear << ',' << residual
                                << ',' << s << ',' << errors.scalar_comp[s] << ',' << errors.flux_comp[s]
                                << ',' << errors.scalar_phys[s] << ',' << errors.flux_phys[s] << ','
                                << errors.div_phys[s] << ',' << errors.mass[s] << '\n';
                }
                if (step == 1 || step == steps || (o.print_interval > 0 && step % o.print_interval == 0)) {
                    std::cout << "step=" << step << '/' << steps << " t=" << time << " picard=" << used
                              << " linear=" << linear << " residual=" << residual
                              << " step_seconds=" << step_seconds << " ksp_seconds=" << step_ksp_seconds
                              << " elapsed_seconds=" << seconds(solve_start) << std::endl;
                    // 有诊断的中间时间步显示当前各组分结果，避免长时间只看到迭代计数。
                    // 最终步稍后由完整结果表统一输出。
                    if (diagnostics && step != steps) {
                        for (size_t s = 0; s < errors.mass.size(); ++s) {
                            std::cout << "  组分 " << s;
                            if (std::isfinite(errors.scalar_comp[s]))
                                std::cout << " 计算域 scalar_L2=" << errors.scalar_comp[s]
                                          << " flux_L2=" << errors.flux_comp[s];
                            std::cout << " mass=" << errors.mass[s];
                            if (s < mass0.size())
                                std::cout << " mass_drift=" << errors.mass[s] - mass0[s];
                            std::cout << '\n';
                        }
                        std::cout << std::flush;
                    }
                }
                // 固定在每步诊断之后采样，避免只凭进程峰值把 Krylov 预热误判为泄漏。
                const auto memory = sample_memory();
                log << "step=" << step << " step_seconds=" << step_seconds
                    << " ksp_seconds=" << step_ksp_seconds << " linear_solve_seconds=" << step_linear_seconds
                    << " elapsed_seconds=" << seconds(solve_start) << " rss_bytes=" << memory.rss_bytes
                    << " gpu_used_bytes=" << memory.gpu_used_bytes << " gpu_free_bytes=" << memory.gpu_free_bytes
                    << '\n';
                log.flush();
                metrics.flush();
                solver.copy(prev, older);
                solver.copy(x, prev);
            }
            // 收敛实验可关闭最终解写盘，避免下载大向量；误差、质量和运行报告仍保存。
            if (o.save_solution) {
                const auto save_start = Clock::now();
                auto sol = x.download();
                std::ofstream out(o.output + "/" + o.files.solution);
                if (!out)
                    throw std::runtime_error("cannot write solution");
                out << std::setprecision(17) << "dof,value\n";
                for (size_t i = 0; i < sol.size(); ++i)
                    out << i << ',' << sol[i] << '\n';
                out.close();
                if (!out)
                    throw std::runtime_error("solution write failed");
                solution_seconds = seconds(save_start);
            }
        }
        // 基准测试与完整求解使用不同状态，防止把算子计时误当作 PDE 收敛。
        double solve_seconds = seconds(solve_start);
        std::ofstream report(o.output + "/" + o.files.report);
        report << std::setprecision(16) << "{\n\"status\":\""
               << (o.benchmark_only ? "benchmark" : "converged") << "\",\n\"equation\":\"" << o.equation
               << "\",\n\"example\":" << o.example << ",\n\"k\":" << o.k << ",\n\"cells\":" << A->host().s.nc
               << ",\n\"dofs\":" << n << ",\n\"time\":" << final_time
               << ",\n\"setup_seconds\":" << setup_seconds << ",\n\"solve_seconds\":" << solve_seconds
               << ",\n\"ksp_seconds\":" << ksp_seconds
               << ",\n\"linear_solve_seconds\":" << linear_solve_seconds
               << ",\n\"average_step_seconds\":" << iteration_seconds / (o.equation == "darcy" ? 1 : o.steps)
               << ",\n\"iteration_seconds\":" << iteration_seconds << ",\n\"error_seconds\":" << error_seconds
               << ",\n\"solution_seconds\":" << solution_seconds
               << ",\n\"linear_iterations\":" << total_linear << ",\n\"residual\":" << residual
               << ",\n\"basis_bytes\":" << A->basis_bytes() << ",\n\"operator_bytes\":" << A->operator_bytes()
               << ",\n\"basis_ms\":" << basis_ms << ",\n\"update_ms\":" << update_ms
               << ",\n\"apply_ms\":" << apply_ms << ",\n\"picard_update_ms\":" << picard_ms
               << ",\n\"initial_mass\":";
        json_array(report, mass0);
        report << ",\n\"petsc_operator\":\"" << (o.assembled ? "seqaijcusparse" : "shell")
               << "\",\n\"petsc_vec\":\"seqcuda\",\n\"petsc_ksp\":\"fgmres\""
               << ",\n\"petsc_pc\":\"" << (o.assembled ? o.petsc_pc : "shell")
               << "\",\n\"petsc_ordering\":\"" << (o.assembled ? o.petsc_ordering : "none")
               << "\",\n\"ilu_levels\":" << o.ilu_levels
               << ",\n\"coo_entries\":" << A->coo_entries()
               << ",\n\"coo_value_bytes\":" << A->coo_entries() * sizeof(PetscScalar)
               << ",\n\"csr_nnz_original\":" << A->sparse_nonzeros()
               << ",\n\"csr_nnz_ordered\":" << A->sparse_nonzeros(true);
        report << ",\n\"mass\":";
        json_array(report, errors.mass);
        report << ",\n\"scalar_comp\":";
        json_array(report, errors.scalar_comp);
        report << ",\n\"flux_comp\":";
        json_array(report, errors.flux_comp);
        report << ",\n\"scalar_phys\":";
        json_array(report, errors.scalar_phys);
        report << ",\n\"flux_phys\":";
        json_array(report, errors.flux_phys);
        report << ",\n\"div_phys\":";
        json_array(report, errors.div_phys);
        report << "\n}\n";
        if (!report)
            throw std::runtime_error("report write failed");
        if (!o.benchmark_only) {
            CaseResult completed;
            completed.cells = A->host().s.nc;
            completed.dofs = n;
            completed.steps = o.equation == "darcy" ? 1 : o.steps;
            completed.linear_iterations = total_linear;
            completed.time = final_time;
            completed.residual = residual;
            completed.setup_seconds = setup_seconds;
            completed.iteration_seconds = iteration_seconds;
            completed.ksp_seconds = ksp_seconds;
            completed.linear_solve_seconds = linear_solve_seconds;
            completed.error_seconds = error_seconds;
            completed.solution_seconds = solution_seconds;
            completed.errors = errors;
            completed.initial_mass = mass0;
            print_case_result(std::cout, completed);
            print_case_result(log, completed);
            if (result)
                *result = completed;
        }
        return 0;
    } catch (const std::exception& e) {
        if (result)
            result->failure = e.what();
        if (!result_directory.empty()) {
            std::ofstream failure_log(result_directory + "/" + o.files.log, std::ios::app);
            failure_log << "ERROR: " << e.what() << "\n";
            std::ofstream failure_report(result_directory + "/" + o.files.report);
            failure_report << "{\"status\":\"failed\"}\n";
        }
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}

} // namespace vgpu
