/**
 * CPU 主机端实验组织与结果汇总。GPU 算子/求解由 run_case 负责。
 * 使用实际网格的 h 和统一 T；读取后的网格复用给求解器，避免双重 I/O。
 */
#include "mesh_study.h"
#include "runtime_metrics.h"
#include "../mesh/straight_mesh.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>

namespace vgpu {
TimeGrid make_time_grid(double h, int k, bool bdf2, const TimeOptions& o) {
    if (!(h > 0) || !std::isfinite(h) || k < 0 || k > 4 || !(o.final_time > 0) ||
        !std::isfinite(o.final_time) || o.max_steps < 1)
        throw std::invalid_argument("invalid h, k, final time or maximum time steps");
    const double exponent = double(k + 1) / (bdf2 ? 2 : 1);
    const double target =
        o.policy == TimePolicy::Balanced ? o.coefficient * std::pow(h, exponent) : o.fixed_dt;
    if (!(target > 0) || !std::isfinite(target))
        throw std::invalid_argument("target time step must be finite and positive");
    // ceil 只会缩小 dt，不能为了凑步数而跨过给定目标；BDF2 至少实际走两步。
    const double required = std::max(bdf2 ? 2. : 1., std::ceil(o.final_time / target));
    if (!std::isfinite(required) || required > o.max_steps)
        throw std::invalid_argument("required time steps exceed max_steps; no computation started");
    int steps = static_cast<int>(required);
    if (o.smooth_steps) {
        long long best = static_cast<long long>(o.max_steps) + 1;
        for (long long a = 1; a <= o.max_steps; a *= 2) {
            for (long long n = a; n <= o.max_steps; n *= 5) {
                if (n >= steps)
                    best = std::min(best, n);
            }
        }
        if (best > o.max_steps)
            throw std::invalid_argument("no 2^a*5^b step count fits max_steps");
        steps = static_cast<int>(best);
    }
    const double dt = o.final_time / steps;
    if (!(dt > 0))
        throw std::invalid_argument("time step underflow");
    return {target, dt, steps};
}

MeshMeasure measure_mesh(const vem::StraightMeshReader& reader) {
    const auto& m = reader.get_mesh_data();
    MeshMeasure r;
    r.cells = m.num_cells;
    if (r.cells < 1)
        throw std::invalid_argument("empty mesh in convergence study");
    for (int i = 0; i < r.cells; ++i) {
        if (!(m.cell_area[i] > 0) || !std::isfinite(m.cell_area[i]) || !(m.cell_diameter[i] > 0) ||
            !std::isfinite(m.cell_diameter[i]))
            throw std::invalid_argument("invalid mesh area or diameter");
        r.area += m.cell_area[i];
        r.h_max = std::max(r.h_max, m.cell_diameter[i]);
    }
    r.h_equivalent = std::sqrt(r.area / r.cells);
    return r;
}

double observed_order(double old_h, double new_h, double old_error, double new_error) {
    if (!(old_h > new_h && new_h > 0 && old_error > 0 && new_error > 0) || !std::isfinite(old_h) ||
        !std::isfinite(new_h) || !std::isfinite(old_error) || !std::isfinite(new_error))
        return std::numeric_limits<double>::quiet_NaN();
    // 对数差避免误差比值上溢，零阶和负阶都是实际结果，不能用 '-' 代替。
    return (std::log(old_error) - std::log(new_error)) / (std::log(old_h) - std::log(new_h));
}

namespace {
using Clock = std::chrono::steady_clock;
std::string csv_string(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"')
            out += '"';
        out += c;
    }
    return out + '"';
}
const std::vector<double>& scalar(const MeshResult& r, ErrorDomain d) {
    return d == ErrorDomain::Computational ? r.solve.errors.scalar_comp : r.solve.errors.scalar_phys;
}
const std::vector<double>& flux(const MeshResult& r, ErrorDomain d) {
    return d == ErrorDomain::Computational ? r.solve.errors.flux_comp : r.solve.errors.flux_phys;
}
void number(std::ostream& out, double value, int width, bool order = false) {
    if (!std::isfinite(value))
        out << std::setw(width) << "-";
    else
        out << std::setw(width) << (order ? std::fixed : std::scientific) << std::setprecision(order ? 3 : 6)
            << value;
}
// 每个组分单独一行，避免三组分的全部指标挤在一条超宽终端行中。
void print_tables(std::ostream& out, const std::vector<MeshResult>& rows, const StudyOptions& study,
                  bool exact) {
    out << "\n>>>> 已完成 " << rows.size() << '/' << study.meshes.size() << " 层网格：" << study.title
        << " <<<<\n";
    if (exact) {
        out << (study.error_domain == ErrorDomain::Computational ? "计算域" : "物理域")
            << "绝对 L2 误差与空间收敛阶（首层 '-' 表示无前一层可比较）\n";
        out << "mesh              comp      scalar_L2    order       flux_L2    order\n";
        for (const auto& r : rows)
            for (size_t s = 0; s < scalar(r, study.error_domain).size(); ++s) {
                out << std::left << std::setw(18) << r.label << std::right << std::setw(4) << s;
                number(out, scalar(r, study.error_domain)[s], 15);
                number(out, r.scalar_order[s], 9, true);
                number(out, flux(r, study.error_domain)[s], 15);
                number(out, r.flux_order[s], 9, true);
                out << '\n';
            }
    } else {
        out << "无解析解：不填写解析误差/收敛阶；检查各组分物理质量守恒\n"
            << "mesh              comp     initial_mass        final_mass        mass_drift\n";
        for (const auto& r : rows)
            for (size_t s = 0; s < r.solve.errors.mass.size(); ++s) {
                out << std::left << std::setw(18) << r.label << std::right << std::setw(4) << s;
                number(out, r.solve.initial_mass[s], 18);
                number(out, r.solve.errors.mass[s], 18);
                number(out, r.solve.errors.mass[s] - r.solve.initial_mass[s], 18);
                out << '\n';
            }
    }
    out << "\n网格尺度、时间网格及墙钟耗时（s）\n"
        << "mesh             cells     dofs          h     h_max         dt  steps   setup   iterate    "
           "error    total  linear   avg_step       KSP\n";
    for (const auto& r : rows) {
        out << std::left << std::setw(16) << r.label << std::right << std::setw(6) << r.solve.cells
            << std::setw(9) << r.solve.dofs << std::scientific << std::setprecision(3) << std::setw(11) << r.h
            << std::setw(10) << r.h_max << std::setw(11) << r.time_grid.dt << std::setw(7) << r.solve.steps
            << std::fixed << std::setprecision(3) << std::setw(8) << r.solve.setup_seconds << std::setw(10)
            << r.solve.iteration_seconds << std::setw(9) << r.solve.error_seconds << std::setw(9)
            << r.wall_seconds << std::setw(8) << r.solve.linear_iterations
            << std::setw(11) << r.solve.iteration_seconds / r.solve.steps
            << std::setw(10) << r.solve.ksp_seconds << '\n';
    }
    out << std::flush;
}
// 每完成一层即写长表；除选定范数外同时保留两种域的原始误差，便于复核。
void append_csv(std::ostream& out, const MeshResult& r, const StudyOptions& study) {
    const auto& e = r.solve.errors;
    for (size_t s = 0; s < e.mass.size(); ++s) {
        const double initial = s < r.solve.initial_mass.size() ? r.solve.initial_mass[s]
                                                               : std::numeric_limits<double>::quiet_NaN();
        out << std::setprecision(17) << csv_string(r.label) << ',' << r.solve.cells << ',' << r.solve.dofs
            << ',' << r.h << ',' << r.h_max << ',' << r.time_grid.target_dt << ',' << r.time_grid.dt << ','
            << r.solve.steps << ',' << r.solve.time << ',' << s << ','
            << (study.error_domain == ErrorDomain::Computational ? "computational" : "physical") << ','
            << scalar(r, study.error_domain)[s] << ',' << r.scalar_order[s] << ','
            << flux(r, study.error_domain)[s] << ',' << r.flux_order[s] << ',' << e.scalar_comp[s] << ','
            << e.flux_comp[s] << ',' << e.scalar_phys[s] << ',' << e.flux_phys[s] << ',' << e.div_phys[s]
            << ',' << initial << ',' << e.mass[s] << ',' << e.mass[s] - initial << ','
            << r.solve.setup_seconds << ',' << r.solve.iteration_seconds << ',' << r.solve.error_seconds
            << ',' << r.solve.solution_seconds << ',' << r.wall_seconds << ',' << r.solve.linear_iterations
            << ',' << r.solve.residual << ',' << r.solve.iteration_seconds / r.solve.steps
            << ',' << r.solve.ksp_seconds << ',' << r.solve.linear_solve_seconds << '\n';
    }
    out.flush();
    if (!out)
        throw std::runtime_error("cannot write convergence table");
}
} // namespace

int run_mesh_study(const SolverOptions& base, const StudyOptions& study) {
    std::ofstream table, transcript;
    try {
        if (study.meshes.empty() || study.table_file.empty() || study.terminal_file.empty())
            throw std::invalid_argument("mesh list and study output files must be specified in main");
        if (base.benchmark || base.benchmark_only)
            throw std::invalid_argument("convergence study requires complete PDE solves");
        const auto absolute = [](const std::string& path) {
            return std::filesystem::weakly_canonical(std::filesystem::absolute(path));
        };
        std::set<std::filesystem::path> paths;
        for (const auto& path : {study.table_file, study.terminal_file})
            if (!paths.insert(absolute(path)).second)
                throw std::invalid_argument("duplicate study output path");
        // 先检查名称，避免运行数小时后下一网格覆盖此前输出；网格本身逐层读取。
        for (const auto& entry : study.meshes) {
            auto o = base;
            o.mesh = entry.file;
            o.output = entry.output;
            o = validated_options(o);
            if (entry.label.empty())
                throw std::invalid_argument("mesh label must not be empty");
            for (const auto& name :
                 {o.files.log, o.files.history, o.files.report, o.files.solution, o.files.parameters})
                if (!paths.insert(absolute((std::filesystem::path(o.output) / name).string())).second)
                    throw std::invalid_argument("duplicate output path across meshes");
        }
        std::filesystem::create_directories(absolute(study.table_file).parent_path());
        std::filesystem::create_directories(absolute(study.terminal_file).parent_path());
        table.open(study.table_file);
        transcript.open(study.terminal_file);
        if (!table || !transcript)
            throw std::runtime_error("cannot open study output files");
        table << "mesh,cells,dofs,h,h_max,dt_target,dt,steps,time,component,error_domain,scalar_error,scalar_"
                 "order,flux_error,flux_order,scalar_comp,flux_comp,scalar_phys,flux_phys,div_phys,initial_"
                 "mass,mass,mass_drift,setup_seconds,iteration_seconds,error_seconds,solution_seconds,total_"
                 "seconds,linear_iterations,residual,average_step_seconds,ksp_seconds,linear_solve_seconds\n";
        const bool exact = base.equation == "darcy" || base.example == 2 || base.example == 4;
        for (std::ostream* out :
             {static_cast<std::ostream*>(&std::cout), static_cast<std::ostream*>(&transcript)}) {
            *out << '\n'
                 << study.title << "\n方程=" << base.equation << " 算例=" << base.example << " k=" << base.k
                 << "\n尺度 h="
                 << (study.scale == MeshScale::Equivalent ? "sqrt(计算域面积/实际单元数)"
                                                          : "计算域最大单元直径")
                 << "；另打印 h_max。总耗时含读网格、GPU 准备、求解、诊断、保存和资源释放。\n";
            if (base.equation == "ms") {
                *out << "所有网格 T=" << study.time.final_time
                     << "；时间格式=" << (base.bdf2 ? "BDF2（首步 Euler）" : "Euler") << '\n';
                if (study.time.policy == TimePolicy::Balanced)
                    *out << "dt_target=" << study.time.coefficient << "*h^"
                         << double(base.k + 1) / (base.bdf2 ? 2 : 1)
                         << "；匹配时间/空间误差阶，不保证有限网格已进入渐近区。\n";
                else
                    *out << "固定目标 dt=" << study.time.fixed_dt << "；时间误差可能影响细网格空间阶。\n";
                *out << "步数向上取整" << (study.time.smooth_steps ? "到 2^a*5^b" : "")
                     << "，dt=T/N；最大步数=" << study.time.max_steps << "，不加半步时间。\n";
            }
            *out << "累计 CSV=" << study.table_file << "\n累计结果记录=" << study.terminal_file << '\n'
                 << std::flush;
        }
        std::vector<MeshResult> rows;
        for (size_t index = 0; index < study.meshes.size(); ++index) {
            const auto& entry = study.meshes[index];
            const auto start = Clock::now();
            std::cout << "\n[" << index + 1 << '/' << study.meshes.size() << "] 正在计算 " << entry.label
                      << " 网格=" << entry.file << " 输出=" << entry.output << std::endl;
            vem::StraightMeshReader mesh;
            const bool vtk = std::filesystem::path(entry.file).extension() == ".vtk";
            if (!(vtk ? mesh.read_mesh_vtk(entry.file) : mesh.read_mesh(entry.file)))
                throw std::runtime_error("mesh load failed: " + entry.file);
            const auto measure = measure_mesh(mesh);
            MeshResult r;
            r.label = entry.label;
            r.h = study.scale == MeshScale::Equivalent ? measure.h_equivalent : measure.h_max;
            r.h_max = measure.h_max;
            if (!rows.empty() && !(rows.back().h > r.h))
                throw std::invalid_argument("mesh sequence must have strictly decreasing h: " + entry.label);
            SolverOptions current = base;
            current.mesh = entry.file;
            current.output = entry.output;
            if (base.equation == "ms") {
                r.time_grid = make_time_grid(r.h, base.k, base.bdf2, study.time);
                current.dt = r.time_grid.dt;
                current.steps = r.time_grid.steps;
            }
            for (std::ostream* out :
                 {static_cast<std::ostream*>(&std::cout), static_cast<std::ostream*>(&transcript)})
                *out << "mesh=" << entry.label << " cells=" << measure.cells << " h=" << std::scientific
                     << std::setprecision(8) << r.h << " h_max=" << r.h_max
                     << " dt_target=" << r.time_grid.target_dt << " dt=" << r.time_grid.dt
                     << " steps=" << r.time_grid.steps << '\n'
                     << std::flush;
            if (run_case(current, &r.solve, &mesh) != 0)
                throw std::runtime_error("solve failed at " + entry.label + ": " + r.solve.failure);
            r.wall_seconds = std::chrono::duration<double>(Clock::now() - start).count();
            const auto released = sample_memory();
            for (auto* out : {static_cast<std::ostream*>(&std::cout), static_cast<std::ostream*>(&transcript)})
                *out << "memory phase=released mesh=" << entry.label << " rss_bytes=" << released.rss_bytes
                     << " gpu_used_bytes=" << released.gpu_used_bytes << " gpu_free_bytes=" << released.gpu_free_bytes
                     << '\n' << std::flush;
            if (base.equation == "ms" &&
                std::abs(r.solve.time - study.time.final_time) >
                    16 * std::numeric_limits<double>::epsilon() * study.time.final_time)
                throw std::runtime_error("final time mismatch; cannot compute a spatial order");
            r.scalar_order.assign(r.solve.errors.mass.size(), std::numeric_limits<double>::quiet_NaN());
            r.flux_order = r.scalar_order;
            if (exact && !rows.empty())
                for (size_t c = 0; c < r.scalar_order.size(); ++c) {
                    const auto& prev = rows.back();
                    r.scalar_order[c] = observed_order(prev.h, r.h, scalar(prev, study.error_domain)[c],
                                                       scalar(r, study.error_domain)[c]);
                    r.flux_order[c] = observed_order(prev.h, r.h, flux(prev, study.error_domain)[c],
                                                     flux(r, study.error_domain)[c]);
                }
            rows.push_back(r);
            append_csv(table, r, study);
            print_tables(std::cout, rows, study, exact);
            print_tables(transcript, rows, study, exact);
            if (!transcript)
                throw std::runtime_error("cannot write study transcript");
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "网格实验停止：" << e.what() << "；已完成的累计结果保留。\n";
        if (transcript)
            transcript << "FAILED: " << e.what() << '\n' << std::flush;
        return 1;
    }
}
} // namespace vgpu
