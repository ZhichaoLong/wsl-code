/**
 * 多算例调度：接收 main 中显式列出的参数，不在内部隐藏算例选择。
 * 各算例顺序占用 GPU；失败后继续后续算例，并把实际报告路径写入汇总表。
 */
#include "solver_driver.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

namespace vgpu {
namespace {
// CSV 路径可能包含逗号或引号；双引号转义避免文件名破坏汇总表的列。
std::string csv_string(const std::string& value) {
    std::string out = "\"";
    for (char c : value) {
        if (c == '"')
            out += '"';
        out += c;
    }
    return out + '"';
}
} // namespace

int run_examples(const std::vector<SolverOptions>& cases, const std::string& summary_file) {
    if (cases.empty() || summary_file.empty())
        throw std::invalid_argument("case list and summary file must not be empty");

    // 在运行前验证整个列表，防止不同算例或汇总文件相互覆盖。
    const auto summary_path = std::filesystem::weakly_canonical(std::filesystem::absolute(summary_file));
    std::set<std::filesystem::path> outputs{summary_path};
    std::vector<SolverOptions> checked;
    for (const auto& entry : cases) {
        auto current = validated_options(entry);
        if (current.benchmark_only || current.benchmark)
            throw std::invalid_argument(
                "the example suite requires full solves; use the benchmark main for timing");
        const auto directory = std::filesystem::weakly_canonical(std::filesystem::absolute(current.output));
        for (const auto& name : {current.files.log, current.files.history, current.files.solution,
                                 current.files.report, current.files.parameters}) {
            const auto file = std::filesystem::weakly_canonical(directory / name);
            if (!outputs.insert(file).second)
                throw std::invalid_argument("duplicate output path in example suite: " + file.string());
        }
        checked.push_back(current);
    }
    std::filesystem::create_directories(summary_path.parent_path());
    std::ofstream summary(summary_path);
    if (!summary)
        throw std::runtime_error("cannot open example summary");
    summary << "equation,example,status,report\n";
    int failures = 0;
    for (const auto& current : checked) {
        std::cout << "\n=== " << current.equation << ' ' << current.example << " ===" << std::endl;
        // run_case 返回前析构 GPU 算子与向量，下一例不会保留上一例的显存。
        const int result = run_case(current);
        failures += result != 0;
        const auto report = std::filesystem::weakly_canonical(
            std::filesystem::absolute(std::filesystem::path(current.output) / current.files.report));
        summary << current.equation << ',' << current.example << ',' << (result == 0 ? "converged" : "failed")
                << ',' << csv_string(report.lexically_relative(summary_path.parent_path()).generic_string())
                << '\n';
        summary.flush();
        if (!summary)
            throw std::runtime_error("cannot write example summary");
    }
    std::cout << "\nExamples converged: " << checked.size() - failures << '/' << checked.size()
              << "; summary: " << summary_path << std::endl;
    return failures == 0 ? 0 : 1;
}

// 兼容原有“统一参数运行六例”的调用；主函数可改用 run_examples 逐例配置。
int run_all_examples(const SolverOptions& options) {
    std::vector<SolverOptions> cases;
    for (const auto* equation : {"darcy", "ms"}) {
        const int count = std::string(equation) == "darcy" ? 2 : 4;
        for (int example = 1; example <= count; ++example) {
            SolverOptions current = options;
            current.equation = equation;
            current.example = example;
            current.output =
                (std::filesystem::path(options.output) / (current.equation + std::to_string(example)))
                    .string();
            cases.push_back(current);
        }
    }
    return run_examples(cases, (std::filesystem::path(options.output) / "summary.csv").string());
}
} // namespace vgpu
