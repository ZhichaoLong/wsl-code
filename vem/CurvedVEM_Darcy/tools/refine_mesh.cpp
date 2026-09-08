/**
 * refine_mesh.cpp
 *
 * 悬点网格生成工具。
 * 对 mesh/data/square_nxn.msh 系列网格做 x 方向 [0.25, 0.75] 条带内的
 * 四分加密，生成带悬点的多边形网格（VTK + 简易文本格式），输出到
 * mesh/data_refined/。
 *
 * 用法：./build/tools/refine_mesh
 * 编译：make -j8
 *
 * 本工具不依赖 PETSc，不修改任何现有功能文件。
 */

#include "../core/mesh_refiner.h"
#include "../mesh/straight_mesh.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

// 需要处理的网格序列（跳过 square_1x1.msh）
const int ns[] = {2, 4, 8, 16, 32, 64, 128};
const int num_ns = sizeof(ns) / sizeof(ns[0]);

// 输出目录
const char* OUT_DIR = "mesh/data_refined";

// 输出信息
void print_stats(const std::string& name, const vem::RefineStats& s) {
    std::cout << "  " << name << ":\n"
              << "    输入 " << s.num_cells_in << " 单元"
              << " -> 输出 " << s.num_cells_out << " 单元"
              << "  (加密 " << s.num_refined << " 个单元)\n"
              << "    悬点 " << s.num_hanging << " 个\n"
              << "    四边形 " << s.num_quads
              << "  五边形 " << s.num_pentagons
              << "  其他 " << s.num_other
              << "  最大边数 " << s.max_poly << "\n";
}

}  // anonymous namespace

int main() {
    // 确保输出目录存在
    std::string cmd = std::string("mkdir -p ") + OUT_DIR;
    int ret = std::system(cmd.c_str());
    (void)ret;

    int total_ok = 0, total_fail = 0;

    for (int i = 0; i < num_ns; ++i) {
        int n = ns[i];
        // 沿用源文件的 n 命名，输出可直接追溯到输入。
        // 注意 square_nxn.msh 实际含 2n x 2n 个单元，名字里的 n 不是单元数。
        std::string stem = "square_" + std::to_string(n) + "x" + std::to_string(n);
        std::string msh_file = "mesh/data/" + stem + ".msh";
        std::string vtk_file = std::string(OUT_DIR) + "/" + stem + "_refined.vtk";
        std::string poly_file = std::string(OUT_DIR) + "/" + stem + "_refined.poly";

        std::cout << "[" << (i + 1) << "/" << num_ns << "] "
                  << stem << " ... ";
        std::cout.flush();

        // 读网格
        vem::StraightMeshReader reader;
        if (!reader.read_mesh(msh_file)) {
            std::cout << "失败：无法读取 " << msh_file << "\n";
            ++total_fail;
            continue;
        }

        const vem::StraightMeshData& in = reader.get_mesh_data();

        // 加密
        vem::PolyMeshData out;
        vem::RefineStats stats;
        if (!vem::refine_x_strip(in, 0.25, 0.75, out, stats)) {
            std::cout << "失败：加密仅支持四边形输入网格\n";
            ++total_fail;
            continue;
        }

        // 写 VTK
        if (!vem::write_vtk(out, vtk_file)) {
            std::cout << "失败：无法写入 " << vtk_file << "\n";
            ++total_fail;
            continue;
        }

        // 写 poly
        if (!vem::write_poly(out, poly_file)) {
            std::cout << "失败：无法写入 " << poly_file << "\n";
            ++total_fail;
            continue;
        }

        // 自检
        std::vector<std::string> msgs;
        int n_err = vem::validate_poly_mesh(out, msgs);
        if (n_err > 0) {
            std::cout << "通过，但自检发现 " << n_err << " 个问题：\n";
            for (size_t j = 0; j < msgs.size(); ++j)
                std::cout << "    - " << msgs[j] << "\n";
        } else {
            std::cout << "通过\n";
        }

        print_stats(stem, stats);
        ++total_ok;
    }

    std::cout << "\n====== 完成：成功 " << total_ok
              << " / " << (total_ok + total_fail)
              << " ======\n";
    return (total_fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}