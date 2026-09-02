/**
 * @file test_ms_loc2glob_main.cpp
 * @brief 测试局部-全局自由度映射 getLocalToGlobalDofs
 *
 * 输出每个单元的全局自由度索引到文件，用于与 FVM 端对比。
 * 同时做基本的自洽性检查。
 */
#include "../solver/MSSolver.h"
#include "../examples/ms_problem.h"
#include "../mesh/straight_mesh.h"

#include <petsc.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <cmath>

int main(int argc, char* argv[]) {
    PetscInitialize(&argc, &argv, NULL, NULL);

    std::string mesh_file = "mesh/data/square_1x1.msh";
    std::string out_file = "build/loc2glob_vem.txt";
    if (argc > 1) mesh_file = argv[1];
    if (argc > 2) out_file = argv[2];

    vem::StraightMeshReader mesh;
    if (!mesh.read_mesh(mesh_file)) {
        std::cerr << "无法读取网格: " << mesh_file << std::endl;
        PetscFinalize();
        return 1;
    }

    const auto& md = mesh.get_mesh_data();

    MaxwellStefan::MSProblem ms_problem;
    ms_problem.set_problem_index(1);
    ms_problem.init_problem();

    const int k = 1;
    MSSolver ms_solver(mesh, ms_problem, 0.01, 1.0, 10, 1e-8, 9, 9, k);
    ms_solver.initialize();

    int n_comp = ms_solver.getNumComponents();
    int total_flux = ms_solver.getTotalDofFluxAll();
    int total_conc = ms_solver.getTotalDofConcAll();
    int total_dof = ms_solver.getTotalDofAll();

    std::cout << "k = " << k << ", 组分数 = " << n_comp << "\n";
    std::cout << "全局通量自由度: " << total_flux << "\n";
    std::cout << "全局浓度自由度: " << total_conc << "\n";
    std::cout << "全局总自由度: " << total_dof << "\n";
    std::cout << "单元数: " << md.num_cells << "\n";

    std::ofstream fout(out_file);
    std::set<PetscInt> all_flux_dofs;
    std::set<PetscInt> all_conc_dofs;
    bool all_pass = true;

    for (int elem = 0; elem < md.num_cells; ++elem) {
        std::vector<PetscInt> dofs = ms_solver.getLocalToGlobalMap(elem);
        int dof_flux = ms_solver.getDofPerElementFlux(elem);
        int dof_conc = ms_solver.getDofPerElementConc();
        int local_size = static_cast<int>(dofs.size());
        int expected = (dof_flux + dof_conc) * n_comp;

        fout << "elem " << elem
             << " ndofs=" << local_size
             << " flux_per_comp=" << dof_flux
             << " conc_per_comp=" << dof_conc << "\n";
        for (int i = 0; i < local_size; ++i) {
            if (i > 0) fout << " ";
            fout << dofs[i];
        }
        fout << "\n";

        // 基本检查
        if (local_size != expected) {
            std::cout << "✗ 单元 " << elem << ": 局部自由度数量不对 "
                      << local_size << " != " << expected << "\n";
            all_pass = false;
        }

        int flux_only = dof_flux * n_comp;
        int conc_only = dof_conc * n_comp;

        // 通量范围检查
        for (int i = 0; i < flux_only; ++i) {
            if (dofs[i] < 0 || dofs[i] >= total_flux) {
                std::cout << "✗ 单元 " << elem << " 通量 dof " << i
                          << " = " << dofs[i] << " 越界 [0, " << total_flux << ")\n";
                all_pass = false;
            }
            all_flux_dofs.insert(dofs[i]);
        }

        // 浓度范围检查
        for (int i = 0; i < conc_only; ++i) {
            PetscInt d = dofs[flux_only + i];
            if (d < total_flux || d >= total_dof) {
                std::cout << "✗ 单元 " << elem << " 浓度 dof " << i
                          << " = " << d << " 越界 [" << total_flux << ", " << total_dof << ")\n";
                all_pass = false;
            }
            all_conc_dofs.insert(d);
        }

        // 单元内无重复
        std::set<PetscInt> unique_check(dofs.begin(), dofs.end());
        if (static_cast<int>(unique_check.size()) != local_size) {
            std::cout << "✗ 单元 " << elem << " 内有重复自由度\n";
            all_pass = false;
        }

        // 打印前几个做人工检查
        std::cout << "\n单元 " << elem << " (local dofs = " << local_size << "):\n";
        std::cout << "  第0组分通量前8个: ";
        for (int i = 0; i < std::min(8, dof_flux); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << dofs[i];
        }
        std::cout << "\n";
        std::cout << "  第0组分浓度: ";
        int conc0_start = dof_flux * n_comp;  // 第0组分浓度起始
        for (int i = 0; i < dof_conc; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << dofs[conc0_start + i];
        }
        std::cout << "\n";
        if (n_comp > 1) {
            std::cout << "  第1组分通量前8个: ";
            for (int i = 0; i < std::min(8, dof_flux); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << dofs[dof_flux + i];
            }
            std::cout << "\n";
        }
    }

    fout.close();

    // 全局去重检查
    int expected_flux = ms_solver.getTotalDofFlux() * n_comp;
    int expected_conc = ms_solver.getTotalDofConc() * n_comp;
    std::cout << "\n========== 全局自洽性检查 ==========\n";
    std::cout << "唯一通量自由度数: " << all_flux_dofs.size()
              << " / 期望 " << expected_flux;
    if (static_cast<int>(all_flux_dofs.size()) == expected_flux) {
        std::cout << " ✓\n";
    } else {
        std::cout << " ✗\n";
        all_pass = false;
    }

    std::cout << "唯一浓度自由度数: " << all_conc_dofs.size()
              << " / 期望 " << expected_conc;
    if (static_cast<int>(all_conc_dofs.size()) == expected_conc) {
        std::cout << " ✓\n";
    } else {
        std::cout << " ✗\n";
        all_pass = false;
    }

    std::cout << "\n映射已写入: " << out_file << "\n";
    std::cout << (all_pass ? "✓ 所有检查通过" : "✗ 存在错误") << "\n";

    PetscFinalize();
    return all_pass ? 0 : 1;
}
