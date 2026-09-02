/**
 * @file test_ms_direction_main.cpp
 * @brief 输出方向调整向量，用于与 FVM 端参考结果对比
 *
 * 输出格式（每个单元一段）：
 *   elem <id> ndofs=<total> flux_per_comp=<f> conc_per_comp=<c> ncomp=<n>
 *   <dir_0> <dir_1> ... <dir_N-1>
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

int main(int argc, char* argv[]) {
    PetscInitialize(&argc, &argv, NULL, NULL);

    std::string mesh_file = "mesh/data/square_1x1.msh";
    std::string out_file = "build/direction_vector_vem.txt";
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

    std::ofstream fout(out_file);
    fout << std::fixed;

    for (int elem = 0; elem < md.num_cells; ++elem) {
        int dof_flux = ms_solver.getDofPerElementFlux(elem);
        int dof_conc = ms_solver.getDofPerElementConc();
        std::vector<double> dir = ms_solver.getDirectionVector(elem);
        int total = static_cast<int>(dir.size());

        fout << "elem " << elem
             << " ndofs=" << total
             << " flux_per_comp=" << dof_flux
             << " conc_per_comp=" << dof_conc
             << " ncomp=" << n_comp << "\n";
        for (int i = 0; i < total; ++i) {
            if (i > 0) fout << " ";
            fout << (dir[i] > 0 ? "1" : "-1");
        }
        fout << "\n";
    }

    fout.close();
    std::cout << "方向向量已写入: " << out_file << "\n";
    std::cout << "共 " << md.num_cells << " 个单元, k=" << k
              << ", ncomp=" << n_comp << "\n";

    // 在屏幕上打印所有单元的详细信息
    for (int elem = 0; elem < md.num_cells; ++elem) {
        int dof_flux = ms_solver.getDofPerElementFlux(elem);
        int edge_count = md.nodes_per_cell[elem];
        std::vector<double> dir = ms_solver.getDirectionVector(elem);

        std::cout << "\n--- 单元 " << elem
                  << " (质心: " << md.cell_centroid_x[elem]
                  << ", " << md.cell_centroid_y[elem]
                  << ", 边数: " << edge_count << ") ---\n";
        std::cout << "通量自由度（第0组分）:\n";
        for (int j = 0; j < dof_flux; ++j) {
            std::cout << "  dof " << std::setw(2) << j << ": "
                      << (dir[j] > 0 ? " 1" : "-1");
            int edge_dofs = (k + 1) * edge_count;
            if (j < edge_dofs) {
                int degree = j / edge_count;
                int le = j % edge_count;
                int ge = md.global_edges[md.cell_edge_indices[elem] + le];
                bool is_bnd = (md.edge_occurrence[ge] == 1);
                std::cout << "  [边" << le << "(g" << ge << "), d="
                          << degree << (is_bnd ? ", 边界" : ", 内部") << "]";
            } else {
                std::cout << "  [泡 " << (j - edge_dofs) << "]";
            }
            std::cout << "\n";
        }
        std::cout << "浓度自由度（所有组分）: 全部 1.0\n";
    }

    PetscFinalize();
    return 0;
}
