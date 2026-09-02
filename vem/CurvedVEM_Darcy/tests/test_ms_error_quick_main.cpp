/**
 * 快速验证：一个时间步后的误差（时间误差可忽略，看空间收敛阶）
 */
#include "../solver/MSSolver.h"
#include <petsc.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, NULL, NULL);
    try {
        const int k = 1;
        const int gauss = 9;
        const double dt = 1e-4;
        const double final_time = 1e-4;  // 只跑一步
        const int picard_max = 50;
        const double picard_tol = 1e-12;

        std::vector<int> nx_list = {1, 2, 4, 8};

        std::cout << "\n  单步验证（dt=" << dt
                  << "，时间误差可忽略）\n\n";
        printf("%-8s %-8s %-14s %-14s %-14s %-14s\n",
               "网格", "单元数", "c0_err", "c1_err", "J0_err", "J1_err");
        std::cout << std::string(72, '-') << "\n";

        std::vector<double> c0_errs, c1_errs, j0_errs, j1_errs, hs;

        for (int nx : nx_list) {
            std::string mesh_file = "mesh/data/square_" + std::to_string(nx)
                                  + "x" + std::to_string(nx) + ".msh";
            vem::StraightMeshReader reader;
            if (!reader.read_mesh(mesh_file))
                throw std::runtime_error("无法读取网格: " + mesh_file);

            MaxwellStefan::MSProblem problem;
            problem.set_problem_index(2);
            problem.init_problem();

            MSSolver solver(reader, problem, dt, final_time,
                            picard_max, picard_tol,
                            gauss, gauss, k, false, "quick_test");
            solver.initialize();
            solver.solveTimeStepping("quick");

            auto ce = solver.computeConcentrationL2Error(gauss);
            auto fe = solver.computeFluxL2Error(gauss);
            int nc = solver.getMeshData()->num_cells;
            double h = 1.0 / nx;

            printf("%-8s %-8d %-14.4e %-14.4e %-14.4e %-14.4e\n",
                   (std::to_string(nx)+"x"+std::to_string(nx)).c_str(),
                   nc, ce[0], ce[1], fe[0], fe[1]);

            c0_errs.push_back(ce[0]);
            c1_errs.push_back(ce[1]);
            j0_errs.push_back(fe[0]);
            j1_errs.push_back(fe[1]);
            hs.push_back(h);
        }

        // 收敛阶
        std::cout << "\n  收敛阶：\n";
        printf("%-16s %-10s %-10s %-10s %-10s\n",
               "网格对", "c0", "c1", "J0", "J1");
        std::cout << std::string(56, '-') << "\n";
        for (size_t i = 1; i < hs.size(); ++i) {
            double logr = std::log(hs[i-1]/hs[i]);
            double oc0 = std::log(c0_errs[i-1]/c0_errs[i]) / logr;
            double oc1 = std::log(c1_errs[i-1]/c1_errs[i]) / logr;
            double oj0 = std::log(j0_errs[i-1]/j0_errs[i]) / logr;
            double oj1 = std::log(j1_errs[i-1]/j1_errs[i]) / logr;
            printf("%-16s %-10.2f %-10.2f %-10.2f %-10.2f\n",
                   (std::to_string((int)(1/hs[i-1]))+"→"+
                    std::to_string((int)(1/hs[i]))).c_str(),
                   oc0, oc1, oj0, oj1);
        }
        std::cout << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        PetscFinalize();
        return 1;
    }
    PetscFinalize();
    return 0;
}
