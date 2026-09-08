/**
 * CurvedVEM MS - 半圆环三组分扩散（悬点网格）
 *
 * 算例 3 原封不动：HalfAnnulusMapping + 径向分层初值 + 零通量边界。
 * 与 main/MS_half_annulus_main.cpp 的唯一区别是网格来源：
 *
 *     reader.read_mesh(".msh")   ->   reader.read_mesh_vtk(".vtk")
 *
 * 目的是检验悬点网格（四边形 + 五边形混合）能否走通整条求解链。
 * 算例 3 之所以适合做这个检验：
 *   1. HalfAnnulusMapping 是全局解析映射，requires_mesh() 为 false、忽略 cell_idx，
 *      不像算例 4 的 Q8 逐单元映射那样对「每单元 4 个角点」有硬性依赖
 *   2. 初值走 initial_concentration_comp（计算域，按 ξ 分层），不需要逆映射
 *   3. 初值的两个间断面恰在 ξ=0.25 / ξ=0.75，正是加密条带的边界；
 *      半圆环映射里 ξ 是径向坐标，加密带正好压在两个扩散锋面上
 *
 * 算例 3 无制造解，不能算收敛阶。判据改用三条守恒/渐近性质（见文末打印）：
 *   A. 组分和恒为 1（Maxwell-Stefan 的代数约束）
 *   B. 零通量边界下每个组分的总质量守恒
 *   C. t 足够大时趋于均匀稳态，稳态值 = 初始质量 / 区域面积
 *
 * 用法：
 *   ./build/MS_half_annulus_hanging [mesh.vtk] [final_time] [dt]
 * 默认：mesh/data_refined/square_8x8_refined.vtk, T=0.5, dt=0.001
 */

#include "../solver/MSSolver.h"

#include <petsc.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
#if defined(PETSC_HAVE_CUDA)
    setenv("CUDA_VISIBLE_DEVICES", "0", 1);
    PetscOptionsSetValue(NULL, "-use_gpu_aware_mpi", "0");
    PetscOptionsSetValue(NULL, "-device_type", "cuda");
    PetscOptionsSetValue(NULL, "-device_select", "0");
#endif

    PetscErrorCode error = PetscInitialize(&argc, &argv, NULL, NULL);
    if (error != 0) {
        return static_cast<int>(error);
    }

    try {
        // ========== 配置参数 ==========
        // 与 MS_half_annulus_main.cpp 保持一致，只有网格文件不同
        std::string mesh_file = "mesh/data_refined/square_8x8_refined.vtk";
        double final_time = 0.5;
        double delta_t = 0.001;

        if (argc > 1) mesh_file = argv[1];
        if (argc > 2) final_time = std::atof(argv[2]);
        if (argc > 3) delta_t = std::atof(argv[3]);

        const int problem_index = 3;
        const int k = 1;
        const int gauss_point_num = 9;
        const int gauss_point_num_1d = 9;
        const int picard_max_iter = 50;
        const double picard_tol = 1e-6;
        const bool save_all_steps = true;
        const bool use_gpu = true;

        // 输出目录：从网格文件名派生，例如 square_8x8_refined.vtk -> ms_half_annulus_hanging_8x8
        std::string tag = "ms_half_annulus_hanging";
        {
            size_t slash = mesh_file.find_last_of('/');
            std::string base = (slash == std::string::npos) ? mesh_file
                                                            : mesh_file.substr(slash + 1);
            size_t us = base.find('_');
            size_t re = base.find("_refined");
            if (us != std::string::npos && re != std::string::npos && re > us) {
                tag += "_" + base.substr(us + 1, re - us - 1);
            }
        }

        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  MS_half_annulus_hanging - 半圆环 MS 三组分扩散（悬点网格）      ║\n";
        std::cout << "║  算例3：半圆环映射 + 径向分层初值 + 零通量边界（与原算例完全一致）║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
        std::cout << "  网格文件 : " << mesh_file << "  (legacy VTK, 含悬点)\n"
                  << "  多项式阶 : k = " << k << "\n"
                  << "  时间     : T = " << final_time << ", dt = " << delta_t
                  << "  (共 " << static_cast<int>(final_time / delta_t + 0.5) << " 步)\n"
                  << "  输出目录 : data/ms_data/" << tag << "/\n\n";

        // ========== 读取网格：唯一与原算例不同的一行 ==========
        vem::StraightMeshReader reader;
        if (!reader.read_mesh_vtk(mesh_file)) {
            throw std::runtime_error("无法读取 VTK 网格: " + mesh_file);
        }

        // 把单元边数分布打出来，确认确实喂进去的是混合多边形网格而不是纯四边形
        {
            const vem::StraightMeshData& m = reader.get_mesh_data();
            int n_quad = 0, n_penta = 0, n_other = 0;
            for (int e = 0; e < m.num_cells; ++e) {
                if (m.nodes_per_cell[e] == 4) ++n_quad;
                else if (m.nodes_per_cell[e] == 5) ++n_penta;
                else ++n_other;
            }
            std::cout << "  单元构成 : " << n_quad << " 四边形 + " << n_penta
                      << " 五边形（悬点）";
            if (n_other > 0) std::cout << " + " << n_other << " 其他";
            std::cout << "\n";
            if (n_penta == 0) {
                std::cout << "  ⚠ 该网格不含五边形，这次跑的不是悬点网格\n";
            }
        }

        // ========== 初始化算例（算例 3，一字未改） ==========
        MaxwellStefan::MSProblem problem;
        problem.set_problem_index(problem_index);
        if (!problem.init_problem()) {
            throw std::runtime_error("无法初始化 MS 算例");
        }

        // ========== 创建求解器 ==========
        MSSolver solver(reader, problem,
                        delta_t, final_time,
                        picard_max_iter, picard_tol,
                        gauss_point_num, gauss_point_num_1d,
                        k, save_all_steps,
                        tag);

        if (!solver.initialize()) {
            throw std::runtime_error("求解器初始化失败");
        }

        std::cout << "\n  单元数: " << solver.getMeshData()->num_cells
                  << "  总自由度: " << solver.getTotalDofAll()
                  << "\n\n";

        // ========== 时间推进 ==========
        PetscErrorCode ierr = solver.solveTimeStepping("ms_half_annulus", use_gpu);
        if (ierr != 0) {
            throw std::runtime_error("时间推进求解失败");
        }

        std::cout << "\n  ✓ 求解完成！解保存在 data/ms_data/" << tag << "/\n\n";

    } catch (const std::exception& exception) {
        std::cerr << "\n  错误：" << exception.what() << "\n\n";
        PetscFinalize();
        return 1;
    }

    PetscFinalize();
    return 0;
}
