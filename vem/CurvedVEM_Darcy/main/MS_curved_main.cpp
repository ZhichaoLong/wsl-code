/**
 * CurvedVEM MS - 主程序入口
 * 功能：Maxwell-Stefan 多组分扩散方程曲边混合虚拟元时间推进求解
 * 配置：k=1, 网格 1x1, 终止时间 0.1, 时间步 0.001
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
    // 如果有CUDA支持，提前设置 GPU 相关选项
#if defined(PETSC_HAVE_CUDA)
    setenv("CUDA_VISIBLE_DEVICES", "0", 1);
    PetscOptionsSetValue(NULL, "-use_gpu_aware_mpi", "0");
    PetscOptionsSetValue(NULL, "-device_type", "cuda");
    PetscOptionsSetValue(NULL, "-device_select", "0");
#endif

    // 初始化 PETSc
    PetscErrorCode error = PetscInitialize(&argc, &argv, NULL, NULL);
    if (error != 0) {
        return static_cast<int>(error);
    }

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║        MS_curved_main - 曲边虚单元法 Maxwell-Stefan 求解器       ║\n";
    std::cout << "║           算例1：正弦扰动曲边映射 + MS 多组分扩散                ║\n";
    std::cout << "║           网格：正方形 4x4 (实际 8x8 单元)                      ║\n";
    std::cout << "║           多项式阶数：k = 1                                      ║\n";
    std::cout << "║           终止时间：0.1    时间步：0.001                        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

    try {
        // ========== 配置参数 ==========
        const std::string mesh_file = "mesh/data/square_4x4.msh";
        const int problem_index = 2;
        const int k = 1;
        const int gauss_point_num = 9;      // 二维高斯积分点数
        const int gauss_point_num_1d = 9;  // 一维高斯积分点数

        const double delta_t = 0.001;
        const double final_time = 0.5;
        const int picard_max_iter = 50;
        const double picard_tol = 1e-8;
        const bool save_all_steps = false;  // 只保存最终时刻

        // ========== 读取网格 ==========
        vem::StraightMeshReader reader;
        if (!reader.read_mesh(mesh_file)) {
            throw std::runtime_error("无法读取网格: " + mesh_file);
        }

        // ========== 初始化算例 ==========
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
                        "solver_data");  // 算例1数据目录

        if (!solver.initialize()) {
            throw std::runtime_error("求解器初始化失败");
        }

        std::cout << "\n  单元数: " << solver.getMeshData()->num_cells
                  << "  总自由度: " << solver.getTotalDofAll()
                  << "\n\n";

        // ========== 时间推进 ==========
        PetscErrorCode ierr = solver.solveTimeStepping("ms_solution");
        if (ierr != 0) {
            throw std::runtime_error("时间推进求解失败");
        }

        std::cout << "\n  ✓ 求解完成！最终解保存在 data/ms_data/solver_data_final/\n\n";

    } catch (const std::exception& exception) {
        std::cerr << "\n  错误：" << exception.what() << "\n\n";
        PetscFinalize();
        return 1;
    }

    PetscFinalize();
    return 0;
}
