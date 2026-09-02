/**
 * CurvedVEM MS - 半圆环三组分扩散主程序
 * 功能：半圆环几何下三组分扩散时间推进求解
 * 配置：k=1, 网格 8x8 (实际 16x16 单元), 终止时间 0.5, 时间步 0.001
 *       保存每一步解
 * 算例：算例 3（半圆环三组分扩散，趋于稳态）
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
    std::cout << "║  MS_half_annulus - 半圆环 Maxwell-Stefan 三组分扩散求解器       ║\n";
    std::cout << "║           算例3：半圆环映射 + 径向分层初值 + 零通量边界            ║\n";
    std::cout << "║           网格：square_8x8.msh (实际 16x16 曲边单元)             ║\n";
    std::cout << "║           多项式阶数：k = 1                                      ║\n";
    std::cout << "║           终止时间：0.5    时间步：0.001  (共 500 步)            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

    try {
        // ========== 配置参数 ==========
        const std::string mesh_file = "mesh/data/square_8x8.msh";
        const int problem_index = 3;
        const int k = 1;
        const int gauss_point_num = 9;
        const int gauss_point_num_1d = 9;

        const double delta_t = 0.001;
        const double final_time = 0.5;
        const int picard_max_iter = 50;
        const double picard_tol = 1e-6;
        const bool save_all_steps = true;
        const bool use_gpu = true;

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
        // 数据保存到 data/ms_data/ms_half_annulus_8x8/
        MSSolver solver(reader, problem,
                        delta_t, final_time,
                        picard_max_iter, picard_tol,
                        gauss_point_num, gauss_point_num_1d,
                        k, save_all_steps,
                        "ms_half_annulus_8x8");

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

        std::cout << "\n  ✓ 求解完成！解保存在 data/ms_data/ms_half_annulus_8x8/\n\n";

    } catch (const std::exception& exception) {
        std::cerr << "\n  错误：" << exception.what() << "\n\n";
        PetscFinalize();
        return 1;
    }

    PetscFinalize();
    return 0;
}
