/**
 * CurvedVEM MS - 算例 4（TriBlockSwirl 逐单元 Q8 映射）逐步存解主程序
 * 功能：在扭曲曲边网格上做三组分时间推进，每个时间步都把解向量落盘，
 *       供后续导出浓度场 / 逐帧绘图 / 合成 GIF。
 * 配置：k=1, 网格 square_8x8.msh（实际 16x16 = 256 曲边单元），
 *       终止时间 0.5，时间步 0.001，共 500 步，保存每一步解。
 * 算例：算例 4（真解、源项、边界条件与算例 2 逐字相同，只有映射不同）
 *
 * 与 MS_half_annulus_main.cpp 的区别只有三处：算例号 3 -> 4、
 *       多一句 set_mesh、保存子目录换名。其余配置刻意保持一字不差，
 *       这样 data/ms_data/ms_triblock_8x8/ 与算例 2 的 ms_conv_8x8/
 *       是同网格同步长同终止时间的两套解，可以直接逐帧对照。
 *
 * 关键：TriBlockSwirlMapping 要靠直边网格建每个单元的 Q8 节点表，
 *       故必须在 init_problem() **之前** 调用 problem.set_mesh(reader)。
 *       漏调会在 init_problem() 里抛异常（requires_mesh() 校验），
 *       不会静默退化成别的映射。
 *
 * 输出：data/ms_data/ms_triblock_8x8/{256}_{时刻}.txt（500 个文件）
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
    std::cout << "║  MS_triblock_frames - TriBlockSwirl 扭曲网格三组分扩散求解器     ║\n";
    std::cout << "║           算例4：逐单元 Q8 映射（真三块 + 全局 swirl）            ║\n";
    std::cout << "║           网格：square_8x8.msh (实际 16x16 曲边单元)             ║\n";
    std::cout << "║           多项式阶数：k = 1                                      ║\n";
    std::cout << "║           终止时间：0.5    时间步：0.001  (共 500 步)            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

    try {
        // ========== 配置参数 ==========
        const std::string mesh_file = "mesh/data/square_8x8.msh";
        const int problem_index = 4;
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
        // set_mesh 必须先于 init_problem：映射在 init_problem 末尾拿到网格才能建 Q8 表
        problem.set_mesh(reader);
        if (!problem.init_problem()) {
            throw std::runtime_error("无法初始化 MS 算例");
        }

        // ========== 创建求解器 ==========
        // 数据保存到 data/ms_data/ms_triblock_8x8/
        MSSolver solver(reader, problem,
                        delta_t, final_time,
                        picard_max_iter, picard_tol,
                        gauss_point_num, gauss_point_num_1d,
                        k, save_all_steps,
                        "ms_triblock_8x8");

        if (!solver.initialize()) {
            throw std::runtime_error("求解器初始化失败");
        }

        std::cout << "\n  单元数: " << solver.getMeshData()->num_cells
                  << "  总自由度: " << solver.getTotalDofAll()
                  << "\n\n";

        // ========== 时间推进 ==========
        PetscErrorCode ierr = solver.solveTimeStepping("ms_triblock", use_gpu);
        if (ierr != 0) {
            throw std::runtime_error("时间推进求解失败");
        }

        std::cout << "\n  ✓ 求解完成！解保存在 data/ms_data/ms_triblock_8x8/\n\n";

    } catch (const std::exception& exception) {
        std::cerr << "\n  错误：" << exception.what() << "\n\n";
        PetscFinalize();
        return 1;
    }

    PetscFinalize();
    return 0;
}
