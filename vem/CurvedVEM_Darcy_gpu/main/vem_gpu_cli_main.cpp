/**
 * 可选命令行兼容入口：保留已有自动化脚本使用的参数解析功能。
 * 日常算例请编辑 main/ 下按方程命名的主函数；本入口只把命令行参数交给相同的求解驱动。
 */
#include "solver_driver.h"
#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    vgpu::PetscSession petsc; // 必须晚于全部 Mat/Vec/KSP 释放再 Finalize。
    try {
        return vgpu::run_case(vgpu::parse_options(argc, argv));
    } catch (const vgpu::HelpRequested&) {
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "命令行算例运行失败：" << e.what() << '\n';
        return 1;
    }
}
