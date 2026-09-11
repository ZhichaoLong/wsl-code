# 独立主函数与六算例完整求解验证（2026-09-10）

本阶段修改入口与共享求解流程的组织，CUDA 数学核和 GMRES 实现未改。

运行命令（工作目录为工程根目录）：

```bash
make -j4
./build/vem_gpu_all --output output/all_examples_20260910
./build/vem_gpu --equation ms --example 2 --scheme euler --output output/single_entry_euler_20260910
```

全部算例使用默认 16 单元网格、k=1；MS 使用 BDF2（Euler 首步），dt=0.001，三步至 t=0.003。以下为完整求解，未使用 benchmark-only。

| 算例 | 自由度 | 最终时间 | 最终相对残差 | 状态 |
|---|---:|---:|---:|---|
| darcy1 | 177 | 0 | 9.447098e-12 | converged |
| darcy2 | 177 | 0 | 3.264558e-12 | converged |
| ms1 | 528 | 0.003 | 9.355729e-12 | converged |
| ms2 | 528 | 0.003 | 9.257095e-12 | converged |
| ms3 | 528 | 0.003 | 9.993948e-12 | converged |
| ms4 | 528 | 0.003 | 9.513362e-12 | converged |

- `all_examples_20260910.log`：真实 GPU 六算例运行标准输出，6/6 收敛。
- `summary.csv` 与各 `*_report.json`：汇总与完整报告快照。summary 中的相对路径对应原输出目录 `output/all_examples_20260910/`。
- `test_main_results.log`：检查六算例身份、时间步数、自由度数量、残差、报告/解输出、MS 1/3 质量守恒、CLI 参数和已有解回归。对照重构前的 Darcy 1、MS 1/2/3/4 最终解，最大绝对差 2.97e-15；另检查单算例入口的 Euler 解。
- `test_main_suite.log`：主机端模拟 MS 2 返回失败，验证仍执行 MS 3/4、只记录一个失败、整体返回非零，并验证公共参数透传和禁止基准模式。日志中的 5/6 是故意注入的模拟失败，不是 GPU 算例失败。
- 临时测试源为 `tests/test_main_suite.cpp`、`tests/test_main_results.py`，通过后按用户要求删除，保留本目录证据。

首次沙箱内运行因 CUDA 驱动不可访问失败；使用获准的沙箱外 GPU 执行后六例通过。本阶段未重新进行 Compute Sanitizer 检查，原 WDDM 限制仍然适用。
