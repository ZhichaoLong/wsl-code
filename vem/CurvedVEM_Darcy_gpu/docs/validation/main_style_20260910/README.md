# main 源码配置与中文注释重构验收（2026-09-10）

主函数直接配置算例、网格、阶次、时间、GPU 参数、输出目录和各文件名；CLI 移至独立兼容入口。GPU 数学核、时间离散公式和预条件方法保持原样。

## 正式入口与实际命令

工作目录为工程根目录，先执行 `make -j4`：

```bash
./build/vem_gpu
./build/vem_gpu_all
./build/vem_gpu_benchmark
./build/vem_gpu_cli --equation darcy --example 2 --mesh mesh/data/hanging_demo.vtk --k 2 --output output/style_cli_hanging
```

前两个主函数使用源码中明确给定的 16 单元、k=1、dt=0.001、三步 BDF2 设置；Darcy 稳态只求解一次。每例的 parameters.txt 保存实际生效值（含解析后的 nq=9）。

| 算例 | 单元数 | 自由度 | 最终时间 | 最终相对残差 |
|---|---:|---:|---:|---:|
| darcy1 | 16 | 177 | 0 | 9.448303e-12 |
| darcy2 | 16 | 177 | 0 | 3.264600e-12 |
| ms1 | 16 | 528 | 0.003 | 9.355729e-12 |
| ms2 | 16 | 528 | 0.003 | 9.257095e-12 |
| ms3 | 16 | 528 | 0.003 | 9.993948e-12 |
| ms4 | 16 | 528 | 0.003 | 9.513362e-12 |

六例全部收敛；与 `output/all_examples_20260910` 的重构前解逐自由度比较，最大绝对差 7.577e-15，四个 MS 的解均完全相同。MS 1/3 的组分质量守恒检查通过；无真解误差保持 null/NaN。

性能主函数完成 4096 单元的基础矩阵、系数更新和算子应用计时，报告状态为 benchmark，没有将它计作 PDE 求解。兼容入口完成 3 单元 VTK 悬点二阶 Darcy 2 求解，解与原记录一致。

## 临时测试与证据

- `test_style_tokens.log`：对 22 个基础/方程/设备/CPU 文件删除注释和空白后的词法序列逐项比较，全部一致；全部 30 个正式 C++/CUDA 文件含中文模块说明。
- `test_style_config.log`：自动积分规则、原低阶规则、非法设置、每例参数透传、自定义报告名、CSV 引号/逗号转义、重复路径拒绝和模拟失败后继续，通过。
- `test_style_results.log`：六例、单例、参数快照、自定义文件名、Euler、CLI、性能入口及已有解回归，全部通过。
- `main_style_build.log`：修正文件头注释中的终止符误用后，最终构建无 warning/error。
- `custom_darcy/`：自定义解为 `flux, pressure.csv`（完整解在 output 中）、报告为 `state, report.json`、日志和参数分别为 `iterations.log` / `settings.txt`。
- `expected_failure/`：故意传不存在的网格，报告正确为 failed；随后 Darcy 和 Euler 仍完整收敛。`custom_summary.csv` 的 2/3 成功属于预期测试。
- 所有 GPU 运行均使用获准的沙箱外驱动访问。本阶段没有重跑受 WDDM 限制的 Compute Sanitizer。

测试源 `tests/test_style_config.cpp`、`tests/test_style_gpu.cpp`、`tests/test_style_tokens.py`、`tests/test_style_results.py` 通过后按用户要求删除。临时 GPU 测试链接为 `build/test_stage`，检查失败返回码与自定义文件真实存在后退出 0，随后删除。保留本目录日志/报告及输出目录中的完整解。

`summary.csv` 的相对报告路径在本目录可直接使用；`custom_summary.csv` 的相对路径对应原 `output/style_custom/`。`before_sha256.json` 保存修改前源文件哈希；`after_sha256.json` 保存本轮正式源码、构建配置与四个程序的哈希。
