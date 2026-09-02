#!/usr/bin/env python3
"""批量生成不同网格密度的压力对比图"""

import subprocess
import os
import sys

# 切换到项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# 要处理的网格列表
mesh_sizes = [1, 2, 4, 8, 16]

for n in mesh_sizes:
    mesh_name = f"square_{n}x{n}"
    mesh_file = f"mesh/data/{mesh_name}.msh"

    print(f"\n{'='*60}")
    print(f"Processing {mesh_name}...")
    print(f"{'='*60}")

    # 1. 导出压力数据
    cmd_export = [
        "./build/export_pressure_plot_data",
        mesh_file,
        "data/pressure_plot/pressure_points.csv",
        "data/pressure_plot/pressure_triangles.csv",
        "data/pressure_plot/curved_edges.csv"
    ]
    print(f"Running: {' '.join(cmd_export)}")
    result = subprocess.run(cmd_export, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error exporting data for {mesh_name}:")
        print(result.stderr)
        continue
    print(result.stdout)

    # 2. 生成对比图
    cmd_plot = [
        sys.executable,
        "scripts/plot_pressure_comparison.py",
        mesh_name
    ]
    print(f"Running: {' '.join(cmd_plot)}")
    result = subprocess.run(cmd_plot, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error plotting {mesh_name}:")
        print(result.stderr)
        continue
    print(result.stdout)

print(f"\n{'='*60}")
print("All plots generated successfully!")
print(f"{'='*60}")
print("\nGenerated files:")
for n in mesh_sizes:
    print(f"  - data/pressure_plot/pressure_comparison_square_{n}x{n}.png")
