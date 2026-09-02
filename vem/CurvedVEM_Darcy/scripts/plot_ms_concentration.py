"""
批量绘制 MS 方程三组分浓度云图（曲边物理域）
- 三个组分共一张图，统一颜色尺度（彩虹色 jet）
- 叠加曲边网格线
- 批量处理目录下所有时刻的浓度数据
用法:
    python plot_ms_concentration.py [数据目录] [输出目录]
"""
import sys
import os
import glob
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


def load_concentration(path):
    """加载单个时刻的浓度点数据"""
    data = []
    with open(path, 'r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            data.append([float(x) for x in parts])
    arr = np.array(data)
    col_idx = {name: i for i, name in enumerate(header)}
    return arr, col_idx


def load_triangles(path):
    tris = []
    with open(path, 'r') as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            tris.append([int(parts[2]), int(parts[3]), int(parts[4])])
    return np.array(tris, dtype=np.int32)


def load_edges(path):
    edges = {}
    with open(path, 'r') as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            edge_id = int(parts[0])
            x = float(parts[2])
            y = float(parts[3])
            if edge_id not in edges:
                edges[edge_id] = ([], [])
            edges[edge_id][0].append(x)
            edges[edge_id][1].append(y)
    return edges


def find_time_files(data_dir):
    """找到所有 concentration_t*.csv 文件，按时刻排序"""
    pattern = os.path.join(data_dir, 'concentration_t*.csv')
    files = glob.glob(pattern)
    # 提取时刻
    time_files = []
    for f in files:
        basename = os.path.basename(f)
        m = re.match(r'concentration_t([0-9.]+)\.csv', basename)
        if m:
            t = float(m.group(1))
            time_files.append((t, f))
    time_files.sort(key=lambda x: x[0])
    return time_files


def plot_one_time(t, points_arr, col_idx, triangles, edges,
                  n_cells, out_path):
    """绘制单个时刻的浓度图（每帧独立颜色尺度，三组分共享同一色条）"""
    x = points_arr[:, col_idx['x']]
    y = points_arr[:, col_idx['y']]
    c0 = points_arr[:, col_idx['c0']]
    c1 = points_arr[:, col_idx['c1']]
    c2 = points_arr[:, col_idx['c2']]

    tri = Triangulation(x, y, triangles)

    comp_names = ['c_1', 'c_2', 'c_3']
    comp_values = [c0, c1, c2]

    # 该时刻三组分统一色条
    all_vals = np.concatenate(comp_values)
    vmin = all_vals.min()
    vmax = all_vals.max()
    pad = (vmax - vmin) * 0.02
    vmin -= pad
    vmax += pad

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.suptitle(
        f'Maxwell-Stefan concentration  t = {t:.3f}  '
        f'({n_cells} cells, curved VEM k=1)',
        fontsize=14, y=0.98)

    levels = np.linspace(vmin, vmax, 50)

    for ax, name, vals in zip(axes, comp_names, comp_values):
        tcf = ax.tricontourf(tri, vals, levels=levels,
                             vmin=vmin, vmax=vmax, cmap='rainbow')

        # 叠加曲边网格
        for edge_id in sorted(edges.keys()):
            ex, ey = edges[edge_id]
            ax.plot(ex, ey, color='black', linewidth=0.3, alpha=0.5)

        ax.set_title(f'Component {name}', fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

    # 统一颜色条
    cbar = fig.colorbar(tcf, ax=axes, shrink=0.82, pad=0.02)
    cbar.set_label('Concentration c', fontsize=11)

    plt.subplots_adjust(left=0.04, right=0.92, top=0.90, bottom=0.12, wspace=0.25)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/ms_concentration_plot'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(data_dir, 'frames')

    os.makedirs(out_dir, exist_ok=True)

    # 加载拓扑（所有时刻共享）
    triangles = load_triangles(os.path.join(data_dir, 'triangles.csv'))
    edges = load_edges(os.path.join(data_dir, 'curved_edges.csv'))

    # 找到所有时刻文件
    time_files = find_time_files(data_dir)
    print(f'找到 {len(time_files)} 个时刻的浓度数据')

    # 先获取单元数（读第一个文件）
    n_cells = 0
    if time_files:
        arr0, col_idx0 = load_concentration(time_files[0][1])
        n_cells = int(arr0[:, col_idx0['cell']].max()) + 1
    print(f'单元数: {n_cells}')
    print(f'每帧使用独立颜色尺度（三组分共享）')

    # 逐帧绘制
    for idx, (t, fpath) in enumerate(time_files):
        arr, col_idx = load_concentration(fpath)
        out_name = f'frame_{idx:04d}_t{t:.6f}.png'
        out_path = os.path.join(out_dir, out_name)
        plot_one_time(t, arr, col_idx, triangles, edges,
                      n_cells, out_path)
        if (idx + 1) % 10 == 0 or idx == len(time_files) - 1:
            print(f'  [{idx+1}/{len(time_files)}] t={t:.3f} → {out_name}')

    print(f'\n全部绘制完成！共 {len(time_files)} 帧')
    print(f'输出目录: {out_dir}')


if __name__ == '__main__':
    main()
