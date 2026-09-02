"""
绘制 MS 方程单个时刻浓度云图：数值解 vs 真解（曲边物理域）
- 3 行 2 列：每行一个组分，左数值解，右真解
- 每组分独立颜色尺度
- 叠加曲边网格线
用法:
    python plot_ms_comparison.py [数据目录] [时刻] [输出png]
示例:
    python plot_ms_comparison.py data/ms_data/ms_conv_8x8_plot 0.5 comparison.png
"""
import sys
import os
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


EPS = 0.05  # SinPerturbationMapping eps=0.05

def exact_concentration(i, x, y, t):
    if i == 0:
        return 0.25 * math.sin(2 * math.pi * x) * math.sin(8 * math.pi * t) + 0.25
    elif i == 1:
        return 0.25 * math.sin(3 * math.pi * y) * math.sin(6 * math.pi * t) + 0.25
    else:
        c0 = 0.25 * math.sin(2 * math.pi * x) * math.sin(8 * math.pi * t) + 0.25
        c1 = 0.25 * math.sin(3 * math.pi * y) * math.sin(6 * math.pi * t) + 0.25
        return 1.0 - c0 - c1


def load_csv(path):
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
        f.readline()
        for line in f:
            parts = line.strip().split(',')
            tris.append([int(parts[2]), int(parts[3]), int(parts[4])])
    return np.array(tris, dtype=np.int32)


def load_edges(path):
    edges = {}
    with open(path, 'r') as f:
        f.readline()
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


def main():
    data_dir = (sys.argv[1] if len(sys.argv) > 1
                else 'data/ms_data/ms_conv_8x8_plot')
    t_target = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    out_file = (sys.argv[3] if len(sys.argv) > 3
                else os.path.join(data_dir, 'comparison.png'))

    # 找数值解文件
    fname = f'concentration_t{t_target:.6f}.csv'
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        # 找最近的
        import glob
        files = sorted(glob.glob(os.path.join(data_dir, 'concentration_t*.csv')))
        print(f'找不到 {fname}，可用文件：')
        for f in files:
            print(f'  {os.path.basename(f)}')
        return

    print(f'数值解: {fpath}')

    # 加载数据
    arr, col_idx = load_csv(fpath)
    triangles = load_triangles(os.path.join(data_dir, 'triangles.csv'))
    edges = load_edges(os.path.join(data_dir, 'curved_edges.csv'))

    x = arr[:, col_idx['x']]
    y = arr[:, col_idx['y']]
    c0_num = arr[:, col_idx['c0']]
    c1_num = arr[:, col_idx['c1']]
    c2_num = arr[:, col_idx['c2']]

    n_cells = int(arr[:, col_idx['cell']].max()) + 1
    print(f'采样点数: {len(x)}, 单元数: {n_cells}')

    # 计算真解
    c0_ex = np.array([exact_concentration(0, xi, yi, t_target)
                      for xi, yi in zip(x, y)])
    c1_ex = np.array([exact_concentration(1, xi, yi, t_target)
                      for xi, yi in zip(x, y)])
    c2_ex = np.array([exact_concentration(2, xi, yi, t_target)
                      for xi, yi in zip(x, y)])

    tri = Triangulation(x, y, triangles)

    # 绘图
    fig, axes = plt.subplots(3, 2, figsize=(13, 16))
    fig.suptitle(
        f'Maxwell-Stefan Concentration  t = {t_target:.3f}  '
        f'({n_cells} cells, curved VEM k=1)',
        fontsize=14, y=0.995)

    comp_names = ['c_1', 'c_2', 'c_3']
    num_vals = [c0_num, c1_num, c2_num]
    exact_vals = [c0_ex, c1_ex, c2_ex]

    for row in range(3):
        vn = num_vals[row]
        ve = exact_vals[row]
        name = comp_names[row]

        vmin = min(vn.min(), ve.min())
        vmax = max(vn.max(), ve.max())
        pad = (vmax - vmin) * 0.02
        vmin -= pad
        vmax += pad
        levels = np.linspace(vmin, vmax, 60)

        # 数值解
        ax = axes[row, 0]
        tcf = ax.tricontourf(tri, vn, levels=levels,
                             vmin=vmin, vmax=vmax, cmap='jet')
        for eid in sorted(edges.keys()):
            ex, ey = edges[eid]
            ax.plot(ex, ey, color='black', linewidth=0.2, alpha=0.4)
        ax.set_title(f'Component {name} — Numerical', fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

        # 真解
        ax = axes[row, 1]
        ax.tricontourf(tri, ve, levels=levels,
                       vmin=vmin, vmax=vmax, cmap='jet')
        for eid in sorted(edges.keys()):
            ex, ey = edges[eid]
            ax.plot(ex, ey, color='black', linewidth=0.2, alpha=0.4)
        ax.set_title(f'Component {name} — Exact', fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

        cbar = fig.colorbar(tcf, ax=axes[row, :], shrink=0.82, pad=0.02)
        cbar.set_label(name, fontsize=11)

    plt.subplots_adjust(left=0.05, right=0.93, top=0.97, bottom=0.04,
                        hspace=0.22, wspace=0.12)
    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'图像已保存: {out_file}')


if __name__ == '__main__':
    main()
