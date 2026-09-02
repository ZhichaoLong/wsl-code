"""
批量绘制 MS 方程精确解浓度云图（曲边物理域）
- 三个组分共一张图，统一颜色尺度
- 叠加曲边网格线
- t=0 到 t=0.5，步长 0.001
用法:
    python plot_ms_exact_frames.py [网格文件] [输出目录]
"""
import sys
import os
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# ======================== 曲边映射 ========================
EPS = 0.05

def physical_coords(xi, eta):
    x = xi + EPS * math.sin(2 * math.pi * eta)
    y = eta + EPS * math.sin(2 * math.pi * xi)
    return x, y

# ======================== 精确浓度 ========================
def exact_concentration(i, x, y, t):
    if i == 0:
        return 0.25 * math.sin(2 * math.pi * x) * math.sin(8 * math.pi * t) + 0.25
    elif i == 1:
        return 0.25 * math.sin(3 * math.pi * y) * math.sin(6 * math.pi * t) + 0.25
    else:
        c0 = 0.25 * math.sin(2 * math.pi * x) * math.sin(8 * math.pi * t) + 0.25
        c1 = 0.25 * math.sin(3 * math.pi * y) * math.sin(6 * math.pi * t) + 0.25
        return 1.0 - c0 - c1

# ======================== 解析 Gmsh 4.1 ========================
def parse_msh4(msh_path):
    nodes = {}
    quads = []

    with open(msh_path, 'r') as f:
        lines = [l.rstrip('\n') for l in f.readlines()]

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line == '$Nodes':
            i += 1
            parts = lines[i].split()
            num_blocks = int(parts[0])
            i += 1
            for _ in range(num_blocks):
                bp = lines[i].split()
                num_in_block = int(bp[3])
                i += 1
                tags = []
                for _ in range(num_in_block):
                    tags.append(int(lines[i]))
                    i += 1
                for tag in tags:
                    cp = lines[i].split()
                    nodes[tag] = (float(cp[0]), float(cp[1]))
                    i += 1
        elif line == '$Elements':
            i += 1
            parts = lines[i].split()
            num_blocks = int(parts[0])
            i += 1
            for _ in range(num_blocks):
                bp = lines[i].split()
                dim = int(bp[0])
                etype = int(bp[2])
                num_in_block = int(bp[3])
                i += 1
                if dim == 2 and etype == 3:
                    for _ in range(num_in_block):
                        ep = lines[i].split()
                        quads.append([int(ep[1]), int(ep[2]),
                                      int(ep[3]), int(ep[4])])
                        i += 1
                else:
                    i += num_in_block
        else:
            i += 1

    sorted_tags = sorted(nodes.keys())
    tag_to_idx = {t: idx for idx, t in enumerate(sorted_tags)}
    node_coords = np.array([nodes[t] for t in sorted_tags])
    quad_idx = [[tag_to_idx[t] for t in q] for q in quads]
    return node_coords, np.array(quad_idx, dtype=int)

# ======================== 采样：每单元 N*N 个点 ========================
def generate_samples(node_coords, quads, N=10):
    """
    在每个四边形单元内生成 N*N 个采样点（计算域均匀），
    同时生成曲边网格边的采样。
    返回:
      pts_x, pts_y: 物理域坐标
      c0, c1, c2: 对应精确浓度（t=0 占位，实际逐帧计算）
      tri: 三角剖分
      edges_phys: 曲边网格线 [(xs, ys), ...]
    """
    n_cells = len(quads)
    all_x = []
    all_y = []
    cell_ids = []
    tri_list = []

    for e in range(n_cells):
        verts = node_coords[quads[e]]  # (4,2) 计算域顶点
        x0, y0 = verts[0]
        x1, y1 = verts[2]

        base = len(all_x)
        # 生成网格点
        for j in range(N + 1):
            for i in range(N + 1):
                xi = x0 + (x1 - x0) * i / N
                eta = y0 + (y1 - y0) * j / N
                px, py = physical_coords(xi, eta)
                all_x.append(px)
                all_y.append(py)
                cell_ids.append(e)

        # 生成三角形（两个三角 per 网格 cell）
        for j in range(N):
            for i in range(N):
                p00 = base + j * (N + 1) + i
                p10 = base + j * (N + 1) + i + 1
                p01 = base + (j + 1) * (N + 1) + i
                p11 = base + (j + 1) * (N + 1) + i + 1
                tri_list.append([p00, p10, p11])
                tri_list.append([p00, p11, p01])

    pts_x = np.array(all_x)
    pts_y = np.array(all_y)
    tri = Triangulation(pts_x, pts_y, np.array(tri_list, dtype=int))

    # 曲边网格线
    edge_set = set()
    for e in range(n_cells):
        v = quads[e]
        for k in range(4):
            a, b = v[k], v[(k + 1) % 4]
            if a > b:
                a, b = b, a
            edge_set.add((a, b))

    edges_phys = []
    for a, b in edge_set:
        p0 = node_coords[a]
        p1 = node_coords[b]
        ns = 30
        xs, ys = [], []
        for s in range(ns + 1):
            t = s / ns
            xi = (1 - t) * p0[0] + t * p1[0]
            eta = (1 - t) * p0[1] + t * p1[1]
            px, py = physical_coords(xi, eta)
            xs.append(px)
            ys.append(py)
        edges_phys.append((xs, ys))

    return pts_x, pts_y, tri, edges_phys, n_cells

# ======================== 主函数 ========================
def main():
    msh_file = (sys.argv[1] if len(sys.argv) > 1
                else 'mesh/data/square_4x4.msh')
    out_dir = (sys.argv[2] if len(sys.argv) > 2
               else 'data/ms_exact_frames')

    t_start = 0.0
    t_end = 0.5
    dt = 0.001

    os.makedirs(out_dir, exist_ok=True)

    print(f'网格: {msh_file}')
    print(f'输出: {out_dir}')
    print(f'时间: {t_start} ~ {t_end}, dt = {dt}')

    # 解析网格
    node_coords, quads = parse_msh4(msh_file)
    print(f'节点: {len(node_coords)}, 单元: {len(quads)}')

    # 生成采样点
    pts_x, pts_y, tri, edges_phys, n_cells = generate_samples(
        node_coords, quads, N=10)
    n_pts = len(pts_x)
    print(f'采样点数: {n_pts}')

    comp_names = ['c_1', 'c_2', 'c_3']

    n_frames = int(round((t_end - t_start) / dt)) + 1
    print(f'总帧数: {n_frames}')

    for f_idx in range(n_frames):
        t = t_start + f_idx * dt

        # 计算三个组分的精确浓度
        c_vals = []
        for c in range(3):
            vals = np.array([exact_concentration(c, x, y, t)
                             for x, y in zip(pts_x, pts_y)])
            c_vals.append(vals)

        # 统一颜色尺度（三组分共享）
        all_v = np.concatenate(c_vals)
        vmin = all_v.min()
        vmax = all_v.max()
        pad = (vmax - vmin) * 0.02
        vmin -= pad
        vmax += pad
        levels = np.linspace(vmin, vmax, 60)

        fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
        fig.suptitle(
            f'Maxwell-Stefan exact concentration  t = {t:.3f}  '
            f'({n_cells} cells, curved)',
            fontsize=14, y=0.98)

        for ax, name, vals in zip(axes, comp_names, c_vals):
            tcf = ax.tricontourf(tri, vals, levels=levels,
                                 vmin=vmin, vmax=vmax, cmap='jet')
            for ex, ey in edges_phys:
                ax.plot(ex, ey, color='black', linewidth=0.3, alpha=0.5)
            ax.set_title(f'Component {name}', fontsize=12)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')

        cbar = fig.colorbar(tcf, ax=axes, shrink=0.82, pad=0.02)
        cbar.set_label('Concentration c', fontsize=11)

        plt.subplots_adjust(left=0.04, right=0.92, top=0.90,
                            bottom=0.12, wspace=0.25)

        out_name = f'frame_{f_idx:04d}_t{t:.6f}.png'
        out_path = os.path.join(out_dir, out_name)
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        if (f_idx + 1) % 20 == 0 or f_idx == n_frames - 1:
            print(f'  [{f_idx+1}/{n_frames}] t={t:.3f} → {out_name}')

    print(f'\n全部完成！共 {n_frames} 帧')
    print(f'输出目录: {out_dir}')


if __name__ == '__main__':
    main()
