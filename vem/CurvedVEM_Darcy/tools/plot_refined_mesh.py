#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_refined_mesh.py

画 mesh/data_refined/ 下的悬点网格。黑白，黑线，一个网格一张图。

用法：
    python3 tools/plot_refined_mesh.py              # 画全部
    python3 tools/plot_refined_mesh.py 2 8          # 只画 square_2x2 和 square_8x8

输出：mesh/data_refined/fig/square_nxn_refined.png
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

MESH_DIR = 'mesh/data_refined'
FIG_DIR = os.path.join(MESH_DIR, 'fig')
ALL_N = [2, 4, 8, 16, 32, 64, 128]


def read_poly(filename):
    """读 .poly：NODES n / 坐标 / CELLS m / 顶点数 + 顶点编号。"""
    with open(filename) as f:
        tokens = f.read().split()

    i = 0
    assert tokens[i] == 'NODES'
    num_nodes = int(tokens[i + 1])
    i += 2

    pts = []
    for _ in range(num_nodes):
        pts.append((float(tokens[i]), float(tokens[i + 1])))
        i += 2

    assert tokens[i] == 'CELLS'
    num_cells = int(tokens[i + 1])
    i += 2

    cells = []
    for _ in range(num_cells):
        nv = int(tokens[i])
        i += 1
        cells.append([int(tokens[i + k]) for k in range(nv)])
        i += nv

    return pts, cells


def find_hanging_nodes(pts, cells):
    """悬点 = 多边形里那个与前后邻点共线、且恰在其中点的顶点。"""
    hanging = set()
    for c in cells:
        nv = len(c)
        if nv <= 4:
            continue
        for k in range(nv):
            a, b, d = pts[c[k - 1]], pts[c[k]], pts[c[(k + 1) % nv]]
            if (abs((a[0] + d[0]) / 2 - b[0]) < 1e-12 and
                    abs((a[1] + d[1]) / 2 - b[1]) < 1e-12):
                hanging.add(c[k])
    return hanging


def plot_one(n, annotate):
    stem = 'square_%dx%d' % (n, n)
    src = os.path.join(MESH_DIR, stem + '_refined.poly')
    if not os.path.exists(src):
        print('  跳过 %s：文件不存在' % stem)
        return

    pts, cells = read_poly(src)
    hanging = find_hanging_nodes(pts, cells)

    # 线宽随网格加密变细，否则密网格会糊成一片黑
    lw = {2: 1.2, 4: 1.0, 8: 0.8, 16: 0.5, 32: 0.3, 64: 0.18, 128: 0.10}.get(n, 0.3)

    fig, ax = plt.subplots(figsize=(6.4, 6.4))

    verts = [[pts[v] for v in c] for c in cells]
    pc = PolyCollection(verts, facecolors='none', edgecolors='black',
                        linewidths=lw)
    ax.add_collection(pc)

    # 小网格才标悬点，大网格标了反而看不清
    if annotate and hanging:
        hx = [pts[h][0] for h in hanging]
        hy = [pts[h][1] for h in hanging]
        ax.plot(hx, hy, 'o', markersize=5, markerfacecolor='white',
                markeredgecolor='black', markeredgewidth=1.2,
                linestyle='none', zorder=5)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.axis('off')

    n_pent = sum(1 for c in cells if len(c) == 5)
    ax.set_title('%s  (%d cells, %d pentagons, %d hanging nodes)'
                 % (stem, len(cells), n_pent, len(hanging)),
                 fontsize=11, color='black')

    if not os.path.isdir(FIG_DIR):
        os.makedirs(FIG_DIR)
    out = os.path.join(FIG_DIR, stem + '_refined.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  %s -> %s' % (stem, out))


def main():
    args = sys.argv[1:]
    ns = [int(a) for a in args] if args else ALL_N
    for n in ns:
        plot_one(n, annotate=(n <= 16))


if __name__ == '__main__':
    main()
