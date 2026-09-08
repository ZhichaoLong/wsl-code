#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 C++ 导出的曲边网格逐层画成单张黑白图，一个网格一张。

只画一样东西：q8_cells_N{N}.txt 里每个单元的闭合边界折线，纯黑细线。
这条数据来自 physical_coords(cell_idx, xi, eta)——**真正参与组装的路径**，
所以图上相邻单元之间有没有缝、有没有叠，就是 (R2) 成不成立。

本脚本不重算映射，只搬点画线；不加颜色、不叠精确曲线、不叠别的网格层，
避免把「不同网格层之间的差」误读成「相邻单元没对上」。

用法: python3 tools/plot_triblock_bw.py [N ...]     默认 2 4 8 16
输出: data/triblock/mesh_bw_N{N}.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "data", "triblock")


def read_cells(fn):
    """读 q8_cells 文件，返回每个单元的边界点数组（不含重复的闭合点）。"""
    cells = []
    with open(fn) as f:
        for line in f:
            t = line.split()
            if not t or t[0].startswith("#"):
                continue
            v = np.array([float(x) for x in t[1:]], dtype=float).reshape(-1, 2)
            cells.append(v)
    return cells


def plot_one(N, lw, dpi=170):
    fn = os.path.join(OUT_DIR, "q8_cells_N%d.txt" % N)
    if not os.path.exists(fn):
        print("跳过 N=%d：找不到 %s" % (N, fn))
        return None
    cells = read_cells(fn)
    # 闭合：末点接回首点，否则每个单元会缺一条边
    segs = [np.vstack([c, c[:1]]) for c in cells]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.add_collection(LineCollection(segs, colors="black", linewidths=lw))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors="0.4", labelsize=9)
    ax.set_title("square_%dx%d  ->  %d cells" % (N, N, len(cells)),
                 fontsize=12, color="black")

    out = os.path.join(OUT_DIR, "mesh_bw_N%d.png" % N)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("N=%2d  %5d 单元  ->  %s" % (N, len(cells), os.path.normpath(out)))
    return out


if __name__ == "__main__":
    Ns = [int(a) for a in sys.argv[1:]] or [2, 4, 8, 16]
    # 单元越多线越细，否则线宽本身会糊掉单元边界
    lws = {2: 1.4, 4: 1.0, 8: 0.7, 16: 0.45, 32: 0.3, 64: 0.18, 128: 0.10}
    # N>=64 时单元宽度已接近像素尺度，只靠减线宽不够，同时提分辨率
    dpis = {64: 300, 128: 450}
    for N in Ns:
        plot_one(N, lws.get(N, 0.5), dpis.get(N, 170))
