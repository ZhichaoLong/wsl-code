#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 triblock_sweep 导出的各组参数画成黑白网格图，一组参数一张。

数据来自 triblock_sweep 的 export_cells()，走的是 physical_coords()——
即真正参与组装的那条路径。脚本不重算映射，只搬点画线。

用法: python3 tools/plot_triblock_sweep.py L0 Z3 m3a110 ...
输出: data/triblock/sweep/bw_<tag>.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "data", "triblock", "sweep")


def read_cells(fn):
    cells = []
    with open(fn) as f:
        for line in f:
            t = line.split()
            if not t or t[0].startswith("#"):
                continue
            v = np.array([float(x) for x in t[1:]], dtype=float).reshape(-1, 2)
            cells.append(v)
    return cells


def plot_one(tag):
    fn = os.path.join(DIR, "cells_%s_N8.txt" % tag)
    if not os.path.exists(fn):
        print("跳过 %s：找不到 %s" % (tag, fn))
        return
    cells = read_cells(fn)
    segs = [np.vstack([c, c[:1]]) for c in cells]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_collection(LineCollection(segs, colors="black", linewidths=0.7))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors="0.4", labelsize=9)
    ax.set_title("%s   (square_8x8 -> %d cells)" % (tag, len(cells)),
                 fontsize=13, color="black")

    out = os.path.join(DIR, "bw_%s.png" % tag)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("%-8s %5d 单元  ->  %s" % (tag, len(cells), os.path.normpath(out)))


if __name__ == "__main__":
    for tag in sys.argv[1:] or ["L0"]:
        plot_one(tag)
