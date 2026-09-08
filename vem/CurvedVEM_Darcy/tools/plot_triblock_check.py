#!/usr/bin/env python3
"""
画 TriBlockSwirl 逐单元 Q8 映射的几何图，用来核对 C++ 实现是否正确。

数据由 build/tools/export_triblock_mesh 导出（在 CurvedVEM_Darcy 目录下运行）。
本脚本只读文件、只画图，不重新实现任何映射公式——如果它自己算一遍再和 C++ 比，
两边都写错同一处就查不出来了。要比的话应该拿 vem/mesh/TriBlockSwirl 的 Python
参考实现来比，那是独立写的。

产出 data/triblock/triblock_check.png，四个子图：
  (a) Q8 单元边界（组装路径），按 detJ 上色 —— 有缝/有叠/有翻转在这里现形
  (b) Q8 边界（黑）叠精确生成器曲线（红虚）—— 看 Q8 二次插值抓没抓住曲边
  (c) detJ 分布 + 界面标注 —— 看「一单元一雅可比」和跨界面跳变
  (d) 三个网格层的 Q8 边界并排 —— 看加密时是否收敛到同一张曲边域
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

OUT_DIR = "data/triblock"


def setup_cjk_font():
    """图上标题带中文，matplotlib 默认的 DejaVu Sans 没有汉字，会画成方框。
    按路径直接注册系统里已有的中文字体，注册不上就退回英文标题——
    宁可标题难看，也不要一张全是方框的图冒充能读。"""
    for path in ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                 "/usr/share/fonts/truetype/arphic-gkai00mp/gkai00mp.ttf",
                 "/mnt/c/Windows/Fonts/msyh.ttc",
                 "/mnt/c/Windows/Fonts/simhei.ttf"):
        if not os.path.exists(path):
            continue
        try:
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
            plt.rcParams["font.sans-serif"] = [name] + \
                list(plt.rcParams["font.sans-serif"])
            plt.rcParams["axes.unicode_minus"] = False
            return name
        except Exception:
            continue
    return None


def read_polylines(path, want_tag=None):
    """读 'tag x0 y0 x1 y1 ...' 或 'idx x0 y0 ...' 格式，返回 [(tag, (M,2) array)]"""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]
            if want_tag is not None and tag != want_tag:
                continue
            v = np.asarray(parts[1:], dtype=float)
            out.append((tag, v.reshape(-1, 2)))
    return out


def read_detj(path):
    d = np.loadtxt(path, comments="#")
    return d[:, 1], d[:, 2], d[:, 3]     # center, min, max


def main():
    font = setup_cjk_font()
    if font is None:
        print("警告：没找到可用的中文字体，图上标题会是方框")
    else:
        print("图形字体: %s" % font)

    if not os.path.isdir(OUT_DIR):
        sys.exit("找不到 %s——先在 CurvedVEM_Darcy 下跑 "
                 "./build/tools/export_triblock_mesh 2 4 8" % OUT_DIR)

    N_MAIN = 4
    cells = [p for _, p in read_polylines("%s/q8_cells_N%d.txt" % (OUT_DIR, N_MAIN))]
    det_c, det_lo, det_hi = read_detj("%s/detj_N%d.txt" % (OUT_DIR, N_MAIN))
    exact = read_polylines("%s/exact_lines_N%d.txt" % (OUT_DIR, N_MAIN))
    iface = dict(read_polylines("%s/interfaces.txt" % OUT_DIR))

    assert len(cells) == len(det_c), "单元数与 detJ 行数不符"

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    # ---- (a) Q8 单元边界，按 detJ 上色 ------------------------------------
    ax = axes[0, 0]
    polys = [c for c in cells]
    pc = PolyCollection(polys, array=det_c, cmap="viridis",
                        edgecolors="k", linewidths=0.6)
    ax.add_collection(pc)
    fig.colorbar(pc, ax=ax, fraction=0.046, label="detJ（质心处）")
    ax.set_title("(a) Q8 单元边界（physical_coords 组装路径）\n"
                 "%d 单元；填充无缝无叠即 (R2) 成立" % len(cells))

    # ---- (b) Q8 边界 vs 精确生成器曲线 -----------------------------------
    ax = axes[0, 1]
    segs = [np.vstack([c, c[:1]]) for c in cells]
    ax.add_collection(LineCollection(segs, colors="k", linewidths=1.0))
    for tag, p in exact:
        ax.plot(p[:, 0], p[:, 1], "r--", lw=0.8, alpha=0.85)
    ax.plot([], [], "k-", lw=1.0, label="Q8 单元边界（C++ 组装路径）")
    ax.plot([], [], "r--", lw=0.8, label="精确生成器 G 沿参考网格线")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("(b) Q8 二次插值 vs 精确曲线\n红虚线看不见 = Q8 完全贴住 G")

    # ---- (c) detJ + 界面标注 ---------------------------------------------
    ax = axes[1, 0]
    pc = PolyCollection(polys, array=det_c, cmap="coolwarm",
                        edgecolors="0.5", linewidths=0.3)
    ax.add_collection(pc)
    fig.colorbar(pc, ax=ax, fraction=0.046, label="detJ")
    for tag, style, lbl in (("D", "k-", "竖直分界线 D（贯穿全高）"),
                            ("H", "k--", "水平界面 H（仅 ξ≥a）")):
        if tag in iface:
            p = iface[tag]
            ax.plot(p[:, 0], p[:, 1], style, lw=2.5, label=lbl)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("(c) detJ 分布与三块界面\n"
                 "detJ ∈ [%.4f, %.4f]，全为正" % (det_lo.min(), det_hi.max()))

    # ---- (d) 三层网格 -----------------------------------------------------
    ax = axes[1, 1]
    colors = {2: "tab:blue", 4: "tab:orange", 8: "0.15"}
    widths = {2: 2.2, 4: 1.1, 8: 0.35}
    for N in (8, 4, 2):
        fn = "%s/q8_cells_N%d.txt" % (OUT_DIR, N)
        if not os.path.exists(fn):
            continue
        cc = [p for _, p in read_polylines(fn)]
        segs = [np.vstack([c, c[:1]]) for c in cc]
        ax.add_collection(LineCollection(segs, colors=colors[N],
                                         linewidths=widths[N], alpha=0.9))
        ax.plot([], [], color=colors[N], lw=widths[N],
                label="square_%dx%d → %d 单元" % (N, N, len(cc)))
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("(d) 三个网格层叠加\n外边界与界面重合 = 加密收敛到同一曲边域")

    for ax in axes.ravel():
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle("算例 4：TriBlockSwirl 逐单元 Q8 映射几何核对", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = "%s/triblock_check.png" % OUT_DIR
    fig.savefig(out, dpi=130)
    print("已保存 %s" % out)

    # 把图上不容易目测的量印出来，免得「看着对」代替了「是对的」
    print("\n单元数 %d" % len(cells))
    print("detJ  质心 [%.6f, %.6f]  单元内极值 [%.6f, %.6f]"
          % (det_c.min(), det_c.max(), det_lo.min(), det_hi.max()))
    print("负 detJ 单元数: %d" % int((det_lo <= 0).sum()))

    # Q8 边界与精确曲线的最大偏差：对每个 Q8 边界点找最近的精确曲线点。
    # 这是插值误差的粗估（下界），量级应是 O(h^3) 的弧高。
    allexact = np.vstack([p for _, p in exact])
    allq8 = np.vstack(cells)
    step = max(1, len(allq8) // 4000)     # 抽稀，避免 O(n^2) 爆掉
    sub = allq8[::step]
    d = np.hypot(sub[:, None, 0] - allexact[None, :, 0],
                 sub[:, None, 1] - allexact[None, :, 1]).min(axis=1)
    print("Q8 边界点到精确曲线的最大距离（抽 %d 点估计）: %.3e" % (len(sub), d.max()))


if __name__ == "__main__":
    main()
