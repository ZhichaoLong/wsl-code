#!/usr/bin/env python3
"""
算例 3（半圆环三组分扩散）在悬点网格上的结果检查图。

判据的来源：算例 3 的初值只依赖计算域坐标 ξ，半圆环映射 r = ξ + 0.5 是纯径向的，
边界条件是零通量 —— 所以真解**严格径向对称**，c_i 只是 r 的函数，与 θ 无关。

于是两张图：
  fig1  二维云图：三组分 × 若干时刻，直观看扩散过程
  fig2  径向塌缩图：把所有采样点画成 (r, c_i) 散点。正确的话必须塌缩成一条细线。
        加密条带 ξ∈[0.25,0.75] 对应 r∈[0.75,1.25]，用竖线标出 ——
        若五边形单元装配有误，曲线会恰在这两条线处折断、分叉或增厚。
        有四边形基线数据时叠加成黑色虚线做对照。

用法:
  python3 tools/plot_ms_annulus_check.py <hanging_csv_dir> [quad_csv_dir]
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R_IN, R_OUT = 0.5, 1.5
# 加密条带 ξ∈[0.25,0.75] 在物理域对应的半径区间（r = ξ + 0.5）
R_REFINE = (0.75, 1.25)
NCOMP = 3


def list_frames(d):
    """返回 [(t, path), ...]，按时刻排序"""
    out = []
    for name in os.listdir(d):
        m = re.match(r"concentration_t([0-9.]+)\.csv$", name)
        if m:
            out.append((float(m.group(1)), os.path.join(d, name)))
    out.sort()
    return out


def load(path):
    a = np.loadtxt(path, delimiter=",", skiprows=1)
    x, y = a[:, 2], a[:, 3]
    c = a[:, 4:4 + NCOMP]
    return x, y, c


def pick_times(frames, n):
    """在所有时刻里等间隔挑 n 个（含首尾）"""
    if len(frames) <= n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in idx]


def fig_contour(frames, out_png):
    times = pick_times(frames, 5)
    fig, axes = plt.subplots(NCOMP, len(times),
                             figsize=(3.1 * len(times), 3.0 * NCOMP),
                             squeeze=False)
    for j, (t, path) in enumerate(times):
        x, y, c = load(path)
        for i in range(NCOMP):
            ax = axes[i][j]
            # 半圆环是凹区域，tricontourf 会跨过内孔，用 mask 把内孔的三角形去掉
            tp = ax.tripcolor(x, y, c[:, i], shading="gouraud",
                              cmap="rainbow", vmin=0.0, vmax=1.0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            # 加密带的两条边界圆弧
            th = np.linspace(-np.pi / 2, np.pi / 2, 200)
            for r in R_REFINE:
                ax.plot(r * np.cos(th), r * np.sin(th), "k--", lw=0.8, alpha=0.9)
            if i == 0:
                ax.set_title("t = %.3f" % t, fontsize=11)
            if j == 0:
                ax.set_ylabel("$c_%d$" % i, fontsize=13)
    fig.colorbar(tp, ax=axes, shrink=0.6, label="concentration")
    fig.suptitle("Case 3 on hanging-node mesh: concentration fields\n"
                 "(black dashed arcs = refined band $r\\in[0.75,1.25]$)",
                 fontsize=12)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  ->", out_png)


def fig_radial(frames, out_png, quad_frames=None):
    times = pick_times(frames, 5)
    fig, axes = plt.subplots(1, len(times),
                             figsize=(3.4 * len(times), 3.4),
                             sharey=True, squeeze=False)
    axes = axes[0]

    quad_by_t = dict(quad_frames) if quad_frames else {}

    for j, (t, path) in enumerate(times):
        ax = axes[j]
        x, y, c = load(path)
        r = np.hypot(x, y)

        colors = ["tab:blue", "tab:orange", "tab:green"]
        for i in range(NCOMP):
            ax.plot(r, c[:, i], ".", ms=0.8, alpha=0.25,
                    color=colors[i], rasterized=True)

        # 四边形基线：同一时刻的解，按 r 排序后画成黑色虚线
        tq = min(quad_by_t, key=lambda s: abs(s - t)) if quad_by_t else None
        if tq is not None and abs(tq - t) < 1e-9:
            xq, yq, cq = load(quad_by_t[tq])
            rq = np.hypot(xq, yq)
            o = np.argsort(rq)
            for i in range(NCOMP):
                ax.plot(rq[o], cq[o, i], "k--", lw=0.7, alpha=0.9)

        for rr in R_REFINE:
            ax.axvline(rr, color="crimson", ls=":", lw=1.2)
        ax.set_xlim(R_IN, R_OUT)
        ax.set_ylim(-0.08, 1.08)
        ax.set_xlabel("r")
        ax.set_title("t = %.3f" % t, fontsize=11)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("concentration")

    handles = [plt.Line2D([], [], color=c, marker=".", ls="", ms=8,
                          label="$c_%d$ (hanging)" % i)
               for i, c in enumerate(["tab:blue", "tab:orange", "tab:green"])]
    if quad_by_t:
        handles.append(plt.Line2D([], [], color="k", ls="--", lw=1,
                                  label="quad mesh baseline"))
    handles.append(plt.Line2D([], [], color="crimson", ls=":", lw=1.2,
                              label="refined band edge"))
    axes[-1].legend(handles=handles, fontsize=8, loc="center right")

    fig.suptitle("Radial collapse test — exact solution depends on $r$ only.\n"
                 "Any spread or kink at $r=0.75,\\,1.25$ would indicate a "
                 "pentagon-assembly defect.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  ->", out_png)


def report_spread(frames):
    """量化 θ 依赖性：真解与 θ 无关，同一 r 上不同 θ 的值必须相等。

    不能简单地按 r 分箱：初值在 r=0.75/1.25 处是**真间断**，跨过间断的箱子里
    同时有 ≈1 和 ≈0 的点，量到的是间断本身而不是 θ 依赖。

    正确做法是按「同一 ξ 列 + 精确同一 r」分组。计算域网格是张量积的，
    同一 ξ 列里各单元的采样点 ξ 坐标完全相同，只是 θ 不同；
    间断落在单元交界上，两侧属于不同 ξ 列，自然被分开。
    """
    print("\n  θ 依赖性（同一 ξ 列、同一 r 上不同 θ 的最大差；真解应为 0）")
    print("  %-10s %-12s %-12s %-10s" % ("t", "全域", "加密带内", "样本组数"))
    for t, path in pick_times(frames, 6):
        a = np.loadtxt(path, delimiter=",", skiprows=1)
        cell = a[:, 1].astype(int)
        r = np.hypot(a[:, 2], a[:, 3])
        c = a[:, 4:4 + NCOMP]

        # 每个单元的 r 跨度 -> 标识它属于哪个 ξ 列
        col_of_cell = {}
        for e in np.unique(cell):
            m = cell == e
            col_of_cell[e] = (round(r[m].min(), 9), round(r[m].max(), 9))

        groups = {}
        for i in range(len(r)):
            key = (col_of_cell[cell[i]], round(r[i], 9))
            groups.setdefault(key, []).append(i)

        worst_all = 0.0
        worst_band = 0.0
        n_group = 0
        for (col, rr), idx in groups.items():
            if len(idx) < 2:
                continue
            n_group += 1
            sub = c[idx]
            d = float(np.max(sub.max(axis=0) - sub.min(axis=0)))
            worst_all = max(worst_all, d)
            if R_REFINE[0] - 1e-9 <= rr <= R_REFINE[1] + 1e-9:
                worst_band = max(worst_band, d)
        print("  %-10.4f %-12.3e %-12.3e %-10d"
              % (t, worst_all, worst_band, n_group))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    hang_dir = sys.argv[1]
    quad_dir = sys.argv[2] if len(sys.argv) > 2 else None

    frames = list_frames(hang_dir)
    if not frames:
        print("没有找到 concentration_t*.csv:", hang_dir)
        sys.exit(1)
    print("悬点网格: %d 个时刻，t ∈ [%.4f, %.4f]"
          % (len(frames), frames[0][0], frames[-1][0]))

    quad_frames = None
    if quad_dir and os.path.isdir(quad_dir):
        quad_frames = list_frames(quad_dir)
        print("四边形基线: %d 个时刻" % len(quad_frames))

    out_dir = os.path.join(hang_dir, "fig")
    os.makedirs(out_dir, exist_ok=True)

    fig_contour(frames, os.path.join(out_dir, "fields.png"))
    fig_radial(frames, os.path.join(out_dir, "radial_collapse.png"), quad_frames)
    report_spread(frames)


if __name__ == "__main__":
    main()
