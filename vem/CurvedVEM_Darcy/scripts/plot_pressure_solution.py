#!/usr/bin/env python3
"""使用 C++ 导出的显式单元拓扑绘制曲边域压力场。"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "pressure_plot"
POINT_FILE = DATA_DIR / "pressure_points.csv"
TRIANGLE_FILE = DATA_DIR / "pressure_triangles.csv"
EDGE_FILE = DATA_DIR / "curved_edges.csv"


def read_edges():
    edges = {}
    with EDGE_FILE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            edge = int(row["edge"])
            edges.setdefault(edge, []).append(
                (int(row["point"]), float(row["x"]), float(row["y"]))
            )
    return {
        edge: sorted(points, key=lambda item: item[0])
        for edge, points in edges.items()
    }


def add_curved_mesh(ax, edges):
    for points in edges.values():
        x = [point[1] for point in points]
        y = [point[2] for point in points]
        ax.plot(x, y, color="#111827", linewidth=0.7, zorder=4)


def draw_pressure(points, connectivity, edges, value_name, title, output,
                  vmin, vmax):
    figure, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    levels = np.linspace(vmin, vmax, 31)
    last_contour = None

    # 每个离散压力单元独立绘制，禁止跨单元插值。原始单元内的耳切
    # 三角形可以共享连续多项式值，但为保持实现简单仍按单元筛选显式拓扑。
    for cell in np.unique(points["cell"]):
        point_mask = points["cell"] == cell
        cell_points = points[point_mask]
        global_ids = cell_points["point_id"].astype(int)
        local_index = {point_id: index for index, point_id in enumerate(global_ids)}

        triangle_mask = connectivity["cell"] == cell
        cell_triangles_global = np.column_stack(
            [connectivity[name][triangle_mask].astype(int)
             for name in ("p0", "p1", "p2")]
        )
        cell_triangles = np.array(
            [[local_index[p] for p in triangle]
             for triangle in cell_triangles_global],
            dtype=int,
        )
        triangulation = mtri.Triangulation(
            cell_points["x"], cell_points["y"], cell_triangles
        )
        last_contour = ax.tricontourf(
            triangulation,
            cell_points[value_name],
            levels=levels,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            extend="both",
            zorder=1,
        )

    add_curved_mesh(ax, edges)
    colorbar = figure.colorbar(last_contour, ax=ax, pad=0.025)
    colorbar.set_label("pressure")
    ax.set_title(title)
    ax.set_xlabel("physical x")
    ax.set_ylabel("physical y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(np.min(points["x"]), np.max(points["x"]))
    ax.set_ylim(np.min(points["y"]), np.max(points["y"]))
    figure.savefig(output, dpi=220, facecolor="white")
    plt.close(figure)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    points = np.genfromtxt(POINT_FILE, delimiter=",", names=True)
    connectivity = np.genfromtxt(TRIANGLE_FILE, delimiter=",", names=True)
    edges = read_edges()
    exact = points["p_exact"]
    numerical = points["p_numerical"]
    vmin = float(min(np.min(exact), np.min(numerical)))
    vmax = float(max(np.max(exact), np.max(numerical)))

    draw_pressure(
        points,
        connectivity,
        edges,
        "p_exact",
        "Exact pressure on curved physical mesh",
        DATA_DIR / "pressure_exact.png",
        vmin,
        vmax,
    )
    draw_pressure(
        points,
        connectivity,
        edges,
        "p_numerical",
        "Numerical pressure on curved physical mesh",
        DATA_DIR / "pressure_numerical.png",
        vmin,
        vmax,
    )

    difference = numerical - exact
    print(f"points: {len(points)}")
    print(f"triangles: {len(connectivity)}")
    print(f"exact range: [{np.min(exact):.6g}, {np.max(exact):.6g}]")
    print(
        f"numerical range: [{np.min(numerical):.6g}, "
        f"{np.max(numerical):.6g}]"
    )
    print(f"sample RMSE: {np.sqrt(np.mean(difference**2)):.6g}")
    print(DATA_DIR / "pressure_exact.png")
    print(DATA_DIR / "pressure_numerical.png")


if __name__ == "__main__":
    main()
