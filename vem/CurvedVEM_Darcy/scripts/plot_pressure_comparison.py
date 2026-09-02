#!/usr/bin/env python3
"""使用 C++ 导出的显式单元拓扑绘制曲边域压力场 - 真解与数值解并排对比"""

from pathlib import Path
import csv
import sys

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


def draw_single_pressure(ax, points, connectivity, edges, value_name, title,
                         vmin, vmax):
    levels = np.linspace(vmin, vmax, 31)
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
        contour = ax.tricontourf(
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
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("physical x", fontsize=10)
    ax.set_ylabel("physical y", fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(np.min(points["x"]), np.max(points["x"]))
    ax.set_ylim(np.min(points["y"]), np.max(points["y"]))
    return contour


def draw_comparison(mesh_name, points, connectivity, edges):
    exact = points["p_exact"]
    numerical = points["p_numerical"]
    vmin = float(min(np.min(exact), np.min(numerical)))
    vmax = float(max(np.max(exact), np.max(numerical)))
    rmse = np.sqrt(np.mean((numerical - exact)**2))

    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    contour1 = draw_single_pressure(ax1, points, connectivity, edges,
                                    "p_exact", "Exact pressure", vmin, vmax)
    contour2 = draw_single_pressure(ax2, points, connectivity, edges,
                                    "p_numerical", f"Numerical pressure (RMSE={rmse:.6f})", vmin, vmax)

    colorbar = figure.colorbar(contour1, ax=[ax1, ax2], pad=0.025)
    colorbar.set_label("pressure")

    figure.suptitle(f"Pressure Comparison - {mesh_name}", fontsize=14, y=1.02)
    output_file = DATA_DIR / f"pressure_comparison_{mesh_name}.png"
    figure.savefig(output_file, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_pressure_comparison.py <mesh_name>")
        print("Example: python3 plot_pressure_comparison.py square_1x1")
        sys.exit(1)

    mesh_name = sys.argv[1]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    points = np.genfromtxt(POINT_FILE, delimiter=",", names=True)
    connectivity = np.genfromtxt(TRIANGLE_FILE, delimiter=",", names=True)
    edges = read_edges()

    draw_comparison(mesh_name, points, connectivity, edges)


if __name__ == "__main__":
    main()
