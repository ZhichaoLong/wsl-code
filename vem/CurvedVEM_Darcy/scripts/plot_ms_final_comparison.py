"""
绘制 MS 方程最终时刻浓度云图：数值解 vs 真解（曲边物理域）
- 每个组分一张子图，左侧数值解，右侧真解
- 三组分共 6 张子图
- 叠加曲边网格线
用法:
    python plot_ms_final_comparison.py [解文件] [网格文件] [输出png]
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
EPS = 0.05  # SinPerturbationMapping eps=0.05

def physical_coords(xi, eta):
    x = xi + EPS * math.sin(2 * math.pi * eta)
    y = eta + EPS * math.sin(2 * math.pi * xi)
    return x, y

def jacobian_det(xi, eta):
    dx_dxi = 1.0
    dx_deta = EPS * 2 * math.pi * math.cos(2 * math.pi * eta)
    dy_dxi = EPS * 2 * math.pi * math.cos(2 * math.pi * xi)
    dy_deta = 1.0
    return dx_dxi * dy_deta - dx_deta * dy_dxi

# ======================== 精确浓度（物理域） ========================
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
    """解析 msh 4.1 文件，返回节点坐标和四边形单元连接（物理编号从 1 开始）"""
    nodes = {}  # tag -> (x, y)
    quads = []  # list of [n1, n2, n3, n4]

    with open(msh_path, 'r') as f:
        lines = [l.rstrip('\n') for l in f.readlines()]

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line == '$Nodes':
            i += 1
            # 头部：numEntityBlocks numNodes minNodeTag maxNodeTag
            parts = lines[i].split()
            num_blocks = int(parts[0])
            total_nodes = int(parts[1])
            i += 1
            for _ in range(num_blocks):
                # entityDim entityTag parametric numNodesInBlock
                bp = lines[i].split()
                num_in_block = int(bp[3])
                i += 1
                # node tags
                tags = []
                for _ in range(num_in_block):
                    tags.append(int(lines[i]))
                    i += 1
                # node coordinates
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
                # entityDim entityTag elementType numElementsInBlock
                bp = lines[i].split()
                dim = int(bp[0])
                etype = int(bp[2])
                num_in_block = int(bp[3])
                i += 1
                if dim == 2 and etype == 3:  # 4-node quadrangle
                    for _ in range(num_in_block):
                        ep = lines[i].split()
                        # tag n1 n2 n3 n4
                        quads.append([int(ep[1]), int(ep[2]),
                                      int(ep[3]), int(ep[4])])
                        i += 1
                else:
                    i += num_in_block
        else:
            i += 1

    # 把节点字典按 tag 排序，建立索引
    sorted_tags = sorted(nodes.keys())
    tag_to_idx = {t: idx for idx, t in enumerate(sorted_tags)}
    node_coords = np.array([nodes[t] for t in sorted_tags])

    # 转换单元节点 tag 为索引
    quad_idx = []
    for q in quads:
        quad_idx.append([tag_to_idx[t] for t in q])

    return node_coords, np.array(quad_idx, dtype=int)

# ======================== 加载解向量 ========================
def load_solution(sol_path):
    with open(sol_path, 'r') as f:
        total_dof = int(f.readline().strip())
        values = []
        for line in f:
            values.append(float(line.strip()))
    return np.array(values), total_dof

# ======================== 从文件名提取时刻 ========================
def extract_time_from_filename(path):
    basename = os.path.basename(path)
    # 格式: 64_0.500000.txt
    name = os.path.splitext(basename)[0]
    parts = name.split('_')
    return float(parts[-1])

# ======================== P1 单项式求值 ========================
def eval_p1_monomials(cx, cy, hD, x, y):
    """
    计算 k=1 (P1) 三个单项式在 (x,y) 处的值
    单项式顺序: 1, (x-cx)/hD, (y-cy)/hD
    """
    xm = (x - cx) / hD
    ym = (y - cy) / hD
    return np.array([1.0, xm, ym])

# ======================== 主绘图 ========================
def main():
    # 参数
    sol_file = (sys.argv[1] if len(sys.argv) > 1
                else 'data/ms_data/ms_conv_4x4_final/64_0.500000.txt')
    msh_file = (sys.argv[2] if len(sys.argv) > 2
                else 'mesh/data/square_4x4.msh')
    out_file = (sys.argv[3] if len(sys.argv) > 3
                else 'data/ms_data/ms_conv_4x4_final/comparison.png')

    t = extract_time_from_filename(sol_file)
    print(f'解文件: {sol_file}')
    print(f'网格: {msh_file}')
    print(f'时刻: t = {t}')

    # 解析网格
    node_coords, quads = parse_msh4(msh_file)
    n_cells = len(quads)
    n_nodes = len(node_coords)
    print(f'节点数: {n_nodes}, 四边形单元数: {n_cells}')

    # 加载解向量
    sol, total_dof = load_solution(sol_file)
    print(f'总自由度: {total_dof}')

    # DOF 布局:
    # [所有通量 DOF | 所有浓度 DOF]
    # 浓度部分: 组分0(按单元排列) -> 组分1 -> 组分2
    # 每单元浓度 DOF: k=1 时为 3 (P1: 1, x, y)
    dof_conc_per_elem = 3
    n_comp = 3
    total_conc_per_comp = n_cells * dof_conc_per_elem
    total_flux = total_dof - n_comp * total_conc_per_comp
    print(f'通量 DOF 总数: {total_flux}')
    print(f'每组分浓度 DOF: {total_conc_per_comp}')

    # 提取每个单元的质心和直径（直边网格）
    cell_centers = np.zeros((n_cells, 2))
    cell_diameters = np.zeros(n_cells)
    for e in range(n_cells):
        verts = node_coords[quads[e]]  # (4, 2)
        cell_centers[e] = verts.mean(axis=0)
        # 直径 = 最大顶点间距
        dmax = 0.0
        for i in range(4):
            for j in range(i+1, 4):
                d = np.linalg.norm(verts[i] - verts[j])
                if d > dmax:
                    dmax = d
        cell_diameters[e] = dmax

    # 每个单元在参考域内的采样点（细分后用于三角剖分画图）
    # 每边采样 N 个点
    N = 8  # 每边采样点数，N*N 个采样点每单元

    # 预计算所有采样点的物理坐标和值
    all_points_x = []
    all_points_y = []
    all_c0_num = []
    all_c1_num = []
    all_c2_num = []
    all_c0_exact = []
    all_c1_exact = []
    all_c2_exact = []
    cell_id = []

    for e in range(n_cells):
        # 单元顶点参考坐标
        verts_ref = node_coords[quads[e]]  # 直边网格节点就是参考坐标
        x0, y0 = verts_ref[0]
        x1, y1 = verts_ref[2]  # 对角点

        # 浓度系数
        cx_center = cell_centers[e, 0]
        cy_center = cell_centers[e, 1]
        hD = cell_diameters[e]

        c_coeffs = []
        for c in range(n_comp):
            base = total_flux + c * total_conc_per_comp + e * dof_conc_per_elem
            c_coeffs.append(sol[base:base+dof_conc_per_elem])

        # 在参考单元内采样
        for i in range(N):
            for j in range(N):
                xi = x0 + (x1 - x0) * (i + 0.5) / N
                eta = y0 + (y1 - y0) * (j + 0.5) / N

                # 物理坐标
                px, py = physical_coords(xi, eta)

                # 数值解（P1 多项式，物理坐标求值）
                monom = eval_p1_monomials(cx_center, cy_center, hD, px, py)
                c0n = np.dot(c_coeffs[0], monom)
                c1n = np.dot(c_coeffs[1], monom)
                c2n = np.dot(c_coeffs[2], monom)

                # 真解
                c0e = exact_concentration(0, px, py, t)
                c1e = exact_concentration(1, px, py, t)
                c2e = exact_concentration(2, px, py, t)

                all_points_x.append(px)
                all_points_y.append(py)
                all_c0_num.append(c0n)
                all_c1_num.append(c1n)
                all_c2_num.append(c2n)
                all_c0_exact.append(c0e)
                all_c1_exact.append(c1e)
                all_c2_exact.append(c2e)
                cell_id.append(e)

    pts_x = np.array(all_points_x)
    pts_y = np.array(all_points_y)
    c0_num = np.array(all_c0_num)
    c1_num = np.array(all_c1_num)
    c2_num = np.array(all_c2_num)
    c0_ex = np.array(all_c0_exact)
    c1_ex = np.array(all_c1_exact)
    c2_ex = np.array(all_c2_exact)

    print(f'采样点数: {len(pts_x)}')

    # 构造三角剖分（用 Delaunay，在物理域内）
    tri = Triangulation(pts_x, pts_y)

    # 生成曲边网格线（每条边采样）
    # 收集所有边界边和内部边
    edge_set = set()
    for e in range(n_cells):
        v = quads[e]
        for k in range(4):
            n1, n2 = v[k], v[(k+1) % 4]
            # 标准化：小的在前
            if n1 > n2:
                n1, n2 = n2, n1
            edge_set.add((n1, n2))

    edges_phys = []
    for n1, n2 in edge_set:
        p1_ref = node_coords[n1]
        p2_ref = node_coords[n2]
        # 沿边采样 20 个点
        ns = 20
        ex, ey = [], []
        for s in range(ns + 1):
            xi = p1_ref[0] + (p2_ref[0] - p1_ref[0]) * s / ns
            eta = p1_ref[1] + (p2_ref[1] - p1_ref[1]) * s / ns
            px, py = physical_coords(xi, eta)
            ex.append(px)
            ey.append(py)
        edges_phys.append((ex, ey))

    # 绘图：3 行 2 列（数值解 | 真解）
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    fig.suptitle(
        f'Maxwell-Stefan Concentration  t = {t:.3f}  '
        f'({n_cells} cells, curved VEM k=1)',
        fontsize=14, y=0.995)

    comp_names = ['c_1', 'c_2', 'c_3']
    num_vals = [c0_num, c1_num, c2_num]
    exact_vals = [c0_ex, c1_ex, c2_ex]

    for row in range(3):
        vals_num = num_vals[row]
        vals_ex = exact_vals[row]
        name = comp_names[row]

        # 统一颜色尺度（数值解和真解共享）
        vmin = min(vals_num.min(), vals_ex.min())
        vmax = max(vals_num.max(), vals_ex.max())
        pad = (vmax - vmin) * 0.02
        vmin -= pad
        vmax += pad
        levels = np.linspace(vmin, vmax, 50)

        # 数值解
        ax = axes[row, 0]
        tcf = ax.tricontourf(tri, vals_num, levels=levels,
                             vmin=vmin, vmax=vmax, cmap='jet')
        for ex, ey in edges_phys:
            ax.plot(ex, ey, color='black', linewidth=0.3, alpha=0.5)
        ax.set_title(f'Component {name} - Numerical', fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

        # 真解
        ax = axes[row, 1]
        ax.tricontourf(tri, vals_ex, levels=levels,
                       vmin=vmin, vmax=vmax, cmap='jet')
        for ex, ey in edges_phys:
            ax.plot(ex, ey, color='black', linewidth=0.3, alpha=0.5)
        ax.set_title(f'Component {name} - Exact', fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

        # 颜色条
        cbar = fig.colorbar(tcf, ax=axes[row, :], shrink=0.85, pad=0.02)
        cbar.set_label(f'{name}', fontsize=11)

    plt.subplots_adjust(left=0.06, right=0.94, top=0.97, bottom=0.04,
                        hspace=0.25, wspace=0.15)
    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'\n图像已保存: {out_file}')


if __name__ == '__main__':
    main()
