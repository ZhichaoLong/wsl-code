#!/usr/bin/env python3
"""
有限体积法结果可视化
绘制每个时间步三个组分并排云图，并生成GIF动图
数据格式：每行 索引, 值, 类型(0=平均值, 1=通量), 组分, 本地索引
只取type=0的平均值（有限体积单元常数）

先从C++输出的mesh文件读取网格信息，保证绘图准确
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

# ============ 配置参数 ============
MESH_INFO_FILE = "meshdata/mesh_64x64.txt"    # C++输出的网格信息文件
INPUT_DIR = "solver_data"          # 输入时间步数据文件夹
OUTPUT_IMAGE_DIR = "python_plot_mesh_con_64"  # 输出图片文件夹
OUTPUT_GIF_DIR = "python_GIF"         # 输出GIF文件夹
OUTPUT_GIF_NAME = "MS_3_64_con.gif"      # GIF文件名
N_COMPONENTS = 3                       # 组分数
COMPONENT_NAMES = ["Component 0", "Component 1", "Component 2"]  # 组分名称
FIG_SIZE = (18, 6)                    # 图像尺寸 (三个并排)
DPI = 120                              # 图像分辨率
GIF_DURATION = 100                     # 每帧持续时间(毫秒)
CMAP = "rainbow"                       # 颜色映射
# =====================================


class MeshInfo:
    """存储从文件读取的网格信息"""
    def __init__(self):
        self.nx = 0
        self.ny = 0
        self.total_elements = 0
        self.elements = []  # 每个元素: {'id': id, 'n_nodes': n, 'vertices': [(x0,y0), ...], 'cx': cx, 'cy': cy, 'area': area}

    def read_from_file(self, filename):
        """从文件读取网格信息"""
        with open(filename, 'r') as f:
            # 第一行：nx ny total_elements
            line = f.readline()
            parts = line.strip().split()
            self.nx = int(parts[0])
            self.ny = int(parts[1])
            self.total_elements = int(parts[2])

            self.elements = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.strip().split()
                elem_id = int(parts[0])
                n_nodes = int(parts[1])
                vertices = []
                ptr = 2
                for i in range(n_nodes):
                    x = float(parts[ptr])
                    y = float(parts[ptr+1])
                    vertices.append((x, y))
                    ptr += 2
                cx = float(parts[ptr])
                cy = float(parts[ptr+1])
                area = float(parts[ptr+2])
                self.elements.append({
                    'id': elem_id,
                    'n_nodes': n_nodes,
                    'vertices': vertices,
                    'cx': cx,
                    'cy': cy,
                    'area': area
                })
        print(f"Read mesh: {self.total_elements} elements, {self.nx}x{self.ny}")
        return self


def read_solution_file(filename, total_elements):
    """
    读取解文件，返回所有三个组分的数组
    返回：solutions[elem_id] = [comp0_val, comp1_val, comp2_val]
    """
    solutions = np.zeros((total_elements, N_COMPONENTS))

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == '#':
                continue
            # 把逗号换成空格
            line = line.replace(',', ' ')
            parts = line.split()
            if len(parts) < 5:
                continue

            index = int(parts[0])
            value = float(parts[1])
            type_val = int(parts[2])
            comp = int(parts[3])
            local_idx = int(parts[4])

            # 只取平均值
            if type_val == 0 and 0 <= comp < N_COMPONENTS:
                if 0 <= local_idx < total_elements:
                    solutions[local_idx, comp] = value

    return solutions


def plot_single_step(mesh, solutions, output_filename, timestep):
    """
    绘制单个时间步，三个组分并排显示
    有限体积法每个单元是常数，逐个单元绘制多边形
    """
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE, dpi=DPI)

    # 找全局色标范围，使得三个图颜色标度一致
    vmin = np.min(solutions)
    vmax = np.max(solutions)

    # 获取颜色映射
    cmap = plt.get_cmap(CMAP)
    # 在每个子图绘制对应组分
    for comp, ax in enumerate(axes):
        # 逐个单元绘制多边形，每个单元填充常数
        for elem in mesh.elements:
            elem_id = elem['id']
            val = solutions[elem_id, comp]
            # 获取顶点坐标
            verts = np.array(elem['vertices'])
            poly = Polygon(verts, facecolor=cmap((val - vmin) / (vmax - vmin)),
                          edgecolor='none', antialiased=False)
            ax.add_patch(poly)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(COMPONENT_NAMES[comp], fontsize=14)
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.set_aspect('equal')

    # 添加颜色条 - 使用ScalerMappable来创建颜色条
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])

    # 为每个子图添加颜色条
    fig.subplots_adjust(wspace=0.3)
    for ax in axes:
        fig.colorbar(sm, ax=ax)

    fig.suptitle(f"Time step {timestep}", fontsize=16, y=1.02)
    plt.tight_layout()
    fig.savefig(output_filename, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def create_gif(image_files, gif_path):
    """
    从一系列图片创建GIF动图
    """
    images = []
    for filename in image_files:
        im = Image.open(filename)
        images.append(im)

    # 保存GIF
    if images:
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=GIF_DURATION,
            loop=0  # 无限循环
        )
    print(f"GIF saved to: {gif_path}")


def create_gif(image_files, gif_path):
    """
    从一系列图片创建GIF动图
    """
    images = []
    for filename in image_files:
        im = Image.open(filename)
        images.append(im)

    # 保存GIF
    if images:
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=GIF_DURATION,
            loop=0  # 无限循环
        )
    print(f"GIF saved to: {gif_path}")


def main():
    # 创建输出文件夹
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_GIF_DIR, exist_ok=True)

    # 第一步：读取网格信息
    print(f"Reading mesh info from {MESH_INFO_FILE}...")
    mesh = MeshInfo()
    if not os.path.exists(MESH_INFO_FILE):
        print(f"ERROR: Mesh file {MESH_INFO_FILE} not found!")
        print(f"Please generate it first with: ./dump_mesh_info input.msh meshdata")
        return
    mesh.read_from_file(MESH_INFO_FILE)

    # 获取所有数据文件并按时间步排序
    all_files = []
    for filename in os.listdir(INPUT_DIR):
        if filename.startswith("solution_step_"):
            # 提取时间步号：solution_step_10.txt -> 10
            try:
                step = int(filename.split('_')[2].split('.')[0])
                all_files.append((step, filename))
            except:
                continue

    # 按时间步排序
    all_files.sort()
    print(f"Found {len(all_files)} time step files")

    # 处理每个时间步
    image_files = []
    for step, filename in all_files:
        print(f"Processing time step {step}...")

        # 读取三个组分的解
        full_path = os.path.join(INPUT_DIR, filename)
        solutions = read_solution_file(full_path, mesh.total_elements)

        # 绘图（三个组分并排）
        output_filename = os.path.join(OUTPUT_IMAGE_DIR, f"step_{step:04d}.png")
        plot_single_step(mesh, solutions, output_filename, step)
        image_files.append(output_filename)

    # 创建GIF
    gif_path = os.path.join(OUTPUT_GIF_DIR, OUTPUT_GIF_NAME)
    create_gif(image_files, gif_path)

    print(f"\nAll done!")
    print(f"  - Images saved in: {OUTPUT_IMAGE_DIR}/")
    print(f"  - GIF saved in: {gif_path}")


if __name__ == "__main__":
    main()
