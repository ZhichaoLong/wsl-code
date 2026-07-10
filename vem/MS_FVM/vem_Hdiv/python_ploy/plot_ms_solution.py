#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maxwell-Stefan方程数值解画图脚本
直接使用分片多项式画图，无需插值
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import os
import re
from pathlib import Path
import subprocess
import tempfile
import shutil
import sys
import time


class MultiIndex2D:
    """二维多重指标，按总次数排序"""
    def __init__(self, k):
        self.k = k
        self.mi = self._init_multi_index()

    def _init_multi_index(self):
        mi = []
        deg = 0
        while True:
            num_in_deg = deg + 1
            for offset in range(num_in_deg):
                m = deg - offset
                n = offset
                mi.append((m, n))
            if deg == self.k:
                break
            deg += 1
        return mi

    def __len__(self):
        return len(self.mi)

    def __getitem__(self, idx):
        return self.mi[idx]


class ScaledMonomialBasis:
    """缩放单项式基函数"""
    def __init__(self, k):
        self.k = k
        self.mi = MultiIndex2D(k)

    def eval(self, m, xD, hD, x, y):
        alpha = self.mi[m]
        xm = (x - xD[0]) / hD
        ym = (y - xD[1]) / hD
        return (xm ** alpha[0]) * (ym ** alpha[1])

    def dim(self):
        return len(self.mi)


class MeshReader:
    """网格文件读取器"""
    def __init__(self, filename):
        self.filename = filename
        self.nx = 0
        self.ny = 0
        self.n_elem = 0
        self.centroid_x = []
        self.centroid_y = []
        self.diameter = []
        self.element_node_coords = []
        self._read()

    def _read(self):
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        parts = lines[0].strip().split()
        self.nx = int(parts[0])
        self.ny = int(parts[1])
        self.n_elem = int(parts[2])

        for line in lines[1:self.n_elem+1]:
            parts = line.strip().split()
            elem_id = int(parts[0])
            n_nodes = int(parts[1])

            coords = []
            for i in range(n_nodes):
                x = float(parts[2 + 2*i])
                y = float(parts[3 + 2*i])
                coords.append([x, y])
            coords = np.array(coords)
            self.element_node_coords.append(coords)

            cx = np.mean(coords[:, 0])
            cy = np.mean(coords[:, 1])
            self.centroid_x.append(cx)
            self.centroid_y.append(cy)

            if n_nodes == 4:
                diag1 = np.linalg.norm(coords[2] - coords[0])
                diag2 = np.linalg.norm(coords[3] - coords[1])
                self.diameter.append(max(diag1, diag2))
            else:
                max_dist = 0
                for i in range(n_nodes):
                    for j in range(i+1):
                        dist = np.linalg.norm(coords[i] - coords[j])
                        max_dist = max(max_dist, dist)
                self.diameter.append(max_dist)

        self.centroid_x = np.array(self.centroid_x)
        self.centroid_y = np.array(self.centroid_y)
        self.diameter = np.array(self.diameter)


class SolutionReader:
    """读取解向量文件"""
    def __init__(self, filename):
        self.filename = filename
        self.data = self._read()

    def _read(self):
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        size = int(lines[0].strip())
        data = np.zeros(size)
        for i in range(size):
            data[i] = float(lines[i+1].strip())
        return data


def initial_condition(pde_index, comp, x, y):
    """
    初始条件函数，与C++代码保持一致

    pde_index: 算例编号
    comp: 组分索引 (0,1,2)
    x,y: 坐标
    """
    if pde_index == 0:
        # 间断初值
        if comp == 0:
            return 1.0 if (x >= 0.0 and x <= 0.5 and y >= 0.0 and y <= 0.5) else 0.0
        elif comp == 1:
            return 1.0 if (x >= 0.0 and x <= 0.5 and y > 0.5 and y <= 1.0) else 0.0
        elif comp == 2:
            u0 = initial_condition(0, 0, x, y)
            u1 = initial_condition(0, 1, x, y)
            return 1.0 - u0 - u1
    elif pde_index == 1:
        # 连续初值
        if comp == 0:
            return 0.4 - 0.1 * x - 0.1 * y
        elif comp == 1:
            return 0.3 + 0.05 * x + 0.15 * y
        elif comp == 2:
            u0 = initial_condition(1, 0, x, y)
            u1 = initial_condition(1, 1, x, y)
            return 1.0 - u0 - u1
    raise ValueError(f"未定义的算例 {pde_index} 或组分 {comp}")


class MSSolutionPlotter:
    """MS方程解画图器"""
    def __init__(self, mesh_filename, solution_dir, n_components=3, k=1, pde_index=0):
        self.mesh_filename = mesh_filename
        self.solution_dir = Path(solution_dir)
        self.n_components = n_components
        self.k = k
        self.pde_index = pde_index

        self.basis = ScaledMonomialBasis(k)
        self.dim_poly = self.basis.dim()

        self.mesh = MeshReader(mesh_filename)
        self.n_elem = self.mesh.n_elem

        # 计算自由度
        dim_pk = (k + 1) * (k + 2) // 2
        dim_grad_mk = dim_pk - 1
        dim_curl_mk = dim_pk - (k + 1)
        internal_dof = dim_grad_mk + dim_curl_mk

        total_edges = (self.mesh.nx + 1) * self.mesh.ny + self.mesh.nx * (self.mesh.ny + 1)
        self.dof_flux_single = total_edges * (k + 1) + self.n_elem * internal_dof
        self.dof_conc_single = dim_pk * self.n_elem
        self.total_flux_dof = self.dof_flux_single * n_components

        print(f"网格: {self.mesh.nx}x{self.mesh.ny} = {self.n_elem} 单元")
        print(f"单个组分: 通量自由度 {self.dof_flux_single}, 浓度自由度 {self.dof_conc_single}")
        print(f"总自由度: {self.total_flux_dof + self.dof_conc_single * n_components}")
        print(f"使用算例 {pde_index} 的初值函数")

        self.solution_files = sorted(self.solution_dir.glob("MS_*.txt"))
        self.times = []
        for f in self.solution_files:
            match = re.search(r"MS_([\d\.]+)\.txt", f.name)
            if match:
                self.times.append(float(match.group(1)))

        print(f"找到 {len(self.solution_files)} 个解文件")

    def extract_concentration_coeffs(self, solution_data, elem_idx, comp_idx):
        comp_conc_start = self.total_flux_dof + comp_idx * self.dof_conc_single
        elem_start = comp_conc_start + elem_idx * self.dim_poly
        return solution_data[elem_start:elem_start + self.dim_poly]

    def evaluate_polynomial(self, coeffs, xD, hD, X, Y):
        val = np.zeros_like(X)
        for m in range(self.dim_poly):
            basis_val = self.basis.eval(m, xD, hD, X, Y)
            val += coeffs[m] * basis_val
        return val

    def compute_concentration_field(self, solution_data, comp_idx, resolution=200):
        """计算整个域的浓度场"""
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        X, Y = np.meshgrid(x, y)
        U = np.zeros_like(X)

        for e in range(self.n_elem):
            coeffs = self.extract_concentration_coeffs(solution_data, e, comp_idx)
            xD = (self.mesh.centroid_x[e], self.mesh.centroid_y[e])
            hD = self.mesh.diameter[e]
            coords = self.mesh.element_node_coords[e]

            x_min = np.min(coords[:, 0])
            x_max = np.max(coords[:, 0])
            y_min = np.min(coords[:, 1])
            y_max = np.max(coords[:, 1])

            mask = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)
            X_elem = X[mask]
            Y_elem = Y[mask]
            U_elem = self.evaluate_polynomial(coeffs, xD, hD, X_elem, Y_elem)
            U[mask] = U_elem

        return X, Y, U

    def compute_initial_condition_field(self, resolution=200):
        """计算初值场（直接使用初值函数，不需要解向量）"""
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        X, Y = np.meshgrid(x, y)
        U_list = []
        for c in range(self.n_components):
            U = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    U[i, j] = initial_condition(self.pde_index, c, X[i, j], Y[i, j])
            U_list.append(U)
        return X, Y, U_list

    def get_initial_value_range(self):
        """获取初值的数值范围"""
        all_values = []
        for c in range(self.n_components):
            # 采样一些点
            for x in np.linspace(0, 1, 50):
                for y in np.linspace(0, 1, 50):
                    all_values.append(initial_condition(self.pde_index, c, x, y))
        return min(all_values), max(all_values)

    def _get_solution_value_range(self, sol_data, sample_points=10):
        """计算单个解向量的真实浓度范围（内部函数）
        对每个单元，在其实际物理范围内采样计算多项式值
        """
        xi = np.linspace(0, 1, sample_points)
        yi = np.linspace(0, 1, sample_points)
        XI, YI = np.meshgrid(xi, yi)
        XI_flat = XI.flatten()
        YI_flat = YI.flatten()

        val_min = float('inf')
        val_max = float('-inf')

        for comp in range(self.n_components):
            for e in range(self.n_elem):
                coeffs = self.extract_concentration_coeffs(sol_data, e, comp)
                xD = (self.mesh.centroid_x[e], self.mesh.centroid_y[e])
                hD = self.mesh.diameter[e]
                coords = self.mesh.element_node_coords[e]

                # 获取该单元的实际物理范围
                x_min = np.min(coords[:, 0])
                x_max = np.max(coords[:, 0])
                y_min = np.min(coords[:, 1])
                y_max = np.max(coords[:, 1])

                # 生成该单元物理范围内的采样点
                X_elem = x_min + XI_flat * (x_max - x_min)
                Y_elem = y_min + YI_flat * (y_max - y_min)

                # 计算该单元在采样点的实际值
                vals = self.evaluate_polynomial(coeffs, xD, hD, X_elem, Y_elem)
                val_min = min(val_min, np.min(vals))
                val_max = max(val_max, np.max(vals))

        return val_min, val_max

    def get_global_value_range(self, sample_points=10):
        """获取所有时间步所有组分的真实浓度范围
        通过在每个单元内采样多个点计算多项式的实际极值
        """
        global_min = float('inf')
        global_max = float('-inf')

        for sol_file in self.solution_files:
            sol_reader = SolutionReader(sol_file)
            sol_data = sol_reader.data
            vmin, vmax = self._get_solution_value_range(sol_data, sample_points)
            global_min = min(global_min, vmin)
            global_max = max(global_max, vmax)

        return global_min, global_max

    def create_combined_plot(self, time_idx, output_folder, vmin=None, vmax=None,
                           show_grid=True, resolution=200, cmap='rainbow', frame_offset=0):
        """创建单个时间步的组合图（1行3列）

        frame_offset: 文件名编号偏移（初值占了0000，解文件从0001开始）
        """
        sol_file = self.solution_files[time_idx]
        t = self.times[time_idx]
        sol_reader = SolutionReader(sol_file)
        sol_data = sol_reader.data

        # 如果没有指定范围，使用当前时间步的真实浓度范围（通过采样点计算）
        if vmin is None or vmax is None:
            vmin, vmax = self._get_solution_value_range(sol_data)
            print(f"  t={t:.4f} 浓度范围: [{vmin:.6f}, {vmax:.6f}]")

        # 创建图形
        fig, axes = plt.subplots(1, self.n_components, figsize=(18, 6))
        if self.n_components == 1:
            axes = [axes]

        fig.suptitle(f"Time = {t:.6f}", fontsize=16, y=0.98)

        # 计算每个组分的浓度场
        U_list = []
        for c in range(self.n_components):
            X, Y, U = self.compute_concentration_field(sol_data, c, resolution)
            U_list.append(U)

        # 绘制每个组分
        for i, (ax, U) in enumerate(zip(axes, U_list)):
            # 使用 imshow 绘制热力图
            im = ax.imshow(U, extent=[0, 1, 0, 1], origin='lower',
                          cmap=cmap, vmin=vmin, vmax=vmax,
                          aspect='equal')

            ax.set_title(f"Component {i+1}", fontsize=14, pad=15)
            ax.set_xlabel('X Coordinate', fontsize=12)
            if i == 0:
                ax.set_ylabel('Y Coordinate', fontsize=12)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')

            # 显示网格线
            if show_grid:
                NX = self.mesh.nx
                NY = self.mesh.ny
                x_ticks = np.linspace(0, 1, NX + 1)
                y_ticks = np.linspace(0, 1, NY + 1)
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                for x in x_ticks:
                    ax.axvline(x=x, color='gray', linewidth=0.5, alpha=0.3, linestyle='-')
                for y in y_ticks:
                    ax.axhline(y=y, color='gray', linewidth=0.5, alpha=0.3, linestyle='-')

        # 添加共享颜色条
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, aspect=30, pad=0.05)
        cbar.set_label('Solution Value', rotation=270, labelpad=20, fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.96], pad=0.2)

        # 保存图片 (编号从 0001 开始，因为初值是 0000)
        output_path = os.path.join(output_folder, f'pic_MS_{time_idx + frame_offset:04d}_combined.png')
        plt.savefig(output_path, dpi=300)
        plt.close(fig)

        return output_path

    def create_initial_condition_plot(self, output_folder, vmin=None, vmax=None,
                                    show_grid=True, resolution=200, cmap='rainbow'):
        """创建初值图"""
        # 如果没有指定范围，使用初值范围
        if vmin is None or vmax is None:
            vmin, vmax = self.get_initial_value_range()
            print(f"初值浓度范围: [{vmin:.6f}, {vmax:.6f}]")

        # 创建图形
        fig, axes = plt.subplots(1, self.n_components, figsize=(18, 6))
        if self.n_components == 1:
            axes = [axes]

        fig.suptitle(f"Time = 0.000000 (Initial Condition)", fontsize=16, y=0.98)

        # 计算每个组分的初值场
        X, Y, U_list = self.compute_initial_condition_field(resolution)

        # 绘制每个组分
        for i, (ax, U) in enumerate(zip(axes, U_list)):
            im = ax.imshow(U, extent=[0, 1, 0, 1], origin='lower',
                          cmap=cmap, vmin=vmin, vmax=vmax,
                          aspect='equal')

            ax.set_title(f"Component {i+1}", fontsize=14, pad=15)
            ax.set_xlabel('X Coordinate', fontsize=12)
            if i == 0:
                ax.set_ylabel('Y Coordinate', fontsize=12)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')

            # 显示网格线
            if show_grid:
                NX = self.mesh.nx
                NY = self.mesh.ny
                x_ticks = np.linspace(0, 1, NX + 1)
                y_ticks = np.linspace(0, 1, NY + 1)
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                for x in x_ticks:
                    ax.axvline(x=x, color='gray', linewidth=0.5, alpha=0.3, linestyle='-')
                for y in y_ticks:
                    ax.axhline(y=y, color='gray', linewidth=0.5, alpha=0.3, linestyle='-')

        # 添加共享颜色条
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, aspect=30, pad=0.05)
        cbar.set_label('Solution Value', rotation=270, labelpad=20, fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.96], pad=0.2)

        # 保存图片 (编号为 0000)
        output_path = os.path.join(output_folder, f'pic_MS_0000_combined.png')
        plt.savefig(output_path, dpi=300)
        plt.close(fig)

        return output_path

    def generate_all_plots(self, output_folder, skip=1, show_grid=True, resolution=200,
                          color_mode='global', vmin=None, vmax=None, cmap='rainbow',
                          include_initial=True):
        """生成所有时间步的组合图

        color_mode:
            - 'global': 所有时间步用统一的全局范围（便于比较绝对值）
            - 'local': 每个时间步用自己的范围（变化更明显，敏感度最高）
            - 'manual': 使用指定的 vmin/vmax
        include_initial: 是否包含初值作为第一张图
        """
        os.makedirs(output_folder, exist_ok=True)

        if color_mode == 'global':
            # 获取全局数值范围（包括初值）
            ic_vmin, ic_vmax = self.get_initial_value_range()
            sol_vmin, sol_vmax = self.get_global_value_range()
            global_vmin = min(ic_vmin, sol_vmin)
            global_vmax = max(ic_vmax, sol_vmax)
            print(f"全局浓度范围(含初值): [{global_vmin:.6f}, {global_vmax:.6f}]")
            use_vmin, use_vmax = global_vmin, global_vmax
        elif color_mode == 'manual':
            use_vmin, use_vmax = vmin, vmax
            print(f"手动指定浓度范围: [{use_vmin:.6f}, {use_vmax:.6f}]")
        else:  # local
            use_vmin, use_vmax = None, None
            print("使用每个时间步独立的浓度范围（颜色敏感度最高）")

        success_count = 0
        total = len(range(0, len(self.solution_files), skip))
        if include_initial:
            total += 1

        if include_initial:
            print(f"处理初值 (0/{total})", end="", flush=True)
            self.create_initial_condition_plot(output_folder, use_vmin, use_vmax,
                                            show_grid, resolution, cmap=cmap)
            success_count += 1
            print(f"\r处理初值 ✓ (1/{total})", end="", flush=True)

        offset = 1 if include_initial else 0

        for i in range(0, len(self.solution_files), skip):
            idx = i // skip + offset
            print(f"\r处理时间步 {idx + 1}/{total}", end="", flush=True)
            # 注意：文件名编号要与初值的 0000 连续
            self.create_combined_plot(i, output_folder, use_vmin, use_vmax,
                                    show_grid, resolution, cmap=cmap, frame_offset=offset)
            success_count += 1

        print(f"\n✓ 成功生成 {success_count} 张图片")
        return success_count

    def generate_gif(self, input_folder, output_path, fps=10, target_width=1200):
        """使用FFmpeg生成GIF"""
        img_paths = sorted(Path(input_folder).glob("pic_MS_*_combined.png"))

        if not img_paths:
            print("❌ 未找到任何图片文件")
            return None

        img_count = len(img_paths)
        print(f"✅ 找到 {img_count} 张图片，开始生成GIF...")

        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ FFmpeg未安装，请先安装FFmpeg")
            return None

        temp_dir = tempfile.mkdtemp(prefix="gif_temp_")

        try:
            # 重命名为连续编号
            for i, img_path in enumerate(img_paths):
                temp_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                shutil.copy2(img_path, temp_path)

            gif_path = output_path
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                '-vf', f"fps={fps},scale={target_width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=full[p];[s1][p]paletteuse=dither=floyd_steinberg",
                '-loop', '0',
                '-compression_level', '6',
                gif_path
            ]

            print(f"⏳ 正在生成GIF（{img_count}帧，{fps} FPS）...")
            start_time = time.time()

            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            process.wait()
            end_time = time.time()

            if process.returncode != 0:
                print(f"❌ FFmpeg执行失败")
                return None

            file_size = os.path.getsize(gif_path) / (1024 * 1024)

            print(f"\n{'='*50}")
            print(f"🎉 GIF生成成功!")
            print(f"📁 文件: {gif_path}")
            print(f"📊 大小: {file_size:.2f} MB")
            print(f"🎞️  帧数: {img_count}")
            print(f"⏱️  时长: {img_count/fps:.1f}秒")
            print(f"⚡ 帧率: {fps} FPS")
            print(f"⏰ 生成耗时: {end_time - start_time:.1f}秒")
            print(f"{'='*50}")

            return gif_path

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def generate_mp4(self, input_folder, output_path, fps=20, target_width=None, crf=23):
        """使用FFmpeg生成MP4视频（推荐用于大量帧）

        crf: 质量参数，18-28之间，越小质量越好
        """
        img_paths = sorted(Path(input_folder).glob("pic_MS_*_combined.png"))

        if not img_paths:
            print("❌ 未找到任何图片文件")
            return None

        img_count = len(img_paths)
        print(f"✅ 找到 {img_count} 张图片，开始生成MP4...")

        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ FFmpeg未安装，请先安装FFmpeg")
            return None

        temp_dir = tempfile.mkdtemp(prefix="mp4_temp_")

        try:
            # 重命名为连续编号
            for i, img_path in enumerate(img_paths):
                temp_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                shutil.copy2(img_path, temp_path)

            mp4_path = output_path

            # 构建FFmpeg命令
            vf_filters = []
            if target_width is not None:
                vf_filters.append(f"scale={target_width}:-2:flags=lanczos")
            vf_str = ','.join(vf_filters) if vf_filters else None

            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.png')
            ]

            if vf_str:
                ffmpeg_cmd.extend(['-vf', vf_str])

            ffmpeg_cmd.extend([
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', str(crf),
                '-movflags', '+faststart',
                mp4_path
            ])

            print(f"⏳ 正在生成MP4（{img_count}帧，{fps} FPS，CRF={crf}）...")
            start_time = time.time()

            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            process.wait()
            end_time = time.time()

            if process.returncode != 0:
                print(f"❌ FFmpeg执行失败")
                print(f"错误信息:\n{process.stderr}")
                return None

            file_size = os.path.getsize(mp4_path) / (1024 * 1024)

            print(f"\n{'='*50}")
            print(f"🎉 MP4生成成功!")
            print(f"📁 文件: {mp4_path}")
            print(f"📊 大小: {file_size:.2f} MB")
            print(f"🎞️  帧数: {img_count}")
            print(f"⏱️  时长: {img_count/fps:.1f}秒")
            print(f"⚡ 帧率: {fps} FPS")
            print(f"📺 CRF质量: {crf}")
            print(f"⏰ 生成耗时: {end_time - start_time:.1f}秒")
            print(f"{'='*50}")

            return mp4_path

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


def generate_video_only(output_folder, output_mp4=None, output_gif=None, fps=20, mp4_quality=23, mp4_width=None):
    """仅从已有图片生成视频，不依赖 plotter"""

    img_paths = sorted(Path(output_folder).glob("pic_MS_*_combined.png"))
    if not img_paths:
        print(f"❌ 错误：在 {output_folder} 中没有找到图片！")
        return
    print(f"✅ 找到 {len(img_paths)} 张图片")

    # 生成 MP4
    if output_mp4:
        print("\n开始生成MP4...")
        temp_dir = tempfile.mkdtemp(prefix="mp4_temp_")
        try:
            for i, img_path in enumerate(img_paths):
                temp_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                shutil.copy2(img_path, temp_path)

            vf_filters = []
            if mp4_width is not None:
                vf_filters.append(f"scale={mp4_width}:-2:flags=lanczos")
            vf_str = ','.join(vf_filters) if vf_filters else None

            ffmpeg_cmd = ['ffmpeg', '-y', '-framerate', str(fps),
                         '-i', os.path.join(temp_dir, 'frame_%06d.png')]
            if vf_str:
                ffmpeg_cmd.extend(['-vf', vf_str])
            ffmpeg_cmd.extend(['-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                              '-crf', str(mp4_quality), '-movflags', '+faststart',
                              str(output_mp4)])

            print(f"⏳ 正在生成MP4（{len(img_paths)}帧，{fps} FPS）...")
            start_time = time.time()
            process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            end_time = time.time()

            if process.returncode == 0:
                file_size = os.path.getsize(output_mp4) / (1024 * 1024)
                print(f"\n{'='*50}")
                print(f"🎉 MP4生成成功!")
                print(f"📁 文件: {output_mp4}")
                print(f"📊 大小: {file_size:.2f} MB")
                print(f"⏰ 耗时: {end_time - start_time:.1f}秒")
                print(f"{'='*50}")
            else:
                print(f"❌ FFmpeg执行失败:\n{process.stderr}")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    # 生成 GIF
    if output_gif:
        print("\n开始生成GIF...")
        temp_dir = tempfile.mkdtemp(prefix="gif_temp_")
        try:
            for i, img_path in enumerate(img_paths):
                temp_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                shutil.copy2(img_path, temp_path)

            ffmpeg_cmd = [
                'ffmpeg', '-y', '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                '-vf', f"fps={fps},scale=1200:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                '-loop', '0', str(output_gif)
            ]

            print(f"⏳ 正在生成GIF（{len(img_paths)}帧，{fps} FPS）...")
            start_time = time.time()
            process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            end_time = time.time()

            if process.returncode == 0:
                file_size = os.path.getsize(output_gif) / (1024 * 1024)
                print(f"\n{'='*50}")
                print(f"🎉 GIF生成成功!")
                print(f"📊 大小: {file_size:.2f} MB")
                print(f"⏰ 耗时: {end_time - start_time:.1f}秒")
                print(f"{'='*50}")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='画Maxwell-Stefan方程数值解')
    parser.add_argument('--grid-size', type=int, help='网格大小（如4表示4x4），如果不指定则从解目录自动识别')
    parser.add_argument('--solution-dir', type=str, help='解目录路径（优先使用此参数）')
    parser.add_argument('--k', type=int, default=1, help='多项式次数')
    parser.add_argument('--n-comp', type=int, default=3, help='组分数')
    parser.add_argument('--output-gif', type=str, default=None, help='输出GIF文件名（留空则不生成）')
    parser.add_argument('--output-mp4', type=str, default='ms_solution_disc_newstable.mp4', help='输出MP4文件名（推荐）')
    parser.add_argument('--output-folder', type=str, default='frames_disc_bigstable', help='帧图片文件夹')
    parser.add_argument('--fps', type=int, default=20, help='动画帧率')
    parser.add_argument('--mp4-quality', type=int, default=23,
                       help='MP4质量参数CRF (18-28,越小质量越好)')
    parser.add_argument('--mp4-width', type=int, default=None,
                       help='MP4输出宽度（如1920，留空则保持原分辨率）')
    parser.add_argument('--skip', type=int, default=1, help='每隔多少帧取一次')
    parser.add_argument('--resolution', type=int, default=200, help='画图分辨率')
    parser.add_argument('--no-grid', action='store_true', help='不显示网格线')
    # 颜色敏感度相关参数
    parser.add_argument('--color-mode', type=str, default='global',
                       choices=['global', 'local', 'manual'],
                       help='颜色范围模式: global(全局统一), local(每个时间步独立,敏感度最高), manual(手动指定)')
    parser.add_argument('--vmin', type=float, default=None, help='手动指定颜色最小值 (--color-mode=manual 时有效)')
    parser.add_argument('--vmax', type=float, default=None, help='手动指定颜色最大值 (--color-mode=manual 时有效)')
    parser.add_argument('--cmap', type=str, default='rainbow',
                       choices=['rainbow', 'viridis', 'plasma', 'inferno', 'jet', 'coolwarm', 'RdBu_r'],
                       help='颜色映射表: plasma/viridis 对小变化更敏感')
    # 初值相关参数
    parser.add_argument('--pde-index', type=int, default=0, choices=[0, 1],
                       help='算例编号: 0=间断初值, 1=连续初值')
    parser.add_argument('--no-initial', action='store_true',
                       help='不生成初值图（默认生成）')
    parser.add_argument('--only-video', action='store_true',
                       help='跳过图片生成，直接从已有图片生成MP4/GIF')

    args = parser.parse_args()

    base_dir = Path("/20184652/lzc/MScode/vem/MS_FVM/vem_Hdiv")
    output_folder = Path(args.output_folder)
    output_gif = Path(args.output_gif) if args.output_gif else None
    output_mp4 = Path(args.output_mp4) if args.output_mp4 else None

    # 如果是只生成视频模式
    if args.only_video:
        print("="*50)
        print("模式：仅生成视频（跳过图片生成）")
        print("="*50)
        print(f"图片文件夹: {output_folder}")
        generate_video_only(
            output_folder,
            output_mp4=output_mp4,
            output_gif=output_gif,
            fps=args.fps,
            mp4_quality=args.mp4_quality,
            mp4_width=args.mp4_width
        )
        return

    # 正常模式：先生成图片再生成视频
    # 确定解目录和网格大小
    if args.solution_dir and args.grid_size:
        solution_dir = Path(args.solution_dir)
        grid_size = args.grid_size
        print(f"使用指定的解目录: {solution_dir.name}")
        print(f"使用指定的网格大小: {grid_size}x{grid_size}")
    elif args.solution_dir:
        solution_dir = Path(args.solution_dir)
        # 从目录名识别网格大小
        dir_name = solution_dir.name
        match = re.search(r'solution_poly_disc_(\d+)', dir_name)
        if match:
            n_elem = int(match.group(1))
            grid_size = int(np.sqrt(n_elem))
            print(f"从解目录自动识别: {grid_size}x{grid_size} 网格")
        else:
            raise ValueError(f"无法从目录名 {dir_name} 识别网格大小，请手动指定 --grid-size")
    elif args.grid_size:
        grid_size = args.grid_size
        solution_dir = base_dir / f"data/solution_poly_disc_{grid_size * grid_size}"
    else:
        # 尝试自动查找最新的解目录
        data_dir = base_dir / "data"
        sol_dirs = sorted(data_dir.glob("solution_poly_disc_*"))
        if sol_dirs:
            solution_dir = sol_dirs[-1]
            dir_name = solution_dir.name
            match = re.search(r'solution_poly_disc_(\d+)', dir_name)
            if match:
                n_elem = int(match.group(1))
                grid_size = int(np.sqrt(n_elem))
                print(f"自动使用最新解目录: {solution_dir.name} ({grid_size}x{grid_size})")
            else:
                raise ValueError("无法识别网格大小，请手动指定 --grid-size 或 --solution-dir")
        else:
            raise ValueError("未找到解目录，请手动指定 --grid-size 或 --solution-dir")

    mesh_filename = base_dir.parent / "meshdata" / f"mesh_{grid_size}x{grid_size}.txt"

    print(f"网格文件: {mesh_filename}")
    print(f"解目录: {solution_dir}")
    print(f"颜色模式: {args.color_mode}")
    print(f"颜色映射: {args.cmap}")
    print(f"算例编号: {args.pde_index}")
    print(f"生成初值图: {'否' if args.no_initial else '是'}")

    plotter = MSSolutionPlotter(
        str(mesh_filename),
        str(solution_dir),
        n_components=args.n_comp,
        k=args.k,
        pde_index=args.pde_index
    )

    if len(plotter.solution_files) == 0:
        print("错误：没有找到解文件！")
        return

    # 生成所有帧
    print("\n开始生成帧图片...")
    success = plotter.generate_all_plots(
        str(output_folder),
        skip=args.skip,
        show_grid=not args.no_grid,
        resolution=args.resolution,
        color_mode=args.color_mode,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        include_initial=not args.no_initial
    )

    if success == 0:
        print("错误：没有生成任何图片！")
        return

    # 生成动画
    # 生成 MP4（推荐，默认）
    if output_mp4:
        print("\n开始生成MP4...")
        plotter.generate_mp4(
            str(output_folder),
            str(output_mp4),
            fps=args.fps,
            target_width=args.mp4_width,
            crf=args.mp4_quality
        )

    # 生成 GIF（可选）
    if output_gif:
        print("\n开始生成GIF...")
        plotter.generate_gif(
            str(output_folder),
            str(output_gif),
            fps=args.fps
        )


if __name__ == "__main__":
    main()
