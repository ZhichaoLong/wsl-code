import meshio
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings("ignore")

# ---------------------- 全局样式配置 ----------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.linewidth": 1.2,          # 坐标轴宽度
    "xtick.color": "#2ca02c",       # X轴颜色（绿色）
    "ytick.color": "#d62728",       # Y轴颜色（红色）
    "xtick.labelcolor": "#2ca02c",  # X轴标签颜色
    "ytick.labelcolor": "#d62728",  # Y轴标签颜色
    "axes.labelcolor": "black",     # 坐标轴标题颜色
    "font.size": 10                 # 基础字体大小
})

# ---------------------- 核心解析函数 ----------------------
def parse_mesh_file(msh_file: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Parse mesh file (supports quad/triangle/polygon)
    Args:
        msh_file: Path to mesh file
    Returns:
        nodes: Node coordinates (N, 2)
        cells: Cell connectivity
        cell_type: Cell type ('quad', 'triangle', 'polygon')
    """
    if not os.path.exists(msh_file):
        raise FileNotFoundError(f"Mesh file not found: {msh_file}")
    
    mesh = meshio.read(msh_file, file_format="gmsh")
    nodes = mesh.points[:, :2]  # Only 2D coordinates
    
    # Get cell data
    cell_type = None
    cells = None
    possible_keys = ["quad", "gmsh:quad", "triangle", "gmsh:triangle", "polygon", "gmsh:polygon"]
    
    for key in possible_keys:
        if key in mesh.cells_dict:
            cells = mesh.cells_dict[key]
            cell_type = key.split(":")[-1] if ":" in key else key
            break
    
    if cells is None and len(mesh.cells) > 0:
        cell_block = mesh.cells[0]
        cells = cell_block.data
        cell_type = cell_block.type.split(":")[-1] if ":" in cell_block.type else cell_block.type
    
    if cells is None:
        raise ValueError(f"No 2D cells found in {msh_file}")
    
    return nodes, cells, cell_type

# ---------------------- 绘图函数 ----------------------
def plot_mesh(
    msh_file: str,
    output_dir: str = "plot_mesh",
    figsize: Tuple[float, float] = (8, 6),
    line_width: float = 1.0,
    line_color: str = "#1f77b4",  # Grid line color (blue)
    alpha: float = 0.8            # Grid line transparency
) -> None:
    """
    Plot single mesh and save as PNG (same name as msh file)
    Args:
        msh_file: Path to mesh file
        output_dir: Output directory for plots
        figsize: Figure size (width, height)
        line_width: Grid line width
        line_color: Grid line color
        alpha: Grid line transparency
    """
    try:
        # Parse mesh
        nodes, cells, cell_type = parse_mesh_file(msh_file)
        mesh_name = Path(msh_file).stem
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")
        
        # Plot grid cells
        for cell in cells:
            cell_nodes = nodes[cell]
            closed_cell = np.vstack([cell_nodes, cell_nodes[0]])  # Close the polygon
            ax.plot(closed_cell[:, 0], closed_cell[:, 1], 
                   color=line_color, linewidth=line_width, alpha=alpha, zorder=2)
        
        # Set axis range (add small padding)
        x_min, x_max = nodes[:, 0].min(), nodes[:, 0].max()
        y_min, y_max = nodes[:, 1].min(), nodes[:, 1].max()
        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.05
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        
        # Axis labels (English only)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(f"Mesh: {mesh_name} (Type: {cell_type})")
        
        # Save plot (same name as msh file)
        output_path = os.path.join(output_dir, f"{mesh_name}.png")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"✅ Plotted: {output_path}")
    
    except Exception as e:
        print(f"❌ Failed to plot {msh_file}: {str(e)}")

# ---------------------- 批量绘图函数 ----------------------
def plot_all_meshes(
    mesh_dir: str = "mesh_data",
    output_dir: str = "plot_mesh",
    figsize: Tuple[float, float] = (8, 6)
) -> None:
    """
    Auto detect all .msh files in mesh_dir and plot them
    Args:
        mesh_dir: Directory containing .msh files
        output_dir: Directory to save plots
        figsize: Figure size for each plot
    """
    # Check input directory
    if not os.path.exists(mesh_dir):
        print(f"❌ Mesh directory '{mesh_dir}' does not exist!")
        return
    
    # Get all .msh files
    msh_files = [f for f in os.listdir(mesh_dir) if f.lower().endswith(".msh")]
    if not msh_files:
        print(f"❌ No .msh files found in '{mesh_dir}'!")
        return
    
    # Plot each mesh
    print(f"📁 Found {len(msh_files)} mesh files in '{mesh_dir}'")
    print(f"📊 Saving plots to '{output_dir}'")
    print("-" * 60)
    
    for idx, msh_file in enumerate(sorted(msh_files), 1):
        msh_path = os.path.join(mesh_dir, msh_file)
        print(f"Processing [{idx}/{len(msh_files)}]: {msh_file}")
        plot_mesh(msh_path, output_dir=output_dir, figsize=figsize)

# ---------------------- 主执行入口 ----------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Mesh Visualization Tool")
    print("=" * 60)
    
    # Plot all meshes in mesh_data directory
    plot_all_meshes(
        mesh_dir="mesh_data",
        output_dir="plot_mesh",
        figsize=(8, 6)
    )
    
    print("=" * 60)
    print("🎉 Mesh plotting completed!")
    print(f"📂 All plots saved in: {os.path.abspath('plot_mesh')}")