import gmsh
import numpy as np
import os
import shutil
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------- 核心算法实现 ----------------------
class LloydRelaxation:
    """Lloyd算法实现CVT（Centroidal Voronoi Tessellation）"""
    @staticmethod
    def lloyd_relaxation(points: np.ndarray, 
                        bounds: Tuple[float, float, float, float], 
                        iterations: int = 100) -> np.ndarray:
        """
        Lloyd迭代优化生成CVT
        :param points: 初始随机点集 (N, 2)
        :param bounds: 边界 [xmin, xmax, ymin, ymax]
        :param iterations: 迭代次数
        :return: 优化后的CVT点集
        """
        xmin, xmax, ymin, ymax = bounds
        relaxed_points = points.copy()
        
        for _ in range(iterations):
            # 1. 构建Voronoi图
            vor = Voronoi(relaxed_points)
            
            # 2. 计算每个Voronoi单元的质心
            new_points = []
            for i in range(len(relaxed_points)):
                # 获取当前点的Voronoi单元顶点
                region = vor.regions[vor.point_region[i]]
                if -1 in region or len(region) == 0:
                    new_points.append(relaxed_points[i])
                    continue
                
                # 获取顶点坐标并裁剪到边界内
                vertices = np.array([vor.vertices[r] for r in region])
                vertices[:, 0] = np.clip(vertices[:, 0], xmin, xmax)
                vertices[:, 1] = np.clip(vertices[:, 1], ymin, ymax)
                
                # 计算多边形质心
                if len(vertices) < 3:
                    centroid = relaxed_points[i]
                else:
                    centroid = LloydRelaxation._polygon_centroid(vertices)
                
                # 限制质心在边界内
                centroid = np.clip(centroid, [xmin, ymin], [xmax, ymax])
                new_points.append(centroid)
            
            relaxed_points = np.array(new_points)
        
        return relaxed_points
    
    @staticmethod
    def _polygon_centroid(vertices: np.ndarray) -> np.ndarray:
        """计算多边形质心（顶点按顺序排列）"""
        x = vertices[:, 0]
        y = vertices[:, 1]
        n = len(vertices)
        
        A = 0.5 * np.abs(np.sum(x[:-1]*y[1:] - x[1:]*y[:-1]))
        if A < 1e-10:
            return np.mean(vertices, axis=0)
        
        cx = (1/(6*A)) * np.sum((x[:-1] + x[1:]) * (x[:-1]*y[1:] - x[1:]*y[:-1]))
        cy = (1/(6*A)) * np.sum((y[:-1] + y[1:]) * (x[:-1]*y[1:] - x[1:]*y[:-1]))
        return np.array([cx, cy])

# ---------------------- Gmsh网格生成核心函数 ----------------------
def initialize_gmsh(model_name: str) -> None:
    """初始化Gmsh并设置基础参数"""
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add(model_name)

def finalize_gmsh(output_file: str) -> None:
    """完成网格生成并保存文件"""
    try:
        gmsh.model.geo.synchronize()
        start_time = time.time()
        gmsh.model.mesh.generate(2)
        gen_time = time.time() - start_time
        gmsh.write(output_file)
        print(f"✅ 网格已保存: {os.path.basename(output_file)} (耗时: {gen_time:.2f}秒)")
    except Exception as e:
        print(f"❌ 网格生成失败: {e}")
        raise
    finally:
        gmsh.finalize()

def generate_triangle_mesh(
    nx: int,          # nx: x方向单元数
    ny: int,          # ny: y方向单元数
    domain: Tuple[float, float, float, float] = (0, 1, 0, 1),
    output_dir: str = "mesh_data"
) -> str:
    """生成均匀三角形网格（nx/ny代表单元数）"""
    x_min, x_max, y_min, y_max = domain
    output_file = f"triangle_{nx}x{ny}.msh"
    output_path = os.path.join(output_dir, output_file)
    Path(output_dir).mkdir(exist_ok=True)

    initialize_gmsh(f"TriangleMesh_{nx}x{ny}")
    
    # 核心修正：步长 = 总长度 / 单元数
    dx = (x_max - x_min) / nx    # x方向单元边长
    dy = (y_max - y_min) / ny    # y方向单元边长
    cell_size = min(dx, dy) * 0.8
    
    # 节点数 = 单元数 + 1
    num_nodes_x = nx + 1
    num_nodes_y = ny + 1

    # 创建结构化点矩阵
    point_tags = np.arange(1, num_nodes_x*num_nodes_y + 1).reshape(num_nodes_x, num_nodes_y)
    tag = 1
    for j in range(num_nodes_y):
        y = y_min + j * dy
        for i in range(num_nodes_x):
            x = x_min + i * dx
            gmsh.model.geo.addPoint(x, y, 0.0, cell_size, tag)
            tag += 1

    # 创建线（先水平后垂直）
    line_tags_h = np.zeros((nx, num_nodes_y), dtype=int)  # nx条水平线
    current_line = 1
    for j in range(num_nodes_y):
        for i in range(nx):  # 单元数nx → 水平线数nx
            line_tags_h[i, j] = current_line
            gmsh.model.geo.addLine(point_tags[i, j], point_tags[i+1, j], current_line)
            current_line += 1
    
    line_tags_v = np.zeros((num_nodes_x, ny), dtype=int)  # ny条垂直线
    for i in range(num_nodes_x):
        for j in range(ny):  # 单元数ny → 垂直线数ny
            line_tags_v[i, j] = current_line
            gmsh.model.geo.addLine(point_tags[i, j], point_tags[i, j+1], current_line)
            current_line += 1

    # 创建四边形环，再分割为三角形
    for i in range(nx):
        for j in range(ny):
            loop_lines = [
                line_tags_h[i, j], line_tags_v[i+1, j],
                -line_tags_h[i, j+1], -line_tags_v[i, j]
            ]
            loop_tag = i * ny + j + 1
            gmsh.model.geo.addCurveLoop(loop_lines, loop_tag)
            gmsh.model.geo.addPlaneSurface([loop_tag], loop_tag)

    # 强制生成三角形网格
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)

    finalize_gmsh(output_path)
    return output_path

def generate_orthogonal_mesh(
    nx: int,          # nx: x方向单元数
    ny: int,          # ny: y方向单元数
    domain: Tuple[float, float, float, float] = (0, 1, 0, 1),
    output_dir: str = "mesh_data"
) -> str:
    """生成严格正交四边形网格（矩形，nx/ny代表单元数）"""
    x_min, x_max, y_min, y_max = domain
    output_file = f"orthogonal_{nx}x{ny}.msh"
    output_path = os.path.join(output_dir, output_file)
    Path(output_dir).mkdir(exist_ok=True)

    initialize_gmsh(f"OrthogonalMesh_{nx}x{ny}")
    
    # 核心修正：步长 = 总长度 / 单元数
    dx = (x_max - x_min) / nx    # x方向单元边长
    dy = (y_max - y_min) / ny    # y方向单元边长
    cell_size = min(dx, dy) * 0.8
    
    # 节点数 = 单元数 + 1
    num_nodes_x = nx + 1
    num_nodes_y = ny + 1

    # 严格结构化点
    point_tags = np.arange(1, num_nodes_x*num_nodes_y + 1).reshape(num_nodes_x, num_nodes_y)
    tag = 1
    for j in range(num_nodes_y):
        y = y_min + j * dy
        for i in range(num_nodes_x):
            x = x_min + i * dx
            gmsh.model.geo.addPoint(x, y, 0.0, cell_size, tag)
            tag += 1

    # 创建线
    line_tags_h = np.zeros((nx, num_nodes_y), dtype=int)  # nx条水平线
    current_line = 1
    for j in range(num_nodes_y):
        for i in range(nx):  # 单元数nx → 水平线数nx
            line_tags_h[i, j] = current_line
            gmsh.model.geo.addLine(point_tags[i, j], point_tags[i+1, j], current_line)
            current_line += 1
    
    line_tags_v = np.zeros((num_nodes_x, ny), dtype=int)  # ny条垂直线
    for i in range(num_nodes_x):
        for j in range(ny):  # 单元数ny → 垂直线数ny
            line_tags_v[i, j] = current_line
            gmsh.model.geo.addLine(point_tags[i, j], point_tags[i, j+1], current_line)
            current_line += 1

    # 创建正交四边形面
    for i in range(nx):
        for j in range(ny):
            loop_lines = [
                line_tags_h[i, j], line_tags_v[i+1, j],
                -line_tags_h[i, j+1], -line_tags_v[i, j]
            ]
            loop_tag = i * ny + j + 1
            gmsh.model.geo.addCurveLoop(loop_lines, loop_tag)
            gmsh.model.geo.addPlaneSurface([loop_tag], loop_tag)

    # 强制四边形网格
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.Optimize", 0)

    finalize_gmsh(output_path)
    return output_path

def generate_square_mesh(
    n: int,           # n: 正方形单元数（n×n个单元）
    domain: Tuple[float, float, float, float] = (0, 1, 0, 1),
    output_dir: str = "mesh_data"
) -> str:
    """
    生成严格正方形网格（结构化四边形网格，无扰动，等长宽比）
    n代表单元数（n×n个单元 → (n+1)×(n+1)个节点）
    """
    x_min, x_max, y_min, y_max = domain
    
    # 确保生成正方形单元（nx=ny=n）
    output_file = f"square_{n}x{n}.msh"
    output_path = os.path.join(output_dir, output_file)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"📌 开始生成 {n}x{n} 正方形网格（{n}×{n}单元）...")
    print(f"   输出文件: {output_file}")
    
    try:
        initialize_gmsh(f"SquareMesh_{n}x{n}")
        
        # 核心修正：步长 = 总长度 / 单元数（保证n×n个单元）
        dx = (x_max - x_min) / n    # 单元边长
        dy = (y_max - y_min) / n    # 单元边长
        # 强制正方形：取最小步长，保证单元为正方形
        cell_size = min(dx, dy) * 0.8
        square_step = min(dx, dy)
        
        print(f"   单元边长: {square_step:.6f}")
        print(f"   实际边界: x=[{x_min:.3f}, {x_min+square_step*n:.3f}], y=[{y_min:.3f}, {y_min+square_step*n:.3f}]")
        
        # 节点数 = 单元数 + 1
        num_nodes = n + 1
        
        # 1. 生成结构化网格点（严格正方形）
        x_grid, y_grid = np.meshgrid(
            np.linspace(x_min, x_min + square_step * n, num_nodes),
            np.linspace(y_min, y_min + square_step * n, num_nodes),
            indexing='ij'
        )
        points = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)
        
        # 2. 创建Gmsh点（无扰动）
        point_tags = np.arange(1, num_nodes*num_nodes + 1).reshape(num_nodes, num_nodes)
        tag = 1
        for i in range(num_nodes):
            for j in range(num_nodes):
                idx = i * num_nodes + j
                x, y = points[idx]
                gmsh.model.geo.addPoint(x, y, 0.0, cell_size, tag)
                tag += 1
        
        # 3. 创建水平线（n条，对应n个单元）
        line_tags_h = np.zeros((n, num_nodes), dtype=int)
        current_line = 1
        for j in range(num_nodes):
            for i in range(n):  # 单元数n → 水平线数n
                line_tags_h[i, j] = current_line
                gmsh.model.geo.addLine(point_tags[i, j], point_tags[i+1, j], current_line)
                current_line += 1
        
        # 4. 创建垂直线（n条，对应n个单元）
        line_tags_v = np.zeros((num_nodes, n), dtype=int)
        for i in range(num_nodes):
            for j in range(n):  # 单元数n → 垂直线数n
                line_tags_v[i, j] = current_line
                gmsh.model.geo.addLine(point_tags[i, j], point_tags[i, j+1], current_line)
                current_line += 1
        
        # 5. 创建四边形面（n×n个单元）
        surface_tags = []
        for i in range(n):
            for j in range(n):
                loop_lines = [
                    line_tags_h[i, j], line_tags_v[i+1, j],
                    -line_tags_h[i, j+1], -line_tags_v[i, j]
                ]
                loop_tag = i * n + j + 1
                gmsh.model.geo.addCurveLoop(loop_lines, loop_tag)
                surface_tag = gmsh.model.geo.addPlaneSurface([loop_tag], loop_tag)
                surface_tags.append(surface_tag)
        
        # 6. 网格参数（与正交网格相同）
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.RecombineAll", 1)          # 强制四边形
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2) # 最优重组算法
        gmsh.option.setNumber("Mesh.ElementOrder", 1)          # 线性单元
        gmsh.option.setNumber("Mesh.Optimize", 0)              # 禁用优化以保持正方形
        gmsh.option.setNumber("Mesh.Smoothing", 0)             # 禁用平滑以保持正方形
        gmsh.option.setNumber("Mesh.Algorithm", 8)             # Frontal-Delaunay适合四边形
        
        finalize_gmsh(output_path)
        
        # 验证文件是否成功生成
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ 正方形网格已生成: {output_file} ({file_size:,} 字节)")
            
            # 可选：验证网格的单元形状
            try:
                import meshio
                mesh = meshio.read(output_path)
                if len(mesh.cells) > 0:
                    cell_type = mesh.cells[0].type
                    cells = mesh.cells[0].data
                    nodes = mesh.points[:, :2]
                    
                    print(f"   网格统计: {len(nodes)} 节点, {len(cells)} 单元")
                    print(f"   单元类型: {cell_type}")
                    print(f"   预期: {(n+1)*(n+1)} 节点, {n*n} 单元")
                    
                    # 计算单元形状统计（可选）
                    if len(cells) > 0 and cell_type == "quad":
                        # 简单验证：检查前几个单元
                        for i in range(min(3, len(cells))):
                            cell_nodes = nodes[cells[i]]
                            # 计算边长
                            edges = [
                                np.linalg.norm(cell_nodes[1] - cell_nodes[0]),
                                np.linalg.norm(cell_nodes[2] - cell_nodes[1]),
                                np.linalg.norm(cell_nodes[3] - cell_nodes[2]),
                                np.linalg.norm(cell_nodes[0] - cell_nodes[3])
                            ]
                            avg_edge = np.mean(edges)
                            std_edge = np.std(edges)
                            if i == 0:
                                print(f"   单元{0}边长统计: 平均={avg_edge:.4f}, 标准差={std_edge:.4f}")
            except Exception as e:
                print(f"   网格验证警告: {e}")
        else:
            print(f"❌ 文件未生成: {output_path}")
            
        return output_path
        
    except Exception as e:
        print(f"❌ 正方形网格生成失败: {e}")
        import traceback
        traceback.print_exc()
        try:
            gmsh.finalize()
        except:
            pass
        raise

def generate_small_perturb_quad_mesh(
    nx: int,          # nx: x方向单元数
    ny: int,          # ny: y方向单元数
    domain: Tuple[float, float, float, float] = (0, 1, 0, 1),
    output_dir: str = "mesh_data",
    perturbation_factor: float = 0.25,
    seed: int = 42
) -> str:
    """
    正确的小扰动四边形网格：对结构化网格节点进行随机偏移
    nx/ny代表单元数（而非节点数）
    """
    x_min, x_max, y_min, y_max = domain
    output_file = f"small_perturb_quad_{nx}x{ny}.msh"
    output_path = os.path.join(output_dir, output_file)
    Path(output_dir).mkdir(exist_ok=True)

    initialize_gmsh(f"SmallPerturbQuadMesh_{nx}x{ny}")
    
    # 核心修正：步长 = 总长度 / 单元数
    dx = (x_max - x_min) / nx    # x方向单元边长
    dy = (y_max - y_min) / ny    # y方向单元边长
    cell_size = min(dx, dy) * 0.8
    
    # 节点数 = 单元数 + 1
    num_nodes_x = nx + 1
    num_nodes_y = ny + 1

    # 1. 生成结构化网格点
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, num_nodes_x),
        np.linspace(y_min, y_max, num_nodes_y),
        indexing='ij'
    )
    points = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)
    
    # 2. 对内部节点添加随机扰动（边界节点保持不变）
    np.random.seed(seed)
    perturb = np.zeros_like(points)
    
    # 标记内部节点（非边界）
    is_interior = (
        (points[:, 0] > x_min + 1e-6) & (points[:, 0] < x_max - 1e-6) &
        (points[:, 1] > y_min + 1e-6) & (points[:, 1] < y_max - 1e-6)
    )
    
    # 生成扰动（基于网格步长的比例）
    max_perturb_x = dx * perturbation_factor
    max_perturb_y = dy * perturbation_factor
    perturb[is_interior, 0] = np.random.uniform(-max_perturb_x, max_perturb_x, np.sum(is_interior))
    perturb[is_interior, 1] = np.random.uniform(-max_perturb_y, max_perturb_y, np.sum(is_interior))
    
    # 应用扰动并限制在边界内
    perturbed_points = points + perturb
    perturbed_points[:, 0] = np.clip(perturbed_points[:, 0], x_min, x_max)
    perturbed_points[:, 1] = np.clip(perturbed_points[:, 1], y_min, y_max)

    # 3. 创建Gmsh点
    point_tags = np.arange(1, num_nodes_x*num_nodes_y + 1).reshape(num_nodes_x, num_nodes_y)
    tag = 1
    for j in range(num_nodes_y):
        for i in range(num_nodes_x):
            idx = i * num_nodes_y + j
            x, y = perturbed_points[idx]
            gmsh.model.geo.addPoint(x, y, 0.0, cell_size, tag)
            tag += 1

    # 4. 创建线和四边形面
    line_tags_h = np.zeros((nx, num_nodes_y), dtype=int)  # nx条水平线
    current_line = 1
    for j in range(num_nodes_y):
        for i in range(nx):  # 单元数nx → 水平线数nx
            line_tags_h[i, j] = current_line
            gmsh.model.geo.addLine(point_tags[i, j], point_tags[i+1, j], current_line)
            current_line += 1
    
    line_tags_v = np.zeros((num_nodes_x, ny), dtype=int)  # ny条垂直线
    for i in range(num_nodes_x):
        for j in range(ny):  # 单元数ny → 垂直线数ny
            line_tags_v[i, j] = current_line
            gmsh.model.geo.addLine(point_tags[i, j], point_tags[i, j+1], current_line)
            current_line += 1

    # 创建四边形面（nx×ny个单元）
    for i in range(nx):
        for j in range(ny):
            loop_lines = [
                line_tags_h[i, j], line_tags_v[i+1, j],
                -line_tags_h[i, j+1], -line_tags_v[i, j]
            ]
            loop_tag = i * ny + j + 1
            gmsh.model.geo.addCurveLoop(loop_lines, loop_tag)
            gmsh.model.geo.addPlaneSurface([loop_tag], loop_tag)

    # 网格参数
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)

    finalize_gmsh(output_path)
    return output_path

def generate_cvt_polygon_mesh(
    nx: int,          # nx: 参考单元数（控制CVT点密度）
    ny: int,          # ny: 参考单元数（控制CVT点密度）
    domain: Tuple[float, float, float, float] = (0, 1, 0, 1),
    output_dir: str = "mesh_data",
    seed: int = 42,
    lloyd_iterations: int = 100
) -> str:
    """
    基于CVT（Centroidal Voronoi Tessellation）的均匀多边形网格
    nx/ny代表参考单元数（控制生成点数量）
    """
    x_min, x_max, y_min, y_max = domain
    output_file = f"cvt_polygon_{nx}x{ny}.msh"
    output_path = os.path.join(output_dir, output_file)
    Path(output_dir).mkdir(exist_ok=True)

    initialize_gmsh(f"CVTPolygonMesh_{nx}x{ny}")
    
    # 1. 计算CVT生成点数（基于参考单元数）
    num_points = nx * ny  # 控制点数量（与参考单元数匹配）
    cell_size = min(x_max - x_min, y_max - y_min) / max(4, nx//2)

    # 2. 生成初始随机点集
    np.random.seed(seed)
    initial_points = np.random.uniform(
        low=[x_min + 0.05, y_min + 0.05],
        high=[x_max - 0.05, y_max - 0.05],
        size=(num_points, 2)
    )

    # 3. Lloyd算法优化生成CVT点集
    print(f"📌 执行Lloyd迭代优化（{lloyd_iterations}次）生成CVT...")
    cvt_points = LloydRelaxation.lloyd_relaxation(
        initial_points, domain, iterations=lloyd_iterations
    )

    # 4. 在Gmsh中创建CVT点
    point_tags = []
    tag = 1
    for (x, y) in cvt_points:
        gmsh.model.geo.addPoint(x, y, 0.0, cell_size, tag)
        point_tags.append(tag)
        tag += 1

    # 5. 创建边界矩形
    border_points = [
        gmsh.model.geo.addPoint(x_min, y_min, 0, cell_size),
        gmsh.model.geo.addPoint(x_max, y_min, 0, cell_size),
        gmsh.model.geo.addPoint(x_max, y_max, 0, cell_size),
        gmsh.model.geo.addPoint(x_min, y_max, 0, cell_size)
    ]
    border_lines = []
    for i in range(4):
        border_lines.append(gmsh.model.geo.addLine(border_points[i], border_points[(i+1)%4]))
    border_loop = gmsh.model.geo.addCurveLoop(border_lines)
    gmsh.model.geo.addPlaneSurface([border_loop], 1)

    # 6. CVT网格生成参数
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.Algorithm", 7)          # Voronoi算法
    gmsh.option.setNumber("Mesh.RandomFactor", 0.01)    # 极小随机因子保证均匀性
    gmsh.option.setNumber("Mesh.ElementOrder", 1)       # 线性单元
    gmsh.option.setNumber("Mesh.SmoothRatio", 5)        # 高平滑度
    gmsh.option.setNumber("Mesh.Optimize", 3)           # 深度优化
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)     # Netgen优化

    finalize_gmsh(output_path)
    return output_path

# ---------------------- 验证与批量生成 ----------------------
def validate_mesh_boundaries(mesh_files: List[str], domain: Tuple[float, float, float, float]) -> None:
    """验证网格边界是否正确"""
    x_min, x_max, y_min, y_max = domain
    tolerance = 1e-6
    print(f"\n{'='*50}")
    print("验证网格边界...")
    print(f"{'='*50}")
    for mesh_file in mesh_files:
        if os.path.exists(mesh_file):
            try:
                import meshio
                mesh = meshio.read(mesh_file)
                nodes = mesh.points[:, :2]
                min_x, max_x = nodes[:, 0].min(), nodes[:, 0].max()
                min_y, max_y = nodes[:, 1].min(), nodes[:, 1].max()
                errors = []
                if abs(min_x - x_min) > tolerance:
                    errors.append(f"左边界: {min_x:.6f} vs 预期 {x_min:.6f}")
                if abs(max_x - x_max) > tolerance:
                    errors.append(f"右边界: {max_x:.6f} vs 预期 {x_max:.6f}")
                if abs(min_y - y_min) > tolerance:
                    errors.append(f"下边界: {min_y:.6f} vs 预期 {y_min:.6f}")
                if abs(max_y - y_max) > tolerance:
                    errors.append(f"上边界: {max_y:.6f} vs 预期 {y_max:.6f}")
                if errors:
                    print(f"⚠️  {os.path.basename(mesh_file)} 边界错误:")
                    for error in errors:
                        print(f"   {error}")
                else:
                    print(f"✓ {os.path.basename(mesh_file)}: 边界验证通过")
            except Exception as e:
                print(f"❌ 验证 {mesh_file} 失败: {e}")

def generate_all_resolution_meshes(
    resolutions: List[Tuple[int, int]] = [(2,2), (4,4), (8,8), (16,16)],
    domain: Tuple[float, float, float, float] = (0, 1, 0, 1),
    output_dir: str = "mesh_data"
) -> Dict[str, List[str]]:
    """批量生成所有类型网格（resolutions中的值代表单元数）"""
    generated_files = {
        "triangle": [],
        "orthogonal": [],       # 正交四边形（矩形）
        "square": [],           # 正方形网格（新增）
        "small_perturb_quad": [],
        "cvt_polygon": []
    }

    # 清空输出目录
    if os.path.exists(output_dir):
        print(f"清理输出目录: {output_dir}")
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"开始生成网格（输出目录: {output_dir}）")
    print(f"目标分辨率: {[f'{nx}x{ny}' for nx, ny in resolutions]}（均为单元数）")
    print(f"{'='*60}")

    for idx, (nx, ny) in enumerate(resolutions):
        print(f"\n【第{idx+1}/{len(resolutions)}组】生成 {nx}x{ny} 分辨率网格（单元数）")
        print("-" * 50)
        
        # 1. 三角形网格
        try:
            file_path = generate_triangle_mesh(nx=nx, ny=ny, domain=domain, output_dir=output_dir)
            generated_files["triangle"].append(file_path)
        except Exception as e:
            print(f"✗ 三角形网格({nx}x{ny})生成失败: {e}")

        # 2. 正交四边形网格（矩形）
        try:
            file_path = generate_orthogonal_mesh(nx=nx, ny=ny, domain=domain, output_dir=output_dir)
            generated_files["orthogonal"].append(file_path)
        except Exception as e:
            print(f"✗ 正交四边形网格({nx}x{ny})生成失败: {e}")

        # 3. 正方形网格（取nx/ny最大值作为单元数）
        try:
            n = max(nx, ny)
            print(f"正方形网格参数: n={n}（{n}×{n}单元）")
            file_path = generate_square_mesh(n=n, domain=domain, output_dir=output_dir)
            generated_files["square"].append(file_path)
        except Exception as e:
            print(f"✗ 正方形网格({n}x{n})生成失败: {e}")
            print(f"   错误详情: {str(e)}")

        # 4. 小扰动四边形网格
        try:
            perturb_factor = 0.20 if nx >= 8 else 0.10
            file_path = generate_small_perturb_quad_mesh(
                nx=nx, ny=ny, domain=domain, output_dir=output_dir, 
                perturbation_factor=perturb_factor
            )
            generated_files["small_perturb_quad"].append(file_path)
        except Exception as e:
            print(f"✗ 小扰动四边形网格({nx}x{ny})生成失败: {e}")

        # 5. CVT多边形网格
        try:
            file_path = generate_cvt_polygon_mesh(
                nx=nx, ny=ny, domain=domain, output_dir=output_dir,
                lloyd_iterations=100 if nx <= 8 else 100  # 高分辨率减少迭代加速
            )
            generated_files["cvt_polygon"].append(file_path)
        except Exception as e:
            print(f"✗ CVT多边形网格({nx}x{ny})生成失败: {e}")
    
    # 验证边界
    all_files = []
    for file_list in generated_files.values():
        all_files.extend(file_list)
    validate_mesh_boundaries(all_files, domain=domain)

    return generated_files


def list_generated_meshes(output_dir: str = "mesh_data") -> None:
    """列出所有生成的网格文件"""
    if not os.path.exists(output_dir):
        print(f"目录 '{output_dir}' 不存在")
        return
    
    def sort_key(filename):
        name_without_ext = filename.replace('.msh', '')
        resolution_part = name_without_ext.split('_')[-1]
        try:
            nx = int(resolution_part.split('x')[0])
        except:
            nx = 0
        type_part = '_'.join(name_without_ext.split('_')[:-1])
        return (type_part, nx)
    
    print(f"\n{'='*60}")
    print(f"{output_dir} 目录下生成的网格文件（共{len(os.listdir(output_dir))}个）")
    print(f"{'='*60}")
    
    files = sorted([f for f in os.listdir(output_dir) if f.endswith('.msh')], key=sort_key)
    
    for file in files:
        filepath = os.path.join(output_dir, file)
        size = os.path.getsize(filepath)
        try:
            import meshio
            mesh = meshio.read(filepath)
            nodes = len(mesh.points)
            cells = sum(len(v) for k, v in mesh.cells_dict.items())
            # 识别单元类型
            cell_type = list(mesh.cells_dict.keys())[0] if mesh.cells_dict else "未知"
            print(f"  {file:35} 节点数: {nodes:4d}, 单元数: {cells:4d}, 类型: {cell_type:8}, 大小: {size:8,} 字节")
        except Exception as e:
            print(f"  {file:35} 大小: {size:8,} 字节 (解析失败: {str(e)[:20]}...)")

# ---------------------- 主程序入口 ----------------------
if __name__ == "__main__":
    # 目标分辨率（均为单元数）
    target_resolutions = [(1,1),(2,2), (4,4), (8,8), (16,16),(32,32),(64,64),(128,128)]
    domain = (0, 1, 0, 1)
    output_dir = "mesh_data"

    # 生成所有网格
    try:
        start_total = time.time()
        generated_files = generate_all_resolution_meshes(
            resolutions=target_resolutions,
            domain=domain,
            output_dir=output_dir
        )
        
        # 列出生成的文件
        list_generated_meshes(output_dir)
        
        # 统计结果
        total_files = sum(len(files) for files in generated_files.values())
        total_time = time.time() - start_total
        
        print(f"\n🎉 所有网格生成完成！")
        print(f"总耗时: {total_time:.2f}秒 | 生成文件数: {total_files}个")
        print(f"文件位置: {os.path.abspath(output_dir)}")
        
        # 详细打印各类型网格生成情况
        print(f"\n📊 各类型网格生成统计:")
        for mesh_type, files in generated_files.items():
            if files:
                print(f"  ✓ {mesh_type}: {len(files)}个文件")
                for f in files:
                    if os.path.exists(f):
                        size = os.path.getsize(f)
                        print(f"     - {os.path.basename(f)} ({size:,} 字节)")
                    else:
                        print(f"     - {os.path.basename(f)} (文件不存在!)")
            else:
                print(f"  ✗ {mesh_type}: 0个文件 (生成失败)")
                
    except Exception as e:
        print(f"\n❌ 网格生成出错: {e}")
        import traceback
        traceback.print_exc()
        
    # 最后检查输出目录
    print(f"\n🔍 最终检查输出目录内容:")
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        if files:
            for f in files:
                fp = os.path.join(output_dir, f)
                size = os.path.getsize(fp) if os.path.isfile(fp) else 0
                print(f"   {f} ({size:,} 字节)")
        else:
            print(f"   目录为空")
    else:
        print(f"   目录不存在")