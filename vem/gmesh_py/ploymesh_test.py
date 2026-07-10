import numpy as np
import os
import matplotlib.pyplot as plt

# 创建保存结果的文件夹
output_dir = "poly_mesh_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. 定义边界框 [xmin, xmax, ymin, ymax]
BdBox = [0, 1, 0, 1]

# 简单但有效的SDF函数 - 单位正方形
def SDF_square(p):
    """单位正方形的符号距离函数"""
    p = np.asarray(p)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    
    x = p[:, 0]
    y = p[:, 1]
    
    # 计算到四条边的距离
    d_left = x
    d_right = 1 - x
    d_bottom = y
    d_top = 1 - y
    
    # 内部点返回负距离，外部点返回正距离
    inside = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
    min_dist_to_edge = np.minimum.reduce([d_left, d_right, d_bottom, d_top])
    
    distances = np.where(inside, -min_dist_to_edge, min_dist_to_edge)
    return distances.reshape(-1, 1)

# 简单的圆形SDF
def SDF_circle(p):
    """圆形的符号距离函数"""
    p = np.asarray(p)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    
    x = p[:, 0] - 0.5
    y = p[:, 1] - 0.5
    
    r = np.sqrt(x**2 + y**2)
    return (r - 0.4).reshape(-1, 1)

# 简单的六边形SDF
def SDF_hexagon_simple(p):
    """简单六边形的符号距离函数"""
    p = np.asarray(p)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    
    x = p[:, 0] - 0.5
    y = p[:, 1] - 0.5
    
    # 计算极坐标
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # 六边形的角度参数
    hex_angle = np.pi / 3  # 60度
    
    # 将角度映射到[0, 60]度
    theta = (theta + np.pi) % (2*np.pi) - np.pi  # 归一化到[-π, π]
    sector = np.floor((theta + np.pi/6) / hex_angle)
    theta_in_sector = theta - sector * hex_angle + np.pi/6
    
    # 计算距离
    distance = 0.4 / np.cos(theta_in_sector - np.pi/6) - r
    
    return distance.reshape(-1, 1)

# 3. 创建Domain对象
try:
    from pyPolyMesher import Domain
    
    print("="*60)
    print("多边形网格生成器")
    print("="*60)
    print("\n选择多边形类型:")
    print("1. 正方形 (简单测试)")
    print("2. 圆形")
    print("3. 六边形")
    
    choice = input("\n请输入选择 (1-3, 默认1): ").strip() or "1"
    
    if choice == "2":
        NewDomain = Domain("Circle Domain", BdBox, SDF_circle)
        polygon_name = "圆形"
        sdf_func = SDF_circle
    elif choice == "3":
        NewDomain = Domain("Hexagon Domain", BdBox, SDF_hexagon_simple)
        polygon_name = "六边形"
        sdf_func = SDF_hexagon_simple
    else:
        NewDomain = Domain("Square Domain", BdBox, SDF_square)
        polygon_name = "正方形"
        sdf_func = SDF_square
    
    print(f"\n成功创建{polygon_name} Domain对象")
    
    # 4. 可视化域
    print("\n生成域可视化...")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 生成网格点用于可视化
    x = np.linspace(-0.1, 1.1, 200)
    y = np.linspace(-0.1, 1.1, 200)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.flatten(), Y.flatten()])
    
    # 计算符号距离
    distances = sdf_func(points)
    Z = distances.reshape(200, 200)
    
    # 绘制等高线
    ax.contour(X, Y, Z, levels=[0], colors='red', linewidths=3, linestyles='-')
    
    # 填充内部区域
    ax.contourf(X, Y, Z, levels=[-np.inf, 0], colors='lightblue', alpha=0.3)
    
    # 设置图形属性
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_title(f'Domain: {polygon_name}', fontsize=16, fontweight='bold')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加边界框
    rect = plt.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='gray', 
                         facecolor='none', linestyle=':')
    ax.add_patch(rect)
    
    # 保存图片
    domain_img_path = os.path.join(output_dir, 'domain_visualization.png')
    plt.savefig(domain_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"域可视化已保存到: {domain_img_path}")
    
    # 5. 生成多边形网格
    from pyPolyMesher import PolyMesher
    
    print("\n" + "="*60)
    print("网格生成参数设置")
    print("="*60)
    
    # 获取用户输入
    try:
        desired_nodes = int(input(f"期望的节点数 (控制网格密度，默认50): ") or "50")
        iterations = int(input("迭代次数 (影响网格质量，默认100): ") or "100")
    except:
        print("使用默认参数...")
        desired_nodes = 50
        iterations = 100
    
    print(f"\n开始生成{polygon_name}多边形网格...")
    print(f"参数: {desired_nodes}个节点, {iterations}次迭代")
    
    try:
        # 方法1：尝试不使用初始点集
        print("尝试方法1: 不使用初始点集...")
        Node, Element, Supp, Load, P = PolyMesher(NewDomain, desired_nodes, iterations, anim=False)
        
        print(f"✓ 网格生成成功!")
        print(f"   实际节点数: {len(Node)}")
        print(f"   多边形单元数: {len(Element)}")
        
    except Exception as e:
        print(f"方法1失败: {e}")
        print("尝试方法2: 使用随机初始点集...")
        
        try:
            # 方法2：创建正确形状的初始点集
            # 在边界框内生成随机点
            np.random.seed(42)  # 固定随机种子以便重现
            
            # 生成指定数量的随机点
            initial_points = np.random.rand(desired_nodes, 2)
            initial_points[:, 0] = initial_points[:, 0] * (BdBox[1] - BdBox[0]) + BdBox[0]
            initial_points[:, 1] = initial_points[:, 1] * (BdBox[3] - BdBox[2]) + BdBox[2]
            
            print(f"生成的初始点集形状: {initial_points.shape}")
            
            # 使用初始点集
            Node, Element, Supp, Load, P = PolyMesher(NewDomain, desired_nodes, iterations, P=initial_points, anim=False)
            
            print(f"✓ 网格生成成功!")
            print(f"   实际节点数: {len(Node)}")
            print(f"   多边形单元数: {len(Element)}")
            
        except Exception as e2:
            print(f"方法2失败: {e2}")
            print("尝试方法3: 使用更简单的参数...")
            
            try:
                # 方法3：使用更保守的参数
                Node, Element, Supp, Load, P = PolyMesher(NewDomain, 25, 50, anim=False)
                print(f"✓ 网格生成成功!")
                print(f"   实际节点数: {len(Node)}")
                print(f"   多边形单元数: {len(Element)}")
                
            except Exception as e3:
                print(f"所有方法都失败: {e3}")
                print("创建简单网格作为替代...")
                
                # 创建简单的结构化网格
                if polygon_name == "正方形":
                    # 正方形区域的简单网格
                    nx = int(np.sqrt(desired_nodes))
                    ny = nx
                    x = np.linspace(0, 1, nx)
                    y = np.linspace(0, 1, ny)
                    X, Y = np.meshgrid(x, y)
                    Node = np.column_stack([X.flatten(), Y.flatten()])
                    
                    # 创建四边形单元
                    Element = []
                    for i in range(ny-1):
                        for j in range(nx-1):
                            n1 = i * nx + j
                            n2 = i * nx + j + 1
                            n3 = (i+1) * nx + j + 1
                            n4 = (i+1) * nx + j
                            Element.append([n1+1, n2+1, n3+1, n4+1])  # 1-based索引
                    
                    Element = np.array(Element)
                    print(f"创建了替代网格: {len(Node)}个节点, {len(Element)}个四边形")
                
    # 6. 保存网格数据
    print("\n保存网格数据...")
    
    # 保存节点信息
    nodes_path = os.path.join(output_dir, 'nodes.txt')
    np.savetxt(nodes_path, Node, fmt='%.6f', 
               header=f'{polygon_name} Mesh - Node Coordinates (x, y)\n{len(Node)} nodes')
    print(f"✓ 节点数据: {nodes_path}")
    
    # 保存单元信息
    elements_path = os.path.join(output_dir, 'elements.txt')
    with open(elements_path, 'w') as f:
        f.write(f'# {polygon_name} Polygon Mesh - Element Connectivity\n')
        f.write(f'# Number of Polygons: {len(Element)}\n')
        f.write('# Format: Polygon_ID num_vertices vertex_indices...\n')
        
        for i, elem in enumerate(Element):
            if isinstance(elem, np.ndarray):
                elem_clean = elem[~np.isnan(elem)] if np.any(np.isnan(elem)) else elem
            else:
                elem_clean = elem
            
            elem_clean = np.array(elem_clean, dtype=int)
            num_vertices = len(elem_clean)
            
            # 1-based索引
            f.write(f'{i+1} {num_vertices} ' + ' '.join(map(str, elem_clean)) + '\n')
    
    print(f"✓ 单元数据: {elements_path}")
    
    # 7. 可视化生成的多边形网格
    print("\n生成网格可视化...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 左图：彩色多边形网格
    colors = plt.cm.tab20c(np.linspace(0, 1, 20))
    
    polygon_counts = {}
    for elem in Element:
        if isinstance(elem, np.ndarray):
            # 检查是否有NaN值
            if np.any(np.isnan(elem)):
                elem_clean = elem[~np.isnan(elem)]
            else:
                elem_clean = elem
            elem_clean = np.array(elem_clean, dtype=int)
        else:
            elem_clean = np.array(elem, dtype=int)
        
        num_sides = len(elem_clean)
        polygon_counts[num_sides] = polygon_counts.get(num_sides, 0) + 1
        
        # 获取多边形坐标
        try:
            # 尝试基于1的索引
            polygon_coords = Node[elem_clean - 1]
        except:
            try:
                # 尝试基于0的索引
                polygon_coords = Node[elem_clean]
            except:
                # 如果索引超出范围，跳过这个多边形
                continue
        
        # 绘制多边形
        if len(polygon_coords) >= 3:
            color_idx = min(num_sides - 3, len(colors) - 1) if num_sides >= 3 else 0
            polygon = plt.Polygon(polygon_coords, 
                                 edgecolor='black', 
                                 facecolor=colors[color_idx],
                                 alpha=0.8, 
                                 linewidth=0.8)
            ax1.add_patch(polygon)
    
    # 绘制节点
    ax1.scatter(Node[:, 0], Node[:, 1], color='red', s=15, zorder=10, alpha=0.6)
    
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect('equal')
    ax1.set_title(f'{polygon_name} Polygon Mesh\n{len(Node)} nodes, {len(Element)} polygons', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.grid(True, alpha=0.2)
    
    # 右图：网格统计
    if polygon_counts:
        sides = list(polygon_counts.keys())
        counts = list(polygon_counts.values())
        
        bars = ax2.bar(range(len(sides)), counts, color=plt.cm.tab20c(np.linspace(0, 1, len(sides))))
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')
        
        ax2.set_xlabel('Number of Sides', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Polygon Type Distribution', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(sides)))
        ax2.set_xticklabels([f'{s}' for s in sides])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加百分比
        total = sum(counts)
        for i, (side, count) in enumerate(zip(sides, counts)):
            percent = count/total*100
            ax2.text(i, count/2, f'{percent:.1f}%', 
                    ha='center', va='center', color='white', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No polygon data available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Polygon Type Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存可视化
    mesh_img_path = os.path.join(output_dir, 'mesh_visualization.png')
    plt.savefig(mesh_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 网格可视化: {mesh_img_path}")
    
    # 8. 创建信息文件
    info_path = os.path.join(output_dir, 'mesh_info.txt')
    with open(info_path, 'w') as f:
        f.write('='*60 + '\n')
        f.write(f'POLYGON MESH INFORMATION - {polygon_name}\n')
        f.write('='*60 + '\n\n')
        
        f.write('DOMAIN INFORMATION:\n')
        f.write(f'  Shape: {polygon_name}\n')
        f.write(f'  Bounding Box: [{BdBox[0]}, {BdBox[1]}] × [{BdBox[2]}, {BdBox[3]}]\n\n')
        
        f.write('MESH STATISTICS:\n')
        f.write(f'  Number of Nodes: {len(Node)}\n')
        f.write(f'  Number of Polygons: {len(Element)}\n\n')
        
        if polygon_counts:
            f.write('POLYGON TYPE DISTRIBUTION:\n')
            total_vertices = 0
            for sides, count in sorted(polygon_counts.items()):
                percent = count/len(Element)*100
                total_vertices += sides * count
                f.write(f'  {sides}-sided polygons: {count:4d} ({percent:5.1f}%)\n')
            
            f.write(f'\n  Average vertices per polygon: {total_vertices/len(Element):.2f}\n')
        
        f.write('\nNODE COORDINATE RANGES:\n')
        f.write(f'  X: [{np.min(Node[:, 0]):.6f}, {np.max(Node[:, 0]):.6f}]\n')
        f.write(f'  Y: [{np.min(Node[:, 1]):.6f}, {np.max(Node[:, 1]):.6f}]\n\n')
        
        f.write('GENERATION PARAMETERS:\n')
        f.write(f'  Desired Nodes: {desired_nodes}\n')
        f.write(f'  Iterations: {iterations}\n\n')
        
        f.write('FILES:\n')
        f.write(f'  {nodes_path}\n')
        f.write(f'  {elements_path}\n')
        f.write(f'  {domain_img_path}\n')
        f.write(f'  {mesh_img_path}\n')
        f.write(f'  {info_path}\n')
    
    print(f"✓ 网格信息: {info_path}")
    
    # 10. 显示总结
    print("\n" + "="*60)
    print("网格生成完成!")
    print("="*60)
    
    print(f"\n{polygon_name}区域:")
    print(f"  • 节点数: {len(Node)}")
    print(f"  • 多边形数: {len(Element)}")
    
    if polygon_counts:
        print("\n多边形类型:")
        for sides, count in sorted(polygon_counts.items()):
            percent = count/len(Element)*100
            print(f"  • {sides}-边形: {count}个 ({percent:.1f}%)")
        
        avg_sides = sum(s*c for s,c in polygon_counts.items())/len(Element)
        print(f"  • 平均边数: {avg_sides:.2f}")
    
    print(f"\n文件保存在: {output_dir}/")
    print("\n使用建议:")
    print("1. 查看网格: cat poly_mesh_data/mesh_info.txt")
    print("2. Python中使用: np.loadtxt('poly_mesh_data/nodes.txt')")
    
except ImportError as e:
    print(f"错误: 无法导入pyPolyMesher - {e}")
    print("\n安装命令:")
    print("pip install pyPolyMesher")
    
except Exception as e:
    print(f"运行时错误: {e}")
    import traceback
    traceback.print_exc()