# CurvedVEM - 新架构记录

这是一个简洁、可维护的混合虚拟元方法架构，目前支持 **Darcy 方程** 和 **Maxwell-Stefan 多组分扩散方程**。

---

## 概述

本项目采用曲边虚拟元法（Curved VEM），通过等参映射将曲边物理单元拉回到直边计算单元，在计算单元上构造 H(div)- 混合虚拟元格式。

- **Darcy 模块**：单组分渗流问题，`u = -κ∇p`, `∇·u = g`
- **Maxwell-Stefan 模块**：多组分非线性扩散，`∂c/∂t + ∇·J = f`, `∇c + A(c)J = 0`

两个模块共享底层基础库（积分、网格、多项式基、PETSc 工具、H(div) 投影矩阵），各自实现专用的组装与求解器。

---

## 目录结构

```
CurvedVEM_Darcy/
├── ARCHITECTURE.md              # 本文档 - 架构记录
├── 工作日志.md                  # Maxwell-Stefan 开发日志
├── Makefile                     # 编译脚本
├── build/                       # 编译产物（二进制文件）
├── core/                        # 核心基础模块（共享）
│   ├── gauss_quadrature.h       # 高斯积分头文件
│   ├── gauss_quadrature.cpp     # 高斯积分实现
│   ├── petsc_utils.h            # PETSc 工具头文件
│   ├── petsc_utils.cpp          # PETSc 工具实现
│   ├── mesh_refiner.h           # 悬点网格生成头文件（服务算例 5）
│   └── mesh_refiner.cpp         # 悬点网格生成实现
├── lib/                         # 底层数学库（共享）
│   ├── polynomial_basis.h       # 显式多项式基函数头文件
│   └── polynomial_basis.cpp     # 显式多项式基函数实现
├── examples/                    # 算例模块
│   ├── darcy_problem.h          # Darcy 算例头文件
│   ├── darcy_problem.cpp        # Darcy 算例实现
│   ├── ms_problem.h             # Maxwell-Stefan 算例头文件
│   └── ms_problem.cpp           # Maxwell-Stefan 算例实现
├── mesh/                        # 网格模块（共享）
│   ├── data/                    # 网格数据文件
│   │   ├── square_*.msh         # 正方形网格系列（1x1~128x128）
│   │   └── orthogonal_*.msh     # 正交网格系列（1x1~128x128）
│   ├── data_refined/            # 悬点网格（core/mesh_refiner 生成，服务算例 5）
│   │   ├── square_*_refined.vtk # legacy VTK，全部单元用 VTK_POLYGON(7)
│   │   ├── square_*_refined.poly# 极简文本格式（NODES/CELLS）
│   │   └── fig/                 # 黑白网格图
│   ├── straight_mesh.h          # 直边网格头文件
│   ├── straight_mesh.cpp        # 直边网格实现
│   ├── curved_mesh.h            # 曲边网格头文件
│   ├── curved_mesh.cpp          # 曲边网格实现
│   ├── polygon_triangulator.h   # 多边形三角剖分头文件
│   └── polygon_triangulator.cpp # 多边形三角剖分实现
├── tests/                       # 测试程序（*_main.cpp 自动发现）
│   ├── test_quadrature_main.cpp      # 积分模块测试
│   ├── test_petsc_utils_main.cpp     # PETSc 工具测试
│   ├── test_polynomial_basis_main.cpp # 多项式基函数测试
│   ├── test_darcy_problem_main.cpp   # Darcy 算例测试
│   ├── test_straight_mesh_main.cpp   # 直边网格测试
│   ├── test_curved_mesh_main.cpp     # 曲边网格测试
│   ├── test_polygon_triangulator_main.cpp # 多边形三角剖分测试
│   ├── test_ms_triblock_q8_main.cpp  # 算例 4 几何准入（detJ>0 / 共享边）
│   └── test_ms_*.cpp                 # MS 各组装环节测试
├── main/                        # 生产算例入口（*_main.cpp 自动发现）
│   ├── Darcy_curved_main.cpp
│   ├── MS_curved_main.cpp            # 算例 1/2
│   ├── MS_curved_bdf2_main.cpp       # 算例 2 BDF2 收敛
│   ├── MS_curved_convergence_main.cpp
│   ├── MS_curved_exact_main.cpp
│   ├── MS_curved_timing_main.cpp
│   ├── MS_half_annulus_main.cpp      # 算例 3
│   ├── MS_triblock_bdf2_main.cpp     # 算例 4 BDF2 收敛
│   └── MS_triblock_frames_main.cpp   # 算例 4 逐时间步出图数据
├── tools/                       # 离线工具（*.cpp 自动发现）
│   ├── export_ms_concentration.cpp   # 解 → 三角剖分 CSV（供 Python 画图）
│   ├── export_triblock_mesh.cpp      # 导出算例 4 网格几何
│   ├── triblock_sweep.cpp            # TriBlockSwirl 参数扫描（detJ 约束）
│   ├── plot_triblock_bw.py           # 黑白网格图，一网格一图
│   ├── plot_triblock_check.py        # 网格自检可视化
│   └── plot_triblock_sweep.py        # 扫参结果可视化
├── scripts/                     # 后处理 Python
│   ├── plot_ms_concentration.py      # 三组分云图 + 曲边网格叠加
│   ├── make_gif.py                   # 帧序列 → GIF
│   └── plot_*.py                     # 其他对比/收敛作图
├── data/                        # 计算与作图产物（.gitignore，不入库）
├── solver/                      # 求解器模块
│   ├── HdivMatrix.h             # H(div) 通用投影矩阵头文件（共享）
│   ├── HdivMatrix.cpp           # H(div) 通用投影矩阵实现（共享）
│   ├── DarcySolver.h            # Darcy 专用组装与求解器
│   ├── DarcySolver.cpp          # Darcy 求解器实现
│   ├── MSSolver.h               # Maxwell-Stefan 专用组装与求解器
│   └── MSSolver.cpp             # MS 求解器实现
├── 混合虚拟元_darcy.md           # Darcy 原始数学文档
└── 混合虚拟元_MaxwellStefan.md   # Maxwell-Stefan 数学文档
```

---

## 已实现模块

### 1. core/gauss_quadrature - 高斯积分模块

**文件**: `core/gauss_quadrature.h` / `core/gauss_quadrature.cpp`

**功能**:
- 一维 Gauss-Legendre 积分 (阶数 1-10)
- 一维 Gauss-Lobatto 积分 (阶数 1-7)
- 二维平面上线段积分
- 二维三角形积分 (支持 1, 3, 4, 7, 9, 12 点)

**主要接口**:
```cpp
namespace vem {

// 积分点结构
struct QuadPoint {
    std::vector<double> x;  // 坐标
    double w;               // 权重
};

class GaussQuadrature {
public:
    // 一维积分 (标准区间[-1,1])
    std::vector<QuadPoint> get_1D_points(QuadratureType type, int order) const;

    // 一维积分 (任意区间[a,b])
    std::vector<QuadPoint> get_1D_points(QuadratureType type, int order,
                                          double a, double b) const;

    // 二维平面线段积分
    std::vector<QuadPoint> get_line_2D_points(QuadratureType type, int order,
                                                double x1, double y1,
                                                double x2, double y2) const;

    // 三角形积分 (标准/任意)
    std::vector<QuadPoint> get_triangle_reference_points(int n_points) const;
    std::vector<QuadPoint> get_triangle_points(const std::vector<double>& vertices,
                                                 int n_points) const;
};

}
```

**设计特点**:
- 简化了命名空间 (`vem` 替代冗长的 `GaussQ`)
- 简化了结构体名 (`QuadPoint` 替代 `QuadraturePoint`)
- 分离了头文件和实现文件，不全部 inline
- 移除了三维预留接口，保持简洁
- 增加了 1 点和 7 点三角形积分选项

**测试**: `tests/test_quadrature_main.cpp`
- 一维 Gauss-Legendre: 所有阶数测试 f(x)=1, f(x)=x, f(x)=x²
- 一维 Gauss-Lobatto: 所有阶数测试 f(x)=1, f(x)=x²
- 二维线段积分: 水平线、竖直线、斜线长度测试
- 三角形积分: 标准三角形和任意三角形，面积和线性函数测试
- **测试结果**: 全部通过 ✓

---

### 2. core/petsc_utils - PETSc 工具模块

**文件**: `core/petsc_utils.h` / `core/petsc_utils.cpp`

**功能**:
- PETSc 资源智能指针管理 (`AutoPetscMat`, `AutoPetscVec`)
- 创建稀疏/稠密矩阵、向量
- 小规模矩阵求逆 (高斯消元)
- 线性方程组求解: GMRES, LU, Schur补分块求解
- 矩阵运算: 乘法、加法、缩放、转置
- 分块矩阵填充
- 调试打印函数

**主要接口**:
```cpp
namespace vem {

// 智能指针类型
using AutoPetscMat = std::unique_ptr<Mat, PetscMatDeleter>;
using AutoPetscVec = std::unique_ptr<Vec, PetscVecDeleter>;

// 创建矩阵/向量
AutoPetscMat create_sparse_matrix(PetscInt rows, PetscInt cols, PetscInt nnz_per_row = 10);
AutoPetscMat create_dense_matrix(PetscInt rows, PetscInt cols);
AutoPetscVec create_vector(PetscInt n);

// 求逆
AutoPetscMat inverse_matrix(const AutoPetscMat& A);

// 求解
PetscErrorCode solve_linear_system(const AutoPetscMat& A, const AutoPetscVec& b,
                                    const AutoPetscVec& x, PetscBool* converged, bool verbose = false);
PetscErrorCode solve_linear_system_lu(...)  // LU 直接求解
PetscErrorCode solve_linear_system_schur(...)  // 鞍点系统 Schur 补求解

// 矩阵运算
AutoPetscMat multiply_matrices(const AutoPetscMat& A, const AutoPetscMat& B, PetscScalar scalar = 1.0);
AutoPetscMat add_matrices(const AutoPetscMat& A, const AutoPetscMat& B, PetscScalar scalar = 1.0);
AutoPetscMat scale_matrix(const AutoPetscMat& A, PetscScalar scalar);
AutoPetscMat transpose_matrix(const AutoPetscMat& A);

// 分块填充
void insert_submatrix(Mat M, const AutoPetscMat& A, PetscInt row_start, PetscInt col_start);

// 打印
void print_matrix(const AutoPetscMat& A, const char* name = "Matrix");
void print_vector(const AutoPetscVec& v, const char* name = "Vector");

}
```

**设计特点**:
- 统一命名空间 (`vem`)
- 使用智能指针自动管理 PETSc 资源，防止内存泄漏
- 分离了头文件和实现
- 提供了多种求解器选项 (迭代 GMRES / 直接 LU / Schur补)
- 丰富的矩阵运算功能
- 方便的调试打印接口

**测试**: `tests/test_petsc_utils_main.cpp`
- 矩阵/向量创建和基本操作
- 小规模矩阵求逆 (验证 A*A⁻¹ = I)
- 矩阵运算 (加、乘、缩放、转置)
- 线性方程组求解 (GMRES 和 LU，验证解的正确性)
- 分块矩阵填充
- **测试结果**: 全部通过 ✓

---

### 2.5 core/mesh_refiner - 悬点网格生成模块

**文件**: `core/mesh_refiner.h` / `core/mesh_refiner.cpp`
**驱动**: `tools/refine_mesh.cpp`（可执行 `build/tools/refine_mesh`）
**绘图**: `tools/plot_refined_mesh.py`
**服务对象**: 算例 5（全新算例，与算例 4 无关）

#### 2.5.1 动机

虚拟元法适配任意多边形单元，因此**悬点不需要约束方程**：一个四边形若某条边上多出一个悬点，
直接把它当五边形单元求解即可，悬点自动升格为该单元的一个顶点。本模块把这个想法落成网格文件，
用来展示 VEM 相对于 FEM 的网格灵活性。

#### 2.5.2 为什么输出不是 .msh

> **勘误（2026-09-03）**：本节初稿断言「`get_nodes_count_by_element_type()` 的 switch 只有
> `case 2` 和 `case 3`，五边形无法写进 .msh」，这是**错的**。该 switch 实际是
> `case 2 → 3`、`case 3 → 4`、`case 11 → 5`、`case 12 → 6`，本项目的 reader 是支持五边形和
> 六边形的。下面是订正后的理由。

Gmsh MSH 4.1 标准里**没有通用多边形单元类型**：类型码 `2 = 三角形`、`3 = 四边形`，
再往上是高阶单元（真实的 Gmsh `11` 是 10 节点四面体、`12` 是 27 节点六面体，都是三维单元）。
本项目的 `get_nodes_count_by_element_type()` 把 `11 / 12` 私自复用成了五边形 / 六边形，
这是**项目自定义约定**，Gmsh 本身永远不会输出这样的文件。

所以理论上确实可以把五边形写成 `case 11` 的 .msh，但那样产出的文件只有本项目能读，
在 Gmsh / ParaView 里打开会被当成三维单元误解。既然是给一个全新算例造网格，
选通用格式更划算：

| 格式 | 用途 |
|------|------|
| legacy VTK（ASCII，UNSTRUCTURED_GRID） | 全部单元写成 `VTK_POLYGON`(类型 7)，可直接在 ParaView 打开；附带 `num_vertices` cell data 方便定位五边形 |
| `.poly` 极简文本 | `NODES n` / 坐标 / `CELLS m` / `顶点数 + 顶点编号`，供后续自己写 reader |

读取见 [2.5.9](#259-读回-vtkstraightmeshreaderread_mesh_vtk)。

#### 2.5.3 数据结构

`PolyMeshData` 与 `StraightMeshData` 的本质区别是**单元节点数可变**：

```cpp
struct PolyMeshData {
    std::vector<double> node_coords;   // [x0, y0, x1, y1, ...]
    int num_nodes;
    std::vector<int> cell_nodes;       // 逆时针，变长
    std::vector<int> cell_offsets;     // 长度 num_cells + 1
    int num_cells;
    int nodes_of_cell(int e) const;    // cell_offsets[e+1] - cell_offsets[e]
};
```

#### 2.5.4 算法

`refine_x_strip(in, x_lo, x_hi, out, stats, tol)`，四步：

1. **标记**：单元的**所有**节点 x 都落在 `[x_lo - tol, x_hi + tol]` 内才算「条带内」。
   用全节点判据而非质心判据——条带边界落在格线上时两者等价，不落在格线上时全节点判据
   不会切出半个单元。非四边形输入直接返回 `false`。
2. **搬节点**：原始节点原样复制，编号不变。**必须先于新节点**，否则后面记下的中点编号会被整体顶掉。
3. **建中点**：遍历所有加密单元的边，用边键 `(min(a,b), max(a,b))` 去重后创建中点。
   一次性建完，第 4 步条带外单元才查得到自己边上有没有悬点。
4. **建单元**：
   - 条带内：4 个边中点 + 1 个形心 → 4 个子四边形（均保持逆时针）
   - 条带外：逐边查 `midpoint`，查到就把悬点顺序插入顶点列表 → 四边形升格为五边形

#### 2.5.5 两个必须注意的坑

- **坐标浮点噪声**：`mesh/data/*.msh` 里 `0.25` 实际存的是 `0.2499999999993471`。
  精确比较会把本该加密的一整列单元漏掉，所以 `tol` 是必需参数（默认 `1e-8`）。
- **中点去重**：相邻两个加密单元看到同一条边时必须得到同一个节点编号。边键保证了这一点，
  也保证了粗单元一侧插入的悬点与细单元一侧的顶点**严格是同一个节点**，而不是两个坐标相同的节点。

#### 2.5.6 自检

`validate_poly_mesh()` 有五组检查，其中**边的流形性**是判断悬点处理是否正确的关键一条：
每条无向边只能出现 1 次（且整条落在区域外边界上）或 2 次。若某个悬点只被细单元一侧认领、
粗单元没升格成多边形，半条边就会「只出现 1 次却又不在区域边界上」，这里立刻抓到。

其余四组：结构自洽（offsets / cell_nodes 长度）、逐单元（索引越界、重复顶点、有向面积为正）、
面积守恒（单元面积之和 == 外接矩形面积，覆盖漏单元与单元重叠）、孤立节点。

#### 2.5.7 生成结果

条带取 `x ∈ [0.25, 0.75]`。跳过 `square_1x1.msh`——它只有 2×2 个单元，格线是 0/0.5/1，
第一个单元就横跨 [0, 0.5]，不可能整体落在中间条带内。

注意 `square_nxn.msh` 实际含 **2n × 2n** 个单元，文件名里的 n 不是单元数。
输出沿用源文件的 n 命名，便于追溯。

| 源文件 | 输入单元 | 输出单元 | 加密单元 | 悬点 | 五边形 |
|--------|---------|---------|---------|------|-------|
| square_2x2 | 16 | 40 | 8 | 8 | 8 |
| square_4x4 | 64 | 160 | 32 | 16 | 16 |
| square_8x8 | 256 | 640 | 128 | 32 | 32 |
| square_16x16 | 1024 | 2560 | 512 | 64 | 64 |
| square_32x32 | 4096 | 10240 | 2048 | 128 | 128 |
| square_64x64 | 16384 | 40960 | 8192 | 256 | 256 |
| square_128x128 | 65536 | 163840 | 32768 | 512 | 512 |

7 张网格自检全部通过，最大单元边数均为 5（无意外的六边形）。
另经独立的 Python 复核：所有五边形的第 5 个顶点恰是所在边的中点；悬点 x 坐标只取 0.25 / 0.75；
**重复坐标的节点组数为 0**（证明悬点是单个共享节点，去重键生效）。

#### 2.5.8 边界

本模块**只生成网格文件**，不参与任何求解流程，不依赖 PETSc，不修改任何既有功能文件。
特别地，算例 4（`TriBlockSwirlMapping`）是**扭曲四边形算例**，其 Q8 映射按每单元 4 个角点建表
（`ms_problem.cpp` 中 `nv != 4` 直接抛异常），**按设计如此**，不在本模块的服务范围内。

#### 2.5.9 读回 VTK：`StraightMeshReader::read_mesh_vtk`

**文件**: `mesh/straight_mesh.h` / `mesh/straight_mesh.cpp`（**纯新增**，原有 `read_mesh` 及其余函数一行未动）

```
read_mesh(.msh)  ──┐
                   ├──► compute_cell_properties ──► generate_element_edges ──► compute_boundary_info
read_mesh_vtk(.vtk)┘         （三个后处理函数两条路径完全复用）
```

只有解析层不同。VTK 的 `CELLS` 段每行自带顶点数，天然支持变边数多边形；
后处理链原样复用，这是"`get_mesh_data()` 两条路径语义一致"最强的保证。
新增私有函数 `read_vtk_nodes_and_cells()` 解析 `POINTS` / `CELLS` / `CELL_TYPES` 三段：

- 用 **token 扫描**（`istream >>` 找关键字）而非逐行匹配，VTK 头部是自由格式
- `POINTS` 强制三分量，z 读出即丢弃
- 顶点顺序统一过一遍 `reorder_polygon_to_ccw()`——VTK 不保证方向
- `CELL_TYPES` **必须校验**且只接受 `5`/`7`/`9`（三角形/多边形/四边形）。
  这不是形式检查：若混进三维或高阶单元，顶点表就不再是"逆时针多边形环"，
  `generate_element_edges()` 会算出错误的边**而不报错**

#### 2.5.10 兼容性验证：`tests/test_vtk_mesh_main.cpp`

```bash
make test_vtk_mesh                                   # 默认 square_2x2_refined.vtk
./build/test_vtk_mesh mesh/data_refined/square_64x64_refined.vtk
```

7 张网格全部 **25 项通过 / 1 项失败**。分七节检查，其中三条是悬点网格独有的判据：

| 节 | 关键判据 |
|----|---------|
| [3] | 面积之和 == 1.0；顶点全部逆时针；**存储质心 vs 真实质心** |
| [4] | 欧拉公式 `V - E + F == 2`；**只出现 1 次的边必须整条躺在区域边界上**——若某悬点只被细单元一侧认领、粗单元没升格，就会出现"用了 1 次却在内部"的半条边 |
| [5] | `boundary_edges` 无 `-1`（相邻边界节点间确实存在真实的边）；下/上边界节点数多于左/右（加密条带竖直穿过） |
| [7] | 耳切法能处理**含三个共线顶点**的五边形 |

**唯一失败项 —— 五边形质心不是真实质心（`compute_cell_properties` 的既有行为，非本次改动引入）**

`compute_cell_properties()` 对 `node_count == 4` 用包围盒中心，对其他边数用**顶点平均**。
顶点平均不是多边形质心。以 `square_2x2_refined` 的单元 21 为例：
存储 `(0.85, 0.375)`，真实 `(0.875, 0.375)`。偏差在 7 张网格上精确等于 **0.1h**
（h=0.5→0.025，h=1/128→0.00078125），即随加密按 O(h) 缩小，但**相对单元直径恒为 10%，不消失**。

影响评估：全项目 `cell_centroid_*` 的使用**只有一种**——`solver/HdivMatrix.cpp`、
`solver/DarcySolver.cpp` 里作为缩放单项式基的中心 `xD`。基 `{1, (x-xD)/hD, ...}`
张成的多项式空间与 `xD` 取在单元内哪一点无关，且所有单元积分都走三角剖分 + 高斯求积、
不假设一阶矩为零。**所以这不影响离散解，只轻微影响条件数。**
面积（鞋带公式）与直径（顶点最大间距）都是正确的。

#### 2.5.11 三角剖分：沿用耳切法

**结论：不改 `PolygonTriangulator`,悬点网格继续走既有的 `earClipping`。**

曾评估过换成"质心扇形"剖分,实测后放弃。原因是耳切法在悬点网格上**本来就是通过的**:
五边形那三个共线顶点(悬点及其两侧端点),`isEar()` 用 `cross <= 1e-8` 直接判为非耳自动跳过,
剩下的凸角照样能切出 `n-2` 个三角形,面积吻合到 3e-14。既然没有故障,
就不动这个被算例 1–4 全部依赖的共享模块——它是所有单元积分的唯一入口,
而源项含 sin/cos 非多项式,换剖分会改动已验证的收敛表末位数字。

留一条备忘:`isPointInsideTriangle()` 用的是**绝对**容差 `1e-6` 比较面积,
而 128×128 加密后小单元面积才 1.5e-5。目前 7 张网格都没出问题,
但若将来把网格加密到更细,这个容差需要复查。

---

### 3. examples/darcy_problem - Darcy 算例模块

**文件**: `examples/darcy_problem.h` / `examples/darcy_problem.cpp`

**功能**:
- 曲边映射抽象基类，提供接口
- 正弦扰动曲边映射实现（来自 md 文档）
- 恒等映射（直边网格）
- Darcy 方程 PDE 数据（渗透张量、边界条件、真解、源项）
- 雅可比矩阵及其相关计算
  - 雅可比矩阵
  - 雅可比行列式
  - 雅可比逆矩阵
  - 雅可比转置
  - 雅可比转置逆

**主要接口**:
```cpp
namespace Darcy {

// 基础几何类型
struct Point2D { double x, y; };
struct Vector2D { double x, y; };
struct Tensor2D { double xx, xy, yx, yy; };

// 曲边映射基类
class IsoparametricMapping {
public:
    virtual Point2D physical_coords(double xi, double eta) const = 0;
    virtual Tensor2D jacobian(double xi, double eta) const = 0;
    virtual double jacobian_det(double xi, double eta) const = 0;
    virtual Tensor2D jacobian_inv(double xi, double eta) const = 0;
    virtual Tensor2D jacobian_transpose(double xi, double eta) const = 0;
    virtual Tensor2D jacobian_transpose_inv(double xi, double eta) const = 0;
};

// 正弦扰动曲边映射
class SinPerturbationMapping : public IsoparametricMapping {
public:
    // x(ξ, η) = ξ + ε sin(2πη), y(ξ, η) = η + ε sin(2πξ)
    SinPerturbationMapping(double eps = 0.05);
    // ... 实现基类接口
};

// 恒等映射（直边）
class IdentityMapping : public IsoparametricMapping { /* ... */ };

// Darcy PDE 数据
struct DarcyPdeData {
    std::function<Tensor2D(double x, double y)> kappa;
    std::function<double(double x, double y)> dirichlet_p;
    std::function<double(double x, double y, double nx, double ny)> neumann_un;
    std::function<double(double x, double y)> solution_p;
    std::function<Vector2D(double x, double y)> solution_u;
    std::function<double(double x, double y)> source_g;
};

// 算例管理类
class DarcyProblem {
public:
    void set_problem_index(int index);
    bool init_problem();
    const DarcyPdeData& get_pde_data() const;
    const IsoparametricMapping& get_mapping() const;

    // 由物理区域解析数据拉回到直边计算区域
    double computational_solution_p(double xi, double eta) const;
    Vector2D computational_solution_u(double xi, double eta) const;
    double computational_source_g(double xi, double eta) const;
    Tensor2D computational_kappa(double xi, double eta) const;
};

}
```

**设计特点**:
- 曲边映射设计为基类+派生类，方便扩展新的映射
- 所有雅可比相关计算统一放在映射类中
- PDE 数据使用 std::function 绑定，灵活可扩展
- 提供多个算例（算例1：正弦真解 + `eps=0.05` 曲边映射；算例2：相同正弦真解 + `eps=0` 映射，严格退化为直边）
- 计算区域解析数据始终由物理解析数据与当前映射在线拉回，避免两套公式失配
- 压力使用复合拉回，速度使用逆 Piola 变换，源项和渗透张量使用对应的守恒变换

**测试**: `tests/test_darcy_problem_main.cpp`
- 测试曲边映射的各项计算正确性
- 验证 J * invJ = I
- 验证物理区域 Darcy 方程 u = -κ∇p 成立
- 验证计算区域 p̂、û、ĝ、K̂ 确实由物理数据映射得到
- 验证正向/逆向 Piola 变换互相恢复
- 验证计算区域 Darcy 方程 û = -K̂∇̂p̂ 和散度方程 div̂(û) = ĝ
- 验证恒等映射下计算区域与物理区域解析数据一致
- 测试失败时返回非零退出码
- **测试结果**: 全部通过 ✓

---

### 3.5 examples/ms_problem - Maxwell-Stefan 算例模块

**文件**: `examples/ms_problem.h` / `examples/ms_problem.cpp`

**功能**:
- 曲边映射基类 + 四个派生映射（恒等 / 正弦扰动 / 半圆环 / 三块 swirl）
- 二元扩散系数与矩阵分解：`c_ij`, `c* = min(c_ij)`, `bar_c_ij = c_ij - c*`
- 三组分制造解（正弦脉动时间依赖）+ 通量 + 源项 + 边界条件
- 本构关系通量 `J = -bar_A(c)^{-1} ∇c`
- 非线性扩散矩阵评估（初值模式 / 多项式系数模式）
- 计算域拉回（Piola 变换、detJ 加权、链式法则）

#### 3.5.1 核心设计：逐单元映射

映射接口的六个雅可比方法**第一个参数一律是 `cell_idx`**。这不是可选的装饰，是整个曲边模块的承重结构：

- **`cell_idx` 不给默认值。** 分单元映射下「静默取到错误单元的雅可比」会让误差表看起来完全正常却是错的——`MSSolver.cpp` 有 14 处直接调用 `jacobian()` / `jacobian_det()`（Piola 变换与面积元），任何一处传错单元都不会报错，只会污染结果。不给默认值，漏传的调用点在编译期就断掉。
- **`(xi, eta)` 始终是全局计算域坐标**，即直边参考网格上的物理坐标。积分点由 `triangulateElement(mesh_data, cell_idx)` 生成，本来就落在 `cell_idx` 号单元内。所以 `cell_idx` 是**纯增量信息，不引入任何坐标变换**：它只回答「这个点属于哪个单元」，从而让映射可以逐单元取不同公式。
- **全局连续映射收下 `cell_idx` 后直接丢弃**，数值行为与加参数前逐位相同。算例 1/2/3 不受影响。

```cpp
class IsoparametricMapping {
public:
    virtual Point2D physical_coords(int cell_idx, double xi, double eta) const = 0;
    virtual Tensor2D jacobian(int cell_idx, double xi, double eta) const = 0;
    virtual double   jacobian_det(int cell_idx, double xi, double eta) const = 0;
    virtual Tensor2D jacobian_inv(int cell_idx, double xi, double eta) const = 0;
    virtual Tensor2D jacobian_transpose(int cell_idx, double xi, double eta) const = 0;
    virtual Tensor2D jacobian_transpose_inv(int cell_idx, double xi, double eta) const = 0;

    // 网格注入通道：分单元映射要靠 cell_idx 去网格里取该单元节点坐标。
    // 持有方式仿 HdivMatrix：只存指针，不拥有所有权。
    void set_mesh(const vem::StraightMeshReader& reader);  // 非虚，内部调 on_mesh_set()
    bool has_mesh() const;

    // 本映射是否**必须**有网格。分单元映射返回 true，
    // init_problem() 据此强制校验，漏调 set_mesh 立刻抛异常。
    virtual bool requires_mesh() const { return false; }

protected:
    virtual void on_mesh_set() {}   // 网格到手的钩子，需要建表的派生类在这里做
    const vem::StraightMeshReader* mesh_reader_ = nullptr;
    const vem::StraightMeshData*   mesh_data_   = nullptr;
};
```

`set_mesh` 保持非虚——指针记账不必在每个派生类里重复一遍，只把「网格到手了」这个时机通过 `on_mesh_set()` 钩子交出去。

#### 3.5.2 四个映射

| 类 | 算例 | 形式 | 与单元相关 |
|---|---|---|---|
| `IdentityMapping` | — | 恒等，直边 | 否 |
| `SinPerturbationMapping` | 1, 2 | `x = ξ + ε sin(2πη)`, `y = η + ε sin(2πξ)`，`ε=0.05` | 否 |
| `HalfAnnulusMapping` | 3 | 半圆环 | 否 |
| `TriBlockSwirlMapping` | 4 | 真三块分区 + 全局 swirl，逐单元 Q8 | **是** |

前三个是全局连续的单一解析映射，`requires_mesh()` 返回 false。

#### 3.5.3 TriBlockSwirlMapping — 两层架构

这是唯一的分单元映射，也是 `cell_idx` 接口存在的理由。它分两层：

**第 1 层：解析生成器 G。** 一个纯 `(X,Y) → (x,y)` 的映射，只由 `Params` 定义，**完全不知道网格存在**。参考域 `[0,1]²` 被切成真三块：竖直分界曲线 D 贯穿全高，水平界面 H 只存在于 D 右侧并终止在 D 上（T 型节点）；再叠加一个在整条外边界上消失的全局 swirl。因为 swirl 在边界消失，物理域严格等于 `[0,1]²`，**外边界保持直**，所以精确解 / 源项 / 边界条件 / 边界节点语义全部原样沿用算例 2，一个字都不用改。

**第 2 层：逐单元 Q8 采样。** 读直边网格，取每个单元的 4 个角点（`cell_nodes`）+ 4 个中边点（相邻角点算术平均），对这 8 个参考点各求一次 G，把结果存成该单元的 Q8 节点表。雅可比就是 Q8 serendipity 形函数导数。

**换网格只改变 G 的采样密度，从不改变 G 的形状**——这是任意网格都能用的原因。`square_8x8.msh` 和 `square_128x128.msh` 扭曲成的是同一个几何，只是离散得更细。

```cpp
class TriBlockSwirlMapping : public IsoparametricMapping {
public:
    struct Params { /* a, c, s_b, s_t, bv, b1, bb3, bt2, br3, br2,
                       bh, y_r, A_H, alpha0, m, cx, cy */ };
    TriBlockSwirlMapping();                        // 用默认 Params
    explicit TriBlockSwirlMapping(const Params& p);
    // 两个构造函数而不是一个带默认实参的：C++11 下 `= Params()` 作为类内
    // 默认实参时，编译器还没处理完 Params 的成员初始值，会直接报错。

    bool requires_mesh() const override { return true; }

    Point2D generator(double xi, double eta) const;   // G 本身，供自检直接调用

    struct CellQ8 {                 // 单元的 Q8 节点数据，供自检检查 (R2)
        double nx[8], ny[8];        // 8 个节点物理坐标，规范序
        double xi_c, eta_c, hx, hy; // 参考单元中心与边长
    };
    const CellQ8& cell(int cell_idx) const;

protected:
    void on_mesh_set() override;    // 建表
};
```

**相邻单元共享边重合 (R2) 按构造成立**：相邻单元看到同一对全局节点，`(x0+x1)/2` 与 `(x1+x0)/2` 在 IEEE 下逐位相同，故共享边的 3 个 Q8 节点数值同一。实测 Q8 节点表差异为 `0.000e+00`。

**一个坑：** `cell_nodes` 保证逆时针，但**起始角点在单元间循环变化**（实测 square_2x2 是 右下→右上→左上→左下）。所以局部坐标不能按「节点 0 就是 (-1,-1)」硬编码，建表时必须按每个角点相对单元中心的实际位置重排成规范 Q8 序。

**参数标定的硬约束是 `detJ > 0`**（`detJ ≤ 0` 意味着单元翻面自交，不是合法网格）。参数由 `tools/triblock_sweep` 扫参选出，约束是 **N=2,4,8,16 四层全部 detJ > 0**。

**N=2 是绑定约束，不是 N=4**——它是收敛研究的最粗一层，也是最容易翻面的一层：G 层生成器 detJ 全正**不代表** Q8 插值不自交，单元太少时 8 节点二次单元跟不上大角度旋转。早期只扫 `{4,8,16}` 会漏掉这个失效。

三个旋钮在 N=2 上的代价极不对称：
- `beta` 统一放大 1.2 倍反而**改善** N=2 条件数（31.2→14.2，分级重分布了单元）；放大到 1.32 倍就翻面
- `A_H` 很贵：0.10→0.14 就把 N=2 cond 推到 87.2
- `s_t - s_b` 倾斜最贵：0.35/0.60 → 0.32/0.63 就让 N=2 cond 爆到 8716
- `m=2` 严格优于 `m=3`：同 `alpha0` 下更扭而 N=2 cond 更低（13.0 vs 36.5）
- `alpha0` 在 `m=2` 下可到 -85°；-90° 时 N=2 cond 163.6 已边缘，-95° 翻面

`detJ_min` 随细化单调收敛到 **0.20，即 G 自身的下界**，每加密一次差距减半（Q8 插值误差 ~h² 消失）：

| N | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|
| detJ_min | 0.2225 | 0.2190 | 0.2101 | 0.2050 | 0.2024 | 0.2011 | 0.2004 |

（以上为 `kG4` 四点 Gauss 采样。`tools/triblock_sweep` 用 9×9 均匀点含边界，更苛刻，数值不同属正常。）

**已知未决**：Q8 逐单元映射只在单元交界处 C⁰，`detJ` 跨界面跳变。混合 H(div) VEM 所需的映射光滑度在设计文档里只是**估计**，未经证明。

#### 3.5.4 PDE 数据签名约定

```cpp
struct MSPdeData {
    int n_components = 3;
    std::vector<std::vector<double>> c_ij, bar_c_ij;
    double c_star;

    // 计算域回调（*_comp）凡内部要用映射的，都带 cell_idx，
    // 参数顺序统一为「组分号 → 单元号 → 坐标 → 时间」。
    std::function<double(int i, int cell, double xi, double eta)> initial_concentration_comp;
    std::function<double(int i, int cell, double xi, double eta, double t)> source_f_comp;
    std::function<double(int i, int cell, double xi, double eta, double t)> exact_concentration_comp;
    std::function<Vector2D(int i, int cell, double xi, double eta, double t)> exact_flux_comp;
    std::function<double(int i, int cell, double xi, double eta,
                         double nx, double ny, double t)> boundary_flux_comp;

    // 物理域回调取 (x, y)，与单元无关，不带 cell_idx。
    std::function<double(int i, double x, double y, double t)> exact_concentration, source_f;
    std::function<Vector2D(int i, double x, double y, double t)> exact_flux;
    std::function<double(int i, double x, double y,
                         double nx, double ny, double t)> boundary_flux;
    // 例外：initial_concentration 虽是物理域，但内部要逆映射，故带 cell_idx。
    std::function<double(int i, int cell, double x, double y)> initial_concentration;

    // 非线性扩散矩阵 A_ij（纯非线性部分，**不含 c*I**）
    //   A_ii(u) = Σ_{k≠i} bar_c_ik u_k ,  A_ij(u) = -bar_c_ji u_i  (i≠j)
    std::function<double(int i, int j, int cell, double x, double y)>   evaluate_A_initial;
    std::function<double(int i, int j, int cell, double xi, double eta)> evaluate_A_comp_initial;
    // 数值解模式：输入单项式系数，全程不碰映射，故不需要 cell_idx
    std::function<double(int i, int j, const std::vector<double>& u_ploy_coeff,
                         double xD_x, double xD_y, double hD,
                         double xi, double eta)> evaluate_A_comp_poly;
};
```

`evaluate_A_comp_poly` 不带 `cell_idx` 是有意的：它只用多项式系数与 `xD/hD`，全程不碰映射。

#### 3.5.5 MSProblem

```cpp
class MSProblem {
public:
    void set_problem_index(int index);

    // 注入网格。可选调用，但若调用**必须在 init_problem() 之前**——
    // init_problem() 会把网格转交给它构造出来的映射对象。
    void set_mesh(const vem::StraightMeshReader& reader);

    bool init_problem();
    const MSPdeData& get_pde_data() const;
    const IsoparametricMapping& get_mapping() const;

    // 由物理坐标反求计算域坐标。映射是分单元的，全局逆映射不再唯一定义，
    // 故必须指明在哪个单元内求逆。
    Point2D invert_mapping(double x, double y, int cell_idx) const;

private:
    bool init_problem_1();  // ~ init_problem_4()
    void setup_manufactured_solution();  // 算例 2 与算例 4 共用
};
```

`init_problem()` 内的强制校验，是分单元映射唯一的防呆闸门：

```cpp
if (mapping_->requires_mesh() && mesh_reader_ == nullptr) {
    throw std::runtime_error("算例 N 使用分单元映射，必须在 init_problem() 之前调用 set_mesh(reader)");
}
if (mesh_reader_) { mapping_->set_mesh(*mesh_reader_); }
```

**所以所有算例 4 的入口都必须是这个顺序**（`main/MS_triblock_*.cpp`、`tools/export_ms_concentration.cpp` 都已遵守）：

```cpp
problem.set_problem_index(4);
problem.set_mesh(reader);      // 必须在前
problem.init_problem();        // 映射在这里才拿到网格建 Q8 表
```

对算例 1/2/3，`set_mesh` 是空操作（那三个映射不读网格），加上它不改变任何行为。

#### 3.5.6 四个算例

| 算例 | 映射 | PDE 数据 | 说明 |
|---|---|---|---|
| 1 | `SinPerturbationMapping(0.05)` | 独立 | 初值驱动 |
| 2 | `SinPerturbationMapping(0.05)` | `setup_manufactured_solution()` | 制造解基准 |
| 3 | `HalfAnnulusMapping` | 独立 | 半圆环几何 |
| 4 | `TriBlockSwirlMapping` | `setup_manufactured_solution()` | **与算例 2 只差映射** |

算例 2 与算例 4 的 PDE 数据一字不差，所以提取成 `setup_manufactured_solution()` 共用，误差表可以直接横向对比。

**制造解**（算例 2 / 4 共用）：
- c₁ = 0.25 sin(2πx) sin(8πt) + 0.25
- c₂ = 0.25 sin(3πy) sin(6πt) + 0.25
- c₃ = 1 - c₁ - c₂
- 扩散参数：c₁₂=0.1, c₁₃=0.2, c₂₃=2.0, c\*=0.1
- 通量由本构关系 `J = -Ā⁻¹∇c` 得出，`exact_flux_comp` 走逆变 Piola
- 源项 `f = ∂c/∂t + ∇·J`，用中心差分（h=1e-6）
- 边界条件 `boundary_flux_comp = Ĵ·n̂`

注意真解定义在**物理坐标**上。因为算例 4 的外边界是直的、物理域仍是 `[0,1]²`，云图上会看到「浓度场是直的、网格是扭的」——这正是对的；若解跟着网格一起扭了，反而说明 Piola 变换或雅可比接错了。

#### 3.5.7 验证

**几何自检** `tests/test_ms_triblock_q8_main.cpp`：逐层检查 `detJ > 0`、条件数、非仿射度、共享边重合。这是算例 4 的准入闸门。

**算例 4 收敛（BDF2, k=1）**，`main/MS_triblock_bdf2_main.cpp`：

| 网格 | 单元 | c₀ | c₁ | c₂ | c₀ 阶 | c₁ 阶 | c₂ 阶 |
|---|---|---|---|---|---|---|---|
| 4x4 | 16 | 5.1175e-02 | 2.1011e-01 | 1.6949e-01 | — | — | — |
| 8x8 | 64 | 1.3728e-02 | 5.2062e-02 | 4.5771e-02 | 1.90 | 2.01 | 1.89 |
| 16x16 | 256 | 2.7300e-03 | 9.7577e-03 | 8.6506e-03 | 2.33 | 2.42 | 2.40 |

| 网格 | J₀ | J₁ | J₂ | J₀ 阶 | J₁ 阶 | J₂ 阶 |
|---|---|---|---|---|---|---|
| 4x4 | 1.5411e+00 | 1.3456e+00 | 1.8524e+00 | — | — | — |
| 8x8 | 7.7696e-01 | 6.1933e-01 | 8.6065e-01 | 0.99 | 1.12 | 1.11 |
| 16x16 | 2.2703e-01 | 2.0716e-01 | 3.1547e-01 | 1.77 | 1.58 | 1.45 |

浓度 L2 阶趋于 2 = k+1，符合预期。**通量的 8x8 那个 0.99 是前渐近，不是掉阶**——算例 2 在对应位置也是 1.58，细化后同样爬向 ~1.8。算例 4 与算例 2 形状一致，误差常数大 3–5 倍，这是分单元 Q8 映射比全局解析映射粗糙的代价。

单层耗时 10.2s / 61.0s / 341.1s（约 5.6×/次加密），32x32 约 30 分钟，64x64 约 3 小时。

**已知遗留**：`MSProblem::invert_mapping`（ms_problem.cpp:328）牛顿迭代跑满 `max_iter = 50` 后**无收敛断言**，直接返回手上的 `(ξ,η)`；同处注释「当前三个映射与单元无关」已过时（现在是四个）。

---

### 4. mesh/straight_mesh - 直边网格模块

**文件**: `mesh/straight_mesh.h` / `mesh/straight_mesh.cpp`

**功能**:
- 读取 Gmsh 格式网格文件（.msh）
- 存储直边网格数据结构：节点、单元、边、边界
- 计算单元属性：质心、面积、直径
- 生成单元边拓扑信息
- 计算边界信息（边界节点、边界边）
- 查找相邻单元（通过单元局部边或全局边编号）

**主要接口**:
```cpp
namespace vem {

// 网格数据结构
struct StraightMeshData {
    std::vector<double> node_coords;  // 节点坐标
    int num_nodes;
    std::vector<int> cell_nodes;      // 单元节点
    std::vector<int> cell_node_indices;
    std::vector<int> nodes_per_cell;
    std::vector<double> cell_centroid_x, cell_centroid_y;
    std::vector<double> cell_area;
    std::vector<double> cell_diameter;
    int num_cells;
    std::vector<int> cell_edge_indices;
    std::vector<int> global_edges;
    std::vector<int> edge_endpoints;
    std::vector<int> edge_occurrence;
    int num_edges;
    std::vector<int> boundary_nodes;
    std::vector<int> bnode_indices;
    std::vector<int> boundary_edges;
    std::vector<int> bedge_indices;
    int num_boundary_nodes, num_boundary_edges;
    double x_min, x_max, y_min, y_max;
};

// 网格读取器
class StraightMeshReader {
public:
    bool read_mesh(const std::string& filename);
    const StraightMeshData& get_mesh_data() const;
    int find_adjacent_cell(int cell_id, int local_edge_id) const;
    std::pair<int, int> find_adjacent_cells_by_edge(int global_edge_id) const;
};

}
```

**设计特点**:
- 参考旧代码 MeshReader，但简化了命名和结构
- 使用 vem 命名空间统一管理
- 分离头文件和实现
- 支持三角形、四边形等多边形单元
- 自动将单元节点逆时针排序
- 支持查询相邻单元关系

**测试**: `tests/test_straight_mesh_main.cpp`
- 读取 2x2 和 4x4 正方形网格
- 验证网格边界范围
- 验证单元属性计算
- 验证边界信息
- 验证相邻单元查找
- **测试结果**: 全部通过 ✓

---

### 5. mesh/curved_mesh - 曲边网格模块

**文件**: `mesh/curved_mesh.h` / `mesh/curved_mesh.cpp`

**功能**:
- 从直边网格和曲边映射生成曲边网格
- 复用直边网格的拓扑信息（单元、边、边界）
- 通过映射计算曲边网格的节点物理坐标
- 计算曲边网格的边界范围
- 简化数据结构（不需要面积、直径、中心点等）

**主要接口**:
```cpp
namespace vem {

// 曲边网格数据结构
struct CurvedMeshData {
    std::vector<double> node_coords;  // 物理坐标
    int num_nodes;
    std::vector<int> cell_nodes;      // 单元节点
    std::vector<int> cell_node_indices;
    std::vector<int> nodes_per_cell;
    int num_cells;
    std::vector<int> cell_edge_indices;
    std::vector<int> global_edges;
    std::vector<int> edge_endpoints;
    std::vector<int> edge_occurrence;
    int num_edges;
    std::vector<int> boundary_nodes;
    std::vector<int> bnode_indices;
    std::vector<int> boundary_edges;
    std::vector<int> bedge_indices;
    int num_boundary_nodes, num_boundary_edges;
    double x_min, x_max, y_min, y_max;
};

// 曲边网格生成器
class CurvedMeshGenerator {
public:
    // 从直边网格和映射生成曲边网格
    bool generate_from_straight(
        const CurvedMeshData& straight_mesh,
        const Darcy::IsoparametricMapping& mapping
    );
    const CurvedMeshData& get_mesh_data() const;
};

}
```

**设计特点**:
- 简化的数据结构（只保留需要的信息）
- 拓扑复用：直边网格的单元-边-边界关系直接复制
- 坐标映射：只对节点坐标进行变换
- 支持任意 IsoparametricMapping 派生类

**辅助功能** (in `straight_mesh.h/cpp`):
- `StraightMeshReader::to_curved_mesh_data()`: 将直边网格转换为曲边网格数据结构

**测试**: `tests/test_curved_mesh_main.cpp`
- 测试恒等映射（验证坐标不变）
- 测试正弦扰动映射（验证坐标正确变换）
- 测试拓扑信息正确复用
- 测试边界范围计算
- **测试结果**: 全部通过 ✓

---

### 6. mesh/polygon_triangulator - 多边形三角剖分模块

**文件**: `mesh/polygon_triangulator.h` / `mesh/polygon_triangulator.cpp`

**功能**:
- 将任意多边形单元剖分为三角形，用于数值积分
- 支持直边网格和曲边网格
- 单个单元剖分和整个网格批量剖分
- 计算每个三角形面积和单元总面积

**主要接口**:
```cpp
namespace vem {

struct Triangle {
    std::vector<int> nodes;      // 三个顶点全局编号
    std::vector<double> vertices; // 顶点坐标 [x0,y0, x1,y1, x2,y2]
    double area;                  // 三角形面积
};

struct TriangulatedElement {
    std::vector<Triangle> triangles;
    double total_area;            // 单元总面积
};

struct TriangulatedMesh {
    std::vector<TriangulatedElement> element_triangulations;
};

class PolygonTriangulator {
public:
    // 直边网格
    TriangulatedElement triangulateElement(const StraightMeshData& mesh, int element_id);
    TriangulatedMesh triangulateMesh(const StraightMeshData& mesh);

    // 曲边网格
    TriangulatedElement triangulateElement(const CurvedMeshData& mesh, int element_id);
    TriangulatedMesh triangulateMesh(const CurvedMeshData& mesh);
};
}
```

**算法**: 耳切法 (Ear Clipping)
- 从多边形顶点中逐个切出"耳朵"（凸顶点且内部不含其他顶点）
- 每次切出一个三角形，移除一个顶点
- 直到剩余3个顶点构成最后一个三角形
- 适用于任意凸多边形和凹多边形

**测试**: `tests/test_polygon_triangulator_main.cpp`
- 直边网格三角剖分，验证面积一致性
- 曲边网格三角剖分
- 更大网格的批量剖分，验证总面积
- 凹五边形剖分，验证生成 `n-2` 个三角形及面积守恒
- **测试结果**: 全部通过 ✓

---

### 7. lib/polynomial_basis - 显式多项式基函数模块

**文件**: `lib/polynomial_basis.h` / `lib/polynomial_basis.cpp`

**功能**:
- 缩放单项式基函数，一维/二维，带一二阶导数
- 多重指标按总次数递增、同次内 x 高阶优先排列
- 边上单项式求值及二维单项式→边一维单项式展开系数
- H(div) 多项式基，分解为梯度空间（$\nabla P_{k+1}/\text{常数}$）和代数补空间 $\mathcal G_k^\perp$
- 支持 $k=0,1,2,3,4$，所有参数严格检查

**主要接口**:
```cpp
namespace vem {
namespace basis {

struct Point1D;
struct Point2D;
struct Vector2D;

class BasisFunctionPloy {
public:
    explicit BasisFunctionPloy(int k);    // k = 0..4

    static long comb(int n, int k);
    static int dimPk(int k, int d);

    int getDimMonomial(int d) const;
    const std::vector<std::vector<int>>& getMultiIndex(int d) const;

    double evalMonomial2D(int nk, const Point2D& xD, double hD,
                          const Point2D& x) const;
    double evalMonomialDeriv2D(int nk, const Point2D& xD, double hD,
                               const Point2D& x, int d_num) const;
    double evalMonomialSecondDeriv2D(int nk, const Point2D& xD, double hD,
                                     const Point2D& x, int d_num) const;
    double evalMonomial1D(int nk, const Point1D& xD, double hD,
                          const Point1D& x) const;

    void getEdgeBasisCoeffs(int nk, const Point2D& xD, double hD,
                            const Point2D& point1, const Point2D& point2,
                            std::vector<double>& coefficients) const;
    double evalEdgeMonomial(int nk, const Point2D& x0, const Point2D& x1,
                            const Point2D& x) const;

    int getDimHdiv() const;
    int getDimGradSpace() const;
    int getDimComplementSpace() const;
    bool isInGradSpace(int nk) const;
    Vector2D evalHdivBasis(int nk, const Point2D& xD, double hD,
                           const Point2D& x) const;
};

}
```

**设计特点**:
- 低依赖：仅 `<vector>`, `<cmath>`, `<stdexcept>`
- 独立 `vem::basis` 命名空间，不与其他模块的类型冲突
- 梯度空间基由 $P_{k+1}$ 多重指标算法生成，补空间基由统一公式给出
- 所有参数严格验证，非法输入立即抛出异常，不静默失败
- 与未来的 `mvem_local_basis`（虚拟局部基，不可显式求值）明确区分

**测试**: `tests/test_polynomial_basis_main.cpp`
- 组合数、多项式空间维数、一维/二维多重指标顺序
- 缩放单项式值、一阶和纯二阶导数，与有限差分交叉验证
- 水平/垂直/斜向边单向式及边展开系数重构
- H(div) 梯度空间与补空间公式 $k=0..4$
- 非法阶数、维数、索引、尺度、退化边等异常路径
- 测试失败时返回非零退出码
- **测试结果**: 全部通过 ✓

---

### 8. solver/HdivMatrix - H(div) 通用投影矩阵模块

**文件**: `solver/HdivMatrix.h` / `solver/HdivMatrix.cpp`

**定位**:
- 只在 Piola 拉回后的直边计算单元上构造通用 H(div) 投影
- 不读取曲边物理网格，不携带物理映射 Jacobian 或 Darcy 渗透系数
- 类成员、public/private 方法名称、参数顺序和返回类型沿用参考工程 `MS_FVM/vem_Hdiv/HdivMatrix`

**积分参数接口**:
```cpp
HdivMatrix(const MeshReader& mesh_reader,
           int gauss_point_num = 9,     // 二维三角形积分点数
           int gauss_point_num_1d = 9,  // 一维 Gauss-Legendre 积分点数
           int k = 1);
```
- `getGaussPointNum()` 保留为兼容接口，返回二维三角形积分点数
- `getGaussPointNum2D()` 明确返回二维三角形积分点数
- `getGaussPointNum1D()` 返回一维边积分点数
- 所有三角形体积分只使用二维参数，所有线积分只使用一维参数
- 当前自动测试只覆盖旧版支持的 `k=1,2`；类保留 `k=3`，但要求二维积分使用 12 点

**自由度排列**:
- 边法向矩按“矩次数块 × 单元局部边号”排列，共 `(k+1)n_e` 个
- 内部梯度矩随后排列，共 `dim(P_k)-1` 个
- 内部补空间矩最后排列，共 `dim(P_k)-(k+1)` 个
- 单元速度自由度总数为 `(k+1)n_e + dim(P_k)-1 + dim(P_{k-1})`

**计算矩阵**:
- `D`: H(div) 多项式基在速度自由度上的取值
- `G`: H(div) 多项式无权 Gram 矩阵
- `W`, `H`, `H_star`: 分部积分及标量多项式体积分矩阵
- `H_edge`, `W_edge`, `H_edge_star`: 每条边的一维矩阵
- `B_1=-H_star H^{-1}W`，`B_2` 为逐边贡献，`B_grad=B_1+B_2`
- `B_curl`: 内部补空间矩对应块，`B=[B_grad; B_curl]`
- `L2Proj_ploy=G^{-1}B`，`L2Proj_basis=DG^{-1}B`

**复用的新架构模块**:
- `StraightMeshReader/StraightMeshData`: 直边计算网格和几何量
- `PolygonTriangulator`: 多边形单元三角剖分
- `GaussQuadrature`: 三角形体积分和二维线段积分；`HdivMatrix` 分别保存二维三角形积分点数和一维 Gauss-Legendre 积分点数
- `vem::basis::BasisFunctionPloy`: 缩放单项式与 H(div) 显式多项式基
- `core/petsc_utils`: PETSc 稠密矩阵、求逆、乘加和子块插入

**测试**: `tests/test_hdiv_matrix_main.cpp`
- `tests/reference/old_hdiv_reference_main.cpp` 独立链接旧工程实现，运行时生成参考矩阵，避免新旧同名类符号冲突和静态 golden 数据失效
- 在 `square_2x2.msh` 与 `orthogonal_2x2.msh` 的两个不同单元上对比旧版支持的 `k=1,2`；`k=1` 全矩阵逐项一致，`k=2` 对不受旧版固定 Lobatto-2 边积分影响的矩阵逐项比较，并验证新版 `BD=G` 与投影幂等
- `k=0` 单独验证最低阶尺寸、`BD=G` 和自由度投影幂等；旧实现因无关的 `k-1` 临时基构造不能作为该阶运行基准
- `k=1` 验证新旧全矩阵一致、`BD=G` 和 `Pi_dof^2=Pi_dof`；`k=2` 验证不受旧版固定低阶边积分影响的矩阵，以及新版代数恒等式
- **测试结果**: 新旧全部目标矩阵对照通过，完整 `make test` 回归通过 ✓

---

### 9. solver/DarcySolver - Darcy 专用组装与求解器（部分实现）

**文件**: `solver/DarcySolver.h` / `solver/DarcySolver.cpp`

**当前状态**:
- 已实现构造初始化：创建维数 `total_flux_dof + total_pressure_dof + 1` 的 PETSc 全零稀疏系统矩阵和全零右端向量；最后一个自由度预留给压力零均值拉格朗日乘子
- 已实现 `getMatrixG_Kappa`、`HdivMatrixG_Kappa` 和 `vemHdiv_gm_gm_Kappa`
- 已实现 `getMatrixK_ac` / `HdivMatrixK_ac`：`K_ac = Pi_poly^T G_Kappa Pi_poly`，输出维数为单元速度自由度数乘自身
- 已实现 `getMatrixK_as` / `HdivMatrixK_as`：`K_as = alpha_E (I-Pi_dof)^T(I-Pi_dof)`；`alpha_E` 对每个单元独立计算为单元平均张量 $|E|^{-1}\int_E\hat K^{-1}$ 的 Frobenius（矩阵 L2）模，不使用固定常系数
- 已实现 `getLocalRhs` / `HdivVecRhs` / `vemHdiv_source_mk`：压力块组装 $-\int_{\hat E}\hat g m_r$，其中 `computational_source_g()` 已包含 $\det J(g\circ F)$，积分时不重复乘 Jacobian；速度块和末尾约束项暂为零
- 已实现 `getVecLagrange` / `HdivVecLagrange`：物理压力零积分约束 $\int_E p=0$ 拉回为 $\int_{\hat E}\hat p\det J=0$，因此每个压力基分量组装 $\int_{\hat E}m_r\det J$；该向量随后作为完整局部矩阵最右列及最下行
- 已实现 `assembleStiffnessMatrix` 及全部辅助函数
  - `getLocalToGlobalDofs`：局部自由度 → 全局自由度编号映射（边/内部速度/压力/约束）；恒等与曲边网格通用
  - `computeAndApplyDirectionAdjustment`：偶次矩在反向边上取反，保证相邻单元通量法向连续
  - `assembleLocalToGlobal` / `assembleLocalRhsToGlobal`：局部矩阵累加至 PETSc 全局系统
  - 局部矩阵结构：$\begin{bmatrix}K_{\mathrm{ac}}+K_{\mathrm{as}}&W^T&0\\W&0&L_E\\0&L_E^T&0\end{bmatrix}$
  - 已实现法向通量强制边界降维：在计算边上组装 $\int_{\hat e}(\hat u\cdot\hat n)\hat m_\ell$，`computational_solution_u()` 已包含逆 Piola 几何因子，不重复乘边 Jacobian；构造边界值向量、修正 $F-KU_{bd}$ 并提取自由自由度子系统
  - 已实现 `solveWithDirichletBC`：使用 `core/petsc_utils` 的 GMRES+ILU 求解降维系统，再将自由解按 PETSc `IS` 恢复到完整解向量；`solve()` 负责先组装再调用该流程
  - 初始化时显式插入所有全局对角线零元以保留稀疏结构，避免 ILU 对缺失对角元报错
- `DarcyProblem::computational_kappa()` 返回 $\hat K$，组装时在每个二维高斯点求 $\hat K^{-1}$，计算 $(\hat K^{-1}m_\beta)\cdot m_\alpha$；积分外不重复乘物理映射 Jacobian
- 其余局部刚度、耦合、全局装配、边界条件和求解方法仍待实现

**新增核心矩阵**:
$$
(G_{\hat K^{-1}})_{\alpha\beta}
=\int_{\hat E}(\hat K^{-1}\boldsymbol m_\beta)
\cdot\boldsymbol m_\alpha\,d\hat x.
$$
代码接口暂命名为 `getMatrixG_Kappa(mesh_idx)` / `HdivMatrixG_Kappa(mesh_idx)`；该矩阵是通用无权 Gram 矩阵 `G` 的 Darcy 系数加权形式。恒等映射且 $K=I$ 时应退化为 `G`。

**已声明的求解层接口**:
- `G_Kappa`、一致性刚度 $K_{ac}=\Pi_{poly}^T G_{\hat K^{-1}}\Pi_{poly}$
- 稳定化刚度、完整速度局部矩阵和速度--压力耦合块
- 完整局部鞍点矩阵、单元右端项和压力零均值约束向量
- 局部到全局自由度映射、方向统一、全局矩阵/右端组装
- 法向通量边界约束、压力边界右端、线性系统求解和误差计算

**已实现验证**:
- `tests/test_darcy_solver_initial_main.cpp` 使用 `orthogonal_1x1.msh`, `k=1`
- 初始化系统维数为 `69 x 69`，系统矩阵和右端均为零
- 恒等映射且 $K=I$ 时 `max|G_Kappa-G|=0`
- `K_ac` 在 `k=1` 四边形单元上为 `11 x 11`，与独立复算的 $\Pi^T G_\kappa\Pi$ 完全一致
- `K_as` 使用逐单元积分得到的变系数尺度 $\alpha_E=\| |E|^{-1}\int_E\hat K^{-1}\|_F$，验证矩阵对称且与公式独立复算一致

**下一步**:
- 组装完整速度局部矩阵 `K_E = K_ac + K_as`，并将压力耦合块和局部右端组合成完整局部鞍点系统

---

## 待实现模块

### solver/DarcySolver - Darcy 专用求解器模块

**当前状态**: 部分实现（见第 9 节）
- ✅ 初始化与系统矩阵/右端创建
- ✅ G_Kappa 系数加权 Gram 矩阵
- ✅ K_ac 一致性刚度
- ✅ K_as 稳定化刚度
- ✅ Lagrange 约束向量
- ✅ 局部刚度矩阵组装
- ✅ 全局装配
- ✅ 边界条件处理
- ✅ 求解与误差计算

### solver/MSSolver - Maxwell-Stefan 专用求解器模块

**文件**: `solver/MSSolver.h` / `solver/MSSolver.cpp`

**当前状态**: 已实现并投入使用（Darcy 与 MS 两条线均已跑通收敛表）

- ✅ 多组分耦合 H(div) 局部矩阵组装
  - `getMatrixG_cmin`：稳定项 c\* 加权 Gram 矩阵（线性部分，所有组分相同）
  - `getMatrixAG_init`：初值模式 A_ij 加权矩阵（第一个时间步）
  - `getMatrixAG_poly`：多项式系数模式 A_ij 加权矩阵（后续 Picard 迭代）
  - 质量矩阵、散度耦合矩阵
- ✅ 全局块结构装配、法向通量 Neumann 边界
- ✅ Picard 迭代 + 时间推进（后向欧拉 / BDF2，构造参数 `use_bdf2`）
- ✅ GPU 求解路径 `solveWithDirichletBC_GPU`（GMRES + BJACOBI on CUDA）
- ✅ 逐步解输出 `saveSolutionToFile`、L2 浓度误差、H(div) 通量误差

**构造接口**:
```cpp
MSSolver(const MeshReader& mesh_reader,
         const MaxwellStefan::MSProblem& problem,
         double delta_t, double final_time,
         int picard_max_iter, double picard_tol,
         int gauss_point_num = 9, int gauss_point_num_1d = 9, int k = 1,
         bool save_all_steps = true,
         const std::string& data_subfolder = "solver_data",
         bool use_bdf2 = false);
```

继承 `HdivMatrix`，网格/自由度/高斯点/k 等属性全部来自父类。

**逐单元映射的接入点**（算例 4 的正确性全押在这里）：
- 14 处直接调用 `jacobian()` / `jacobian_det()`（Piola 变换与面积元），全部传 `mesh_idx`
- `applyNormalFluxBoundaryConditions` **按单元遍历**（`for elem`），靠 `edge_occurrence[global_edge] != 1` 跳过内部边，再把 `elem` 传进 `boundary_flux_comp`。边界边只属于唯一单元，故不存在两侧取到不同雅可比的问题
- `computeConcentrationL2Error` / `computeFluxL2Error` 同样逐单元传 `elem`

**输出约定**：`saveSolutionToFile` 路径前缀硬编码为 `data/ms_data/{subfolder}/`，文件名 `{总自由度}_{时间:.6f}.txt`，首行为总自由度数，其后每行一个值（`setprecision(12)`）。

**与 Darcy 的区别**:
- 多组分耦合（3 组分 → 块大小 ×3）
- 非线性系数（需要 Picard 迭代）
- 时间依赖（质量矩阵 / 时间步长）
- 没有压力零均值约束（MS 方程不需要 Lagrange 乘子）
- 稳定项 c\* 对应线性部分，与 Darcy 的 K=I 情况类似

---

## 后处理流水线

算例 4 逐时间步动画的完整链路（其他算例同构，只换 `problem_index` 与目录名）：

```bash
# 1. 求解：500 步，dt=0.001，T=0.5 → data/ms_data/ms_triblock_8x8/
./build/MS_triblock_frames

# 2. 导出：解 → 三角剖分 CSV（第 4 参数为算例号，第 5 为细分密度）
./build/tools/export_ms_concentration \
    mesh/data/square_8x8.msh data/ms_data/ms_triblock_8x8 \
    data/ms_triblock_8x8_plot 4 12

# 3. 画帧：三组分云图 + 曲边网格叠加 → 501 张 PNG
python3 scripts/plot_ms_concentration.py \
    data/ms_triblock_8x8_plot data/ms_triblock_8x8_frames

# 4. 合成
python3 scripts/make_gif.py \
    data/ms_triblock_8x8_frames data/gif/ms_triblock_8x8.gif
```

**`subdivisions`（第 5 参数）要按单元数往下调。** 采样点数按 `(s+1)(s+2)/2` 增长，CSV 体积同步增长：256 个单元下 `s=24` 每帧约 17MB（500 帧共 8.6G），`s=12` 每帧约 4.4MB（共 2.4G），每单元仍有约 300 个采样点，云图看不出差别。默认值保持 24，算例 1/2/3 已有的跑法不受影响。

`export_ms_concentration` 内部建了一张 **边 → 属主单元** 表：映射现在是分单元的，而边是全局遍历的，必须先确定每条边归谁才能取对雅可比。它同时导出 `curved_edges.csv` 供画图叠加网格。

`MS_triblock_frames_main.cpp` 与 `MS_half_annulus_main.cpp` 只差三处——算例号 3→4、多一句 `set_mesh`、保存子目录换名，其余配置刻意保持一字不差。这样 `data/ms_data/ms_triblock_8x8/` 与算例 2 的 `ms_conv_8x8/` 是**同网格同步长同终止时间**的两套解，可以直接逐帧对照。实测两者总自由度均为 7872，完全一致。

---

## 设计原则

1. **简洁性**: 不过度分层，一个模块尽量一个头文件+一个实现
2. **可读性**: 变量名和函数名清晰表达意图
3. **实用性**: 只实现当前需要的，不过度设计
4. **可维护**: 代码结构清晰，易于修改和扩展

---

## 编译和运行测试

### 自动识别规则

- `tests/` 中所有以 `_main.cpp` 结尾的文件都会自动识别为测试入口；新增测试无需修改 Makefile。
- `core/`、`examples/`、`mesh/`、`lib/`、`solver/` 中的 `.cpp` 会自动识别为公共模块。
- 所有测试统一使用 `mpicxx` 编译并链接 PETSc，不再区分 PETSc 与非 PETSc 测试。
- 公共模块只编译一次，目标文件和自动依赖文件保存在 `build/obj/`，并保留源码目录层级以避免同名冲突。

PETSc 路径可通过环境变量或 Make 命令行覆盖：
```bash
export PETSC_DIR=/home/lzccode/petsc
export PETSC_ARCH=arch-linux-c-opt
```

### 常用命令

```bash
make                 # 自动发现并编译全部测试程序，不运行
make test            # 编译并运行全部自动发现的测试
make list            # 查看自动发现的公共源码、测试入口和目标
make test_basis      # 运行多项式基函数测试（兼容简短别名）
make test_darcy      # 运行 Darcy 算例测试（兼容简短别名）
make test_petsc      # 运行 PETSc 工具测试（兼容简短别名）
make test_hdiv_matrix # 运行新旧 HdivMatrix 全矩阵对照测试
make test_quadrature # 也可直接使用自动生成的完整测试名
make clean           # 删除 build 目录
```

---

## 开发日志

- **2026-09-03**: 悬点网格生成模块（服务算例 5）✓
  - 新增 `core/mesh_refiner.{h,cpp}` + `tools/refine_mesh.cpp` + `tools/plot_refined_mesh.py`，详见 §2.5
  - **格式决策**：改输出 VTK + `.poly`，理由见 §2.5.2。
    ⚠️ 初稿写的理由（"reader 的 switch 只有 `case 2`/`case 3`，五边形无法写回 .msh"）是**错的**，
    当天核实后已在 §2.5.2 加勘误：该 switch 有 `case 11 → 5`、`case 12 → 6`，reader 本来就支持五边形
  - **两个实现坑**：①原始节点必须先于中点搬进输出数组，否则中点编号被整体顶掉（初版写反了，重写修正）；
    ②`0.25` 在 msh 里存成 `0.2499999999993471`，精确比较会漏掉一整列单元，`tol` 是必需参数
  - **验证三层**：C++ 自检（含边流形性检查，这是判断悬点是否被粗单元认领的关键）→ 独立 Python 几何复核
    （五边形第 5 顶点是否为边中点、重复坐标节点组数是否为 0）→ 黑白图目视
  - 7 张网格（2x2 ~ 128x128）全部通过，最大单元边数均为 5；跳过 1x1（格线 0/0.5/1，无单元整体落在 [0.25,0.75]）
  - 未触碰任何既有功能文件；Makefile 自动发现 `core/`、`tools/` 下的新文件，无需改动
  - **后续未做**：如何读取 `.vtk`/`.poly` 回 solver，以及算例 5 本身，都尚未开始

- **2026-09-03**: 悬点网格读回 + 质心扇形剖分（服务算例 5）✓
  - `mesh/straight_mesh.{h,cpp}` 新增 `read_mesh_vtk()` / `read_vtk_nodes_and_cells()`，详见 §2.5.9。
    **纯新增 172 行、0 删除**（`git diff | grep "^-"` 为空），原有 `read_mesh` 字节不变
  - `mesh/polygon_triangulator.{h,cpp}` **未改动**：曾加过质心扇形剖分，
    实测耳切法在共线悬点上本来就通过，遂整体回退，详见 §2.5.11
  - 新增 `tests/test_vtk_mesh_main.cpp`，7 张网格各 25 项通过 / 1 项失败
  - **唯一失败项**：`compute_cell_properties()` 对非四边形用顶点平均当质心，偏差恒为 0.1h。
    这是**既有行为**，非本次引入；经排查 `cell_centroid_*` 全项目只用作缩放单项式基中心，
    不影响离散解（§2.5.10）。**未修改该函数**——它与 .msh 路径共用
  - **一次自己造成的破坏**：用"`old_string` 带尾换行、`new_string` 不带"的方式做插入，
    把 `straight_mesh.cpp` 第 43-44 行合并了。立即 Read 确认、当场说明、一次 Edit 修回并核验 diff
  - **后续未做**：算例 5 本身尚未开始

- **2026-09-03**: 算例 4 逐时间步动画流水线跑通 ✓
  - 新增 `main/MS_triblock_frames_main.cpp`：算例 4 + `square_8x8.msh`，dt=0.001，T=0.5，Picard 50/1e-6，k=1，GPU 求解，逐步保存
  - 刻意照抄 `MS_half_annulus_main.cpp` 且只改三处（算例号、`set_mesh`、子目录名），使产物与算例 2 的 `ms_conv_8x8` 同网格同步长同终止时间，可逐帧对照；实测两者总自由度均为 7872
  - `tools/export_ms_concentration.cpp` 两处修改：
    - 补 `problem.set_mesh(reader)`。不补则算例 4 在 `init_problem()` 的 `requires_mesh()` 校验处抛异常；对算例 1/2/3 是空操作
    - `subdivisions` 提为第 5 个可选命令行参数，默认仍为 24。256 单元下 `s=24` 全程 8.6G，改用 `s=12` 降到 2.4G，每单元仍约 300 采样点，云图无差别
  - 产物：500 个时间步解（75M）→ 501 时刻 CSV（2.4G）→ 501 帧 PNG → `data/gif/ms_triblock_8x8.gif`（35.2 MB）
  - 云图特征：**浓度场是直的、网格是扭的**。真解定义在物理坐标上，而算例 4 外边界保持直，所以两者本就不应一致；若解跟着网格扭了反而说明 Piola 变换或雅可比接错
  - 编译 `-Wall -Wextra` 零警告

- **2026-09-03**: 新增算例 4（`TriBlockSwirlMapping` 分单元曲边网格）✓
  - `MSProblem::init_problem_4()`：构造 `TriBlockSwirlMapping` + 复用 `setup_manufactured_solution()`
  - 把算例 2 与算例 4 共用的制造解数据（真解 / 通量 / 源项 / 边界 / A_ij）提取为 `setup_manufactured_solution()`。两个算例只有映射不同，PDE 数据一字不差，误差表可直接横向对比
  - `init_problem()` 增加防呆闸门：`requires_mesh() && mesh_reader_ == nullptr` 时立刻抛异常，而不是留着 nullptr 等某个 Gauss 点崩掉、或更糟——静默算出错的雅可比
  - BDF2 收敛验证（`main/MS_triblock_bdf2_main.cpp`）：浓度 L2 阶 1.90→2.33（趋于 k+1=2）；通量阶 0.99→1.77
  - 通量在 8x8 的 0.99 是**前渐近而非掉阶**——算例 2 在对应位置也是 1.58，细化后同样爬向 ~1.8。算例 4 与算例 2 曲线形状一致，误差常数大 3–5 倍，即分单元 Q8 映射相对全局解析映射的代价
  - 单层耗时 10.2s / 61.0s / 341.1s（约 5.6×/次加密）

- **2026-09-03**: 实现 `TriBlockSwirlMapping` 两层曲边网格 ✓
  - 第 1 层解析生成器 G：真三块分区（竖直分界曲线贯穿全高 + 水平界面终止于其上形成 T 型节点）叠加全局 swirl。swirl 在整条外边界消失，故物理域严格等于 `[0,1]²`，外边界保持直，精确解 / 源项 / 边界条件语义全部原样沿用
  - 第 2 层逐单元 Q8 采样：读直边网格，取每单元 4 角点 + 4 中边点求 G，存该单元 Q8 节点表。**换网格只改采样密度，不改 G 的形状**，故任意网格文件都可扭曲
  - 相对设计文档 §7.1 的偏离：不用「全局半步网格 (2n+1)² 求值」，改为逐单元从自己的 4 个角点构造。收益是不需要推出 n、不假设结构化 n×n、不需要把 `cell_idx` 反推成 (ci,cj)
  - 相邻单元共享边重合 (R2) 按构造成立：`(x0+x1)/2` 与 `(x1+x0)/2` 在 IEEE 下逐位相同，实测 Q8 节点表差异 `0.000e+00`
  - 踩坑：`cell_nodes` 虽保证逆时针，但**起始角点在单元间循环变化**（square_2x2 实测为 右下→右上→左上→左下），局部坐标不能硬编码「节点 0 = (-1,-1)」，建表时须按角点相对单元中心的实际位置重排成规范 Q8 序
  - 参数由 `tools/triblock_sweep` 扫参标定，硬约束为 **N=2,4,8,16 四层全部 detJ > 0**。N=2 是绑定约束而非 N=4：G 层 detJ 全正不代表 Q8 插值不自交，早期只扫 `{4,8,16}` 会漏掉该失效
  - `detJ_min` 随细化单调收敛到 0.20（G 自身下界），每加密一次差距减半：0.2225 / 0.2190 / 0.2101 / 0.2050 / 0.2024 / 0.2011 / 0.2004（N=2…128）
  - 新增 `tests/test_ms_triblock_q8_main.cpp` 几何准入测试、`tools/export_triblock_mesh.cpp` 与 `tools/plot_triblock_bw.py`（黑白线，一网格一图）
  - 遗留：Q8 逐单元映射只在单元交界处 C⁰，`detJ` 跨界面跳变；混合 H(div) VEM 所需光滑度在设计文档中仅为估计，未经证明

- **2026-09-03**: 映射接口改为逐单元（新增 `cell_idx` 参数）✓
  - `IsoparametricMapping` 六个雅可比方法的第一个参数统一改为 `int cell_idx`，**不给默认值**
  - 不给默认值是刻意的：`MSSolver.cpp` 有 14 处直接调用 `jacobian()` / `jacobian_det()`，传错单元不会报错，只会让误差表看起来正常却是错的。不给默认值可让漏传的调用点在编译期断掉
  - `(xi, eta)` 语义不变，仍是全局计算域坐标；`cell_idx` 是纯增量信息，只回答「这个点属于哪个单元」，不引入任何坐标变换
  - 基类增加网格注入通道 `set_mesh()` / `has_mesh()` / `requires_mesh()` 与 `on_mesh_set()` 钩子。持有方式仿 `HdivMatrix`：只存指针，不拥有所有权。`set_mesh` 保持非虚，指针记账不在每个派生类重复
  - `MSPdeData` 计算域回调（`*_comp`）凡内部用映射的一律加 `cell_idx`，参数顺序统一为「组分号 → 单元号 → 坐标 → 时间」；物理域回调不加。例外：`initial_concentration` 虽为物理域但内部要逆映射，故带 `cell_idx`；`evaluate_A_comp_poly` 只用多项式系数，故不带
  - `MSProblem::invert_mapping` 增加 `cell_idx`：映射分单元后全局逆映射不再唯一定义，必须指明在哪个单元内求逆
  - 同步改完 `MSSolver.cpp` 全部调用点与相关 test/tool，编译通过
  - 三个既有映射（`SinPerturbationMapping` / `HalfAnnulusMapping` / `IdentityMapping`）收下 `cell_idx` 后直接丢弃，算例 1/2/3 数值行为与改动前逐位相同
  - 遗留未修：`MSProblem::invert_mapping` 牛顿迭代跑满 50 步后无收敛断言，直接返回手上的 `(ξ,η)`；同处注释「当前三个映射与单元无关」已过时

- **2026-07-24**: 实现 `examples/ms_problem` Maxwell-Stefan 算例模块 ✓
  - MS 专用曲边映射：`x = ξ + 0.1 sin(2πη + π/3)`, `y = η + 0.2 sin(2πξ + π/4)`
  - 扩散参数：c₁₂=0.1, c₁₃=0.2, c₂₃=2.0, c*=0.1
  - 三组分正弦脉动精确解 + 通量 + 源项
  - 4 种非线性扩散矩阵评估接口
    - `evaluate_diffusion_matrix(u, bar_A)`：完整矩阵 bar_A = c*I + A(u)
    - `evaluate_nonlinear_part(u, A)`：纯非线性部分 A(u)
    - `element_diffusion_coeff(i, j, u_avg)`：单元均值输入（对应 MS_FVM mid_A_coeff）
    - `polynomial_diffusion_coeff(i, j, u_ploy_coeff, xD_x, xD_y, hD, x, y)`：单项式系数输入（对应 MS_FVM poly_A_coeff）
  - 关键设计：纯非线性部分 A(u) **不含 c***，稳定项在组装时单独加（与 MS_FVM 完全一致）
  - 计算域拉回：c 直接映射，J 用 Piola 变换，f 乘 detJ，∇c 用链式法则
  - 8 项全面验证测试全部通过（几何、质量守恒、M-矩阵性质、本构关系、多项式系数、Piola 变换、源项）

- **2026-07-20**: 实现 `DarcySolver::HdivMatrixK_as` 变系数稳定化矩阵 ✓
  - 按 $K_{as}=\alpha_E(I-\Pi_{dof})^T(I-\Pi_{dof})$ 组装投影核空间稳定化
  - 稳定化系数不是旧常系数，而是逐单元计算：先积分 $\hat K^{-1}$ 得到单元平均张量，再取其 Frobenius（矩阵 L2）模
  - 平均值分母采用与系数积分相同的三角剖分实际面积，保证分子和测度一致
  - 增加投影矩阵空指针、方阵维数及单元速度自由度相容性检查
  - 恒等映射测试中 $K=I$ 时系数自然退化为 $\sqrt2$（仅为该输入的计算结果，不是硬编码）；公式复算误差 `7.11e-15`，曲边情况下稳定化矩阵保持对称 ✓

- **2026-07-20**: 实现 `DarcySolver::HdivMatrixK_ac` 一致性局部刚度 ✓
  - 公共接口 `getMatrixK_ac(G_Kappa, L2Proj_ploy)` 转发到私有组装方法
  - 按 $K_{ac}=\Pi_{poly}^T G_{\hat K^{-1}}\Pi_{poly}$ 依次完成加权乘法、投影转置和最终乘法
  - 增加空矩阵和维数相容性检查，拒绝不满足 $G_\kappa\in\mathbb R^{2N_k\times2N_k}$、$\Pi\in\mathbb R^{2N_k\times n_d}$ 的输入
  - 测试中 `K_ac` 为 `11 x 11`，公式独立复算误差为 0；恒等和曲边映射下对称误差均约为机器精度 ✓

- **2026-07-20**: 实现 `DarcySolver` 含 $\det J$ 的 Lagrange 约束向量 ✓
  - 物理压力零积分约束 $\int_E p=0$ 通过 Piola 拉回为 $\int_{\hat E}\hat p\det J=0$
  - 每个单元局部 Lagrange 向量的压力分量组装 $\int_{\hat E}m_r\det J$，速度分量和末尾乘子位置为零
  - 同时复用旧的 `vemHdiv_mk` 用于压力零积分约束（直边无曲边，与旧代码一致）
  - 测试结果：恒等映射下 Lagrange 常数分量与积分面积一致，曲边映射下因 $\det J\neq1$ 而发生变化，速度块和乘子位置为零 ✓

- **2026-07-19**: 实现 `DarcySolver` 初始化与 `G_Kappa` 系数加权 Gram 矩阵 ✓
  - 初始化创建 `N_flux + N_pressure + 1` 维 PETSc 全零稀疏矩阵和全零右端，最后一个自由度预留给压力零均值约束
  - `G_Kappa` 在直边计算单元的三角形高斯点调用 `DarcyProblem::computational_kappa()`，逐点求 $\hat K^{-1}$ 后组装
  - 严格采用 $(\hat K^{-1}m_\beta)\cdot m_\alpha$，不在积分外重复乘曲边映射 Jacobian
  - 新增 `tests/test_darcy_solver_initial_main.cpp`：恒等映射时验证 `G_Kappa=G`，曲边映射时验证加权矩阵发生变化且保持对称
  - 测试结果：系统维数 `69 x 69`，恒等映射差值为 0，曲边 `G_Kappa` 对称误差 `1.39e-17` ✓

- **2026-07-19**: 建立 `solver/DarcySolver` 求解器接口骨架 ◇
  - 参考旧工程 `DarcySolver` 的继承关系、局部/全局组装和求解职责，但暂不迁移具体实现
  - 新增系数加权 Gram 矩阵 `G_Kappa` 接口，对应 $G_{\hat K^{-1}}$，供曲边映射后的变系数一致性刚度使用
  - 声明一致性刚度、稳定化、速度--压力耦合、局部鞍点矩阵、右端项、全局装配、边界条件、求解与误差接口
  - 构造参数沿用 `HdivMatrix` 的二维/一维独立积分点数，并保存 `DarcyProblem`
  - 原接口骨架中的初始化与 `G_Kappa` 已在 `solver/DarcySolver.cpp` 实现；其余接口继续按数学步骤逐项实现

- **2026-07-18**: 实现 `solver/HdivMatrix` 直边计算域 H(div) 投影模块 ✓
  - 保留参考工程 `MS_FVM/vem_Hdiv/HdivMatrix` 的类成员、public/private 方法名称、参数顺序和矩阵接口
  - 底层依赖重构为新架构的 `StraightMeshReader`、`PolygonTriangulator`、`GaussQuadrature`、`vem::basis::BasisFunctionPloy` 和 `core/petsc_utils`
  - 实现 `D/G/W/H/H_star`、三类边矩阵、`B_1/B_2/B_grad/B_curl/B` 及两种 L2 投影矩阵
  - 投影只作用于 Piola 拉回后的直边计算单元，不携带物理曲边 Jacobian 或 Darcy 系数
  - 新增独立旧实现参考导出器；`make test_hdiv_matrix` 每次运行旧代码实时生成结果，再由新版逐矩阵比较，避免同名类冲突和手工 golden 数据
  - 对 `square_2x2`、`orthogonal_2x2` 的 `k=1,2` 验证目标矩阵、自由度、节点顺序和边法向；旧版不支持 `k=3`
  - 新版将二维三角形积分点数与一维 Gauss-Legendre 积分点数拆分为独立构造参数，默认均为 9；所有体积分与边积分分别使用对应参数
  - `k=1` 与旧版全矩阵逐项一致；`k=2` 不复制旧版固定 Lobatto-2 的边自由度积分，而验证新版 `BD=G` 与投影幂等
  - 新版额外支持 `k=0`，验证 `BD=G` 和自由度投影幂等；旧实现的无关 `k-1` 临时基构造使其无法作为最低阶运行基准
  - 公开接口运行样例在 `orthogonal_2x2, k=1` 得到 `D=11x6`、`B=6x11`，`max|BD-G|=5.55e-17`；非法 `k=5` 返回明确错误
  - 测试结果：`make test_hdiv_matrix` 与完整 `make test` 全部通过 ✓

- **2026-07-17**: Makefile 改为自动发现测试与公共模块 ✓
  - `tests/*_main.cpp` 自动识别为测试入口，所有测试文件统一使用 `_main.cpp` 后缀
  - 自动扫描 `core/`、`examples/`、`mesh/`、`lib/` 下所有公共 `.cpp`
  - 公共模块仅编译一次，并通过 `-MMD -MP` 自动追踪头文件依赖
  - `build/obj/` 保留源码目录层级，避免不同目录同名目标文件冲突
  - 全部测试统一使用 `mpicxx` 和 PETSc 链接，不再区分 PETSc 测试
  - 保留 `make` 只编译、`make test` 全部运行，并兼容 `test_basis`、`test_darcy` 等简短目标
  - 新增 `make list` 显示自动发现结果；今后添加测试不需要再修改 Makefile

- **2026-07-17**: 实现 `lib/polynomial_basis` 显式多项式基函数模块 ✓
  - 缩放单项式基函数（一维/二维）及一阶、纯二阶导数
  - 多重指标按总次数、同次内 x 高阶优先排列
  - 边上单项式求值及二维单项式至边一维单项式展开系数
  - H(div) 多项式基：梯度空间（$\nabla P_{k+1}$）与代数补空间 $\mathcal G_k^\perp$
  - 梯度空间由 $P_{k+1}$ 多重指标算法生成，补空间由统一公式给出
  - 支持 $k=0..4$，严格参数检查，非法输入抛出异常
  - 独立 `vem::basis` 命名空间，低依赖，纯 C++11
  - 编写 `tests/test_polynomial_basis_main.cpp` 测试（1117 项检查）
  - 验证数值、导数（含有限差分交叉验证）、边展开重构、H(div) 公式、异常路径
  - 更新 Makefile 增加 `test_basis` 并纳入 `compile`/`test` 目标
  - 更新 `ARCHITECTURE.md` 和 `混合虚拟元_darcy.md` 将其与未来 `mvem_local_basis` 区分
  - 测试结果：`make test_basis` 与 `make test` 全部通过，编译无警告 ✓

- **2026-07-17**: 增加直边计算区域解析数据及映射证明测试 ✓
  - 在 `DarcyProblem` 增加 `computational_solution_p/u`、`computational_source_g` 和 `computational_kappa` 接口
  - 计算区域压力采用复合拉回 `p̂=p∘F`
  - 计算区域速度采用逆 Piola 变换 `û=det(J)J⁻¹(u∘F)`
  - 源项采用 `ĝ=det(J)(g∘F)`，渗透张量采用 `K̂=det(J)J⁻¹(K∘F)J⁻ᵀ`
  - 所有计算区域数据在线复用当前映射和物理解析函数，不维护重复解析表达式
  - 未初始化访问会抛出明确异常
  - 重写 `tests/test_darcy_problem_main.cpp` 的 `main` 测试，失败时返回非零退出码
  - 逐点验证物理/计算数据变换、Piola 往返恢复、`K̂` 对称正定及恒等映射退化
  - 用独立中心差分验证 `û=-K̂∇̂p̂` 和 `div̂(û)=ĝ`
  - 修正 `混合虚拟元_darcy.md` 中算例1源项误写为零的问题，并补充计算区域变换推导
  - 测试结果：`make test_darcy` 与 `make test` 全部通过，编译无警告 ✓

- **2026-07-17**: 实现 `mesh/polygon_triangulator` 多边形三角剖分模块 ✓
  - 使用耳切法将任意多边形单元划分为三角形
  - 同时支持 `StraightMeshData` 和 `CurvedMeshData`
  - 支持单元剖分和整个网格批量剖分
  - 保存三角形顶点、全局节点编号、面积和单元总面积
  - 增加节点编号及单元索引范围检查
  - 编写 `tests/test_polygon_triangulator_main.cpp`
  - 验证直边网格、映射后网格、批量网格和凹五边形剖分
  - 更新 Makefile，增加 `test_triangulator`，并将其加入完整测试目标
  - 测试结果：全部通过 ✓

- **2026-07-17**: 调整 Makefile 默认行为
  - `make` 和 `make compile` 只编译，不运行测试
  - `make test` 编译并运行全部非 PETSc 测试

- **2026-07-15**: 实现 `mesh/curved_mesh` 曲边网格模块 ✓
  - 实现 CurvedMeshData 数据结构（简化版）
  - 实现 CurvedMeshGenerator 生成器
  - 支持从直边网格和映射生成曲边网格
  - 复用直边网格的拓扑信息
  - 在 straight_mesh 添加 to_curved_mesh_data() 方法
  - 编写 `tests/test_curved_mesh_main.cpp` 测试程序
  - 更新 Makefile 添加 test_curved_mesh 目标

- **2026-07-15**: 补充网格文件到 `mesh/data/` ✓
  - 复制 square_1x1 ~ square_128x128
  - 复制 orthogonal_1x1 ~ orthogonal_128x128
  - 更新直边网格测试，添加节点坐标验证

- **2026-07-15**: 实现 `mesh/straight_mesh` 直边网格模块 ✓
  - 复制测试网格文件到 `mesh/data/`
  - 实现 StraightMeshData 数据结构
  - 实现 StraightMeshReader 读取器
  - 支持读取 Gmsh .msh 格式
  - 单元属性计算、边拓扑生成、边界信息计算
  - 相邻单元查找功能
  - 编写 `tests/test_straight_mesh_main.cpp` 测试程序
  - 更新 Makefile 添加 test_mesh 目标

- **2026-07-14**: 实现 `core/gauss_quadrature` 高斯积分模块，完整测试通过 ✓
  - 编写 `tests/test_quadrature_main.cpp` 测试程序
  - 编写 `Makefile` 简化编译和测试流程

- **2026-07-14**: 实现 `core/petsc_utils` PETSc 工具模块，完整测试通过 ✓
  - 智能指针资源管理
  - 多种求解器支持
  - 矩阵运算和分块填充
  - 编写 `tests/test_petsc_utils_main.cpp` 测试程序

- **2026-07-14**: 实现 `examples/darcy_problem` Darcy 算例模块，完整测试通过 ✓
  - 曲边映射基类和派生类（正弦扰动、恒等）
  - 雅可比矩阵/行列式/逆/转置/转置逆计算
  - Darcy PDE 数据和算例管理
  - 编写 `tests/test_darcy_problem_main.cpp` 测试程序

- **2026-07-14**: Makefile 优化 ✓
  - 编译产物单独放在 `build/` 目录
  - 使用管道 `| $(BUILDDIR)` 自动创建目录
  - 清理改为 `rm -rf build`

- **2026-07-14**: 命名空间修正 ✓
  - 算例模块命名空间从 `vem` 改为 `Darcy`
  - 原因：该模块是 Darcy 方程专用的算例
  - `IsoparametricMapping` 基类，子类继承实现具体映射
