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
│   └── petsc_utils.cpp          # PETSc 工具实现
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
│   ├── straight_mesh.h          # 直边网格头文件
│   ├── straight_mesh.cpp        # 直边网格实现
│   ├── curved_mesh.h            # 曲边网格头文件
│   ├── curved_mesh.cpp          # 曲边网格实现
│   ├── polygon_triangulator.h   # 多边形三角剖分头文件
│   └── polygon_triangulator.cpp # 多边形三角剖分实现
├── tests/                       # 测试程序
│   ├── test_quadrature_main.cpp      # 积分模块测试
│   ├── test_petsc_utils_main.cpp     # PETSc 工具测试
│   ├── test_polynomial_basis_main.cpp # 多项式基函数测试
│   ├── test_darcy_problem_main.cpp   # Darcy 算例测试
│   └── test_ms_problem_main.cpp      # Maxwell-Stefan 算例测试
│   ├── test_straight_mesh_main.cpp   # 直边网格测试
│   ├── test_curved_mesh_main.cpp     # 曲边网格测试
│   └── test_polygon_triangulator_main.cpp # 多边形三角剖分测试
├── solver/                      # 求解器模块
│   ├── HdivMatrix.h             # H(div) 通用投影矩阵头文件（共享）
│   ├── HdivMatrix.cpp           # H(div) 通用投影矩阵实现（共享）
│   ├── DarcySolver.h            # Darcy 专用组装与求解器
│   ├── DarcySolver.cpp          # Darcy 求解器实现
│   ├── MsSolver.h               # Maxwell-Stefan 专用组装与求解器（待实现）
│   └── MsSolver.cpp             # MS 求解器实现（待实现）
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
- MS 专用曲边映射（`x = ξ + 0.1 sin(2πη + π/3)`, `y = η + 0.2 sin(2πξ + π/4)`）
- 二元扩散系数与矩阵分解：`c_ij`, `c* = min(c_ij)`, `bar_c_ij = c_ij - c*`
- 三组分精确解（正弦脉动时间依赖）
- 本构关系通量 `J = -bar_A(c)^{-1} ∇c`
- 非线性扩散矩阵评估（3 种接口，与 MS_FVM 完全兼容）
- 计算域拉回函数（Piola 变换、detJ 加权、链式法则）

**主要接口**:
```cpp
namespace MaxwellStefan {

// 曲边映射基类（与 Darcy 共用接口设计）
class IsoparametricMapping {
    virtual Point2D physical_coords(double xi, double eta) const = 0;
    virtual Tensor2D jacobian(double xi, double eta) const = 0;
    virtual double jacobian_det(double xi, double eta) const = 0;
    virtual Tensor2D jacobian_inv(double xi, double eta) const = 0;
    virtual Tensor2D jacobian_transpose(double xi, double eta) const = 0;
    virtual Tensor2D jacobian_transpose_inv(double xi, double eta) const = 0;
};

// MS 专用曲边映射
class MS_Mapping : public IsoparametricMapping { /* ... */ };

// MS PDE 数据结构
struct MSPdeData {
    int n_components;
    std::vector<std::vector<double>> c_ij;     // 二元扩散系数倒数
    double c_star;                              // 最小值 c* = min(c_ij)
    std::vector<std::vector<double>> bar_c_ij;  // bar_c_ij = c_ij - c*

    // 物理域函数
    std::function<double(int i, double x, double y)> initial_concentration;
    std::function<double(int i, double x, double y, double t)> solution_c;
    std::function<Vector2D(int i, double x, double y, double t)> solution_J;
    std::function<Vector2D(int i, double x, double y, double t)> grad_c;
    std::function<double(int i, double x, double y, double t)> source_f;
    std::function<double(int i, double x, double y, double nx, double ny, double t)> boundary_flux;

    // 非线性扩散矩阵评估（用于 Picard 迭代）
    // 1. 完整矩阵 bar_A(u) = c*I + A(u)
    std::function<void(const std::vector<double>& u,
                       std::vector<std::vector<double>>& bar_A)>
        evaluate_diffusion_matrix;
    // 2. 纯非线性部分 A(u)（退化，列和=0）
    std::function<void(const std::vector<double>& u,
                       std::vector<std::vector<double>>& A)>
        evaluate_nonlinear_part;
    // 3. 单元均值输入（对应 MS_FVM mid_A_coeff）
    //    返回纯非线性部分 A(u)，不含 c*（稳定项单独加）
    std::function<double(int i, int j, const std::vector<double>& u_avg)>
        element_diffusion_coeff;
    // 4. 单项式系数输入（对应 MS_FVM poly_A_coeff）
    //    u_ploy_coeff 布局：[comp0_monom0, ..., comp1_monom0, ...]
    //    返回纯非线性部分 A(u)，不含 c*
    std::function<double(int i, int j, const std::vector<double>& u_ploy_coeff,
                         double xD_x, double xD_y, double hD, double x, double y)>
        polynomial_diffusion_coeff;
};

// 算例管理类
class MSProblem {
public:
    void set_problem_index(int index);
    void set_time(double t);
    bool init_problem();
    const MSPdeData& get_pde_data() const;
    const IsoparametricMapping& get_mapping() const;

    // 计算域拉回函数
    double computational_solution_c(int i, double xi, double eta) const;
    Vector2D computational_solution_J(int i, double xi, double eta) const;  // Piola
    double computational_source_f(int i, double xi, double eta) const;     // detJ加权
    Vector2D computational_grad_c(int i, double xi, double eta) const;     // 链式法则
};

}
```

**设计特点**:
- **与 MS_FVM 接口完全一致**：`element_diffusion_coeff` 对应 `mid_A_coeff`，`polynomial_diffusion_coeff` 对应 `poly_A_coeff`
- **非线性系数分解一致**：纯非线性部分 A(u) 不含 c*，稳定项 c* 在 Hdiv 组装时单独加到对角线
- **单项式系数存储布局一致**：所有组分平铺，`u_ploy_coeff[comp * n_monomial + m]`
- **曲边映射与 Darcy 共用基类设计**，方便后续复用组装代码
- **计算域数据全部由物理域在线拉回**，不维护两套公式
- **Piola 变换用于通量**，源项乘 detJ，梯度用链式法则 J^{-T}∇_ξ

**算例 1：三组分正弦脉动扩散**
- c₁ = 0.25 sin(2πx) sin(8πt) + 0.25
- c₂ = 0.25 sin(3πy) sin(6πt) + 0.25
- c₃ = 1 - c₁ - c₂
- 扩散参数：c₁₂=0.1, c₁₃=0.2, c₂₃=2.0, c*=0.1

**验证测试**: `examples/test_ms_problem.cpp`（8 项全通过 ✓）
1. 曲边映射几何正确性（雅可比逆、detJ>0、转置逆）
2. 质量守恒 c₁+c₂+c₃=1
3. 初始条件正确性（t=0 时 c₁=c₂=0.25, c₃=0.5）
4. 非线性扩散矩阵 M-矩阵性质
   - 对角正、非对角负
   - A(u) 列和=0（退化性质）
   - bar_A(u) 列和=c*（正则化）
   - element_diffusion_coeff 与 A(u) 一致、不含 c*
5. 本构关系 ∇c + bar_A(c)·J = 0
6. 多项式系数评估（单项式还原 + A(u) 评估 + 与单元均值一致）
7. Piola 变换正向/逆向一致性
8. 计算域源项 detJ 加权正确性

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

### solver/MsSolver - Maxwell-Stefan 专用求解器模块

**功能规划**:
- 多组分耦合的 H(div) 局部矩阵组装
  - 稳定项 c* 对应的 G 加权 Gram 矩阵（线性部分）
  - 非线性部分 A(u) 对应的加权 Gram 矩阵（Picard 迭代中用当前解更新）
  - 质量矩阵（时间离散 ∂c/∂t 项）
  - 散度耦合矩阵（速度-浓度耦合）
- 全局系统组装（块结构，多组分耦合）
- Picard 迭代循环
  - 第一步用初值函数计算初始扩散系数
  - 后续时间步用上一时间步的单项式系数还原浓度，计算 A(u)
  - 求解线性系统，更新浓度单项式系数
  - 迭代直到收敛
- 时间推进（向后欧拉 / 可能扩展 Crank-Nicolson）
- 边界条件处理（Neumann 法向通量）
- 后处理与误差计算（L2 浓度误差、H(div) 通量误差）

**与 Darcy 的区别**:
- 多组分耦合（3 组分 → 块大小 ×3）
- 非线性系数（需要 Picard 迭代）
- 时间依赖（质量矩阵 / 时间步长）
- 没有压力零均值约束（MS 方程不需要 Lagrange 乘子）
- 稳定项 c* 对应线性部分，与 Darcy 的 K=I 情况类似

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
