/**
 * CPU Maxwell–Stefan 算例与映射接口：包含正弦、半圆环和逐单元 Q8 几何。
 */
#ifndef CURVEDVEM_MS_PROBLEM_H
#define CURVEDVEM_MS_PROBLEM_H

#include "../mesh/straight_mesh.h"

#include <cmath>
#include <vector>
#include <functional>
#include <stdexcept>
#include <memory>
#include <string>

namespace MaxwellStefan {

// ============================================================================
// 基础向量/矩阵类型定义（与 Darcy 保持一致）
// ============================================================================

struct Point2D {
    double x, y;
    Point2D() : x(0.0), y(0.0) {}
    Point2D(double x_, double y_) : x(x_), y(y_) {}
};

struct Vector2D {
    double x, y;
    Vector2D() : x(0.0), y(0.0) {}
    Vector2D(double x_, double y_) : x(x_), y(y_) {}
};

struct Tensor2D {
    double xx, xy, yx, yy;
    Tensor2D() : xx(1.0), xy(0.0), yx(0.0), yy(1.0) {}
    Tensor2D(double xx_, double xy_, double yx_, double yy_)
        : xx(xx_), xy(xy_), yx(yx_), yy(yy_) {}
};

// ============================================================================
// 曲边映射基类（分单元映射）
//
// 坐标约定：(xi, eta) 始终是**全局计算域坐标**，即直边参考网格上的物理坐标。
//   积分点由 triangulateElement(mesh_data, cell_idx) 生成，本来就落在
//   cell_idx 号单元内，并直接喂给以单元质心/直径缩放的单项式基。
//   所以 cell_idx 是纯增量信息，不引入任何坐标变换：
//   它只回答「这个点属于哪个单元」，从而让映射可以逐单元取不同的公式。
//
// cell_idx 一律不给默认值：分单元映射下「静默取到错误单元的雅可比」会让误差表
//   看起来正常却是错的，必须让漏传的调用点在编译期就报错。
//
// 现有三个派生类（SinPerturbationMapping / HalfAnnulusMapping /
//   IdentityMapping）都是全局连续的单一解析映射，收下 cell_idx 后直接丢弃，
//   数值行为与加参数前逐位相同。
// ============================================================================

class IsoparametricMapping {
public:
    IsoparametricMapping() = default;
    virtual ~IsoparametricMapping() = default;

    virtual Point2D physical_coords(int cell_idx,
                                    double xi, double eta) const = 0;
    virtual Tensor2D jacobian(int cell_idx,
                              double xi, double eta) const = 0;
    virtual double jacobian_det(int cell_idx,
                                double xi, double eta) const = 0;
    virtual Tensor2D jacobian_inv(int cell_idx,
                                  double xi, double eta) const = 0;
    virtual Tensor2D jacobian_transpose(int cell_idx,
                                        double xi, double eta) const = 0;
    virtual Tensor2D jacobian_transpose_inv(int cell_idx,
                                            double xi, double eta) const = 0;

    // ------------------------------------------------------------------
    // 网格访问基础
    //
    // 分单元映射要靠 cell_idx 去网格里取该单元的节点坐标才能算出雅可比，
    // 所以映射本身需要持有网格。持有方式仿 HdivMatrix（HdivMatrix.h:17-18,
    // 27, 80-81）：只存指针，不拥有所有权，生命周期由调用方保证。
    //
    // 注意：当前三个派生类与单元无关，**没有任何代码读取这两个指针**。
    // 它们是为后续分单元映射（如 TriBlockSwirlMapping）预留的注入通道。
    // 不调用 set_mesh 时两者保持 nullptr，一切照旧。
    // ------------------------------------------------------------------
    void set_mesh(const vem::StraightMeshReader& reader) {
        mesh_reader_ = &reader;
        mesh_data_ = &reader.get_mesh_data();
        on_mesh_set();
    }
    const vem::StraightMeshReader* mesh_reader() const { return mesh_reader_; }
    const vem::StraightMeshData* mesh_data() const { return mesh_data_; }
    bool has_mesh() const { return mesh_data_ != nullptr; }

    // 本映射是否**必须**有网格才能工作。
    // 分单元映射（TriBlockSwirlMapping）返回 true，MSProblem::init_problem()
    // 会据此强制校验：漏调 set_mesh 时立刻抛异常，而不是留着 nullptr
    // 等到某个 Gauss 点上崩掉、或者更糟——静默算出错的雅可比。
    virtual bool requires_mesh() const { return false; }

protected:
    // 网格注入完成后的回调。需要预处理网格的派生类在这里建表。
    // set_mesh 保持非虚（指针记账不必在每个派生类里重复一遍），
    // 只把「网格到手了」这个时机通过钩子交出去。
    virtual void on_mesh_set() {}

    const vem::StraightMeshReader* mesh_reader_ = nullptr;
    const vem::StraightMeshData* mesh_data_ = nullptr;
};

// ============================================================================
// 正弦扰动的曲边映射（与 Darcy 相同形式，默认 eps=0.05）
//   x = ξ + eps * sin(2π η)
//   y = η + eps * sin(2π ξ)
// 全局连续的单一解析映射，与单元无关，故忽略 cell_idx。
// ============================================================================

class SinPerturbationMapping : public IsoparametricMapping {
public:
    SinPerturbationMapping(double eps_ = 0.05) : eps(eps_) {}

    Point2D physical_coords(int cell_idx,
                            double xi, double eta) const override {
        (void)cell_idx;
        double x = xi + eps * std::sin(2.0 * M_PI * eta);
        double y = eta + eps * std::sin(2.0 * M_PI * xi);
        return Point2D(x, y);
    }

    Tensor2D jacobian(int cell_idx, double xi, double eta) const override {
        (void)cell_idx;
        double dxdxi = 1.0;
        double dxdeta = eps * 2.0 * M_PI * std::cos(2.0 * M_PI * eta);
        double dydxi = eps * 2.0 * M_PI * std::cos(2.0 * M_PI * xi);
        double dydeta = 1.0;
        return Tensor2D(dxdxi, dxdeta, dydxi, dydeta);
    }

    double jacobian_det(int cell_idx, double xi, double eta) const override {
        (void)cell_idx;
        double dxdxi = 1.0;
        double dxdeta = eps * 2.0 * M_PI * std::cos(2.0 * M_PI * eta);
        double dydxi = eps * 2.0 * M_PI * std::cos(2.0 * M_PI * xi);
        double dydeta = 1.0;
        return dxdxi * dydeta - dxdeta * dydxi;
    }

    Tensor2D jacobian_inv(int cell_idx, double xi, double eta) const override {
        double detJ = jacobian_det(cell_idx, xi, eta);
        if (std::abs(detJ) < 1e-14) {
            throw std::runtime_error("Jacobian is singular in MS mapping!");
        }
        Tensor2D J = jacobian(cell_idx, xi, eta);
        double invdet = 1.0 / detJ;
        return Tensor2D(
            J.yy * invdet,  -J.xy * invdet,
            -J.yx * invdet, J.xx * invdet
        );
    }

    Tensor2D jacobian_transpose(int cell_idx,
                                double xi, double eta) const override {
        Tensor2D J = jacobian(cell_idx, xi, eta);
        return Tensor2D(J.xx, J.yx, J.xy, J.yy);
    }

    Tensor2D jacobian_transpose_inv(int cell_idx,
                                    double xi, double eta) const override {
        Tensor2D invJ = jacobian_inv(cell_idx, xi, eta);
        return Tensor2D(invJ.xx, invJ.yx, invJ.xy, invJ.yy);
    }

private:
    double eps;
};

// ============================================================================
// 半圆环映射
//   r = ξ + 0.5,  θ = π(η - 0.5)
//   x = r cosθ,  y = r sinθ
//   物理域：右半圆环，内半径 0.5，外半径 1.5，θ ∈ [-π/2, π/2]
// 全局连续的单一解析映射，与单元无关，故忽略 cell_idx。
// ============================================================================

class HalfAnnulusMapping : public IsoparametricMapping {
public:
    HalfAnnulusMapping() = default;

    Point2D physical_coords(int cell_idx,
                            double xi, double eta) const override {
        (void)cell_idx;
        double r = xi + 0.5;
        double theta = M_PI * (eta - 0.5);
        double x = r * std::cos(theta);
        double y = r * std::sin(theta);
        return Point2D(x, y);
    }

    Tensor2D jacobian(int cell_idx, double xi, double eta) const override {
        (void)cell_idx;
        double r = xi + 0.5;
        double theta = M_PI * (eta - 0.5);
        double cos_t = std::cos(theta);
        double sin_t = std::sin(theta);
        // ∂x/∂ξ = cosθ,  ∂x/∂η = -r π sinθ
        // ∂y/∂ξ = sinθ,  ∂y/∂η =  r π cosθ
        double dxdxi = cos_t;
        double dxdeta = -r * M_PI * sin_t;
        double dydxi = sin_t;
        double dydeta = r * M_PI * cos_t;
        return Tensor2D(dxdxi, dxdeta, dydxi, dydeta);
    }

    double jacobian_det(int cell_idx, double xi, double eta) const override {
        (void)cell_idx; (void)eta;
        double r = xi + 0.5;
        // detJ = cosθ * r π cosθ - (-r π sinθ) * sinθ = r π (cos²θ + sin²θ) = r π
        return r * M_PI;
    }

    Tensor2D jacobian_inv(int cell_idx, double xi, double eta) const override {
        (void)cell_idx;
        double r = xi + 0.5;
        double theta = M_PI * (eta - 0.5);
        double cos_t = std::cos(theta);
        double sin_t = std::sin(theta);
        double detJ = r * M_PI;
        // J = [cosθ,  -rπ sinθ; sinθ,  rπ cosθ]
        // inv(J) = (1/rπ) * [rπ cosθ,  rπ sinθ; -sinθ,  cosθ]
        //         = [cosθ,  sinθ; -sinθ/(rπ),  cosθ/(rπ)]
        return Tensor2D(
            cos_t,              sin_t,
            -sin_t / detJ,      cos_t / detJ
        );
    }

    Tensor2D jacobian_transpose(int cell_idx,
                                double xi, double eta) const override {
        Tensor2D J = jacobian(cell_idx, xi, eta);
        return Tensor2D(J.xx, J.yx, J.xy, J.yy);
    }

    Tensor2D jacobian_transpose_inv(int cell_idx,
                                    double xi, double eta) const override {
        Tensor2D invJ = jacobian_inv(cell_idx, xi, eta);
        return Tensor2D(invJ.xx, invJ.yx, invJ.xy, invJ.yy);
    }
};

// ============================================================================
// 恒等映射（直边网格）
// 与单元无关，故忽略 cell_idx。
// ============================================================================

class IdentityMapping : public IsoparametricMapping {
public:
    IdentityMapping() = default;

    Point2D physical_coords(int cell_idx,
                            double xi, double eta) const override {
        (void)cell_idx;
        return Point2D(xi, eta);
    }

    Tensor2D jacobian(int cell_idx, double xi, double eta) const override {
        (void)cell_idx; (void)xi; (void)eta;
        return Tensor2D(1.0, 0.0, 0.0, 1.0);
    }

    double jacobian_det(int cell_idx, double xi, double eta) const override {
        (void)cell_idx; (void)xi; (void)eta;
        return 1.0;
    }

    Tensor2D jacobian_inv(int cell_idx, double xi, double eta) const override {
        (void)cell_idx; (void)xi; (void)eta;
        return Tensor2D(1.0, 0.0, 0.0, 1.0);
    }

    Tensor2D jacobian_transpose(int cell_idx,
                                double xi, double eta) const override {
        (void)cell_idx; (void)xi; (void)eta;
        return Tensor2D(1.0, 0.0, 0.0, 1.0);
    }

    Tensor2D jacobian_transpose_inv(int cell_idx,
                                    double xi, double eta) const override {
        (void)cell_idx; (void)xi; (void)eta;
        return Tensor2D(1.0, 0.0, 0.0, 1.0);
    }
};

// ============================================================================
// TriBlockSwirl 逐单元 Q8 等参映射（算例 4）
//
// 设计出处：vem/mesh/TriBlockSwirl/设计文档.md（第 2 版）。
// 参考实现：tri_block_swirl_gen.py（生成器 G = T∘S）、tri_block_swirl_q8.py。
//
// 与前三个映射的本质区别：**每个单元一个雅可比**。单元内的映射由该单元的
// 8 个 Q8 节点决定，单元内逐点不同，跨单元边界允许跳变——这正是本算例要
// 测试的特征，而不是缺陷（设计文档 §2：H(div) VEM 只要求单元内光滑 (R1)
// 与共享边几何重合 (R2)，不要求全局 J 连续）。
//
// 几何结构：参考域 [0,1]² 被切成真三块——竖直分界曲线 D 贯穿全高，
//   水平界面 H 只存在于 D 右侧并终止在 D 上（T 型节点）。再叠加一个在整条
//   外边界上消失的全局 swirl T，故物理域严格等于 [0,1]²，外边界保持直，
//   精确解/源项/边界条件/边界节点语义全部原样沿用。
//
// 节点构造方式（相对设计文档 §7.1 的偏离，理由见下）：
//   文档用「全局半步网格 (2n+1)² 求值一次，按 (ci,cj) 取每单元 8 节点」。
//   这里改为**逐单元从它自己的 4 个参考角点构造 8 个 Q8 节点**：
//   角点取 cell_nodes，中边点取相邻两角点的算术平均，对这 8 个参考点各求一次 G。
//
//   (R2) 照样按构造成立：相邻单元看到同一对全局节点，(x0+x1)/2 与
//   (x1+x0)/2 在 IEEE 下逐位相同，故共享边的 3 个 Q8 节点数值同一。
//
//   收益：不需要推出 n、不假设结构化 n×n、不需要把 cell_idx 反推成 (ci,cj)，
//   也就不依赖 straight_mesh.cpp:360-371 那个「四边形质心用外接盒算」的实现。
//
// 一个坑：cell_nodes 保证逆时针，但**起始角点在单元间循环变化**
//   （实测 square_2x2 是 右下→右上→左上→左下）。所以局部坐标 (a_i,b_i)
//   不能按「节点 0 就是 (-1,-1)」硬编码，必须按每个角点在参考域相对单元
//   中心的实际位置来定。建表时就把 8 个节点重排成规范 Q8 序，之后雅可比
//   便可使用教科书固定公式。
// ============================================================================

class TriBlockSwirlMapping : public IsoparametricMapping {
public:
    // 调参约束（同设计文档 §10）：a, c 必须落在宏观网格线上；
    //   增大 |alpha0| / A_H / |s_t - s_b| 提高难度，须重新确认 detJ_min > 0。
    // §10 还写着「br2 != -br3 且 bt2 != -bb3，否则界面处 C¹ 光滑、detJ 无跳变」，
    //   但默认值一直是 br3=1.1/br2=-1.1（现 1.32/-1.32），恰好违反前半条，
    //   且没有任何测试断言它——test_ms_triblock_q8 只查 detJ/cond/nonaff/共享边。
    //   即：这条约束目前是未生效的设计意图，不是代码事实。
    // 下列这组值是 tools/triblock_sweep 扫参选出的，约束是**四层 N=2,4,8,16
    //   全部 detJ > 0**。N=2（square_2x2，16 个单元）是收敛研究的最粗一层
    //   （见 main/MS_triblock_bdf2_main.cpp: meshes = {2,4,8,16,32}），也是
    //   最容易翻面的一层：G 层生成器 detJ 全正不代表 Q8 插值不自交，单元太少时
    //   8 节点二次单元跟不上大角度旋转。早期只扫 {4,8,16} 会漏掉这个失效。
    //
    // 实测：N=2 detJ_min 1.337e-01 cond 29.3->44.1 nonaff 1.544 坏元 0
    //       N=4/8/16 detJ_min ~0.197/0.199/0.200，cond 20.6/9.2/8.6（随细化下降）
    //
    // 三个旋钮在 N=2 上的代价极不对称，调参时别搞反：
    //   - beta 统一放大 1.2 倍（即下列展开值）反而**改善** N=2 cond（31.2->14.2），
    //     因为分级重分布了单元；再放大到 1.32 倍就翻面。
    //   - A_H 很贵：0.10->0.14 就把 N=2 cond 推到 87.2。
    //   - s_t-s_b 倾斜最贵：0.35/0.60 -> 0.32/0.63 就让 N=2 cond 爆到 8716。
    //   - m=2 严格优于 m=3：同 alpha0 下 nonaff 更高（更扭）而 N=2 cond 更低
    //     （13.0 vs 36.5）。m=1 太散，swirl 冲到边界，N=2 翻面。
    //   - alpha0 在 m=2 下可到 -85；-90 时 N=2 cond 163.6 已边缘，-95 翻面。
    struct Params {
        double a      = 0.5;    // 竖直界面参考位置
        double c      = 0.5;    // 水平界面参考位置
        double s_b    = 0.35;   // 分界线底端 x（再倾斜 N=2 立刻恶化，勿动）
        double s_t    = 0.60;   // 分界线顶端 x
        double bv     = 1.08;   // 全局纵向分级
        double b1     = 0.6;    // 左块横向分级
        double bb3    = 0.84;   // 右下块底边横向分级
        double bt2    = -0.72;  // 右上块顶边横向分级
        double br3    = 1.32;   // 右下块右边纵向分级
        double br2    = -1.32;  // 右上块右边纵向分级
        double bh     = 0.48;   // 水平界面横向分级（两块共用，不可分设）
        double y_r    = 0.62;   // 水平界面在 x=1 处的高度
        double A_H    = 0.10;   // 水平界面弯曲幅度（对 N=2 很贵，勿加）
        double alpha0 = -85.0 * M_PI / 180.0;  // 中心旋转角
        int    m      = 2;      // swirl 在边界的消失阶（2 最优，见上）
        // swirl 旋转支点。必须与权重 w = 16X(1-X)Y(1-Y) 的峰值 (0.5,0.5) 一致：
        // 否则「转最多的点」和「绕着转的点」不是同一个，图上表现为整体偏心。
        // 早期版本用 T 型节点 curve_D(c) 当支点，在 bv/s_b/s_t 较大时
        // 该点会漂到 (0.39,0.39) 附近，故改为独立可调。
        double cx     = 0.5;
        double cy     = 0.5;
    };

    // 两个构造函数而不是一个带默认参数的：C++11 下 `= Params()` 作为默认实参
    // 出现在类内时，编译器还没处理完 Params 的成员初始值，会直接报错。
    TriBlockSwirlMapping() {}
    explicit TriBlockSwirlMapping(const Params& p) : p_(p) {}

    bool requires_mesh() const override { return true; }

    Point2D physical_coords(int cell_idx,
                            double xi, double eta) const override;
    Tensor2D jacobian(int cell_idx, double xi, double eta) const override;
    double jacobian_det(int cell_idx, double xi, double eta) const override;
    Tensor2D jacobian_inv(int cell_idx, double xi, double eta) const override;
    Tensor2D jacobian_transpose(int cell_idx,
                                double xi, double eta) const override;
    Tensor2D jacobian_transpose_inv(int cell_idx,
                                    double xi, double eta) const override;

    // 生成器 G = T∘S（设计文档 §4-§6）。公开出来供自检测试直接调用。
    Point2D generator(double xi, double eta) const;

    // 单元的 Q8 节点数据。公开出来供自检测试检查 (R2) 与边界直性。
    struct CellQ8 {
        double nx[8];   // 8 个 Q8 节点物理 x，规范序
        double ny[8];   // 8 个 Q8 节点物理 y，规范序
        double xi_c;    // 参考单元中心
        double eta_c;
        double hx;      // 参考单元边长
        double hy;
    };
    const CellQ8& cell(int cell_idx) const;
    int num_cells() const { return static_cast<int>(cells_.size()); }

protected:
    void on_mesh_set() override;  // 建表

private:
    // 分级函数 g_β(u) = (e^{βu} - 1)/(e^β - 1)，g_0(u) = u
    static double grade(double u, double b);
    static double grade_deriv(double u, double b);

    double s_of(double y) const;      // 分界线 x 偏移 smoothstep
    double v_of(double eta) const;    // 全局纵向分级
    Point2D curve_D(double eta) const;      // 分界曲线
    Point2D curve_H(double pp) const;       // 水平界面曲线
    Point2D block1(double xi, double eta) const;
    Point2D block2(double xi, double eta) const;
    Point2D block3(double xi, double eta) const;
    Point2D swirl(double X, double Y) const;

    // Q8 serendipity 形函数导数（设计文档 §7.2），规范序
    static void q8_shape_deriv(double a, double b,
                               double dNda[8], double dNdb[8]);
    static void q8_shape(double a, double b, double N[8]);

    Params p_;
    std::vector<CellQ8> cells_;
};

// ============================================================================
// Maxwell-Stefan 方程 PDE 数据
//
// 签名约定：计算域回调（*_comp）凡是内部要用到映射的，都带 cell_idx，
//   参数顺序统一为「组分号 → 单元号 → 坐标 → 时间」。
//   物理域回调取 (x, y)，与单元无关，不带 cell_idx。
//   evaluate_A_comp_poly 只用多项式系数与 xD/hD，也不需要 cell_idx。
// ============================================================================

struct MSPdeData {
    int n_components = 3;  // 组分数

    // 二元扩散系数倒数 c_ij
    std::vector<std::vector<double>> c_ij;
    double c_star;                          // 最小扩散系数倒数
    std::vector<std::vector<double>> bar_c_ij;  // c_ij - c_star

    // 计算域上的初始浓度 c_i(ξ, η)（cell_idx 号单元内）
    // 直接用计算域坐标定义，映射到物理域后自然是曲边间断
    std::function<double(int i, int cell_idx, double xi, double eta)>
        initial_concentration_comp;

    // 物理域上的初始浓度 c_i(x, y)
    // 通过逆映射找到对应的 (ξ,η)，再用计算域初值求值。
    // 逆映射是分单元的，故需 cell_idx 指明在哪个单元内求逆。
    std::function<double(int i, int cell_idx, double x, double y)>
        initial_concentration;

    // 物理域源项 f_i(x, y, t)（质量守恒右端）
    std::function<double(int i, double x, double y, double t)> source_f;

    // 计算域源项 f̂_i(ξ, η, t) = f_i(x(ξ,η), y(ξ,η), t)（cell_idx 号单元内）
    std::function<double(int i, int cell_idx,
                         double xi, double eta, double t)> source_f_comp;

    // 物理域法向通量边界条件：J_i · n(x, y, t)
    std::function<double(int i, double x, double y, double nx, double ny, double t)> boundary_flux;

    // 计算域法向通量边界条件：Ĵ_i · n̂(ξ, η, t)（cell_idx 号单元内）
    // Piola 恒等式：J_i · n ds = Ĵ_i · n̂ dŝ，
    // 因此直接给定计算域法向通量 Ĵ·n̂，供边界积分使用
    std::function<double(int i, int cell_idx, double xi, double eta,
                         double nx, double ny, double t)> boundary_flux_comp;

    // ===== 真解（用于误差分析 / 制造解算例） =====

    // 物理域精确浓度 c_i(x, y, t)
    std::function<double(int i, double x, double y, double t)> exact_concentration;

    // 计算域精确浓度 ĉ_i(ξ, η, t) = c_i(x(ξ,η), y(ξ,η), t)（cell_idx 号单元内）
    std::function<double(int i, int cell_idx,
                         double xi, double eta, double t)> exact_concentration_comp;

    // 物理域精确通量矢量 J_i(x, y, t)（返回 x,y 两个分量）
    // 由本构关系 J = -Ā(c)^{-1} ∇c 计算
    std::function<Vector2D(int i, double x, double y, double t)> exact_flux;

    // 计算域精确通量矢量 Ĵ_i(ξ, η, t)（cell_idx 号单元内）
    // Piola 变换：Ĵ = J · det(J) · J^{-1} （逆变 Piola）
    std::function<Vector2D(int i, int cell_idx,
                          double xi, double eta, double t)> exact_flux_comp;

    // 初值模式：物理域上评估非线性扩散矩阵 A_ij（纯非线性部分，不含 c*I）
    //   A_ii(u) = Σ_{k≠i} bar_c_ik · u_k
    //   A_ij(u) = -bar_c_ji · u_i   (i ≠ j)
    // 浓度 u 从 initial_concentration 获取（内部通过逆映射求值，故需 cell_idx）
    std::function<double(int i, int j, int cell_idx,
                         double x, double y)> evaluate_A_initial;

    // 初值模式：计算域上评估非线性扩散矩阵 A_ij（纯非线性部分，不含 c*I）
    //   公式同上，浓度 u 从 initial_concentration_comp 获取（cell_idx 号单元内）
    std::function<double(int i, int j, int cell_idx,
                         double xi, double eta)> evaluate_A_comp_initial;

    // 数值解模式：计算域上评估 A_ij（纯非线性部分，不含 c*I）
    // 输入单项式系数向量，先还原各组分浓度值，再计算 A_ij。
    // 全程不碰映射，故不需要 cell_idx。
    //   u_ploy_coeff: 所有组分的单项式系数平铺
    //                布局：[comp0_monom0, ..., comp1_monom0, ...]
    //                长度 = n_components * n_monomial
    //   xD_x, xD_y: 单元质心（缩放基的中心）
    //   hD: 单元特征尺度
    //   xi, eta: 计算点（计算域坐标）
    std::function<double(int i, int j, const std::vector<double>& u_ploy_coeff,
                         double xD_x, double xD_y, double hD,
                         double xi, double eta)> evaluate_A_comp_poly;
};

// ============================================================================
// MS 算例管理类
// ============================================================================

class MSProblem {
public:
    MSProblem() : problem_index_(0) {}
    ~MSProblem() = default;

    // 禁用拷贝
    MSProblem(const MSProblem&) = delete;
    MSProblem& operator=(const MSProblem&) = delete;

    void set_problem_index(int index) { problem_index_ = index; }

    // 注入网格，作为分单元映射的网格调用基础。
    //
    // 可选调用，但若调用则必须在 init_problem() 之前——init_problem() 会把
    // 网格转交给它构造出来的映射对象。只存指针，不拥有所有权，
    // reader 的生命周期必须覆盖本对象的使用期。
    //
    // 不调用时 mesh_data_ 保持 nullptr。当前三个映射都是全局连续的解析映射，
    // 与单元无关，没有任何代码读取网格，所以不调用 set_mesh 完全正常，
    // 现有算例的跑法一行都不用改。
    void set_mesh(const vem::StraightMeshReader& reader) {
        mesh_reader_ = &reader;
        mesh_data_ = &reader.get_mesh_data();
    }

    bool init_problem();

    const MSPdeData& get_pde_data() const { return pde_data_; }

    const IsoparametricMapping& get_mapping() const {
        if (!mapping_) {
            throw std::runtime_error("MS problem is not initialized!");
        }
        return *mapping_;
    }

    const vem::StraightMeshReader* get_mesh_reader() const {
        return mesh_reader_;
    }
    const vem::StraightMeshData* get_mesh_data() const { return mesh_data_; }
    bool has_mesh() const { return mesh_data_ != nullptr; }

    // 由物理坐标 (x,y) 反求计算域坐标 (ξ,η)
    // 用牛顿迭代求解 x(ξ,η) = x_target, y(ξ,η) = y_target
    //
    // 映射是分单元的，全局逆映射不再唯一定义，故必须指明在哪个单元内求逆：
    // cell_idx 号单元的映射公式决定了这次牛顿迭代解的是哪个方程。
    Point2D invert_mapping(double x, double y, int cell_idx) const;

private:
    int problem_index_;
    MSPdeData pde_data_;
    std::unique_ptr<IsoparametricMapping> mapping_;

    // 网格调用基础（仿 HdivMatrix：只存指针，不拥有所有权）
    const vem::StraightMeshReader* mesh_reader_ = nullptr;
    const vem::StraightMeshData* mesh_data_ = nullptr;

    // 初始化具体算例
    bool init_problem_1();
    bool init_problem_2();
    bool init_problem_3();
    bool init_problem_4();

    // 算例 2 与算例 4 共用的制造解数据（真解 / 通量 / 源项 / 边界 / A_ij）。
    // 两个算例只有映射不同，PDE 数据一字不差，所以提取出来共用，
    // 误差表也就能直接横向对比。调用前 mapping_ 必须已构造。
    void setup_manufactured_solution();
};

}  // namespace MaxwellStefan

#endif  // CURVEDVEM_MS_PROBLEM_H
