# 曲边混合虚拟元求解 Maxwell-Stefan 扩散方程 - 完整理论与架构文档

> **文档版本**：v2.0（对照 LaTeX 理论稿全面修订方程表述与双线性型定义）
> **修订日期**：2026-08-26
> **代码位置**：`vem/CurvedVEM_Darcy/`
> **参考文档**：`MaxwellStefanVem理论.tex`（MS 方程理论推导）、`混合虚拟元_darcy.md`（Darcy 方程参考）

---

## 第一部分 数学理论推导

### 1.1 Maxwell-Stefan 扩散方程强形式

在二维光滑曲边多边形域 $\Omega$ 中，考虑 $n$ 组分的 **Maxwell-Stefan 扩散方程组**。设浓度体积分数为 $\mathbf{u}=(u_1, u_2, \ldots, u_n)$，浓度通量为 $\mathbf{J}=(J_1, J_2, \ldots, J_n)$，其中 $u_i \in [0,1],\ i=1,2,\ldots,n$。方程的向量形式为：

$$
\left\{
\begin{aligned}
&\frac{\partial \mathbf{u}}{\partial t} + \nabla \cdot \mathbf{J} = \mathbf{f}, \\
&\nabla \mathbf{u} + \bar{A}(\mathbf{u})\mathbf{J} = \mathbf{0}.
\end{aligned}
\right.
$$

其中组分形式为：

$$
\left\{
\begin{aligned}
&\frac{\partial u_i}{\partial t} + \nabla \cdot J_i = f_i, \\
&\nabla u_i + \sum_{j=1}^{n} \bar{A}_{ij}(\mathbf{u}) J_j = 0,
\end{aligned}
\right.
\quad i=1,2,\ldots,n.
$$

$\mathbf{f}=(f_1,\dots,f_n)$ 为体积源项（反应项或外部源）。边界条件按组分给定：
- 通量边界 $\Gamma_N$：$J_i \cdot \boldsymbol{n} = g_{N,i}$
- 浓度边界 $\Gamma_D$：$u_i = g_{D,i}$

其中 $\Gamma_N \cup \Gamma_D = \partial\Omega$，$\Gamma_N \cap \Gamma_D = \emptyset$。

#### 扩散矩阵 $\bar{A}(\mathbf{u})$ 的定义

扩散矩阵 $\bar{A}(\mathbf{u}) \in \mathbb{R}^{n \times n}$ 的元素为：

$$
\bar{A}(\mathbf{u}) = \begin{bmatrix}
c^* + \sum\limits_{\substack{1\leq j \leq n \\ j\neq 1}} \bar{c}_{1j}u_j & -\bar{c}_{12}u_1 & -\bar{c}_{13}u_1 & \cdots & -\bar{c}_{1n}u_1 \\
-\bar{c}_{21}u_2 & c^* + \sum\limits_{\substack{1\leq j \leq n \\ j\neq 2}} \bar{c}_{2j}u_j & -\bar{c}_{23}u_2 & \cdots & -\bar{c}_{2n}u_2 \\
-\bar{c}_{31}u_3 & -\bar{c}_{32}u_3 & c^* + \sum\limits_{\substack{1\leq j \leq n \\ j\neq 3}} \bar{c}_{3j}u_j & \cdots & -\bar{c}_{3n}u_3 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
-\bar{c}_{n1}u_n & -\bar{c}_{n2}u_n & -\bar{c}_{n3}u_n & \cdots & c^* + \sum\limits_{\substack{1\leq j \leq n \\ j\neq n}} \bar{c}_{nj}u_j
\end{bmatrix}
$$

**参数说明**：
- $c_{ij}$：第 $i$ 组分与第 $j$ 组分之间的二元扩散系数的倒数
- $c^* = \min\limits_{i,j} \{c_{ij}\}$：所有二元扩散系数倒数的最小值
- $\bar{c}_{ij} = c_{ij} - c^*$：相对扩散系数（差值），恒非负

#### 线性-非线性分解

将扩散矩阵分解为线性自扩散项 + 非线性交叉扩散耦合项：

$$
\bar{A}(\mathbf{u}) = c^* I + A(\mathbf{u})
$$

其中非线性交叉扩散矩阵 $A(\mathbf{u}) \in \mathbb{R}^{n \times n}$ 的元素为：

$$
A(\mathbf{u}) = \begin{bmatrix}
\sum\limits_{\substack{1\leq j \leq n \\ j\neq 1}} \bar{c}_{1j}u_j & -\bar{c}_{12}u_1 & \cdots & -\bar{c}_{1n}u_1 \\
-\bar{c}_{21}u_2 & \sum\limits_{\substack{1\leq j \leq n \\ j\neq 2}} \bar{c}_{2j}u_j & \cdots & -\bar{c}_{2n}u_2 \\
\vdots & \vdots & \ddots & \vdots \\
-\bar{c}_{n1}u_n & -\bar{c}_{n2}u_n & \cdots & \sum\limits_{\substack{1\leq j \leq n \\ j\neq n}} \bar{c}_{nj}u_j
\end{bmatrix}
$$

分解后的方程形式为：

$$
\left\{
\begin{aligned}
&\frac{\partial u_i}{\partial t} + \nabla \cdot J_i = f_i, \\
&\nabla u_i + c^* J_i + \sum_{j=1}^{n} A_{ij}(\mathbf{u}) J_j = 0,
\end{aligned}
\right.
\quad i=1,2,\ldots,n.
$$

**矩阵性质**：
- **对角元**：$A_{ii} = \sum_{j \neq i} \bar{c}_{ij} u_j$（非线性自扩散，由其他组分浓度加权和构成）
- **非对角元**：$A_{ij} = -\bar{c}_{ji} u_i$（$i \neq j$，非线性交叉扩散）
- **行和为零**：$\sum_{j=1}^n A_{ij}(\mathbf{u}) = 0$（质量守恒结构）
- 所有 $\bar{c}_{ij} \ge 0$，保证对角元恒正，矩阵具有 M 矩阵结构

---

### 1.2 混合变分弱形式（鞍点问题）

#### 函数空间定义

定义 $H(\mathrm{div}; \Omega)$ 空间及其齐次迹空间：
$$
H(\mathrm{div}; \Omega) = \{ \mathbf{v} \in [L^2(\Omega)]^2 : \nabla\cdot\mathbf{v} \in L^2(\Omega) \}
$$
$$
\mathring{H}(\mathrm{div}; \Omega) = \{ \mathbf{v} \in H(\mathrm{div}; \Omega) : \mathbf{v}\cdot\mathbf{n} = 0 \text{ on } \partial\Omega \}
$$

多组分空间定义为单组分空间的直积：
- **通量检验空间**：$\boldsymbol{V} = \big[\mathring{H}(\mathrm{div}; \Omega)\big]^n$
- **浓度空间**：$Q = \big[L^2(\Omega)\big]^n$

#### 双线性型与三线性型定义

对第 $i$ 组分，定义以下双线性型和三线性型：

1. **线性扩散双线性型 $\mathscr{C}$**（对应 $c^* I$ 项）：
   $$
   \mathscr{C}(J_{i}, v_{i}) = c^* \int_\Omega J_i \cdot v_i \, dx
   $$

2. **非线性交叉扩散三线性型 $\mathscr{A}_{ij}$**（对应 $A_{ij}(\mathbf{u})$ 项）：
   $$
   \mathscr{A}_{ij}(\mathbf{u}; J_j, v_i) = \int_\Omega A_{ij}(\mathbf{u}) \, J_j \cdot v_i \, dx
   $$

3. **散度耦合双线性型 $\mathscr{B}$**：
   $$
   \mathscr{B}(v_i, q_i) = \int_\Omega (\nabla \cdot v_i) \, q_i \, dx
   $$
   第一个自变量为通量（向量场），第二个自变量为浓度（标量场）。

4. **时间质量双线性型 $\mathscr{M}$**：
   $$
   \mathscr{M}\left(\frac{\partial u_i}{\partial t}, q_i\right) = \int_\Omega \frac{\partial u_i}{\partial t} q_i \, dx
   $$

5. **源项右端泛函**：
   $$
   F_Q(q_i) = \int_\Omega f_i \, q_i \, dx
   $$

#### 连续变分问题

求 $J_i \in \mathring{H}(\mathrm{div};\Omega)$ 和 $u_i \in L^2(\Omega)$，使得对任意 $v_i \in \mathring{H}(\mathrm{div};\Omega)$ 和任意 $q_i \in L^2(\Omega)$ 成立：

$$
\begin{cases}
\mathscr{C}(J_i, v_i) + \displaystyle\sum_{j=1}^n \mathscr{A}_{ij}(\mathbf{u}; J_j, v_i) + \mathscr{B}(v_i, u_i) = 0, \\[10pt]
\mathscr{M}\left(\dfrac{\partial u_i}{\partial t}, q_i\right) + \mathscr{B}(J_i, q_i) = F_Q(q_i),
\end{cases}
\quad i = 1, 2, \dots, n.
$$

写成全向量形式：

$$
\begin{cases}
\mathscr{C}(\mathbf{J}, \mathbf{v}) + \mathscr{A}(\mathbf{u}; \mathbf{J}, \mathbf{v}) + \mathscr{B}(\mathbf{v}, \mathbf{u}) = \mathbf{0}, \\[8pt]
\mathscr{M}\left(\dfrac{\partial \mathbf{u}}{\partial t}, q\right) + \mathscr{B}(\mathbf{J}, q) = F_Q(q).
\end{cases}
$$

其中 $\mathscr{A}(\mathbf{u}; \mathbf{J}, \mathbf{v}) = \sum\limits_{i,j} \mathscr{A}_{ij}(\mathbf{u}; J_j, v_i)$ 为全体组分的非线性耦合项之和。

**说明**：第一个方程包含散度耦合项 $\mathscr{B}(v_i, u_i)$，将浓度梯度信息转化为通量空间的检验内积；第二个方程为质量守恒的变分形式。$\mathscr{B}$ 是通用散度耦合双线性型，不依赖扩散系数和几何映射。

---

### 1.3 曲边单元等参映射与 Piola 变换

#### 参考元 → 物理单元映射
标准参考单元 $\hat{E} = [0,1]^2$，局部坐标 $(\xi, \eta)$。曲边物理单元 $E$ 的等参映射：
$$
\boldsymbol{x}: \hat{E} \to E, \quad
\boldsymbol{x} = (x(\xi, \eta), y(\xi, \eta))
$$

雅可比矩阵及其行列式：
$$
\mathbb{J}(\xi, \eta) =
\nabla_{(\xi,\eta)} \boldsymbol{x} =
\begin{pmatrix}
\partial_\xi x & \partial_\eta x \\
\partial_\xi y & \partial_\eta y
\end{pmatrix},
\qquad
J = \det(\mathbb{J}) > 0
$$

物理梯度 ↔ 参考梯度变换：
$$
\nabla_{\boldsymbol{x}} \phi(\boldsymbol{x}) =
\mathbb{J}^{-\top}(\xi, \eta) \,
\nabla_{(\xi,\eta)} \hat{\phi}(\xi, \eta),
\qquad
\hat{\phi}(\xi, \eta) = \phi(\boldsymbol{x}(\xi, \eta))
$$

#### H(div) 相容 Piola 变换
通量向量场采用逆变 Piola 变换，以保证跨单元法向通量连续：

$$
J_i(\boldsymbol{x}) =
\frac{1}{J(\xi, \eta)}
\mathbb{J}(\xi, \eta) \,
\hat{J}_i(\xi, \eta)
$$

**Piola 散度恒等式（核心化简关系）**：
$$
\nabla_{\boldsymbol{x}} \cdot J_i =
\frac{1}{J(\xi, \eta)} \,
\hat{\nabla} \cdot \hat{J}_i
$$

这保证了散度耦合项的几何不变性。

#### 曲边界 Nanson 积分变换
物理曲边上的法向测度变换：
$$
\boldsymbol{n}(\boldsymbol{x}) \, ds =
J \, \mathbb{J}^{-\top} \hat{\boldsymbol{n}} \, d\hat{s}
$$

分离单位法向量与弧长微元：
$$
ds = J \, \|\mathbb{J}^{-\top} \hat{\boldsymbol{n}}\| \, d\hat{s},
\qquad
\boldsymbol{n}(\boldsymbol{x}) =
\frac{\mathbb{J}^{-\top} \hat{\boldsymbol{n}}}
{\|\mathbb{J}^{-\top} \hat{\boldsymbol{n}}\|}
$$

所有曲边界积分均可拉回至参考直边计算。

---

### 1.4 单元积分拉回参考元

全域积分分解为单元积分之和，逐单元将物理曲边单元 $E$ 上的积分拉回到直边参考单元 $\hat{E}$ 上计算。以下分别给出各双线性型和三线性型的拉回公式。

#### 线性扩散双线性型 $\mathscr{C}$ 的拉回

$$
\begin{aligned}
\mathscr{C}_E(J_i, v_i)
&= c^* \int_E J_i \cdot v_i \, dx \\
&= c^* \int_{\hat{E}}
\left( \frac{\mathbb{J} \hat{J}_i}{J} \right)
\cdot
\left( \frac{\mathbb{J} \hat{v}_i}{J} \right)
J \, d\hat{\boldsymbol{x}} \\
&= \int_{\hat{E}}
\underbrace{\frac{c^*}{J} \, \mathbb{J}^\top \mathbb{J}}
_{\mathbb{K}_\xi^C}
\hat{J}_i \cdot \hat{v}_i \, d\hat{\boldsymbol{x}}
= \hat{\mathscr{C}}_E(\hat{J}_i, \hat{v}_i).
\end{aligned}
$$

$\mathbb{K}_\xi^C$ 为曲几何修正的线性扩散张量。

#### 非线性交叉扩散三线性型 $\mathscr{A}_{ij}$ 的拉回

使用非线性矩阵 $A(\mathbf{u})$（不含 $c^*$，线性项已归入 $\mathscr{C}$），其元素为：
- 对角元：$A_{ii}(\mathbf{u}) = \sum\limits_{k \neq i} \bar{c}_{ik} u_k$
- 非对角元：$A_{ij}(\mathbf{u}) = -\bar{c}_{ji} u_i \quad (i \neq j)$

$$
\begin{aligned}
\mathscr{A}_{ij,E}(\mathbf{u}; J_j, v_i)
&= \int_E A_{ij}(\mathbf{u}) \, J_j \cdot v_i \, dx \\
&= \int_{\hat{E}} A_{ij}(\hat{\mathbf{u}}) \,
\frac{1}{J} \hat{J}_j^\top \mathbb{J}^\top \mathbb{J} \hat{v}_i
\, d\hat{\boldsymbol{x}} \\
&= \int_{\hat{E}}
\underbrace{\frac{A_{ij}(\hat{\mathbf{u}})}{J} \, \mathbb{J}^\top \mathbb{J}}
_{\mathbb{K}_\xi^{A,ij}}
\hat{J}_j \cdot \hat{v}_i \, d\hat{\boldsymbol{x}}
= \hat{\mathscr{A}}_{ij,E}(\hat{\mathbf{u}}; \hat{J}_j, \hat{v}_i).
\end{aligned}
$$

**分块说明**：
- 对角块（$i = j$）：纯非线性自扩散，系数为 $\sum\limits_{k \neq i} \bar{c}_{ik} u_k$
- 非对角块（$i \neq j$）：交叉扩散耦合，系数为 $-\bar{c}_{ji} u_i$

> **注意**：线性扩散项 $c^*$ 与非线性交叉扩散项 $A(\mathbf{u})$ 的几何修正张量形式完全一致（均为 $\frac{\cdot}{J}\mathbb{J}^\top\mathbb{J}$），仅标量系数不同。程序中可复用同一套加权 Gram 矩阵构造代码。

#### 散度耦合双线性型 $\mathscr{B}$ 的拉回（无几何修正）

利用 Piola 散度恒等式，雅可比 $J$ 恰好抵消：

$$
\begin{aligned}
\mathscr{B}_E(v_i, q_i)
&= \int_E (\nabla \cdot v_i) \, q_i \, dx \\
&= \int_{\hat{E}}
\frac{1}{J} (\hat{\nabla} \cdot \hat{v}_i) \cdot \hat{q}_i
\cdot J \, d\hat{\boldsymbol{x}} \\
&= \int_{\hat{E}}
\hat{q}_i \, (\hat{\nabla} \cdot \hat{v}_i)
\, d\hat{\boldsymbol{x}}
= \hat{\mathscr{B}}_E(\hat{v}_i, \hat{q}_i).
\end{aligned}
$$

**关键结论**：曲边单元与直边单元的散度耦合积分公式完全一致，无额外几何修正项。这是混合虚拟元方法的核心优势之一。

#### 时间质量双线性型 $\mathscr{M}$ 的拉回

浓度为标量场，拉回后乘面积雅可比权重：

$$
\begin{aligned}
\mathscr{M}_E(u_i, q_i)
&= \int_E u_i \, q_i \, dx \\
&= \int_{\hat{E}}
\hat{u}_i \, \hat{q}_i \, J(\xi, \eta) \, d\hat{\boldsymbol{x}}
= \hat{\mathscr{M}}_E(\hat{u}_i, \hat{q}_i).
\end{aligned}
$$

#### 体积源项右端

$$
F_{Q,E}(q_i) =
\int_E f_i \, q_i \, dx =
\int_{\hat{E}}
f_i(\boldsymbol{x}(\xi, \eta)) \, \hat{q}_i \, J \, d\hat{\boldsymbol{x}}
= \hat{F}_{Q,E}(\hat{q}_i).
$$

---

### 1.5 混合虚拟元空间离散

#### 全局离散空间

$$
\begin{aligned}
V_h =& \big\{ \mathbf{v}_h \in H(\mathrm{div}; \Omega) \cap H(\mathrm{rot}; \Omega) : \; \mathbf{v}_h\cdot\mathbf{n}_{|e} \in \mathbb{P}_k(e), \ \forall e \in \mathcal{E}_h, \\
&\quad (\nabla \cdot \mathbf{v}_h)_{|E} \in \mathbb{P}_{k}(E), \ (\nabla \times \mathbf{v}_h)_{|E} \in \mathbb{P}_{k-1}(E), \ \forall E \in \mathcal{T}_h \big\}.\\
Q_h =& \big\{q_h \in L^2(\Omega) :  q_{h|E} \in \mathbb{P}_k(E), \ \forall E \in \mathcal{T}_h \big\}.
\end{aligned}
$$

多组分为单组分空间的直积：$\mathbf{V}_h = (V_h)^n$，$\mathbf{Q}_h = (Q_h)^n$。

#### 单元离散空间 $\hat{V}_h(\hat{E})$

在直边参考单元 $\hat{E}$ 上，单组分的 $H(\mathrm{div})$ 虚拟元空间定义为：
$$
\begin{aligned}
\hat{V}_h(\hat{E}) = \big\{ \hat{v} \in H(\mathrm{div}; \hat{E}) \cap H(\mathrm{rot}; \hat{E}) : \;
& \hat{v} \cdot \hat{\boldsymbol{n}} \in \mathbb{P}_k(\hat{e}), \ \forall \hat{e} \subset \partial\hat{E}, \\
& \hat{\nabla} \cdot \hat{v} \in \mathbb{P}_k(\hat{E}), \\
& \hat{\nabla} \times \hat{v} \in \mathbb{P}_{k-1}(\hat{E}) \big\}.
\end{aligned}
$$

**关键说明**：
- 边法向迹为 $k$ 次多项式 → 散度由散度定理恰好为 $k$ 次多项式
- 旋度为 $k-1$ 次多项式 → 用于确定补空间维度
- 多项式向量空间分解：$[\mathbb{P}_k(\hat{E})]^2 = \nabla \mathbb{P}_{k+1}(\hat{E}) \oplus \mathcal{G}_k^\perp(\hat{E})$
  - $\nabla \mathbb{P}_{k+1}$ 的维数：$\dim \mathbb{P}_{k+1} - 1 = \frac{(k+2)(k+3)}{2} - 1$
  - 补空间 $\mathcal{G}_k^\perp$ 的维数：$\dim \mathbb{P}_{k-1} = \frac{k(k+1)}{2}$

#### 单元三类自由度（单组分）

1. **边法向矩**：每条边 $k+1$ 个
   $$
   \int_{\hat{e}} (\hat{v} \cdot \hat{\boldsymbol{n}}) \, m_\ell \, d\hat{s}, \quad
   m_\ell \in \mathbb{P}_k(\hat{e}), \quad \ell = 0, \dots, k
   $$
   共 $(k+1) \cdot n_{\text{edge}}$ 个。

2. **内部梯度矩**（对应 $\nabla \mathbb{P}_{k+1}/\mathbb{R}$）：
   $$
   \int_{\hat{E}} \hat{v} \cdot \hat{\nabla} m \, d\hat{\boldsymbol{x}}, \quad
   m \in \mathbb{P}_{k+1}(\hat{E}) / \mathbb{R}
   $$
   共 $\dim \mathbb{P}_{k+1} - 1 = \frac{(k+2)(k+3)}{2} - 1$ 个。

3. **内部旋度补空间矩**（对应 $\mathcal{G}_k^\perp$）：
   $$
   \int_{\hat{E}} (\hat{\nabla} \times \hat{v}) \, m \, d\hat{\boldsymbol{x}}, \quad
   m \in \mathbb{P}_{k-1}(\hat{E})
   $$
   共 $\dim \mathbb{P}_{k-1} = \frac{k(k+1)}{2}$ 个。

单组分单元自由度总数：
$$
n_{\text{dof}} = (k+1)n_{\text{edge}} + \big(\dim \mathbb{P}_{k+1} - 1\big) + \dim \mathbb{P}_{k-1}.
$$

$n$ 组分为 $n \cdot n_{\text{dof}}$。

#### 浓度离散空间 $\hat{Q}_h(\hat{E})$
分片多项式，无跨单元连续性要求，次数与通量散度匹配：
$$
\hat{Q}_h(\hat{E}) = \mathbb{P}_k(\hat{E})
$$

单组分单元浓度自由度：$\dim \mathbb{P}_k = \frac{(k+1)(k+2)}{2}$。

#### 加权 L² 投影算子
任意通量基函数 $\hat{\phi}_i \in \hat{V}_h$ 分解为多项式相容部分 + 虚拟不稳定部分：
$$
\hat{\phi}_i =
\Pi_k^0 \hat{\phi}_i +
(I - \Pi_k^0) \hat{\phi}_i
$$

曲边单元的加权正交条件（带雅可比权重 $J$）：
$$
\int_{\hat{E}}
\left( \hat{\phi}_i - \Pi_k^0 \hat{\phi}_i \right)
\cdot \boldsymbol{r} \
J(\xi, \eta) \, d\hat{\boldsymbol{x}} = 0,
\quad
\forall \boldsymbol{r} \in [\mathbb{P}_k(\hat{E})]^2
$$

**投影矩阵代数构造**：
设 $[\mathbb{P}_k(\hat{E})]^2$ 的标准多项式基为 $\{g_\alpha\}$，自由度矩阵 $D_{i\alpha} = \mathrm{dof}_i(g_\alpha)$，加权 Gram 矩阵：
$$
G_{\alpha\beta} =
\int_{\hat{E}}
g_\alpha \cdot g_\beta \
J(\xi, \eta) \, d\hat{\boldsymbol{x}}
$$

右端投影矩矩阵：
$$
B_{\alpha i} =
\int_{\hat{E}}
\hat{\phi}_i \cdot g_\alpha \
J(\xi, \eta) \, d\hat{\boldsymbol{x}}
$$

则投影算子为：
$$
\Pi_k^0 = D \, G^{-1} \, B
$$

---

### 1.6 单元离散双线性/三线性型

借助 $L^2$ 投影算子 $\Pi_k^0$，构造可计算的离散双线性型和三线性型来逼近连续变分形式。每个双线性型均由 **一致性项**（多项式投影部分）+ **稳定化项**（投影核控制）组成。

设稳定化双线性型 $\mathscr{S}^E(\cdot,\cdot)$ 满足谱等价性：存在与网格无关的常数 $s_*, s^* > 0$，使得
$$
s_*\|\mathbf{v}_h\|_{0,E}^2 \leq \mathscr{S}^E(\mathbf{v}_h, \mathbf{v}_h) \leq s^*\|\mathbf{v}_h\|_{0,E}^2, \quad \forall \mathbf{v}_h \in V_h(E).
$$

程序中采用自由度差分型稳定化：设 $\{\mathrm{dof}_m\}_{m=1}^{n_{\text{dof}}}$ 为单元通量自由度，则
$$
\mathscr{S}^E(\delta \hat{v}, \delta \hat{w}) =
\sum_{m=1}^{n_{\text{dof}}}
\mathrm{dof}_m(\delta \hat{v}) \cdot
\mathrm{dof}_m(\delta \hat{w}).
$$

---

#### 线性扩散双线性型 $\mathscr{C}$

对应连续形式中的 $c^* I$ 项。曲边单元的一致性项包含几何修正张量 $\mathbb{K}_\xi^C = \frac{c^*}{J}\mathbb{J}^\top\mathbb{J}$：

$$
\mathscr{C}_E(\hat{J}_{i,h}, \hat{v}_{i,h}) :=
\int_{\hat{E}}
\mathbb{K}_\xi^C
(\Pi_k^0 \hat{J}_{i,h}) \cdot
(\Pi_k^0 \hat{v}_{i,h}) \, d\hat{\boldsymbol{x}}
+ c^* \mathscr{S}^E\big((I-\Pi_k^0)\hat{J}_{i,h}, (I-\Pi_k^0)\hat{v}_{i,h}\big).
$$

稳定化强度取线性扩散系数 $c^*$，与投影核的尺度匹配。

---

#### 非线性交叉扩散三线性型 $\mathscr{A}_{ij}$

采用分裂稳定策略：**仅自扩散对角项附加稳定化，交叉扩散项不加稳定化**。

1. **自扩散项（$i = j$）**：
   $$
   \mathscr{A}_{ii,E}(\hat{\mathbf{u}}_h; \hat{J}_{i,h}, \hat{v}_{i,h}) :=
   \int_{\hat{E}}
   \mathbb{K}_\xi^{A,ii}
   (\Pi_k^0 \hat{J}_{i,h}) \cdot
   (\Pi_k^0 \hat{v}_{i,h}) \, d\hat{\boldsymbol{x}}
   + |A_{ii}(\hat{\mathbf{u}}_h)|_E \,
   \mathscr{S}^E\big((I-\Pi_k^0)\hat{J}_{i,h}, (I-\Pi_k^0)\hat{v}_{i,h}\big).
   $$

   其中 $\mathbb{K}_\xi^{A,ii} = \frac{A_{ii}(\hat{\mathbf{u}}_h)}{J}\mathbb{J}^\top\mathbb{J}$ 为曲几何修正的非线性扩散张量，$|A_{ii}(\hat{\mathbf{u}}_h)|_E$ 为单元平均的对角元尺度。

2. **交叉扩散项（$i \neq j$）**：
   $$
   \mathscr{A}_{ij,E}(\hat{\mathbf{u}}_h; \hat{J}_{j,h}, \hat{v}_{i,h}) :=
   \int_{\hat{E}}
   \mathbb{K}_\xi^{A,ij}
   (\Pi_k^0 \hat{J}_{j,h}) \cdot
   (\Pi_k^0 \hat{v}_{i,h}) \, d\hat{\boldsymbol{x}}.
   $$

   其中 $\mathbb{K}_\xi^{A,ij} = \frac{A_{ij}(\hat{\mathbf{u}}_h)}{J}\mathbb{J}^\top\mathbb{J}$。交叉项为纯耦合项，不含投影核稳定化。

---

#### 散度耦合双线性型 $\mathscr{B}$（与直边完全相同）

利用 Piola 散度恒等式，几何因子全部抵消，曲边公式与直边完全一致：

$$
\mathscr{B}_E(\hat{\mathbf{v}}_h, \hat{q}_h) =
\sum_{i=1}^n \int_{\hat{E}}
\hat{q}_{i,h} \,
(\hat{\nabla} \cdot \hat{v}_{i,h})
\, d\hat{\boldsymbol{x}}.
$$

离散耦合矩阵元素：$W_{\alpha i} = \mathscr{B}_E(\hat{\phi}_i, \hat{q}_\alpha)$。

---

#### 时间质量双线性型 $\mathscr{M}$

浓度为标量场，拉回后乘雅可比权重：

$$
\mathscr{M}_E(\hat{\mathbf{u}}_h, \hat{q}_h) =
\sum_{i=1}^n \int_{\hat{E}}
\hat{u}_{i,h} \, \hat{q}_{i,h} \,
J(\xi, \eta) \, d\hat{\boldsymbol{x}}.
$$

单元质量矩阵为按组分的对角块结构（不同组分之间无耦合）。

---

### 1.7 时间离散：向后欧拉法（BDF1）与二阶向后差分法（BDF2）

设时间步长 $\Delta t > 0$，记时间层 $t^n = n\Delta t$，引入记号 $u^n := u(t^n)$。本节给出一阶向后欧拉（BDF1）与二阶向后差分（BDF2）两种时间离散方案。BDF1 用于初始步启动，BDF2 在后续时间步使用以获得更高的时间精度。

---

#### 1.7.1 一阶向后欧拉（BDF1）

向后欧拉时间离散为：
$$
\delta_t u^n = \frac{u^n - u^{n-1}}{\Delta t}.
$$

##### 全离散变分格式

求 $J_{i,h}^n \in V_h$ 和 $u_{i,h}^n \in Q_h$，使得对任意 $v_{i,h} \in V_h$ 和任意 $q_{i,h} \in Q_h$ 成立：

$$
\begin{cases}
\mathscr{C}(J_{i,h}^n, v_{i,h}) +
\displaystyle\sum_{j=1}^n
\mathscr{A}_{ij}(\mathbf{u}_h^n; J_{j,h}^n, v_{i,h}) +
\mathscr{B}(v_{i,h}, u_{i,h}^n) = 0, \\[12pt]
\mathscr{M}(\delta_t u_{i,h}^n, q_{i,h}) +
\mathscr{B}(J_{i,h}^n, q_{i,h}) =
(f_i^n, q_{i,h}),
\end{cases}
\quad i = 1,2,\dots,n.
$$

##### 整理为标准鞍点形式

将时间差分离散代入，把含 $u_h^n$ 的项放在左端，含 $u_h^{n-1}$ 的项放在右端：

$$
\begin{cases}
\mathscr{C}(J_{i,h}^n, v_{i,h}) +
\displaystyle\sum_{j=1}^n
\mathscr{A}_{ij}(\mathbf{u}_h^n; J_{j,h}^n, v_{i,h}) +
\mathscr{B}(v_{i,h}, u_{i,h}^n) = 0, \\[12pt]
\dfrac{1}{\Delta t} \mathscr{M}(u_{i,h}^n, q_{i,h}) +
\mathscr{B}(J_{i,h}^n, q_{i,h}) =
F_Q(q_{i,h}) +
\dfrac{1}{\Delta t} \mathscr{M}(u_{i,h}^{n-1}, q_{i,h}).
\end{cases}
$$

右端包含上一时间层的质量贡献，携带 $1/\Delta t$ 因子。

---

#### 1.7.2 二阶向后差分（BDF2）

BDF2 利用前两个时间层的信息构造二阶精度的时间导数逼近：
$$
\delta_t u^n = \frac{3u^n - 4u^{n-1} + u^{n-2}}{2\Delta t}.
$$

时间截断误差为 $O(\Delta t^2)$，在 A-稳定的线性多步法中具有最大的虚轴稳定区间，非常适合扩散型方程。

##### 启动策略（第一步用 BDF1）

BDF2 需要前两个时间层的解（$u^{n-1}$ 和 $u^{n-2}$）才能推进到 $u^n$，因此初始时刻 $t^0$ 只有初值 $u^0$，无法直接启动 BDF2。采用如下启动策略：

- **第 1 步**（$n=1$，从 $t^0$ 到 $t^1$）：使用一阶 BDF1 计算 $u^1$，时间步长取为 $\Delta t$（或取更小的启动步长 $\Delta t/2$ 以减小启动误差）。
- **第 2 步及之后**（$n \geq 2$）：已有 $u^{n-1}$ 和 $u^{n-2}$，切换为 BDF2 继续推进。

> **关于启动误差的说明**：BDF1 的一阶启动误差仅为局部的，对全局时间精度的影响通常被吸收或在后续步中被扩散耗散。若需严格保持全局二阶，可采用更小的启动步长（如 $\Delta t_{\text{start}} = \Delta t / 2$ 或 $\Delta t^2$ 量级），或用 RK2 等二阶自启动单步法做第一步。本文实现采用等步长 BDF1 启动，实践中对 Maxwell-Stefan 扩散问题已足够。

##### 全离散变分格式（$n \geq 2$）

求 $J_{i,h}^n \in V_h$ 和 $u_{i,h}^n \in Q_h$，使得对任意 $v_{i,h} \in V_h$ 和任意 $q_{i,h} \in Q_h$ 成立：

$$
\begin{cases}
\mathscr{C}(J_{i,h}^n, v_{i,h}) +
\displaystyle\sum_{j=1}^n
\mathscr{A}_{ij}(\mathbf{u}_h^n; J_{j,h}^n, v_{i,h}) +
\mathscr{B}(v_{i,h}, u_{i,h}^n) = 0, \\[12pt]
\mathscr{M}\!\left(\dfrac{3u_{i,h}^n - 4u_{i,h}^{n-1} + u_{i,h}^{n-2}}{2\Delta t},\, q_{i,h}\right) +
\mathscr{B}(J_{i,h}^n, q_{i,h}) =
(f_i^n, q_{i,h}),
\end{cases}
$$
对 $i = 1,2,\dots,n$ 成立。

##### 整理为标准鞍点形式

将时间差分离散代入，左端为当前时间层 $u^n$ 的质量项（系数 $3/(2\Delta t)$），右端包含前两个历史层的贡献：

$$
\begin{cases}
\mathscr{C}(J_{i,h}^n, v_{i,h}) +
\displaystyle\sum_{j=1}^n
\mathscr{A}_{ij}(\mathbf{u}_h^n; J_{j,h}^n, v_{i,h}) +
\mathscr{B}(v_{i,h}, u_{i,h}^n) = 0, \\[12pt]
\dfrac{3}{2\Delta t} \mathscr{M}(u_{i,h}^n, q_{i,h}) +
\mathscr{B}(J_{i,h}^n, q_{i,h}) =
F_Q(q_{i,h}) +
\dfrac{4}{2\Delta t} \mathscr{M}(u_{i,h}^{n-1}, q_{i,h}) -
\dfrac{1}{2\Delta t} \mathscr{M}(u_{i,h}^{n-2}, q_{i,h}).
\end{cases}
$$

右端的时间历史项包含两个历史层的线性组合，系数分别为 $4/(2\Delta t)$ 和 $-1/(2\Delta t)$。相比 BDF1，BDF2 的左端质量矩阵系数由 $1/\Delta t$ 变为 $3/(2\Delta t)$，右端增加了 $u^{n-2}$ 贡献项。

##### 与 BDF1 的统一形式

为便于程序实现，可将 BDF1 与 BDF2 写成统一的多步形式：
$$
\frac{1}{\Delta t} \mathscr{M}\!\left( \sum_{k=0}^{p} \alpha_k u^{n-k},\, q \right),
$$
其中 $p$ 为方法阶数（BDF1: $p=1$，BDF2: $p=2$），系数为：

| 方法 | $\alpha_0$ | $\alpha_1$ | $\alpha_2$ |
|------|-----------|-----------|-----------|
| BDF1 | $1$       | $-1$      | —         |
| BDF2 | $3/2$     | $-2$      | $1/2$     |

矩阵层面，左端质量块乘 $\alpha_0 / \Delta t$，右端历史项为 $-\sum_{k=1}^{p} (\alpha_k / \Delta t) \cdot M \mathbf{u}^{n-k}$。

---

### 1.8 全局离散鞍点矩阵结构

#### 单元局部矩阵分块（单组分，单单元）

设单元通量自由度为 $N_J$，浓度自由度为 $N_u$。单元局部鞍点矩阵为 $(N_J + N_u) \times (N_J + N_u)$ 的分块矩阵：

$$
K_E(\mathbf{u}_h) =
\begin{bmatrix}
K_C + K_A(\mathbf{u}_h) + K_S & W^\top \\
W & \dfrac{1}{\Delta t} M_E
\end{bmatrix}
$$

其中：
- $K_C$：线性扩散相容刚度（常数，预计算）
- $K_A(\mathbf{u}_h)$：非线性交叉扩散相容刚度（依赖当前浓度，每步更新）
- $K_S$：虚拟元稳定化刚度（仅自扩散对角部分，尺度因子为单元平均）
- $W$：散度耦合矩阵，$W_{\alpha i} = \mathscr{B}(\hat{\phi}_i, \hat{q}_\alpha)$（与 Darcy 方程完全相同，无几何修正）
- $M_E$：曲边单元加权质量矩阵（含雅可比权重 $J$，可预计算）

**多组分扩展**：$n$ 组分的全局矩阵为按组分的块对角结构，不同组分之间通过 $K_A$ 的非对角块耦合。

#### 单元局部右端向量
$$
\mathbf{F}_E =
\begin{bmatrix}
\mathbf{0} \\
F_{Q,E}(\hat{q}_\alpha) +
\dfrac{1}{\Delta t}
\mathscr{M}_E(\mathbf{u}_h^{n-1}, \hat{q}_\alpha)
\end{bmatrix}
$$

上块为通量方程右端（齐次边界条件下为零向量），下块为体积源项加上**时间历史贡献（携带 $1/\Delta t$ 因子）**。

#### BDF2 的矩阵形式扩展

对于 BDF2 时间离散（$n \geq 2$），单元局部鞍点矩阵的质量块系数由 $1/\Delta t$ 替换为 $3/(2\Delta t)$：

$$
K_E(\mathbf{u}_h) =
\begin{bmatrix}
K_C + K_A(\mathbf{u}_h) + K_S & W^\top \\
W & \dfrac{3}{2\Delta t} M_E
\end{bmatrix}
$$

单元右端向量的时间历史项包含两个历史层：

$$
\mathbf{F}_E =
\begin{bmatrix}
\mathbf{0} \\
F_{Q,E}(\hat{q}_\alpha) +
\dfrac{4}{2\Delta t}
\mathscr{M}_E(\mathbf{u}_h^{n-1}, \hat{q}_\alpha) -
\dfrac{1}{2\Delta t}
\mathscr{M}_E(\mathbf{u}_h^{n-2}, \hat{q}_\alpha)
\end{bmatrix}
$$

矩阵结构和耦合形式与 BDF1 完全相同，仅质量块标量系数与右端历史项不同，因此代码层面可通过参数化 BDF 阶数实现灵活切换。第一步（$n=1$）退化为 BDF1。

---

### 1.9 非线性迭代求解：Picard 不动点迭代

由于 $A(\mathbf{u})$ 的非线性，离散系统为非线性鞍点问题。采用 Picard 线性化迭代：固定浓度值，将非线性项显式处理，求解线性化鞍点系统，反复迭代直到收敛。

**算法 1：Picard 迭代求解 Maxwell-Stefan**

1. **初始化**：给定初始猜测 $\mathbf{u}_h^{(0)} = \mathbf{u}_h^{n-1}$
2. **迭代直到收敛**：对于 $k = 0, 1, 2, \dots$
   a. 固定浓度 $\mathbf{u}_h^{(k)}$，计算非线性扩散矩阵 $A(\mathbf{u}_h^{(k)})$：
      - 对角元：$A_{ii} = \sum_{j \neq i} \bar{c}_{ij} u_j^{(k)}$
      - 非对角元：$A_{ij} = -\bar{c}_{ji} u_i^{(k)}$（$i \neq j$）
   b. 组装全局线性化鞍点矩阵：
      $$
      K_h^{(k)} =
      \begin{bmatrix}
      K_C + K_A(\mathbf{u}_h^{(k)}) + K_S & W^\top \\
      W & \dfrac{1}{\Delta t} M
      \end{bmatrix}
      $$
   c. 求解线性鞍点系统：
      $$
      K_h^{(k)}
      \begin{bmatrix}
      \mathbf{J}_h^{(k+1)} \\
      \mathbf{u}_h^{(k+1)}
      \end{bmatrix} =
      \begin{bmatrix}
      \mathbf{0} \\
      F_Q + \dfrac{1}{\Delta t} M \mathbf{u}_h^{n-1}
      \end{bmatrix}
      $$
   d. **收敛检验**：
      $$
      \|\mathbf{u}_h^{(k+1)} - \mathbf{u}_h^{(k)}\|_{L^2} < \epsilon_{\text{tol}}
      $$
3. **收敛后**：令 $\mathbf{J}_h^{n} \leftarrow \mathbf{J}_h^{(k+1)}$，$\mathbf{u}_h^{n} \leftarrow \mathbf{u}_h^{(k+1)}$，进入下一时间步

**收敛性说明**：
- 当 $A(\mathbf{u})$ 的非线性较弱或 $\Delta t$ 较小时，Picard 迭代通常线性收敛
- 由于对角元恒正且矩阵具有 M 矩阵结构，Picard 迭代通常稳定
- 强非线性情形可改用 Newton 迭代（需组装雅可比矩阵）或 JFNK 方法

---

### 1.10 边界条件离散处理

#### 法向通量 Neumann 边界（强制约束）
属于 H(div) 空间的本质边界条件，直接约束边矩自由度。对物理曲边界 $\Gamma_N$ 上的边 $e$，其拉回参考边 $\hat{e}$ 上的边矩约束：
$$
\int_{\hat{e}}
(\hat{J}_i \cdot \hat{\boldsymbol{n}}) \,
\hat{m}_\ell \, d\hat{s} =
\int_{\hat{e}}
\hat{g}_{N,i} \, \hat{m}_\ell \, d\hat{s},
\quad
\ell = 0, \dots, k
$$

其中 $\hat{g}_{N,i} = g_{N,i} \circ \boldsymbol{x}$ 为拉回后的边界通量数据。所有几何因子已包含在 Piola 变换中，积分公式与直边完全相同。

实现方式：边界自由度降维消元，修正右端向量 $F \leftarrow F - K U_{\text{bd}}$。

**⚠️ 注意**：
- 通量检验空间 $\boldsymbol{V}$ 定义了齐次 Neumann 边界（$J_i \cdot n = 0$ on $\Gamma_N$）
- 非齐次 Neumann 边界 $J_i \cdot n = g_{N,i}$ 通过上述强制约束方式施加

#### 浓度 Dirichlet 边界（自然边界）
进入速度方程的边界泛函 $F_V(\boldsymbol{v})$，通过曲边界积分实现：
$$
F_{V,E,i}(\hat{v}_i) =
-\int_{\hat{\Gamma}_{D,E}}
\hat{g}_{D,i} \,
(\hat{v}_i \cdot \hat{\boldsymbol{n}}) \, d\hat{s}
$$

该式不显含几何因子，与直边公式相同。

---

### 1.11 后处理与误差估计

#### L² 投影恢复多项式场
数值通量 $\boldsymbol{J}_h$ 为虚拟元函数，通过 L² 投影恢复为显式多项式场以便后处理：
$$
\Pi_k^0 \boldsymbol{J}_h =
\sum_\alpha c_\alpha \boldsymbol{g}_\alpha,
\quad
c_\alpha =
\sum_i (G^{-1} B)_{\alpha i} \, J_{h,i}
$$

浓度 $\boldsymbol{u}_h$ 本身就是分片多项式，无需额外投影。

#### 误差范数计算（计算区域上）
1. **浓度 L² 误差**：
   $$
   \|\boldsymbol{u}_h - \hat{\boldsymbol{u}}_{\text{exact}}\|_{L^2(\hat{\Omega})}^2 =
   \sum_E \int_{\hat{E}}
   |\boldsymbol{u}_h - \hat{\boldsymbol{u}}_{\text{exact}}|^2
   \, d\hat{\boldsymbol{x}}
   $$

2. **通量 L² 误差**：
   $$
   \|\boldsymbol{J}_h - \hat{\boldsymbol{J}}_{\text{exact}}\|_{L^2(\hat{\Omega})}^2 =
   \sum_E \int_{\hat{E}}
   |\Pi_k^0 \boldsymbol{J}_h - \hat{\boldsymbol{J}}_{\text{exact}}|^2
   \, d\hat{\boldsymbol{x}}
   $$

3. **通量散度 L² 误差**：
   $$
   \|\nabla \cdot (\boldsymbol{J}_h - \boldsymbol{J}_{\text{exact}})\|_{L^2(\Omega)}^2 =
   \sum_E \int_{\hat{E}}
   \left|
   \frac{1}{J} \hat{\nabla} \cdot
   (\Pi_k^0 \hat{\boldsymbol{J}}_h - \hat{\boldsymbol{J}}_{\text{exact}})
   \right|^2
   J \, d\hat{\boldsymbol{x}}
   $$

#### 预期收敛阶
在解足够正则、网格满足形状正则条件，且时间步长充分小（时间误差可忽略）时，理论收敛阶为：
- 浓度 $L^2$ 误差：$\mathcal{O}(h^{k+1})$
- 通量 $L^2$ 误差：$\mathcal{O}(h^{k+1})$（超收敛，数值上通常观察到）
- 通量散度 $L^2$ 误差：$\mathcal{O}(h^k)$

> **说明**：对标准混合元，通量的最优收敛阶为 $O(h^k)$。在虚拟元方法中，当使用投影算子和适当的稳定化时，通量的 $L^2$ 误差常表现出 $O(h^{k+1})$ 的超收敛行为，与数值实验观察一致。

##### 时间收敛阶

时间误差由时间离散格式决定：

| 时间格式 | 时间收敛阶 | 启动方式 |
|---------|-----------|---------|
| BDF1（向后欧拉） | $O(\Delta t)$ | 自启动 |
| BDF2 | $O(\Delta t^2)$ | 第一步用 BDF1 启动 |

当空间网格充分细（空间误差可忽略）时，全局 $L^2$ 误差由时间误差主导，表现为相应的时间收敛阶。实际计算中需根据空间精度 $k$ 匹配时间步长，使空间误差与时间误差同量级以达到计算效率最优。

---

## 第二部分 代码架构设计与实现

### 2.1 与 Darcy 方程的架构复用性对比

| 模块 | DarcySolver | MsSolver | 复用策略 |
|------|-------------|----------|----------|
| HdivMatrix 通用投影 | ✅ | ✅ | 直接复用基类 |
| 等参映射与 Piola 变换 | ✅ | ✅ | 共享 `IsoparametricMapping` 接口 |
| 系数加权 Gram 矩阵 | $G_{\kappa^{-1}}$ | $G_{A(u)}$ | 派生实现：Darcy 用渗透率，MS 用扩散矩阵 |
| 一致性刚度 $K_{ac}$ | $\Pi^\top G_{\kappa^{-1}} \Pi$ | $\Pi^\top (G_C + G_A) \Pi$ | 通用矩阵乘法框架 |
| 稳定化刚度 $K_{as}$ | 尺度 $\|\bar{\kappa}^{-1}\|_F$ | 尺度 $\|A_{ii}\|_{\text{单元平均}}$ | 共享 $(I-\Pi)^\top (I-\Pi)$ 模板 |
| 散度耦合 $W$ | ✅ | ✅ | 完全相同 |
| 质量矩阵 $M$ | ❌ | ✅ | MS 新增 |
| Lagrange 约束 | 压力零均值 | 通常不需要 | 条件编译 |
| 边界条件处理 | ✅ | ✅ | 共享降维消元代码 |
| 时间步进 | 稳态/瞬态可选 | 必选瞬态 | 基类 `TimeDependentSolver` 抽象 |
| 非线性迭代 | ❌ | ✅ | MS 新增 Picard 驱动 |
| PETSc 线性求解 | GMRES + Schur 补 | GMRES + Schur 补 | 共享预条件器实现 |

**推荐重构**：
- 将 `DarcySolver` 中的通用组装代码提取到 `Hdiv saddle point assembler` 基类
- `DarcySolver` 和 `MsSolver` 均派生自该基类，仅覆盖系数相关的组装方法

---

### 2.2 新增文件清单与功能定位

```
vem/CurvedVEM_Darcy/
├── solver/
│   ├── MsDiffusionMatrix.h      # 非线性扩散矩阵评估器
│   ├── MsDiffusionMatrix.cpp
│   ├── MsAssembler.h            # MS 方程单元组装器
│   ├── MsAssembler.cpp
│   ├── MsSolver.h               # MS 全局求解器（时间步进+非线性迭代）
│   └── MsSolver.cpp
├── examples/
│   ├── ms_problem.h             # MS 算例定义接口
│   ├── ms_problem.cpp
│   ├── ms_sine_pulse.h         # 正弦脉冲扩散算例
│   └── ms_sine_pulse.cpp
└── tests/
    ├── test_ms_matrix_main.cpp  # 单元矩阵测试
    ├── test_ms_solver_main.cpp  # 求解器端到端测试
    └── test_ms_convergence_main.cpp  # 收敛阶验证
```

---

### 2.3 核心类设计与数学对应关系

#### 类 1：MsDiffusionMatrix（非线性扩散矩阵评估器）

**对应数学**：$\bar{A}_{ij}(\boldsymbol{u}) = c^*\delta_{ij} + A_{ij}(\boldsymbol{u})$、$\mathbb{K}_\xi^{C} = \frac{c^*}{J}\mathbb{J}^\top\mathbb{J}$、$\mathbb{K}_\xi^{A,ij} = \frac{A_{ij}(\boldsymbol{u})}{J} \mathbb{J}^\top \mathbb{J}$

**功能**：
- 存储二元扩散系数倒数矩阵 $c_{ij}$
- 自动计算 $c^* = \min(c_{ij})$ 和相对扩散系数 $\bar{c}_{ij} = c_{ij} - c^*$
- 评估给定点的完整扩散矩阵 $\bar{A}(\boldsymbol{u}) = c^* I + A(\boldsymbol{u})$ 及其非线性部分 $A(\boldsymbol{u})$
- 计算曲几何修正的线性扩散张量 $\mathbb{K}_\xi^{C}$（对应 $\mathscr{C}$）和非线性交叉扩散张量 $\mathbb{K}_\xi^{A,ij}$（对应 $\mathscr{A}_{ij}$）
- 提供自扩散稳定化尺度因子（单元平均对角元）

**关键接口**：
```cpp
class MsDiffusionMatrix {
public:
    // 构造：传入组分数目、完整的二元扩散系数倒数矩阵 c_ij
    // 内部自动计算：c* = min(c_ij), bar_c_ij = c_ij - c*
    MsDiffusionMatrix(int num_species,
                      const std::vector<std::vector<double>>& binary_coeffs);

    // 获取参数
    int numSpecies() const { return n_species_; }
    double getCStar() const { return c_star_; }
    double getBarC(int i, int j) const { return bar_c_ij_[i][j]; }

    // 给定点评估全扩散矩阵（含线性项）：
    //   bar_A_ii = c* + sum_{k != i} bar_c_ik * u_k
    //   bar_A_ij = -bar_c_ji * u_i   (i != j)
    void evaluateFullMatrix(const std::vector<double>& u,
                            std::vector<std::vector<double>>& bar_A) const;

    // 仅评估非线性部分 A(u) = bar_A(u) - c*I
    //   A_ii = sum_{k != i} bar_c_ik * u_k
    //   A_ij = -bar_c_ji * u_i   (i != j)
    void evaluateNonlinearPart(const std::vector<double>& u,
                               std::vector<std::vector<double>>& A) const;

    // 曲几何修正后的扩散张量 K_xi^{A,ij}
    //   K_xi^{A,ij} = (bar_A_ij(u) / detJ) * J^T * J
    void evaluateGeometricallyCorrected(
        const std::vector<double>& u,
        double detJ,
        const Tensor2D& J_mat,  // 雅可比矩阵
        std::vector<std::vector<Tensor2D>>& K_xi_A) const;

    // 获取第 i 组分的稳定化尺度因子（单元平均对角元）
    double getStabilizationScale(int species,
                                 const std::vector<double>& u_elem_avg) const;

private:
    int n_species_;
    double c_star_;                           // c* = min(c_ij)
    std::vector<std::vector<double>> bar_c_ij_;  // bar_c_ij = c_ij - c*
};
```

**使用示例（构造三组分扩散矩阵）**：
```cpp
// 三组分二元扩散系数倒数矩阵（c_ij）
std::vector<std::vector<double>> c_ij = {
    {-1.0,  0.1,  0.2},   // c_12 = 0.1, c_13 = 0.2
    { 0.1, -1.0,  2.0},   // c_21 = 0.1, c_23 = 2.0
    { 0.2,  2.0, -1.0}    // c_31 = 0.2, c_32 = 2.0
};

// 构造非线性扩散矩阵
MsDiffusionMatrix diffusion(3, c_ij);

// 验证参数
assert(diffusion.getCStar() == 0.1);                  // min(c_ij)
assert(diffusion.getBarC(0, 1) == 0.0);                // c_12 - c* = 0.1 - 0.1 = 0
assert(diffusion.getBarC(0, 2) == 0.1);                // c_13 - c* = 0.2 - 0.1 = 0.1
assert(diffusion.getBarC(1, 2) == 1.8);                // c_23 - c* = 2.0 - 0.1 = 1.8

// 给定点评估矩阵
std::vector<double> u = {0.25, 0.25, 0.5};  // 浓度
std::vector<std::vector<double>> bar_A(3, std::vector<double>(3));
diffusion.evaluateFullMatrix(u, bar_A);
// 结果：
//   bar_A[0][0] = 0.1 + 0.0*0.25 + 0.1*0.5 = 0.15
//   bar_A[0][1] = 0
//   bar_A[0][2] = -0.1*0.25 = -0.025
//   bar_A[1][1] = 0.1 + 0.0*0.25 + 1.8*0.5 = 1.0
//   bar_A[1][2] = -1.8*0.25 = -0.45
//   bar_A[2][2] = 0.1 + 0.1*0.25 + 1.8*0.25 = 0.575
```

---

#### 类 2：MsAssembler（MS 方程单元组装器）

**对应数学**：单元局部矩阵 $K_E(\boldsymbol{u})$、右端向量 $\boldsymbol{F}_E$、全局组装

**功能**：
- 复用 `HdivMatrix` 的通用投影矩阵
- 组装线性扩散刚度 $K_C$
- 组装当前浓度下的非线性交叉扩散刚度 $K_A(\boldsymbol{u})$
- 组装稳定化刚度 $K_S$（自扩散部分，单元平均尺度）
- 组装散度耦合块 $W$（复用 Darcy）
- 组装质量矩阵 $M$（带雅可比权重）
- 组装单元右端向量（源项 + 时间历史 + 边界载荷）
- 局部 → 全局自由度映射（多组分）

**关键接口**：
```cpp
class MsAssembler {
public:
    MsAssembler(const HdivMatrix& hdiv_base,
                const MsDiffusionMatrix& diffusion,
                const IsoparametricMapping& mapping);

    // ========== 预计算常数矩阵（与 u 无关） ==========

    // 线性扩散相容刚度 K_C（所有组分的块对角矩阵）
    AutoPetscMat assembleLinearConsistency(int elem) const;

    // 稳定化刚度 K_S（仅自扩散部分，尺度因子可更新）
    AutoPetscMat assembleStabilization(int elem,
        const std::vector<double>& scale_per_species) const;

    // 质量矩阵 M（带雅可比权重，标量块对角）
    AutoPetscMat assembleMassMatrix(int elem) const;

    // 散度耦合矩阵 W（与 Darcy 完全相同，多组分块对角）
    AutoPetscMat assembleDivergenceCoupling(int elem) const;

    // ========== 非线性更新矩阵（依赖当前浓度 u） ==========

    // 非线性交叉扩散相容刚度 K_A(u)
    AutoPetscMat assembleNonlinearConsistency(
        int elem,
        const std::vector<double>& elem_u_avg) const;  // 单元平均浓度

    // 完整单元鞍点矩阵（Picard 迭代线性化矩阵）
    AutoPetscMat assembleLocalSaddleMatrix(
        int elem,
        const std::vector<double>& elem_u_avg,
        double dt) const;

    // 单元右端向量（源项 + 时间历史）
    AutoPetscVec assembleLocalRhs(
        int elem,
        const std::vector<double>& u_prev,  // 上一时间步
        double dt,
        const std::function<double(int, double, double)>& source) const;

    // ========== 全局组装 ==========
    void assembleGlobalSystem(
        const std::vector<std::vector<double>>& elem_u_avg,  // 各单元平均浓度
        double dt,
        Mat global_matrix,
        Vec global_rhs) const;

    // ========== 多组分自由度映射 ==========
    PetscInt getGlobalDof(int species, int elem,
                          DofType type, int index) const;
};
```

---

#### 类 3：MsSolver（全局求解器，时间步进 + 非线性迭代）

**对应数学**：时间离散、Picard 迭代、边界条件、线性求解、后处理误差

**功能**：
- 时间步循环管理
- Picard 非线性迭代控制（单元平均浓度评估）
- 边界条件施加（Neumann 通量强制约束 + Dirichlet 浓度自然边界）
- PETSc 线性鞍点求解（GMRES + Schur 补预条件）
- 后处理 L² 投影恢复多项式场
- 误差计算与收敛阶验证

**关键接口**：
```cpp
class MsSolver {
public:
    MsSolver(const HdivMatrix& hdiv,
             MsAssembler& assembler,
             const MsProblem& problem);

    // 设置时间步进参数
    void setTimeParameters(double t_final, double dt);

    // 设置非线性迭代参数
    void setNonlinearParameters(int max_picard_iter,
                                double tolerance = 1e-8);

    // ========== 主求解入口 ==========

    // 瞬态求解：从 t=0 积分到 t_final
    bool solveTransient();

    // 单时间步求解（含 Picard 迭代）
    bool solveTimeStep(double t_prev, double dt,
                       const Vec u_prev, Vec u_next, Vec J_next);

    // ========== 非线性迭代内核 ==========

    // 一次 Picard 迭代：固定 u_old，求解线性鞍点得到 (J_new, u_new)
    bool picardIteration(const Vec u_old, double dt,
                         Vec J_new, Vec u_new);

    // 计算每个单元的平均浓度（用于非线性评估）
    void computeElementAverageConcentration(
        const Vec u,
        std::vector<std::vector<double>>& elem_u_avg) const;

    // 非线性收敛检验
    bool checkConvergence(const Vec u_old, const Vec u_new) const;

    // ========== 边界条件 ==========

    // 施加法向通量边界（强制约束，降维消元）
    void applyNeumannBoundary(Vec boundary_values,
                              IS& free_dofs,
                              Vec adjusted_rhs);

    // 施加浓度 Dirichlet 边界（自然边界，修正右端）
    void applyDirichletBoundary(Vec rhs);

    // ========== 后处理与误差 ==========

    // 通量 L² 投影恢复多项式系数
    void projectFluxToPolynomial(const Vec J_dofs,
        std::vector<std::vector<double>>& poly_coeffs) const;

    // 计算浓度 L² 误差
    double computeL2ErrorConcentration(const Vec u_num,
        double t_exact) const;

    // 计算通量 L² 误差
    double computeL2ErrorFlux(const Vec J_num,
        double t_exact) const;
};
```

---

### 2.4 多组分自由度映射方案

设网格有 $N_{\text{elem}}$ 个单元，每个单元 $n_{\text{edge}}$ 条边，多项式阶数 $k$。

**全局自由度编号规则**：

1. **通量自由度（边矩 + 内部矩）**：
   - 第 $s$ 组分、第 $e$ 条全局边、第 $\ell$ 阶矩的编号：
     $$
     \text{idx} = s \cdot N_{\text{edges}} \cdot (k+1) + \ell \cdot N_{\text{edges}} + e
     $$
   - 第 $s$ 组分、第 $i$ 个单元、第 $m$ 个内部梯度矩的编号：
     $$
     \text{idx} = n_{\text{species}} \cdot N_{\text{edges}} \cdot (k+1) +
     s \cdot N_{\text{elem}} \cdot \dim(P_{k-1}) +
     i \cdot \dim(P_{k-1}) + m
     $$

2. **浓度自由度**：
   - 第 $s$ 组分、第 $i$ 个单元、第 $m$ 个浓度多项式系数的编号：
     $$
     \text{idx} = \text{total_flux_dofs} +
     s \cdot N_{\text{elem}} \cdot \dim(P_{k-1}) +
     i \cdot \dim(P_{k-1}) + m
     $$

3. **Lagrange 乘子（若需要压力/浓度零均值约束）**：
   - 可选，通常 MS 方程不需要全局零均值约束（与 Darcy 不同）

---

### 2.5 完整求解流程伪代码

```
程序入口：main()
├── 1. 读取网格，构造直边计算网格数据结构
├── 2. 构造曲边等参映射（如正弦扰动映射）
├── 3. 初始化 HdivMatrix：计算通用投影矩阵 D, G, W, H, H*, B, Π
├── 4. 构造 MsDiffusionMatrix：设定 d* 和非线性函数
├── 5. 构造 MsAssembler：绑定 hdiv, diffusion, mapping
├── 6. 构造 MsProblem：设定精确解、源项、边界条件
├── 7. 构造 MsSolver：绑定 assembler, problem
├── 8. 设置时间参数：t_final, dt
├── 9. 设置非线性迭代参数：max_iter, tol
├── 10. 初始化解向量 u0 = 初始条件
├── 11. 时间步进循环：
│    for n = 0; n < N_steps; n++
│    │
│    │    Picard 迭代（固定点迭代）：
│    │    ├── u^{(0)} = u^n
│    │    ├── for k = 0; k < max_iter; k++
│    │    │    ├── 由 u^{(k)} 计算每个单元的平均浓度
│    │    │    ├── MsAssembler 组装全局鞍点矩阵 K(u^{(k)})
│    │    │    ├── 组装右端向量 F = F_source + (1/dt) M u^n
│    │    │    ├── 施加边界条件（降维消元）
│    │    │    ├── PETSc GMRES + Schur 补预条件求解
│    │    │    ├── 得到 (J^{(k+1)}, u^{(k+1)})
│    │    │    └── if ||u^{(k+1)} - u^{(k)}|| < tol: break
│    │    └── end
│    │
│    │    u^{n+1} = u^{(k+1)}, J^{n+1} = J^{(k+1)}
│    │    每 10 步输出误差（若有精确解）
│    │    可选：保存可视化数据
│    └── end
└── 12. 输出最终误差，程序退出
```

---

## 第三部分 数值验证算例

### 3.0 官方测试算例：三组分正弦脉动扩散

#### 3.0.1 曲边网格映射

逻辑坐标（计算域）$(\xi, \eta) \in [0,1]^2$ 到物理坐标 $(x, y)$ 的映射：

$$
\begin{cases}
x = \xi + 0.1\sin\left(2\pi \eta + \dfrac{\pi}{3}\right)\\
y = \eta + 0.2\sin\left(2\pi \xi + \dfrac{\pi}{4}\right)
\end{cases}
$$

**雅可比矩阵**：
$$
\mathbb{J}(\xi, \eta) =
\nabla_{(\xi,\eta)} \boldsymbol{x} =
\begin{pmatrix}
\dfrac{\partial x}{\partial \xi} & \dfrac{\partial x}{\partial \eta} \\
\dfrac{\partial y}{\partial \xi} & \dfrac{\partial y}{\partial \eta}
\end{pmatrix}
=
\begin{pmatrix}
1 & 0.2\pi \cos\left(2\pi \eta + \dfrac{\pi}{3}\right) \\
0.4\pi \cos\left(2\pi \xi + \dfrac{\pi}{4}\right) & 1
\end{pmatrix}
$$

**雅可比行列式**：
$$
J(\xi, \eta) = \det(\mathbb{J}) =
1 - 0.08\pi^2 \cos\left(2\pi \xi + \dfrac{\pi}{4}\right)
\cos\left(2\pi \eta + \dfrac{\pi}{3}\right)
$$

**C++ 代码可直接使用的表达式**：

```cpp
// 物理坐标映射
double x = xi + 0.1 * std::sin(2 * M_PI * eta + M_PI / 3.0);
double y = eta + 0.2 * std::sin(2 * M_PI * xi + M_PI / 4.0);

// 雅可比矩阵元素
double J_00 = 1.0;
double J_01 = 0.2 * M_PI * std::cos(2 * M_PI * eta + M_PI / 3.0);
double J_10 = 0.4 * M_PI * std::cos(2 * M_PI * xi + M_PI / 4.0);
double J_11 = 1.0;

// 雅可比行列式
double detJ = J_00 * J_11 - J_01 * J_10;
// 验证：detJ = 1 - 0.08 * M_PI * M_PI *
//          std::cos(2 * M_PI * xi + M_PI / 4.0) *
//          std::cos(2 * M_PI * eta + M_PI / 3.0);

// 雅可比逆矩阵 (2x2 直接公式)
double inv_detJ = 1.0 / detJ;
double invJ_00 =  J_11 * inv_detJ;
double invJ_01 = -J_01 * inv_detJ;
double invJ_10 = -J_10 * inv_detJ;
double invJ_11 =  J_00 * inv_detJ;
```

#### 3.0.2 物理域精确解（三组分）

$$
\begin{cases}
c_1(x, y, t) = 0.25\sin(2\pi x)\sin(8\pi t) + 0.25\\
c_2(x, y, t) = 0.25\sin(3\pi y)\sin(6\pi t) + 0.25\\
c_3(x, y, t) = 1 - c_1(x, y, t) - c_2(x, y, t)
\end{cases}
$$

**浓度和约束**：$c_1 + c_2 + c_3 = 1$，恒满足质量守恒。

**初始条件（$t=0$）**：
$$
c_1(x, y, 0) = 0.25, \quad
c_2(x, y, 0) = 0.25, \quad
c_3(x, y, 0) = 0.5
$$

#### 3.0.3 扩散本构与通量（与第 1 部分一致的精确形式）

**完整本构方程（Fick 型推广形式）**：
$$
\nabla c_i + \sum_{j=1}^n \bar{A}_{ij}(\boldsymbol{c}) J_j = 0,
\quad \text{或等价地} \quad
\nabla \boldsymbol{c} + \bar{A}(\boldsymbol{c})\boldsymbol{J} = \mathbf{0}.
$$

其中完整扩散矩阵 $\bar{A}(\boldsymbol{c}) = c^* I + A(\boldsymbol{c})$，其元素为：

**对角元**（$i = 1, 2, 3$）：
$$
\bar{A}_{ii}(\boldsymbol{c}) = c^* + \sum_{j \neq i} \bar{c}_{ij} c_j
$$

**非对角元**（$i \neq j$）：
$$
\bar{A}_{ij}(\boldsymbol{c}) = -\bar{c}_{ji} c_i
$$

**线性-非线性分裂形式**（对应数值离散的 $\mathscr{C} + \mathscr{A}$ 分解）：
$$
\nabla c_i + c^* J_i + \sum_{j=1}^n A_{ij}(\boldsymbol{c}) J_j = 0,
$$

其中非线性交叉扩散矩阵 $A(\boldsymbol{c})$ 的对角元为 $\sum\limits_{j \neq i} \bar{c}_{ij} c_j$，非对角元为 $-\bar{c}_{ji} c_i$。

**二元扩散参数（算例设定）**：

设三组分系统的二元扩散系数倒数为（值越大表示扩散越慢）：
- $c_{12} = 0.1$（组分 1-2 之间扩散最快）
- $c_{13} = 0.2$（组分 1-3 之间扩散中等）
- $c_{23} = 2.0$（组分 2-3 之间扩散最慢，相差一个数量级）

完整的对称矩阵形式：
$$
c_{ij} =
\begin{pmatrix}
- & 0.1 & 0.2 \\
0.1 & - & 2.0 \\
0.2 & 2.0 & -
\end{pmatrix}
$$

其中 $c_{ij} = c_{ji}$ 为第 $i$ 组分与第 $j$ 组分之间的二元扩散系数的倒数。

**参数计算**：
- 最小扩散系数倒数：$c^* = \min(c_{ij}) = 0.1$
- 相对扩散系数：$\bar{c}_{ij} = c_{ij} - c^*$
$$
\bar{c}_{ij} =
\begin{pmatrix}
- & 0.0 & 0.1 \\
0.0 & - & 1.8 \\
0.1 & 1.8 & -
\end{pmatrix}
$$

**矩阵显式形式（3 组分）**：
$$
\bar{A}(\boldsymbol{c}) =
\begin{pmatrix}
c^* + \bar{c}_{12} c_2 + \bar{c}_{13} c_3 & -\bar{c}_{21} c_1 & -\bar{c}_{31} c_1 \\
-\bar{c}_{12} c_2 & c^* + \bar{c}_{21} c_1 + \bar{c}_{23} c_3 & -\bar{c}_{32} c_2 \\
-\bar{c}_{13} c_3 & -\bar{c}_{23} c_3 & c^* + \bar{c}_{31} c_1 + \bar{c}_{32} c_2
\end{pmatrix}
$$

代入数值：
$$
\bar{A}(\boldsymbol{c}) =
\begin{pmatrix}
0.1 + 0.0 c_2 + 0.1 c_3 & -0.0 c_1 & -0.1 c_1 \\
-0.0 c_2 & 0.1 + 0.0 c_1 + 1.8 c_3 & -1.8 c_2 \\
-0.1 c_3 & -1.8 c_3 & 0.1 + 0.1 c_1 + 1.8 c_2
\end{pmatrix}
$$

**简化形式**（注意 $\bar{c}_{12} = 0$）：
$$
\bar{A}(\boldsymbol{c}) =
\begin{pmatrix}
0.1 + 0.1 c_3 & 0 & -0.1 c_1 \\
0 & 0.1 + 1.8 c_3 & -1.8 c_2 \\
-0.1 c_3 & -1.8 c_3 & 0.1 + 0.1 c_1 + 1.8 c_2
\end{pmatrix}
$$

**矩阵性质**：
- 稀疏结构（组分1与组分2之间无直接非线性耦合，$\bar{c}_{12} = 0$）
- 对角元恒正（M 矩阵性质，保证适定性）
- 每行和为 $c^*$（不是零，因为有背景扩散项）
- 对称正定（因为 $\bar{c}_{ij} = \bar{c}_{ji}$）
- 非对称强度：组分2-3之间强非线性耦合（$\bar{c}_{23} = 1.8$），组分1-3之间弱非线性耦合（$\bar{c}_{13} = 0.1$），组分1-2之间无线性耦合（$\bar{c}_{12} = 0$）

**通量求解**：
$$
\boldsymbol{J} = -\bar{A}(\boldsymbol{c})^{-1} \nabla \boldsymbol{c}
$$

由于 $c^* > 0$，矩阵 $\bar{A}(\boldsymbol{c})$ 非奇异，可直接求逆（不需要广义逆）。

#### 3.0.4 体积源项（右端项）推导

由质量守恒方程：
$$
\frac{\partial c_i}{\partial t} + \nabla \cdot J_i = f_i
$$

得体积源项：
$$
f_i = \frac{\partial c_i}{\partial t} + \nabla \cdot J_i
$$

**组分 1 源项**：
$$
\begin{aligned}
\frac{\partial c_1}{\partial t} &=
0.25 \cdot 8\pi \sin(2\pi x) \cos(8\pi t) =
2\pi \sin(2\pi x) \cos(8\pi t) \\
\nabla c_1 &=
\begin{pmatrix}
0.5\pi \cos(2\pi x) \sin(8\pi t) \\
0
\end{pmatrix} \\
f_1 &= 2\pi \sin(2\pi x) \cos(8\pi t) + \nabla \cdot J_1
\end{aligned}
$$

**组分 2 源项**：
$$
\begin{aligned}
\frac{\partial c_2}{\partial t} &=
0.25 \cdot 6\pi \sin(3\pi y) \cos(6\pi t) =
1.5\pi \sin(3\pi y) \cos(6\pi t) \\
\nabla c_2 &=
\begin{pmatrix}
0 \\
0.75\pi \cos(3\pi y) \sin(6\pi t)
\end{pmatrix} \\
f_2 &= 1.5\pi \sin(3\pi y) \cos(6\pi t) + \nabla \cdot J_2
\end{aligned}
$$

**组分 3 源项**（由和约束自动满足）：
$$
f_3 = -f_1 - f_2
$$

验证：$\frac{\partial}{\partial t}(c_1 + c_2 + c_3) = 0$，$\nabla \cdot (J_1 + J_2 + J_3) = 0$，因此 $f_1 + f_2 + f_3 = 0$ 自动成立。

#### 3.0.5 法向通量边界条件推导

**边界映射**：

计算域四条边对应物理曲边：

| 计算域边界 | 物理曲边参数化 | 切向量 | 法向量 |
|-----------|----------------|--------|--------|
| 左边界 $\xi = 0$ | $x = 0.1\sin(2\pi \eta + \pi/3)$<br>$y = \eta + 0.2\sin(\pi/4)$ | $\boldsymbol{t}_\eta = \frac{\partial \boldsymbol{x}}{\partial \eta}$ | $\boldsymbol{n} = (-t_y, t_x)$ |
| 右边界 $\xi = 1$ | $x = 1 + 0.1\sin(2\pi \eta + \pi/3)$<br>$y = \eta + 0.2\sin(9\pi/4)$ | $\boldsymbol{t}_\eta = \frac{\partial \boldsymbol{x}}{\partial \eta}$ | $\boldsymbol{n} = (t_y, -t_x)$ |
| 下边界 $\eta = 0$ | $x = \xi + 0.1\sin(\pi/3)$<br>$y = 0 + 0.2\sin(2\pi \xi + \pi/4)$ | $\boldsymbol{t}_\xi = \frac{\partial \boldsymbol{x}}{\partial \xi}$ | $\boldsymbol{n} = (t_y, -t_x)$ |
| 上边界 $\eta = 1$ | $x = \xi + 0.1\sin(7\pi/3)$<br>$y = 1 + 0.2\sin(2\pi \xi + \pi/4)$ | $\boldsymbol{t}_\xi = \frac{\partial \boldsymbol{x}}{\partial \xi}$ | $\boldsymbol{n} = (-t_y, t_x)$ |

**法向通量计算**：

对每一条曲边界，通量的法向分量为：
$$
g_{N,i} = J_i \cdot \boldsymbol{n}
$$

其中 $J_i$ 由扩散本构关系确定：
$$
J_i = -\frac{1}{d^*} A(\boldsymbol{c})^{-1}_{ij} \nabla c_j
$$

**Piola 拉回后的法向通量**（计算域边积分使用）：

$$
\hat{J}_i \cdot \hat{\boldsymbol{n}} = J_i \cdot \boldsymbol{n} \cdot \frac{ds}{d\hat{s}}
$$

**简化实现方式**（推荐）：

由于 $\boldsymbol{J}_i = \frac{1}{J} \mathbb{J} \hat{\boldsymbol{J}}_i$ 且 $\boldsymbol{n} ds = J \mathbb{J}^{-\top} \hat{\boldsymbol{n}} d\hat{s}$，有：
$$
(J_i \cdot \boldsymbol{n}) ds = (\hat{J}_i \cdot \hat{\boldsymbol{n}}) d\hat{s}
$$

因此在计算域上的法向通量矩可以直接使用：
$$
\int_{\hat{e}} (\hat{J}_i \cdot \hat{\boldsymbol{n}}) \hat{m}_\ell d\hat{s} =
\int_e (J_i \cdot \boldsymbol{n}) \hat{m}_\ell \circ \boldsymbol{x}^{-1} ds
$$

**实现时的步骤**：
1. 在物理曲边上的高斯点求精确通量 $J_i(x, y, t)$
2. 求物理外法向量 $\boldsymbol{n}(x, y)$
3. 计算法向通量 $g_{N,i} = J_i \cdot \boldsymbol{n}$
4. 直接作为计算域对应边的边矩强制约束值

#### 3.0.6 计算域上的精确解（拉回形式）

在虚拟元求解中，所有量都在计算域 $(\xi, \eta)$ 上定义：

**浓度（标量直接复合）**：
$$
\begin{aligned}
\hat{c}_1(\xi, \eta, t) &=
0.25\sin\left(2\pi \left[\xi + 0.1\sin\left(2\pi \eta + \frac{\pi}{3}\right)\right]\right)
\sin(8\pi t) + 0.25 \\
\hat{c}_2(\xi, \eta, t) &=
0.25\sin\left(3\pi \left[\eta + 0.2\sin\left(2\pi \xi + \frac{\pi}{4}\right)\right]\right)
\sin(6\pi t) + 0.25 \\
\hat{c}_3(\xi, \eta, t) &= 1 - \hat{c}_1 - \hat{c}_2
\end{aligned}
$$

**通量（Piola 变换）**：
$$
\hat{J}_i(\xi, \eta, t) =
J(\xi, \eta) \cdot
\mathbb{J}^{-1}(\xi, \eta) \cdot
J_i(\boldsymbol{x}(\xi, \eta), t)
$$

**浓度梯度（链式法则）**：
$$
\nabla_{\boldsymbol{x}} c_i =
\mathbb{J}^{-\top}(\xi, \eta) \cdot
\nabla_{(\xi,\eta)} \hat{c}_i
$$

**源项（带雅可比权重）**：
$$
\hat{f}_i(\xi, \eta, t) =
J(\xi, \eta) \cdot
f_i(\boldsymbol{x}(\xi, \eta), t)
$$

#### 3.0.7 算例特点总结

| 特性 | 说明 |
|-----|------|
| **组分数量** | 3 组分（$c_1 + c_2 + c_3 = 1$ 约束） |
| **非线性强度** | 中等（扩散矩阵依赖浓度的线性组合） |
| **边界条件类型** | 全边界 Neumann 法向通量（4 条曲边） |
| **是否需要 Lagrange 乘子** | ❌ 不需要（瞬态问题，初始条件确定唯一解） |
| **时间依赖** | ✅ 强时间依赖（正弦脉动，频率 6π / 8π） |
| **曲边扰动幅度** | x 方向 0.1，y 方向 0.2（中等扰动） |
| **推荐网格分辨率** | 从 4×4 到 64×64 做收敛分析 |
| **推荐多项式阶数** | $k = 1$（平衡精度和计算量） |
| **时间步长建议** | $\Delta t = 0.001 \sim 0.01$（满足 CFL 条件） |
| **Picard 迭代收敛预期** | 通常 3-5 步收敛到 $10^{-8}$ |

---

### 3.1 manufactured solution：正弦脉冲扩散（备用算例）

**物理区域**：$\Omega = (0,1) \times (0,1)$ 的小正弦扰动曲边形

**等参映射**：
$$
x(\xi, \eta) = \xi + 0.05 \sin(2\pi \xi) \sin(2\pi \eta),
\quad
y(\xi, \eta) = \eta + 0.05 \sin(2\pi \xi) \sin(2\pi \eta)
$$

**精确解（双组分）**：
$$
\begin{cases}
u_1(\boldsymbol{x}, t) = \sin(\pi x) \sin(\pi y) e^{-t}, \\
u_2(\boldsymbol{x}, t) = \cos(\pi x) \cos(\pi y) e^{-t}.
\end{cases}
$$

**非线性扩散矩阵**（简单交叉扩散模型）：
$$
\Lambda(\boldsymbol{u}) =
\begin{pmatrix}
0 & \alpha u_2 \\
\alpha u_1 & 0
\end{pmatrix},
\quad
\alpha = 0.1
$$

线性扩散系数：$d^* = 1.0$

**精确通量**（由 MS 本构方程反解）：
$$
(d^* I + \Lambda) \boldsymbol{J} = -\nabla \boldsymbol{u}
\implies
\boldsymbol{J} = -(d^* I + \Lambda)^{-1} \nabla \boldsymbol{u}
$$

**体积源项**（由质量守恒方程确定）：
$$
f_i = \partial_t u_i + \nabla \cdot J_i
$$

**边界条件**：全边界 Neumann，按精确解指定法向通量

**时间参数**：$t_{\text{final}} = 0.5$，$\Delta t = 0.01$

**预期收敛阶**：
- $k=0$：浓度 $\mathcal{O}(h)$，通量 $\mathcal{O}(h)$
- $k=1$：浓度 $\mathcal{O}(h^2)$，通量 $\mathcal{O}(h^2)$
- $k=2$：浓度 $\mathcal{O}(h^3)$，通量 $\mathcal{O}(h^3)$

---

### 3.2 网格收敛性验证策略

1. **网格序列**：$N \times N$ 均匀网格，$N = 4, 8, 16, 32, 64$
2. **多项式阶数**：$k = 0, 1, 2$ 分别测试
3. **Picard 迭代容差**：$10^{-10}$（确保非线性误差可忽略）
4. **线性求解容差**：$10^{-12}$（远小于离散误差）
5. **每步输出**：L² 浓度误差、L² 通量误差、迭代次数、CPU 时间
6. **收敛阶计算**：$\log_2(e_{2h} / e_h)$

---

## 第四部分 实现优先级与开发计划

### Phase 1：核心基础设施（1-2 周）
1. ✅ 确认 `HdivMatrix` 多组分扩展能力
2. ✅ 设计并实现 `MsDiffusionMatrix` 接口
3. 🚧 实现单元质量矩阵 $M$ 组装
4. 🚧 实现多组分自由度映射
5. 🚧 线性扩散刚度 $K_C$ 组装（常数矩阵）
6. 🚧 单元测试验证：$K_C, M, W$ 的维度与对称性

### Phase 2：非线性组装与 Picard 迭代（2-3 周）
1. 实现非线性交叉扩散刚度 $K_A(\boldsymbol{u})$ 组装
2. 实现局部鞍点矩阵完整分块构造
3. 实现全局组装循环
4. 实现 Picard 迭代驱动与收敛检验
5. 单元测试：单单元非线性矩阵、小网格迭代收敛

### Phase 3：边界条件与时间步进（1-2 周）
1. 实现 Neumann 法向通量边界强制约束
2. 实现 Dirichlet 浓度自然边界
3. 实现时间步进循环
4. manufactured solution 算例实现

### Phase 4：后处理、验证与文档（1-2 周）
1. 实现 L² 误差计算（浓度、通量）
2. 网格收敛性验证
3. 性能 profiling 与优化
4. 完善文档、注释和使用说明

---

## 附录：问题修正清单与关键公式速查

### 数学审阅修正清单

| 优先级 | 问题 | 修正状态 |
|--------|------|----------|
| 🔴 高 | 双线性型与三线性型定义符号不统一 | ✅ 已统一为 $\mathscr{}$ 花体符号体系 |
| 🔴 高 | 第一个变分方程散度耦合项符号错误 | ✅ 已修正为 $+\mathscr{B}(v_i, u_i)$，与 LaTeX 参考一致 |
| 🔴 高 | 扩散矩阵分解符号前后不一致 | ✅ 已统一为 $\bar{A} = c^*I + A$ |
| 🔴 高 | 本构方程中 $c^*$ 重复计算 | ✅ 已修正：完整形式用 $\bar{A}$，分裂形式用 $c^* + A$ |
| 🔴 高 | 1.4 节单元积分拉回符号未更新 | ✅ 已全面修订，非线性项改用 $A(\mathbf{u})$ |
| 🔴 高 | 离散空间多项式次数错误（散度/浓度误为 $P_{k-1}$） | ✅ 已修正为 $P_k$ |
| 🔴 高 | 时间离散载荷向量缺失 1/Δt 因子 | ✅ 已修正 |
| 🔴 高 | 稳定项系数取单元平均值而非逐点值 | ✅ 已修正 |
| 🟡 中 | MS 本构为 Fick 型推广而非标准形式 | ✅ 已补充说明 |
| 🟡 中 | 稳定双线性型 $\mathscr{S}^E$ 未定义 | ✅ 已补充自由度差分定义 |
| 🟢 低 | 投影权重一致性问题 | ✅ 已说明稳定项与投影使用相同内积 |
| 🟢 低 | Neumann 非齐次边界处理说明 | ✅ 已补充 |
| 🟢 低 | 关键公式速查表符号与正文不一致 | ✅ 已更新为统一符号体系 |

**结论**：对照 LaTeX 理论稿全面修订后，方程表述、双线性型/三线性型定义、符号体系、空间离散、全局矩阵均已自洽，理论体系完整可直接实现。

---

### 关键公式速查表

| 物理量 | 物理域公式 | 参考元拉回公式 |
|--------|-----------|----------------|
| 标量梯度 | $\nabla_{\boldsymbol{x}} u_i$ | $\mathbb{J}^{-\top} \nabla_{\hat{\boldsymbol{x}}} \hat{u}_i$ |
| 通量 Piola 变换 | $J_i(\boldsymbol{x})$ | $\frac{1}{J} \mathbb{J} \hat{J}_i$ |
| 散度 | $\nabla \cdot J_i$ | $\frac{1}{J} \hat{\nabla} \cdot \hat{J}_i$ |
| 线性扩散双线性型 $\mathscr{C}$ | $c^* \int_\Omega J_i \cdot v_i dx$ | $\int_{\hat{E}} \frac{c^*}{J} (\mathbb{J}^\top \mathbb{J}) \hat{J}_i \cdot \hat{v}_i d\hat{x}$ |
| 非线性交叉扩散三线性型 $\mathscr{A}_{ij}$ | $\int_\Omega A_{ij}(\mathbf{u}) J_j \cdot v_i dx$ | $\int_{\hat{E}} \frac{A_{ij}(\hat{\mathbf{u}})}{J} (\mathbb{J}^\top \mathbb{J}) \hat{J}_j \cdot \hat{v}_i d\hat{x}$ |
| 散度耦合双线性型 $\mathscr{B}$ | $\int_\Omega (\nabla \cdot v_i) q_i dx$ | $\int_{\hat{E}} \hat{q}_i (\hat{\nabla} \cdot \hat{v}_i) d\hat{x}$（无几何修正） |
| 时间质量双线性型 $\mathscr{M}$ | $\int_\Omega u_i q_i dx$ | $\int_{\hat{E}} \hat{u}_i \hat{q}_i J d\hat{x}$ |
| Dirichlet 边界载荷 | $-\int_{\Gamma_D} g_{D,i} (v_i \cdot \boldsymbol{n}) ds$ | $-\int_{\hat{\Gamma}_D} \hat{g}_{D,i} (\hat{v}_i \cdot \hat{\boldsymbol{n}}) d\hat{s}$（无几何修正） |
| 加权 $L^2$ 投影正交性 | — | $\int_{\hat{E}} (\hat{\phi} - \Pi_k^0 \hat{\phi}) \cdot \boldsymbol{r} \, J d\hat{x} = 0, \quad \forall \boldsymbol{r} \in [\mathbb{P}_k]^2$ |
| 一致性刚度矩阵 | — | $\Pi^\top (G_C + G_A) \Pi$ |
| 稳定化刚度矩阵 | — | $c^* (I - \Pi)^\top (I - \Pi)$（仅自扩散对角项） |
| 时间离散右端 | — | $F_Q + \frac{1}{\Delta t} M u^{n-1}$ |

---

### 代码实现核对清单（实现时逐一确认）

- [ ] 质量矩阵 $M_E$ 带雅可比权重 $J$
- [ ] 时间离散鞍点矩阵的 $M / \Delta t$ 因子位置正确
- [ ] 右端时间历史项携带 $1/\Delta t$ 因子
- [ ] 稳定化系数使用单元平均而非逐点值
- [ ] 非线性扩散矩阵评估使用单元平均浓度
- [ ] 多组分自由度映射编号连续无冲突
- [ ] Piola 变换后的边界积分不重复乘几何因子
- [ ] 散度耦合项与 Darcy 实现完全一致

---

## 4. 算例描述

### 4.1 算例 1：三组分间断初值扩散（曲边域）

**算例类型**：三组分 Maxwell–Stefan 扩散，间断初始条件，零通量边界，无源项。

**计算域与曲边映射**：
- 计算域：单位正方形 $[0,1]^2$
- 物理域映射：正弦扰动等参映射
  $$
  x(\xi,\eta) = \xi + \varepsilon \sin(2\pi\eta), \qquad
  y(\xi,\eta) = \eta + \varepsilon \sin(2\pi\xi)
  $$
  其中扰动幅度 $\varepsilon = 0.05$。

**组分数与扩散参数**：
- 组分数 $N = 3$
- 二元扩散系数倒数：
  $c_{12} = 0.1,\quad c_{13} = 0.2,\quad c_{23} = 2.0$
- $c^* = \min\{c_{ij}\} = 0.1$
- $\bar{c}_{ij} = c_{ij} - c^*$

**初始条件**（计算域上定义，物理域通过逆映射获得）：
- 组分 1（$c_1$）：左下块 $(\xi\in[0,0.5),\ \eta\in[0,0.5])$ 取 0.8，其余区域取 0.1
- 组分 2（$c_2$）：左上块 $(\xi\in[0,0.5),\ \eta\in(0.5,1])$ 取 0.8，其余区域取 0.1
- 组分 3（$c_3$）：$c_3 = 1 - c_1 - c_2$，三组分之和恒为 1

**源项**：$f_i = 0$（$i=1,2,3$）

**边界条件**：所有边界为零法向通量（Neumann），即 $J_i \cdot \boldsymbol{n} = 0$。

**物理意义**：考察三组分在曲边物理域中的非对称扩散行为。由于扩散系数差异较大（$c_{23}$ 远大于 $c_{12}, c_{13}$），组分 2 与 3 之间的扩散交换远快于组分 1，可用于验证非线性 Maxwell–Stefan 系统的数值稳定性与保和性。

---

### 4.2 算例 2：制造解——正弦脉动扩散（曲边域）

**算例类型**：三组分 Maxwell–Stefan 扩散制造解算例，用于收敛阶验证。

**计算域与曲边映射**：与算例 1 完全相同（正弦扰动映射，$\varepsilon = 0.05$）。

**组分数与扩散参数**：与算例 1 完全相同（$c_{12}=0.1,\ c_{13}=0.2,\ c_{23}=2.0,\ c^*=0.1$）。

**精确解**（物理域上定义）：
$$
\begin{aligned}
c_1(x,y,t) &= 0.25 \sin(2\pi x) \sin(8\pi t) + 0.25, \\
c_2(x,y,t) &= 0.25 \sin(3\pi y) \sin(6\pi t) + 0.25, \\
c_3(x,y,t) &= 1 - c_1(x,y,t) - c_2(x,y,t).
\end{aligned}
$$
三组分之和恒为 1。浓度时均水平为 0.25，振幅 0.25，故取值范围为 $[0, 0.5]$，保证正定。

**通量本构**：
$$
\boldsymbol{J} = -\bar{A}(\boldsymbol{c})^{-1} \nabla \boldsymbol{c},
\qquad \bar{A}(\boldsymbol{c}) = c^* I + A(\boldsymbol{c}),
$$
其中 $A(\boldsymbol{c})$ 为非线性交叉扩散部分，非对角元 $A_{ij} = -\bar{c}_{ji}\, c_i$，对角元 $A_{ii} = \sum_{k\neq i} \bar{c}_{ik}\, c_k$。

**源项**：由制造解反演得到
$$
f_i = \frac{\partial c_i}{\partial t} + \nabla \cdot \boldsymbol{J}_i,
$$
散度项通过中心差分近似计算。

**边界条件**：法向通量边界，通量值由精确解计算（非零）。

**物理意义**：该算例具有解析的精确解，用于验证曲边 H(div) 混合虚元方法对 Maxwell–Stefan 方程的空间收敛阶。解在空间和时间上均为正弦振荡形式，可同时检验时间离散与空间离散的精度。

---

### 4.3 算例 3：半圆环三组分扩散（曲边域，趋于稳态）

**算例类型**：三组分 Maxwell–Stefan 扩散，径向分层间断初值，零通量边界，无源项，扩散趋于稳态。

**计算域与曲边映射**：
- 计算域：单位正方形 $[0,1]^2$
- 物理域映射：半圆环极坐标映射
  $$
  x(\xi,\eta) = r\cos\theta, \qquad
  y(\xi,\eta) = r\sin\theta,
  $$
  其中 $r = \xi + 0.5$，$\theta = \pi(\eta - 0.5)$。
- 物理域为右半圆环：内半径 0.5，外半径 1.5，角度范围 $\theta \in [-\pi/2, \pi/2]$。
- $\xi=0$ 对应内圆边界，$\xi=1$ 对应外圆边界，$\eta=0, 1$ 对应两条直边。

**组分数与扩散参数**：与算例 1、2 完全相同（$c_{12}=0.1,\ c_{13}=0.2,\ c_{23}=2.0,\ c^*=0.1$）。

**初始条件**（计算域上按径向坐标 $\xi$ 分层，与角向 $\eta$ 无关）：
- 内圈（$\xi \in [0, 0.25]$，即 $r \in [0.5, 0.75]$）：$(c_1, c_2, c_3) = (1, 0, 0)$（纯组分 1）
- 中间（$\xi \in (0.25, 0.75)$，即 $r \in (0.75, 1.25)$）：$(c_1, c_2, c_3) = (0, 0, 1)$（纯组分 3）
- 外圈（$\xi \in [0.75, 1]$，即 $r \in [1.25, 1.5]$）：$(c_1, c_2, c_3) = (0, 1, 0)$（纯组分 2）

三组分之和恒为 1，初始时各组分在其区域内形成完整的半圆环带。

**源项**：$f_i = 0$

**边界条件**：所有边界（内圆、外圆、两条直边）均为零法向通量，即 $J_i \cdot \boldsymbol{n} = 0$。

**物理意义**：考察三组分在半圆环几何中的径向扩散过程。初始时刻形成内圈纯组分 1、中间纯组分 3、外圈纯组分 2 的三层环状分布，在零通量边界约束下沿径向相互扩散，最终趋于均匀稳态。该算例可用于验证曲边 H(div) 混合虚元方法在非平凡几何下的保和性、质量守恒与长期稳定性。

---

## 5. 数值结果

以下数值实验均采用曲边 H(div) 混合虚元法（$k=1$ 或 $k=2$），Picard 迭代处理非线性，时间离散采用向后欧拉，线性求解器为 GPU 加速 FGMRES。误差为绝对 L² 误差。

### 5.1 算例 2（制造解）收敛阶验证

#### 5.1.1 k = 1

时间步长 $dt=10^{-4}$，共 50 步，$T=5\times10^{-3}$ s，时间离散误差可忽略。

**浓度 L² 误差与收敛阶：**

| 网格 | 单元数 | h | c₀ 误差 | c₁ 误差 | c₂ 误差 | c₀ 阶 | c₁ 阶 | c₂ 阶 |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 2x2 | 4 | 0.5000 | 9.68e-03 | 1.03e-02 | 1.41e-02 | — | — | — |
| 4x4 | 16 | 0.2500 | 3.35e-03 | 3.99e-03 | 5.19e-03 | 1.53 | 1.37 | 1.45 |
| 8x8 | 64 | 0.1250 | 8.03e-04 | 1.25e-03 | 1.58e-03 | 2.06 | 1.68 | 1.72 |
| 16x16 | 256 | 0.0625 | 1.66e-04 | 3.14e-04 | 3.59e-04 | 2.27 | 1.99 | 2.13 |
| 32x32 | 1024 | 0.0312 | 3.78e-05 | 6.80e-05 | 7.73e-05 | 2.14 | 2.21 | 2.22 |
| 64x64 | 4096 | 0.0156 | 9.28e-06 | 1.57e-05 | 1.82e-05 | 2.03 | 2.11 | 2.09 |

**通量 L² 误差与收敛阶：**

| 网格 | 单元数 | h | J₀ 误差 | J₁ 误差 | J₂ 误差 | J₀ 阶 | J₁ 阶 | J₂ 阶 |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 2x2 | 4 | 0.5000 | 2.74e-01 | 1.38e-01 | 2.04e-01 | — | — | — |
| 4x4 | 16 | 0.2500 | 9.99e-02 | 5.88e-02 | 9.55e-02 | 1.45 | 1.23 | 1.09 |
| 8x8 | 64 | 0.1250 | 2.54e-02 | 1.31e-02 | 2.19e-02 | 1.98 | 2.17 | 2.13 |
| 16x16 | 256 | 0.0625 | 6.32e-03 | 2.97e-03 | 5.05e-03 | 2.01 | 2.14 | 2.11 |
| 32x32 | 1024 | 0.0312 | 1.58e-03 | 7.09e-04 | 1.21e-03 | 2.00 | 2.07 | 2.07 |
| 64x64 | 4096 | 0.0156 | 3.96e-04 | 1.72e-04 | 2.92e-04 | 1.99 | 2.04 | 2.05 |

**结论**：$k=1$ 时，浓度与通量均达到约 $O(h^2)$ 收敛，与混合元理论预期（通量 $k$ 阶、浓度 $k+1$ 阶）相符。

#### 5.1.2 k = 2

时间步长 $dt=10^{-5}$，共 50 步，$T=5\times10^{-4}$ s，时间离散误差可忽略。

**浓度 L² 误差与收敛阶：**

| 网格 | 单元数 | h | c₀ 误差 | c₁ 误差 | c₂ 误差 | c₀ 阶 | c₁ 阶 | c₂ 阶 |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 2x2 | 4 | 0.5000 | 1.98e-04 | 4.26e-04 | 4.68e-04 | — | — | — |
| 4x4 | 16 | 0.2500 | 2.84e-05 | 7.39e-05 | 7.90e-05 | 2.80 | 2.53 | 2.57 |
| 8x8 | 64 | 0.1250 | 4.71e-06 | 1.01e-05 | 1.11e-05 | 2.59 | 2.87 | 2.83 |
| 16x16 | 256 | 0.0625 | 5.89e-07 | 1.31e-06 | 1.44e-06 | 3.00 | 2.95 | 2.95 |
| 32x32 | 1024 | 0.0312 | 7.10e-08 | 1.66e-07 | 1.80e-07 | 3.05 | 2.98 | 3.00 |

**通量 L² 误差与收敛阶：**

| 网格 | 单元数 | h | J₀ 误差 | J₁ 误差 | J₂ 误差 | J₀ 阶 | J₁ 阶 | J₂ 阶 |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 2x2 | 4 | 0.5000 | 1.32e-02 | 6.29e-03 | 1.04e-02 | — | — | — |
| 4x4 | 16 | 0.2500 | 2.34e-03 | 1.06e-03 | 1.81e-03 | 2.49 | 2.57 | 2.53 |
| 8x8 | 64 | 0.1250 | 2.47e-04 | 1.32e-04 | 2.06e-04 | 3.25 | 3.01 | 3.13 |
| 16x16 | 256 | 0.0625 | 2.89e-05 | 1.57e-05 | 2.37e-05 | 3.09 | 3.07 | 3.12 |
| 32x32 | 1024 | 0.0312 | 3.55e-06 | 1.74e-06 | 2.73e-06 | 3.03 | 3.17 | 3.12 |

**结论**：
- 浓度收敛阶约 $O(h^3)$，符合 $k=2$ 混合元理论预期（浓度 $k+1$ 阶收敛）。
- 通量收敛阶约 $O(h^3)$，高于理论 $O(h^k)=O(h^2)$，存在超收敛现象。
- 最密网格 32x32（1024 单元）浓度误差量级 $10^{-7}$，通量误差量级 $10^{-6}$。

#### 5.1.3 时间离散误差的影响

当时间步长较大时，时间累积误差会主导总误差，导致空间收敛阶在细网格下出现平台。以 $k=2$、$dt=10^{-4}$、100 步（$T=0.01$ s）为例：

| 网格 | c₀ 误差 | c₁ 误差 | c₂ 误差 | c₀ 阶 | c₁ 阶 | c₂ 阶 |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 2x2 | 1.24e-02 | 1.18e-02 | 1.62e-02 | — | — | — |
| 4x4 | 1.18e-03 | 2.23e-03 | 2.57e-03 | 3.40 | 2.40 | 2.66 |
| 8x8 | 1.15e-04 | 3.00e-04 | 3.07e-04 | 3.35 | 2.89 | 3.07 |
| 16x16 | 1.30e-05 | 3.14e-05 | 3.24e-05 | 3.15 | 3.26 | 3.24 |
| 32x32 | 5.16e-06 | 4.34e-06 | 7.07e-06 | 1.33 | 2.86 | 2.20 |

32x32 最密网格处收敛阶明显下降，误差进入时间误差主导的平台区。与 $dt=10^{-5}$ 的结果对比：时间步缩小 10 倍后，32x32 浓度误差从 $5\times10^{-6}$ 降至 $7\times10^{-8}$（约 70 倍），验证了向后欧拉一阶时间误差的瓶颈效应。

---

### 5.2 算例 3（半圆环扩散）数值实验

8x8 网格（实际 16x16 = 256 个曲边单元），$k=1$，$dt=0.001$，共 500 步，$T=0.5$ s。初始为内圈纯组分 1、中间纯组分 3、外圈纯组分 2 的三层环状分布，在零通量边界下沿径向扩散，最终趋于均匀稳态。

> 该算例无精确解，主要用于演示曲边几何下非线性扩散过程的数值稳定性与保和性。结果可视化见浓度云图动画。
