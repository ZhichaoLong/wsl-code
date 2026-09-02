# 曲边网格混合虚拟元法求解 Darcy 方程：理论与数值实验

## 摘要
本文针对曲边多边形网格上的 Darcy 渗流方程，建立一套完整的混合虚拟元（Mixed Virtual Element Method, MVEM）离散框架。全文采用”物理曲边域 Darcy 问题经 Piola 变换拉回直边计算域，再在计算域上进行单元组装”的技术路线，系统介绍了等参曲边映射、H(div) 投影矩阵链、一致性刚度与稳定化构造、散度耦合、曲边边界条件处理以及全局鞍点系统的形成。数值实验部分给出了 k=1, 2, 3 阶曲边 MVEM 的收敛阶结果，验证了压力与通量均达到 k+1 阶的最优收敛阶，并展示了曲边压力场云图。

---
## 1. 连续问题模型（Darcy渗流方程）
### 1.1 控制方程与边界条件
设有界曲边多边形区域$\Omega\subset\mathbb{R}^2$，边界$\partial\Omega=\Gamma_D\cup\Gamma_N$，$\Gamma_D\cap\Gamma_N=\emptyset$。Darcy渗流方程组：
$$
\begin{cases}
\boldsymbol{u} + \mathbb{K}\nabla p = 0,\quad &\text{in } \Omega\\
\nabla\cdot\boldsymbol{u} = g,\quad &\text{in } \Omega
\end{cases}
$$
边界条件：
1. 压力Dirichlet边界：$p = p_D,\quad \Gamma_D$
2. 法向通量Neumann边界：$\boldsymbol{u}\cdot\boldsymbol{n} = u_N,\quad \Gamma_N$

符号说明：
- $\boldsymbol{u}$：渗流速度场（矢量未知量）；
- $p$：孔隙压力（标量未知量）；
- $\mathbb{K}\in\mathbb{R}^{2\times2}$：对称正定渗透张量；
- $g$：体积质量源；
- $\boldsymbol{n}$：区域外法向单位向量。

### 1.2 混合变分弱形式（鞍点问题）
引入函数空间：
- 速度空间：$\boldsymbol{H}(\text{div};\Omega) = \{\boldsymbol{v}\in[L^2(\Omega)]^2,\ \nabla\cdot\boldsymbol{v}\in L^2(\Omega)\}$；
- 压力空间：$L^2(\Omega)$。

对Darcy第一式乘测试速度$\boldsymbol{v}\in\boldsymbol{H}(\text{div};\Omega)$，第二式乘测试压力$q\in L^2(\Omega)$，分部积分得混合弱形式：
找$(\boldsymbol{u},p)\in\boldsymbol{H}(\text{div};\Omega)\times L^2(\Omega)$，满足
$$
\begin{cases}
a(\boldsymbol{u},\boldsymbol{v}) + b(\boldsymbol{v},p) = 0, &\forall \boldsymbol{v}\in\boldsymbol{H}(\text{div};\Omega)\\
b(\boldsymbol{u},q) = G(q), &\forall q\in L^2(\Omega)
\end{cases}
$$
其中双线性型、线性泛函定义：
1. 渗透双线性型（对称正定）
$$
a(\boldsymbol{w},\boldsymbol{v}) = \int_\Omega \mathbb{K}^{-1}\boldsymbol{w}\cdot\boldsymbol{v} \,\mathrm{d}\Omega
$$
2. 散度耦合双线性型（鞍点约束）
$$
b(\boldsymbol{v},q) = -\int_\Omega q\,\nabla\cdot\boldsymbol{v} \,\mathrm{d}\Omega
$$
3. 质量源泛函
$$
G(q) = -\int_\Omega g\,q \,\mathrm{d}\Omega
$$
法向通量边界条件$\boldsymbol{u}\cdot\boldsymbol{n}=u_N$通过本质约束（自由度强置）方式施加，对应速度空间取满足边界条件的仿射子空间。

## 2. 曲边多边形虚拟单元剖分与等参映射（核心曲边网格基础）
### 2.1 曲边单元剖分$\mathcal{T}_h$
$\mathcal{T}_h$为$\Omega$的曲边多边形剖分，任意单元$E\in\mathcal{T}_h$为**曲边多边形**，边界由直线段+曲线边构成；每条边$\gamma\subset\partial E$可直可曲。
对每个曲边物理单元$E$，建立**标准参考多边形单元$\hat{E}$（正则凸参考元，直边）**，构造**等参可逆映射**：
$$
\boldsymbol{F}_E:\hat{E}\to E,\quad \boldsymbol{x} = \boldsymbol{F}_E(\hat{\boldsymbol{x}}),\quad \hat{\boldsymbol{x}}\in\hat{E},\ \boldsymbol{x}\in E
$$
映射可微可逆，雅可比矩阵：
$$
\mathbb{J}_E(\hat{\boldsymbol{x}}) = \nabla_{\hat{\boldsymbol{x}}} \boldsymbol{F}_E(\hat{\boldsymbol{x}})
=\begin{pmatrix}
\partial_{\hat{x}_1}F_{E,1} & \partial_{\hat{x}_2}F_{E,1}\\
\partial_{\hat{x}_1}F_{E,2} & \partial_{\hat{x}_2}F_{E,2}
\end{pmatrix}
$$
雅可比行列式（恒正，映射保定向）：
$$
J_E(\hat{\boldsymbol{x}}) = \det\mathbb{J}_E(\hat{\boldsymbol{x}})>0
$$
逆雅可比：$\mathbb{J}_E^{-1}(\hat{\boldsymbol{x}})$，梯度变换法则（关键，曲边单元导数转换）：
$$
\nabla_{\boldsymbol{x}} \phi(\boldsymbol{x}) = \mathbb{J}_E^{-\top}(\hat{\boldsymbol{x}}) \nabla_{\hat{\boldsymbol{x}}} \hat{\phi}(\hat{\boldsymbol{x}}),\quad \hat{\phi}(\hat{\boldsymbol{x}})=\phi(\boldsymbol{F}_E(\hat{\boldsymbol{x}}))
$$

### 2.2 标量积分变换与 $H(\mathrm{div})$ Piola 变换
对任意物理单元 $E$ 上的标量可积函数 $\varphi(\boldsymbol{x})$，有
$$
\int_E \varphi(\boldsymbol{x})\,\mathrm{d}\boldsymbol{x}
=\int_{\hat{E}}\varphi\big(\boldsymbol{F}_E(\hat{\boldsymbol{x}})\big)
J_E(\hat{\boldsymbol{x}})\,\mathrm{d}\hat{\boldsymbol{x}}.
$$
曲边 $\gamma_E=\boldsymbol{F}_E(\hat{\gamma})$ 上的标量边积分为
$$
\int_{\gamma_E}\psi(\boldsymbol{x})\,\mathrm{d}\Gamma
=\int_{\hat{\gamma}}\psi\big(\boldsymbol{F}_E(\hat{s})\big)
 j_\gamma(\hat{s})\,\mathrm{d}\hat{s}.
$$

但是速度属于 $\boldsymbol{H}(\mathrm{div})$，不能只作函数复合。记
$\mathbb{A}_E=\mathbb{J}_E=\nabla_{\hat{\boldsymbol{x}}}\boldsymbol{F}_E$、
$J_E=\det\mathbb{A}_E$，正向和逆向 contravariant Piola 变换为
$$
\boldsymbol{v}(\boldsymbol{F}_E(\hat{\boldsymbol{x}}))
=\frac{1}{J_E}\mathbb{A}_E\hat{\boldsymbol{v}}(\hat{\boldsymbol{x}}),
\qquad
\hat{\boldsymbol{v}}
=J_E\mathbb{A}_E^{-1}(\boldsymbol{v}\circ\boldsymbol{F}_E).
$$
该变换保持散度和法向通量：
$$
\nabla_{\boldsymbol{x}}\cdot\boldsymbol{v}
=\frac{1}{J_E}\hat{\nabla}\cdot\hat{\boldsymbol{v}},
$$
$$
(\boldsymbol{v}\cdot\boldsymbol{n})\,\mathrm{d}\Gamma
=(\hat{\boldsymbol{v}}\cdot\hat{\boldsymbol{n}})\,\mathrm{d}\hat{\Gamma}.
$$
因此
$$
\int_E q\,\nabla\cdot\boldsymbol{v}\,\mathrm{d}\boldsymbol{x}
=\int_{\hat{E}}\hat{q}\,\hat{\nabla}\cdot\hat{\boldsymbol{v}}
\,\mathrm{d}\hat{\boldsymbol{x}},
\qquad \hat{q}=q\circ\boldsymbol{F}_E,
$$
这里不再额外出现 $J_E$。

### 2.3 计算域路线
本文不在物理曲边单元上重新定义一套投影，而是先通过 Piola 变换把物理 Darcy 问题拉回直边计算单元 $\hat{E}$，然后在 $\hat{E}$ 上使用与直边网格相同的 H(div) 虚拟元投影。物理数据变换为
$$
\hat{p}=p\circ\boldsymbol{F}_E,
\qquad
\hat{\boldsymbol{u}}=J_E\mathbb{A}_E^{-1}
(\boldsymbol{u}\circ\boldsymbol{F}_E),
$$
$$
\hat{g}=J_E(g\circ\boldsymbol{F}_E),
\qquad
\hat{\mathbb{K}}=J_E\mathbb{A}_E^{-1}
(\mathbb{K}\circ\boldsymbol{F}_E)\mathbb{A}_E^{-\top}.
$$
对应的逆渗透张量为
$$
\hat{\mathbb{K}}^{-1}
=\frac{1}{J_E}\mathbb{A}_E^\top
(\mathbb{K}^{-1}\circ\boldsymbol{F}_E)\mathbb{A}_E.
$$

这一选择带来明确的分层：

1. 通用投影矩阵 $D,G,W,H,H^*,B$ 只依赖直边计算单元的几何和自由度，不依赖物理曲边映射；
2. 曲率通过 $\hat{\mathbb{K}}$、$\hat{g}$、变换后的边界数据和解析解进入 Darcy 专用组装；
3. 恒等映射下，整个构造严格退化为参考直边 H(div) 实现。

若改为直接在物理曲边单元上投影，则必须把多项式向量基作 Piola 推送，并使用度量加权 Gram 矩阵。这是另一套可行表述，但不是本文采用的路线，不能与上述计算域投影混写。

## 3. 直边计算域上的混合虚拟元空间与投影
记
$$
N_k=\dim\mathbb{P}_k(\hat{E})=\frac{(k+1)(k+2)}{2},
\qquad N_{-1}=0.
$$

### 3.1 压力离散空间
压力在每个直边计算单元上取 $k$ 次多项式：
$$
\hat{Q}_h=\{\hat{q}_h\in L^2(\hat{\Omega}):
\hat{q}_h|_{\hat{E}}\in\mathbb{P}_k(\hat{E}),\ \forall\hat{E}\in\hat{\mathcal{T}}_h\}.
$$
每个单元有 $N_k$ 个压力自由度。最低阶 $k=0$ 时 $N_0=1$，才退化为分片常数压力。

### 3.2 速度虚拟元空间和自由度
局部速度空间满足
$$
\hat{\boldsymbol{V}}_h(\hat{E})\subset\boldsymbol{H}(\mathrm{div};\hat{E}),
\qquad
\hat{\nabla}\cdot\hat{\boldsymbol{v}}_h\in\mathbb{P}_k(\hat{E}),
\qquad
\hat{\boldsymbol{v}}_h\cdot\hat{\boldsymbol{n}}|_{\hat{e}}
\in\mathbb{P}_k(\hat{e}).
$$
用来承载 $L^2$ 投影的向量多项式空间按标准分解为
$$
[\mathbb{P}_k(\hat{E})]^2
=\nabla\mathbb{P}_{k+1}(\hat{E})\oplus\mathcal{G}_k^\perp(\hat{E}).
$$
其维数为
$$
\dim\nabla\mathbb{P}_{k+1}=N_{k+1}-1=N_k+k+1,
$$
$$
\dim\mathcal{G}_k^\perp=N_{k-1}=N_k-(k+1),
\qquad
(N_{k+1}-1)+N_{k-1}=2N_k.
$$

若单元有 $n_e$ 条边，局部速度自由度为：

1. **边法向矩**：每条边 $k+1$ 个，
   $$
   \int_{\hat{e}}(\hat{\boldsymbol{v}}_h\cdot\hat{\boldsymbol{n}})
   \hat{m}_\ell^{\hat{e}}\,\mathrm{d}\hat{s},
   \qquad \ell=0,\ldots,k;
   $$
2. **内部梯度矩**：$N_k-1$ 个，对应 $\nabla\mathbb{P}_k$ 中由非常数标量单项式产生的梯度；
3. **内部补空间矩**：$N_{k-1}$ 个，对应 $\mathcal{G}_k^\perp$ 的低阶部分。

所以
$$
n_{\mathrm{dof},\hat{E}}
=(k+1)n_e+(N_k-1)+N_{k-1}.
$$
特别地，$k=0$ 时只有每条边一个常数法向通量矩，**没有单元内部散度自由度**。散度常数由边通量通过散度定理确定。

虚拟元局部基函数通常没有可用的逐点显式表达；后续计算只需要自由度向量和显式多项式基，不应在高斯点直接求“虚拟基函数值”。

### 3.3 系数无关的基础矩阵
在一个直边计算单元 $\hat{E}$ 上，令
$\{\boldsymbol{m}_\alpha\}_{\alpha=1}^{2N_k}$ 为
$[\mathbb{P}_k]^2$ 基，$\{m_i\}_{i=1}^{N_k}$ 为标量 $\mathbb{P}_k$ 基，局部速度自由度数为 $n_d$。构造以下基础矩阵。

#### 3.3.1 自由度矩阵 $D$
$$
D\in\mathbb{R}^{n_d\times2N_k},
\qquad
D_{i\alpha}=\mathrm{dof}_i(\boldsymbol{m}_\alpha).
$$
它把多项式系数映射成该多项式的自由度向量。

#### 3.3.2 无权向量 Gram 矩阵 $G$
$$
G\in\mathbb{R}^{2N_k\times2N_k},
\qquad
G_{\alpha\beta}
=\int_{\hat{E}}\boldsymbol{m}_\alpha\cdot\boldsymbol{m}_\beta
\,\mathrm{d}\hat{\boldsymbol{x}}.
$$
$G$ 不含渗透张量，也不含物理曲边映射。

#### 3.3.3 标量矩阵 $H$、$H^*$ 和 $W$
$$
H\in\mathbb{R}^{N_k\times N_k},
\qquad
H_{ij}=\int_{\hat{E}}m_i m_j\,\mathrm{d}\hat{\boldsymbol{x}}.
$$
令 $\{m^+_a\}_{a=1}^{N_{k+1}-1}$ 为 $\mathbb{P}_{k+1}$ 中去掉常数后的有序单项式，则
$$
H^*\in\mathbb{R}^{(N_{k+1}-1)\times N_k},
\qquad
H^*_{aj}=\int_{\hat{E}}m^+_a m_j\,\mathrm{d}\hat{\boldsymbol{x}}.
$$
分部积分矩阵
$$
W\in\mathbb{R}^{N_k\times n_d}
$$
满足
$$
W_{ij}
=-\int_{\hat{E}}\hat{\boldsymbol{\phi}}_j\cdot\hat{\nabla} m_i
\,\mathrm{d}\hat{\boldsymbol{x}}
+\sum_{\hat{e}\subset\partial\hat{E}}
\int_{\hat{e}}(\hat{\boldsymbol{\phi}}_j\cdot\hat{\boldsymbol{n}})m_i
\,\mathrm{d}\hat{s}.
$$
右端两项都可由内部矩、边法向矩和显式多项式积分计算，不需要 $\hat{\boldsymbol{\phi}}_j$ 的点值。

每条边还需要
$$
H_{\hat{e}},\quad W_{\hat{e}},\quad H^*_{\hat{e}},
$$
分别表示边多项式 Gram、边法向自由度矩阵和 $m_a^+$ 在该边上的交叉矩。

### 3.4 投影右端矩阵 $B$ 与两个投影算子
梯度部分按分部积分分成体积项和边界项：
$$
B_1=-H^*H^{-1}W,
$$
$$
B_2=\sum_{\hat{e}\subset\partial\hat{E}}
H^*_{\hat{e}}H_{\hat{e}}^{-1}W_{\hat{e}},
$$
其中每个边矩阵的列按该边在局部边自由度中的编号嵌入总矩阵。于是
$$
B_{\mathrm{grad}}=B_1+B_2.
$$
补空间部分 $B_\perp$ 直接由内部补空间矩得到。总矩阵为
$$
B=\begin{bmatrix}B_{\mathrm{grad}}\\B_\perp\end{bmatrix}
\in\mathbb{R}^{2N_k\times n_d}.
$$
应满足多项式一致性恒等式
$$
BD=G.
$$

从自由度向量到 $[\mathbb{P}_k]^2$ 系数的 $L^2$ 投影为
$$
\Pi_{\mathrm{poly}}=G^{-1}B
\in\mathbb{R}^{2N_k\times n_d}.
$$
投影多项式再取自由度，得到自由度空间中的投影
$$
\Pi_{\mathrm{dof}}=D\,G^{-1}B
\in\mathbb{R}^{n_d\times n_d}.
$$
它应满足 $\Pi_{\mathrm{dof}}^2=\Pi_{\mathrm{dof}}$。以上矩阵都属于通用 H(div) 投影层，与具体 PDE 系数无关。

## 4. Darcy 专用双线性型、稳定化与耦合
通用投影层只提供 $D,G,W,H,H^*,B$ 和两个投影算子，不直接决定 Darcy、Maxwell--Stefan 等问题的局部刚度。问题相关的加权 Gram、一致性项和稳定化尺度应由上层派生/组装类提供。

### 4.1 系数加权一致性项
在直边计算单元上组装
$$
(G_{\hat{K}^{-1}})_{\alpha\beta}
=\int_{\hat{E}}
\big(\hat{\mathbb{K}}^{-1}\boldsymbol{m}_\beta\big)
\cdot\boldsymbol{m}_\alpha\,\mathrm{d}\hat{\boldsymbol{x}}.
$$
注意：物理曲边的 Jacobian 和度量已经包含在
$\hat{\mathbb{K}}^{-1}=J_E^{-1}\mathbb{A}_E^\top
(\mathbb{K}^{-1}\circ\boldsymbol{F}_E)\mathbb{A}_E$ 中，不能再在该积分外重复乘一个 $J_E$。

一致性刚度矩阵为
$$
K_{\mathrm{ac}}
=\Pi_{\mathrm{poly}}^\top G_{\hat{K}^{-1}}
\Pi_{\mathrm{poly}}.
$$
当映射为恒等且 $\mathbb{K}=\mathbb{I}$ 时，
$G_{\hat{K}^{-1}}=G$。变系数或多组分问题只需改变加权 Gram 的组装，而复用同一投影算子。

### 4.2 稳定化项
自由度稳定化写成
$$
K_{\mathrm{as}}
=\alpha_{\hat{E}}
(I-\Pi_{\mathrm{dof}})^\top
(I-\Pi_{\mathrm{dof}}).
$$
它只作用于投影核空间，并自动在多项式子空间上为零。这里
$\alpha_{\hat{E}}>0$ 是问题相关尺度：常系数问题可取固定的量纲匹配值，变系数问题可由单元上的系数积分、均值或谱尺度确定。

### 4.3 完整速度局部矩阵
$$
K_{\hat{E}}=K_{\mathrm{ac}}+K_{\mathrm{as}}
\in\mathbb{R}^{n_d\times n_d}.
$$
这一矩阵由 Darcy 专用组装层生成，而不是由通用投影基类直接生成。

### 4.4 速度--压力耦合矩阵 $C_{\hat{E}}$
为避免与投影右端矩阵 $B$ 混淆，本文将鞍点系统中的局部耦合矩阵记为 $C_{\hat{E}}$。若 $\{q_r\}_{r=1}^{N_k}$ 是压力基，则
$$
(C_{\hat{E}})_{ir}
=-\int_{\hat{E}}q_r\,
\hat{\nabla}\cdot\hat{\boldsymbol{\phi}}_i
\,\mathrm{d}\hat{\boldsymbol{x}},
$$
$$
C_{\hat{E}}\in\mathbb{R}^{n_d\times N_k}.
$$
该积分可通过分部积分和已有自由度矩精确计算，无需虚拟基的点值。由于 Piola 保持散度配对，从物理单元拉回时该式不再额外乘 $J_E$。

最低阶 $k=0$ 时 $N_0=1$，$C_{\hat{E}}$ 才退化为一列。

### 4.5 单元矩阵职责分层
推荐的调用关系为
$$
\text{多项式基}
\longrightarrow
\text{H(div) 通用投影 }(D,G,W,H,H^*,B,\Pi)
\longrightarrow
\text{Darcy 组装 }(G_{\hat{K}^{-1}},K_{\mathrm{ac}},K_{\mathrm{as}},C_{\hat{E}}).
$$
中间层为通用 H(div) 投影，与具体方程无关；最后一层由问题的系数和方程形式决定，可针对不同算例派生不同的局部刚度，而复用同一套投影算子。

## 5. 全局离散鞍点线性系统

全局未知向量分块为

$$
\boldsymbol{X}
=
\begin{pmatrix}
\boldsymbol{U} \\
\boldsymbol{P}
\end{pmatrix},
$$

其中 $\boldsymbol{U}$ 为全局速度自由度，$\boldsymbol{P}$ 为全局压力自由度。全局速度--压力耦合矩阵记为 $\mathbb{C}$，以避免与局部投影矩阵 $B$ 混淆。未加入压力零均值约束时，全局离散系统写为

$$
\begin{pmatrix}
\mathbb{K}_h & \mathbb{C} \\
\mathbb{C}^{\mathsf{T}} & \mathbb{O}
\end{pmatrix}
\begin{pmatrix}
\boldsymbol{U} \\
\boldsymbol{P}
\end{pmatrix}
=
\begin{pmatrix}
\boldsymbol{0} \\
\boldsymbol{G}
\end{pmatrix}.
$$

这里采用局部约定

$$
C_{\hat{E}}
\in
\mathbb{R}^{n_d\times N_k},
$$

所以速度--压力块为 $\mathbb{C}$，压力--速度块为 $\mathbb{C}^{\mathsf{T}}$。若采用相反的行列约定，只需整体转置，但所有局部与全局组装必须保持一致。

引入全局 Lagrange 乘子 $\lambda$ 施加压力零积分约束。此时完整系统为

$$
\begin{pmatrix}
\mathbb{K}_h & \mathbb{C} & \boldsymbol{0} \\
\mathbb{C}^{\mathsf{T}} & \mathbb{O} & \boldsymbol{L} \\
\boldsymbol{0}^{\mathsf{T}} & \boldsymbol{L}^{\mathsf{T}} & 0
\end{pmatrix}
\begin{pmatrix}
\boldsymbol{U} \\
\boldsymbol{P} \\
\lambda
\end{pmatrix}
=
\begin{pmatrix}
\boldsymbol{F} \\
\boldsymbol{G} \\
0
\end{pmatrix},
$$

其中 $\boldsymbol{L}$ 由各计算单元上的压力基积分累加得到：

$$
(L_{\hat{E}})_r
=
\int_{\hat{E}}
 m_r(\hat{\boldsymbol{x}})
 \det A(\hat{\boldsymbol{x}})
\,\mathrm{d}\hat{\boldsymbol{x}}.
$$

分块说明：

1. $\mathbb{K}_h$：由每个计算单元的 $K_{\hat{E}}=K_{\mathrm{ac}}+K_{\mathrm{as}}$ 累加；
2. $\mathbb{C}$：由局部散度耦合矩阵 $C_{\hat{E}}$ 累加，等价于 $W^{\mathsf{T}}$；
3. $\mathbb{O}$：压力--压力零块；
4. $\boldsymbol{L}$：压力零积分约束向量；
5. $\boldsymbol{G}$：变换后的质量源在压力空间中的矩。

速度方程右端为零向量，法向通量边界条件通过本质约束（自由度强置）方式施加。

## 6. 曲边网格边界条件离散处理

### 6.1 法向通量边界：计算域强制边自由度

在混合 Darcy 离散中，速度属于 $H(\mathrm{div})$，边界上的法向通量矩是速度空间的边自由度。若物理边界上给定
$$
\boldsymbol{u}\cdot\boldsymbol{n}=u_N,
$$
则对物理曲边 $e=F(\hat{e})$，其第 $\ell$ 个边自由度为
$$
d_{e,\ell}(\boldsymbol{u})
=
\int_e
(\boldsymbol{u}\cdot\boldsymbol{n})
\bigl(\hat{m}_\ell\circ F^{-1}\bigr)
\,\mathrm{d}s,
\qquad \ell=0,\ldots,k.
$$
这里 $\hat{m}_\ell$ 是直边计算边 $\hat{e}$ 上的一维缩放单项式。

记映射雅可比矩阵和行列式为
$$
A=\nabla_{\hat{\boldsymbol{x}}}F,
\qquad
j=\det A>0.
$$
物理速度与计算速度之间采用 contravariant Piola 变换：
$$
\boldsymbol{u}\circ F
=\frac{1}{j}A\hat{\boldsymbol{u}},
\qquad
\hat{\boldsymbol{u}}
=jA^{-1}(\boldsymbol{u}\circ F).
$$
根据 Nanson 公式，物理曲边上的法向测度满足
$$
\boldsymbol{n}\,\mathrm{d}s
=
 jA^{-\top}\hat{\boldsymbol{n}}\,\mathrm{d}\hat{s}.
$$
因此
$$
\begin{aligned}
(\boldsymbol{u}\cdot\boldsymbol{n})\,\mathrm{d}s
&=
\left(\frac{1}{j}A\hat{\boldsymbol{u}}\right)
\cdot
\left(jA^{-\top}\hat{\boldsymbol{n}}\right)
\,\mathrm{d}\hat{s}\\
&=
(\hat{\boldsymbol{u}}\cdot\hat{\boldsymbol{n}})
\,\mathrm{d}\hat{s}.
\end{aligned}
$$
即 Piola 变换保持法向通量：
$$
\boxed{
(\boldsymbol{u}\cdot\boldsymbol{n})\,\mathrm{d}s
=
(\hat{\boldsymbol{u}}\cdot\hat{\boldsymbol{n}})
\,\mathrm{d}\hat{s}
}.
$$
于是物理曲边上的通量自由度可等价地在直边计算边上计算：
$$
\boxed{
d_{e,\ell}(\boldsymbol{u})
=
\int_{\hat{e}}
(\hat{\boldsymbol{u}}\cdot\hat{\boldsymbol{n}})
\hat{m}_\ell
\,\mathrm{d}\hat{s}
}.
$$

由 Piola 变换，计算区域上的解析速度为
$$
\hat{\boldsymbol{u}}
=jA^{-1}(\boldsymbol{u}\circ F),
$$
因此边界自由度直接按下式计算：
$$
U_{e,\ell}^{\mathrm{bd}}
=
\int_{\hat{e}}
\left(
\hat{\boldsymbol{u}}_{\mathrm{exact}}
\cdot\hat{\boldsymbol{n}}
\right)
\hat{m}_\ell
\,\mathrm{d}\hat{s}.
$$
这里使用计算直边上的普通一维 Gauss--Legendre 积分，**不能再额外乘物理曲边的边 Jacobian**，因为曲边长度变化已经包含在 Piola 速度的法向分量中。

若不使用 $\hat{\boldsymbol{u}}$，而直接使用物理法向通量 $\boldsymbol{u}\cdot\boldsymbol{n}$，则必须显式保留边映射 Jacobian $j_e=\mathrm{d}s/\mathrm{d}\hat{s}$：
$$
\int_{\hat{e}}
\bigl[(\boldsymbol{u}\cdot\boldsymbol{n})\circ F\bigr]
\hat{m}_\ell j_e
\,\mathrm{d}\hat{s}.
$$
这与上式数学等价，但两种表述不能混用，否则会重复计入曲边几何因子。

法向通量边界是 $H(\mathrm{div})$ 速度空间的强制边界约束。离散实现采用降维消元：先计算所有边界边自由度，构造完整边界值向量 $\boldsymbol{U}_{\mathrm{bd}}$，再修正右端
$$
\boldsymbol{G}
\leftarrow
\boldsymbol{G}-\mathbb{C}^{\mathsf{T}}\boldsymbol{U}_{\mathrm{bd}},
$$
并将速度方程右端保持为零。提取自由自由度对应的子矩阵和子向量求解，最后将自由解与边界值恢复为完整解向量。

还必须保持边自由度的全局方向一致。边方向反转时，法向产生一次负号，而 $\ell$ 次边单项式产生 $(-1)^\ell$，所以边通量矩整体产生 $(-1)^{\ell+1}$。这与局部矩阵组装时使用的方向调整规则相同：偶次边矩改变符号，奇次边矩保持不变。

### 6.2 压力边界数据

压力 Dirichlet 数据在混合弱形式中作为速度方程的自然边界泛函进入右端项；它不属于本节所述的法向通量强制自由度。
## 7. 直边计算域上的单元组装流程
对每个计算单元 $\hat{E}$，单元矩阵按以下顺序构造：

1. 由直边计算单元的顶点、边、质心、直径确定几何量与局部自由度数；
2. 构造标量 $\mathbb{P}_k$ 和向量 $[\mathbb{P}_k]^2$ 的显式多项式基；
3. 由边法向矩、内部梯度矩、内部补空间矩以及分部积分组装
   $D,G,H,H^*,W,H_e,W_e,H_e^*$；
4. 计算 $B_1,B_2,B_{\mathrm{grad}},B_\perp,B$，满足 $BD=G$；
5. 计算多项式投影
   $\Pi_{\mathrm{poly}}=G^{-1}B$ 和
   自由度投影 $\Pi_{\mathrm{dof}}=DG^{-1}B$；
6. 在计算域高斯点作用曲边映射，得到
   $\hat{\mathbb{K}}^{-1}$、$\hat{g}$ 和变换后的边界数据；
7. 构造 Darcy 单元矩阵
   $G_{\hat{K}^{-1}}$、$K_{\mathrm{ac}}$、$K_{\mathrm{as}}$、
   $C_{\hat{E}}$ 及局部右端；
8. 将局部速度块、耦合块和右端按全局自由度映射累加至全局系统。

注意：步骤 3--5 不依赖物理曲边 Jacobian；步骤 6--7 才引入物理到计算域的数据变换。

---

## 8. 数值验证算例

### 8.1 精确解与问题定义

取渗透张量 $\mathbb{K}=\mathbb{I}$（单位矩阵），物理域为单位正方形的小扰动曲边形，达西方程的**精确解**定义如下：

$$
p(\boldsymbol{x}) = \sin(\pi x)\cos(\pi y), \quad
\boldsymbol{u}(\boldsymbol{x}) =
\begin{pmatrix}
-\pi \cos(\pi x) \cos(\pi y) \\
\pi \sin(\pi x) \sin(\pi y)
\end{pmatrix}
$$

（注：第一分量符号与标准达西方程 $\boldsymbol{u}=-\mathbb{K}\nabla p$ 保持一致）

**推导源项**：
$$
\nabla\cdot\boldsymbol{u} = \frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y}
= \pi^2\sin(\pi x)\cos(\pi y) + \pi^2\sin(\pi x)\cos(\pi y)
= 2\pi^2\sin(\pi x)\cos(\pi y)
$$
因此体积质量源
$$
g(\boldsymbol{x})=2\pi^2\sin(\pi x)\cos(\pi y).
$$

**边界条件**：全程采用Neumann边界条件（第二类边界条件），法向通量：
$$
u_N = \boldsymbol{u}\cdot\boldsymbol{n} \quad \text{on } \partial\Omega
$$
在四个边界上可根据精确解预先计算 $u_N$ 的表达式。

---

### 8.2 曲边网格映射

参考单元为单位正方形 $\hat{\Omega}=[0,1]\times[0,1]$，物理单元为带小扰动的曲边形，等参映射：

$$
\boldsymbol{x}(\hat{\boldsymbol{\xi}})
=
\begin{pmatrix}
\xi + 0.05\sin(2\pi \eta) \\
\eta + 0.05\sin(2\pi \xi)
\end{pmatrix}
$$

其中 $\hat{\boldsymbol{\xi}}=(\xi,\eta)\in\hat{\Omega}$ 为参考坐标。

**雅可比矩阵**：
$$
\mathbb{J}(\hat{\boldsymbol{\xi}}) = \nabla_{\hat{\boldsymbol{\xi}}}\boldsymbol{x} = \begin{pmatrix}
1 & 0.1\pi\cos(2\pi \eta) \\
0.1\pi\cos(2\pi \xi) & 1
\end{pmatrix}
$$

**雅可比行列式**：
$$
\det\mathbb{J}(\hat{\boldsymbol{\xi}}) = 1 - 0.01\pi^2\cos(2\pi\xi)\cos(2\pi\eta)
$$
满足 $\det\mathbb{J} \ge 1 - 0.01\pi^2 \approx 0.901 > 0$，映射保定向且可逆。

### 8.3 直边计算区域上的精确解

计算与后续误差比较在直边参考区域 $\hat{\Omega}$ 上进行。不能直接把物理速度的两个分量仅作函数复合，而应采用保持法向通量和散度结构的逆 Piola 变换。记
$\boldsymbol{x}=\boldsymbol{F}(\hat{\boldsymbol{\xi}})$、
$\mathbb{J}=\nabla_{\hat{\boldsymbol{\xi}}}\boldsymbol{F}$、
$J=\det\mathbb{J}$，则计算区域解析数据定义为

$$
\hat{p}(\hat{\boldsymbol{\xi}})
= p\big(\boldsymbol{F}(\hat{\boldsymbol{\xi}})\big),
$$

$$
\hat{\boldsymbol{u}}(\hat{\boldsymbol{\xi}})
=J(\hat{\boldsymbol{\xi}})\mathbb{J}^{-1}(\hat{\boldsymbol{\xi}})
\boldsymbol{u}\big(\boldsymbol{F}(\hat{\boldsymbol{\xi}})\big),
$$

$$
\hat{g}(\hat{\boldsymbol{\xi}})
=J(\hat{\boldsymbol{\xi}})g\big(\boldsymbol{F}(\hat{\boldsymbol{\xi}})\big),
$$

$$
\hat{\mathbb{K}}(\hat{\boldsymbol{\xi}})
=J(\hat{\boldsymbol{\xi}})\mathbb{J}^{-1}(\hat{\boldsymbol{\xi}})
\mathbb{K}\big(\boldsymbol{F}(\hat{\boldsymbol{\xi}})\big)
\mathbb{J}^{-\top}(\hat{\boldsymbol{\xi}}).
$$

这些量在直边计算区域上满足

$$
\hat{\boldsymbol{u}}=-\hat{\mathbb{K}}\hat{\nabla}\hat{p},
\qquad
\hat{\nabla}\cdot\hat{\boldsymbol{u}}=\hat{g}.
$$

相应的正向 Piola 关系为

$$
\boldsymbol{u}\big(\boldsymbol{F}(\hat{\boldsymbol{\xi}})\big)
=\frac{1}{J(\hat{\boldsymbol{\xi}})}\mathbb{J}(\hat{\boldsymbol{\xi}})
\hat{\boldsymbol{u}}(\hat{\boldsymbol{\xi}}),
$$

因此物理区域与计算区域的法向通量及散度方程保持一致。恒等映射下
$\mathbb{J}=\mathbb{I}$、$J=1$，上述计算区域解析数据自然退化为物理区域解析数据。

---

## 9. 数值实验结果

本节给出正弦扰动曲边映射算例下，混合虚拟元法的收敛阶实测结果，验证上述数学构造的正确性与精度。

### 9.1 实验设置

- **物理域**：单位正方形经正弦扰动（eps=0.05）得到的曲边四边形
- **精确解**：$p(\boldsymbol{x}) = \sin(\pi x)\cos(\pi y)$，$\boldsymbol{u} = -\nabla p$
- **边界条件**：全边界 Neumann（法向通量给定）
- **积分规则**：
  - $k=1,2$：三角形 9 点高斯（2D），线段 9 点高斯（1D 边界）
  - $k=3$：三角形 12 点高斯（2D），线段 9 点高斯（1D 边界）
- **网格序列**：$2\times 2, 4\times 4, 8\times 8, 16\times 16, 32\times 32, 64\times 64$（$k=1,2$ 额外算至 $128\times 128$，$k=3$ 算至 $64\times 64$）
- **求解器**：PETSc FGMRES + ILU（CPU）/ BJACOBI（GPU）

### 9.2 压力云图（$k=1$，$32\times 32$ 网格）

下图给出 $k=1$ 最低阶情形下，$32\times 32$ 网格的数值解与精确解对比。左为数值解，右为精确解，可以看出两者几乎完全一致。

![压力云图对比 - 16x16 网格 k=1](../data/pressure_plot/pressure_comparison_square_16x16.png)

### 9.3 $k=1$（二阶）收敛阶结果

| 网格 | 总自由度 | 压力 $L^2$ 误差 | 压力阶 | 通量 $L^2$ 误差 | 通量阶 |
|------|---------|----------------|--------|----------------|--------|
| $2\times 2$     |     49 | 1.770e-01 |   —    | 6.046e-01 |   —    |
| $4\times 4$     |    177 | 3.886e-02 | 2.1876 | 1.615e-01 | 1.9044 |
| $8\times 8$     |    673 | 8.888e-03 | 2.1286 | 4.024e-02 | 2.0047 |
| $16\times 16$   |   2625 | 2.119e-03 | 2.0682 | 9.916e-03 | 2.0210 |
| $32\times 32$   |  10369 | 5.218e-04 | 2.0220 | 2.464e-03 | 2.0085 |
| $64\times 64$   |  41217 | 1.299e-04 | 2.0059 | 6.150e-04 | 2.0025 |
| $128\times 128$ | 164353 | 3.245e-05 | 2.0015 | 1.537e-04 | 2.0007 |

**结论**：压力与通量均以 $O(h^{k+1})=O(h^2)$ 速率收敛，符合混合虚元理论预测。最密网格（$128\times 128$）下压力 2.0015 阶、通量 2.0007 阶，误差单调下降。

### 9.4 $k=2$（三阶）收敛阶结果

| 网格 | 总自由度 | 压力 $L^2$ 误差 | 压力阶 | 通量 $L^2$ 误差 | 通量阶 |
|------|---------|----------------|--------|----------------|--------|
| $2\times 2$     |     93 | 3.219e-02 |   —    | 1.608e-01 |   —    |
| $4\times 4$     |    345 | 4.375e-03 | 2.8791 | 1.988e-02 | 3.0158 |
| $8\times 8$     |   1329 | 5.929e-04 | 2.8835 | 2.823e-03 | 2.8165 |
| $16\times 16$   |   5217 | 7.257e-05 | 3.0304 | 3.455e-04 | 3.0304 |
| $32\times 32$   |  20673 | 9.005e-06 | 3.0105 | 4.274e-05 | 3.0149 |
| $64\times 64$   |  82305 | 1.123e-06 | 3.0029 | 5.324e-06 | 3.0050 |
| $128\times 128$ | 328449 | 1.404e-07 | 3.0000 | 6.648e-07 | 3.0014 |

**结论**：压力与通量均以 $O(h^{k+1})=O(h^3)$ 速率收敛。最密网格（$128\times 128$）下压力 3.0000 阶、通量 3.0014 阶，再次验证曲边 VEM 框架的正确性。高阶（$k=2$）下曲边映射的几何处理仍然精确。

### 9.5 $k=3$（四阶）收敛阶结果

| 网格 | 总自由度 | 压力 $L^2$ 误差 | 压力阶 | 通量 $L^2$ 误差 | 通量阶 |
|------|---------|----------------|--------|----------------|--------|
| $2\times 2$   |    149 | 1.007e-02 |   —    | 4.344e-02 |   —    |
| $4\times 4$   |    561 | 8.158e-04 | 3.6260 | 4.117e-03 | 3.3991 |
| $8\times 8$   |   2177 | 4.763e-05 | 4.0982 | 2.503e-04 | 4.0397 |
| $16\times 16$ |   8577 | 2.942e-06 | 4.0173 | 1.548e-05 | 4.0151 |
| $32\times 32$ |  34049 | 1.830e-07 | 4.0067 | 9.608e-07 | 4.0102 |
| $64\times 64$ | 135681 | 1.150e-08 | 3.9916 | 6.003e-08 | 4.0005 |

**结论**：压力与通量均以 $O(h^{k+1})=O(h^4)$ 速率收敛。最密网格（$64\times 64$）下压力 3.99 阶、通量 4.00 阶，在高阶（$k=3$）下仍达到理论最优收敛阶，充分验证了：
1. 曲边等参映射与 Piola 变换的正确性；
2. 投影矩阵链（$D,G,W,H,H^*,B$）的高阶一致性；
3. 法向通量边界条件处理的精确性；
4. 12 点二维高斯积分足以匹配 $k=3$ 的多项式精度。

### 9.6 数值实验小结

1. **收敛阶**：$k=1,2,3$ 三档多项式阶数下，压力和通量均严格达到 $k+1$ 阶理论收敛速度，验证了曲边混合虚元法的最优收敛性。
2. **曲边映射**：所有结果均在带正弦扰动的曲边网格上获得，说明通过 Piola 变换拉回计算域、再用直边投影矩阵组装的路线是正确的。
3. **阶数通用性**：从 $k=1$ 到 $k=3$ 复用同一套投影矩阵构造，仅通过多项式阶数参数控制空间维度，体现了方法的通用性。