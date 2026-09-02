# 曲边混合虚拟元中法向通量边界条件的 Piola 变换与矩阵处理

## 1. 问题与符号

考虑 Darcy 型混合问题

$$
\begin{aligned}
\boldsymbol u&=-\boldsymbol\kappa\nabla p+\boldsymbol f,
&&\text{in }\Omega,\\
\nabla\cdot\boldsymbol u&=f_0,
&&\text{in }\Omega,
\end{aligned}
$$

并在边界的通量部分 $\Gamma_N$ 上给定

$$
\boxed{
\boldsymbol u\cdot\boldsymbol n=g
\qquad\text{on }\Gamma_N.
}
$$

其中 $\boldsymbol n$ 是物理区域边界上的单位外法向。速度属于

$$
\boldsymbol u\in H(\operatorname{div},\Omega),
$$

因此法向通量条件是速度空间的本质边界条件，应通过边界速度自由度强制施加。

设曲边单元 $E$ 由参考单元 $\widehat E$ 映射得到：

$$
F_E:\widehat E\longrightarrow E,
\qquad
\boldsymbol x=F_E(\widehat{\boldsymbol x}),
$$

并记

$$
J(\widehat{\boldsymbol x})=DF_E(\widehat{\boldsymbol x}),
\qquad
J_E(\widehat{\boldsymbol x})=\det J(\widehat{\boldsymbol x})>0.
$$

参考边 $\widehat e\subset\partial\widehat E$ 映射到物理曲边

$$
e=F_E(\widehat e)\subset\partial E.
$$

参考边和物理边上的单位外法向分别记为

$$
\widehat{\boldsymbol n},
\qquad
\boldsymbol n.
$$

## 2. 物理真解复合与 Piola 拉回的区别

已知物理速度真解 $\boldsymbol u_{\mathrm{ex}}(\boldsymbol x)$ 时，首先将参考点映射到物理点：

$$
\boldsymbol x=F_E(\widehat{\boldsymbol x}),
$$

然后代入真解，得到

$$
\widetilde{\boldsymbol u}(\widehat{\boldsymbol x})
=
\boldsymbol u_{\mathrm{ex}}(F_E(\widehat{\boldsymbol x})).
$$

这里 $\widetilde{\boldsymbol u}=\boldsymbol u_{\mathrm{ex}}\circ F_E$ 只是物理速度在对应物理点的取值。它的分量仍然是相对于物理坐标基底表示的，不能直接与参考法向 $\widehat{\boldsymbol n}$ 做法向通量计算。

真正属于参考 $H(\operatorname{div})$ 空间的速度，是物理速度的逆 Piola 拉回：

$$
\boxed{
\widehat{\boldsymbol u}_{\mathrm{ex}}
=
J_EJ^{-1}
\bigl(\boldsymbol u_{\mathrm{ex}}\circ F_E\bigr).
}
$$

因此必须依次完成两步：

$$
\boxed{
\widehat{\boldsymbol x}
\xrightarrow{F_E}
\boldsymbol x
\xrightarrow{\boldsymbol u_{\mathrm{ex}}}
\boldsymbol u_{\mathrm{ex}}(\boldsymbol x)
\xrightarrow{J_EJ^{-1}}
\widehat{\boldsymbol u}_{\mathrm{ex}}(\widehat{\boldsymbol x}).
}
$$

坐标映射负责找到物理真解的取值位置，Piola 拉回负责把物理速度转换为保持散度和法向通量结构的参考速度。

## 3. 速度的逆变 Piola 变换

参考速度 $\widehat{\boldsymbol u}$ 到物理速度 $\boldsymbol u$ 的逆变 Piola 变换为

$$
\boxed{
\boldsymbol u(\boldsymbol x)
=
\frac{1}{J_E}
J\widehat{\boldsymbol u}(\widehat{\boldsymbol x}),
\qquad
\boldsymbol x=F_E(\widehat{\boldsymbol x}).
}
$$

反变换为

$$
\boxed{
\widehat{\boldsymbol u}(\widehat{\boldsymbol x})
=
J_EJ^{-1}
\boldsymbol u(F_E(\widehat{\boldsymbol x})).
}
$$

该变换保证散度满足

$$
\boxed{
\nabla_{\boldsymbol x}\cdot\boldsymbol u
=
\frac{1}{J_E}
\widehat\nabla\cdot\widehat{\boldsymbol u}.
}
$$

这正是混合虚拟元和混合有限元中速度变量使用 Piola 变换的原因。

## 4. 物理法向与参考法向的变换

物理曲边法向不能直接取成参考法向。二者满足

$$
\boxed{
\boldsymbol n
=
\frac{J^{-T}\widehat{\boldsymbol n}}
{\left\|J^{-T}\widehat{\boldsymbol n}\right\|}.
}
$$

定义边界雅可比

$$
\boxed{
J_e
=
J_E\left\|J^{-T}\widehat{\boldsymbol n}\right\|.
}
$$

边测度的变换为

$$
\boxed{
\mathrm ds=J_e\,\mathrm d\widehat s.
}
$$

法向量与边测度的组合形式为

$$
\boxed{
\boldsymbol n\,\mathrm ds
=
J_EJ^{-T}\widehat{\boldsymbol n}\,\mathrm d\widehat s.
}
$$

在二维情况下，若参考边以参数 $t$ 表示为 $\widehat{\boldsymbol\gamma}(t)$，物理曲边为

$$
\boldsymbol\gamma(t)
=
F_E(\widehat{\boldsymbol\gamma}(t)),
$$

则

$$
\boldsymbol\gamma'(t)
=
J(\widehat{\boldsymbol\gamma}(t))
\widehat{\boldsymbol\gamma}'(t).
$$

当 $\widehat{\boldsymbol\gamma}$ 是单位速度参数化时，

$$
\boxed{
J_e(t)=\|\boldsymbol\gamma'(t)\|
=
\|J\widehat{\boldsymbol t}\|,
}
$$

其中 $\widehat{\boldsymbol t}$ 是参考边单位切向量。

## 5. 法向通量的 Piola 不变性

将速度 Piola 变换和法向变换结合，有

$$
\begin{aligned}
\boldsymbol u\cdot\boldsymbol n
&=
\left(\frac{1}{J_E}J\widehat{\boldsymbol u}\right)
\cdot
\left(
\frac{J_EJ^{-T}\widehat{\boldsymbol n}}{J_e}
\right)\\
&=
\frac{1}{J_e}
\widehat{\boldsymbol u}^{T}J^TJ^{-T}\widehat{\boldsymbol n}\\
&=
\frac{1}{J_e}
\widehat{\boldsymbol u}\cdot\widehat{\boldsymbol n}.
\end{aligned}
$$

因此

$$
\boxed{
\widehat{\boldsymbol u}\cdot\widehat{\boldsymbol n}
=
J_e(\boldsymbol u\cdot\boldsymbol n).
}
$$

再乘边测度可得

$$
\boxed{
(\boldsymbol u\cdot\boldsymbol n)\,\mathrm ds
=
(\widehat{\boldsymbol u}\cdot\widehat{\boldsymbol n})
\,\mathrm d\widehat s.
}
$$

所以 Piola 变换保持的是法向通量微元和法向通量积分，而不是法向分量的点值。一般情况下

$$
\boldsymbol u\cdot\boldsymbol n
\ne
\widehat{\boldsymbol u}\cdot\widehat{\boldsymbol n}.
$$

只有当 $J_e=1$ 时，这两个点值才相同。

## 6. 物理边界条件在参考边上的表达式

物理曲边上给定

$$
\boldsymbol u\cdot\boldsymbol n=g.
$$

则参考边上的边界数据为

$$
\boxed{
\widehat{\boldsymbol u}\cdot\widehat{\boldsymbol n}
=
\widehat g
=
J_e\,(g\circ F_E).
}
$$

因此不能直接令

$$
\widehat{\boldsymbol u}\cdot\widehat{\boldsymbol n}
=g\circ F_E,
$$

否则会遗漏曲边的边界雅可比。

若 $g$ 没有单独给出，而是由速度真解产生，则

$$
g(\boldsymbol x)
=
\boldsymbol u_{\mathrm{ex}}(\boldsymbol x)
\cdot\boldsymbol n(\boldsymbol x).
$$

参考边上的通量可以直接由 Piola 拉回计算：

$$
\boxed{
\widehat g(\widehat{\boldsymbol x})
=
\widehat{\boldsymbol u}_{\mathrm{ex}}
\cdot\widehat{\boldsymbol n}
=
J_E
\widehat{\boldsymbol n}^{T}J^{-1}
\boldsymbol u_{\mathrm{ex}}(F_E(\widehat{\boldsymbol x})).
}
$$

这个公式不需要显式计算物理单位法向 $\boldsymbol n$，与下面的物理形式完全等价：

$$
\boxed{
\widehat g
=
J_e
\left[
\boldsymbol u_{\mathrm{ex}}(F_E(\widehat{\boldsymbol x}))
\cdot
\frac{J^{-T}\widehat{\boldsymbol n}}
{\|J^{-T}\widehat{\boldsymbol n}\|}
\right].
}
$$

## 7. 由压力真解计算边界通量

若算例只给出压力真解 $p_{\mathrm{ex}}$，则先由 Darcy 定律计算速度：

$$
\boldsymbol u_{\mathrm{ex}}
=
-\boldsymbol\kappa\nabla p_{\mathrm{ex}}
+\boldsymbol f.
$$

于是物理边界函数为

$$
\boxed{
g
=
\left(
-\boldsymbol\kappa\nabla p_{\mathrm{ex}}
+\boldsymbol f
\right)\cdot\boldsymbol n.
}
$$

若

$$
p_{\mathrm{ex}}(F_E(\widehat{\boldsymbol x}))
=
\widehat p_{\mathrm{ex}}(\widehat{\boldsymbol x}),
$$

则梯度变换为

$$
\nabla p_{\mathrm{ex}}
=
J^{-T}\widehat\nabla\widehat p_{\mathrm{ex}}.
$$

因此参考边通量为

$$
\boxed{
\widehat g
=
J_E\widehat{\boldsymbol n}^{T}J^{-1}
\left[
-\boldsymbol\kappa J^{-T}
\widehat\nabla\widehat p_{\mathrm{ex}}
+\boldsymbol f
\right].
}
$$

若 $\boldsymbol f=\boldsymbol0$ 且 $\boldsymbol\kappa=\kappa I$，则

$$
\boxed{
\widehat g
=
-\kappa J_E
\widehat{\boldsymbol n}^{T}
J^{-1}J^{-T}
\widehat\nabla\widehat p_{\mathrm{ex}}.
}
$$

## 8. 混合虚拟元的边法向矩自由度

设 $k$ 阶混合虚拟元在边 $e$ 上的法向矩自由度为

$$
\boxed{
\operatorname{dof}_{e,\alpha}(\boldsymbol v)
=
\frac{1}{c_{e,\alpha}}
\int_e
(\boldsymbol v\cdot\boldsymbol n_e)
m_\alpha^e
\,\mathrm ds,
\qquad
m_\alpha^e\in\mathbb P_k(e).
}
$$

其中 $c_{e,\alpha}$ 由采用的归一化方式决定。例如：

$$
c_{e,\alpha}=1
$$

对应积分型自由度，而

$$
c_{e,\alpha}=|e|
$$

常用于边平均矩型自由度。

在通量边界 $e\subset\Gamma_N$ 上，必须强制赋值

$$
\boxed{
U_{e,\alpha}^{\mathrm{bc}}
=
\frac{1}{c_{e,\alpha}}
\int_e g\,m_\alpha^e\,\mathrm ds.
}
$$

如果 $g$ 由真解产生，则

$$
\boxed{
U_{e,\alpha}^{\mathrm{bc}}
=
\frac{1}{c_{e,\alpha}}
\int_e
(\boldsymbol u_{\mathrm{ex}}\cdot\boldsymbol n_e)
m_\alpha^e
\,\mathrm ds.
}
$$

## 9. 边矩自由度的参考域计算

定义物理边单项式的拉回

$$
\widehat m_\alpha(\widehat{\boldsymbol x})
=
m_\alpha^e(F_E(\widehat{\boldsymbol x})).
$$

利用法向通量积分不变性，得到

$$
\begin{aligned}
U_{e,\alpha}^{\mathrm{bc}}
&=
\frac{1}{c_{e,\alpha}}
\int_e
(\boldsymbol u_{\mathrm{ex}}\cdot\boldsymbol n_e)
m_\alpha^e\,\mathrm ds\\
&=
\frac{1}{c_{e,\alpha}}
\int_{\widehat e}
(\widehat{\boldsymbol u}_{\mathrm{ex}}
\cdot\widehat{\boldsymbol n}_e)
\widehat m_\alpha
\,\mathrm d\widehat s.
\end{aligned}
$$

所以

$$
\boxed{
U_{e,\alpha}^{\mathrm{bc}}
=
\frac{1}{c_{e,\alpha}}
\int_{\widehat e}
J_E\widehat{\boldsymbol n}_e^TJ^{-1}
\boldsymbol u_{\mathrm{ex}}(F_E(\widehat{\boldsymbol x}))
\widehat m_\alpha
\,\mathrm d\widehat s.
}
$$

等价地，若先计算物理边界函数 $g$，则

$$
\boxed{
U_{e,\alpha}^{\mathrm{bc}}
=
\frac{1}{c_{e,\alpha}}
\int_{\widehat e}
J_e(g\circ F_E)
\widehat m_\alpha
\,\mathrm d\widehat s.
}
$$

上面两个公式只能选择一种实现：

1. 使用 Piola 拉回速度 $\widehat{\boldsymbol u}_{\mathrm{ex}}$ 时，不再额外乘 $J_e$；
2. 使用物理法向分量 $g=\boldsymbol u_{\mathrm{ex}}\cdot\boldsymbol n$ 时，参考边积分必须乘 $J_e$。

不能同时在 $\widehat{\boldsymbol u}_{\mathrm{ex}}$ 中包含 Piola 因子，又在积分中额外乘一次 $J_e$。

## 10. 一阶边自由度

若边上的法向矩空间取 $\mathbb P_1(e)$，则每条边有两个法向矩自由度。可取一组尺度化物理边基

$$
m_0^e=1,
\qquad
m_1^e=\frac{s-s_e}{h_e},
$$

其中 $s$ 是物理曲边上的弧长坐标，$s_e$ 是边弧长中点坐标，$h_e=|e|$。

两个边界自由度为

$$
\boxed{
U_{e,0}^{\mathrm{bc}}
=
\frac{1}{c_{e,0}}
\int_e g\,\mathrm ds,
}
$$

以及

$$
\boxed{
U_{e,1}^{\mathrm{bc}}
=
\frac{1}{c_{e,1}}
\int_e g\,m_1^e\,\mathrm ds.
}
$$

参考域形式为

$$
\boxed{
U_{e,0}^{\mathrm{bc}}
=
\frac{1}{c_{e,0}}
\int_{\widehat e}
(\widehat{\boldsymbol u}_{\mathrm{ex}}
\cdot\widehat{\boldsymbol n})
\,\mathrm d\widehat s,
}
$$

$$
\boxed{
U_{e,1}^{\mathrm{bc}}
=
\frac{1}{c_{e,1}}
\int_{\widehat e}
(\widehat{\boldsymbol u}_{\mathrm{ex}}
\cdot\widehat{\boldsymbol n})
\widehat m_1
\,\mathrm d\widehat s,
}
$$

其中

$$
\widehat m_1
=
m_1^e\circ F_E.
$$

需要注意：如果 $m_1^e$ 是用物理弧长定义的，那么在一般非线性曲边映射下，$\widehat m_1$ 不一定等于参考参数 $t$。只有当自由度本身明确采用参考边多项式基，或者物理弧长与参考参数为仿射关系时，才能直接写成

$$
\widehat m_1(t)=t.
$$

因此程序中必须与实际自由度定义保持一致，不能把物理弧长单项式和参考参数单项式混用。

## 11. 参考边上的高斯积分

设参考边参数为

$$
\widehat{\boldsymbol\gamma}(t),
\qquad t\in[-1,1].
$$

则

$$
\mathrm d\widehat s
=
\left\|\widehat{\boldsymbol\gamma}'(t)\right\|\mathrm dt.
$$

边界自由度可写成

$$
\boxed{
\begin{aligned}
U_{e,\alpha}^{\mathrm{bc}}
=
\frac{1}{c_{e,\alpha}}
\int_{-1}^{1}
&J_E(t)\widehat{\boldsymbol n}^{T}J(t)^{-1}
\boldsymbol u_{\mathrm{ex}}(\boldsymbol x(t))\\
&\cdot\widehat m_\alpha(t)
\left\|\widehat{\boldsymbol\gamma}'(t)\right\|
\,\mathrm dt,
\end{aligned}
}
$$

其中

$$
\boldsymbol x(t)
=
F_E(\widehat{\boldsymbol\gamma}(t)).
$$

采用高斯点 $(t_q,w_q)$ 后，数值积分为

$$
\boxed{
\begin{aligned}
U_{e,\alpha}^{\mathrm{bc}}
\approx
\frac{1}{c_{e,\alpha}}
\sum_{q=1}^{N_q}w_q
&J_{E,q}\widehat{\boldsymbol n}^{T}J_q^{-1}
\boldsymbol u_{\mathrm{ex}}(\boldsymbol x_q)\\
&\cdot\widehat m_\alpha(t_q)
\left\|\widehat{\boldsymbol\gamma}'(t_q)\right\|.
\end{aligned}
}
$$

若参考边直接是 $[-1,1]$ 上的水平边或竖直边，并以单位速度参数化，则

$$
\left\|\widehat{\boldsymbol\gamma}'(t)\right\|=1.
$$

此时公式简化为

$$
\boxed{
U_{e,\alpha}^{\mathrm{bc}}
\approx
\frac{1}{c_{e,\alpha}}
\sum_{q=1}^{N_q}w_q
J_{E,q}\widehat{\boldsymbol n}^{T}J_q^{-1}
\boldsymbol u_{\mathrm{ex}}(\boldsymbol x_q)
\widehat m_\alpha(t_q).
}
$$

## 12. 二维曲边参数化下的等价公式

设物理曲边按区域边界逆时针方向参数化为

$$
\boldsymbol\gamma(t)
=
\begin{bmatrix}
x(t)\\y(t)
\end{bmatrix},
\qquad t\in[a,b].
$$

则

$$
\boldsymbol n\,\mathrm ds
=
\begin{bmatrix}
y'(t)\\-x'(t)
\end{bmatrix}\mathrm dt.
$$

若

$$
\boldsymbol u_{\mathrm{ex}}
=
\begin{bmatrix}
u_{1,\mathrm{ex}}\\u_{2,\mathrm{ex}}
\end{bmatrix},
$$

则边界自由度也可直接写成

$$
\boxed{
\begin{aligned}
U_{e,\alpha}^{\mathrm{bc}}
=
\frac{1}{c_{e,\alpha}}
\int_a^b
&\left[
u_{1,\mathrm{ex}}(\boldsymbol\gamma(t))y'(t)
-u_{2,\mathrm{ex}}(\boldsymbol\gamma(t))x'(t)
\right]\\
&\cdot m_\alpha^e(\boldsymbol\gamma(t))
\,\mathrm dt.
\end{aligned}
}
$$

如果曲边参数方向是顺时针，上式整体需要乘以 $-1$，以保证使用区域单位外法向。

该公式与 Piola 参考域公式应给出相同的数值，可以用于程序交叉验证。

## 13. 全局边方向与符号

在 $H(\operatorname{div})$ 离散空间中，通常为每条全局边预先规定一个全局单位法向 $\boldsymbol n_e^{\mathrm g}$。单元 $E$ 的外法向 $\boldsymbol n_E$ 与全局边法向之间满足

$$
\boldsymbol n_E|_e
=
\sigma_{E,e}\boldsymbol n_e^{\mathrm g},
\qquad
\sigma_{E,e}\in\{-1,1\}.
$$

局部边自由度和全局边自由度之间相应满足

$$
\boxed{
U_{E,e,\alpha}^{\mathrm{local}}
=
\sigma_{E,e}
U_{e,\alpha}^{\mathrm{global}}.
}
$$

对于物理边界，最方便的做法是令全局边法向直接等于区域外法向。此时

$$
\sigma_{E,e}=1.
$$

如果程序的全局边方向不是这样定义的，则由真解算出的边界通量矩必须乘相应的方向符号。

## 14. 混合离散系统

混合虚拟元离散系统写成

$$
\boxed{
\begin{bmatrix}
A&-B^T\\
B&0
\end{bmatrix}
\begin{bmatrix}
U\\P
\end{bmatrix}
=
\begin{bmatrix}
F\\G
\end{bmatrix}.
}
$$

其中

$$
U\in\mathbb R^{N_u}
$$

是全部速度自由度，

$$
P\in\mathbb R^{N_p}
$$

是压力自由度。

设 $D$ 是位于通量边界 $\Gamma_N$ 上的速度自由度编号集合，由真解或边界函数计算得到的规定值组成向量

$$
\overline U_D
=
\left\{U_{e,\alpha}^{\mathrm{bc}}\right\}_{(e,\alpha)\in D}.
$$

其余未知速度自由度编号集合记为 $I$。

## 15. 分块消元推导

按照内部速度自由度、边界速度自由度和压力自由度重新排列系统：

$$
\begin{bmatrix}
A_{II}&A_{ID}&-B_I^T\\
A_{DI}&A_{DD}&-B_D^T\\
B_I&B_D&0
\end{bmatrix}
\begin{bmatrix}
U_I\\U_D\\P
\end{bmatrix}
=
\begin{bmatrix}
F_I\\F_D\\G
\end{bmatrix}.
$$

强制代入

$$
U_D=\overline U_D.
$$

保留内部速度方程和质量守恒方程，得到

$$
A_{II}U_I-B_I^TP
=
F_I-A_{ID}\overline U_D,
$$

以及

$$
B_IU_I
=
G-B_D\overline U_D.
$$

因此约化系统为

$$
\boxed{
\begin{bmatrix}
A_{II}&-B_I^T\\
B_I&0
\end{bmatrix}
\begin{bmatrix}
U_I\\P
\end{bmatrix}
=
\begin{bmatrix}
F_I-A_{ID}\overline U_D\\
G-B_D\overline U_D
\end{bmatrix}.
}
$$

边界通量不仅修改速度方程右端

$$
F_I^{\mathrm{new}}
=
F_I-A_{ID}\overline U_D,
$$

还必须修改质量守恒方程右端

$$
\boxed{
G^{\mathrm{new}}
=
G-B_D\overline U_D.
}
$$

遗漏第二项会破坏离散质量守恒。

## 16. 保持完整矩阵尺寸的对角线置一方法

定义完整鞍点矩阵、未知量和右端：

$$
K=
\begin{bmatrix}
A&-B^T\\B&0
\end{bmatrix},
\qquad
X=
\begin{bmatrix}
U\\P
\end{bmatrix},
\qquad
R=
\begin{bmatrix}
F\\G
\end{bmatrix}.
$$

把 $D$ 理解为完整未知向量 $X$ 中对应边界速度自由度的编号。若希望同时清零边界自由度对应的行和列，以保持矩阵的对称结构，正确顺序为：

### 第一步：利用原矩阵列修正右端

$$
\boxed{
R\leftarrow R-K(:,D)\overline U_D.
}
$$

这一步必须在清零矩阵列之前完成。

### 第二步：清零边界自由度的行和列

$$
\boxed{
K(D,:)=0,
\qquad
K(:,D)=0.
}
$$

### 第三步：边界自由度对角线置一

$$
\boxed{
K(D,D)=I.
}
$$

### 第四步：边界行右端写入规定自由度

$$
\boxed{
R(D)=\overline U_D.
}
$$

最终边界行就是

$$
U_D=\overline U_D.
$$

完整算法概括为

$$
\boxed{
\begin{aligned}
R&\leftarrow R-K(:,D)\overline U_D,\\
K(D,:)&\leftarrow0,\\
K(:,D)&\leftarrow0,\\
K(D,D)&\leftarrow I,\\
R(D)&\leftarrow\overline U_D.
\end{aligned}
}
$$

如果只清零边界行而保留矩阵列，则其他方程中仍保留已知的 $U_D$，此时不能再做同样的列消元右端修正。但这种处理会破坏原矩阵对称性，通常不如行列同时消元清晰。

## 17. 求解后的完整速度自由度

真解只用于计算通量边界上的速度自由度

$$
U_D=\overline U_D.
$$

内部速度自由度 $U_I$ 和压力自由度 $P$ 仍由离散系统求得。求解后，完整速度向量为

$$
\boxed{
(U_h)_i
=
\begin{cases}
(\overline U_D)_i,
&i\in D,\\[2mm]
(U_I)_i,
&i\in I.
\end{cases}
}
$$

因此计算过程是

$$
\boxed{
\boldsymbol u_{\mathrm{ex}}
\longrightarrow
\overline U_D
\longrightarrow
\text{修改混合系统}
\longrightarrow
\text{求解 }(U_I,P)
\longrightarrow
U_h.
}
$$

不能把真解在所有边上的自由度都强制写入系统，否则相当于预先指定了全部速度未知量，不再是在求解原偏微分方程。

## 18. 数值解与真解插值的区别

为计算误差，可以在所有边上构造真解的虚拟元插值自由度：

$$
\boxed{
(U_{\mathrm{ex}}^I)_{e,\alpha}
=
\frac{1}{c_{e,\alpha}}
\int_e
(\boldsymbol u_{\mathrm{ex}}\cdot\boldsymbol n_e)
m_\alpha^e
\,\mathrm ds.
}
$$

其中：

- $U_h$ 是数值系统的解，只有边界部分被强制指定；
- $U_{\mathrm{ex}}^I$ 是真解在离散自由度上的插值，用于误差比较；
- $U_{\mathrm{ex}}^I$ 不能替代 $U_h$ 参与正常求解。

## 19. 边界通量相容性与压力唯一性

若整个边界都给定法向通量，即

$$
\Gamma_N=\partial\Omega,
$$

则由散度定理必须满足相容条件

$$
\boxed{
\int_{\partial\Omega}g\,\mathrm ds
=
\int_\Omega f_0\,\mathrm d\boldsymbol x.
}
$$

离散情况下，对边界零阶通量自由度求和应与离散体源积分一致。

纯通量边界问题中的压力通常只确定到一个常数，因此还需要施加例如

$$
\boxed{
\int_\Omega p_h\,\mathrm d\boldsymbol x=0
}
$$

的零均值条件，或固定一个等价的压力约束。

## 20. 程序计算逻辑

对每条通量边界曲边 $e$，以及该边的每个法向矩指标 $\alpha$，执行以下计算：

1. 取得参考边、参考外法向 $\widehat{\boldsymbol n}$、全局边编号和方向符号；
2. 在参考边高斯点 $t_q$ 上计算 $\widehat{\boldsymbol x}_q$；
3. 通过 $F_E$ 得到物理坐标 $\boldsymbol x_q$；
4. 计算 $J_q$、$J_{E,q}=\det J_q$；
5. 在物理坐标 $\boldsymbol x_q$ 处计算真速度 $\boldsymbol u_{\mathrm{ex}}(\boldsymbol x_q)$；
6. 计算参考法向通量

   $$
   \widehat g_q
   =
   J_{E,q}\widehat{\boldsymbol n}^{T}J_q^{-1}
   \boldsymbol u_{\mathrm{ex}}(\boldsymbol x_q);
   $$

7. 计算与自由度定义一致的 $\widehat m_\alpha(t_q)$；
8. 累加高斯积分，得到 $U_{e,\alpha}^{\mathrm{bc}}$；
9. 乘全局边方向符号并写入 $\overline U_D$；
10. 对完整混合矩阵执行右端修正和行列消元；
11. 求解系统并恢复完整速度向量 $U_h$。

## 21. 可用于程序验证的恒等式

在每个边高斯点上，可以同时计算

$$
\widehat g_q^{(1)}
=
J_{E,q}\widehat{\boldsymbol n}^{T}J_q^{-1}
\boldsymbol u_{\mathrm{ex}}(\boldsymbol x_q),
$$

以及

$$
\widehat g_q^{(2)}
=
J_{e,q}
\left[
\boldsymbol u_{\mathrm{ex}}(\boldsymbol x_q)
\cdot\boldsymbol n_q
\right],
$$

其中

$$
\boldsymbol n_q
=
\frac{J_q^{-T}\widehat{\boldsymbol n}}
{\|J_q^{-T}\widehat{\boldsymbol n}\|},
\qquad
J_{e,q}
=
J_{E,q}\|J_q^{-T}\widehat{\boldsymbol n}\|.
$$

理论上必须满足

$$
\boxed{
\widehat g_q^{(1)}=\widehat g_q^{(2)}.
}
$$

还可以比较物理曲边积分和参考边积分：

$$
\boxed{
\int_e
(\boldsymbol u_{\mathrm{ex}}\cdot\boldsymbol n)m_\alpha^e
\,\mathrm ds
=
\int_{\widehat e}
(\widehat{\boldsymbol u}_{\mathrm{ex}}\cdot\widehat{\boldsymbol n})
\widehat m_\alpha
\,\mathrm d\widehat s.
}
$$

这两个恒等式适合检查雅可比、法向方向、边参数方向和积分权重是否正确。

## 22. 最终结论

曲边混合虚拟元的法向通量边界处理可以概括为

$$
\boxed{
\begin{aligned}
\boldsymbol x
&=F_E(\widehat{\boldsymbol x}),\\[1mm]
\widehat{\boldsymbol u}_{\mathrm{ex}}
&=J_EJ^{-1}
\boldsymbol u_{\mathrm{ex}}(F_E(\widehat{\boldsymbol x})),\\[1mm]
\widehat{\boldsymbol u}_{\mathrm{ex}}
\cdot\widehat{\boldsymbol n}
&=J_e
(\boldsymbol u_{\mathrm{ex}}\cdot\boldsymbol n),\\[1mm]
U_{e,\alpha}^{\mathrm{bc}}
&=
\frac{1}{c_{e,\alpha}}
\int_{\widehat e}
(\widehat{\boldsymbol u}_{\mathrm{ex}}\cdot\widehat{\boldsymbol n})
\widehat m_\alpha
\,\mathrm d\widehat s,\\[1mm]
R
&\leftarrow R-K(:,D)\overline U_D,\\[1mm]
K(D,:)&=0,\qquad K(:,D)=0,\\[1mm]
K(D,D)&=I,\qquad R(D)=\overline U_D.
\end{aligned}
}
$$

最关键的逻辑是：

1. $\boldsymbol u_{\mathrm{ex}}\circ F_E$ 只是物理真解在映射点的取值，不是参考速度；
2. 参考速度必须再乘 $J_EJ^{-1}$ 完成 Piola 拉回；
3. Piola 保持法向通量积分，不保持法向分量点值；
4. 一阶边界条件需要同时计算零阶矩和一阶矩；
5. 行列同时消元时，必须先用原矩阵列修正包括压力方程在内的完整右端。
