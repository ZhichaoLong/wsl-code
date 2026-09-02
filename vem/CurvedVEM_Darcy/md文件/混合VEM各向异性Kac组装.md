# 混合虚拟元方法中各向异性系数的 \(K_c^a\) 矩阵组装

## 1. 各向异性系数

设单元 \(E\) 上的各向异性系数为

$$
\nu(\boldsymbol x)=
\begin{bmatrix}
\nu_{11}(\boldsymbol x) & \nu_{12}(\boldsymbol x)\\
\nu_{21}(\boldsymbol x) & \nu_{22}(\boldsymbol x)
\end{bmatrix}.
$$

通常扩散张量是对称正定的，因此

$$
\nu_{12}=\nu_{21}.
$$

如果原方程中给出的是扩散张量 \(\kappa\)，则混合形式中的系数为

$$
\nu=\kappa^{-1}.
$$

这里的逆是矩阵逆。

## 2. 混合 VEM 的一致性项

混合虚拟元方法中的局部一致性双线性型为

$$
a_{h,c}^E(\boldsymbol u_h,\boldsymbol v_h)
=
\int_E
\left(\Pi_k^0\boldsymbol v_h\right)^T
\nu(\boldsymbol x)
\left(\Pi_k^0\boldsymbol u_h\right)
\,\mathrm d\boldsymbol x.
$$

设局部虚拟元空间的基函数为

$$
\{\boldsymbol\phi_i\}_{i=1}^{N_{\mathrm{dof}}},
$$

则局部一致性矩阵定义为

$$
(K_c^a)_{ij}
=
a_{h,c}^E(\boldsymbol\phi_j,\boldsymbol\phi_i).
$$

因此

$$
(K_c^a)_{ij}
=
\int_E
\left(\Pi_k^0\boldsymbol\phi_i\right)^T
\nu
\left(\Pi_k^0\boldsymbol\phi_j\right)
\,\mathrm d\boldsymbol x.
$$

## 3. \(L^2\) 投影展开

设

$$
\{\boldsymbol g_I^k\}_{I=1}^{2n_k}
$$

是 \([\mathbb P_k(E)]^2\) 的一组向量多项式基。将投影写成

$$
\Pi_k^0\boldsymbol\phi_j
=
\sum_{J=1}^{2n_k}
(\Pi_k^{*0})_{Jj}\boldsymbol g_J^k.
$$

同理，

$$
\Pi_k^0\boldsymbol\phi_i
=
\sum_{I=1}^{2n_k}
(\Pi_k^{*0})_{Ii}\boldsymbol g_I^k.
$$

论文中的投影系数矩阵为

$$
\boxed{
\Pi_k^{*0}=G^{-1}B
}
$$

其中无权 Gram 矩阵为

$$
G_{IJ}
=
\int_E
\boldsymbol g_I^k\cdot\boldsymbol g_J^k
\,\mathrm d\boldsymbol x.
$$

即使 \(\nu\) 是各向异性矩阵，投影矩阵仍由无权矩阵 \(G\) 计算：

$$
\Pi_k^{*0}=G^{-1}B.
$$

不能将其改成

$$
\Pi_k^{*0}=(G^\nu)^{-1}B.
$$

## 4. 各向异性带权矩阵 \(G^\nu\)

把投影展开代入一致性项：

$$
\begin{aligned}
(K_c^a)_{ij}
={}&
\int_E
\left[
\sum_{I=1}^{2n_k}
(\Pi_k^{*0})_{Ii}\boldsymbol g_I^k
\right]^T
\nu
\left[
\sum_{J=1}^{2n_k}
(\Pi_k^{*0})_{Jj}\boldsymbol g_J^k
\right]
\,\mathrm d\boldsymbol x\\
={}&
\sum_{I=1}^{2n_k}
\sum_{J=1}^{2n_k}
(\Pi_k^{*0})_{Ii}
(\Pi_k^{*0})_{Jj}
\int_E
(\boldsymbol g_I^k)^T
\nu
\boldsymbol g_J^k
\,\mathrm d\boldsymbol x.
\end{aligned}
$$

定义新的带权 Gram 矩阵

$$
\boxed{
(G^\nu)_{IJ}
=
\int_E
(\boldsymbol g_I^k)^T
\nu(\boldsymbol x)
\boldsymbol g_J^k
\,\mathrm d\boldsymbol x
}
$$

其中

$$
G^\nu\in\mathbb R^{2n_k\times2n_k}.
$$

于是

$$
(K_c^a)_{ij}
=
\sum_{I=1}^{2n_k}
\sum_{J=1}^{2n_k}
(\Pi_k^{*0})_{Ii}
(G^\nu)_{IJ}
(\Pi_k^{*0})_{Jj}.
$$

因此矩阵形式为

$$
\boxed{
K_c^a
=
(\Pi_k^{*0})^T
G^\nu
\Pi_k^{*0}.
}
$$

## 5. \(G^\nu\) 的逐元素展开

令

$$
\boldsymbol g_I^k=
\begin{bmatrix}
g_{I,x}\\
g_{I,y}
\end{bmatrix},
\qquad
\boldsymbol g_J^k=
\begin{bmatrix}
g_{J,x}\\
g_{J,y}
\end{bmatrix}.
$$

首先有

$$
\nu\boldsymbol g_J^k
=
\begin{bmatrix}
\nu_{11}g_{J,x}+\nu_{12}g_{J,y}\\
\nu_{21}g_{J,x}+\nu_{22}g_{J,y}
\end{bmatrix}.
$$

因此

$$
\begin{aligned}
(\boldsymbol g_I^k)^T\nu\boldsymbol g_J^k
={}&
g_{I,x}(\nu_{11}g_{J,x}+\nu_{12}g_{J,y})\\
&+g_{I,y}(\nu_{21}g_{J,x}+\nu_{22}g_{J,y})\\
={}&
\nu_{11}g_{I,x}g_{J,x}
+\nu_{12}g_{I,x}g_{J,y}\\
&+\nu_{21}g_{I,y}g_{J,x}
+\nu_{22}g_{I,y}g_{J,y}.
\end{aligned}
$$

所以 \(G^\nu\) 的每个元素是四个积分之和：

$$
\boxed{
\begin{aligned}
(G^\nu)_{IJ}
={}&
\int_E \nu_{11}g_{I,x}g_{J,x}\,\mathrm d\boldsymbol x\\
&+\int_E \nu_{12}g_{I,x}g_{J,y}\,\mathrm d\boldsymbol x\\
&+\int_E \nu_{21}g_{I,y}g_{J,x}\,\mathrm d\boldsymbol x\\
&+\int_E \nu_{22}g_{I,y}g_{J,y}\,\mathrm d\boldsymbol x.
\end{aligned}
}
$$

## 6. 使用分量单项式基时的分块形式

设标量多项式基为

$$
\{m_\alpha\}_{\alpha=1}^{n_k}.
$$

定义简单向量多项式基

$$
\boldsymbol m_\alpha^x=
\begin{bmatrix}
m_\alpha\\0
\end{bmatrix},
\qquad
\boldsymbol m_\alpha^y=
\begin{bmatrix}
0\\m_\alpha
\end{bmatrix}.
$$

基函数按照先 \(x\) 分量、后 \(y\) 分量的顺序排列：

$$
\boldsymbol m_1^x,\ldots,\boldsymbol m_{n_k}^x,
\boldsymbol m_1^y,\ldots,\boldsymbol m_{n_k}^y.
$$

定义四个 \(n_k\times n_k\) 矩阵

$$
\boxed{
(H^{\nu_{ab}})_{\alpha\beta}
=
\int_E
\nu_{ab}m_\alpha m_\beta
\,\mathrm d\boldsymbol x,
\qquad a,b\in\{1,2\}.
}
$$

在这组分量单项式基下，带权 Gram 矩阵为

$$
\boxed{
\widehat G^\nu
=
\begin{bmatrix}
H^{\nu_{11}} & H^{\nu_{12}}\\
H^{\nu_{21}} & H^{\nu_{22}}
\end{bmatrix}.
}
$$

四个分块分别来自

$$
(\boldsymbol m_\alpha^x)^T
\nu\boldsymbol m_\beta^x
=
\nu_{11}m_\alpha m_\beta,
$$

$$
(\boldsymbol m_\alpha^x)^T
\nu\boldsymbol m_\beta^y
=
\nu_{12}m_\alpha m_\beta,
$$

$$
(\boldsymbol m_\alpha^y)^T
\nu\boldsymbol m_\beta^x
=
\nu_{21}m_\alpha m_\beta,
$$

以及

$$
(\boldsymbol m_\alpha^y)^T
\nu\boldsymbol m_\beta^y
=
\nu_{22}m_\alpha m_\beta.
$$

## 7. 从分量单项式基变换到论文中的向量基

论文使用的基

$$
\{\boldsymbol g_I^k\}_{I=1}^{2n_k}
$$

由梯度空间基和补空间基组成。设它与分量单项式基之间的关系为

$$
\boldsymbol g_I^k
=
\sum_{A=1}^{2n_k}
T_{IA}\boldsymbol m_A.
$$

其中 \(T\in\mathbb R^{2n_k\times2n_k}\) 是基变换矩阵。

于是

$$
\begin{aligned}
(G^\nu)_{IJ}
&=
\int_E
(\boldsymbol g_I^k)^T
\nu\boldsymbol g_J^k
\,\mathrm d\boldsymbol x\\
&=
\sum_{A=1}^{2n_k}
\sum_{C=1}^{2n_k}
T_{IA}T_{JC}
\int_E
\boldsymbol m_A^T\nu\boldsymbol m_C
\,\mathrm d\boldsymbol x.
\end{aligned}
$$

因此

$$
\boxed{
G^\nu=T\widehat G^\nu T^T.
}
$$

代入一致性矩阵，得到

$$
\boxed{
K_c^a
=
(\Pi_k^{*0})^T
T
\begin{bmatrix}
H^{\nu_{11}} & H^{\nu_{12}}\\
H^{\nu_{21}} & H^{\nu_{22}}
\end{bmatrix}
T^T
\Pi_k^{*0}.
}
$$

## 8. 等价的四项分块表达式

令

$$
Q=T^T\Pi_k^{*0}
=
\begin{bmatrix}
Q_x\\Q_y
\end{bmatrix},
$$

其中

$$
Q_x,Q_y\in\mathbb R^{n_k\times N_{\mathrm{dof}}}.
$$

则

$$
K_c^a
=
Q^T\widehat G^\nu Q.
$$

将其按照分块矩阵乘法展开，得到

$$
\boxed{
\begin{aligned}
K_c^a
={}&
Q_x^TH^{\nu_{11}}Q_x
+Q_x^TH^{\nu_{12}}Q_y\\
&+Q_y^TH^{\nu_{21}}Q_x
+Q_y^TH^{\nu_{22}}Q_y.
\end{aligned}
}
$$

这四项分别对应 \(\nu\) 的四个分量。

## 9. 单元内常系数的情形

如果 \(\nu\) 在单元 \(E\) 内为常矩阵，定义普通标量质量矩阵

$$
H_{\alpha\beta}
=
\int_E
m_\alpha m_\beta
\,\mathrm d\boldsymbol x.
$$

则

$$
H^{\nu_{11}}=\nu_{11}H,
\qquad
H^{\nu_{12}}=\nu_{12}H,
$$

$$
H^{\nu_{21}}=\nu_{21}H,
\qquad
H^{\nu_{22}}=\nu_{22}H.
$$

因此

$$
\widehat G^\nu
=
\begin{bmatrix}
\nu_{11}H & \nu_{12}H\\
\nu_{21}H & \nu_{22}H
\end{bmatrix}.
$$

按照当前的基排列方式，也可写为

$$
\widehat G^\nu=\nu\otimes H.
$$

所以

$$
\boxed{
K_c^a
=
(\Pi_k^{*0})^T
T(\nu\otimes H)T^T
\Pi_k^{*0}.
}
$$

## 10. 对称性与矩阵维数

若

$$
\nu^T=\nu,
$$

则

$$
(G^\nu)^T=G^\nu
$$

并且

$$
(K_c^a)^T=K_c^a.
$$

矩阵维数为

$$
\Pi_k^{*0}
\in\mathbb R^{2n_k\times N_{\mathrm{dof}}},
$$

$$
G^\nu\in\mathbb R^{2n_k\times2n_k},
$$

因此

$$
\underbrace{(\Pi_k^{*0})^T}_{N_{\mathrm{dof}}\times2n_k}
\underbrace{G^\nu}_{2n_k\times2n_k}
\underbrace{\Pi_k^{*0}}_{2n_k\times N_{\mathrm{dof}}}
=
\underbrace{K_c^a}_{N_{\mathrm{dof}}\times N_{\mathrm{dof}}}.
$$

## 11. 最终结论

各向异性混合 VEM 一致性矩阵的数学组装过程为

$$
\boxed{
\begin{aligned}
\Pi_k^{*0}
&=G^{-1}B,\\[2mm]
(G^\nu)_{IJ}
&=
\int_E
(\boldsymbol g_I^k)^T
\nu\boldsymbol g_J^k
\,\mathrm d\boldsymbol x,\\[2mm]
K_c^a
&=
(\Pi_k^{*0})^T
G^\nu
\Pi_k^{*0}.
\end{aligned}
}
$$

其中，\(G^\nu\) 的每个元素等于 \(\nu_{11}\)、\(\nu_{12}\)、\(\nu_{21}\) 和 \(\nu_{22}\) 所对应的四个积分之和。
