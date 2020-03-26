---
layout:     post
title:      凸优化第二章
subtitle:   学习笔记
date:       2020-03-22
author:     DT
header-img: img/post-bg-debug.png
catalog: true
tags:
    - notes
    - 凸优化



---

> 凸优化第二章学习笔记

# 2 凸集

## 2.1 仿射集合和凸集

**直线、线段**：$x_1$和$x_2$为$\mathbb{R}^n$中两个相异的点，具有下列形式的点
$$
y=\theta x_1 + (1-\theta)x_2=x_2+\theta(x_1-x_2),\quad \theta \in \mathbb{R}
$$
（注意上式中$x_1$和$x_2$是给定的，$\theta$则是任意变化的）组成了一条穿过$x_1$和$x_2$的**直线**。当$0\leq\theta\leq1$的时候，上式表示的则是由$x_1$和$x_2$构成的**线段**。当$\theta > 1$时，表示直线上越过$x_1$的部分；当$\theta < 0$时，表示直接上越过$x_2$的部分。（上述陈述可以通过平行四边形法则简单地即可验证）

**仿射集合**：如果集合$C$中任意两点所形成的直线仍然在集合$C$中，那么称集合$C$是仿射的；等价说法：对于任意的$x_1 \in \mathbb{R}^n, x_2 \in \mathbb{R}^n, \theta \in \mathbb{R}$，有$\theta x_1+(1-\theta)x_2 \in C$；等价说法：$C$中包含了$C$中任意两点的系数之和为1的线性组合。（想想之前线性函数的定义，以及线性规划的定义；自然地，可以将线性规划/函数 与纺射集和联系到一起）

**仿射组合**：如果$\theta_1+...+\theta_k=1$，$\theta_1x_1+...+\theta_kx_k$称为$x_1,...,x_k$的仿射组合。一个仿射集合包含其中任意点的仿射组合，即：若$C$为一个仿射集合，$x_1,...,x_k \in C$，且$\theta_1+...+\theta_k=1$，那么$\theta_1x_1+...+\theta_kx_k$也在$C$中。（Q2.1：上述陈述到底是一个概念的推论还是推演出来的一个结论？若是前者，仿射集合的精确定义到底是什么？若是后者怎么推导？）

**子空间**：如果$C$为一个仿射集合，且$x_0 \in C$，那么集合
$$
V = \{x-x_0|x \in C\}
$$
为一个子空间，即关于加法和数乘是封闭的。（证明：令$v_1,v_2 \in V$，则有$v_1+x_0,v_2+x_0 \in C$，（$x_0 \in C$自然不必说），$\alpha(v_1+x_0)+\beta(v_2+x_0)+(1-\alpha-\beta)x_0 \in C=\alpha v_1+\beta v_2+x_0 \in C$，则得到 $\alpha v_1+\beta v_2 \in V$ （这里$\alpha ,\beta \in \mathbb{R}$），显然对加法和数乘封闭）因此，仿射集合$C$可以表示为
$$
C = V + x_0=\{x+x_0|x \in V\},
$$
即为子空间加一个偏移量。与仿射集合$C$相关联的子空间$V$的选取与$x_0$无关，所以$x_0$可以是$C$中的任意一点。

**仿射集合的维数**：定义仿射集合$C$的维数为子空间$V=C-x_0$的维数，其中$x_0$为$C$中的任意元素。

**仿射包，集合$C$的仿射包**：称由**集合$C$中的点所形成的*所有仿射组合*所构成的集合**为$C$的仿射包，记为$\text{aff}\,C$
$$
\text{aff}\,C = \{\theta_1x_1+...+\theta_kx_k|x_1,...,x_k \in C,\theta_1+...+\theta_k=1\}
$$
**仿射包是包含$C$的最小仿射集合**，i.e.，如果集合$S$是一个仿射集合且$C\subseteq S$，那么$\text{aff}\,C \subseteq S$。

**仿射维数，集合$C$的仿射维数**：定义集合$C$的仿射维数为其仿射包的维数。如果集合$C \subseteq \mathbb{R}^n$的仿射维数小于$n$，那么这个集合在仿射集合$\text{aff}\,C\neq \mathbb{R}^n$中。（个人觉得这里的意思是，因为一个集合$C$是必然$\subseteq$$\text{aff}\,C$的，但是也有可能是$C=$$\text{aff}\,C$，所以这里应该强调的是$C$在仿射包**中**，也就是没有等于的可能性）

**集合$C$的相对内部**：定义集合$C$的内部为$\text{aff}\,C$的内部，记为$\text{relint}\,C$。不太理解；p21 p34

**集合$C$的相对边界**：定义集合$C$的相对边界为$\text{cl}\,C\,\backslash\,\text{relint}\,C$，$\text{cl}\,C$表示$C$的闭包。不太理解p21p3。

**凸集**：称集合$C$为凸集：如果集合$C$中任意两点之间的**线段**仍然在$C$中，也即：对于任意的$x_1,x_2\in C$以及满足$0\leq\theta\leq1$的$\theta$都有：
$$
\theta x_1+(1-\theta)x_2 \in C
$$
比较仿射集和凸集的定义，可见仿射集合必然是凸集。

**凸组合**：称点$\theta_1x_1+...+\theta_kx_k$为$x_1,...,x_k$的一个**凸组合**，其中$\theta_1+...+\theta_k=1$，且$\theta_i \geq 0,i=1,...,k$。 As with affine sets, it can be shown that a set is convex if and only if it contains every convex combination of its points. （Q2.2：这句话怎么证明）；**凸组合的概念可以拓展到无穷级数、积分以及大多数形式的概率分布**（p35 p22）。

**凸包**：称由集合$C$中的点所形成的**所有凸组合**所构成的集合为凸包，记为$\text{conv}\,C$：
$$
\text{conv}\,C = \{\theta_1x_1+...+\theta_kx_k|x_i \in C, \theta_i \geq 0, i=1,...,k, \theta_1+...+\theta_k=1\}
$$
凸包为包含了集合$C$的最小凸集，i.e.，若集合$B$是包含了集合$C$的凸集，那么$\text{conv}\,C \subseteq B$。

**锥/非负齐次**：如果对于任意的$x\in C$以及$\theta \geq0$都有$\theta x \in C$，那么称集合$C$是锥或者是非负齐次。

**凸锥**：如果集合$C$为锥，且是凸的（也就是即为锥集又是凸集），那么称$C$为凸锥，即：对于任意的$x_1,x_2 \in C,\theta_1,\theta_2\geq0$，都有
$$
\theta_1x_1+\theta_2x_2 \in C
$$
**锥组合，非负线性组合**：称点$\theta_1x_1+...+\theta_kx_k,\theta_1,...,\theta_k \geq0$为$x_1,...,x_k$的锥组合（非负线性组合）； a set $C$ is a convex cone if and only if it contains all conic combinations of its elements. （一个集合$C$是凸锥的充要条件是该集合包含了其元素的所有锥组合；Q2.3：怎么推出来的？）；同凸/仿射组合一样，锥组合的概念也可以拓展到无穷级数和积分中去。

**锥包，集合$C$的锥包**：称$C$中元素的**所有锥组合**为集合$C$的锥包，即：
$$
\{\theta_1x_1+...+\theta_kx_k|x_i \in C,\theta_i\geq0,i=1,...,k\}
$$
他是包含了集合$C$的最小凸锥。