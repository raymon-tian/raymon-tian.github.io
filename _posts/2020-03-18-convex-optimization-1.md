---
layout:     post
title:      凸优化第一章
subtitle:   读书笔记
date:       2020-03-18
author:     DT
header-img: img/post-bg-debug.png
catalog: true
tags:
    - notes
    - 凸优化


---

> 凸优化第一章学习笔记

# 1 引言

## 1.1 数学优化

数学优化问题，优化问题：数学优化问题或者优化问题可以写成如下形式:
$$
\text{minimize} \quad f_0(x) \\
\text{subject to} \quad f_i(x) \leq b_i, \quad i = 1,...,m
\tag{1.1}
\label{eq1.1}
$$


在公式$\eqref{eq1.1}$中，向量$x=(x_1,...,x_n)$称为问题的优化变量，$f_0 : \mathbb{R}^n \to \mathbb{R}$称为目标函数，$f_i(x) \leq b_i,i=1,...,m$称为（不等式）约束函数，$b_i,...,b_m$称为约束上限或者约束边界；问题$\eqref{eq1.1}$的解或者最优解即为满足约束条件下的使得目标函数最小的向量$x^*$。

线性函数：若对任意的$x \in \mathbb{R}^n$，$y \in \mathbb{R}^n$, $\alpha \in \mathbb{R}$, $\beta \in \mathbb{R}$ 都有
$$
f_i(\alpha x + \beta y) = \alpha f_i(x) + \beta f_i(y)
\tag{1.2}
\label{eq1.2}
$$
则称函数$f_i(x)$为线性函数。

线性规划，非线性规划：如果问题$\eqref{eq1.1}$中的目标函数以及约束函数，$f_0(x),...,f_m(x)$均为线性函数，则此优化问题称为**线性规划**；若优化问题不是线性的，则称之为**非线性规划**。

凸函数：对于任意的$x \in \mathbb{R}^n$，$y \in \mathbb{R}^n$，以及任意的$\alpha \in \mathbb{R}$, $\beta \in \mathbb{R}$ 且满足$\alpha + \beta = 1$ ，$\alpha \geq 0$，$\beta \geq 0$，下列不等式成立
$$
f_i(\alpha x + \beta y) \leq \alpha f_i(x) + \beta f_i(y)
\tag{1.3}
\label{eq1.3}
$$


则$f_i(x)$为**凸函数**。

凸优化：若问题$\eqref{eq1.1}$中的目标函数以及约束函数都是凸函数，那么该优化问题即为**凸优化问题**。线性规划问题也是凸优化问题，可以将凸优化问题视为线性优化问题的拓展。

优化问题是稀疏的：如果一个优化问题的**每个约束函数**仅仅依赖于为数不多的几个变量，那么**此问题称为是稀疏的**。

## 1.2 最小二乘和线性规划

广为人知且广泛应用的两类特殊凸优化问题：最小二乘、线性规划。

**最小二乘**：最小二乘是这样一类优化问题：其没有约束函数，且目标函数是若干项的平方之和，每项具有的形式为$a_i^Tx-b_i$ （向量点积减去一个标量），具体形式如下：
$$
\text{minimize} \quad f_0(x) = \left\|Ax-b\right\|_2^2=\sum_{i=1}^{k}(a_i^Tx-b_i)^2
\tag{1.4}
\label{eq1.4}
$$
其中，$A\in\mathbb{R}^{k\times n}(k \geq n)$ （Q1.2：为什么必须$k\geq n$），$a_i^T$为矩阵$A$的行向量，$x \in \mathbb{R}^n$为优化变量。

最小二乘的解：求解公式$\eqref{eq1.4}$转换成求解线性方程组$(A^TA)x=A^Tb$，得到解析解$x=(A^TA)^{-1}A^Tb$。（Q1.3：怎么推出来的）

判断一个优化问题是否为最小二乘：只需要检验目标函数是否为二次函数（然后检验此二次函数是否半正定）（Q1.4：怎么说）

加权最小二乘：$\sum_{i=1}^{k}w_i(a_i^Tx-b_i)^2$，其中$w_1,...,w_k$均大于0。（Q1.5：$w_i$是常数吗？）

正则化最小二乘：$\sum_{i=1}^{k}(a_i^Tx-b_i)^2+\rho\sum_{i=1}^{n}x_i^2，(\rho >0)$是求解最小二乘的另一种技术。（Q1.6：怎么推）

线性规划：线性规划（其目标函数和约束函数皆为线性函数），可以表述如下：
$$
\text{minimize} \quad c^Tx \\
\text{subject to} \quad a_i^Tx \leq b_i, i=1,...,m
\tag{1.5}
\label{eq1.5}
$$
其中，向量$c,a_1,...,a_m \in \mathbb{R}^n$，$b_i \in \mathbb{R}$皆为问题参数。

## 1.3 凸优化

凸优化：
$$
\text{minimize} \quad f_0(x) \\
\text{subject to} \quad f_i(x) \leq b_i, \quad i = 1,...,m
\tag{1.8}
\label{eq1.8}
$$
其中函数$f_0,f_1,...,f_m : \mathbb{R}^n \to \mathbb{R}$皆为凸函数。即为对于任意的$x \in \mathbb{R}^n$，$y \in \mathbb{R}^n$，以及任意的$\alpha \in \mathbb{R}$, $\beta \in \mathbb{R}$ 且满足$\alpha + \beta = 1$ ，$\alpha \geq 0$，$\beta \geq 0$，下列不等式成立：
$$
f_i(\alpha x + \beta y) \leq \alpha f_i(x) + \beta f_i(y)
$$
凸优化问题的解并没有一个解析表达式。

## 1.4 非线性规划

非线性规划：描述这样一类问题，目标函数**或者**约束函数是非线性函数，**且不一定是凸函数**。对于一般的非线性规划问题式$\eqref{eq1.1}$，目前还没有有效解，一般是在对其放松某些指标的条件下，采取不同的途径进行求解。

局部优化中利用凸优化进行初始值的选取；非凸优化中的凸启发式算法；

# 疑问

1. 公式$\eqref{eq1.1}$的优化解法是否可以泛化到其余约束条件下的优化问题，譬如公式$\eqref{eq1.1}$中的约束函数为等式，是否依旧可以照搬公式$\eqref{eq1.1}$的解法？