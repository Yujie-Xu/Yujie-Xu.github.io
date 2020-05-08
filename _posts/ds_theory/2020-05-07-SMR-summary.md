---
title: SVM Algorithms Summary
date: 2020-05-07 -0400
categories: [machine-learning, algorithm]
tag: [machine-learning]
mathjax:  True
---



{% include mathjax.html %}

<h1> Linear SVM </h1>

Given a dataset $\mathcal{D}=\\{(\mathbf{x}_1, y_1),..., (\mathbf{x}_N, y_N) \\}$, where $y_i \in \\{1, -1\\}$, we want to find a hyperplane $\mathbf{w}^T\mathbf{x}+b=0$ to classify the points based on the value of $y_i$ in the dataset $\mathcal{D}$. We use $h=(\mathbf{w},b)$ to represent the hyperplane.

<h2> Hard-Margin SVM </h2>
In this case, we assume that the points in $\mathcal{D}$ are separable. Formally, there exist a hyperplane that can separate the points.

There are two equivalent definitions of separate hyperplane.

__Def. 1 (Separating Hyperplane)__: The hyperlane separates the data in $\mathcal{D}$ if and only if for $n=\{1,...,N\}$,
$$y_i(\mathbf{w}^T\mathbf{x}_i+b) >0.$$


__Def. 2 (Separating Hyperplane)__: The hyperlane separates the data in $\mathcal{D}$ if and only if it can be represented by parameters that satisfy

$$\min_i y_i(\mathbf{w}^T\mathbf{x}_i+b) = 1.$$

Note: Note that hyperplanes $\mathbf{w}^T\mathbf{x}+b=0$ and $\frac{\mathbf{w}^T\mathbf{x}}{\rho}+\frac{b}{\rho}=0$ are equivalent. In Def. 2, all the weights are normarlized by $\rho = \min_i y_i(\mathbf{w}^T\mathbf{x}_i+b).$


__Def. 3 (Distance between a point and a hyperpalce)__  The distance between a point $\mathbf{x}$ and a hyperplane $h$ is defined by the distance between the point and the closest point in the hyperplane, i.e.,

$$\min_{\mathbf{x}^{'}} \|\mathbf{x}^{'}-\mathbf{x}\|, ~~ \text{s.t.} ~~ \mathbf{w}^T\mathbf{x}^{'}+b=0.$$

Solving the optimization problem above, we have the distance denoted by $dis(\mathbf{x},h)$

$$dis(\mathbf{x},h)=\frac{|\mathbf{w}^T\mathbf{x}+b|}{\|\mathbf{w}\|}.$$


The minimum distance among all the points in $\mathcal{D}$ to the hyperplane $h$, denoted by $d^{*}$ is


$$d^*=\min_n dis(\mathbf{x}_n,h)=\min_n\frac{|\mathbf{w}^T\mathbf{x}_n+b|}{\|\mathbf{w}\|}  \overset{(a)}= \frac{1}{\|\mathbf{w}\|},$$


where $(a)$ is based on the Def 2, and the fact that $y_i\in \\{1, -1\\}.$

Hence, the optimization problem to find the fattest separating hyperplace is

$$
\max _{\mathbf{w}, b} \frac{1}{\|\mathbf{w}\|} \quad \text{s.t.} \quad  \min_i y_i(\mathbf{w}^T\mathbf{x}_i+b) = 1.
$$  


Equivalently, the optimization problem is


$$
\text{P}_1: ~~ \min _{\mathbf{w}, b} \frac{1}{2}\mathbf{w}^T\mathbf{w} \quad \text{s.t.} \quad  \min_i y_i(\mathbf{w}^T\mathbf{x}_i+b) = 1.
$$

Furthermore, we can show that the optimization problem can be further simplified as

$$
\text{P}_2:  ~~ \min _{\mathbf{w}, b} \frac{1}{2}\mathbf{w}^T\mathbf{w} \quad \text{s.t.} \quad   y_i(\mathbf{w}^T\mathbf{x}_i+b) \geq 1, ~~ \forall ~ n.
$$

To show the equivalence between $\text{P}_1$ and $\text{P}_2$, we have the following argument. Assume that $(\mathbf{w}_i^{\*}, b\_i)$ is the optimal solution to problem $\text{P}_i$, and the optimal objective is $o_i^{\*}=\frac{1}{2}{\mathbf{w}_i^{\*}}^T\mathbf{w}_i^{\*}$.

First, we claim that $o_1^{\*} \geq o_2^{\*}$ since the feasible set in problem $\text{P}_2$ is larger.

Then we claim that  $(\mathbf{w}_2^{\*}, b_2)$ will satisfy the constraint in problem  $\text{P}_2$, and hence $(\mathbf{w}_2^{\*}, b_2)$ is a feasbile point in problem $\text{P}_1$. Assume that for every $n$, we have $y_i({\mathbf{w}_2^\*}^T\mathbf{x}_i+b) > 1.$ Let $\rho=\min_n y_i({\mathbf{w}_2^\*}^T\mathbf{x}_i+b)$. Then $\rho>1$. Then the solution $(\frac{\mathbf{w}_2^{\*}}{\rho}, \frac{b_2}{\rho})$ is still a feasible soltion to problem $\text{P}_2$, and achieves a smaller objective. Hence, there must exist i, such that $y_i({\mathbf{w}_2^\*}^T\mathbf{x}_i+b) = 1.$ Thus solution $(\mathbf{w}_2^{\*}, b_2)$ is also a feasible solution to $\text{P}_1$, Hence, $o_2^{\*} \geq o_1^{\*}$.

Hence, $o_2^{\*} = o_1^{\*}$, and achive the optimal at the same point(s).


<h2> Soft-Margin SVM </h2>
Hard-Margin SVM is based on the assumption that the points in the dataset is separable. This is a very strict assumption. The soft-margin svm introduces _soft-margin svm to relax the constraint that the points are separable. Mathmatically, the problem is formulated as:

$$
\text{P}_3:  ~~ \min _{\mathbf{w}, b, \xi_i} \frac{1}{2}\mathbf{w}^T\mathbf{w} + C\sum_n \xi_i \\ ~~~~ \text{s.t.} \quad   y_i(\mathbf{w}^T\mathbf{x}_i+b) \geq 1-\xi_i, ~~ \forall ~ n, \\
\xi_i \geq 0, ~~\forall ~n.
$$


The introduction of $\xi_i$ guarantees that there exist feasible solutions to problem $\text{P}_3$. If $\xi_i$ approaches $\infty$, the first constraint is always satified. Thus, the feasible set is guranteed to be non-empty.


<h1> Dual Problem </h1>


Based on the standard Lagrangian analysis, the dual problem of $P_3$ is given by


$$
\text{P}_4:  ~~ \min_{\mathbf{\alpha}} \frac{1}{2}\mathbf{\alpha}^T \mathbf{\alpha} - \mathbf{1}^T\mathbf{\alpha} \\
\text{s.t.} ~~ \mathbf{y}^T\mathbf{\alpha} =0; \\
\mathbf{0} \leq \mathbf{\alpha} \leq C * \mathbf{1}.
$$

Let $\alpha^\*, \mathbf{w}^\*, b^\*, \xi_i^\*$ the optimal solution to the dual problme and the original problem. Part of the KKT conditions are

$$
\alpha_n^* (y_n({\mathbf{w}^*}^T\mathbf{x}_n+b^*)-1+\xi_n^*) = 0 \\
(C-\alpha_n^*)\xi_n^* = 0
$$

Based on these two equalities, we have the three cases:

(1) if $\alpha_n^* = 0$, then we have $y_n({\mathbf{w}^\*}^T\mathbf{x}_n, b^\*)\leq 1$. Then point $(\mathbf{x}_n, y_n)$ is in the right subspace cut by the corresponding fat-hyperplane.

(2) if $0< \alpha_n^* < C$, then we have $y_n({\mathbf{w}^\*}^T\mathbf{x}_n, b^\*)=1$. Then point $(\mathbf{x}_n, y_n)$ is on the fat hyperplane.

(3) if $\alpha_n^* = C$, then we have $y_n({\mathbf{w}^\*}^T\mathbf{x}_n, b^\*) \leq 1$. Then point $(\mathbf{x}_n, y_n)$ is in the wrong subspace cut by fat hyperplane.
