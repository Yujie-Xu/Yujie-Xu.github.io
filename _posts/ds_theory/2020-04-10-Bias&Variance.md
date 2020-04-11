---
title: Bias & Variance in Machine Learning
date: 2020-04-10 -0400
categories: [machine-learning, theory]
tag: [machine-learning]
mathjax:  False
---

{% include mathjax.html %}

In this article, we focus on studying the relationship between out-of-sample error, bias and varaince.

We assume that the target function is a deterministic function $f(\mathbf{x})$. We randomly draw a dataset $\mathcal{D}$ from input space $\mathcal{X}$.

Given a dataset $\mathcal{D}$, we have the final chosen hypothesis denoted by $g^{(\mathcal{D})}(\mathbf{x})$. Then out-of-sample is given by


$$
E_{\text{out}}(g^{\mathcal{D}})=\mathbf{E}_{\mathbf{x}}[(g^{\mathcal{D}}(\mathbf{x})-f(\mathbf{x}))^2].
$$

The out-of-sample error $E_{\text{out}}(g^{\mathcal{D}})$ depends on the selection of the dataset $\mathcal{D}$. We further get the following <strong>expected</strong> out-of-sample error as


$$
\mathbf{E}_{\mathcal{D}}[E_{\text{out}}(g^{\mathcal{D}})] = \mathbf{E}_{\mathcal{D}}[\mathbf{E}_{\mathbf{x}}[(g^{\mathcal{D}}(\mathbf{x})-f(\mathbf{x}))^2]].
$$


First, let us define


 $$\bar{g}(\mathbf{x})=\mathbf{E}_{\mathcal{D}}[g^{\mathcal{D}}(\mathbf{x})].$$

After some math manipulations, we can show that

$$
\mathbf{E}_{\mathcal{D}}[E_{\text{out}}(g^{\mathcal{D}})] = \mathbf{E}_{\mathbf{x}}[\text{bias}(\mathbf{x})
+\text{var}(\mathbf{x})],
$$
where



$$\text{bias}(\mathbf{x}) = (\bar{g}(\mathbf{x})-f(\mathbf{x}))^2,$$

and

$$\text{var}(\mathbf{x}) = \mathbf{E}_{\mathcal{D}}[(g^{\mathcal{D}}(\mathbf{x})-\bar{g}(\mathbf{x}))^2].$$


To futher study the bias and variance, we will further discuss $\bar{g}(\mathbf{x})$, $\text{bias}(\mathbf{x})$, and $\text{var}(\mathbf{x})$.

* $g^{\mathcal{D}}(\mathbf{x})$ is the chosen hypothesis given $\mathcal{D}$.
* $\bar{g}(\mathbf{x})$ is the <strong>mean</strong> of the hypothesises over different <strong>realizations</strong> of $\mathcal{D}$.
* $f(\mathbf{x})$ is the target function.
* $\text{bias}(\mathbf{x})$ is the bias w.r.t <strong>the dataset $\mathcal{D}$ </strong> given any points $\mathbf{x}$ in input space $\mathcal{X}$.
* $\text{var}(\mathbf{x})$ is the variance w.r.t <strong>the dataset $\mathcal{D}$ </strong> given any points $\mathbf{x}$ in input space $\mathcal{X}$.
