---
title: Validation Set
date: 2020-04-13 -0400
categories: [machine-learning, theory]
tag: [machine-learning]
---



It is very common that we have a validation set on top of
a training set in machine learning. Assume that a dataset
\\(\mathcal{D}\\)  of size \\(N\\) is split into
* a training set \\(\mathcal{D}_t\\) of size \\(N-K\\),
* a validation set \\(\mathcal{D}_v\\) of size \\(K\\)


<h2> Model Selection </h2>
Vailication set is usually used to select proper models. Assume that we divide the original hypothesis set \\(\mathcal{H}\\) into subsets \\(\mathcal{H}_1\\),..., \\(\mathcal{H}_M\\). The division may be based on different <strong>hyper-parameters</strong>. We select one candidate hypothesis \\(g_m^{-}\\) from hypothesis set \\(\mathcal{H}_m\\) based on the training dataset \\(\mathcal{D}_t\\). Then we select the final hypothesis \\(g\_{m^\*}^{-}\\) (or at least, hyper-parameters) from hypothesis set \\(\{g_1^{-}, ..., g_M^{-}\}\\).
![Validation_illustration]({{ "/assets/img/ds/valication.png" | relative_url }})


Based on the VC dimesion, we have

\\[
E_{out}(g_{m^\*}) \leq E_{out}(g_{m^\*}^-) \leq E_{out}(g_{m^\*}^-) + O(\sqrt{\frac{\ln M}{K}}),
\\]
where hypothesis \\(g_{m^\*}\\) is trained on the <strong>the whole data set</strong> \\(\mathcal{D}\\) using the hyper-parameters in hypothesis on the <strong>train data set </strong> \\(\mathcal{D}_t\\).


Our goal is to minimze \\(E_{out}(g_{m^\*})\\). Based on the inequality, we need the following two steps:

* If we want \\(E_{out}(g_{m^\*})\\) to be close to \\(E_{out}(g_{m^\*}^-)\\), we need the training set to be similar, i.e, \\(K\\) is small.
* If we want \\(E_{out}(g_{m^\*}^-)\\) to be close to \\(E_{out}(g_{m^\*}^-)\\), we need \\(K\\) to be large, so that the bound is tight.


Therefore, there is a trade of selecting a proper \\(K\\). Rule of thumb is \\(K=N/5\\).


<h2> Cross Validation </h2>

Instead of having a separate validation set, people have proposed of "reusing" the data for validation purpose.

One method is <strong>leave-one-out</strong> method. Assume the orginal data set is \\(\mathcal{D}=\{(\mathbf{x}_1, y_1), ...(\mathbf{x}_N, y_N)\}\\). We remove point \\((\mathbf{x}_n, y_n)\\), and denote by \\(\mathcal{D}_n\\) the collection of remaining points. We train on the dataset \\(\mathcal{D}_n\\), and obtain the hypothesis \\(g_n^-\\). We further do the validation on the point \\((\mathbf{x}_n, y_n)\\), and obtain the error as
\\[
e_n = e(g_n^-(\mathbf{x}_n), y_n).
\\]. This process is iterated from the point \\((\mathbf{x}_1, y_1)\\) to the point \\((\mathbf{x}_N, y_N)\\). The average error termed as <strong> cross valication error</strong> is defined as
\\[
E\_\{cv\}=\frac{1}{N}\sum\_{n=1}^N e_n .
\\]


It can be shown that for each individual error \\(e_n\\), we have

\\[
\mathbf{E}\_{\mathcal{D}}[e\_n] = \bar{E}\_{out}(N-1),         (1)
\\]
where \\( \bar{E}\_{out}(N) = \mathbf{E}\_{\mathcal{D}}[E_{out}(g)]. \\)


* \\( \bar{E}\_{out}(N)\\) is the expected out-of-sample error over the dataset \\(\mathcal{D}\\) of size \\(N\\). Note that \\(\mathcal{D}\\) is the random variable here, and the model is given. Specifically, the model means the mapping from a given dataset to the chosen hypothesis.

* Equation (1) shows that each individual error \\(e_n\\) is an <strong>unbiased estimate </strong> of the expected out-of-sample error trained on a dataset of size \\(N-1\\). Note the expectation is over the dataset of size \\(N-1\\). Hence, \\(E_{cv}\\) is also an <strong> unbiased estimate </strong> of the expected out-of-sample error trained on a dataset of size \\(N-1\\).





{% include mathjax_header.html %}
