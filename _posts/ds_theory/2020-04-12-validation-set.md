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


{% include mathjax_header.html %}
