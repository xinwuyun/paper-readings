---
alias: OutrageouslyLargeNeuralNetworksSparselyGatedMixtureofExpertsLayer
tags: MoE NLP 稀疏性 GateNetwork
rating: ⭐
share: false
ptype: article
---

# Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
<cite>* Authors: [[Noam Shazeer]], [[Azalia Mirhoseini]], [[Krzysztof Maziarz]], [[Andy Davis]], [[Quoc Le]], [[Geoffrey Hinton]], [[Jeff Dean]]</cite>


* [Local library](zotero://select/items/1_C67EYBPQ)

***

### 初读印象

comment:: 稀疏gate的 MoE 层，通过条件计算增大模型容量，同时避免大量计算 

### 文章骨架
%%创新点到底是什么?%%
novelty:: 引入了一个**稀疏门控**的 MoE 层，由多达数千个前馈子网络组成，实现了通过条件计算增加机器学习模型参数数量而不成比例增加计算量。

%%有什么意义？%%
significance:: 基于 MoE 层，可以做到增加参数量从而提高模型能力，但不增加计算量。

%%有什么潜力?%% 
potential:: 可以用来构建大型语言模型，在机器翻译基准任务上，可以超过 state-of-the-art，同时计算量更小。
![Pasted image 20220820145954.png](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/08/22/ce427c628feaa645a2c43b4bc4f9cd3c-Pasted%20image%2020220820145954-fbd3e6.png)

这里，MoE层嵌入在堆叠的[[../../10-Learning/6.RNN与NLP应用/RNN笔记|RNN]] 中的。在这种情况下，稀疏门控函数选择两位专家进行计算。它们的输出由门控网络的输出调制。

每个专家是一个简单的前馈神经网络。

> 前馈神经网络（Feedforward Neural Network），图是 DAG，比如 CNN, FCN, GAN
> 反馈神经网络有环，有环，比如 RNN, LSTM, 

不同的专家往往会根据语法和语义高度专业化
评估显示，引入 MoE 层能够以很小的计算成本改进了最佳的发布结果。

## 结构

我们用G(x)（gate）和$E_i$(x)（Expert）表示对于给定的输入x，门控网络的输出和第i个专家网络的输出。MoE模块的输出y可以写成这样

$$
y=\sum_{i=1}^{n} G(x)_{i} E_{i}(x)
$$
$G(x)$输出具有稀疏性，节省了计算。$G(x)_{i}=0$时对应的$E_i(x)$就不用计算。
### Gating Network

有两种形式

1. Softmax Gated: $G_{\sigma}(x)=\operatorname{Softmax}\left(x \cdot W_{g}\right)$, $W_g$ 是权重矩阵，可训练
2. Noisy Top-K Gating: 在 softmax 中加入**稀疏性**和噪音。softmax之前加入可调高斯噪声，保留top-k。其余设置为0
$$
\begin{gathered}
G(x)=\operatorname{Softmax}(\operatorname{KeepTop} K(H(x), k)) \\\\
H(x)_{i}=\left(x \cdot W_{g}\right)_{i}+\operatorname{StandardNormal}() \cdot \operatorname{Softplus}\left(\left(x \cdot W_{\text {noise }}\right)_{i}\right) \\\\
\text { KeepTopK }(v, k)_{i}= \begin{cases}v_{i} & \text { if } v_{i} \text { is in the top } k \text { elements of } v \\\\
-\infty & \text { otherwise. }\end{cases}
\end{gathered}
$$
$W_{noise}$ 可训练
 ### Gating Network 的训练

  反向传播。和模型的其他部分一起训练。

## 解决性能挑战

### 1. Shrinking Batch Problem

在现代的cpu和gpu上，为了提高计算效率，需要更大的batch size，以分摊参数加载和更新的开销。

对于大小变为 b 的 batch，假设对于每个样本取 k 个专家，每个专家会得到大概 $\frac{kb}{n}{\ll}b$，随着专家增多，一般的 MoE 实现会比较低效。

解决方法是尽可能让 batch size  增大。但是这对内存需求太大，需要以下技术增加批处理大小

+ 混用数据并行和模型并行
+ 利用卷积：在语言模型中（RNN），对前一层的每个time step应用相同的MoE。如果我们等待上一层完成，我们可以将MoE应用到所有的时间步骤一起作为一个大批。这样做将增加到MoE层的输入批处理的大小，其大小是展开时间步数的一个因数。
+ Recurrent MoE，把MoE嵌入到 RNN 层里。例如，一个 LSTM 或其他 RNN 的权矩阵用一个 MoE 来代替。Gruslys et al. (2016) describe a technique for drastically reducing the number of stored  activations in an unrolled RNN, at the cost of recomputing forward activations. This would allow  for a large increase in batch size.

### 网络带宽

通信会影响效率。为了维持计算效率，一个专家的计算量与输入输出量的比率必须超过计算设备的计算量与网络容量的比率。

方便的是，我们可以通过使用更大的隐藏层或更多的隐藏层来提高计算效率。

> 具体细节见论文吧

## 平衡专家利用率

实验表明，门网络趋于收敛时，有几个专家很受欢迎，给他们的权重总是很大。这种不平衡是**自我强化的**，因为受青睐的专家被训练得更快，因此被门控网络选择得更多。

> 马太效应？

>已经有两个工作研究了该问题。  
Eigen et al. (2013)  describe the same phenomenon, and use a hard constraint at the beginning of training to avoid this  local minimum. Bengio et al. (2015) include a soft constraint on the batch-wise average of each gate

作者使用软约束方法。给每个专家定义了一个重要性，每个专家的重要性等于在一组样本上每次分配的权重和，并为重要性了一个 loss 函数，加到了总体的 loss 函数中。

这个 loss 函数等于重要性的变异系数的平方，乘一个超参数$w_{importance}$。所有专家重要性相等时 loss 最小。
$$
\begin{gathered}
\operatorname{Importance}(X)=\sum_{x \in X} G(x) \\
L_{\text {importance }}(X)=w_{\text {importance }} \cdot C V(\text { Importance }(X))^{2}
\end{gathered}
$$

虽然这种损失函数可以确保同等的重要性，但专家仍然可能收到不同数量的例子。例如，一个专家可能收到一些权重较大的示例，而另一个专家可能收到许多权重较小的示例。这可能会导致分布式硬件上的内存和性能问题。

为了避免这个问题，引入另一个 Loss 函数。

## 结论

+ 这项工作首次证明了条件计算在深度网络中的主要优势。
+ 作者也提出了，条件计算的设计考虑因素和挑战，也提出了算法和工程上的解决方案。
+ 作者主要关注了NLP，条件计算也可以应用到其他领域
