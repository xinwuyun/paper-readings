---
alias: FeatureLearningInfiniteWidthNeuralNetworks2022
tags: unread
rating: ⭐
share: false
ptype: article
---

# Feature Learning in Infinite-Width Neural Networks
<cite>* Authors: [[Greg Yang]], [[Edward J. Hu]]</cite>


* [Local library](zotero://select/items/1_JHUULM33)

***

### 初读印象

comment:: 

当深度神经网络的宽度趋于无穷大时，如果适当地参数化(例如NTK参数化)，它在梯度下降下的行为可以**简化和可预测**(例如由神经切线核(NTK)给出)。
然而，我们表明神经网络的标准和NTK参数化不允许学习特征的无限宽度限制，而这对于预训练和迁移学习(如BERT)是至关重要的。However, we show that the standard and NTK parametrizations of a neural network  do not admit infinite-width limits that can learn features, which is crucial for pre-training and transfer learning such as with BERT. 

我们建议对标准参数化进行简单的修改，以允许在极限范围内进行特征学习。利用张量程序技术，我们推导出这些极限的显式公式。在Word2Vec和Omniglot上通过MAML进行的 few-shot 学习(这两个典型的任务非常依赖特征学习)上，我们精确地计算出了这些极限。我们发现它们优于NTK基线和有限宽度网络，后者随着宽度的增加接近无限宽度的特征学习性能。

更一般地，我们分类神经网络参数化的自然空间，它概括了标准、NTK和平均场参数化。  我们证明了1)该空间中的任何参数化要么允许特征学习，要么具有核梯度下降给出的无限宽度训练动态，但不能两者兼有; 2)任何这样的无限宽度极限都可以用张量程序技术计算出来

