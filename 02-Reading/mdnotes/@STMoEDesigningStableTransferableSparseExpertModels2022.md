---
alias: STMoEDesigningStableTransferableSparseExpertModels2022
tags: unread HPO AI MoE sparse
rating: ⭐
share: false
ptype: article
---

# ST-MoE: Designing Stable and Transferable Sparse Expert Models
<cite>* Authors: [[Barret Zoph]], [[Irwan Bello]], [[Sameer Kumar]], [[Nan Du]], [[Yanping Huang]], [[Jeff Dean]], [[Noam Shazeer]], [[William Fedus]]</cite>


* [Local library](zotero://select/items/1_MYJALAHL)

***

### 初读印象

comment:: Stable and transferable [[MoE笔记]] 

### Introduction

系数专家神经网络展现了规模的优势，是静态神经网络架构有效的替代方法。

动态选择每个输入使用哪些参数。这允许网络极大扩展他们的参数数量，同时保证每个 token 的 FLOP 大致不变。

这种方法产生了 SOTA 翻译模型，并且有 4-7 x 的 speed-up。对于 GPU 。

**存在的问题**

1.6T 参数 MoE 模型预训练相比之前的 SOTA 有 4x 的是speed-up，但是在 SuperGLUE 等常用基准上进行微调时，落后于较小的模型。

存在训练不稳定性的影响。

![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/26/555ea82cd0b5f4d869995f4efd81f275-20221026212502-646a72.png)
![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/26/2c7b476fbbbdc9721a651a85f0946f6e-20221026212507-7a4b83.png)

有几种提高稳定度的方法 
1. Remove muliplicative interactions
2. Inject model noise
3. Constrain activations, and gradients

最后，我们提出了我们的建议:一种新的辅助损耗，路由器z损耗，它显著提高了训练的稳定性，而且没有质量下降。这是对Mesh Tensorflow代码库中用于最终softmax logits的z-loss的调整(Shazeer等人，2018)

>Mesh-tensorflow: Deep  learning for supercomputers. In Advances in Neural Information Processing Systems, pages  10414–10423, 2018.

稳定稀疏模型总结 
1. 有很多方法，但是他们都会影响模型性能 
2. 使用 router z-loss 可以稳定模型并且不损害性能
3. Transformer modifications with more multiplicative components (GEGLU, RMS normalization) worsen stability, but boost quality.







