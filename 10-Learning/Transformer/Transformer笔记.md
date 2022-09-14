# Transformer
#transformer #attention #NLP
> 前置知识 [[ATTENTION机制]] 
+ 是一个 Seq2Seq model
+ 不是 RNN 模型，没有循环结构
+ 只基于 Attention 和 Dense 层（全连接层）
+ 性能超高，完爆所有RNN 
+ 现在工业界没人用 RNN 了，都用 Transformer 或 BERT

## Attention for Seq2Seq 
![[Pasted image 20220816225544.png]]
![[Pasted image 20220816225747.png]]
## Attention without RNN 

我们需要设计一个 Attention 层
![[Pasted image 20220816231904.png]]
Decoder 的输入是依次生成的。思路和前面的翻译一致 
#### Key 和 Value

![[Pasted image 20220816232022.png]]
![[Pasted image 20220816232041.png]]
#### Query

![[Pasted image 20220816232101.png]]
![[Pasted image 20220816232122.png]]

> 注意，一共有三个**权重向量**

#### 计算 $s_1$

##### step1 计算权重 $\alpha_1$

计算 $q_1$ 对应到 所有 $k$ 的权重
$$
\boldsymbol{\alpha}_{: 1}=\operatorname{Softmax}\left(\mathbf{K}^{T} \mathbf{q}_{: 1}\right) \in \mathbb{R}^{m}
$$

![[Pasted image 20220816232218.png]]

##### step2 计算Context Vector $c_1$

注意，这里权重乘的是 **value**
$$
\mathbf{c}_{: 1}=\alpha_{11} \mathbf{v}_{: 1}+\cdots+\alpha_{m 1} \mathbf{v}_{: m}=\mathbf{V} \boldsymbol{\alpha}_{: 1}
$$

##### step3 获得下一个输入

![[Pasted image 20220816231809.png]]

##### step 4 重复前面的，获得输出矩阵 

![[Pasted image 20220816232702.png]]

注意：
$$
\mathbf{c}_{: j}=\mathbf{V} \cdot \operatorname{Softmax}\left({\mathbf{K}^{T} {q_{:j}}}\right)
$$
计算一个 c 的值需要
+ 整个 V，包含整个句子的信息
+ 整个 K，包含整个句子的信息
+ 和一个 q，当前输入的德语单词
并且由此可以得到下一个德语单词输入 

> 显然，可以用 Attention layer 代替 [[RNN笔记]] 

## Attention Layer 

![[Pasted image 20220816233034.png]]

## Self-Attention Layer (without RNN)

> 原理完全一样，用 Self-Attention 代替 RNN

+ 不是 Seq2Seq 
+ $C=Attn(X,X)$
> 注意这里和上面 Attention Lay 的对应关系

![[Pasted image 20220816233403.png]]

### step1 计算 Query Key Value

![[Pasted image 20220816233535.png]]![[Pasted image 20220816233558.png]]
### step2 计算权重 

![[Pasted image 20220816233620.png]]
依次计算出所有权重

### step3 计算出所有Context Vector 

![[Pasted image 20220816233747.png]]

## 总结

+ Attention 最初用于 **Seq2Seq RNN model**
+ Self-attention: attention for **all the RNN model** 
+ Attention without RNN 
+ 我们学习了如何构建 Attention Layer 和 Self-Attention Layer

Reference:
1. Bahdanau, Cho, \& Bengio. Neural machine translation by jointly learning to align and translate. In ICLR, $2015 .$
2. Cheng, Dong, \& Lapata. Long Short-Term Memory-Networks for Machine Reading. In EMNLP, $2016 .$
3. Vaswani et al. Attention Is All You Need. In NIPS, $2017 .$

![[Pasted image 20220816234206.png]]
![[Pasted image 20220816234226.png]]

----
# 用 Attention 构建深度神经网络
## 单头和多头层

![[Pasted image 20220816234558.png]]
![[Pasted image 20220816234606.png]]
![[Pasted image 20220816235358.png]]

## 用这两种层搭建深度深度神经网络

### 搭建 encoder 网络

![[Pasted image 20220817005708.png]]

记住每个全连接层完全相同，有同一个参数矩阵 $W_u$

![[Pasted image 20220817005741.png]]

$u_i$ 与所有 x 都有关

![[Pasted image 20220817005812.png#center|One Block of Transformer]]
![[Pasted image 20220817005916.png#center|Transformer encoder]]
+ 有 6 个 block
+ 每个 block 有两层
+ block 之间不共享参数

### 搭建 Decoder 网络 

![[Pasted image 20220817010040.png]]
![[Pasted image 20220817010055.png]]

encoder 输入和输出都是 512×m 
![[Pasted image 20220817010144.png]]

![[Pasted image 20220817010159.png]]

![[Pasted image 20220817010341.png]]

### Transformer 

![[Pasted image 20220817010517.png]]

> 对比一下RNN
> ![[Pasted image 20220817010640.png]]
> 可以看到，RNN 的输入输出和 transformer 一样，两者使用方法一致 

## 总结 
![[Pasted image 20220817010730.png]]
![[Pasted image 20220817010744.png]]

![[Pasted image 20220817010801.png]]

### 总结 Transformer 模型

+ 是 Seq2Seq model，包含一个 encoder 和 decoder
+ 不是 RNN
+ 基于 Attention 和 [[Self-Attention]] 
+ 精度超高
