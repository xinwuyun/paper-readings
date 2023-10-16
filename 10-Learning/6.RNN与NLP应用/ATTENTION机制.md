#attention
# Attention for Seq2Seq model
注意力机制
+ Attention 能显著提高性能
+ model 不会忘记输入
+ Decoder 会知道应该关注那个状态
+ 但是计算量太大

## SimpleRNN + Attention 

![[Pasted image 20220816181431.png]]

需要计算每个 $h_i$ 与 $s_0$ 的相关性(权重)

$$
\text { Weight: } \quad \alpha_{i}=\operatorname{align}\left(\mathbf{h}_{i}, \mathbf{s}_{0}\right)

$$
$\sum_{i=1}^{m}{\alpha_i}=1$

### 计算权重的方法

##### 方法1：（原论文方法）

![[Pasted image 20220816181839.png]]
> 矩阵$v$和$w$都是要训练的参数 

 #### 方法2：更流行，且和 transformer 相同
![[Pasted image 20220816182322.png]]

> $W_k$ 和 $W_Q$ 通过训练优化

### 计算Context Vector

![[Pasted image 20220816191744.png]]

### 如何更新 $S$

![[Pasted image 20220816191611.png]]


$$
\mathbf{s}_{1}=\tanh \left(\mathbf{A}^{\prime} \cdot\left[\begin{array}{l}
\mathbf{x}_{1}^{\prime} \\
\mathbf{s}_{0} \\
\mathbf{C}_{0}
\end{array}\right]+\mathbf{b}\right)
$$
> 回忆一下 $c_0$  是encoder 中所有 h 的加权平均 

这里使用 $s_1$ 重新计算 $\alpha$，计算新的 $c_1$

![[Pasted image 20220816192400.png]]

![[Pasted image 20220816192902.png]]


Attention 提高了准确率，代价是，巨大的计算
![[Pasted image 20220816193815.png]]

![[Pasted image 20220816193957.png]]

用这个例子，说明权重的实际意义。这个例子是英语到法语的翻译。
+ 下面是英语，上面是法语。
+ 中间连接线是 encoder 的 s 和下面各个 token 的权重关系。线越粗表示权重越大，也就是**相关性**越大。

可以看到，中间的 zone-Area 有较大相关性，实际上法语的 zone ，就是英语的 Area。

也就是权重 $\alpha$ 会告诉 decoder 应该关注 encoder 的哪个状态。这就是 Attention 的含义

## Summary

+ 一般的 Seq2Seq 会出现遗忘 
+ Attention: decoder 会考察前面的所有状态
+ Attention: decoder 会知道应该 focus 哪个状态 
+ <font color=#ED7001>缺点</font>：较高时间复杂度

![[Pasted image 20220816194628.png]]
