# Self-Attention 

- Self-Attention [2]: attention [1] beyond Seq2Seq models
- 原始论文用了LSTM
- 本文为了简单起见，用 Simple-RNN
>原始论文
>1. Bahdanau, Cho, \& Bengio. Neural machine translation by jointly learning to align and translate. In ICLR, $2015 .$
>2. Cheng, Dong, \& Lapata. Long Short-Term Memory-Networks for Machine Reading. In EMNLP, 2016.

## SimpleRNN + Self-Attention 

![[Pasted image 20220816204247.png]]

> 注意，A 后面向量不包含 h。

![[Pasted image 20220816204404.png]]

### 计算 C
$$
\text { Weights: } \quad \alpha_{i}=\operatorname{align}\left(\mathbf{h}_{i}, \mathbf{h}_{2}\right) \text {. }
$$
再对已有的 h 加权得到 C

![[Pasted image 20220816204602.png]]

以此类推

## 总结

![[Pasted image 20220816205203.png]]
