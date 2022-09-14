# Neural Machine Translation
#RNN #机器翻译 
[视频链接](https://www.youtube.com/watch?v=gxXJ58LR684&list=PLvOO0btloRnuTUGN4XqO85eKPeFSZsEqK&index=8)
用 sequence-to-sequence Model 把英语翻译成德语

## 准备机器翻译的数据

![[Pasted image 20220816160401.png]]

数据预处理

+ lower case
+ 去掉标点符号

### step1 Tokenization & Build Dictionary

![[Pasted image 20220816164944.png]]

+ 需要使用两个 dictionary
+ tokenization 可以是 char-level，也可以是 word-level

> 本文用 char-level。实际的都用 word-level tokenization ，因为他们的数据集足够大

![[Pasted image 20220816165130.png]]

> 为什么要用两个 tokenizer
![[Pasted image 20220816165231.png]]
![[Pasted image 20220816165259.png]]

并且，不同语言的分词方法也不同

![[Pasted image 20220816165339.png]]

要往德语中加入一个 start sign 和 一个 end sign。不能和已有 token 冲突。

### step 2 one-hot encoding

![[Pasted image 20220816165507.png]]
![[Pasted image 20220816165511.png]]

再进行转化为**输入矩阵**
+ 一个字符用一个向量表示
+ 一句话用一个矩阵表示

![[Pasted image 20220816165626.png]]

> 这样输入数据就准备好了

## Train a Seq2Seq Model

包含一个 encoder，和一个 decoder。encoder 是 LSTM encoder 或 其他RNN encoder，取最后一个状态。

![[Pasted image 20220816165829.png]]

**最后一个 state 包含整个句子的信息（特征）**
+ 最后一个 h
+ 和最后的传输带 C

将随后一个状态，输入到 LSTM Decoder。

![[Pasted image 20220816165933.png]]

Decoder 作为一个**文本生成器**生成德语。

![[Pasted image 20220816171216.png]]

### 如何改进？
#### 1. 使用 Bi-LSTM 作为 encoder
+ encoder 的最后一个状态包含整个英语句子的完整信息
+ 如果句子太长，最后一个状态**会忘记早前的输入**
+ Bi-LSTM（双向LSTM）有更久的记忆
![[Pasted image 20220816171403.png]]
**用双向LSTM替换encoder**

> 但是decoder不能用双向LSTM

#### 2. Word-Level  Tokenization 

+ 减小序列长度
+ LSTM不容易遗忘
![[Pasted image 20220816171625.png]]

**但是**需要更大的数据集
+ one-hot 向量维度过大，大约是 1w
+ 因此，必须用 word-embedding
+ Embedding 层参数非常多很容易导致**overfitting**

#### 3. 多任务学习

![[Pasted image 20220816173216.png]]
这样Encoder的只有一个，但是训练数据却多了很多倍，**可以得到更好的 encoder**。

最终的翻译效果也会更好

#### Attention 机制！**最强的**
 

