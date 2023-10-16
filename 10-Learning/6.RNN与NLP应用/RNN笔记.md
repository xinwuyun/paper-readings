### Word Embedding

### step 1: one-hot encoding

![[Pasted image 20220816103053.png]]

### step 2 : word embedding

![[Pasted image 20220816103117.png]]

xi就是一个**词向量**，d 的大小会影响模型的表现。

这里 d 是二维的。可以映射到坐标系中。

![[Pasted image 20220816103838.png]]

训练完成后，应当形如右面的坐标系。

### 代码

![[Pasted image 20220816104137.png]]
...
![[Pasted image 20220816123949.png]]

### 总结
![[Pasted image 20220816124417.png]]

## Simple RNN

> 目前RNN已经过时了，如果训练数据足够多，不如transformer

### 如何对时序数据建模

> 全连接神经网络和卷积神经网络都是 one-to-one 模型。要求一个输入对应一个输出，很适合图片问题，不适合文本问题，因为输入输出不固定

对于时序数据，更好的模型是 many-to-one，比如 RNN

![[Pasted image 20220816124933.png]]

### Recurrent Neural Networks

![[Pasted image 20220816125005.png]]

### Simple RNN Model

![[Pasted image 20220816125415.png]]

> 为什么用 tanh 激活函数？

![[Pasted image 20220816130841.png]]

参数数量是多少？
![[Pasted image 20220816131052.png]]

### Simple RNN 解决电影评价分类

![[Pasted image 20220816131300.png]]
+ 词向量维度自己设置，应当通过交叉验证决定。
+ h维度同上
	+ h 和 x 的维度通常不同，这里是巧合 
$h_t$ 涵盖了前 t 个词的信息。最后一个 h 涵盖了前面所有x 值的信息。
+ 只提取 $h_t$

![[Pasted image 20220816131528.png]]

![[Pasted image 20220816131722.png]]
![[Pasted image 20220816131734.png]]
![[Pasted image 20220816131844.png]]

也可以在 rnn 层返回所有 h，再加一个 flatten 层
  ![[Pasted image 20220816131950.png]]
![[Pasted image 20220816132028.png]]

### 缺陷

![[Pasted image 20220816132304.png]]这个例子，Simple RNN 很擅长

但是，Simple RNN 不擅长 **long-term dependence**
![[Pasted image 20220816132329.png]]

如果改变 $x_1$， $h_{100}$几乎不会变。

**举个🌰**

![[Pasted image 20220816132437.png]]

Simple RNN 已经把前文忘记了，所以很难预测出来。

**Summary**

![[Pasted image 20220816132552.png]]

## LSTM（Long Short term Memory）

> 是对Simple RNN 的改进![[Pasted image 20220816132710.png]]

![[Pasted image 20220816132751.png]]

### 内部结构
#### 传输带 Conveyor Gate

过去的信息通过传输带传输到未来。
![[Pasted image 20220816135530.png]]
> 可以避免梯度消失问题 

#### 遗忘门 Forget Gate

![[Pasted image 20220816135558.png]]

![[Pasted image 20220816135735.png]]

遗忘门 f 有选择得让元素通过。

![[Pasted image 20220816135800.png]]
$w_f$是要训练的参数

#### 输入门 Input Gate 

![[Pasted image 20220816140023.png]]

#### New Value

![[Pasted image 20220816140056.png]]

#### Update the Conveyor Belt

![[Pasted image 20220816140132.png]]
> 如果 $f_t$ 中的某个元素为 0，则 $c_{t-1}$ 中的值就会被遗忘


#### Ouput Gate 

![[Pasted image 20220816140410.png]]
![[Pasted image 20220816140450.png]]

一个 $h_t$ 作为 LSTM 的输出，一个输入到下一个 block

#### 参数量

![[Pasted image 20220816140551.png]]

### 代码 

![[Pasted image 20220816140635.png]]

![[Pasted image 20220816140658.png]]![[Pasted image 20220816140705.png]]![[Pasted image 20220816140716.png]]
![[Pasted image 20220816140852.png]]
![[Pasted image 20220816140826.png]]

### 总结 

![[Pasted image 20220816141035.png]]
## Text Generation 自动文本生成
### RNN for Text Prediction

![[Pasted image 20220816143925.png]]

#### 如何训练？

![[Pasted image 20220816144153.png]]

+ 红色片段作为输入 
+ 蓝色字符作为label
+ Traning data: (segment, next_char) pair

其实就是个**多分类问题**
![[Pasted image 20220816144300.png]]

> 应用：
> **起名字**
> ![[Pasted image 20220816144403.png]]
> 如果用 linux 源码作为输入 
> ![[Pasted image 20220816144432.png]]
> > 当然他无法编译通过
> 
> 如果用 LaTex 源码训练
> ![[Pasted image 20220816144502.png]]

### 训练过程

#### step 1 准备训练数据

![[Pasted image 20220816154251.png]]

得到 （segment, next_char） pair

#### step 2 Char to vector

使用 one-hot encoding 

![[Pasted image 20220816154434.png]]

可以把 segment 转换为矩阵
+ 行数 = 字符数
+ 列数 = 字符个数

> 注意：这里是 character-level。维数比较低，所以不需要进行 word-embedding 

![[Pasted image 20220816154703.png]]

#### step 3 构建网络 

![[Pasted image 20220816154907.png]]

#### step 4 训练 

![[Pasted image 20220816155003.png]]


#### step 5 预测

![[Pasted image 20220816155112.png]]
![[Pasted image 20220816155119.png]]
![[Pasted image 20220816155152.png]]
![[Pasted image 20220816155211.png]]
![[Pasted image 20220816155247.png]]
> 可以让概率更大的值更大，概率更小的值更小
![[Pasted image 20220816155533.png]]

#### 一个例子

种子：输入的字符串

输入的字符串长度在后续的迭代中不变，加一个，去掉后面一个。

![[Pasted image 20220816155801.png]]

##### 训练一个 epoch
![[Pasted image 20220816155818.png]]

同样的种子每次可能生成不同的句子。

##### 20个
![[Pasted image 20220816155930.png]]

基本收敛了，句子恒好了，但是还是不够好。如果用更大的模型，可能能得到更逼真的文本。

### 总结 

**训练**
![[Pasted image 20220816160046.png]]

**文本生成**
![[Pasted image 20220816160155.png]]


