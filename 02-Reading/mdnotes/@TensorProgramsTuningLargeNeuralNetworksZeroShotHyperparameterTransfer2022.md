---
alias: TensorProgramsTuningLargeNeuralNetworksZeroShotHyperparameterTransfer2022
tags: unread HPO
rating: ⭐
share: false
ptype: article
---

# Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer
<cite>* Authors: [[Greg Yang]], [[Edward J. Hu]], [[Igor Babuschkin]], [[Szymon Sidor]], [[Xiaodong Liu]], [[David Farhi]], [[Nick Ryder]], [[Jakub Pachocki]], [[Weizhu Chen]], [[Jianfeng Gao]]</cite>


* [Local library](zotero://select/items/1_DUUVT4RM)

***

### 初读印象

comment:: 
### 文章骨架
%%创新点到底是什么?%%
novelty:: 提出了 μTransfer

%%有什么意义？%%
significance:: 可以将(任意)大模型的调优问题简化为(固定大小的)小模型的调优问题。

%%有什么潜力?%% 
potential:: 

将(任意)大模型的调优问题简化为(固定大小的)小模型的调优问题。我们的整个过程，我们称之为μTransfer，在算法1和图2中得到了总结，我们覆盖的hp在表1和表2中得到了总结

方法的好处

1. Better performance：μTransfer不仅仅是预测 SP（Standard Parameterization） 中最优学习率是如何扩展的。还能比 SP 收敛的模型性能更好（training loss 更低）。
![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/22/488b798a05fe9f849f8511020afd5b3b-20221022122108-728396.png)

3. Speedup
4. Tune once for whole family
5. Better compute utilization


### 改进

1. 跨深度 transfer 不是很好，跨深度 transfer 不适用于 post-laynorm [[Transformer笔记]] 
2. 对于较小的模型，最优 HP 仍有一些变化
3. Finally, it will be interesting to study if there s a way to transfer regularization HPs as a function of both the model size and data size, especially in the context of finetuning of pretrained models

### µTransfer 的优势

1. Better performance : 相同超参数，*µ*Transer 性能更好
2. Speedup: 
3. Tune once for whole family:
4. Better compute utilization:
5. Painless transition from exploration to scaling up:

本文主要关注与 traning loss 相关的 HP transfer。

#TODO 什么是 per-layer HP

![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/27/46fb8b02769cd5ba3ed6c0b9503682fc-20221027125708-74282c.png)

贡献
1. We demonstrate it is possible to zero-shot transfer near optimal HPs to a large model from a  small version via the Maximal Update Parametrization (μP) from [57] .
2. While [57] only covered SGD, here we derive μP for Adam as well (Table 3).
3. We propose a new HP tuning technique, μTransfer, for large neural networks based on this observation that provides massive speedup over conventional methods and covers both SGD  and Adam training;
4. 一个 PyTorch 包

# 翻译

## 摘要

深度学习中的超参数(HP)调优是一个昂贵的过程，对于拥有数十亿个参数的神经网络(nn)来说更是如此。我们表明，在最近发现的最大更新参数化(maximum Update parameter ization， μP)中，即使模型大小发生变化，许多最优hp也保持稳定。这带来了一种新的HP调优范式，我们称之为μTransfer:用μP参数化目标模型，在更小的模型上间接调优HP，并将它们零样本转移到完整的模型上，也就是说，完全不直接调优后者。我们在Transformer和ResNet上验证μTransfer。例如，1)通过从一个13M参数的模型中转移预训练的hp，我们的性能优于已发布的BERT-large (350M参数)的数量，总调优成本相当于预训练BERT-large一次;2)通过40M参数的转移，我们的性能优于6.7B GPT-3模型的公布数据，调优成本仅为总预训练成本的7%。

## Introduction 

超参数(HP)调优是深度学习的关键。选择不当的hp会导致低水平的表现和训练不稳定。由于不同程度的HP调优，许多发布的基线很难相互比较。当训练非常大的深度学习模型时，这些问题会更加严重，因为拥有数十亿参数的最先进的网络变得非常昂贵，难以调优。

最近，[57]研究表明，不同的神经网络参数化可以产生不同的无限宽度极限，并提出了最大更新参数化(缩写为μP)(表3所示)，可以在极限范围内实现最大特征学习。直观地说，它保证了在训练过程中，无论宽度如何，每一层都按相同的 order 更新（也就是说，在较大的宽度限制下，更新对激活的影响基本上与宽度无关）。相比之下，虽然标准参数化(SP)确保激活在初始化时是单位顺序的，但它实际上会导致它们在训练[57]期间在宽模型中爆炸，这主要是由于每层学习率的不平衡(也见图5)。在这项工作中，我们利用μP零样本将hp从小模型转移到大模型，也就是说，我们在大模型上获得了接近最优的hp，而根本不需要直接调整它!尽管从业人员总是从小模型中猜测大模型的hp，但由于参数化不正确，结果最多只能是随机的。例如，如图1所示，在一个Transformer中，最优学习率在μP(右)的宽度下是稳定的，但在标准参数化(左)中则远非如此。除了宽度之外，我们还通过经验验证，在一些注意事项的情况下，hp也可以跨深度(见第6.1节)、批大小、语言模型序列长度和训练时间(见附录G.2.1)进行转移。这将(任意)大模型的调优问题简化为(固定大小的)小模型的调优问题。我们的整个过程，我们称之为μTransfer，在算法1和图2中得到了总结，我们覆盖的hp在表1和表2中得到了总结。

我们的方法有几个好处:**1. 更好的性能**:μTransfer不仅仅是预测最佳学习率如何在SP中扩展。一般来说，我们期望μTransfer模型在优化学习率的情况下优于SP模型。例如，这是图1中宽度为8192的变压器的情况。我们将在第5节和附录C. 2中讨论原因。2. **加速**:它为大型模型的调优提供了巨大的加速。例如，我们仅通过零样本HP转移就能够超越已发布的(350M) BERT-large[11]，而调优成本大约等于1个BERT-large预训练。同样，我们的性能优于发布的6.7B GPT-3模型[7]，调优成本仅为总训练前成本的7%。对于这种规模的模型，如果没有我们的方法，HP调优是完全不可行的。3.**只需为整个家族调优一次**:对于任何宽度和深度不同的固定模型家族(如BERT家族或GPT-3家族)，我们只需要调优单个小模型，并可以为家族中的所有模型重用它的hp例如，我们将使用这种技术，通过从13M模型转移，同时调优BERT-base (110M参数)和BERT-large (350M参数)。4. **更好的计算利用率**:虽然大型模型训练需要分布在多个gpu上，但小型模型调优可以发生在单个gpu上，极大地提高了调优的并行性水平(在组织计算集群的上下文中，还可以提高调度和利用率)。5. **从探索到放大的无痛过渡**:通常，研究人员在小模型上探索新想法，但当放大时，发现他们在探索期间优化的hp在大模型上表现不佳。μTransfer可以解决这个问题。

除了HP稳定性属性外，我们发现在μP的训练中，与SP(第8节)相比，更宽更好。这增加了深度学习中模型缩放的可靠性。

在这项工作中，我们主要关注与训练损失相关的超参数转移。在正则化不是测试性能的瓶颈的情况下，就像在我们这里的所有实验中一样，这也转化为测试损失方面的有效性。在其他设置中，比如微调小数据集上的模型，μTransfer可能是不够的，我们将在第6.1节中讨论

表2:μ可转移超参数的例子。下面的所有参数也都可以特化为每层的超参数。
| 优化器相关的                                                     | 初始化             | 参数乘数 |
| ---------------------------------------------------------------- | ------------------ | -------- |
| 学习速率(LR)，momentum，Adam beta后的每层乘法常数，LR schedule等 | 每一层的初始化方差 | 权重/偏置后的乘法常数，等等         |
我们的贡献 
+ 我们证明了通过从[57]的最大更新参数化(μP)，可以从一个小版本向一个大模型进行接近最优hp的零样本转移。
+ 虽然[57]只涵盖SGD，在这里我们也为Adam推导出μP(表3)
+ 基于这一观察结果，我们提出了一种新的HP调优技术μTransfer，用于大型神经网络，它比传统方法提供了巨大的加速，并涵盖了SGD和Adam训练
+ 我们在机器翻译和大型模型预训练上(见第7.3节)以及图像分类(见附录G.1)对我们的方法进行了完全的验证
+ 我们发布了一个PyTorch[35]包来轻松实现μTransfer。附件H给出了这个方案的草图。

**术语**
有时，为了不那么含糊，我们经常将大模型称为目标模型，因为它是我们希望最终调优的模型，而将小模型称为代理模型，因为它代理了HP调优过程。对于变压器的尺寸，我们遵循标准符号$d_{model},d_{head} = d_k,d_v,n_{head},d_{ffn}$;我们可以看看图11来复习一下。

*Tensor Programs Series* 本文是*Tensor Programs Series*的第5部分。虽然它是自成体系的，目标受众是从业者和实证研究人员，但本文提出了在以前的著作中建立的理论基础的第一个主要实践收益[53-58]

## 2. 参数化问题:入门
在本节中，我们将对为什么正确的参数化可以允许跨宽度的HP传输进行非常基础的介绍，但要了解更多(数学)细节，请参阅附录J.1到J.3。

中心极限定理(CLT)说，如果$x_1，…，x_n$为来自零均值、单位方差分布的iid样本，则当$n \rightarrow \infty$时$\frac{1}{\sqrt{n}} (x_1 +···+ x_n)$收敛于标准高斯$\mathcal{N}(0,1)$

因此，我们可以说 $\frac{1}{\sqrt{n}}$ 是*缩放系数*$c_n$的正确的阶，使得 $c_n(x_1+...+x_n)$收敛域某个非平凡的东西。与此相反，如果我们设置 $c_n=\frac{1}{n}$，那么 $c_n(x_1+...+x_n) \rightarrow 0$ ；或者如果 $c_n=1$ 那么随着$n \rightarrow \infty$ $c_n(x_1+...+x_n)$  的方差会爆炸。

现在，假设我们想要最小化函数

$$
F_n(c) \stackrel{\text { def }}{=} \underset{x_1, \ldots, x_n}{\mathbb{E}} f\left(c\left(x_1+\cdots+x_n\right)\right)
$$
其中，$c \in \mathbb{R}$，对于有界连续函数$f: \mathbb{R}\rightarrow \mathbb {R}$。如果我们重新设置$c=\alpha / \sqrt{n}$, 其中$\alpha \in \mathbb{R}$，根据中心极限定理，$G_n(\alpha) \stackrel{\text { def }}{=} F_n(c) \rightarrow \mathbb{E} f\left(\mathcal{N}\left(0, \alpha^2\right)\right)$会随着 $n\rightarrow \infty$ 稳定为$\alpha$的函数。当n足够大时，对于任意 $N > n$ 最优 $\alpha_n^* \stackrel{\text { def }}{=} \arg \min _\alpha G_n(\alpha)$  应当接近于 $\alpha^{*}_{N}$。并且，事实上，对于 $N=\infty$, 这就意味着我们可以把一个小问题($F_n$)的最佳$c^*_n$或$\alpha^*_n*$转移到一个大问题($F_N$)上：$G_N$ 近似被 $\alpha^*_n$ 最小化，$F_N$ 近似被 $c^*_n \sqrt{n/N}$ 最小化。因为迁移算法只是简单的复制 $\alpha$ ，那么我们说参数化 $c=\alpha / \sqrt{n}$ 是问题正确的参数化。

在本文研究的场景中，$x_1，…，x_n$类似于宽度为$n$的神经网络的随机初始化参数，$c$ 类似于学习率等HP, $f$ 为网络训练后的测试集性能，因此$F_n$给出了其对随机初始化的期望。就像在这个例子中一样，如果我们正确地参数化学习率和其他hp，那么我们可以直接将窄网络的最优hp复制到宽网络中，并期望近似最优的性能，这就是我们在这里提出的(零样本)超参数传输。结果表明，在[57]中提出的最大更新参数化(μP)是正确的(类似于上面α中的参数化)，而标准参数化(SP)是不正确的(类似于c中的参数化)。我们将很快回顾这两种参数化方法。  理论上，μP网络有一个定义良好的无限宽度限制，类似于$(x_1+...+x_n)/\sqrt{n}$有一个 CLT 的 $\mathcal{N}(0,1)$ 极限，而SP网络没有(这个极限会爆炸)[57]。事实上，基于[57]中奠定的理论基础，我们在附录J.3中认为μP也应该是允许跨宽度传递HP的唯一参数化。有关术语参数化和传输的更正式的讨论，请参见附录a。

我们强调，为了确保任何超参数(例如学习率)的可移植性，仅仅重新参数化那个超参数是不够的，相反，我们需要识别并正确地重新参数化表2中的所有超参数。例如，在图1中，SP中的宽模型仍然不如μP的等效模型，即使优化了学习率。  
这正是因为与μP相比，SP不能正确缩放参数乘数和输入/输出层学习率(见表3)。参见附录C，通过我们这里的例子的延续获得更多的直观感受。我们还将在第5节神经网络的背景下更具体地解释这一点。

## 3 使用常规方式超参数不能迁移

在社区中，对于HP的稳定性似乎有相互矛盾的假设。从先验角度来看，不同尺寸的模型没有理由共享最佳的hp。事实上，旨在获得最先进结果的论文经常分别调整它们。另一方面，深度学习中有相当一部分的论文在与基线比较时固定了所有的hp，这反映了一个假设，即最优hp不仅在相同的不同尺寸的模型中是稳定的，而且在不同设计的模型中也是稳定的，因此这样的比较是公平的。在这里，我们在标准参数化的MLP和变压器中明确地演示了跨宽度的HP不稳定性。我们将只考虑训练损失，以排除正则化的影响。

**使用标准参数化的MLP** 我们从一个带有激活函数φ的2隐藏层MLP开始，使用标准参数化（即通常深度学习框架提供的默认参数化。请参见表3的回顾）和LeCun初始化（这里的关键是初始方差 $\propto$ 1/fan_in ），类似于PyTorch中的默认值。

![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/11/13/cb0308bdc818032c7fe8e59dc8710952-20221113221118-3fe02a.png)

其中 $W^1 \in \mathbb{R}^{d_{i n} \times n}, b^1 \in \mathbb{R}^n$， $W^2 \in \mathbb{R}^{n \times n}, b^2 \in \mathbb{R}^n, W^3 \in \mathbb{R}^{n \times {d_{out}}}$ ；$d_{in}$，$n$ 和 $d_{out}$ 分别是输入，隐层和输出的维数。我们使用的特定MLP具有φ = ReLU和交叉熵(xent)损失函数。我们将MLP的宽度定义为隐藏大小n，从256到8192不等。模型在CIFAR-10上训练了20个周期，这足以保证收敛。

![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/11/13/d1230e4b81c58e626ce6b55da1c62c0e-20221113222154-c51cb6.png)

图3:使用SGD在CIFAR-10上训练20个epoch的MLP宽度不同的隐藏大小。左采用标准参数化(SP);right使用最大更新参数化(μP)。μP网络比SP网络具有更好的学习速率稳定性

如图3左侧所示，当宽度从256增加到8192时，最优学习率大约会发生数量级的变化;在最大模型上使用最小模型的最优学习，即使没有发散，也会带来非常糟糕的性能

**使用标准参数化的Transformer** 对于更复杂的体系结构(如Transformer)，这也许并不令人惊讶，如图1(左)所示。我们将宽度定义为$d_{model}$，其中 $d_k=d_q=d_v=d_{model}/n_{head}$ ，$d_{ffn}=4d_{model}$，模型在wikitext-2上训练了5个epoch。在附录的图18中，我们还展示了初始化尺度和其他hp的不稳定性。

## 4. 用 µP 解锁零样本超参数迁移 

我们将展示μP可以解决我们在第3节中看到的问题。

**使用 µP 的多层感知机**。对于第3节中的多层感知机，要转换为μP，只需要修改Eq.(2)对最后一层的初始化，以及对第一层和最后一层的学习率和偏差的学习率

$$
\begin{gathered}
\text { initialize } W^1 \sim \mathcal{N}\left(0,1 / d_{i n}\right), W^2 \sim \mathcal{N}(0,1 / n), W^3 \sim \mathcal{N}\left(0,1 / n^2\right), b^{\{1,2\}}=0 \\
\text { with SGD learning rates } \quad \eta_{W^1}=\eta_{b^1}=\eta_{b^2}=\eta n, \eta_{W^2}=\eta, \eta_{W^3}=\eta n^{-1}
\end{gathered}
$$

![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/11/17/fbcc1c48cb7af13148b67a45f9a9f6c6-20221117204039-017f60.png)

这里，η指定“主学习率”，我们用紫色突出显示两种参数化的差异。这种基本形式明确了参数化的宽度为n的缩放，但在实践中，我们通常会在每个 n 前插入(可能是可调的)乘法常数。例如，当我们希望与宽度为基本宽度 $n_0$ 时与SP MLP一致时，这是很有用的。我们可以按如下方式插入尝试：For $\tilde{n} \stackrel{\text { def }}{=} n / n_0$

$$
\begin{gathered}
\text { initialize } W^1 \sim \mathcal{N}\left(0,1 / d_{i n}\right), W^2 \sim \mathcal{N}(0,1 / n), W^3 \sim \mathcal{N}(0,1 / n \cdot \tilde{n}), b^{\{1,2\}}=0 \\
\text { with SGD learning rates } \quad \eta_{W^1}=\eta_{b^1}=\eta_{b^2}=\eta \tilde{n}, \eta_{W^2}=\eta, \eta_{W^3}=\eta \tilde{n}^{-1} .
\end{gathered}
$$
![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/11/17/a87dd37f32a1e3f90895bfb654437c22-20221117204759-ab9709.png)

那么在宽度$n=n_0$处，上面所有的**紫色因子都为1**，参数化与宽度 SP（$n=n_0$）相同。
当 n 从 $n_0$ 开始增加时，则Eq.(4)**迅速偏离**Eq.(2)。
换句话说，对于特定的 n ， μP和SP可以在选择某些常数(在这种情况下是 $n_0$ )之前相同，但μP决定的网络和优化轨迹与 SP 决定的 n 不同。正如我们将在下一节的经验中看到的，这种偏差对HP转移至关重要。

事实上，在图3(右)中，我们绘制了$n_0=128$的 μP mlp 在不同学习率和宽度下的CIFAR10性能。与SP相比，μP条件下的最优学习率是稳定的。这意味着，带宽为128的网络的最佳学习率同样适用于带宽为8192的μP网络，即HP传输有效，但SP传输无效。

这个MLP μP示例可以很容易地推广到SGD或Adam下训练的一般神经网络，如表3所示，该表由附录J派生。

![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/11/20/92776594a757d40b19e222f82ca37286-20221120194236-72b37f.png)
图4: **4个具有代表性的超参数** 在 μP 的预LN transformer上的稳定性的经验验证: 学习率、最后一层权重乘子α输出、权初始化标准差和学习率表。我们使用以下学习速率表:(a)线性衰减;(b) StepLR @ [5k, 8k]，衰减系数为0.1;(c)衰减因子为0.3的StepLR @ [4k, 7k];(d)余弦退火;(e)常数;(f)反平方根衰减。所有模型都在wikitext-2上进行10k步的训练。当图例中没有指定时，使用的宽度为256，深度为2，批次大小为20，序列长度为256,LR调度常数。我们扫描对应于每一列的特定HP，同时将其他所有HP固定为常数。有关这些结果的讨论，请参阅第6.1节。

**使用 µP 的 Transformer**：我们重复实验，基本宽度 $n_0 = 128$的变压器。

**定义 4.1** 表3给出了变压器的最大更新参数化(Maximum  Update Parameterization, μP)。其中，attention 层使用 $1/d$ ，而不是 $1/\sqrt{d}$，也就是说 attion logit 的计算方法是 $q^{T}k/d$ 而不是 $q^{T}k/\sqrt{d}$，其中 query 矩阵 q 和 key 矩阵 k 的维度是 d （这大致是因为在训练过程中，q和k是相关的，所以由于**大数定律** $q^{T}k$实际上像d一样缩放，与最初的动机相反，q和k在初始化时是不相关的，因此应用中心极限。参见附录 J.2.1 获得更深入的讨论）。

结果如图1右侧所示，最优学习率稳定，性能随宽度单调提高。有关μP的进一步解释，请参见附录B。

## SP 的缺陷及 μP 如何修复这些缺陷

SP vs μP 的问题在[57]中已经进行了详细的研究。在这里，我们旨在概括主要见解，并在附录 J.3 中给出更多解释。

一个启发性的例子 如[57]和附录J.3所示，在SP中，SGD的1步之后，网络输出会以宽度膨胀。


