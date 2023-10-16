---
alias: HyperParameterOptimizationReviewAlgorithmsApplications2020
tags: unread HPO
rating: ⭐
share: false
ptype: article
---

# Hyper-Parameter Optimization: A Review of Algorithms and Applications
<cite>* Authors: [[Tong Yu]], [[Hong Zhu]]</cite>


* [Local library](zotero://select/items/1_I79XGGLZ)

---

### 初读印象

comment:: HPO 综述，本文调查了 HPO 算法、工具以及未来的挑战

*自动*超参数优化 Automated HPO

超参优化关注如下问题
+ 模型训练和结构相关的关键超参数（搜索空间）：重要性、取值范围等问题
+ 不同的算法：搜索算法，提前停止策略

---
### 文章结构

*Section 1*: 介绍
*Section 2*: 哪些超参数？超参数的影响、搜索空间 
*Section 3*: HPO 算法及它们在不同机器学习模型上的效率和应用性。分为搜索算法和基于提前停止的 Trial Schedulers
*Section 4*: 介绍主流 HPO 工具和服务
*Section 5*: 全面比较各种方法，强调了模型评价的有效方法
*Section 6*: 结论

---
### 背景

神经网络应用很高效，但是或的模型很低效。认为是一种蛮力方法（brute），这是应为网络用一个随机状态初始化的，并且数据集很大。

对于模型设计、算法设计以及超参数选择，往往是基于经验的。众所周知，这很玄学。

AutoML 以计算资源为代价，自动设计和训练神经网络。超参数优化(HPO)是AutoML在神经网络结构和模型训练过程中寻找最优超参数的重要组成部分。

如何利用计算资源和设计一个**有效的搜索空间**的问题导致了各种关于HPO算法和工具箱的研究。

---
### HPO 的目的
1.  减少专家的繁重工作
2.  提高神经网络的准确性和效率 [NNI](https://github.com/microsoft/nni#nni-released-reminder)
3.  使超参数集的选择更具说服力，训练结果更具可重复性

**Reference**

+ Matthias Feurer and Frank Hutter. Hyperparameter optimization. In Automated Machine Learning, pages 3–33. Springer, 2019
+ [https://github.com/microsoft/nni#nni-released-reminder](https://github.com/microsoft/nni#nni-released-reminder)
+ James Bergstra, Daniel Yamins, and David Daniel Cox. Making a science of model search: Hyperparameter optimization in hundreds of dimensions for vision architectures. 2013.

---

### 深度学习近年的趋势

+ 第一个趋势是更大规模的模型 
+ 第二个趋势是设计一个巧妙的轻量级模型，以较少的权重和参数提供令人满意的精度

---

### Section 2: 主要超参数和搜索空间

---

优先考虑重要性较大的超参数。一般来说，重要性越高的研究越多，其重要性已由以往的经验决定。

---

### 超参数的分类

+ 用于训练
+ 用于模型设计

---

### Optimizer

The most adopted optimizer for training a deep neural network is 
+ SGD with momentum
+ Variants of SGD with momentum: AdaGrad, RMSprop, Adam 

---

### Batch size and learning rate

Batch size and LR determine *the speed of convergence*

Its tuning is essential.

---

### HP for model design

与神经网路结构相关的 HP ，最典型的是模型的深度（隐层数量）和宽度。

这些 HP 通常**衡量模型的学习能力**

---

### Section 3: Search Algorithms and Trial Schedulers 
---

HPO是一个有着悠久历史的问题，但近年来在深度学习网络的应用中受到广泛关注。总的原理是一样的 : 确定要调优的超参数和它们的搜索空间，从粗到细调整它们，用不同的参数集评估模型的性能，并确定最佳组合(Strauss, 2007)。

---

 一般而言：搜索算法用于采样，trial scheduler 主要处理提前停止方法。

---

### 提前停止策略

HPO通常是一个耗时的过程，计算成本很高。在现实场景中，有必要用有限的可用资源来设计HPO流程。当专家手工调优超参数时，他们偶尔可以利用自己的参数经验缩小搜索空间，在**训练过程中**对模型进行评估，并决定是停止训练还是继续训练。

这在神经网络中尤其常见，因为用**每个超参数集训练一个模型需要很长时间**。早期停止策略是一系列**模仿AI专家行为**的方法，以最大化有希望的超参数集的计算资源预算。

除了高效的搜索算法，评估试验并决定是否提前停止试验的策略是另一个活跃的研究领域。下列的提前停止算法可以与搜索算法结合使用，也可以单独用于搜索和暂停。

**HPO 的早期停止算法与神经网络训练的早期停止算法相似**，但在神经网络训练中，是为了避免过拟合，采用了早期停止算法。这允许用户在完成整个训练之前终止试验，从而释放计算资源，用于有前景的超参数集的试验。

---

### 中间止损 Median Stopping

无模型，使用范围广。
A trial *X* is stopped at *step  S* if the best objective value by *step S* is strictly worse than the median value of the running average of all  completed trials objective value ported at *step S* (Golovin et al., 2017)

**Reference**
[Google vizier: A service for black-box optimization]([https://research.google.com/pubs/archive/46180.pdf](https://research.google.com/pubs/archive/46180.pdf))

---
### 曲线拟合 Curve Fitting

曲线拟合是一种LPA(学习、预测、评估)算法(Kohavi和John, 1995;provst et al.， 1999)。它适用于第2节中提出的搜索算法的组合，由谷歌Vizier和NNI支持。这个早期停止规则通过从一组完成或部分完成的试验中回归的性能曲线对最终的目标值(例如，准确性或损失)进行预测。如果对最终目标值的预测远远低于试验历史中最优值的容忍值，试验X将在步骤S停止。

与 Median Stopping 相比，曲线拟合法是一种带参数的模型。建立模型也是一个训练过程。当与搜索算法相结合时，预测终止加速优化过程，然后找到最先进的网络。

Freeze-Thaw BO可以被视为 BO 和曲线拟合的组合

---

### SHA (successive halving) and HyperBand

本节和下一小节将讨论几种基于 bandit 的算法，它们在优化深度学习超参数方面具有很强的性能。深度神经网络中的HPO更可能是准确性和计算资源之间的权衡，因为训练DNN是一个非常昂贵的过程。SHA (Successive halving )和HyperBand在为HPO节省资源方面优于传统的不提前停止搜索算法，采用随机搜索作为采样方法和基于土匪的提前停止策略。

![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/24/15bf7f137f4ec55b77722f6eee805aba-20221024103937-bdf5b6.png)
![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/24/ceae1f09846a4a6acc8cc6fa082d4b42-20221024104005-a346fc.png)



**Reference**
Kevin Jamieson and Ameet Talwalkar. Non-stochastic best arm identification and hyper-parameter optimization. In Artificial Intelligence and Statistics, pages 240–248, 2016

---

SHA将HPO转换为非随机最佳臂识别**，将更多的计算资源分配到更有前途的超参数集

与BO相比，SHA在理论上更容易理解，计算效率更高。SHA对中间结果进行评估，以决定是否终止它，而不是在模型被完全训练到收敛时对其进行评估。SHA的主要缺点是资源的分配。
...
如果n过大，则每次试验的预算较小，可能导致提前终止，而n不足则无法提供足够的可选选项。此外，过大的预算可能导致选择的浪费，而过小的值可能不能保证最优

---

### ASHA ( Asynchronous Successive Halving ) and Bayesian Optimization HyperBand

### 工具
#### Amazon：SageMaker
![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/25/2057783bd1c268b7677b3e0445bef931-20221025003547-5528cd.png)

![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/25/5396b744c1b17b8691863e28fa7ad18a-20221025003513-41d5a0.png)
![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/25/184a1e925e8ddeee6e04c7a54eaa35bf-20221025003533-81a64f.png)


![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/25/7e53bfd09d86e85402829105834c0a1f-20221025003429-0a4333.png)
