---
alias: zhengAlpaAutomatingInter2022
tags: 
rating: ⭐
share: false
ptype: article
---

# Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning
<cite>* Authors: [[Lianmin Zheng]], [[Zhuohan Li]], [[Hao Zhang]], [[Yonghao Zhuang]], [[Zhifeng Chen]], [[Yanping Huang]], [[Yida Wang]], [[Yuanzhong Xu]], [[Danyang Zhuo]], [[Eric P. Xing]], [[Joseph E. Gonzalez]], [[Ion Stoica]]</cite>


* [Local library](zotero://select/items/1_IYCWLURG)

***

### 初读印象

comment:: 用于分布式深度学习的内部和内部运算符并行自动化

# 视频讲解

[https://www.youtube.com/watch?v=g_E7UfpXusk&ab_channel=uwsampl](https://www.youtube.com/watch?v=g_E7UfpXusk&ab_channel=uwsampl)

## Backgound

State-of-the-art model is getting larger

![[Pasted image 20220813113808.png]]

大规模模型参数过多，只能通过分布式训练进行

![[Pasted image 20220813145506.png]]

+ data parallelism
+ tensor partitioning （Model parallelism ）
+ pipeline parallelism 

> 什么是pipeline parallelism?
> 答：流水线并行
> 参考资料[https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/collective/collective_mp/pipeline.html](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/collective/collective_mp/pipeline.html)
>![[Pasted image 20220813150429.png]]提高计算效率和设备并发度，提高设备利用率。
>
> 模型并行？
> ![[Pasted image 20220813160559.png]]
> [https://zhuanlan.zhihu.com/p/432907816](https://zhuanlan.zhihu.com/p/432907816)
> ![[Pasted image 20220813152832.png]]

为了训练更大的模型要结合这些技术

![[Pasted image 20220813154743.png]]

无法使用数据并行，因为模型本身很大，无法放到一个GPU中。所以必须切分权重。

![[Pasted image 20220813160714.png]]

针对不同的模型需要指定不同的策略。这非常有挑战性。

## 架构

这个项目目的是综合这些策略，构建一个编译器来自动生成这些策略的最优集合

![[Pasted image 20220813162356.png]]

把传统的并行技术分为两种：
+ Intra-op
+ Inter-op

我们需要更高效的 Auto-Parallelization，前人的工作有以下限制

![[Pasted image 20220813164959.png]]

 ![[Pasted image 20220813165212.png]]

+ 作者设计了一个两层的分层系统。
+ 设计了算法

### Intra-op

![[Pasted image 20220813174248.png]]

我们需要为每个op选择一个策略，最小化总的时间消耗。

![[Pasted image 20220813175207.png]]
每个 op 有若干个策略可供选择，不同 op 策略不同，可以做到逐个列举。

![[Pasted image 20220813180649.png]]
枚举每个 op 的可能的 partition 策略，计算他们的通信cost，最小化总的时间耗时

![[Pasted image 20220813181048.png]]

> 有人问到了通信之间的 overlapping

### Inter-op 

![[Pasted image 20220813212023.png]]

划分子图为 stages，给设备集群分片为 submesh。
一个子图对应一个submesh，进行 pipeline parallelism 

![[Pasted image 20220813212158.png]]

使用 DP 算法最小化**流水线延迟**
![[Pasted image 20220813212309.png]]
![[Pasted image 20220813212325.png]]
![[Pasted image 20220813212546.png]]

> 时间复杂度很高，有个 $K^5$。$K$ 是图中的操作数。
> 为了解决这个问题，先做一个预处理，使用图聚类算法来吧相似的图聚类为一个layer。

> 有人提问：这个 DP 算法是什么时候运行，运行多少次，是一开始运行一次还是每个stage都要进行？
> 答：这里前面的 overview 的图里可以找到答案。
> ![[Pasted image 20220813213404.png]]

## 实现

使用 jax
> pathways 也用的这个， [[@barhamPathwaysAsynchronousDistributed2022|barhamPathwaysAsynchronousDistributed2022]]

![[Pasted image 20220813221333.png]]

![[Pasted image 20220813223406.png]]

## 实验

>[High performance computing](https://en.wikipedia.org/wiki/High_performance_computing "High performance computing") has two common notions of scalability:
  + _Strong scaling_ is defined as how the solution time varies with the number of processors for a fixed _total_ problem size 
  > _Weak scaling_ is defined as how the solution time varies with the number of processors for a fixed problem size _per processor_.[[10]](https://en.wikipedia.org/wiki/Scalability#cite_note-10)

> 这里随着 GPU 数量增加，实验的参数量也会增加

![[Pasted image 20220813231534.png]]

> Megatron-LM 是手工设计并行策略的

> Linear-Scaling，和 Communication Bound 相关

> Megatron-LM 是专为 transformer model 设计的，有人问到之前人们都是怎么训练这么大的模型？
> 答：还是 Megatron-LM ，因为目前的大型模型和GPT3 比较像，MoE 有点不一样。
> ![[Pasted image 20220813233520.png]]
> https://www.oneflow.org/a/share/jishuboke/75.html

> 有人问到 tofu 和 alpa 的区别
> 答：tofu 只考虑了 intra-op，没有使用流水线
## Summary

![[Pasted image 20220813232112.png]]

# 精读

[知乎:[阅读笔记] Alpa/Parax @OSDI 2022](https://zhuanlan.zhihu.com/p/521211578)

