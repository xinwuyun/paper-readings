---
alias: TensorFlowLargeScaleMachineLearningHeterogeneousDistributedSystems2016
tags: unread
rating: ⭐
share: false
ptype: article
---

# TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems
<cite>* Authors: [[Martín Abadi]], [[Ashish Agarwal]], [[Paul Barham]], [[Eugene Brevdo]], [[Zhifeng Chen]], [[Craig Citro]], [[Greg S. Corrado]], [[Andy Davis]], [[Jeffrey Dean]], [[Matthieu Devin]], [[Sanjay Ghemawat]], [[Ian Goodfellow]], [[Andrew Harp]], [[Geoffrey Irving]], [[Michael Isard]], [[Yangqing Jia]], [[Rafal Jozefowicz]], [[Lukasz Kaiser]], [[Manjunath Kudlur]], [[Josh Levenberg]], [[Dan Mane]], [[Rajat Monga]], [[Sherry Moore]], [[Derek Murray]], [[Chris Olah]], [[Mike Schuster]], [[Jonathon Shlens]], [[Benoit Steiner]], [[Ilya Sutskever]], [[Kunal Talwar]], [[Paul Tucker]], [[Vincent Vanhoucke]], [[Vijay Vasudevan]], [[Fernanda Viegas]], [[Oriol Vinyals]], [[Pete Warden]], [[Martin Wattenberg]], [[Martin Wicke]], [[Yuan Yu]], [[Xiaoqiang Zheng]]</cite>


* [Local library](zotero://select/items/1_WE3KMH9G)

***

### 初读印象

comment:: Tensorflow 白皮书

[论文翻译](https://www.jianshu.com/p/65dc64e4c81f)

## Introduction

前身：DistBelief

### 相关工作 

+ 单机框架
	+ Caffe
	+ Torch
+ Batch dataflow system 
	+ MapReduce
	+ Spark
+ Parameter servers
+ MXNet

参数服务器架构能满足大部分需求，DistBelief 使用带有类似Caffe的模型定义格式的参数服务器效果很好。但是这个架构的扩展性不够，因为添加一个新的优化算法或者尝试一个非传统的模型架构将需要用户修改参数服务器的实现。

虽然使用该系统的一些从业者对这些更改感到舒适，但是大多数人习惯使用高级语言编写模型，而高性能参数服务器的实现是比较复杂的。

因此，使用TF，我们寻求一个*高级编程模型*，允许用户定制运行在系统所有部分的代码

## TF执行模型 
TensorFlow使用单个数据流图来表示机器学习算法中的所有计算和状态，包括单独的数学操作，参数及其更新规则，以及输入预处理(图1)。数据流使子计算之间的通信显式，因此很容易在并行下执行独立计算，并跨多个分布式设备划分计算。TensorFlow在两个方面不同于batch dataflow 系统(2.2)
+ 该模型支持对整个图的重叠子图进行多个并发执行。
+ 单个顶点可能具有可变的状态，可以在图的不同执行之间共享。
具有可变状态的数据流使Tensor- Flow能够模拟参数服务器的功能，但具有额外的灵活性，因为它可以在承载共享模型参数的机器上执行任意数据流子图。


#### Partial Execution

Tensorflow allow executing arbitrary subgraph. Can inject arbitrary data along any edge. 

Each node has a name. Each output of a node has a number （port）. 
e.g., “bar:0”  refers to the 1st output of the “bar” node, while “bar:1”  refers to the 2nd output

Two arguments can define a subgraph that will be executed.
+ Inputs
+ Outputs

![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/08/12b69659897b60caedaad4916caccdd5-20221008213808-157461.png)
### Device Constraints


规定某些 node 只能在某些 devices 上执行

### 