---
alias: TVMAutomatedEndtoEndOptimizingCompilerDeepLearning
tags: 深度学习编译器 TVM 陈天奇
rating: ⭐
share: false
ptype: article
---

# TVM: An Automated End-to-End Optimizing Compiler for Deep Learning
<cite>* Authors: [[Tianqi Chen]], [[Thierry Moreau]], [[Ziheng Jiang]], [[Lianmin Zheng]], [[Eddie Yan]], [[Meghan Cowan]], [[Haichen Shen]], [[Leyuan Wang]], [[Yuwei Hu]], [[Luis Ceze]], [[Carlos Guestrin]], [[Arvind Krishnamurthy]]</cite>


* [Local library](zotero://select/items/1_GHW5IZP6)

***

### 初读印象

comment:: 陈天奇的TVM，很重要的深度学习编译器

### 文章骨架
%%创新点到底是什么?%%
novelty:: 端到端图优化，支持多种硬件后端，自动化端到端优化

%%有什么意义？%%
significance:: 自动化的端到端优化可以减少人力工作，使得深度学习编译器可以更好支持更多后端。机器学习模型可以更轻松地部署到更多种设备

%%有什么潜力?%% 
potential:: 已经大有可为了

## 摘要

作者希望能够更轻松地把机器学习模型部署到更多种设备上。当前流行的框架，依赖于特定供应商的操作程序库，针对**较窄的服务器级 GPU 进行优化**。

我们提出了TVM，这是一种公开**图级和操作级优化**的编译器，为**跨不同硬件后端**的深度学习工作负载提供性能可移植性。

**解决了哪些挑战**

+ high-level operator  fusion
+ Mapping to arbitary hardware primitives
+ memory latency hiding
+ etc

还实现了自动优化 low-level program, 基于的是一种 learning-based cost model.

## Intro

### 当代框架存在的问题
Current DL frameworks, such as TensorFlow, MXNet, Caffe, and PyTorch, rely on a computational graph intermediate representation to implement optimizations,  e.g., auto differentiation and dynamic memory management [3, 4, 9].

Graph-level optimizations, however, are often too high-level to handle hardware back-end- specific operator-level transformations. Most of these frameworks focus on a narrow class of server-class GPU devices and delegate target-specific optimizations to highly engineered and vendor-specific operator libraries

> 当代框架的不能适用于所有设备

即使对于受支持的后端，框架也必须做出艰难的选择:
1. 避免图优化产生的新操作符不在预定义的操作符库中
2. 使用这些新操作符的未优化实现。

### TVM 提出的解决方法 

**Fundamentally different, end-to-end appraoch.**

*We built TVM, a compiler that takes a high-level specification of a deep learning program from existing frameworks and generates low-level optimized code for **a diverse set of hardware back-ends.***

TVM 为了得到更具竞争力的性能，解决了如下关键挑战

#### 1. Leveraging Specific Hardware Features and Abstractions.

#### 2. Large Search Space for Optimization

不需要手动优化就能生成高效的代码。

**内存进程、线程模式和新的硬件原语的组合选择**为生成的代码创建了巨大的配置空间(例如，loop tiles和ordering、caching、unrolling)，如果我们实现blackbox auto-tuning，将会产生很大的搜索成本。
我们需要对每种硬件类型建立一个 cost model，而且现代硬件越来越复杂，这个模型也会也来越大。

**TVM 解决这些挑战使用的三个模块**
1. *tensor expression language*：build operators and provide program transformation primitives that generate different versions of the pro- gram with various optimizations. 学习了 Halide 的 compute/schedule 分离思想。
2. *automated program optimization framework*：寻找优化的张量ops。optimizer 由一个基于 ml 的 cost model 指导，**该模型随着我们从硬件后端收集更多的数据而调整和改进。**
> 所谓的 ML for ML
3. *graph rewriter*

> 具体见论文

通过这三个模块的结合，TVM可以
1. 从现有的深度学习框架中获取**模型描述**，
2. 进行高级和低级的联合优化，
3. 进而生成针对后端硬件的优化代码，如cpu、gpu和基于fpga的专用加速器


## 2. Overview

![](../../08-Assets/Pasted%20image%2020220824134029.png)

1. 该系统首先从现有框架中获取一个模型作为输入，并将其转换为 Computational Graph 表示。
2. 执行 high-level dataflow rewriting 来生成一个优化的图。
操作符由声明式张量表达式语言指定;执行细节未指定。
*TVM identifies  **a collection of possible code optimizations** for a given  hardware target’s operators*

这个搜索空间非常大，所以作者使用 ***ML-based cost model*** 来找到最优的 optimized ops.

最后，系统把生成的代码打包为 deployable module

### **代码示例**

从已有的深度学习框架中获取 model。调用 TVM api。
```python
import tvm as t  
# Use keras framework as example, import model  
graph, params = t.frontend.from_keras(keras_model)  
target = t.target.cuda()  
graph, lib, params = t.compiler.build(graph, target, params)
```
最终的运行时模块包含三部分，
1. graph: 优化后的计算图
2. lib: generated operators 
3. params: **module parameters**


他们可以用来 deploy the model 到目标后端。
```python
import tvm.runtime as t  
module = runtime.create(graph, lib, t.cuda(0))  
module.set_input(**params)  
module.run(data=data_array)  
output = tvm.nd.empty(out_shape, ctx=t.cuda(0))  
module.get_output(0, output)
```
## 3. Optimizing Computational Graph 

![](../../08-Assets/Pasted%20image%2020220824155741.png)
双层CNN示例，每个节点表示一个 operation ，每个 operation 会消耗一个或多个张量，并产生若干个张量。
张量操作可以通过属性参数化来配置它们的行为(例如，填充或步长)。

与 LLVM IR 类似，计算图可以转换成等价的优化图。
TVM 还利用普通 DL 工作负载中的形状特异性类优化一组固定的输入形状。
TVM 实现了以下图级别优化
+ operator fusion: 融合算子
+ constant folding：预先计算 Graph 中的静态部分，节省执行成本
+ data layout transformations：将 internel data 转换为对后端友好的形式，有的后端不支持一些数据结构

### 算子融合 

将多个 op 融合成一个 kernel，这个 kernel 在**内存中不保留中间结果，可以减少 memory access**。能很好地减少执行时间尤其是 GPU 和一些特定加速器。

将 graph operator 分为四类
1. injective (one-to-one map，比如 add)
2. reduction (比如 sum)
3. complex-out-fusable(可以融合元素级映射到输出)
4. opaque（不能融合，比如 sort）

**通用的规则**
+ 多个 Injective 可以融合为另一个 injective
+ reduction op 可以和 input injective 融合
+ complex-out-fusable: fuse element-wise operators to its output
通过这些规则，可以把 computational graph 转换为一个融合的版本

**效果图**

![](../../08-Assets/Pasted%20image%2020220824171016.png)
融合前后对比，有 1.2x 到 2.0x 的加速

### Data Layout Transformation 

对于 computational graph 的给定的 tensor，有多种存储方法，比如行主序，列主序等。

针对不同硬件的内存结构，可以对数据布局进行优化。

实际上主流框架也有算子融合优化，但是他们要求融合后的算子在operator library 中上有针对特定硬件的实现。 

随着硬件种类越来越多，这种方法将变得不可持续，因为所需的融合模式实现的数量会随着必须支持的数据布局、数据类型和加速器intrinsic的数量的增加而增加。

全手工优化操作是不可行的。最后，我们接下来提出一种代码生成方法，可以为给定的 model operator 生成各种可能的实现。


![](../../08-Assets/Pasted%20image%2020220824190455.png)
上图左边是 TVM 生成 low-level code 的过程，表中展示了 Halide 的原语和 TVM 独有的 scheduling 原语。*Tensorization* 是专门用在加速器上的，但是 CPU 和 GPU 也能用。
Latency Hiding 专用在 TPU-like 加速器中的。

## 4. Generating Tensor Operations

TVM 通过在每个 hardware back-end 生成许多有效的实现并选择一个优化的实现来为每个 op 生成高效的代码。

这个过程基于 Halide 将 description 和计算解耦的思想，并进行了扩展，支持了各种硬件后端和一些新的优化: 
1. nested parallemlism
2. tensorization 
3. latency hiding 
 
### 4.1 Tensor Expr and Schedule Space 张量表达式和调度空间

与张量操作的实现不透明的高级计算图表示不同，每个操作都用索引公式表达式语言描述。下面是一个计算*转置矩阵乘法*的张量表达式
![](../../08-Assets/Pasted%20image%2020220824175434.png)
每个张量表达式都指定输出 tensor 的 shape。TVM 的 tensor expression language 支持通用的数学 op 并且涵盖常见的 DL op 模式。
表达式不指定循环结构和许多其他执行细节，为不同 backend 的实现提供灵活性。

TVM 使用 Halide 的原理，用 **a schedule 来表示张量表达式到 low-level 代码的映射**。一个 function 有很多可能得 schedule 。

TVM 通过递增地应用**保持程序逻辑等价的基本转换**(调度原语)来构建schedule。
![](../../08-Assets/Pasted%20image%2020220824185654.png)
在专门加速器上优化及矩阵乘法调度转换

为了在许多后端上实现高性能，我们必须支持足够多的调度原语，以覆盖不同硬件后端上的各种优化集。

之后会介绍怎么**得到最优的调度**


### 4.2 Nested Parallemlism with Cooperation

大部分解决方案使用一种数据并行策略，一种fork-join方案，称为 *nested parallemlism *

> 嵌套并行（nested parallelism）指在一个并行程序中调用另一个并行程序，也就是递归并行
> *This model requires a parallel schedule primitive to parallelize a data parallel task; each task can be further recursively subdivided into subtasks to exploit the target architecture’s multi-level thread hierarchy (e.g., thread groups in GPU).*

这里称该 model 为 *shared-notion nested parallemlism* ，线程之间不共享数据。

> 与之相对的，标题中的 Cooperation 就是有共享数据的

线程组可以协作地获取他们都需要的数据把数据放到*shared memory space* 中。这种优化可以利用GPU内存层次结构，并通过共享内存区域实现线程间的数据重用。TVM支持这种众所周知的GPU优化，使用调度原语来实现最佳性能。下面是一个优化矩阵乘法的 GPU 代码示例 
![](../../08-Assets/Pasted%20image%2020220824194947.png)

![](../../08-Assets/Pasted%20image%2020220824195343.png)
使用共享内存的TVM, shared-nothing-TVM 和 cuBLAS 的矩阵乘法性能对比（NIVIDIA Titan X）

### 4.3 Tensorization 

DL工作负载具有很高的运算强度，通常可以分解为 tensor ops ，如矩阵-矩阵乘法或一维卷积。

These new natural decomposition have led to the recent trend of adding  *tensor compute primitives*

> [2016-TensorFlow](../../08-Assets/pdfs/2016-TensorFlow.pdf)

作者引入 *tensornization* 类似于 SIMD 体系结构的向量化。指令输入是多维的，有固定长度或可变长度，每个指令都有不同的数据布局。

更重要的是，我们不能支持一组固定的原语，因为新的加速器正在以它们自己的张量指令变体出现。因此，**我们需要一个可扩展的解决方案**。

We make tensorization extensible by separating the   target hardware intrinsic from the schedule with a mechanism for tensor-intrinsic declaration. We use the same  tensor expression language to declare both the behavior  of each new hardware intrinsic and **the lowering rule associated with it.**

![](../../08-Assets/Pasted%20image%2020220824204651.png)

https://tvm.apache.org/docs/how_to/work_with_schedules/tensorize.html

### 4.4 Explicit Memory Latency Hiding 显示内存延迟隐藏

**延迟隐藏是指内存操作和计算 overlapping 的过程**，以最大限度地利用内存和计算资源。根据 target 硬件后端，他需要不同的策略。在 CPU 上，memory latency hiding 通过  simultaneous multithreading [14] or hardware  prefetching [10, 20].实现的 。
对于特殊的DL加速器，如TPU[21]，通常倾向于使用*decoupled access-execute(DAE)* 架构[35]进行更精简的控制，将细粒度的同步问题推给软件解决。

> 感觉类似于 FasterMoE 中的通信和计算重叠。

使用 DAE 硬件流水线可以减少 runtime latency。与单片系统相比，可以隐藏几乎所有内存访问开销，几乎可以 100% 利用计算资源。

为了获得更高的利用率，**指令流必须增加细粒度的同步操作**。
因此，DAE硬件流水线需要细粒度的流水线stage之间的入队离队操作，以确保正确执行，如图9的指令流所示。
![](../../08-Assets/Pasted%20image%2020220824223010.png)

编程需要显式实现底层同步的 DAE 加速器是很困难的。为了减轻编程负担，TVM 引入了一个 virtual threading scheduling primitives，它允许程序员指定高级数据并行程序，就像他们指定支持多线程的硬件后端程序一样。

然后，TVM通过底层显式同步自动将程序降低到单个结构内流，如图8所示。
![](../../08-Assets/Pasted%20image%2020220824224117.png)

1. 该算法从一个高级多线程程序调度开始，然后插入必要的低级同步操作，以保证在每个线程中正确执行
2. 接下来，它将所有虚拟线程的操作交织到一个单一的指令流中。
3. 最后，硬件恢复指令流中的底层同步所指示的可用流水线并行性

**评估**
![](../../08-Assets/Pasted%20image%2020220824225308.png)

## 5. Automating Optimization 
ML for ML
> Given the rich set of schedule primitives, our remaining problem is to find optimal operator implementations for each layer of a DL model.

这样的组合选择为每个硬件后端创建了很大的运算符实现搜索空间。作者构建了一个自动化的 schedule optimizer，有两个主要组件：
1. schedule explorer: 提出有希望的 config
2. ML cost model: 预测 config 的性能

![](../../08-Assets/Pasted%20image%2020220824232154.png)
+ TVM 构建了schedule template specification API ，开发者可以按需要加入自己的代码
+ 为每个 back-end 创建了一个通用的 master 模板，这个模板可以根据使用张量表达式语言表达的计算描述自动提取可能得 knob
+ 在较高的层次上，我们希望考虑尽可能多的配置，并让优化器来管理选择负担

![](../../08-Assets/Pasted%20image%2020220822170003.png)
### 5.3 Schedule Exploration(模拟退火算法)

**训练规则**
选择了 Cost Model 后，我们可以使用它去选择有希望的 config ，在骑上迭代运行真实的度量。每次迭代时，资源管理器使用 ML模型的预测来选择一批候选对象来运行 measurement。手机的数据作为tarining data 来更新模型。如果不存在初始的训练数据，explorer 就随机挑选候选者阿里运行 measurement 

**搜索算法**
最简单的算法：枚举

启发式算法：搜索空间过大时搜索空间难以驾驭，所以作者使用模拟退火算法。

**过程**
1. 探索者从随机 config 开始，在每一步，**随机**走到附近的 config。如果成本如**成本模型所预测**的那样下降，那么这种转变就是成功的
2. 随机游走倾向于收敛于成本模型**所预测的成本更低的配置**。搜索 状态在成本模型更新期间持续存在;在这些更新之后，我们继续从最后一个配置开始。
  
>Exploration states  persist across cost model updates; we continue from the  last configuration after these updates.


### 5.4 分布式设备池和RPC
分布式设备池扩大了硬件试验的运行，并支持在多个优化作业之间共享细粒度的资源。
TVM实现了一个定制的、基于rpc的**分布式设备池**，使客户端能够在特定类型的设备上运行程序。我们可以使用该接口在主机编译器上编译程序，请求远程设备，远程运行该函数，并在主机上使用相同的脚本访问结果。
TVM的RPC支持动态上传并运行使用其运行时约定的交叉编译模块和函数。

因此，**相同的基础设施**可以执行**单个工作负载优化和端到端图推断**。我们的方法在多个设备上自动化编译、运行和配置步骤。
这种基础结构对于嵌入式设备尤其重要，传统上嵌入式设备需要进行冗长的*手工*编译、代码部署和度量。

## 6. Evaluation

## 7. 相关工作 

## 8. 结论
We proposed an end-to-end compilation stack to solve fundamental optimization challenges for deep learning across a diverse set of hardware back-ends.
我们提出了一个**端到端的编译栈**，以解决跨**不同硬件后端集**的深度学习的基本优化挑战。

Our system includes automated end-to-end optimization, which is historically a labor-intensive and highly specialized task.
我们的系统包括**自动化**的端到端优化，这在历史上是一项劳动密集型和高度专业化的任务

We hope this work will encourage additional studies of end-to-end compilation approaches and open new op- portunities for DL system software-hardware co-design techniques.
我们希望这项工作将鼓励更多的端到端编译方法的研究，并为DL系统的软硬件协同设计技术开辟新的机会。




---
**以下为视频笔记**

## Motivation

设计一个加速器需要支持 Caffe、TensorFlow 等框架，我们不能设计一个硬件就设计一套上层的软件。

我们希望解决这个问题，做一个全栈的优化。

## TVM Stack Goal

端到端深度学习编译器stack，可以适配各种加速器

### High level User's Perspective 

```python
import tvm 
import nnvm.frontend
import nnvm.compiler
```
![](../../08-Assets/Pasted%20image%2020220822132348.png)

### 为什么做到这个很难？

![](../../08-Assets/Pasted%20image%2020220822162521.png)

> 目前已经能够超过手写优化效果

### 计算图优化

计算图本身描述计算的需求，可以对图进行等价的变换。跑变换后的程序可能会比变换前的程序快。

![](../../08-Assets/Pasted%20image%2020220822163053.png)
> 比如上图，三个op转为一个op能更快

计算图优化带来的问题，比如我要做算子融合
+ 人类的力量有限，我们需要做一些自动的方法，做自动的代码生成

在极速那图上的优化 
1. 内存优化
2. 算子融合
3. 数据排布的转换，ncw，矩阵乘法，行优先，列优先。

> XLA 基本上是一个计算图的框架，这种框架的问题在于，每个算子只是图中的一个节点，手写算子的优化选择很多，比如向量化。给定一个卷积在不同硬件上做优化很难做很好。

![](../../08-Assets/Pasted%20image%2020220822164600.png)

![](../../08-Assets/Pasted%20image%2020220822164940.png)

自动搜索算子的最优实现。

![](../../08-Assets/Pasted%20image%2020220822170003.png)



# Another video

### deploying your model.

1. On what?
2. How fast / accurate?
3. Inside of what?![](../../08-Assets/Pasted%20image%2020220822182200.png)
![](../../08-Assets/Pasted%20image%2020220822182350.png)

### TVM 

![](../../08-Assets/Pasted%20image%2020220822182430.png)


## What problems does TVM address?

![](../../08-Assets/Pasted%20image%2020220822183524.png)

### TVM for Portability

![](../../08-Assets/Pasted%20image%2020220822183836.png)
![](../../08-Assets/Pasted%20image%2020220822183850.png)

### TVM for Efficiency

![](../../08-Assets/Pasted%20image%2020220822184219.png)

![](../../08-Assets/Pasted%20image%2020220822184251.png)

### TVM for Software Support

 ![](../../08-Assets/Pasted%20image%2020220822185709.png)
 
挑战：build a software stack 

![](../../08-Assets/Pasted%20image%2020220822185747.png)

![](../../08-Assets/Pasted%20image%2020220823121857.png)

## Impact on Industry
![](../../08-Assets/Pasted%20image%2020220823121919.png)

## Overview of this talk

![](../../08-Assets/Pasted%20image%2020220823125120.png)

import model -> 模型的图

multi-stage lowering 可以最终 target 某个 backend。LLVM C CUDA....

![](../../08-Assets/Pasted%20image%2020220823125547.png)

# 一、TVM op-level optimization 
## Example
![](../../08-Assets/Pasted%20image%2020220823125738.png)

然而必须的是专业的工程师才能写出右边的代码。

### TFLite 中的一个真实例子

![](../../08-Assets/Pasted%20image%2020220823125927.png)

这个代码有大约1.3w行。为什么这么多行呢？
1. 大量汇编代码 
2. 各种不同的变体 
3. Special impl for vector intrinsics (NEON on ARM)

## Operator optimization Challenge

 如何达到这种手写调整的代码同时保证代码可读性和简洁？
+ 有许多op FC conv2d conv3d
+ 每个 op 输入张量各不相同，使用不同的 dilation , strides, padding size
+ 如果 hardware target 变了怎么办？

## Halide programming model 

+ Functional Defination: **What** should this function do?
+ **Schedule defination**: **How** should this function do it?

![](../../08-Assets/Pasted%20image%2020220823130704.png)

## TVM schedule: 矩阵乘法 
![](../../08-Assets/Pasted%20image%2020220823132150.png) 

 这会生成一个tir程序 
 
![](../../08-Assets/Pasted%20image%2020220823133137.png)

这个 tir 可以**lower down to** lvm IR , cuda code, c code, opencl et

## TVM schedule: Tiling and Reordering

![](../../08-Assets/Pasted%20image%2020220823134309.png)


> 可以更好利用cache
![](../../08-Assets/Pasted%20image%2020220823134743.png)

## TVM schedule: After serval Steps

![](../../08-Assets/Pasted%20image%2020220823134429.png)

速度能快200倍

## 矩阵乘法总结

![](../../08-Assets/Pasted%20image%2020220823142907.png)



# 二、Automated Optimization Search

auto scheduling

## Taking optimization to next level 
+ 前面的例子中，我们能大概达到MLK 60% 的性能 
+ 我们如何进一步提升性能呢？
	+ 调度基本是一种艺术。factors used for splits, vectorization, etc. and loop orders are a function of the target hardware architecture, and input tensor shapes.

面对一个op，可能存在上亿种调度方案。

我们希望 ML 来帮我们做这件事。

 ## Problem formalization 

 ![](../../08-Assets/Pasted%20image%2020220823150903.png)
在整个 search space 里有一个方案可以最小化执行时间

### ML based Model 
> **Using ML to optimize ML**

![](../../08-Assets/Pasted%20image%2020220823151249.png)

使用一个 统计学的 Cost Model  
优点：自动适配硬件类型 

效果非常好！
![](../../08-Assets/Pasted%20image%2020220823151755.png)


# 三、TVM graph-level optimization 

 > 上面我们讨论了如何在一个独立kernel 上优化，比如 conv2d。
 
 在图优化中，TVM 使用了 relay。
 原始论文：Relay:A High-Level Compiler for Deep Learning. Roesch et al.ArXiv 19

### High-Level Data Flow Graph 

TVM uses Relay, 一种 Functional and statically typed **IR** to describe ML computation  

+ Ralay 实现了 ML 架构的常见 features 比如 quantization 和 shape inference 

> Quantization for deep learning is the process of approximating a neural network that uses floating-point numbers by a neural network of low bit width numbers.

![](../../08-Assets/Pasted%20image%2020220823155459.png)

### Graph-Level Optimization：Operator Fusion

算子融合

![](../../08-Assets/Pasted%20image%2020220823160258.png)
![](../../08-Assets/Pasted%20image%2020220823160328.png) 

比如这个例子里，我们知道 relu 是一个非常简单的函数，我们完全可以把他融合地哦啊前面的batch norm中。这样，中间很多 DRAM 访问就不用了。

融合后算子编程下面这样
![](../../08-Assets/Pasted%20image%2020220823171819.png)
![](../../08-Assets/Pasted%20image%2020220823171836.png)
+ The idea is to **fuse multiple ops into a single op to minimize memory access**
+ If you have to rely on an operator library, you need to add implementations for fused operators! That's a long list of ops.
+ Thankfully with more flexible code-generation approaches like TVM, we can generate fused kernels on demand


效果
![](../../08-Assets/Pasted%20image%2020220823172357.png)

## Graph-Level Optimization: Quantization 
 ![](../../08-Assets/Pasted%20image%2020220823175026.png)
 ![](../../08-Assets/Pasted%20image%2020220823174243.png)

一般来说 Quantization 用在框架里。如果设备没有能力运行一个模型，你需要修改他的数据类型 ，同时获得更高的性能，可以在编译时使用 Quantization 。

## Results with TVM 
![](../../08-Assets/Pasted%20image%2020220823175916.png)
![](../../08-Assets/Pasted%20image%2020220823175956.png)

  # 四、MicroController Support 

用于高度限制的环境。演讲时还在研发中。
