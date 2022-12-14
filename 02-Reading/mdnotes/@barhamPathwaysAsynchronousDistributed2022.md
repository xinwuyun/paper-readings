---
alias: barhamPathwaysAsynchronousDistributed2022
tags: 分布式机器学习 异步
rating: ⭐⭐
share: false
ptype: article
---

# Pathways: Asynchronous Distributed Dataflow for ML
<cite>* Authors: [[Paul Barham]], [[Aakanksha Chowdhery]], [[Jeff Dean]], [[Sanjay Ghemawat]], [[Steven Hand]], [[Dan Hurt]], [[Michael Isard]], [[Hyeontaek Lim]], [[Ruoming Pang]], [[Sudip Roy]], [[Brennan Saeta]], [[Parker Schuh]], [[Ryan Sepassi]], [[Laurent El Shafey]], [[Chandramohan A. Thekkath]], [[Yonghui Wu]]</cite>


* [Local library](zotero://select/items/1_C8B2R5W5)

***

### 初读印象

comment:: 感觉是一篇硬核的不懂的点会比较多的文章

https://zhuanlan.zhihu.com/p/497461172

---

## 论文精读
[https://www.bilibili.com/video/BV1xB4y1m7Xi](https://www.bilibili.com/video/BV1xB4y1m7Xi)

![[Pasted image 20220807115735.png]]

这是系统论文名字最常见的命名方法，<系统名字>：<系统用途>

> 作者有两位大神，Jeff Dean, Sanjay Ghemawat，经常结对编程 

### 导言

 随着硬件的进步，模型的进步，软件系统也需要隔几年更新。PathWays相信会被未来的机器学习任务需要，

当前机器学习模型一般都是 SPMD 模型。要完成分布式的话需要进行数据的交换。

> 前面学习时[参数服务器](obsidian://open?vault=working&file=02-Reading%2Fmdnotes%2F%40Li2014)用了数据并行，每个节点拿到各自的数据计算梯度，最后加到一起

---
**BSP模型**

整体同步并行计算模型
并发计算 → 通信 → 同步
![Pasted image 20220809142718.png](app://local/Users/chentong/Documents/projects/working/08-Assets/Pasted%20image%2020220809142718.png?1660026438629)[BSP模型wiki](https://zh.wikipedia.org/wiki/%E6%95%B4%E4%BD%93%E5%90%8C%E6%AD%A5%E5%B9%B6%E8%A1%8C%E8%AE%A1%E7%AE%97%E6%A8%A1%E5%9E%8B)

---
### 硬件上的变化

最近，随着每一代加速器的出现机器学习的集群越来越异构。通过高带宽互连对同质加速器的大孤岛进行独占访问是昂贵的，而且常常是浪费的，因为单个用户程序必须设法使所有加速器持续处于繁忙状态。

> [zhihu: MIT2020年力作：机器学习加速器综述](https://zhuanlan.zhihu.com/p/374573729)

# 翻译

## 2. DESIGN MOTIVATION

分布式ML系统的设计选择通常由底层目标硬件加速器的属性驱动。在这里，我们关注现有分布式ML系统的一些设计和实现选择如何使它们难以支持大型的、稀疏的或不规则的模型。

<font color=#ED7001>用于训练最先进的SPMD模型的分布式ML系统</font>通常采用<font color=#ED7001>多控制器体系结构</font>，其中相同的客户机可执行文件直接在系统中的所有主机上运行，在程序执行期间独占这些主机上的资源。这种架构的例子包括MPI (Clarke等人，1994年)，PyTorch (Paszke等人，2019年)，JAX (Bradbury等人，2018年)，以及TensorFlow的最新配置(Shazeer等人，2018年;Agrawal等人，2019年）**这种架构的关键优势是分发加速器计算的低延迟**(参见图1a)，因为用户代码的相同副本运行在每个加速器主机上，并且分派补丁只涉及通过(相对)快速的PCIe链路进行通信。所有其他跨主机的通信只能通过使用专用互连的集合进行，如NVLink (Foley和Danskin, 2017)和ICI (Jouppi等人，2020)，而不通过主机内存<font color=#FF0211>。然而，这种架构与使用流水线或计算稀疏性的现代ML工作负载不太匹配</font>。在多控制器系统中，任何超出标准集合的通信都需要用户实现他们自己的协调原语。多控制器方法<font color=#FF0211>通常还假定独占硬件资源</font>。这不仅将确保昂贵加速器的高利用率的责任转移到用户身上，还使构建高效的集群级ML基础设施所需的资源虚拟化和多路复用等特性的设计变得复杂。

> **多控制器系统**与目前最先进的流水线或者计算稀疏性的现代机器学习工作负载不匹配。

**单控制器系统**，如TensorFlow v1 (Abadi等人，2016)提供了一个非常通用的分布式数据流模型，包括优化的图内控制流(Yu等人，2018)。
1. TensorFlow (TF) Python客户端构建一个计算图 
2. 将计算图移交给协调运行时 
3. 协调运行时将图划分为每个worker的子图，并将子图的执行委托给worker上的本地运行时。
Worker之间的协调是通过数据和控制<font color=#FF0211>边缘在数据中心网络(DCN)上传递消息</font>来执行的。虽然单控制器设计提供了灵活的编程模型和虚拟化资源，但它提出了实现的挑战。

> TF1 使用的就是单控制器系统

1. 首先，多控制器系统只需要**通过PCIe通信来分派加速器计算**(图1a)，而单控制器系统中的客户端距离较远，分派延迟涉及<font color=#ED7001>通过DCN通信</font>，通常比PCIe慢一个数量级(图1b)。
2. 其次，利用SPMD子计算支持MPMD程序的并发执行，每一个子计算都跨越<font color=#FF0211>从共享集群中提取的加速器子集</font>，运行时必须有某种机制来支持加速器计算的分组调度。对于tpu来说，分组调度是必不可少的，因为它们是单线程的，并且只运行不可抢占的内核，所以如果通信计算没有按照一致的顺序进入队列，系统将发生死锁。即使对于gpu或其他可以执行并发计算的加速器，分组调度也可以更有效地执行集合(Feitelson和Rudolph, 1992)。因此，ML的单控制器系统需要一种分布式调度机制来对代表不同程序排队的计算进行排序。
3. 最后，现代ML工作负载系统的设计必须能够运行分布在**数千个加速器上的计算**，并对分片表示和数据结构提供一流的支持。例如，表示M路分片计算和N路分片计算之间的一条边的原始数据流图将需要M + N个节点和M×N条边，很快就会变得难以处理。

TF v1所做的实现选择过于专业化，它假设一个单独的、小型的、独占的加速器孤岛。这种过度专门化使得在当代或未来的ML工作负载中使用TF实际上是不可行的。虽然TF可以通过send和recv操作运行需要跨主机协调或数据传输的计算(图1c)，但只有在传输完成后才会触发主机端在目的地的工作，如调度加速器计算。在涉及许多跨主机传输的程序中，例如具有大量阶段的流水线模型，这些分派延迟累积起来，导致低效的加速器利用率。

![[Pasted image 20220811233944.png]]
图1。多控制器和单控制器系统之间的调度开销和通信模式的比较。(a) Jax或PyTorch SPMD在快速PCIe上独立异步排队加速器计算;(b) TensorFlow v1 SPMD在较慢的DCN上需要控制消息;(c) TensorFlow v1非spmd程序需要通过显式的send (S)和recv (R)操作进行跨主机协调或数据传输。

虽然TF v1用户可以(低效地)在单个程序中强制执行分组调度的一致顺序，但通过使用控制边缘，在TF v1这样的单控制器系统中缺乏集中式调度器，这使得无法确保跨程序的计算之间的一致顺序。TF还实现了完整的分片计算图，当分片数量达到数千个时，在图的序列化和执行方面带来了大量的开销，导致子计算之间有数百万条图边。

PATHWAYS结合了单控制器框架的灵活性和多控制器的性能。我们采用单控制器模型，因为我们相信它比多控制器在计算稀疏性和异构性方面提供了更好的机会，并通过使集群管理系统促进资源共享和虚拟化，从而实现了新的高效的ML计算。我们的设计不同于旧的单控制器ML系统，它**使用异步调度来匹配多控制器系统的性能**，支持集中的资源管理和调度，对组SPMD加速器计算提供一流的支持，并使用分片数据流系统来高效协调

> 整个Pathways 解决的是 tf1 的数据流模型，怎么做到异构的上千个加速器，解决高延时的瓶颈问题。
>  ![[Pasted image 20220812003227.png]]

## 3. 编程模型 

我们已经实现了用TensorFlow和JAX编写的源程序对Pathways的支持，但在本文中我们将重点放在JAX上进行评估。JAX用户可以显式地用decorator包装标准Python代码，以指示应该编译成(可能是SPMD) XLA计算的片段。这些XLA计算的特点通常是已知的输入和输出类型和形状，有界循环，并且很少(如果有的话)条件(详见附录B)，这使得提前估计计算的资源需求是可行的。我们将这些已知资源需求的计算称为编译函数。每个这样的函数都映射到PATHWAYS程序中的单个(分片)计算节点。

由于JAX程序在多控制器配置下运行，使用XLA集合传输所有数据，因此目前JAX无法扩展到单个TPU吊舱之外，而XLA集合目前只能在TPU上的ICI上使用。**PATHWAYS可以用作JAX后端插件的替代品**，允许JAX代码不加修改地运行，但SPMD计算现在不仅可以访问本地连接的TPU核心，而且可以访问系统中提供的所有核心。由于pathway可以通过ICI和DCN进行通信，这使得JAX程序可以第一次扩展到多个TPU pods，其中包含了数千个TPU核心。

![[Pasted image 20220811183540.png]]
图2。Python用户代码示例，用于跨多个TPU岛运行分片计算

> 前端是JAX

运行未经修改的JAX代码的能力很方便，但不能完全释放PATHWAYS的性能。PATHWAYS 用户可以请求一组虚拟设备，对设备类型、位置或互连拓扑有可选的约束，然后能够在这些设备上放置特定的编译函数(图2)。系统将自动处理所有数据移动和相关计算之间的重分片。

默认情况下，我们将每个编译后的函数转换为只包含一个(分片)计算的独立的 PATHWAYS程序，这意味着如果用户想要背对背运行多个函数，则每个函数都需要一个单独的Python调用和从客户端到协调器的RPC。因此，我们还实现了一个新的程序跟踪器(图2)，用户可以将调用许多编译函数的Python代码块包装起来。**跟踪程序生成一个单独的路径程序，其中每个编译后的函数都由数据流图中的一个计算节点表示。**

JAX支持跟踪代码转换的思想与我们想要探索的研究方向很吻合。例如，JAX有一个名为FLAX的配套库(Heek et al.， 2020)，用于表示分层的DNN模型，我们已经编写了一个库，可以自动将FLAX模型转换为流水线的PATHWAYS程序。此外，JAX还支持对每个示例的Python函数进行向量化的转换，从而产生高效的批处理代码，而这样的转换是探索依赖数据的向量化控制流的新形式的良好基础，我们将在后面(6.3)简要介绍这一点。

## 4. Pathways System Architecture

PATHWAYS广泛地建立在以前的系统上，包括XLA (TensorFlow, 2019)来表示和执行TPU计算、TensorFlow图和执行器来表示和执行分布式CPU计算，以及Python编程框架，包括JAX (Bradbury等人，2018)和TensorFlow api。通过利用这些构建块，我们能够专注于PATHWAYS的新协调方面，同时能够以最小的代码更改运行现有的ML模型

### 4.1 Resource Manager

路径后端由一组加速器组成，这些加速器被分组成紧密耦合的岛屿，**这些岛屿依次通过DCN相互连接**(图3)。
> ![[Pasted image 20220812224955.png]]图中框起来的是岛。集群中设备被分成很多组，每个组包含一批互联成 mesh 拓扑的同质设备，称为一个 island （一般表示一个 POD），一个 island 内部的设备可以通过称为 ICI 的高速连接通信，island 之间则需要通过带宽低一些的DCN（RDMA）来通信。

+ PATHWAYS有一个资源管理器，负责对所有岛屿上的设备进行集中管理。
+ 客户可能要求使用符合其通信模式的特定2D或3D网格形状的岛屿虚拟切片。每个虚拟切片（*virtual slices*）包含虚拟设备，允许客户端表达如何在网格上布局计算。
+ **资源管理器**为满足互连拓扑结构、内存容量等要求的**虚拟设备动态分配物理设备。**

我们最初的资源管理器实现使用了一个简单的启发式方法，它试图通过在所有可用设备上分散计算来静态平衡负载，并在虚拟设备和物理设备之间保持一对一的映射。如果未来的工作负载需要它，我们可以采用更复杂的分配算法，例如考虑所有客户端计算的资源需求和系统的当前状态，以近似地为计算分配物理设备。

> 切片之间甚至会有重叠，涉及到分配就能联想到更复杂的启发式方法进行分配，但是这里作为 future works 了

### 4.2 Client

当用户想要运行一个 traced program（*前面被 `@pw.program`包裹的函数*） 时，它调用PATHWAYS客户端库，它首先分配虚拟设备给以前没有运行过的计算，然后用资源管理器注册计算，触发服务器在后台编译计算。然后客户端为程序构建一个设备位置无关的PATHWAYS间接表示(IR)，表示为自定义MLIR方言(Lattner等人，2021)。IR通过一系列标准的编译器传递逐渐降低，最终输出一个包含物理设备位置的低级表示。这个低级程序考虑到物理设备之间的网络连接，并包括对将源计算碎片的输出传输到目标碎片的位置，包括在需要数据交换时进行分散和收集操作。在虚拟设备位置不变的常见情况下，重复运行低级程序是有效的，如果资源管理器改变了虚拟设备和物理设备之间的映射，可以重新降低程序。
旧的单控制器系统中的客户端可能很快成为性能瓶颈，因为它需要协调数千个单独的计算和数据缓冲区，对应于分布在数千个加速器上的每个计算碎片。PATHWAYS客户端使用一个切分的缓冲区抽象来表示一个可以分布在多个设备上的逻辑缓冲区。这种抽象通过在逻辑缓冲区(而不是单个碎片)的粒度上分摊记账任务(包括引用计数)的成本，从而帮助客户扩展。

### 4.3 Coordination implementation

PATHWAYS依赖于PLAQUE进行所有使用DCN的跨主机协调。
PLAQUE是一个现有的(闭源)生产分片数据流系统，在谷歌用于许多面向客户的服务，其中需要高fanout或高fanin通信，可伸缩性和延迟都很重要。

低级的PATHWAYS IR被直接转换为一个PLAQUE程序，以数据流图的形式表示。PATHWAYS对其协同底物有严格的要求，而所有这些要求都被PLAQUE满足。首先，用于描述PATHWAYS  IR的表示必须包含每个分片计算的单个节点，以确保跨多个分片的计算有一个紧凑的表示，即一个包含N个计算分片的2个计算a和B的链执行应该在数据流表示中有4个节点:Arg Compute(a) Compute(B) Result，无论选择N。在PLAQUE运行时实现中，每个节点生成带有目标分片标签的输出数据元组，因此当执行数据并行执行时，N个数据元组将流动，每个相邻的IR节点对之间有一个数据元组。

![[Pasted image 20220811201423.png]]
图3。PATHWAYS概述。(左)表示为DAG的分布式计算，其中每个节点代表一个单独的编译函数，节点之间的边代表函数之间的数据流。(中)资源管理器为每个编译后的函数分配孤岛加速器的子集(虚拟片)。(右)集中式调度程序用于每个孤岛群调度计算，然后由每个分片执行程序分派。红色箭头表示控制消息，蓝色箭头表示数据路径传输。

协调运行时还必须支持沿着分片边缘的稀疏数据交换，其中消息可以在动态选择的分片子集之间发送，使用标准的进度跟踪机制(Akidau等人，2013;Murray et al.， 2013)来检测一个分片的所有消息何时已经接收。高效的稀疏通信是避免DCN成为加速器上依赖数据的控制流的瓶颈的必要条件，这是我们希望PATHWAYS启用的关键功能之一。

协调基板用于发送DCN消息，这些消息位于传输调度消息和数据句柄的关键路径(图4)，因此它必须以低延迟发送关键消息，并在需要高吞吐量时批量发送到同一主机的消息。

使用可扩展的、通用的数据流引擎来处理DCN通信也很方便，因为这意味着PATHWAYS也可以使用它来完成后台的内部管理任务，例如分发配置信息、监视程序、清理程序、在故障时传递错误等等。

我们认为，使用Ray (Moritz等人，2018)等其他分布式框架而不是PLAQUE来实现底层协调框架，重新实现完整的PATHWAYS设计是可行的。在这样的实现中，PATHWAYS执行器和调度器将被长期运行的Ray actor所取代，它将在底层Ray集群调度之上实现PATHWAYS调度，并且执行器可以使用PyTorch进行GPU计算和集合。为了达到类似的性能(见5)，可能需要增加一些功能，例如，Ray缺乏HBM对象存储，或者通过GPU互连有效地传输远程对象的原语。


### 4.4 Gang-scheduled dynamic dispatch

> **中心化调度**，为避免一些死锁问题，Pathways 通过为每个 island 准备一个中心化的 scheduler 来实现群调度（gang scheduling）。
>  scheduler per island 的方案还是有缺陷的。

如前所述(2)，在一组共享加速器上支持SPMD计算的一个要求是支持高效的分组调度。在PATHWAYS运行时，每个岛都包含一个集中的调度器，它们一致地对岛上的所有计算进行排序。当PATHWAYS将程序排队等待执行时，PLAQUE数据流程序负责
1. 在每个加速器上将本地编译函数的执行排队，以缓冲区期望（地址）作为输入;
2. 通过函数执行对网络发送到远程加速器的缓冲区期货输出进行排队;
3. 与调度器通信，以确定在岛上运行的所有程序的函数执行的一致顺序。调度器必须实现以毫秒为时间尺度分配加速器的策略。我们目前的实现只是按照FIFO的顺序排队，但是更复杂的调度程序可能会根据估计的执行时间重新排序计算。

### 4.5 并行异步分发


![[Pasted image 20220811202757.png]]
图4。三节点程序的串行分发与并行分发。当一个计算在设备上的执行时间比调度、资源分配和协调花费的时间短时，异步管道会因为顺序调度中的主机端工作而停止。并行异步分派通过并行运行主机端工作，利用常规编译函数的静态已知资源使用情况，从而克服了这一瓶颈。为了简洁，省略了调度程序。

>有三个计算任务 A, B, C 分别运行在三个不同的设备 A, B, C 上产和消费关系是 A->B->C。
>1. *host A* 把任务 A 异步地插入设备 A 的任务队列
>2. *host A* 计算（黄方块）之前通知 *host B* 为任务 B 的分配内存，*host B* 把内存地址传给 *host A* （同时在 *host B* 上启动recv），并为启动任务 B 做好必要准备。
>3. 当任务 A 执行结束时，它的输出被直接通过高速连接 ICI 从设备 A 发送到为任务 B 分配好的输入缓冲区，随后 *host B* 可以在设备 B 上启动任务 B。从 B 到 C 的执行过程和从A到B类似
> 弊端：**如果计算时间太短，就像图中所示的那样，异步管道就会停顿，host 上的执行逻辑就会成为执行的关键瓶颈。**
> <font color=#ED7001>解决思路</font>：大部分深度学习都是静态的，下游输入对应的张量形状可以在编译阶段获得，不需要非要等到上游任务进入队列后通知，如图b

当在加速器上运行计算时，系统可以利用异步api使计算与协调重叠(Kwon等人，2020)。考虑图4a中的三节点图，其中的正方形对应三个节点A、B和C，它们运行在主机A、B和C所连接的加速器上。所有节点计算都是常规的编译函数。主机A排队到节点A，接收到A输出的未来数据，并将未来数据传输给主机B。主机B分配B的输入，将输入缓冲区发送给主机A，完成启动节点B功能的大部分准备工作。当节点A完成时，它的输出通过加速器互连直接发送到节点B的输入缓冲区，然后主机B启动节点B。一个节点完成和下一个节点启动之间的延迟可以比数据传输时间多一点。
当前一个节点的计算时间超过调度、资源分配和主机间协调所花费的时间时，上述设计是有效的。但是，**如果计算时间太短，就像图中所示的那样，异步管道就会停止，主机端工作就会成为执行的关键瓶颈，从而减少整个计算序列。**
由于编译的函数都是规则的，因此实际上可以在前一个节点的计算进入队列之前计算后继节点的输入形状。
因此，我们引入了一种新的并行异步分派设计，如图4b所示，它利用常规编译函数的静态已知资源使用情况，并行地运行计算节点的大部分宿主端工作，而不是序列化工作，以便在前面的节点加入队列后才发生由于工作只能在函数是规则的情况下进行并行调度，所以PATHWAYS将并行调度视为**一种优化**，并在一个节点的资源需求直到前一个计算完成后才知道的情况下(例如，由于依赖数据的控制流)，退回到传统的模型。

> 并不是所有情况下都可以这样，所以视为一种优化 

当一个计算的子图可以被静态调度时，程序会向调度器发送一条消息(**描述整个子图)**，调度器能够将子图中所有活动的分片依次执行。使用单个消息的目的是最小化网络流量，但不需要调度器实际将所有子图的分片作为批处理排队:计算仍然可能与其他并发执行程序提交的计算交错。我们在5中评估了不同的调度机制的成本。

![[Pasted image 20220811211509.png]]图5。与TF、JAX和Ray相比，PATHWAYS的分派开销。在所有配置上，PATHWAYS优于TF和Ray等单控制器系统，并与多控制器JAX在fusion (-F)和Chained (-C)配置下的性能，分别适用于多达1000个和256个TPU核心。每个计算包括单个标量AllReduce和一个标量加法。图6。最小的计算来匹配pathway和JAX之间的吞吐量，掩盖单控制器开销。对于配置为128 tpu的16台主机(B)，路径匹配JAX吞吐量为至少2.3 ms，对于配置为2048 tpu的512台主机(a)，路径匹配的计算大小为至少35 ms。

### 4.6 Data Management

每个主机管理一个分片的对象存储，类似于Ray的对象存储(Moritz等人，2018)，但扩展到跟踪每个分片中加速器HBM中持有的缓冲区。客户端程序可以在远程主机或加速内存中保存对对象的引用，客户端和服务器使用不透明的句柄引用它们，从而允许系统在需要时迁移它们。中间程序值也保存在对象存储中，例如当系统等待在加速器之间传输它们，或将它们传递给后续计算时。对象被标记为所有权标签，以便在程序或客户端失败时可以对其进行垃圾收集。如果由于其他计算缓冲区暂时占用HBM而无法分配内存，我们可以使用简单的背压机制（back pressure）来暂停计算。

> 不太看得懂

>一些很重要但 Ray 还没做的事，包括 
>1. object store 支持 HBM 等设备内存管理；
>2. 支持设备间高速数据传输，譬如NVLink 和 RDMA；
>3. 基于 Ray 开发一个数据流引擎进行依赖管理。

>“在深度学习系统里，数据搬运是一等公民，深度学习框架需要像对待数据计算一样重视且全权管理，不能把数据搬运委托给更底层的机制以至于数据搬运隐式地在背后发生，丧失宝贵的确定性和可预测性，必须把数据搬运像计算一样作为算子显式的调度管理。虽然 TensorFlow 没有基于 Ray 开发，但也存在这个问题，感兴趣的朋友可以阅读《**[数据搬运的“诅咒”](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzU5ODY2MTk3Nw%3D%3D%26mid%3D2247485398%26idx%3D1%26sn%3D6db4e248b9fe15e9a2fc3505b497669e%26chksm%3Dfe4189e0c93600f689b945cd6827416bd0aee0f6dfc4ddad9ed23ac2a7e316cd4b5bd5191103%26scene%3D21%23wechat_redirect)**》。多说一句，做 AI 芯片的朋友对这一点体会最深，通用 CPU 都使用 cache 机制，硬件自动完成预取的工作，让软件开发更简单，但仍存在 cache miss rate 的问题，AI 芯片不能承受这个代价，再加上深度学习任务有显著的规律，所以更希望让软件显式的管理缓存何时预取数据，这里普遍使用一种叫 scratchpad 的技术，以实现100%的”命中“。”
>https://zhuanlan.zhihu.com/p/496736889


## 实验  

### 配置 
+ A：4TPUs $\times$ 512hosts，共2048个用ICI连接的TPU
+ B：8TPUs $\times$ 64hosts，共512个TPU 
+ C：使用 4 个岛，每个岛有 4 个 hosts 共 32 个 TPU 

当在GPU上评估Ray时，我们使用Ray v1.3和PyTorch 1.8.1运行在p3.2xlarge VMs1（  1×V100 GPU and 8×CPU cores）上，主机通过DCN连接，使用Amazon放置组调度。

+ 主要将 PATHWAYS 和多控制器的 JAX 进行对比，因为 JAX 的性能目前是 state-of-the-art。
+ 也会和 TF 和 Ray 比较，用来测试分布式系统性能的某些方面，展示运行在PATHWAYS上的TF模型的流水线性能。



# 总结
PATHWAYS匹配最先进的多控制器性能，目前的ML模型是单租户SPMD。我们确保了与多控制器JAX的严格兼容性，正如我们在5中演示的，在非常大的系统规模上，除了最小的计算外，PATHWAYS匹配JAX的性能。

与此同时，PATHWAYS颠覆了JAX程序的执行模型，将用户代码拉回单控制器模型，并在客户端和加速器之间插入一个集中的资源管理和调度框架。单控制器编程模型允许用户简单地访问更丰富的计算模式。资源管理和调度层要求重新引入集群管理策略，包括多租户共享、虚拟化和弹性，所有这些都是为ML工作负载和加速器的需求量身定制的。

我们的微基准测试显示了并发客户机工作负载的交错，以及高效的流水线执行，令人信服地证明了我们所构建的系统机制是快速和灵活的，并为研究利用它们的新策略奠定了坚实的基础。

我们已经证明，仔细的系统设计和工程设计可以让我们获得两个世界的最佳效果，在今天的ML模型上匹配性能，同时交付编写未来模型所需的特性。