---
alias: MLSysNewFrontierMachineLearningSystems
tags: MLSys AI system overview
rating: ⭐
share: false
ptype: article
---

# MLSys: The New Frontier of Machine Learning Systems
<cite>* Authors: [[Alexander Ratner]], [[Dan Alistarh]], [[Gustavo Alonso]], [[David G. Andersen]], [[Peter Bailis]], [[Sarah Bird]], [[Nicholas Carlini]], [[Bryan Catanzaro]], [[Jennifer Chayes]], [[Eric Chung]], [[Bill Dally]], [[Jeff Dean]], [[Inderjit S. Dhillon]], [[Alexandros Dimakis]], [[Pradeep Dubey]], [[Charles Elkan]], [[Grigori Fursin]], [[Gregory R. Ganger]], [[Lise Getoor]], [[Phillip B. Gibbons]], [[Garth A. Gibson]], [[Joseph E. Gonzalez]], [[Justin Gottschlich]], [[Song Han]], [[Kim Hazelwood]], [[Furong Huang]], [[Martin Jaggi]], [[Kevin Jamieson]], [[Michael I. Jordan]], [[Gauri Joshi]], [[Rania Khalaf]], [[Jason Knight]], [[Jakub Konečný]], [[Tim Kraska]], [[Arun Kumar]], [[Anastasios Kyrillidis]], [[Aparna Lakshmiratan]], [[Jing Li]], [[Samuel Madden]], [[H. Brendan McMahan]], [[Erik Meijer]], [[Ioannis Mitliagkas]], [[Rajat Monga]], [[Derek Murray]], [[Kunle Olukotun]], [[Dimitris Papailiopoulos]], [[Gennady Pekhimenko]], [[Theodoros Rekatsinas]], [[Afshin Rostamizadeh]], [[Christopher Ré]], [[Christopher De Sa]], [[Hanie Sedghi]], [[Siddhartha Sen]], [[Virginia Smith]], [[Alex Smola]], [[Dawn Song]], [[Evan Sparks]], [[Ion Stoica]], [[Vivienne Sze]], [[Madeleine Udell]], [[Joaquin Vanschoren]], [[Shivaram Venkataraman]], [[Rashmi Vinayak]], [[Markus Weimer]], [[Andrew Gordon Wilson]], [[Eric Xing]], [[Matei Zaharia]], [[Ce Zhang]], [[Ameet Talwalkar]]</cite>


* [Local library](zotero://select/items/1_UMKLJM9F)

***

### 初读印象

comment:: (AI system白皮书)MLSys: 机器学习系统的新前沿

## 摘要

当前，机器学习（ML）技术得到广泛应用 ，但是实际部署中支撑 ML 模型的系统是一个重大障碍，这很大原因是 
1. radically different development and deployment profile of modern ML methods. 
2. the range of **pratical concerns** that come with broader adoption<font color=#77777> (of machine learning)</font>

因此，作者建议在**传统系统和机器学习**研究社区的交汇处建立一个新的 system machine learning research community，专注于诸如
+ 用于机器学习的硬件系统 
+ 用于机器学习的软件系统 
+ 用于预测精度以外指标的优化的机器学习 

这就是 MLSys

## Introduction

### 发展趋势
+ 大公司投资数十亿美元，把自己重塑为人工智能中心的企业
+ 众多学科将机器学习纳入他们的研究中
+ 公众对人工智能也更加兴奋

这是由几个因素造成的，
+ 主要是新的深度学习方法
+ 不断增加的数据和计算资源
+ 开源框架**有效地将模型设计和规范从系统中解耦**
	+ Caffe
	+ Theano
	+ MXNet
	+ TensorFlow
	+ PyTorch

### 挑战

不幸的是，***System***变成了瓶颈。
+ 基于 ML 的应用程序需要新型的软硬件和工程系统 
+ 它们越来越以不同于传统软件的方式开发，例如，通过收集、预处理、标记和重塑训练数据集而不是编写代码
+ 也以不同的方式部署，例如，利用专门的硬件、新型的质量保证方法和新的端到端工作流程。
+ 这一转变为ML开发的高级接口、执行ML模型的低级系统以及在传统计算机系统代码中嵌入learned component的接口带来了令人兴奋的研究挑战和机会。

现代的ML方法还需要新的解决方案来解决随着这些技术在**不同的现实环境**中得到更广泛的使用而自然产生的一系列问题。包括小型和大型组织的成本和其他效率指标...。ML 用户在增加，但是不是所有人有 ML 的博士学位。

#TODO "Modern ML approaches also require new solutions for the s" 这一段没太看懂

各种各样的现状指出，需要对机器学习的系统进行的一致的研究。
MLSys 将重点放在广泛的、全栈的问题上，包括：
1. 应该如何设计 *software system* 来支持机器学习的整个生命周期，从程序设定接口和数据预处理到输出解释、调试和监控?
> 大概是如何设计框架
3. 如何为机器学习设计 *hardware system* ?
> 专门的、异构的硬件，专门用来训练和部署机器学习模型。
> 如何进行 trade-off。
> 如何设计分布式应用
5. 应当如何设计机器学习系统来满足预测准确性之外的指标，比如功耗和内存效率、可访问性、成本、延迟、隐私、安全性、公平性和可解释性

划分这些研究主题的另一种方法是将这些研究主题划分为

1. 支持ML开发的接口和工作流的**高级ML系统**(类似于编程语言和软件工程方面的传统工作)
2. 和涉及硬件或软件的**低级ML系统**(通常模糊两者之间的界限以支持模型的训练和执行)，类似于编译器和体系结构方面的传统工作。

考虑到它们的全栈特性，我们认为它们最好的答案是一个研究社区，混合了传统机器学习和系统社区的观点。

...

最后，我们认为系统机器学习社区是一个理想的起点，甚至更大范围和更广泛的问题，超越了如何与单一模型[4]接口，训练，执行或评估。例如，
+ 我们如何管理以复杂方式相互作用的模型的整个生态系统?我们如何维护和评估追求长期目标的系统?
+ 我们如何衡量ML系统对社会、市场等方面的影响?
+ 我们如何在社会范围内共享和重用数据和模型，同时维护隐私和其他经济、社会和法律问题?所有这些问题

## Why Now? The Rise of Full Stack Bottlenecks in ML

机器学习模型的性能突飞猛进，是时候将机器学习的承诺在现实生活中兑现了，为了兑现这个承诺我们需要 Full stack 上的各种新系统。

实际部署中，有很多问题，一系列平静开始浮现出水面，关键是全栈、系统级问题，无关核心算法的特性，包括
+ **Deployment concerns** ：对抗性影响或其他虚假因素的稳健性;更广泛地考虑安全性;隐私和安全，特别是敏感数据越来越多地被使用;可解释性，在法律上和操作上越来越需要;公平，因为ML算法开始对我们的日常生活产生重大影响;还有许多其他类似的担忧。
+ **Cost**：训练甚至数据处理的成本都很大
+ **Accessibility**：越来越多的人希望将 ML 用到实际生产中，我们希望即便不是机器学习和系统方向的专家也可以上手 ML 系统。

共同的痛点是，他们是全栈问题，不仅需要在核心 ML 算法上进行推理，还需要在硬件、软件和支持他们的整体系统。

## MLSys: Building a New Conference at the Intersection of Systems +  Machine Learning

这一节再次强调了建立该会议的原因。委员会的人来自 system 和 ML 两个领域。

## Conclusion

在传统机器学习和系统社区的交汇处，有一系列令人难以置信的令人兴奋的研究挑战可以被独特地解决，无论是今天还是未来。解决这些挑战将需要**理论、算法、软件和硬件**的进步，并将带来令人兴奋的新的执行 ML 算法的 low-level 系统，用于指定、监控和与它们交互的高级系统，除此之外，新的范式和框架将塑造机器学习**如何与整个社会交互。**

