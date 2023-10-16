---
title: OpenMLSys-计算图
date: 2022-10-01 21:30:27
excerpt: 为啥要做？做完后有何收获感想体会？
tags: 
rating: ⭐
status: inprogress
destination: 
share: false
obsidianUIMode: source
---

https://openmlsys.github.io/chapter_computational_graph

-   计算图是计算框架的核心理念之一，了解主流计算框架的设计思想，有助于深入掌握这一概念，建议阅读 [TensorFlow 设计白皮书](https://arxiv.org/abs/1603.04467)、 [PyTorch计算框架设计论文](https://arxiv.org/abs/1912.01703)、[MindSpore技术白皮书](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/white_paper/MindSpore_white_paperV1.1.pdf)。
    
-   图外控制流直接使用前端语言控制流，熟悉编程语言即可掌握这一方法，而图内控制流则相对较为复杂，建议阅读[TensorFlow控制流](http://download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf)论文。
    
-   动态图和静态图设计理念与实践，建议阅读[TensorFlow Eager 论文](https://arxiv.org/pdf/1903.01855.pdf)、[TensorFlow Eager Execution](https://tensorflow.google.cn/guide/eager?hl=zh-cn)示例、[TensorFlow Graph](https://tensorflow.google.cn/guide/intro_to_graphs?hl=zh-cn)理念与实践、[MindSpore动静态图](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/design/dynamic_graph_and_static_graph.html)概念。