---
title: MLSys思维导图
date: 2022-08-25 12:29:06
obsidianUIMode: source
---

<!--可使用快捷键`Ctrl+Alt+M`预览思维导图。-->

# MLSys 
## 分布式机器学习
### ML 
1. 单机优化算法
2. 分布式优化算法：同步、异步
3. 分布式机器学习理论
	1. 收敛性分析
	2. 加速比分析 
	3. 泛化分析
	4. 数据与模型聚合
### System
+ 分布式 ML 系统设计
+ 通信机制
+ 计算资源调度
+ Data Management
+ 数据与模型并行（流水线）

## 深度学习模型加速 

### Quantized 

### 深度学习加速器

### 优化 
+ Op-level
+ Graph-level 
+ 针对特定硬件优化
+ 调优方式
	+ 自用优化
	+ 手工优化

## 深度学习框架
+ 算子和 *tensor*  的支持
+ Computational Graph 
+ 自动求导
+ 优化器
+ Operator library
	+ cuDNN
	+ openBLAS
	+ MKL 
+ 深度学习编译器
	+ TVM
	+ XLA
	+ Multi-Level IR

