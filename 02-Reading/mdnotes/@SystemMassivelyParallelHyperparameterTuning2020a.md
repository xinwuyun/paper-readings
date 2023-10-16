---
alias: SystemMassivelyParallelHyperparameterTuning2020a
tags: unread HPO
rating: ⭐
share: false
ptype: article
---

# A System for Massively Parallel Hyperparameter Tuning
<cite>* Authors: [[Liam Li]], [[Kevin Jamieson]], [[Afshin Rostamizadeh]], [[Ekaterina Gonina]], [[Moritz Hardt]], [[Benjamin Recht]], [[Ameet Talwalkar]]</cite>


* [Local library](zotero://select/items/1_FF6P72JA)

***

### 初读印象

comment:: 异步 ASHA，SHA 的变体，针对大规模并行超参调优 

![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/24/891534dcbef653eb006f88baa8b2a4d6-20221024105641-cbd3bf.png)

![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/24/905d9140b5ac8d32e8127c2c11c72afb-20221024105650-d7e385.png)

![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/24/8a57ad1e0ed0c46ccc437e3fcff9d757-20221024110359-fc4594.png)

### Downsampling Iterations

![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/24/671dd336900f1ff90ca8f40db72b70ce-20221024112702-c0bb04.png)

Throw out the non-promising iterations.

![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/24/e62cadad8d0f9d68f72223bb39a3b8e3-20221024112735-04a360.png)

### Issues SHA 

![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/24/573dd178dfa66c6857adfe8b77ab1f89-20221024113502-a4d04e.png)

如何安全地提前停止呢？ -> Hyperband

![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/24/61762d85c09bb16e269fb7b1fea53f04-20221024113556-1817c9.png)

### 1st ingredients: Successive Halving (SHA)

![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/24/73a1db95e520102a7973352dfbb167be-20221024115107-bee2b3.png)

Is downsampling "Safe"?

Is the stop too late or too early?

![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/24/83d740791e932b8e8d1d765f0f562d13-20221024115213-801296.png)

### Safe downsampling —— Hyperband 
![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/24/bc08f3609bea6fcc78fcb6087542052e-20221024115646-c2eb7f.png)

![](https://cdn.jsdelivr.net/gh/xinwuyun/pictures@main/2022/10/24/2aaac59b015df54a189acdf48dedd9cd-20221024115842-4fc647.png)

 Where iteration = 5r， random and adaptive methods had only considered 5 configurations, that of hyperband is *256*

## Toolkits

### 分类

一般有两种超参数调优的 toolkit：
+ 开源工具
+ 依赖云资源的服务


