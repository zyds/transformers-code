![手把手带你实战Transformers](./imgs/1.png)

# 简介

手把手带你实战Transformers课程的代码仓库

## 代码适配

- transformers==4.36.2

# 课程规划

- 基础入门篇：Transformers入门，从环境安装到各个基础组件的介绍，包括Pipeline、Tokenizer、Model、Datasets、Evaluate、Trainer，并通过一个最基本的文本分类实例将各个模块进行串讲

- 实战演练篇：Transformers实战，通过丰富的实战案例对Transformers在NLP任务中的解决方案进行介绍，包括命名实体识别、机器阅读理解、多项选择、文本相似度、检索式对话机器人、掩码语言模型、因果语言模型、摘要生成、生成式对话机器人

- 高效微调篇：Transformers模型高效微调，以PEFT库为核心，介绍各种常用的参数高效微调方法的原理与实战，包括BitFit、Prompt-tuning、P-tuning、Prefix-Tuning、Lora和IA3

- 低精度训练篇：Transformers模型低精度训练，基于bitsandbytes库，进行模型的低精度训练，包括LlaMA2-7B和ChatGLM2-6B两个模型的多个不同精度训练的实战演练，包括半精度训练、8bit训练、4bit训练（QLoRA）

- 分布式训练篇：Transformers模型分布式训练，基于accelerate库讲解transformers模型的分布式训练解决方案，介绍分布式训练的基本原理以及accelerate库的基本使用方式，包括与Deepspeed框架的集成

- 对齐训练篇: ...

- 性能优化篇: ...

- 系统演示篇: ...


# 课程地址

课程视频发布在B站与YouTube，代码与视频会逐步进行更新，目前课程主要更新在B站，YouTube后续会持续更新

- [Bilibili](https://www.bilibili.com/video/BV1ma4y1g791)

- [YouTube](https://www.youtube.com/@lunatic-zzz)

## Transformers 基础入门篇 (已更新完成)

- 01- 基础知识与环境安装

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1ma4y1g791) | [YouTube](https://www.youtube.com/watch?v=ddCfxkCh-O8)

- 02 基础组件之 Pipeline | 

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1ta4y1g7bq) | [YouTube](https://www.youtube.com/watch?v=Xeu3qFTP9qY&t=7s)

- 03 基础组件之 Tokenizer

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1NX4y1177c) | [YouTube](https://www.youtube.com/watch?v=G4JmQu-VWrU)

- 04 基础组件之 Model(上) 基本使用

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1KM4y1q7Js) | [YouTube](https://www.youtube.com/watch?v=xK-6VcLqa94)

- 04 基础组件之 Model(下) BERT文本分类代码实例

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV18T411t7h6) | [YouTube](https://www.youtube.com/watch?v=nkwOQQDCDvc)

- 05 基础组件之 Datasets

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1Ph4y1b76w) | [YouTube](https://www.youtube.com/watch?v=LRhcUjbSOEk)

- 06 基础组件之 Evaluate

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1uk4y1W7tK) | [YouTube](https://www.youtube.com/watch?v=tpE2bleqk6A)

- 07 基础组件之 Trainer

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1KX4y1a7Jk) | [YouTube](https://www.youtube.com/watch?v=YzS-BvHeSGE)

## Transformers 实战演练篇 (已更新完成)

- 08 基于 Transformers的 NLP解决方案

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV18N411C71F) | [YouTube](https://www.youtube.com/watch?v=WRBPd86T1Fc)

- 09 实战演练之 命名实体识别
   
   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1gW4y197CT) | [YouTube](https://www.youtube.com/watch?v=3xQR-7sly_I)

- 10 实战演练之 机器阅读理解（上，过长截断策略）
   
   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1rs4y1k7FX) | [YouTube](https://www.youtube.com/watch?v=-rzKZIpELOk)

- 10 实战演练之 机器阅读理解（下，滑动窗口策略）

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1uN411D7oy) | [YouTube](https://www.youtube.com/watch?v=oTlpbISOkaE)

- 11 实战演练之 多项选择 

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1FM4y1E77w) | [YouTube](https://www.youtube.com/watch?v=xHM1PjIihJs)

- 12 实战演练之 文本相似度（上，基于交互策略） 

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1Tm4y1J7EF) | [YouTube](https://www.youtube.com/watch?v=SElN5_LqZls)

- 12 实战演练之 文本相似度（下，基于匹配策略） 

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV13P411C7UD) | [YouTube](https://www.youtube.com/watch?v=7zxNXBBDqwA)

- 13 实战演练之 检索式对话机器人

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1Lh4y117KJ) | [YouTube](https://www.youtube.com/watch?v=gHOUoqqXb8I)

- 14 实战演练之 预训练模型

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1B44y1c7x2) | [YouTube](https://www.youtube.com/watch?v=jHRo2qgtE7Y)

- 15 实战演练篇之 文本摘要（上，基于T5模型）

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1Kp4y137ar) | [YouTube](https://www.youtube.com/watch?v=5AusJJbpWaA)

- 15 实战演练篇之 文本摘要（下，基于GLM模型）

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1CF411y7hw) | [YouTube](https://www.youtube.com/watch?v=BK2wUNZZbRg)

- 16 实战演练篇之 生成式对话机器人（基于Bloom）

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV11r4y197Ht) | [YouTube](https://www.youtube.com/watch?v=McE0XUG5Gw4)

## Transformers 参数高效微调篇 (已更新完成)

- 17 参数高效微调与BitFit实战

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1Xu4y1k7Ls) | [YouTube](https://www.youtube.com/watch?v=ynBE40yVTSk)

- 18 Prompt-Tuning 原理与实战

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1Fu4y1C7tJ) | [YouTube](https://www.youtube.com/watch?v=aAbVsm6tWIM)

- 19 P-Tuning 原理与实战

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV17V411N7Ld) | [YouTube](https://www.youtube.com/watch?v=xNC12IhNuw4)

- 20 Prefix-Tuning 原理与实战

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1Ru411g7Qa) | [YouTube](https://www.youtube.com/watch?v=EYd-sJHXCio)

- 21 LoRA 原理与实战

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV13w411y7fq) | [YouTube](https://www.youtube.com/watch?v=-xVJtu9pyoA)

- 22 IA3 原理与实战

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1Y8411k7yD) | [YouTube](https://www.youtube.com/watch?v=WOrHqOkMqxY)

- 23 PEFT 进阶操作

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1YH4y1o7rg) | [YouTube](https://www.youtube.com/watch?v=KJljAinRXs8)
   

## Transformers 低精度训练篇

- 24 低精度训练与模型下载

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1y34y1M7t1) | [YouTube](https://www.youtube.com/watch?v=mWiXtVs9ZzY)

- 25 半精度模型训练（上，基于LLaMA2的半精度模型训练）

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1CB4y1R78v) | [YouTube](https://www.youtube.com/watch?v=Is4T8u1Astk)

- 25 半精度模型训练（下，基于ChatGLM3的半精度模型训练）

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1aw411M7Cv) | [YouTube](https://www.youtube.com/watch?v=8SmlpNuY_pU)

- 26 量化与8bit模型训练

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1EN411g7Yn) | [YouTube](https://www.youtube.com/watch?v=XKImkaWv7-Y)

- 27 4bit量化与QLoRA模型训练

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1DQ4y1t7e8) | [YouTube](https://www.youtube.com/watch?v=CY0jTExZlKE)

## Transformers 分布式训练篇

- 28 分布式训练基础与环境配置

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1cK4y1z7Mv) | [YouTube](https://www.youtube.com/watch?v=eNOoIlUCX6Q)

## Transformers 番外技能篇

- 基于Optuna的Transformers模型自动调参

   - 视频地址：[Bilibili](https://www.bilibili.com/video/BV1NN4y1S7i8) | [YouTube](https://www.youtube.com/watch?v=ugiAW2ukZZw)

# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=zyds/transformers-code&type=Date)](https://star-history.com/#zyds/transformers-code&Date)


# 请作者喝杯奶茶

![](./imgs/wx.jpg)