---
tags: [LoRA, MTX, Stable Diffusion]
title: LoRA of Stable Diffusion
created: '2023-07-07T08:34:23.283Z'
modified: '2023-07-08T07:25:34.395Z'
---

# LoRA of Stable Diffusion

## Attention

导师让我关注背后知识，于是我决定先看看 Attention 是什么。

于是找到了 [号称史上最直白的 Attention 讲解](https://blog.csdn.net/weixin_44048622/article/details/126906918)。直观地理解 Attention 很像人类看一张图片时的逻辑，比如我们看到一张图片，往往不会看清全部内容，而是将注意力集中在图像焦点上，而类似地，在深度学习中 Attention 机制就是通过矩阵运算的方式将模型的注意力集中在输入信息的重点特征上，从而节省资源并快速获取最有效的信息。

这玩意一般用来处理 **seq2seq** 问题，大概是由一个序列到另一个序列，如机翻。在 Attention 出现之前，人们常用**多个 RNN 网络**来处理这类问题，但其中有一个严重缺陷，即假如序列较长则会丢失较远时间的信息，会丢失很多细节信息，而假如引入了Attention机制，则会有一个重要程度的分布。

那么Attention机制是怎么实现对于不同信息的重要性判断的呢？在 self-attention 中，会有三种矩阵向量，即 $Q$ (Query) 查询向量，$K$ (Key) 键值向量，$V$ (Value) 值向量。（在传统 Attention 中 Q 来自外部，而 self-attention 中 $Q$ 来自内部）。在进行 Attention 运算时，首先把当前单词产生的 $Q$ 和所有的 $K$ 做矩阵乘法得到一个中间结果，最后把自己的 $V$ 向量乘上这个中间结果矩阵，得到一个含有句子所有词语上下文信息的新向量。~~这一段没太懂，反正知道 QKV 是啥就行了（bushi~~

## LoRA

然后来看 [LoRA 正文](https://huggingface.co/blog/zh/lora)，这玩意可以对大型模型进行微调，只需要较小开销就可以对数十亿以上参数的大模型进行微调。自然而然地，这个东西也可以用于 Stable Diffusion，LoRA 可以应用于**将图像表示与描述它们的提示相关联**的交叉关注层（Cross-Attention Layer，~~原来 cross attention 就是用来搞图像描述的啊~~。

[![latent-diffusion.png](https://img1.imgtp.com/2023/07/08/MRNq9z6K.png)](https://img1.imgtp.com/2023/07/08/MRNq9z6K.png)

这张图里面 Latent Space 中那几个黄色的 $QKV$ 就是 Cross-Attention 层，它接收 Conditioning 中处理过的图像生成条件（如 Text Prompt, Images, Semantic Map 等）作为输入，让图文之间建立联系。

在使用了 LoRA 之后，对时间及空间的占用起到了巨大的优化作用，而且训练模型也小了很多，总之就是非常地牛逼。它可以在不修改 SD 的前提下，利用少数训练数据训练出某一种画风的作品来实现个性化需求。用数据公式来表示 LoRA 如下：

$W=W_0+BA$

什么意思呢？其中 $W_0$ 就是**初试 SD 参数（Weights）**，$BA$ 为低秩矩阵（low rank matrics）也就是 **LoRA 模型的参数**，$W$ 代表**被 LoRA 模型影响后的最终 SD 模型参数**。这是一个线性模型，相当于直接在原 SD 模型上叠加了一个 LoRA 模型得到一个新模型。

## 关于应用与训练

关于怎么用 LoRA 模型，确实是非常地简单啊，直接把 LoRA 模型放到 SD 的模型库里，然后在 prompt 里加上 `<lora:filename:multiplier>` 就完事了（这里的 `multiplier` 指的是 LoRA 的强度）。

```
prompt_part_1, prompt_part_2, (...), prompt_end, <lora:a_lora_model:0.7>
```

是的，就是这么简单。

那么自己训练 LoRA 呢？这个项目 [kohya_ss](https://github.com/bmaltais/kohya_ss) 它提供了 GUI 界面，非常的人性化。先找几张训练图片（一般使用$50$张以上效果比较好？不过当然是越多越好辣），然后在这个软件里给你训练的东西加 tag，在 prompt 里面加上这些 tag 才能正确生成这些效果，然后命名模型，开始训练，就行了。

~~貌似 LoRA 就这么些东西，但是貌似还是没搞懂底层原理诶（~~

## 补充学习

不太行得再看看（




