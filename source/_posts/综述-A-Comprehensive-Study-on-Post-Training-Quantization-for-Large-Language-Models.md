---
title: >-
  综述: A Comprehensive Study on Post-Training Quantization for Large Language
  Models
date: 2024-02-20 22:23:47
tags:
categories:
    - 模型压缩
---



> 这篇综述对不同的量化方式做了很多实验，对后续有很大帮助

**整篇简短的总结一下**

![image-20240220222305139.png](https://s2.loli.net/2024/02/20/zXWBtZmF8e3UpHD.png)

<!--more-->

paper：

<br>
 
 
{% pdf  ./综述.pdf %} 
 
 
<br>

## 全面分析量化效果

### 敏感性分析

- **INT8仅加权量化**没有任何模型质量影响。对于**INT4仅加权量化**，与相对较小的模型相比，较大的模型通常表现出更好的量化容限。（也就是int4权重量化会较大得影响小模型得效果）
- 与权重量化相比，**激活量化**通常对量化更敏感。较小的模型通常比相对较大的模型具有更好的激活量化性能
- 不同的模型家族表现出完全不同的**INT 8激活量化**行为。特别是对于大型模型，BLOOM-176 B仍然具有有意义的精度（about 1 perplexity, PPL in short, point drop），但OPT-30 B和-66 B的性能要差得多

### 现有PTQ方法分析

- 现有的方法可以显着减少量化误差相比RTN。不同的PTQ方法都有各自的最佳工作场景。
- 对于仅 **INT4 权重**或 **W4A8** （即 INT4 权重和 INT8 激活）量化，当前现有方法几乎无法实现小于 0.1 PPL 点的降级。

### 细粒度量子化效应

- 在细粒度量化的进一步帮助下，PTQ能够在仅加权量化或加权和激活量化的情况下，对于大型模型（> 13 B）实现<0.1 PPL点降级。
- 与较小的模型（e.g., block size 32/64 for OPT-30B）相比，较大的模型可以使用相对粗粒度的权重量化（e.g., block size 128/256 for BLOOM-176B）来实现良好的量化误差。
- 对于 BLOOM-176B，与具有较低位的细粒度量化（e.g., 4 bits with 32 elements as quantization block size）相比，具有较高位（e.g., 5 bits）的粗粒度（per-row）权重量化始终会带来更好的精度 ，即使实际位精度相似。

### 建议量化LLM的设置

- 对于**较大的模型**（> 10 B），可使用细粒度（block size 64–256）**4位权重量化加上8位激活量化**（block size 64–256）和PTQ方法进行真实的部署
- 对于**中等大小的模型**（<10B and >1B），per-row的**INT8权重量化**加上细粒度（block size 64–256）**INT8激活量化**
- 对于**较小的模型**（<1B），直接应用per-row的**W8A8**RTN就足够了

## 总结

- INT8仅权重量化可以用作标准（几乎）无精度降级的方式，以帮助降低LLM的内存成本。

- **小模型的INT 4仅权重量化**导致显著的准确性降低，并且这种影响随着模型大小变大而减小。然而，即使对于较大的模型，准确度降级也可能高于使用较大模型的增益。(e.g., 4-bit asymmetric quantized OPT-30B has worse performance than 8-bit quantized OPT-13B in Table 2.)

![image-20240220175043517.png](https://s2.loli.net/2024/02/20/rqCyxNWA1jczp5M.png)

- **INT8激活**导致小模型的精度下降最小，而较大模型的趋势变得更大。另一个有趣的事情是激活量化灵敏度与模型族高度相关，例如，表3中的BLOOM的结果比表2中的OPT的结果好得多。

- 大模型可以使用4位，小模型小心使用4位

- 在细粒度激活量子化下，**对称和非对称方案的质量退化是相似的**。对于更大的模型（>10B），权重和激活量化与仅权重量化之间的差异可以忽略不计。

- 当块大小为 256 时，细粒度激活量化的好处就会消失。

## PTQ现有方法

比如baseline得RTN，混合精度的LLM.int8()，异常值预变换的smoothquant和AWQ，重新排序的RPTQ，移位缩放处理的Outlier Suppression+，采用离群-受害者对的OliVe，简化流水线的FastGEMM。。。目前现有的方法还是比较多，就不对所有论文一一介绍了

![image-20240220152024713.png](https://s2.loli.net/2024/02/20/fcSX8IEH47zNJ3D.png)

