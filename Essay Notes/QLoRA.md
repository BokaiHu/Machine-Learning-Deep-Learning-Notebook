# Quantized Low-Rank Adaptation (QLoRA)

[[2305.14314] QLoRA: Efficient Finetuning of Quantized LLMs (arxiv.org)](https://arxiv.org/abs/2305.14314 "QLoRA")

## Contribution:

1. NormalFloat4:
   信息理论上最优的数据类型。因为pretrained model的weights基本都服从一个高斯分布$N(0, \sigma)$。使用NormalFloat4进行量化分为三步

   1. 使用norm.ppf计算$N(0, 1)$的$2^k+1$个quantile
   2. 把这些quantiles用absmax映射到[-1, 1]之间。
   3. 将weights的值用absmax映射到[-1, 1]之间，然后寻找对应的quantile.

   假如这里$k=4$, 为了fully utilize这$2^4=16$个数字，我们需要把[-1, 1]这个区间分成16份(按照quantile)，这样每个数落在哪个区间内就会被映射到哪个值上，但这样是**取不到0**的。为了使值能够被映射到0，作者使用了一个asymetric的映射方式，在负半轴上有$2^{k-1}$个点，在正半轴上有$2^{k-1}+1$个点，然后再删去一个共同的0得到$2^k$个点。此外因为无法从0或1开始计算quantile，所以需要一个**offset**，即计算的区间为**[1-offset, offset]**. 原文中使用的offset为$\frac{1}{2}(1-\frac{1}{2*15}+1-\frac{1}{2*16})=0.9677083$, 我的理解是考虑负半轴时有15个半(不理解什么是半，也不理解为什么是15个)，按照这种方式拓展到全数轴时应该是2\*15个，考虑正半轴时有16个半，拓展到全数轴时是2\*16个。$1-\frac{1}{2*15}$相当于找出了最右边分位数的值，$1-\frac{1}{2*16}$同理，然后对两个值求均值保证左右两侧面积相等。
   Quantize之后的值应该是Weight: 保存quantize值的index; index用于在cookbook里查找对应的值。

   **为什么要使用NormalFloatk的表示方法：这个方法确保落在每个区间内的weights的个数是相同的，如果等距切分根据高斯分布的性质，大量的点会聚集在0附近的quantize值。**
2. Double Quantization: 对32bits的quantization constant再做一次quantization，这里使用block size=256的8-bit float quantization，因为作者观察到这种方式没有performance degradation。原先对于一个block size=64的32bits quantization constant，相当于每个参数额外加了$\frac{32}{64}=0.5$bit。现在是$\frac{8}{64}+\frac{32}{64*256}=0.127$.因为每256个(64block)才做一次double quantization，所以是$\frac{32}{64*256}$。
