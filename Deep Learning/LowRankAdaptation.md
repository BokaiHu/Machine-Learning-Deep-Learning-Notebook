# Low Rank Adaptation (LoRA)

一种能够减少训练参数量微调大模型的方法。

## Quantization

Computer stores 32-bits float in the following way:

$$
|0|10000001|11100100010101111100000| \\
|sign|exponent|fraction| \\
a = (-1)^s*2^{e-bias}*1.f \\
bias=127
$$

The parameters of the models are usually saved in 32-bits float. We can load the model in 4-bits int, 8-bits int, or 16-bits float to save memory and train faster. This may lose model performance as well.

## LoRA

大模型在针对下游任务微调时不需要更新所有的参数，只需要更新一部分dimension的参数即可 (intrinsic dimension)，假设模型在微调前后的参数之差$\Delta W$也存在这种现象。

### Training

将$\Delta W$低秩分解为两个低秩矩阵$A,\,B$，现在微调的过程可以表示成

$$
W_0+\Delta W=W_0+BA \\
B\in \mathbb{R}^{d\times r},\,A\in\mathbb{R}^{r\times k}
$$

$d,\,k$是大模型参数的矩阵size, $r<<min(d,\,k)$. 为了保证初始值为0，B的初始值全部为0，A从标准高斯分布中随机取样。

### Inference

$$
h=W_0x+\Delta Wx=W_0x+\frac{\alpha}{r}BAx
$$

$\alpha$是scaling factor，用来控制添加LoRA的比例，如果想要使用原始模型->$\alpha=0$. $r$是low-rank matrices的rank，用于scale $BA$.

### Benefits

- Less memory, less training time, less parameters
- Easily applied during inference time
- Can easily load different LoRAs on the same base model

## Code

Transformers peft library (Parameter-Efficient training)
