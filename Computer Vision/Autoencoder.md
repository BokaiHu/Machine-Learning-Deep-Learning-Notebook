# Auto-encoder

用于对无标签数据进行压缩，学习这些数据的低维embedding. Encoder将高维输入转化为一个维数较低的latent vector，而decoder尽可能将这个低维latent vector转换回原数据。

Objective: minimizing the reconstruction loss

$$
loss=||x-\hat{x}||_2=||x-d_{\theta}(e_\phi(x))||_2
$$

## Assumption

- 假设存在一个低维空间，将原数据映射到该低维空间上能够达到最大压缩，当维度继续降低时会造成大量信息损失。和主成分分析Principle Component Analysis很像，但Auto-encoder的优势在于能够通过非线性的神经网络学习到更加复杂的低维数据表示。

## Note

- 如之前在Variational Auto-encoder中提到的，Auto-encoder不能被用于生成。这是因为Auto-encoder的隐空间是不规则的，即只存在少量区域拥有对应的数据点，如果我们从隐空间进行采样，生成的结果大概率是无效输出。

## Some Tips

* Feature Disentanglement: 我们假设encoder输出的latent vector包括输入的不同方面的信息，例如对于一段输入的语音，latent vector可能包含音色，内容等等。[1904.05742.pdf (arxiv.org)](https://arxiv.org/pdf/1904.05742.pdf)；[1804.02812.pdf (arxiv.org)](https://arxiv.org/pdf/1804.02812.pdf)；[1905.05879.pdf (arxiv.org)](https://arxiv.org/pdf/1905.05879.pdf).
* Latent Representations: 可以考虑约束模型输出不同类型的latent vector，例如用两个seq2seq做encoder和decoder，第一个seq2seq产生的sequence就是latent vector。但这个sequence转换成人类可识别的类型(语音、文本等)可能是人类无法理解的，但是decoder可以理解。为了让这里的sequence产生有效的输出，可以额外添加一个discriminator来做监督(分辨哪个是人类的输入哪个是机器的latent vector). [1810.02851.pdf (arxiv.org)](https://arxiv.org/pdf/1810.02851.pdf)
* Anomaly detection: 用来检测新的sample是否和training data一致，如果一致的话reconstruction loss应该不会很大，如果不一致会有较大的loss.
