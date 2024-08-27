# Variational Auto-Encoder

### Intuition

对输入数据通过encoder-decoder结构进行无损编码和解码，通过对隐空间中的值随机采样并且解码生成新的样本。

### Problem

隐空间中的编码数据缺乏结构性。仅仅使用生成内容和原始输入之间的损失最小化无法保证生成有结构性的隐空间，并且可能会导致强烈的过拟合。

### Idea

让encoder返回一个distribution(均值和方差，通常是高斯分布)而不是一个点，在解码时从这个分布中随机抽样还原成原始数据。此时，假如一张满月图片被编码为$G(0.1, 1)$，而月食图片被编码为$G(-0.1, 1)$，如果sample出的点位于0.08，那么这个编码值应该更容易被还原为满月图片，因为这个区域的点很容易在训练目标为满月图片时被sample出来；如果sample出来的点位于0，那么这个编码值可能会被还原为满月和月食中间状态的图片，因为这个点在不同的distribution中都有可能被sample到，因此其训练目标可能是多样的。

### Method

我们希望隐空间具有连续性(相邻点应该呈现相似的内容)和完整性(从隐空间采样的点能够被解码成有意义的输出)，然而仅仅通过让encoder输出分布是不够的，模型可以通过输出极小的方差或者均值差距很大的分布来使得编码结果近似于一个点，因此我们需要额外的正则项来保证模型的输出具有一定的方差并且均值之间相差不远。这里选择了输出分布与标准高斯分布$G(0, I)$之间的Kullback-Leibler散度作为正则化项，确保编码输出的协方差矩阵接近$I$，均值之间相互接近$0$。

### Mathematical Induction

我们用$x$表示原始数据，$z$表示latent variable。

对于训练过程：从分布$P_{Z|X}(z|x)$中采样了一个$z$，然后将$z$通过$P_{X|Z}(x|z)$还原为原始数据$x$。

对于推理过程：从prior $P_Z(z)$中采样一个$z$，然后将$z$通过$P_{X|Z}(x|z)$还原为原始数据$x$。

实际上，我们的encoder由$P_{Z|X}(z|x)$定义，decoder由$P_{X|Z}(x|z)$定义。我们假设先验分布$P_Z(z)$为标准高斯分布$G(0, I)$，$P_{X|Z}(x|z)$也是高斯分布，则根据共轭先验，$P_{Z|X}(z|x)$也应该是高斯分布。

根据贝叶斯定理：

$$
P_{Z|X}(z|x)=\frac{P_{X|Z}(x|z) \cdot P_Z(z)}{\int_u P_{X|Z}(x|u) \cdot P_Z(u)du}
$$

但由于分母中的积分不好计算，所以可以使用变分推理等方式近似。

### Variational Inference

In a group of parametrized distributions, find a best approximation of the target distribution, which can minimize the measurement error (usually KL divergence). Obtain the parameters using gradient descent.

我们使用$q_{Z|X}(z|x)$来估计$P_{Z|X}(z|x)$，假设$q_{Z|X}(z|x)=G(g(x), h(x))$，$p_{X|Z}(x|z)=G(f(z), \sigma^2 I)$。为了方便，后文将不再标注概率分布中的脚标。我们的目标是找到

$$
\begin{aligned}
	g^*(x), h^*(x)&=argmin_{g, h}KL(q(z|x), P(z|x))\\
	&=argmin_{g, h}[q(z|x)\cdot log(\frac{q(z|x)}{P(z|x)})] \\
	&=argmin_{g, h}\{E_{z\sim q}[log(q(z|x))] - E_{z\sim q}[log(P(z|x))]\} \\
	&=argmin_{g, h}\{E_{z\sim q}[log(q(z|x))] - E_{z\sim q}[log(P(x|z))] - E_{z\sim q}[log(P(z))]\} \\
	&=argmax_{g, h}\{E_{z\sim q}[log(P(x|z))] - KL(q(z|x), P(z))\} \\
	&=argmax_{g, h}\{E_{z\sim q}[-\frac{(x-f(z))^2}{2\sigma^2}] - KL(q(z|x), P(z))\} \\
\end{aligned}
$$

$E_{z\sim q}[log(P(x|z))] - KL(q(z|x), P(z))$ 该式子正是$logP_X(x)$的estimate lower bound (ELBO), 可以使用EM算法最大化ELBO。

Encoder的目标：

$$
\begin{aligned}
	g^*(x), h^*(x)&=argmin_{g, h}KL(q(z|x), P(z|x))\\
	&=argmin_{g, h}\{E_{z\sim q}[log(P(x))] - ELBO\} \\
	&=argmax_{g, h}ELBO
\end{aligned}
$$

Decoder的目标：

$$
\begin{aligned}
f^*(x)&=argmax_f\{E_{z\sim q^*}[-\frac{(x-f(z))^2}{2\sigma^2}] - KL(q(z|x), P(z))\} \\
&=argmax_fELBO
\end{aligned}
$$

最终的$f, g, h$可以在同一个框架中共同优化：

$$
f^*(x), g^*(x), h^*(x)=argmax_{f, g, h}\{E_{z\sim q}[-\frac{(x-f(z))^2}{2\sigma^2}] - KL(q(z|x), P(z))\}
$$

可以看出最终的objective的第一部分是在最大化$P(X|Z)$对数似然，第二部分是为了减小$q(z|x)$与先验之间的KL散度 (正则化项)。

### Introducing Neural Network

我们用一个神经网络来拟合编码器分布$P_{Z|X}(z|x)$，当输入一个x时，编码器网络会输出对应分布的均值和方差，我们用该分布与标准高斯分布的KL散度作为正则项进行约束。此处获得$z$时使用了重参数化技巧使得梯度能够反向传播到编码器。(Sampling from $a\sim G(0, I)$, 计算$z=\sigma a+\mu$.)

我们假设解码器分布$P_{X|Z}(x|z)$具有恒定的协方差矩阵，使用神经网络对该分布的均值$f(z)$进行拟合，并且通过最小化均方误差$||x-f(z)||^2$的方法进行优化。

对模型做出了一些假设，包括：

- 编码器网络输出的协方差矩阵为对角矩阵(变量相互独立)，从而将其变为一维向量。
- 解码器网络只拟合均值 (降低复杂度；提高稳定性；提高计算效率；From GPT4: 如果解码器同时模拟均值和方差，模型可能会倾向于忽略潜在变量$z$中的信息，仅通过调整方差来最小化重建误差。这种现象称为“后验坍塌”，它会使得潜在空间的表示能力减弱。)

# Vector Quantized-Variational Autoencoder

### Intuition

将图片等内容编码为离散向量感觉更加合理，因为是否为老人，性别，年龄等都是离散的。

### Method

其实VQ-VAE并不是VAE，而是AE，它直接将原图片$x$映射到一个连续变量$z$，然后在codebook中寻找$z$的最近邻$z_q$并将$z$映射到$z_q$，然后将$z_q$解码输出图片。

对于AE，优化目标为：

$$
||x-decoder(encoder(x))||^2
$$

但在VQ-VAE中，我们实际上是对$z_q$解码，所以优化目标好像是：

$$
||x-decoder(z_q)||^2
$$

两个都考虑，使用Straight-through estimation:

$$
||x-decoder(encoder(x) + stop\_gradient[z_q - encoder(x)])||^2
$$

stop_gradient的意义是：

$$
stop\_gradient(x)=\left\{
\begin{array}{rcl}
x       &      & {forward}\\
0     &      & {backward}
\end{array} \right.
$$

为了维护编码表，我们可以在目标函数中添加额外的项$||encoder(x)-z_q||^2$。

另外$encoder(x)$需要在反向传播时维持重构图片的效果，而编码表$z_q$是更加自由的，所以我们希望$z_q$能够更靠近$encoder(x)$.因此在目标函数修改为

$$
||x-decoder(encoder(x) + sg[z_q - encoder(x)])||^2+\gamma||sg[z_q]-encoder(x)||^2+\beta||z_q-sg[encoder(x)]||^2
$$

$\gamma < \beta$, 原论文中$\gamma=0.25\beta$.
