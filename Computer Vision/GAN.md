# Generative Adversarial Network

### Idea

GAN由两个神经网络组成，分别是Discriminator和Generator。Generator的作用是从一个simple distribution中sample出一个样本，并将其转化成一个复杂分布中的样本；Discriminator的作用是分辨一个样本是由Generator生成的还是从复杂样本中sample出来的。

### Objective

Discriminator:

$$
D^*=argmax_DV(D, G) \\
V(D, G)=E_{y\sim p_{data}}[log(D(y))]+E_{y\sim P_G}[log(1 - D(y))]
$$

我们希望Discriminator对于来自真实数据的输出越大越好，来自生成数据的输出越小越好，其实Discriminator就是一个二元分类器，训练目标即为maximize negative cross entropy loss. $V(D, G)$和JS散度相关，因此对于Generator

$$
\begin{aligned}
	G^*&=argmin_GDiv(P_{data}, P_G) \\
	&=argmin_Gargmax_DV(D, G)
\end{aligned}
$$

所以GAN的优化步骤如下：

1. Fix generator G, update discriminator D.
2. Fix discriminator D, update generator G.

### Problem

1. The overlap between $P_{data}$ and $P_G$ is usually neglegible.
2. JS divergence is always $log2$ if two distributions do not overlap, therefore the loss does not provide useful guidance for model optimization.

# Wasserstein-GAN

Wasserstein-GAN (wGAN), using Wasserstein distance (Earth moving distance?) as the objective function. Different from JS divergence, Wasserstein distance can provide useful information even though the distributions do not overlap.

Modified objective function using Wasserstein distance

$$
\begin{aligned}
	argmax_{D\sim 1-Lipschitz}\{E_{x\sim p_{data}}[log(D(x))]+E_{x\sim P_G}[log(D(x))]\}
\end{aligned}
$$

Three optimization methods:

1. Weight Clip
   Clip the parameters of the discriminator between $c$ and $-c$. If $w>c\rightarrow w=c$, if $w<-c\rightarrow w=-c$. Does not really ensure $1-Lipschitz$.
2. Gradient Penalty
   Interpolate point $\hat{x}$ between $x$ and $G(z)$, 使判别器相对这个插值点的梯度趋近于1，在训练目标中添加以下正则项来直接约束梯度。

$$
   (||\nabla_{\hat{x}}D(\hat{x})||_2-1)^2)
$$
3. Spectral Normalization

   每次迭代时根据discriminator参数的最大奇异值对参数进行归一化

$$
   W_{norm}=\frac{W}{\sigma(W)}
$$

# Conditional-GAN

此时generator G的输入不仅是Gaussian distribution sample出的噪音$z$，还包括另一个条件向量$t$，该向量可以是由seq2seq模型获得的text vector等等。Discriminator D也要将条件$t$作为输入，判断生成图片质量如何，是否与condition相对应。

### Applications

Text-to-image, image-to-image (pix2pix), sound-to-image...

# Cycle GAN

要实现learning from unpaired data，例如将人像$x_1$转换成二次元图像$x_2$，这时我们大概率没有所有的人像图片对应的二次元图片，图片之间没有对应关系。

最简单的想法：直接将人像作为generator $G_{x1\rightarrow x2}$的输入$x_1$生成一个结果$\hat{x_2}$，然后使用discriminator $D_1$对$\hat{x_2}$和$x_2$进行学习。但此时的一个问题在于generator学习不到人像与对应的二次元图像之间的关系，它很有可能会随机生成一个和人像无关的二次元图像，这点和conditional-GAN很像，但此时我们没有办法用condition来约束discriminator的分辨学习。

更进一步：额外添加一个reconstructor $G_{x2\rightarrow x1}$，将$G_{x1\rightarrow x2}$生成的$\hat{x_2}$还原为$x_1$。此时可以进行两组训练，一组是依次经过$G_{x_1\rightarrow x_2}、G_{x_2\rightarrow x_1}$和$D_1$，$x_1\rightarrow \hat{x_2} \rightarrow x_1$，$D(x_2, \hat{x_2})$；另一组是依次经过$G_{x_2\rightarrow x_1}、G_{x_1\rightarrow x_2}$和$D_2$，$x_2\rightarrow \hat{x_1} \rightarrow x_2$，$D(x_1, \hat{x_1})$

主要思想就是$G_{x_1\rightarrow x_2}$用来生成，$D$保证生成质量，$G_{x_2\rightarrow x_1}$保证生成内容与原图相关，通过这种循环的训练保证内容一致。

### Application

风格迁移

# Evaluation Metrics

### Inception Score (IS)

生成的单张图片用一个Inception Net分类，所有图片的类别做平均，如果每个类非常接近，说明生成的图片有一定的diversity。单张图片用Inception Net分类，如果某类的值特别高，说明quality很好；如果各类的值比较平均，说明quality一般。Good quality + large diversity = large IS.

### Frechet Inception Distance (FID)

一张图片扔进Inception Net中，取最后一个encoder输出的hidden latent vector代表一张图片。假设这个vector是从一个multivariate Gaussian中sample出来的，计算生成图片和真实图片的两个hidden latent vector站之间的Frechet distance，就得到了Frechet Inception Distance，越小越好。该metric的问题在于需要sample比较大量的点才能获得distribution，并且简单的把latent vector认为是Gaussian distribution可能是有问题的。
