# 矩阵向量求导

分子布局：分子决定行，分子不同则行不同。

$$
\frac{d\textbf{y}}{dx}=\begin{bmatrix}
\frac{dy_1}{dx} \\ 
\frac{dy_2}{dx} \\ 
... \\
\frac{dy_n}{dx} \\ 
\end{bmatrix},\,\,\frac{dy}{d\textbf{x}} =\begin{bmatrix}
\frac{dy}{dx_1} & \frac{dy}{dx_2} & ... & \frac{dy}{dx_n}
\end{bmatrix}
$$

假设$\textbf{x},\,\textbf{y}$均为列向量。

$$
\frac{d\textbf{a}^T\textbf{x}}{d\textbf{x}}=\frac{d\textbf{x}^T\textbf{a}}{d\textbf{x}}=\textbf{a}^T \\

\, \\ \frac{d\textbf{x}^T\textbf{x}}{d\textbf{x}}=2\textbf{x}^T \\
\, \\ 
\frac{d\textbf{A}\textbf{x}}{d\textbf{x}}=\textbf{A} \\
\, \\ 
\frac{d\textbf{x}^T\textbf{A}}{d\textbf{x}} =\textbf{A}^T \\
\, \\ 
\frac{d\textbf{x}^T\textbf{A}\textbf{x}}{d\textbf{x}} = \textbf{x}^T(\textbf{A}+\textbf{A}^T)\\
$$

# What is Machine Learning?

Find a function $F$ such that $y=F(x)$.

- Regression: $y$ is a scalar.
- Classification: $y$ is catagorical.
- Challenging problems: structured data, such as images, texts, audios...

General framework of ML:

1. Write an unknown function such that $y = F(x)$.
2. Define a proper loss function  $L(\theta)$.
3. Optimize and find the parameters through $argmin_{\theta}L(\theta)$.

### General Guide

Identify problems in your framework:

Loss of the training data:

1. Large
   1. ***Model bias***, increase the complexity of the model.
   2. Problems about the ***optimization** *process.
2. Small, see loss of the test data
   1. Large
      1. ***Overfittting***, reduce the complexity of the model.
      2. ***Mismatching***, existing distribution shift between the training and testing data.
   2. Small, great!

### Possible problems during optimization:

1. Gradient = 0, local minima, local maxima, and saddle point.

   * Taylor expansion: $L(\theta)=L(\theta')+(\theta-\theta')g+(\theta-\theta')^TH(\theta-\theta')$, $g$ is the gradient and $H$ is the Hessian matrix. When $g$ is $0$, if $H$ positive definite (all eigenvalues $>0$), local minima; if $H$ negative definite (all eigenvalues  $<0$), local maxima; else (some eigenvalues $>0$ and some $<0$), saddle point.
   * Local minima and maxima are hard to escape, but escape from saddle points is much easier.
2. Batch size, gradient descent, batch gradient descent, and stochastic gradient descent

   |                                       | Small  | Large  |
   | ------------------------------------- | ------ | ------ |
   | Single update<br />(GPU acceleration) | Same   | Same   |
   | Single epoch                          | Slower | Faster |
   | Gradient                              | Noisy  | Stable |
   | Optimization                          | Better | Worse  |
   | Generalization                        | Better | Worse  |

   #### Optimizers


   * Momentum: Simulate momentum in physical world.
     * Parameters: $\theta_i$, parameters at batch $i$; $m_i$, real movement; $g_i$, gradient.
     * For $batch\,i$:
       * Calculate $g_i$, $m_0=0$.
       * $m_i=\lambda m_{i-1}-\gamma g_{i-1}$
       * $\theta_{i}=\theta_{i-1}+m_i$
     * $m_i$ is the exponential moving average of $g$.
   * AdaGrad: Adaptively adjusting learning rate
     * $\theta_{t}=\theta_{t-1}-\frac{\eta}{\sqrt{v_{t-1}}+\epsilon}g_{t-1}$; $v_{t-1}=\sum_{k=1}^{t-1}g^2_k$
     * Pros: When the gradient is large, learning rate will be smaller and vice versa.
     * Cons: If the gradient is large at first and then becomes small, the learning rate will not change.
   * RMSprop: Adaptively adjusting learning rate with EMA
     * $\theta_{t}=\theta_{t-1}-\frac{\eta}{\sqrt{v_{t-1}}+\epsilon}g_{t-1}^2$; $v_{t-1}=\alpha\sigma^2_{t-2}+(1-\alpha) g^2_{t-1}$; $0<\alpha<1$
     * Pros: Introducing the decay factor $\alpha$, which means the gradient closer to the current iteration should have a larger impact on the current parameter update.
   * Adam: Combining SGD with momentum and RMSprop
     * $\theta_t=\theta_{t-1}-\frac{\eta}{\sqrt{\hat{v_{t-1}}}+\epsilon}\hat{m_{t-1}}$; $\hat{m_t}=\frac{m_t}{1-\beta_1^t}$; $\hat{v_t}=\frac{v_t}{1-\beta_2^t}$, $\beta_1=0.9,\,\beta_2=0.999$.
   * Use cases: In computer vision (CV), SGD with Momentum is commonly used; while in natural language processing (NLP), reinforcement learning (RL), and generative adversarial networks (GANs)... Adam and its variants are generally preferred.

   #### Scheduler and Warm Up

   * Scheduler: Gradually decreasing learning rate. (Cosine Annealing...)
   * Warm up: Increase the learning rate at first and gradually decrease.
