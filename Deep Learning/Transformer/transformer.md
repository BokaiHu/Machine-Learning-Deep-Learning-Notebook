# Transformer Modules

## Self-Attention

Denote: Input embedding vectors $a_i$; Weight matrices $W_q,\,W_k,\,W_v$; Transformed vectors $q,\,k,\,v$; Attention score $\alpha_i$.

Note: All the input embedding $a_i$ are row vectors.

1. Obtain the query, key, value vectors $q_i,\,k_i,\,v_i$ by multiplying the weight matrices $W_q,\,W_k,\,W_v$ and the input vector $a_i$.

   $$
   \begin{bmatrix}
   	— & q_0 & — \\
   	— & q_1 & — \\
   	— & ... & — \\
   	— & q_n & — \\
   \end{bmatrix}
   =
   \begin{bmatrix}
   	— & a_0 & — \\
   	— & a_1 & — \\
   	— & ... & — \\
   	— & a_n & — \\
   \end{bmatrix}
   W_q
   $$
2. Calculate the correlation between $a_i$ and all the other input vectors through dot-product.

   $$
   \begin{bmatrix}
   	— & \alpha_{0, 0:n} & — \\
   	— & \alpha_{1, 0:n} & — \\
   	— & ... & — \\
   	— & \alpha_{n, 0:n} & — \\
   \end{bmatrix}=
   \begin{bmatrix}
   	— & q_0 & — \\
   	— & q_1 & — \\
   	— & ... & — \\
   	— & q_n & — \\
   \end{bmatrix}
   \begin{bmatrix}
   	— & k_0 & — \\
   	— & k_1 & — \\
   	— & ... & — \\
   	— & k_n & — \\
   \end{bmatrix}^T
   $$
3. Calculate Softmax in each row.

   $$
   \begin{bmatrix}
   	— & \alpha'_{0, 0:n} & — \\
   	— & \alpha'_{1, 0:n} & — \\
   	— & ... & — \\
   	— & \alpha'_{n, 0:n} & — \\
   \end{bmatrix}=
   \begin{bmatrix}
   	Softmax(— & \alpha_{0, 0:n} & —) \\
   	Softmax(— & \alpha_{1, 0:n} & —) \\
   	Softmax(— & ... & —) \\
   	Softmax(— & \alpha_{n, 0:n} & —) \\
   \end{bmatrix}
   $$
4. Obtain output by calculating the weighted sum of $v_i$

   $$
   \begin{bmatrix}
   	— & b_{0} & — \\
   	— & b_{1} & — \\
   	— & ... & — \\
   	— & b_{n} & — \\
   \end{bmatrix}=
   \begin{bmatrix}
   	— & \alpha'_{0, 0:n} & — \\
   	— & \alpha'_{1, 0:n} & — \\
   	— & ... & — \\
   	— & \alpha'_{n, 0:n} & — \\
   \end{bmatrix}
   \begin{bmatrix}
   	— & v_{0} & — \\
   	— & v_{1} & — \\
   	— & ... & — \\
   	— & v_{n} & — \\
   \end{bmatrix}
   $$

If using *Multi-head self-attention*, replace $W_q$ with $[W_{q1}, W_{q2}, ..., W_{qn}]$. Multiple heads are like different neurons in CNN, capturing different types of correlation in a sequence.

The only learnable parameters are $W_q, W_k, W_v$.

## Attention Variants

Transformer和Attention module主要的计算开销在于计算$q$与$k$的点积，如果sequence length=N，则Time complexity=$O(N^2)$.

### Local Attention / Truncated Attention

Only consider neighboring tokens when calculating $q\cdot k^T$. Set other tokens as 0 if not considered. Similar to convolutional neural network.

### Stride Attention

Only consider tokens with stride = $k$, the stride depends on the task.

### Global Attention

Add special tokens into the original sequence, these special tokens calculate dot product with all tokens; all tokens only calculate dot product with these special tokens.

### Reduce Number of *Keys*

- Compressed Attention: Using convolution layer to reduce the number of keys.
- Linformer: Shape of *Keys*: (dim token, seq length). multiply another transformation matrix with shape: (seq length, K)

## Positional Embedding

Add positional embedding $e_i$ to the input embedding vector $a_i$ to integrate positional information.

The final input of the self-attention should be $a'_i=e_i+a_i$.

Two implementations (see positional_embedding.ipynb)

- Sinusoidal Positional Embedding
- Rotary Positional Embedding
  - $R_\theta^T=R_{-\theta}$
  - $R_\theta \cdot R_\phi=R_{\theta+\phi}$
  - $(R_\theta k)^T\cdot R_\phi q=k^T\cdot R_{-\theta}\cdot R_\phi \cdot q=k^T\cdot R_{\phi-\theta}\cdot q$.

## Self-attention v.s. CNN

CNN: restricted self-attention in a receptive field

Self-attention: CNN with learnable receptive field.

Self-attention is a complex version of CNN, thus requires larger amount of training data.

## Self-attention v.s. RNN

RNN: Non-parallel computing; hard to consider long-term memory; uni-directional.

Self-attention: Parallel computing; consider long-term memory; bi-directional.

# Transformer Architecture

Note: Input = sequence embedding + positional embedding; Output of the decoder block should be one selected word / character from the vocabulary (include a special token **END**) by choosing the maximum after a Softmax.

- Encoder block:
  - Input -> Self-attention -> add residual & layer normalization -> Feedforward network -> add residual & layer normalization;
- Decoder block:
  - Autoregressive:
    - Input -> Masked self-attention ->add residual & layer normalization -> Cross-attention -> add residual & layer normalization -> Feedforward network -> add residual & layer normalization -> Linear & Softmax
    - Masked self-attention: When calculating the output $b_i$, only use input vectors no more than $i$, i.e. $\{a_0, ...,\, a_i\}$ and set${}$ $\{a_{i+1},\,...,\,a_n\}=0$.
    - Generate the output token using the past outputs. Generate the 1st token $b_1$ with input **BEGIN**, generate the 2nd output $b_2$ using the $\{$**BEGIN**, $b_1\}$...
    - Cross-attention has the same structure as the self-attention, it takes two inputs: one from the masked self-attention and one from the encoder.
  - Non-Autoregressive:
    - Generate all the output tokens at once with all the inputs are **BEGIN**.
    - Use an additional classifer to determine the length of the output given the input sequence.
    - Set a limit for output length $N$, feed the decoder block with $N$ **BEGIN**s, and cut the output sequence at the first **END** token.
    - Advantage: Parallel; controllable output length.

# Training & Testing

## Training

Minimizing the **cross entropy** of all the output token with the ground truth. **NOTE** that we need input for both encoder and decoder. During training, the input for decoder should be the ground truth (teacher forcing).

## Tips for training:

- Copy mechanism: directly copying part of the input sequence. (Pointer Network)
- Guided attention: 例如语音合成或语音识别，输出的attention score应该是从左到右而非随机位置的。对于这种特殊的任务可以强迫attention聚焦在某个位置，keywords: monotonic attention, locatoin-aware attention.
- Beam search: 每次取概率最大的$k$个输出，一共维持$k$条output sequence. greedy decoding不一定是最好的，beam search是对搜索全局路径的一个approximation. 有时候work(不需要创造力的)有时候不work(需要创造力的)，因为永远找全局最优对于生成文本来说也不一定是最好的。[10.8. Beam Search — Dive into Deep Learning 1.0.3 documentation (d2l.ai)](https://d2l.ai/chapter_recurrent-modern/beam-search.html)
- Parallel scheduled sampling: 在inference的时候decoder的输入是之前的output，其中可能会有错误的内容导致之后的输出全部collapse。考虑在train的时候给decoder的输入中替换一些错误信息进去来缓解这个问题。
- Some others: 语音合成在test的时候要加noise; loss function不知道如何optimize可以把它当reward用RL来train;

# Others

Batch norm: normalize over the *channel* dimension; Layer norm: normalize over the *batch* dimension.

- Why using layer norm instead of batch norm?

  1. The length of the input sequence may be different.
