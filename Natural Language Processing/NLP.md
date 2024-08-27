## Before LLM

### Word2Vec

- Skipgram: 根据滑动窗口中心的词输出上下文的prob
- Continuous Bag of Words (CBOW): 根据上下文输出滑动窗口中心词的prob

对于Skipgram模型, Word2Vec的训练方法是，从数据集sample词以及它的上下文(例如This is not a cat, 得到四个正例[not, this], [not is], [not, a], [not, cat], 再随机sample一些词和not放在一起作为负例, 例如[not, bluff]...)，训练的模型包括两个矩阵embedding和context, 把not输入embedding, 上下文输入context, 然后计算两个word embedding的cosine similarity经过sigmoid后计算loss. 训练到一定程度就可以得到一个可以使用的embedding矩阵.

CBOW将上下文转换为embedding后做加权平均，经过神经网络预测中间的词然后softmax得到概率计算loss

### TF-IDF

整体的思想就是：**如果某个词在某类或某条文本中出现的频率很高，但在整个数据集中出现的频率并不高，那么这个词就很适合用于分类。**

- Term Frequency: 某个词在某条文本中出现的频率。 $TF=\frac{count(w)}{|x|}$.
- Inverse document frequency: 总文本数比上某个词在文本中出现的频率。$IDF=log\frac{|D|}{\sum_i (w\in x_i)}$

### FastText ##TODO

### Recurrent Neural Network

每一步接收当前时间步的input $x_t$和上一个时间步的output $h_{t-1}$, 在RNN cell里用不同的全连接层处理得到当前时间步的输出$h_t$.

$$
h_t = tanh(x_tW_x+b_x+h_{t-1}W_h+b_h)
$$

多层RNN: 下一层RNN的input $x_{n, t}$是上一层当前时间步的output $h_{n-1, t}$.

$$
h_{n,\,t} = tanh(h_{n-1,\,t}W_x+b_x+h_{n,\,t-1}W_h+b_h)
$$

在实践中，RNN往往难以学习到长距离依赖关系。

### Long-Short Term Memory

添加了遗忘门，更新门和输入门，解决了RNN中的梯度爆炸/梯度消失以及无法获取长距离依赖的问题。

$$
f_t=\sigma(W_f[h_{t-1}, x_t]+b_f) \\
i_t=\sigma(W_i[h_{t-1}, x_t]+b_i) \\
\hat{C_t}=tanh(W_C[h_{t-1}, x_t]+b_C) \\
C_t=C_{t-1}*f_t+\hat{C_t}*i_t \\
o_t = \sigma(W_o[h_{t-1}, x_t]+b_o) \\
h_t = tanh(C_t) * o_t
$$

Bidircetional LSTM就是把$x$正向输出给前向LSTM, 把$x$反转后输入给后向LSTM, 再把后向LSTM的输出再反转和前向输出的timestep对齐, 然后再hidden dim维度上concat起来。

$$
h_{forward}=lstm_{forward}(x) \\
h_{backward}=lstm_{backward}(reverse(x)) \\
h_t = Concat[h_{forward}, reverse(h_{backward})]
$$

## Details

### Tokenization

分为三类:

- Word-level: 词表太大，有些词出现频率过低导致未被完全训练。
- Character-level: 词表小，但无法表征单词语义，并且转化后的token序列很长，可能导致模型处理时间变长。
- Subword-level: 好。

#### Byte-Pair Encoding (BPE)

将文档拆成单个字符放进初始化词表中。给定一个最大词表大小，统计连续两个token出现的频率，然后将频率最大的两个子token合并成新的一个token放进词表代替之前的两个子token。重复此过程直到词表大小符合要求。

#### Byte-level BPE

将文档拆成单个Byte。初始化词表中包含特殊token(`<unk>, <bos>, <eos>等`)和256个字节(<0x00>, <0x01>, ..., <0xFF>)。然后在Byte层面上统计连续两个Byte出现的频率进行合并。

优点在于：

- 对多语言支持
- 不会出现OOV

UTF-8编码不会出现多解的情况，例如"E7 B4 C9"只可能为汉字而非三个英文字母。

### LayerNorm vs BatchNorm vs RMSNorm

LayerNorm (theoretical): 每个词向量根据整个句子的mean和std做normalization

LayerNorm (PyTorch): 每个词向量根据它自己的mean和std做normalization

BatchNorm: 某个pixel根据整个batch中其所在channel的所有pixel的mean和std做normalization

RMSNorm: 简化版LayerNorm, 不需要计算std和mean, 只计算均方差即可。

为什么NLP用LayerNorm:

- 句子长度不同可能导致和padding做normalization
- 不同句子中的单词不一定具有相似特征，所以不应该一起normalize.

为什么LLaMa用RMSNorm: 相比于LayerNorm参数更少，运算量更小。

### Activation function

- $Swish(x) = x * Sigmoid(\beta x)$. 也叫SiLU, Sigmoid GLU, $\beta$ is learnable.
- $GeLU(x) = x * \Phi(x)$, $\Phi(x)=P(X<x),\,X\sim N(0, 1)$.
- $SwiGLU(x) = Swish(xW+b) * (xV+b)$.
- $GeGLU(x) = GeLU(xW+b) * (xV+b)$.

SwiGLU, GeGLU包含两个FC层, 用来代替transformers FFN中的第一个FC layer和activation fn.

### Sinusoidal Positional Embedding

- 为什么transformers中要加入positional embedding？

  - RNN, LSTM是天然具备了输入序列的顺序的，而transformer由于并行化是无法天然蕴含位置信息的。
- 绝对位置编码：偶数位是sin，奇数位是cos

  $$
  pe_{pos, 2i}=sin(\frac{pos}{10000^\frac{2i}{d_{model}}}) \\
  pe_{pos, 2i+1}=cos(\frac{pos}{10000^\frac{2i}{d_{model}}})
  $$

移动一定距离的pos+k能够表示为pos的embedding的线性变换：

$$
pe_{pos+k, 2i}=sin(w_{2i}*(pos+k)) \\
pe_{pos+k, 2i+1}=cos(w_{2i}*(pos+k)) \\
pe_{pos+k, 2i}=sin(w_{2i}*pos)cos(w_{2i}*k)+cos(w_{2i}*pos)sin(w_{2i}*k) \\
pe_{pos+k, 2i+1}=cos(w_{2i}*pos)cos(w_{2i}*k)-sin(w_{2i}*pos)sin(w_{2i}*k)
$$

因为$k$是常数，令$u=cos(w_{2i}*k),\,v=sin(w_{2i}*k)$, 可得

$$
\begin{bmatrix}
pe_{pos+k, 2i} \\
pe_{pos+k, 2i+1}
\end{bmatrix}=
\begin{bmatrix}
u, v \\
-v, u
\end{bmatrix}*
\begin{bmatrix}
pe_{pos, 2i} \\
pe_{pos, 2i+1}
\end{bmatrix}
$$

另外，两个pos之间点积能够表示其的相对位置信息：

$$
pe_{pos+k}*pe_{pos}=\sum_isin(w_{2i}*(pos+k))sin(w_{2i}(pos))+cos(w_{2i}*(pos+k))cos(w_{2i}(pos)) \\
=\sum_icos(((pos+k)-pos)*w_{2i})=\sum_icos(k*w_{2i})
$$

当相对距离$k$增大时，点积会减小。但问题在于该方法并不能表征方向性，即$pe_{pos+k}*pe_{pos}=pe_{pos-k}*pe_{pos}$, 和原来的embedding相加后再做点积就会有

### RoPE 旋转位置编码

我们只要想办法让$q_m,\,k_n$的点积表现为一个只和他们之间距离之差$m-n$的相关的函数，就能让transformer学习到相对位置信息。借助复数域，在二维的情况下：

$$
q_m=r_{q,\,m}e^{i\Theta_{q,\,m}} \\
k_n=r_{k,\,n}e^{i\Theta_{k,\,n}} \\
q_m\cdot k_n=Re[q_m\cdot k_n^*]=Re[r_{q,\,m}r_{k,\,n}e^{i(\Theta_{q,\,m}-\Theta_{k,\,n})}] \\
$$

假设现在存在一个函数$g(q, k, m-n)$使得

$$
r_{q,\,m}r_{k,\,n}e^{i(\Theta_{q,\,m}-\Theta_{k,\,n})}=g(q,k,m-n)=r_{q,k,m-n}e^{i\Theta_{q,k,m-n}} \\
r_{q,\,m}r_{k,\,n}=r_{q,k,m-n}=r_{q,k,0}=r_qr_k=||q||||k||,\,when\,m=n \\
\Theta_{q,\,m}-\Theta_{k,\,n}=\Theta_{q,k,m-n}=\Theta_{q,k,0}=\Theta_q-\Theta_k,\,when\,m=n
$$

所以当$m=n$时，$r_q,\,r_k$都与$m$无关，我们简单认为他们就是$||q||,\,||k||$。同样的$\Theta_{q,\,m}-\Theta_{k,\,m}=\Theta_q-\Theta_k \Rightarrow \Theta_{q,\,m}-\Theta_q=\Theta_{k,\,m}-\Theta_k=\phi(m)$。现在令$n=m-1$, 则有

$$
\Theta_{q,\,m}-\Theta_{k,\,m-1}=\Theta_q+\phi(m)-\Theta_k-\phi(m-1)=\Theta_{q,k,0}+\phi(m)-\phi(m-1) \\
\phi(m)-\phi(m-1)=\Theta_{q,k,1}-\Theta_{q,k,0}
$$

右侧这个是一个和$m$无关的常数，所以$\phi(m)$是一个关于$m$的等差数列, 令$\Theta_{q,k,1}-\Theta_{q,k,0}=\theta$, 则$\phi(m)=m\theta$. 所以最终$q_m=qe^{i\theta m}$. 该式又可以表示为旋转矩阵:

$$
q_m=(q_0, q_1)\times (cos(\theta m),sin(\theta m))=q_0cos(\theta m)+q_oisin(\theta m)+q_1icos(\theta m)-q_1sin(\theta m) \\
=\begin{bmatrix}
cos\theta m & -sin\theta m \\
sin\theta m & cos\theta m
\end{bmatrix}
\begin{bmatrix}
q_0 \\
q_1
\end{bmatrix}
$$

可以将二维情况直接拼接扩展到$d_{model}$维：

$$
R_m = \begin{bmatrix}
\cos(\theta_0 m) & -\sin(\theta_0 m) & 0 & 0 & \cdots & 0 \\
\sin(\theta_0 m) & \cos(\theta_0 m)  & 0 & 0 & \cdots & 0 \\
0 & 0 & \cos(\theta_1 m) & -\sin(\theta_1 m) & \cdots & 0 \\
0 & 0 & \sin(\theta_1 m) & \cos(\theta_1 m)  & \cdots & 0 \\
0 & 0 & 0 & \cdots & \cos(\theta_{d_{model} / 2} m) & -\sin(\theta_{d_{model} / 2} m) \\
0 & 0 & 0 & \cdots & \sin(\theta_{d_{model} / 2} m) & \cos(\theta_{d_{model} / 2} m) 
\end{bmatrix}
$$

这样在点乘时：

$$
(R_mq_m)^T(R_nk_n)=q_m^TR_m^TR_nk_n=q_m^TR_{m-n}k_n
$$

实验显示RoPE的外推性好。外推性指的是在推理阶段遇到了预训练阶段没见过的长度，应用该位置编码是否会出现性能下降。RoPE在该情况下下降不明显，而绝对位置编码下降明显。

- 远程衰减：低维高频，高维低频，假设$q_i=k_i=1, $那么计算应用RoPE后的向量点积时可以得到$\mathbf{1}R_m^TR_n\mathbf{1}=2\sum_i cos((m-n)\theta_i)$, 假如$m-n$比较大，那么它会超出低维区域的周期，但始终处于高维区域的下降1/4区间内，这就导致随着$m-n$的增大，低维区域的编码始终震荡，而高维区域的编码单调递减，二者相加得到一个总体趋势递减的函数。

### ALiBi

在softmax之后添加一个不可学习的参数，这个线性bias表示了$q$和$k$之间的位置偏差。对于attention mask:

$$
\begin{bmatrix}
q_1k_1 & -1e9 \\
q_2k_1 & q_2k_2
\end{bmatrix}
$$

我们添加的偏置矩阵为:

$$
\begin{bmatrix}
0 & 0 \\
-1 & 0
\end{bmatrix} * m
$$

$m$的取值取决于注意力头的数量, 当num_heads=8时, $m=\frac{1}{2^k}$, $k=\frac{8}{i}$, $i$是第几个头。

# Pretrained Language Model

## Basic

### 参数量估计

$V: Vocab\_size,\,h:hidden\_size,\,n:num\_layers$

- Embedding: $V*h$
- Attention: qkv+out proj: $4h^2+4h$
- FFN: 2 h->4h FCs: $8h^2+5h$; SwiGLU/GeGLU + FC: $12h^2+9h$
- 2 LayerNorms: 2 weight + 2bias: $4h$
- lm_head: same as embedding: $V*h$
- Overall: $Vh+n(12h^2+13h)$, shared embedding

### 显存估计

Model parameters: $x$ B.

- 模型本身:
  - FP32: $4x$ Gb
  - FP16: $2x$ Gb
  - int8: $x$ Gb
- 梯度:
  - FP16: $2x$ Gb
- 优化器状态(Adam):
  - 模型主参数(必须为FP32): $4x$ Gb
  - 动量: $4x$ Gb
  - 方差: $4x$ Gb
- Overall: $(4+12)x$ Gb

### 预训练任务

- Language Modeling: GPT类预训练任务, 根据之前词语预测下一个.
- Permuted Language Modeling: XLNet预训练任务，把序列文本打乱，在所有可能的sequence上进行Language Modeling.
- Masked Language Modeling: BERT预训练任务之一, 根据上下文预测中间被MASK掉的词语.
- Denoising AutoEncoder (DAE): 更广泛的MLM任务，包含
  - 随机MASK掉某个词汇
  - 随机删除某个词汇并让模型确定删除词汇的位置
  - 采样不同长度的文本替换为单个[MASK]标记，模型需要预测被mask掉标记的个数
- 对比学习方法
  - Next Sentence Prediction (NSP): BERT预训练任务之一，预测B是否是A的下一句，但RoBERTa和XLNet认为该任务反而对模型预训练起到负作用。
  - Sentence Order Prediciton (SOP): AlBERT预训练任务，模型需要预测两个句子的顺序。

### 继续预训练 Continual Pretraining

对上下游任务影响的因素：

- Warm-up的影响：Warm-up基本不会影响微调后模型在上下游任务上的性能(即不管使用多大的warmup比例最终收敛到的性能都是一样的)。但更大的Warm-up在训练前期时，在上下游任务上的Loss均会更小。
- 学习率的影响：学习率越大，下游任务性能越好，上游任务性能越差。在训练初期，学习率越小loss越小。

注意到的点：

- 进行了继续预训练的模型都比从头开始预训练的模型效果更好。即在数据集A上先进行预训练，然后再在数据集B上进行预训练的效果优于只在数据集B上进行预训练。
- 当我们的数据不能完全训练时(即只考虑早期训练效果)，可以考虑小学习率+大warm-up。
- 另外，如果先在数据集A的一部分上进行预训练，然后再接着用数据集A继续预训练，这时使用Warm-up会损伤性能，且Warm-up越大损失越厉害。

## RLHF

### Reward Modeling

训练一个能够对模型生成的输出进行打分的模型，这个模型通常是使用已经经过SFT训练的LM，通过把原先的LM head替换成一个用来输出标量值的线性层对输出进行打分。

- 模型选择：一般会选择比生成模型小一些的模型，例如175B的GPT选择了6B作为RM，因为他们发现175B的RM生成结果不稳定。另外从直觉上来说，打分判别任务比生成任务更简单。
- 训练方式：我们让模型的输出是对当前回答的评分，但我们的训练方法是针对模型对不同回答的排序计算一个loss。因为人和人的评分可能相差很大，但排序是基本一致的。所以我们要将针对一个Query的所有Response放在一个batch内，一次性生成模型对他们的预测评分，然后两两取出，计算评分之间的logsigmoid，损失函数为：

  $$
  Loss=-\frac{1}{\binom{k}{2}}log(\sigma(RM(x_w)-RM(x_l)))
  $$

  并且原论文中提到要将所有的关于同一个query的response放在同一个batch中，因为这样针对一个query只进行了一次梯度更新，如果将所有数据拆开分散在数据集中的话就需要对同一个query进行$\binom{k}{2}$次梯度更新，会过拟合。

### PPO Proximal Policy Optimization

- 用一个reward model估计reward, $R_{actual}=R_{RM}-\lambda_{kl}\frac{P(a_t|s_t)}{P_{ref}(a_t|s_t)}$. 这里加入了reference model和training policy之间的KL divergence保证二者相差不要太大. 最初的Actor loss为$Loss=-V_tlogP(a_t|s_t)$, 此处的$V$是由Value head预测, 相当于critic model, Value head是接在LM transformer层后面预测每个token的value的。这里我们用优势函数$A_t$代替Value, $A_t=Q_t-V_t=R_t+\gamma V_{t+1}-V_t$. 并且用GAE估计$A_t=R_t+\gamma V_{t+1}-V_t +\gamma*\lambda A_{t+1}$. Actor loss变为$Loss=-\sum_t^T A_tlogP(a_t|s_t)$.
- 明确一下在PPO训练过程中，我们的language model有**三个状态**: 1. 完全没有进行过RL的SFT reference model, 2. 每隔一个batch更新一次的Old policy, 3. 在batch内多次更新的New policy。当一个batch数据进来时，我们会用***Old policy***对这个batch的query生成response(采样)，然后计算***reference model***对这个response的log prob，这二者的比值会被用在Reward计算中利用KL divergence确保policy更新不偏离reference太远。由于计算效率的问题，我们会用这一个batch的数据进行ppo_epoch次更新，每次更新我们都需要用***New policy***生成***Old policy***采样结果的log prob。但由于这里采样是用***Old policy***采的，所以我们需要用重要性采样(Importance Sampling)，把Loss变为$Loss=-\sum_t^T A_tlog\frac{P(a_t|s_t)}{P_{old}(a_t|s_t)}$, 这里的$P_{old}$指的是batch输入时更新前的***Old policy***, 而$P$就是每个ppo_epoch更新后的***New policy**。*

### DPO Direct Preference Optimization

通过重参数化避免了奖励建模，优化目标为

$$
L_{DPO}(\pi_{\theta}, \pi_{ref})=-E_{\pi_{\theta}}(\sigma(\beta\frac{\pi_{\theta}(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\frac{\pi_{\theta}(y_l|x)}{\pi_{ref}(y_l|x)}))
$$

这个式子长得就很像奖励模型训练目标，所以其实有一个隐式的奖励建模：$\hat{r_{\theta}}=\beta\frac{\pi_{\theta}(y_w|x)}{\pi_{ref}(y_w|x)}$.

### BERT系列

- BERT: encoder only, MLM + NSP
- StructBERT: 修改了NSP loss, 1/3概率A是B的下一句, 1/3概率B是A的下一句, 1/3概率AB来自不同文本。添加了重构loss，随机选取三个token打乱让encoder学习复原顺序。
- ALBERT: BERT + embedding 矩阵分解 + 参数共享 - NSP + SOP。
  - 把embedding矩阵分解成vocab_size * embedding_size + embedding_size * hidden_size, 其中embedding_size << hidden_size.
  - encoder blocks共享参数，不再需要串行12个encoder block，而是单个encoder循环12次。
  - 抛弃了StructBERT中AB来自不同文本的构造，只有1/2和1/2。原因是作者认为有两个任务: topic prediction和coherence prediction, MLM中已经包含了topic prediction, 只需要加强模型学习coherence的能力。
- RoBERTa:
  - 动态构造mask，不是预处理数据而是在输入过程中随机选取，增大多样性
  - 放弃NSP loss，输入修改为拼接完整句子直到达到truncation上限512词。
  - 增大batch size。
  - 把word级tokenization修改为subword级。
- TinyBERT: 对BERT进行多阶段多层次蒸馏
  - 多层次: 分别对Embedding输出，每隔n层的TransformerBlock输出的hidden state和attention matrix以及最终的logits进行模仿学习。假如原来的12层BERT distill到 4层的TinyBERT，那就是TinyBERT embedding + up projection = BERT embedding用MSE训练；Bert中每三层最后的attn score以及hidden state对应TinyBERT中一层的attn score和hidden state用MSE训练；最后Logits用CE训练。
  - 多阶段：先在预训练数据集上蒸馏，再在特定任务数据集上蒸馏
- DeBERTa:
  - 在Transformer层内处理时添加相对位置编码
  - 在最后一层transformer层后添加绝对位置编码
  - 虚拟对抗训练：在输入单词嵌入上添加一个小的扰动，增强模型训练的鲁棒性

### Embedding模型

和CLIP的预训练方式类似，都是Contrastive learning，构造一个句子和相似句子的数据集，每个batch里AB两组对应位置就是相似的句子，剩下其他的都看作不相似的为负样本。Embedding模型基本都是BERT类的Encoder-only架构。

1. **Cosine Similarity**: 把两句话分别输入BERT得到embedding，然后计算embedding的相似度。
2. **ReRank**：将两句话拼接起来输入BERT计算embedding，然后通过一个classifier判断这两个句子之间的匹配程度。

在RAG中通常使用Cosine Similarity方式，因为我们可以提前将知识库内的文本处理成向量保存下来。在ReRank中每处理一条Query都需要将Query和知识库中的每一个Block拼接起来进行一次Forward pass，消耗时间远超Similarity方法。但ReRank精准度相对更高。

**如何评估Embedding模型？**

- 在特定任务上线之前：
  - Relatedness，wordsim353数据集，找一些词，看和他们空间距离近的词，是否和人想到的词相近。
  - Analogy, A+B-C=D，例如King-Man+Woman=Queen; China + Beijing - Japan = Tokyo等
  - Categorization, 查看每个词在不同类别中的概率。
  - 降维聚类查看词向量分布。
  - MTEB数据集，包含八种task：
    - Bitext mining: 两种语言数据集，用第一个数据集里的句子找第二个数据集里最匹配的。
    - Classification: 训练测试集用模型编码后，训练集用来train一个clf，然后在测试集上测试。
    - STS: 对句子对编码后判断他们的相似度，输出一个0-5之间的float。
    - Summarization: 对于一个文本段-摘要pair，分别编码后计算相似度。
    - Clustering: 用模型对数据进行编码后训练一个k-means，然后进行打分。（应该是看分类是否正确或者相似句子是否包含在同一类中）
    - Pair Classification: 给一对句子进行编码，然后根据cosine similarity判断标签。
    - ReRanking: 给一个Query，从一个包含相关和不相关的句子list里取出来和Query拼一起，然后通过FFN输出相关程度。
    - Retrieval: 模型对Query和语料进行编码后计算相似度，然后考虑排名前$k$个看召回率。
- 在特定任务上评价效果：
  - Hit Rate@k: 在前$k$个候选中包含正确的概率。
  - MRR (Mean Reciprocal Rank): 看相关文档排名的倒数。如果相关文档排第一就是1，相关文档排第二就是$\frac{1}{2}$.

### 长上下文

问题在于

1. 模型在推理或微调时存在远超训练上下文长度的情况，这时我们需要使用到没被训练到的位置编码。
2. Transformer的时间复杂度随序列长度平方增长，且显存占用量会极大增加，且训练数据中超长文本数量不多。
3. 模型在长文本上训练会稀释原先在短文本上的注意力，会导致在短窗口上的性能下降。

解决问题一的办法：

**位置插值(Position Interpolation)**。假如训练时文本长度为$L\in[0,2048]$, 推理时出现了$L'\in[0, 4096]$长度的文本，那么，这时把$L'$放缩到支持的长度内。

$$
f(x, L')\Rightarrow f(x, \frac{2048L'}{4096})
$$

**这种方法的问题在于平等的对待所有的位置编码，但问题在于低维的信息频率本身就较高，再经过插值会导致信息密度过大，模型无法很好的区分不同位置的编码。**

**NTK-aware interpolation**: 但在LLaMa2-Long中发现这种方法会大幅缩减两个连续位置embedding之间的距离，使得模型对距离不那么敏感。所以提出一种新的方法**Adjust base frequency**， 直接修改计算theta时的基底。

连续的位置编码可以看成：

$$
cos(\frac{pos}{\beta^i}),\,\,i\in[0,...,\frac{d_{model}}{2}-1],\,\,\beta=10000^{\frac{2}{d_{model}}}
$$

我们给$\beta$乘以一个$\lambda=k^{\frac{2}{d_{model}-2}}$, 这样对于高维的就相当于$cos(\frac{\frac{pos}{k}}{\beta^i})$, 对于低维的则是$cos(\frac{\frac{pos}{k^{\frac{2}{d_{model}-2}}}}{\beta^i})$, 因为$d_{model}$一般很大，所以$\lambda$本身很接近于1.这就相当于对高维做了scale=k的插值，而低维的只做了k略大于1的插值。

**NTK-by-parts:** 只在低频内插，高频不操作。

**Dynamic-NTK:** 在推理时根据序列长度动态内插，选择$s=max(1, l'/L)$作为内插的scale，$l'$是当前seq的长度。

**YaRN:** NTK-by-parts + temperture的缩放。发现在softmax之前加一个temperature对logits进行缩放能变好。因为我们把RoPE实现为两个向量和qk相乘，所以可以把temperature一起在RoPE里实现。通过给RoPE的两个向量添加一个$\sqrt{\frac{1}{t}}$的缩放实现对logits的缩放。

**LongRoPE:** 低维信息比较密集插值应该较少，高维相对稀疏。用进化搜索算法对每个维度不同token的RoPE进行搜索确定插值方案。先在256k上下文上进行微调，然后再用LongRoPE的非均匀插值直接扩展到8倍大小(即2048k)。之后发现在短文本上的性能下降，所以在扩展后的大模型8k上下文长度内重新进行搜索来恢复短文本上的性能。推理时可根据输入文本的长度自动调整插值比例。

    实现：搜索得到每两个维度的插值比例，然后相当于一个adaptive的有参数尺度缩放。定义一系列PI，NTK和YaRN作为初始化种群，通过进化算法选择最优个体(取得了最低的困惑度)进行变异和交叉重组获得新的个体。

    先搜索参数插值到128k进行微调，然后根据初始base_length再搜索另一组参数插值到256k再微调，如果target length大于256k，例如2048k，则再根据初始base_length进行一次搜索直接插值，不做微调。最后再对短context(例如4k, 8k)做一次搜索插值微调。

解决问题二需要：

1. 修改Attention mechanism, 使用滑动窗口attention, 局部attention来减小输入模型的序列长度。
2. **LongLoRA:** 使用了S2Attn (Shifted sparse attention)，将文段分成小段，每段有$k$个token，每个token假设有$n$个heads，将其中一半的heads shift $k // 2$个位置，来确保每个小段之间都能相互交换信息。

## PEFT

### LoRA

预训练模型的权重矩阵往往是满秩的，但在微调过程中更新的权重矩阵$\Delta W$往往是低秩的，因此可以用矩阵分解的方法:$\Delta W=W_AW_B$并保持原始预训练模型参数不变，这样能够极大减少需要训练的参数量。

### Adapter

在预训练模型内部插入一些参数量很低的模块，在下游任务微调时只对这部分参数进行训练。例如在transformer中可以在TransformerBlock中插入一些FFN：Self-Attention -> Adapter -> Add & Norm -> FFN -> Adapter -> Add & Norm.

- Mixture of Adaptation: 结合了MoE思想，每个Adapter就是一个Expert

### Prefix系列

- Prefix-tuning: 在下游任务微调时，在模型每个transformer的输入前都拼上一个embedding prefix $P_{\theta}$, 其他参数冻结只训练这个参数，但作者发现只微调$P_{\theta}$会导致训练不稳定，于是将$P_{\theta}$拆分成一个更小的embedding $P_{\theta'}$和一个MLP $f_{\phi}$, $P_{\theta}=f_{\phi}(P_{\theta'})$，训练时联合优化这两个，最后只保存$P_{\theta}$即可。
- Prompt-tuning: Prefix-tuning的简化版，只在embedding层插入prefix，并且不需要额外MLP直接优化$P_{\theta}$，至于为什么没有Prefix-tuning中提到的训练不稳定以及效果不好的情况，估计是模型参数规模增大带来的好处。
- P-tuning: 不同于以上两个方法插入虚拟的embedding，P-tuning插入的是未使用的token，在BERT中是[unused1]-[unused100]。这是因为作者认为这些virtual embedding之间也应该存在自然语言的相关性，所以不应该随机初始化，所以作者用一个LSTM + MLP / MLP对virtual token进行编码得到virtual embedding后再插入模型输入中。并且这里的virtual token不一定非要是前缀，例如对于GPT而言。
- P-tuning-v2: 作者认为prompt tuning 和 p-tuning中存在一些问题：
  - prompt只插入在embedding层，没有在深层进行优化
  - 在模型参数大于100B的时候效果逼近全量微调，然而在参数量小的时候表现不如全量微调。
  - 所以决定在每一层都插入prefix，其实就是改进版的Prefix-tuning。
    - 作者发现用来Prefix-tuning中用来重参数化MLP和P-tuning中的LSTM加入对于模型帮助并不大，所以他们移除了这些模块。
    - 对于不同难度的任务插入不同长度的embedding，简单的情感分析插入短的，词性标注插入长的。
    - 不像之前的方法在MASK位置接LM head来预测tokens获得标签，这篇文章回归了传统的CLS label直接预测标签。

### Model Quantization

两种量化方式：

- 直接将原数据分布rescale到量化目标分布上，例如目标为INT8, 那么量化函数为$x_{quant}=round(x*c_{quant})$, 这里$c_{quant}=\frac{128}{max_i(x)}$, 反量化时即为$x_{dequant}=\frac{x_{quant}}{c_{quant}}$. 此时量化后的权重矩阵元素均为[-128, 127]之间的INT8数。
  - 优点在于实现简单，效率较高。缺点在于当数据中存在outlier时会导致量化误差较大，可以通过分块量化缓解。
- 聚类量化/分位数量化：以INT8 聚类量化为例，现在权重矩阵中计算$2^8$个聚类中心并保存在一个向量里，然后将权重矩阵中的元素map到对应的聚类中心，此时权重矩阵中只保存对应聚类中心在向量里的index, 为[0, 255]. 此时我们还需要额外保存一个聚类中心的向量。
  - 优点在于能够适应不同的数据分布，对离群点的适应性很强。但缺点在于计算要求高，且相比于直接量化需要保存额外的向量。

absmax量化包括：

- 对称量化：找出所有权重中绝对值最大的那个，然后rescale到对应范围。

  $$
  x_{quant}=round(\frac{x}{s}),\,s=\frac{max(abs(x))}{2^{k-1}-1} \\
  x_{dequant}=x_{quant}*s
  $$

  这种方法的缺点在于大部分模型权重都不是关于零点对称的，这会导致量化不均匀。优点在于计算量较小。
- 非对称量化：不以量化前的0和量化后的0做对应，而是选一个其他的零点。

$$
x_{quant}=round(\frac{x}{s}+z),\,s=\frac{max(x)-min(x)}{2^k-1},\,z=round(-\frac{min(x)}{s}) \\
x_{dequant}=s*(x_{quant}-z)
$$

    优点在于能够适配不对称的数据分布，量化误差相对较低。缺点在于增加了额外的计算量。

Post-training quantization: 在模型训练完毕后再进行量化。

Quantization-aware training: 在模型训练过程中先将权重量化再反量化以模拟量化误差，这样模型可以学习如何适应量化误差，取得更好的性能。

### QLoRA ##TO DO

## RAG

基础RAG流程：知识库向量化存储，查询文本向量化->计算相似度->取出相似度最大的K个->作为上下文，输入LLM进行回答。

优化方向：

- 检索前优化：
  - **重写Query**
    - 利用LLM对用户输入进行理解然后重写得到多条查询，每条查询都去进行检索然后对检索得到的结果用Reciprocal rank fusion进行排序后放进LLM进行生成(RAG-fusion)。
    - 用LLM生成一些相关的假设性问题来增大搜到相关信息的概率(HyDE)
    - 对用户Query进行降噪：在很多情况下Query中的部分词是没有意义的，我们只需要提取关键词即可。很多开源库中维护了一些Stop word可以借用。
  - 文本分块：使用**特定长度**，或者**根据语义**对长文本进行**分块**，实现更细粒度的文本语义查询。
- 检索时优化：
  - 使用更强大的embedding模型：不使用固定的静态编码，而是使用能够根据上下文生成不同编码的模型。
  - 使用混合编码方式：可以将现代编码模型与传统编码(例如TF-IDF, ngram, 关键词等)结合起来。
  - 还可以通过知识图谱进行召回。
  - 通常使用**多路召回**(就是同时使用多种召回方式: 向量召回+关键词召回+图谱召回)方式，然后进行排序交给大模型。
- 检索后优化：
  - 压缩检索得到的提示。
    - 对多路召回的结果进行**去重**。
    - 根据文件的归属关系，如果多个子片段都来自于同一个父片段，可以将这些片段合并。
  - 计算检索结果的相关度进行排序。

# 加速框架

## 多卡并行训练

假设一台机器上有$n$张显卡。一个transformer模型中包含$m$层transformer block。

### 数据并行 Data Parallel

将模型复制$n$份分发到所有显卡上，将一个batch的数据分割成$n$份，每张卡只负责这一小份数据的训练，训练完毕后进行汇总 (loss 或者 gradient accumulation)

**缺点: 如果单张卡无法容纳完整模型则无法使用。**

### 流水线并行 Pipeline Parallel

将模型按层拆开分发到不同显卡上。即每张卡都会拿到$\frac{m}{n}$个transformer block, 数据从第一张卡进入依次经过所有显卡完成训练。

**缺点: 这种方法就相当于将$n$张卡合并成一张卡，显存扩大了$n$倍。导致在一张卡上进行计算时其他卡全部都是空闲的。并且最开始计算得到的activation需要被一直保存直到轮到当前GPU进行反向传播。**

**解决方法:**

- **GPipe:** 将一个minibatch再进行拆分，拆成更小的shard (例如batchsize=64, minibatch=16, shard=4), 每完成一个shard的计算就将其放入下一张卡中，这样就确保了有$\frac{64}{4}$张卡能够同时进行计算的.
- **PipeDream:** 一个minibatch完成所有的前向传播后立即开始反向传播。bubble数量和GPipe相同，但可以更早的释放掉一部分完成反向传播的激活，对显存的要求降低。

### 张量并行 Tensor Parallel

由于transformer中的大部分运算都是矩阵乘法，可以将模型的权重矩阵根据行或者列拆开进行运算，每张卡保存一个模型中所有block的一部分参数。列并行需要对输出结果进行拼接，行并行需要对输出结果进行加和。多个全连接层只需要在最后输出最终结果前通信即可。

多头注意力并行同理，由于多头本身就是相互独立的，因此每张卡上单独计算几个头即可。

2D / 2.5D 并行：将权重矩阵和激活值都进行分割。

3D 并行：结合DP, PP和TP。

## Flash Attention

### FlashAttention 1

**Motivation:** Transformer中的运算可以分为两类，**矩阵乘法**和**位运算**。其中矩阵乘法都是计算密集型的，即从运算耗费的时间是瓶颈。而位运算都是IO密集型的，即从HBM获取数据的时间占用是瓶颈。然而目前，GPU的发展并非是IO速度和运算速度同比例增长的。运算速度的增长要超过IO速度的增长，并且人们针对运算做了很多的优化，有高效矩阵乘法这类算法。因此这就导致IO的时间反倒成为了整个过程中占用较大时间的部分。因此Flash attention的目的是想避免直接对大块数据进行运算，因为这样需要将数据从HBM加载再存回HBM消耗大量IO时间。因此想要将数据分块运算利用SRAM高速读取。

对于位运算分块运行是完全没有问题的，问题在于**Softmax**运算需要整个数据来计算，所以这里的想法是：将矩阵拆成B个小块，每个小块都计算最大值$M(x_1), M(x_2)$... 然后矩阵内部用这个最大值计算一个exponential和softmax结果，也就是$e^{x_{1,0}-M(x_1)}, e^{x_{1,1}-M(x_1)}$... 然后每次合并几个小块，合并的过程中要比较两个小块的最大值，并且取较大的那个作为新的最大值, 即$M(x)=max(M(x_1), M(x_2))$。然后所有的exponential都需要乘以$e^{M(x_i)-M(x)}$，其中$M(x_i)$是这个小块原先的最大值，$M(x)$是合并后总体的最大值，同理原来小块的softmax分母也需要乘以这个值。然后逐个块合并即可。

### FlashAttention 2

指出在FlashAttention1中每一个小block计算完都除以softmax分母是没有必要的，因此**只有在合并最后一个块的时候才除以这个分母**。另外FlashAttention 1中是将KV作为外层循环，而Q作为内层循环。在FlashAttention 2中将这两者反过来了，**Q作为外层**。这样更加合理因为对于一个Q可以一次性遍历所有的K和V，直接计算出当前的输出矩阵O。如果把Q放在内层则需要把每个块的输出先保存回HBM。

## KV Cache

将之前token计算过的Key 和 Value保存下来，在计算新token时直接读取之前的key和value，节省计算时间。

在第一个token时，KV Cache为空，这时需要计算所有的key value值，是compute bound。

在计算第二个到最后一个token时，KV Cache有值，只需要从Cache中读取，并且将新生成的key value不断添加即可，是memory bound。

## vLLM

### Paged Attention

*KV Cache*通常会在显存中预分配连续的一段空间，这就有可能导致显存中存在额外的碎片空间无法被利用。因此借鉴了OS中的虚拟内存动态分配空间。

- 虚拟内存：每个进程都分成不同的页，在该进程使用到某些页的时候将他们加载到物理内存上，这些内存在物理内存上可能是碎片化的，但在他们各自的虚拟内存中是连续的，且只包含这一个进程的信息。

Paged Attention中存在几个概念：一条请求=一个进程；Logical KV block=虚拟内存，每个block中能够保存的token数是固定的，在vLLM中为16；block table=映射表；Physical KV block=物理内存。

**实际上，Logical KV block只是一个抽象的概念，在硬件层面是不存在的。vLLM做的事是将Token sequence拆分成小块存入显存中，这样就减小了出现碎片的概率。在计算时，通过映射表将Logical block映射到Physical block获取需要加载的数据，然后在连续的Logical block中进行计算。**

虽然现在能够处理更多prompt，但是由于没有预留额外空间，如果某一时刻显存满了该怎么办？

- 两个原则：1. 先来的先被服务；2. 如果有抢占的需要，后来的先被抢占。
- 在GPU资源不足的情况下，可以先释放后来的请求的KV cache，等到先来的请求处理完毕再重新加载。

  1. Swapping：将需要被释放的block全部转移到CPU上，等到GPU空闲了再加载回来。
  2. Recomputation：当GPU资源充足时，将被卸载到CPU上的资源重新加载到GPU上完成推理。

**多卡情景：**

- vLLM存在一个中心调度器，负责管理每张卡上的映射表。
- 在分布式计算时，中心调度器将映射表广播到每张卡上，分别进行KV block的管理。
- 在张量并行情况下，每张卡只负责模型中的几个layer，所以这时候每张卡的映射表都应该是相同的，只不过Physical KV block中保存的元素不同。

### Continuous Batching

多条请求合并成一个batch进行处理的时候，由于每条请求的completion长度不一定相同，那么结束一个batch的处理需要等到batch中最长的那条生成完毕。这样会导致大量的无效使用。

Continuous Batching提出在一个batch生成的过程中，可以在生成完毕的序列处直接插入一条新的序列而无需等待最长的序列完成。
