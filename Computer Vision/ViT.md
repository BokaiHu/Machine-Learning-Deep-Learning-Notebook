## Vision Transformer

1. 将输入分割成$r*r$的小块，这一步可以用一个$r*r$大小的卷积实现。(256x256 -> 256 * (16x16))
   1. 实现方法：PatchEmbedding：用一个kernel_size=16, stride=16的Conv2d，out_channel=embedding_dim, 因为相当于是把一个16x16方格里的元素转换成embedding_dim维的向量了。
   2. 例如输入大小是(1, 3, 1024, 1024), 经过一个(768, 3, 16, 16)的Conv2d，输出大小为(1, 768, 64, 64), 然后再Permute成(1, 64, 64, 768)，此时相当于seq_len=16x16，embeddim_dim=768.
2. 展平成$r^2$长度的embedding，在序列前concat一个CLS token，并且添加位置编码。
3. 经过$k$个encoder层，使用CLS token位置的输出经过全连接层得到分类结果。

## Swin Transformer

## Detection Tranformer (DETR)

1. 先使用一个CNN进行特征提取将输出转换为(bs, 2048, h, w)大小的feature map, 可以再用1x1的卷积层对2048做一个降维。
2. 将feature map展平输入transformer encoder中。
3. Decoder的输入是很多的Query token，确保输出的框的数量大于图片中可能的object数量。此时decoder输出的每一个token都是一个框的信息。
4. 最后经过reg head和cls head获得类别以及框的中心，宽度长度信息。
5. 训练过程：对于100个输出的框，使用匈牙利算法(二分图)进行匹配，匹配的cost是分类损失+框回归损失。分类损失就是Cross entropy，框回归损失为MAE+IoU loss。这里还添加了一个额外的类作为背景类。

## Masked AutoEncoder (MAE)

一种预训练方法，和NLP中的BERT相似。

1. 每张图片随机mask掉75%的block，将剩余可见的block输入ViT Encoder得到一系列Embeddings。
2. 然后将掩码标记 `<MASK>`和encoder的输出一起输入decoder。Decoder是一个Transformer Decoder结构，只不过没有Causal Attention(初始Transformer Decoder中需要Causal Attention是因为模型不能看到后面的字，但这里Decoder输入全部都是 `<MASK>`所以无所谓了？)。
3. 然后让Decoder预测这些masked patch，最后和原图像计算一个MSE Loss。

## SegmentAnythingModel (SAM)

- 模型设计：一个图像编码器负责编码图像，一个prompt 编码器负责编码prompt，然后这二者在Mask解码器中解码输出分割mask。
  - 图像编码器：MAE预训练的ViT，该模块在处理一张图片时只会被调用一次。
  - 提示编码器：两类提示：稀疏提示包括：框，点，和文本。密集提示包括：掩码。
- 数据引擎：模型辅助三阶段数据标注

**Prompt Encoder:**

- **提示编码器：**包含四个不同的Point Embedding(分别对应 前景点，背景点，左上角点和右下角点)。对点进行编码时，先把点进行映射，然后根据坐标计算**位置编码**，再把位置编码和对应的点Embedding权重相加即可。还有一个额外的padding点，以及一个专门的not_a_point_embedding，如果需要padding则按照同样处理流程只不过这里加上not_a_point_embedding的权重即可。
- **如何生成Positional Embedding**：先归一化放缩到[-1, 1]之间，然后乘以一个随机高斯初始化的matrix。输入为(1, 1, 2)shape, 乘以(2, 128)的随机高斯矩阵得到(1, 1, 128)的positional embedding, 然后在最后一个维度上把sin(pe), cos(pe)做concat得到(1, 1, 256)的positional embedding。然后和对应的Point embedding权重相加。**其实这里的位置编码和其他的位置编码不同，点自身带有位置信息，所以这里不需要我们加一个额外的位置信息进来，其实只是对这个点做了一个升维而已。**
- 这里的positional embedding还负责生成图像的位置编码，做法就是生成一个n * n大小的matrix，其中元素为点的坐标，从左上角到右下角的坐标为(0.5, 0.5)到(n-0.5, n-0.5)。然后同上方法对每个点进行处理。

**Mask Decoder:**

- 主体是一个transformer decoder，目前已有的元素：
  - Image embedding(来自Image Encoder, shape=(1, 64, 64, 256))
  - Sparse Prompt Embedding(来自Prompt Encoder (bs, 3, 256) = Point Embedding (bs, 1, 256) + Box Embedding (bs, 2, 256))
  - Dense Prompt Embedding(来自Prompt Encoder, shape=(1, 64, 64, 256))
  - Image Positional Embedding(来自Prompt Encoder, shape=(1, 64, 64, 256))。
- 首先定义两组额外的token：
  - 【IoU token】 (shape=(bs, 1, 256))
  - 【Mask token】(shape=(bs, 4, 256), 包括Single-mask的(bs, 1, 256)和Multi-mask的(bs, 3, 256)，同时支持这两种情况的输出)。
  - 将【这两个token】和【Sparse Prompt Embedding】拼接起来得到一个 (bs, 8, 256) shape的token sequence作为Query。
  - 将【Image Embedding】和【Dense Prompt Embedding】相加作为Key和Value。
- 然后将【相加后的Image Embedding】和【拼接后的Token Sequence】输入transformer。
- 最后模型的输出为【Query：Shape=(1, 8, 256)】，【Key：Shape=(1, 64, 64, 256)】。
- Query中的[:, 1:5, :] (Shape=(4, 256)) 取出来经过一个MLP变为Shape=(4, 32)，Key reshape成(1, 256, 64, 64)经过降维上采样变成(1, 32, 256, 256)。两者相乘变成(1, 4, 256, 256)的四个mask输出。
- Query中的[:, 0, :]取出来经过MLP预测四个mask的IoU。

## Grounding
