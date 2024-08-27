# Bidirectional Encoder Representations from Transformer

BERT的结构主要是由transformer encoder组成的。

General input format:

- [CLS]  $s_{1}$  $s_{2}$
- [CLS]  $s_{1,1}$  $s_{1, 2}$  [SEP]  $s_{2, 1}$  $s_{2,2}$  $s_{2,3}$

## Self-supervised learning

- What is self-supervised learning: Only have data $x$, no label $y$. Divide the data $x$ in two parts $x'$ and $x''$. Use $x'$ as the input and $x''$ as the ground truth label. Optimize the model to minimize the difference between model output $y$ and $x''$.
- Self-supervised learning in **BERT**:
  - Randomly mask some tokens in the input sequence or substitute them with random tokens, predict these masked words.
    input -> [BERT (Output at the corresponding position) -> Linear] -> Softmax -> output; Modules in the brackets are trained.
  - Next sentence prediction: concatenate two sentences with [CLS] at first and [SEP] between two sentences. Apply a binary classification using the output correspond to the [CLS]. *Proved not very useful.*
    input -> [BERT (Output correspond to the [CLS]) -> Linear] -> Softmax -> output; Modules in the brackets are trained.
  - In **ALBERT**, senetence order prediction.

## How to Use BERT?

BERT: pretrained; Linear: random initialized.

### Case I: Sequence Classification

- Input: 1 sequence; Output: 1 class.
- Example: Sentiment analysis
- Input a sequence start with the [CLS] token, pass the first output token of BERT (correspond to the [CLS] token) through a Linear layer and obtain a classification result.

### Case II: Sequence Translation

* Input: 1 sequence; Output: 1 sequence with equal length.
* Example: 词性标注
* Input a sequence start with the [CLS] token, pass output tokens of BERT (except the one correspond to the [CLS] token) through a Linear layer and obtain a classification result.

### Case III: Sequence Relation Recognition

* Input: 2 sequences; Output: 1 class.
* Example: Natural language inference
* Concatenate two sequences start with the [CLS] token and separate them using the [SEP] token, pass output tokens of BERT (correspond to the [CLS] token) through a Linear layer and obtain a classification result.

### Case IV: Content Extraction

* Input: 1 document, 1 query; Output: 1
* Example: Extraction-based question answering
* Two randomly initialized vector $\{v_1,\,v_2\}$. Concatenate the query and the document, start with the [CLS] token and separate them using the [SEP] token. 计算$v_1,\,v_2$与所有document token对应的输出计算inner product，经过Softmax后找最大概率的输出，即为答案的start和end.
