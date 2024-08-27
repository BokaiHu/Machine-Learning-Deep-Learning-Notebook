# Convolution Neural Network (CNN)

## 归纳偏置

1. 局部性
2. 平移不变性

这两点是卷积核的性质导致的，由于卷积核是共享的所以特征不管移动到图片中的哪个部位输出都是相同的。

Transformer不具备归纳偏置，它学习到的是全局的特征之间的联系。所以如果有大量的数据，transformer能够学到更好的效果，但小数据集上一般CNN更佳。

## Features of CNN

- *神经元（Neuron）*：在这里，神经元指的是卷积层中的一个卷积块（convolutional block）。
- 感受野（Receptive field）：在卷积神经网络中，一个*神经元*只关注输入的一个小部分，这部分被称为感受野。
- 参数共享（Parameter sharing）：由于一个*神经元*识别一种模式，那么我们不需要在不同的感受野中为同一种模式使用不同的神经元。我们使用一个神经元在整个图像和不同的图像中检测这种模式。
- **一般来说，一个神经元对应一种模式。我们用很多神经元（过滤器或卷积层）扫描图像，看看是否出现某些模式。示例：3通道输入，64通道输出和$3\*3$核大小意味着我们想要从图像中找到64种模式（这些模式可以用$3*3$的块来表示）。**
- 特征图（Feature map）：一个卷积层的输出。例如：输入大小为(3, 256, 256)，一个卷积层的形状为$shape = (64, 3, 3, 3)$，且$padding = 1$，输出**（特征图）**大小为(64, 256, 256)。
- 池化（Pooling）：
  - 最大池化（Max pooling）：在滑动窗口中选择最大值（通常为$2*2$）。
  - 平均池化（Average pooling）：使用滑动窗口中所有数值的平均值。
  - 目的：减少计算量，可以用步幅不等于1的卷积层替代。

## 计算感受野

$$
R_s=R_{s-1}+(k_s-1)*\prod_{i=1}^{s-1} s_i \\
start_l=start_{l-1}+(\frac{k_l-1}{2}-p_l)*\prod_{i=1}^{s-1} s_i
$$

$R_s$为感受野大小，$start_l$为起点位置。

如果有空洞卷积，就当作是修改了卷积核的大小，使用$d=dilation(k_s-1)+1$替换$k_s$即可。

## ConvTranspose

参数解释：

- Kernel_size 同 Conv2d
- Stride其实是Conv2d中的dilation，即在相邻元素之间插入"0"来扩大图像。
- Padding ($p'$): 与Conv2d中不同，ConvTranspose2d中自带(Kernel_size-1)大小的padding，且实际padding的大小为$p=k-1-p'$. 例如Kernel_size=3，padding=1，此时实际padding大小为1，输出大小与输入大小相同。
- Output_padding：当有stride时，如果最后多出几列，此时这几列是不会参与运算的，从而导致结果缺少几行几列，所以这时加上output_padding在矩阵右侧和下侧加上几行几列数据。

# Object Detection 目标检测

## 基于ROI

### R-CNN系列

#### R-CNN

1. 先选择性裁剪出2000个候选框，resize到特定shape。
2. 将这些框全部输入CNN进行特征提取。
3. 将提取出的features通过SVM进行分类，如果判断某个部分属于某个类的概率很大，则挑出来。
4. 对挑选出的图块进行NMS，即计算它们的pairwise IoU，如果IoU超过某个阈值，则判断为同一个object，只保留其中之一。
5. 对保留下来的图块对应的features通过一个回归器计算出四个偏移量(dx, dy, dw, dh)，最后计算正确的bounding box.

#### Fast R-CNN

* 先选择性裁剪出2000个候选框。
* 将原图输入CNN进行特征提取，将2000个图块按照比例map到feature map上，并进行池化获得2000个固定7*7大小的feature map。
* 将提取出的features展平通过全连接层，最后输入两个不同的头，一个进行分类，一个进行回归。

#### Faster R-CNN

- Faster R-CNN = RPN + Fast R-CNN。只是使用RPN来进行候选框的选择，而非选择性裁剪。
- RPN: 使用全图经过CNN之后的feature map(假如是256x7x7)，先用1x1conv变成18x7x7, 这里的18就代表了9个anchor box的前/后景的概率，然后直接reshape经过softmax判断7x7中每个点是否为前景or后景。同时把feature map用1x1conv变成36x7x7，代表了每个点anchor box的预测位置和shape。
- 然后用这些信息对anchor box进行NMS筛选，最后和fast RCNN一样用预测的box位置映射到CNN的feature map上输入ROI pooling进行后续预测。

### YOLO系列

#### YOLOv1

1. 整张图片输入CNN中，得到一个feature map。
2. 将原始图片上目标的中心点transform到feature map上。
3. 将feature map分割为SxS块，输入分类和回归头进行预测。
4. 取出包含原始目标中心点的grid的结果计算loss。
