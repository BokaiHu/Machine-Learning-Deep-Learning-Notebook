## Network Pruning

1. Evaluate the importance of a weight / neuron
2. Remove weights / neurons that have small importance.
   * Usually the performance will drop a little.
3. Fine-tune the rest of the network on the dataset.
4. Go back to 1; or end.

## Knowledge Distillation

用Student Network学习Teacher Network的输出分布，例如{0: 0.7, 1: 0.2, 2: 0.1}。即使teacher是错误的也要学习。可以将一个ensemble model作为teacher，让student学习ensemble的输出来极大的提高运算效率。

- Using temperature for softmax: smoother classes after softmax.
  - Example: $y_1=100,\,y_2=10,\,y_3=1\Rightarrow y_1'=1,\,y_2'=0,\,y_3'=0$; If using temperature = 100: $y_1=1,\,y_2=0.1,\,y_3=0.01\Rightarrow y_1'=0.56,\,y_2'=0.23,\,y_3'=0.21$, 能够学得更好。

## Parameter Quantization

1. 用更少的bits来储存一个值: 32 bits -> 8 bits 降低参数精度，performance不会掉很多。
2. Weight clustering: 对网络的参数进行clustering，用每个cluster的均值来代替这个cluster内的所有参数。

## Architecture Design

Example: Depthwise separable convolution.

## Dynamic Computation

- Dynamic depth
  - 每一层都给一个output，minimize所有的loss。
- Dynamic width
  - 同一个network中最后一层用不同数量的neuron做判断。
