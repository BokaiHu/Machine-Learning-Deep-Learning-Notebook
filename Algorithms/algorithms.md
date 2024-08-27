## 修改与查询

- 不修改，区间查询：前缀和
- 区间修改，单点查询：差分，树状数组
- 单点修改，区间查询：树状数组
- 区间修改，区间查询：线段树

### 前缀和

遍历求出前缀和数组，之后求区间和时只需要找 `prefixSum[right] - prefixSum[left-1]`

### 差分

遍历所有区间修改的操作，构建差分数组：`diff[left]+=k`, `diff[right+1]-=k`

### 树状数组

lowbit操作：求出一个数从末尾开始的所有0以及第一个1构成的数字。求法：`d = x & (-x)`.

```python
class arrTree:
    def __init__(self, arr):
        self.size = len(arr)
        self.tree = [0] * (self.size + 1)
        for i in range(0, len(arr)):
            self.update(i+1, arr[i])

    def update(self, index, d):
        while index <= self.size:
            self.tree[index] += d
            index += index & (-index)
  
    def query(self, index):
        total = 0
        while index > 0:
            total += self.tree[index]
            index -= index & (-index)
        return total

    def query_range(self, left, right):
        return self.query(right) - self.query(left - 1)
```

## 并查集

用于判断不相交集合的查询，合并问题。由list pre[], func find(x)和func join(x, y)组成。

```python
class UnionSet:
    def __init__(self, arr):
        self.pre = arr
  
    def find(self, x):
        if self.pre[x] == x:
            return x
        else:
            self.pre[x] = self.find(self.pre[x])
            return self.pre[x]
  
    def join(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
          self.pre[px] = py
```

## 哈夫曼树和加权路径和

给定一个数组，每次从数组中取出最小的两个数相加记为新的节点存回数组里。这个新的节点作为最小两个数的父节点。

加权路径和：所有 叶子节点的层 * 叶子结点的值

```python
class HuffmanNode:
	def __init__(self, char, freq):
		self.freq = freq
		self.char = char
		self.left = None
		self.right = None

	def __lt__(self, other):
		return self.freq < other.freq

def build_huffman_tree(text):
	heap = []
	chr_freq = Counter(text)
	for char, freq in chr_freq.items():
		heapq.heappush(heap, HuffmanNode(char, freq))
	while len(heap) > 1:
		left = heapq.heappop()
		right = heapq.heappop()
		new_node = HuffmanNode(None, left.freq+right.freq)
		new_node.left = left
		new_node.right = right
		heapq.heappush(heap, new_node)
	return new_node
```

可用于数据压缩。
