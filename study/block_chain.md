# token 序列如何映射到 block 列表：以 GPT-2、block=512、TP=8 为例

本文继续沿用 `decode_example.md` 的设定，用 GPT-2 风格 decoder-only 模型解释：

```text
输入 token 序列
  -> 如何切成 512 token 一个 block
  -> 如何形成 block list / block table
  -> TP=8、模型 8 个 attention head 时，每张 GPU 存什么
  -> 每个 block 是否知道自己在序列中的编号
```

这里的 `block` 指 KV cache block，也就是推理服务里常说的 paged attention block，不是 Transformer layer/block。

## 1. 先区分两个 block

大模型里很容易把两个概念混在一起：

```text
Transformer block:
  模型结构里的层，比如 GPT-2 有 12 层 block。
  每层包含 self-attention + MLP。

KV cache block:
  推理系统为了管理 KV cache，把 token 序列按固定长度切块。
  例如每 512 个 token 一个 block。
```

本文说的是第二种：KV cache block。

ASCII 图：

```text
模型结构上的 block:

input hidden
    |
    v
+------------------+
| Transformer blk0 |
+------------------+
    |
    v
+------------------+
| Transformer blk1 |
+------------------+
    |
   ...


KV cache 管理上的 block:

token positions:
0 1 2 ... 511 | 512 ... 1023 | 1024 ...
+-------------+--------------+---------+
| KV block 0  | KV block 1   | block 2 |
+-------------+--------------+---------+
```

## 2. 输入文本先变成 token 序列

原始 prompt：

```text
今年4月初，英国《简氏防务周刊》首飞放出清晰卫星图像和分析称，东大正在建造一艘尺寸巨大的新型补给舰，舰长达到了约270米，远大于现役901综合补给舰的240米。而若是以901综合补给舰满排4.8万吨的吨位作为对比，在建的这艘新型补给舰满排可能会达到6.5万吨甚至7万吨，达到与航母“山东舰”（约6.5万吨）一个级别。
```

tokenizer 会把它变成 token ids：

```text
text
  |
  v
[tok_0, tok_1, tok_2, ..., tok_N-1]
```

每个 token 有两个重要编号：

```text
token_id: 这个 token 在词表里的编号，比如 318、50256 等
position: 这个 token 在当前序列里的位置，比如 0、1、2、3...
```

两者不是一回事：

```text
token_id 由词表决定
position 由它在当前 prompt/generated sequence 中的位置决定
```

GPT-2 使用绝对位置编码，因此第 `i` 个 token 会加上第 `i` 个 position embedding：

```python
x = token_embedding[token_id] + position_embedding[position]
```

## 3. 假设 block size = 512

假设输入 prompt 被 tokenizer 后有 `N = 1230` 个 token。

按 512 个 token 一个 block 切：

```text
block_size = 512

logical block 0: positions 0    ~ 511
logical block 1: positions 512  ~ 1023
logical block 2: positions 1024 ~ 1229
```

ASCII 图：

```text
position:

0                                                    511
|------------------------------------------------------|
                 logical block 0

512                                                 1023
|------------------------------------------------------|
                 logical block 1

1024                 1229
|----------------------|
     logical block 2, partially filled
```

换成一条链：

```text
sequence S
  |
  v
+-------------------+     +-------------------+     +-------------------+
| logical block 0   | --> | logical block 1   | --> | logical block 2   |
| pos 0..511        |     | pos 512..1023     |     | pos 1024..1229    |
+-------------------+     +-------------------+     +-------------------+
```

这里的 `logical block 0/1/2` 是序列内部的逻辑编号，不一定等于底层显存里的物理 block 编号。

## 4. logical block 和 physical block

推理引擎通常不会要求一个序列的 KV cache 在显存中连续摆放。它会维护一个 block table：

```text
logical block id -> physical block id
```

例如：

```text
sequence_id = 42

logical block 0 -> physical block 17
logical block 1 -> physical block 81
logical block 2 -> physical block 23
```

ASCII 图：

```text
                    block table for sequence 42

logical chain:      [0] --------> [1] --------> [2]
                     |            |             |
                     v            v             v
physical blocks:   [17]          [81]          [23]

GPU memory pool:

+-----+-----+-----+-----+-----+-----+-----+
|  0  |  1  | ... | 17  | ... | 23  | ... |
+-----+-----+-----+-----+-----+-----+-----+
                    ^           ^
                    |           |
                 block0      block2

+-----+-----+-----+-----+-----+-----+-----+
| ... | 81  | ... | 94  | ... | 120 | ... |
+-----+-----+-----+-----+-----+-----+-----+
        ^
        |
     block1
```

为什么要这样做？

```text
1. 不同请求长度不同，连续分配容易产生碎片。
2. decode 每次只追加一个 token，需要按需扩容。
3. 多个请求 batch 在一起时，block table 可以快速找到每个请求的 KV。
```

这和操作系统的页表有点像：

```text
逻辑地址 -> 页表 -> 物理页

token position -> block table -> physical KV block
```

## 5. 每个 block 知道自己在序列中的编号吗？

短答案：

```text
通常不需要，也通常不靠 block 自己知道。
```

更准确地说：

```text
block 本体:
  主要是一段 KV tensor 存储空间。
  它可以有 physical_block_id，用于显存池管理。
  但它里面的 K/V 张量通常不自带“我是序列里的第几个 block”这个语义。

block table / metadata:
  知道某个 sequence 的 logical block 顺序。
  知道 logical block 0/1/2 分别映射到哪些 physical block。

position_ids:
  知道每个 token 在序列中的绝对位置。
  GPT-2 依赖这个位置来加 position embedding。
```

所以答案是：

```text
block 自己不必知道“我在序列中排第几”；
推理引擎的 block table 知道；
模型计算时的 position_ids 知道每个 token 的绝对位置。
```

用图表示：

```text
                 sequence metadata
                 +-----------------------------+
                 | seq_id = 42                 |
                 | length = 1230               |
                 | block_size = 512            |
                 | block_table = [17, 81, 23]  |
                 +-----------------------------+
                              |
                              v
           +------------------+------------------+
           |                  |                  |
           v                  v                  v
    physical block 17  physical block 81  physical block 23
    +-------------+    +-------------+    +-------------+
    | KV tensors  |    | KV tensors  |    | KV tensors  |
    | no seq pos  |    | no seq pos  |    | no seq pos  |
    +-------------+    +-------------+    +-------------+
```

但有些工程实现可能会在 block metadata 里缓存更多信息，例如：

```text
owner sequence id
logical block id
ref count
filled token count
hash prefix
```

这些是调度和缓存复用需要，不是 attention 数学上必须要求的。

## 6. position 如何映射到 block

给定一个 token 的绝对位置 `pos`：

```python
block_size = 512
logical_block_id = pos // block_size
offset_in_block = pos % block_size
```

例如：

```text
pos = 0
  logical_block_id = 0
  offset_in_block = 0

pos = 511
  logical_block_id = 0
  offset_in_block = 511

pos = 512
  logical_block_id = 1
  offset_in_block = 0

pos = 1229
  logical_block_id = 2
  offset_in_block = 205
```

再通过 block table 找到物理 block：

```python
physical_block_id = block_table[logical_block_id]
```

完整映射：

```text
pos
 |
 | pos // 512
 v
logical_block_id
 |
 | block_table[logical_block_id]
 v
physical_block_id
 |
 | pos % 512
 v
offset_in_block
```

ASCII：

```text
token position 1229
       |
       | 1229 // 512 = 2
       v
logical block 2
       |
       | block_table[2] = 23
       v
physical block 23
       |
       | 1229 % 512 = 205
       v
slot 205 inside physical block 23
```

## 7. TP=8，模型只有 8 个头

现在设定：

```text
tp = 8
num_heads = 8
local_heads = num_heads / tp = 1
```

也就是说每张 GPU 只负责一个 attention head：

```text
GPU/rank 0 -> head 0
GPU/rank 1 -> head 1
GPU/rank 2 -> head 2
GPU/rank 3 -> head 3
GPU/rank 4 -> head 4
GPU/rank 5 -> head 5
GPU/rank 6 -> head 6
GPU/rank 7 -> head 7
```

输入 token 序列不会按 token 维度切给不同 GPU。通常每张 GPU 都知道同一个请求的 token 位置和 block table，只是每张 GPU 只保存自己 head 的 K/V。

图：

```text
same sequence positions:

pos: 0 1 2 ... 511 | 512 ... 1023 | 1024 ... 1229
     \____________/ \_____________/ \_____________/
       block 0          block 1          block 2


rank 0 stores KV for head 0 only:

block table: [17, 81, 23]
physical block 17: K/V for head 0, pos 0..511
physical block 81: K/V for head 0, pos 512..1023
physical block 23: K/V for head 0, pos 1024..1229


rank 1 stores KV for head 1 only:

block table: [17, 81, 23]  # logical mapping same, local memory pool may use local ids
physical block 17: K/V for head 1, pos 0..511
physical block 81: K/V for head 1, pos 512..1023
physical block 23: K/V for head 1, pos 1024..1229

...

rank 7 stores KV for head 7 only.
```

工程上，physical block id 可以是“每个 rank 本地显存池的 id”，也可以由调度器维护成一致编号。重点不是编号是否数值相同，而是：

```text
同一个 logical block 在每个 rank 上都能找到对应的 local KV block。
```

## 8. 每张卡上的 KV block 形状

假设：

```text
batch = 1
num_layers = 12
num_heads = 8
tp = 8
local_heads = 1
head_dim = 64
block_size = 512
```

每张 GPU、每一层、每个 physical block 存：

```text
K block: [block_size, local_heads, head_dim]
V block: [block_size, local_heads, head_dim]
```

代入：

```text
K block: [512, 1, 64]
V block: [512, 1, 64]
```

也有人把维度排成：

```text
[local_heads, block_size, head_dim]
```

或者为了 kernel 访存效率再做 layout 优化。语义一样：一个 block 里放的是 512 个 token 在本 rank 负责的 head 上的 K/V。

完整层级：

```text
rank 0
  layer 0
    physical block 17
      K: [512, 1, 64]
      V: [512, 1, 64]
    physical block 81
      K: [512, 1, 64]
      V: [512, 1, 64]
  layer 1
    physical block 17
      K: [512, 1, 64]
      V: [512, 1, 64]
  ...

rank 1
  layer 0
    physical block 17
      K/V for head 1
  ...
```

注意每一层都有自己的 KV cache。不是所有层共用一份 KV。

## 9. prefill 时如何填 block

prefill 输入完整 prompt，例如 1230 tokens。

逻辑步骤：

```text
1. tokenizer 得到 input_ids: [1, 1230]
2. 生成 position_ids: [0, 1, 2, ..., 1229]
3. 分配 3 个 logical blocks
4. 为每个 logical block 分配 physical blocks
5. 模型逐层计算 K/V
6. 把每个 token 的 K/V 写入对应 block slot
```

伪代码：

```python
BLOCK_SIZE = 512


def position_to_block_slot(pos, block_table):
    logical_block_id = pos // BLOCK_SIZE
    offset = pos % BLOCK_SIZE
    physical_block_id = block_table[logical_block_id]
    return physical_block_id, offset


def prefill_write_kv(layer_id, rank, k, v, block_table):
    # k/v: [seq_len, local_heads, head_dim]
    seq_len = k.shape[0]

    for pos in range(seq_len):
        physical_block_id, offset = position_to_block_slot(pos, block_table)

        kv_cache[layer_id][rank].K[physical_block_id][offset] = k[pos]
        kv_cache[layer_id][rank].V[physical_block_id][offset] = v[pos]
```

实际内核不会用 Python for-loop 一个 token 一个 token 写，这里只是表达映射关系。

ASCII：

```text
positions:    0   1   2        511 | 512 513       1023 | 1024 ... 1229
              |   |   |         |  |  |   |          |  |   |
              v   v   v         v  |  v   v          v  |   v
logical:    block 0                 | block 1           | block 2
              |                     |                   |
              v                     v                   v
physical:  block 17              block 81            block 23
slot:        0   1   2        511    0   1        511     0 ... 205
```

## 10. decode 时如何追加 token

decode 每次只来一个新 token。

假设 prefill 后长度是 1230，下一个生成 token 的位置就是：

```text
pos = 1230
```

计算：

```text
logical_block_id = 1230 // 512 = 2
offset = 1230 % 512 = 206
```

所以它继续写到 logical block 2，也就是 physical block 23 的 slot 206。

图：

```text
before decode:

logical block 2 / physical block 23

slot:  0   1   2        205 206 207 ... 511
       +---+---+--- ... +---+---+--- ... +---+
       | K | K | K      | K |   |       |   |
       +---+---+--- ... +---+---+--- ... +---+
                            ^
                            next write offset


after generating one token:

slot:  0   1   2        205 206 207 ... 511
       +---+---+--- ... +---+---+--- ... +---+
       | K | K | K      | K | K |       |   |
       +---+---+--- ... +---+---+--- ... +---+
```

当某次 decode 的位置来到 1536：

```text
1536 // 512 = 3
1536 % 512 = 0
```

说明需要分配新 block：

```text
logical block 3 -> new physical block, say 44
```

block table 从：

```text
[17, 81, 23]
```

变成：

```text
[17, 81, 23, 44]
```

ASCII：

```text
old chain:

[logical 0] -> [logical 1] -> [logical 2]
     |              |              |
     v              v              v
   phys17         phys81         phys23


append at pos 1536:

[logical 0] -> [logical 1] -> [logical 2] -> [logical 3]
     |              |              |              |
     v              v              v              v
   phys17         phys81         phys23         phys44
```

## 11. attention 如何读 block list

decode 某个 token 时，query 是当前位置的 Q。它要和历史所有 K 做 attention。

如果当前位置是 `pos = 1230`，它需要读：

```text
positions 0..1230 的 K/V
```

也就是：

```text
block 0: slots 0..511
block 1: slots 0..511
block 2: slots 0..206
```

通过 block table：

```text
block 0 -> phys17
block 1 -> phys81
block 2 -> phys23
```

ASCII：

```text
query position 1230
       |
       v
read keys/values from:

logical block 0        logical block 1        logical block 2
slots 0..511           slots 0..511           slots 0..206
      |                     |                       |
      v                     v                       v
physical 17            physical 81              physical 23
```

每张 rank 只读自己的 head：

```text
rank 0:
  Q head0 attends to K/V head0 in [phys17, phys81, phys23]

rank 1:
  Q head1 attends to K/V head1 in [phys17, phys81, phys23]

...

rank 7:
  Q head7 attends to K/V head7 in [phys17, phys81, phys23]
```

每个 rank 独立得到一个 local context：

```text
rank 0 -> context for head0
rank 1 -> context for head1
...
rank 7 -> context for head7
```

然后进入 output projection，并通过 `all_reduce` merge。

## 12. TP=8 下的计算和 block 关系

对于一个 Transformer layer：

```text
输入 hidden state 每张卡都有一份
QKV projection 按 head 切分
每张卡生成自己的 Q/K/V
K/V 写入本 rank 的 KV block
Q 读取本 rank 的 KV block list
attention 得到本 rank 的 local context
out_proj 后 all_reduce 合并
MLP 也做类似的 tensor parallel merge
```

ASCII：

```text
same token position p, same sequence block_table

                         hidden[p]
                             |
         +-------------------+-------------------+
         |                   |                   |
         v                   v                  ...
      rank0               rank1               rank7
      head0               head1               head7
         |                   |                   |
         v                   v                   v
  read KV blocks      read KV blocks      read KV blocks
  for head0           for head1           for head7
         |                   |                   |
         v                   v                   v
  local context0      local context1      local context7
         |                   |                   |
         v                   v                   v
  partial out0        partial out1        partial out7
         \                   |                   /
          \                  |                  /
           +----------- all_reduce ------------+
                             |
                             v
                     merged hidden output
```

重点：

```text
block list 管 sequence 维度；
TP 管 head/hidden/intermediate 维度；
两者是正交的。
```

也就是说：

```text
不是 GPU0 负责 block0，GPU1 负责 block1；
而是 8 张 GPU 都会看到同一个 block chain，
只是每张 GPU 负责不同 head 的 KV 内容。
```

## 13. block 是否包含 position embedding

这个问题要分两层看。

GPT-2 的计算：

```python
x[pos] = token_embedding[input_ids[pos]] + position_embedding[pos]
k[pos] = x[pos] @ W_k
v[pos] = x[pos] @ W_v
```

所以写入 KV cache 的 K/V 已经是“用 position 参与计算后的结果”。但是 K/V 张量本身一般不会额外存一个 `pos` 字段。

因此：

```text
K/V 数值受到 position 的影响；
KV block 存储结构通常不显式记录每个 slot 的绝对 position；
推理引擎通过 sequence length、block table、slot offset 推导 position。
```

图：

```text
token_id + position_id
        |
        v
   hidden state
        |
        v
      K / V  --------------------+
        |                        |
        v                        v
 stored in KV block       block itself does not need
                           to store position_id field
```

## 14. 一个最小 block table 数据结构

可以用这样的结构理解：

```python
from dataclasses import dataclass, field


BLOCK_SIZE = 512


@dataclass
class SequenceState:
    seq_id: int
    length: int = 0
    block_table: list[int] = field(default_factory=list)

    def locate(self, pos: int) -> tuple[int, int, int]:
        logical_block_id = pos // BLOCK_SIZE
        offset = pos % BLOCK_SIZE
        physical_block_id = self.block_table[logical_block_id]
        return logical_block_id, physical_block_id, offset


seq = SequenceState(
    seq_id=42,
    length=1230,
    block_table=[17, 81, 23],
)

print(seq.locate(1229))
# (2, 23, 205)
```

追加 token：

```python
def append_token_slot(seq: SequenceState, allocate_block):
    pos = seq.length

    if pos % BLOCK_SIZE == 0:
        seq.block_table.append(allocate_block())

    logical_block_id, physical_block_id, offset = seq.locate(pos)
    seq.length += 1
    return logical_block_id, physical_block_id, offset
```

这段逻辑表达了：

```text
序列长度决定新 token 的 position；
position 决定 logical block 和 offset；
block table 决定 physical block；
block 本体只是被写入的存储页。
```

## 15. 和那段补给舰文本结合看

假设那段中文 prompt token 化后比较长，超过 512 个 token，那么它就会跨多个 block。

例如：

```text
block 0:
  今年4月初，英国《简氏防务周刊》...

block 1:
  ...901综合补给舰的240米。而若是以901...

block 2:
  ...6.5万吨甚至7万吨，达到与航母“山东舰”...
```

这只是逻辑切块。模型并不会因为文本被切成 block 就“忘记”前面的内容。

attention 看到的是：

```text
当前 query position
  attends to
所有历史 positions 的 K/V
```

block 只是 KV cache 的分页管理方式：

```text
从语义上看：这仍然是一条连续 token 序列
从存储上看：它被拆成多个 512-token KV block
```

ASCII：

```text
semantic sequence:

今年4月初 -> 简氏防务周刊 -> 270米 -> 901 -> 4.8万吨 -> 6.5万吨 -> 山东舰
      \_______________________________________________________________/
                         one continuous context


storage layout:

+---------------------+  +---------------------+  +---------------------+
| KV block 0          |  | KV block 1          |  | KV block 2          |
| positions 0..511    |  | positions 512..1023 |  | positions 1024..    |
+---------------------+  +---------------------+  +---------------------+
```

## 16. prefill 和 decode 下的 block chain 总图

prefill：

```text
prompt tokens: 0 ... 1229

allocate:

logical blocks:   L0 ----------> L1 ----------> L2
                   |             |              |
physical blocks:  P17           P81            P23

write:

P17 slots 0..511
P81 slots 0..511
P23 slots 0..205
```

decode：

```text
step 1:
  new token position = 1230
  L2, P23, slot 206

step 2:
  new token position = 1231
  L2, P23, slot 207

...

when position = 1536:
  need new logical block L3
  allocate new physical block P44
  block_table = [17, 81, 23, 44]
```

TP=8：

```text
rank0: same block chain, KV for head0
rank1: same block chain, KV for head1
rank2: same block chain, KV for head2
rank3: same block chain, KV for head3
rank4: same block chain, KV for head4
rank5: same block chain, KV for head5
rank6: same block chain, KV for head6
rank7: same block chain, KV for head7
```

## 17. 最后一遍回答核心问题

问题：

```text
输入的 token 序列，如何映射到 block 列表？
```

答案：

```text
token 的绝对 position 决定 logical block:
  logical_block_id = position // 512
  offset = position % 512

sequence 的 block table 决定 physical block:
  physical_block_id = block_table[logical_block_id]
```

问题：

```text
tp=8，模型只有8个头，怎么分？
```

答案：

```text
每张 GPU 负责一个 attention head。
token 序列和 block table 不按 GPU 切。
每张 GPU 都有同一条 sequence 的 block 映射，
但每张 GPU 的 KV block 里只存自己那个 head 的 K/V。
```

问题：

```text
每个 block 知道自己在序列中的编号吗？
```

答案：

```text
通常 block 本体不知道，也不需要知道。
block table 知道 logical block 顺序；
sequence length 知道当前写到哪里；
position_ids 知道 token 的绝对位置；
physical block 只是 KV 存储页。
```

一句话总结：

```text
序列顺序不在 KV block 自己身上，而在 sequence metadata / block table / position_ids 里；
KV block 只是承载一段 token 的 K/V 张量，TP=8 时这段 K/V 还会按 head 分片到 8 张 GPU 上。
```
