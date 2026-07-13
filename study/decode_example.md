# 用 GPT-2 理解一次大模型推理：单机 8 卡、merge、prefill/decode 分离

本文用 GPT-2 风格的 decoder-only Transformer 解释一段文本在大模型中是如何被“读入、理解上下文、继续生成”的。

示例 prompt：

```text
今年4月初，英国《简氏防务周刊》首飞放出清晰卫星图像和分析称，东大正在建造一艘尺寸巨大的新型补给舰，舰长达到了约270米，远大于现役901综合补给舰的240米。而若是以901综合补给舰满排4.8万吨的吨位作为对比，在建的这艘新型补给舰满排可能会达到6.5万吨甚至7万吨，达到与航母“山东舰”（约6.5万吨）一个级别。
```

这里讨论的是模型推理机制，不讨论这段军事内容本身是否真实。你可以把它当成一段输入材料，模型要基于其中的数字、实体和比较关系继续生成分析。

## 1. GPT-2 推理在做什么

GPT-2 是 decoder-only 模型。它的输入是一串 token，输出是“下一个 token 的概率分布”。

如果 prompt 是：

```text
舰长约270米，远大于901综合补给舰的240米，所以
```

模型不是一次性直接输出整段话，而是反复执行：

```text
输入已有 token -> 预测下一个 token -> 采样/选择一个 token -> 拼回输入 -> 再预测下一个 token
```

例如：

```text
所以 -> 这 -> 意味 -> 着 -> 该 -> 舰 -> ...
```

从外部看，模型好像在“推理”；从计算上看，它每一步都在做矩阵乘法、attention、softmax 和采样。所谓“推理过程”，就是这些 token 之间的信息被 attention 聚合，然后在最后一层 logits 上体现为某些下一个词更可能。

## 2. 这段话在模型里的信息流

这段 prompt 里有几类关键信息：

```text
时间：今年4月初
来源：英国《简氏防务周刊》
证据：清晰卫星图像和分析
主体：东大正在建造新型补给舰
长度：约270米
对比对象：901综合补给舰，240米，满排4.8万吨
推测吨位：6.5万吨甚至7万吨
参照：山东舰，约6.5万吨
```

模型生成下一段时，attention 会让后面的 token 关注前面的关键 token。例如，当模型生成“这意味着该舰可能...”时，它会更关注：

```text
270米
240米
4.8万吨
6.5万吨
7万吨
山东舰
补给舰
```

GPT-2 每一层大致做：

```python
x = token_embedding(input_ids) + position_embedding(position_ids)

for block in transformer_blocks:
    x = x + self_attention(layer_norm(x), kv_cache)
    x = x + mlp(layer_norm(x))

logits = lm_head(layer_norm(x))
next_token = sample(logits[:, -1, :])
```

注意最后只用 `logits[:, -1, :]`，也就是最后一个位置的 logits 来预测下一个 token。

## 3. prefill 和 decode

大模型推理通常分成两个阶段：

```text
prefill：一次性处理完整 prompt，建立 KV cache
decode：每次只处理新生成的 1 个 token，复用 KV cache
```

### 3.1 prefill 阶段

prefill 输入的是完整 prompt：

```python
input_ids.shape = [batch_size, prompt_len]
```

比如：

```text
prompt_len = 220 tokens
```

模型会并行处理这 220 个 token，并为每一层保存 attention 用的 K/V：

```python
kv_cache[layer].key.shape   = [batch, num_heads, prompt_len, head_dim]
kv_cache[layer].value.shape = [batch, num_heads, prompt_len, head_dim]
```

prefill 的特点：

```text
计算量大，因为 prompt 中所有 token 之间要做 causal attention
并行度高，因为 prompt_len 个位置可以一起算
吞吐优先，适合大矩阵乘法
```

极简代码：

```python
@torch.no_grad()
def prefill(model, input_ids):
    # use_cache=True 会返回 past_key_values，也就是 KV cache
    out = model(input_ids=input_ids, use_cache=True)
    kv_cache = out.past_key_values
    logits = out.logits
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    return next_token, kv_cache
```

### 3.2 decode 阶段

decode 输入的是刚生成出来的 1 个 token：

```python
input_ids.shape = [batch_size, 1]
```

它不再重复计算整个 prompt，而是把新 token 的 Q 和历史 KV cache 做 attention：

```python
@torch.no_grad()
def decode_one_token(model, token, kv_cache):
    out = model(
        input_ids=token,
        past_key_values=kv_cache,
        use_cache=True,
    )
    new_kv_cache = out.past_key_values
    logits = out.logits
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    return next_token, new_kv_cache
```

decode 的特点：

```text
每步计算量较小
必须串行，因为第 t+1 个 token 依赖第 t 个 token
更容易受显存带宽、通信延迟、batch 调度影响
```

完整生成流程：

```python
@torch.no_grad()
def generate(model, input_ids, max_new_tokens=128):
    token, kv_cache = prefill(model, input_ids)
    generated = [token]

    for _ in range(max_new_tokens - 1):
        token, kv_cache = decode_one_token(model, token, kv_cache)
        generated.append(token)

    return torch.cat(generated, dim=1)
```

## 4. 单机 8 卡怎么做分布式

一机 8 卡做 GPT-2 推理，常见方式是 tensor parallel，也就是把一个模型层切到 8 张 GPU 上共同算。

启动方式：

```bash
torchrun --nproc_per_node=8 decode_gpt2_tp.py
```

每个进程绑定一张 GPU：

```python
import os
import torch
import torch.distributed as dist

def init_dist():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank
```

单机 8 卡时：

```text
world_size = 8
rank 0 -> cuda:0
rank 1 -> cuda:1
...
rank 7 -> cuda:7
```

## 5. GPT-2 层如何切到 8 卡

GPT-2 block 主要有两块：

```text
self-attention
MLP
```

可以把 attention heads 分到 8 张卡上。假设有 16 个 heads：

```text
GPU 0: head 0, 1
GPU 1: head 2, 3
...
GPU 7: head 14, 15
```

如果是 GPT-2 small，只有 12 个 heads，不能均匀除以 8。工程上一般会选择 head 数能整除并行度的模型配置，或者把 tensor parallel size 设小一点，比如 4。本文为了讲清楚 8 卡逻辑，可以假设一个 GPT-2-like 配置：

```text
n_layer = 12
n_head = 16
n_embd = 1024
tp_size = 8
local_heads = 2
```

### 5.1 Attention 的切法

标准 attention：

```python
qkv = x @ W_qkv
q, k, v = split(qkv)
attn = softmax(q @ k.transpose(-1, -2) / sqrt(head_dim))
context = attn @ v
out = context @ W_o
```

tensor parallel 后：

```text
W_qkv 按输出维度切开，每张卡只算一部分 heads 的 Q/K/V
每张卡独立算自己的 local attention
W_o 按输入维度切开，每张卡算 partial output
最后 all_reduce，把 8 张卡的 partial output 加起来
```

示意代码：

```python
class TensorParallelAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, tp_rank, tp_size):
        super().__init__()
        assert num_heads % tp_size == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.local_heads = num_heads // tp_size
        self.head_dim = hidden_size // num_heads
        self.local_qkv_dim = 3 * self.local_heads * self.head_dim

        # Column parallel: W_qkv 的输出维度被切开
        self.qkv = torch.nn.Linear(hidden_size, self.local_qkv_dim, bias=True)

        # Row parallel: W_o 的输入维度被切开
        self.out_proj = torch.nn.Linear(
            self.local_heads * self.head_dim,
            hidden_size,
            bias=False,
        )

    def forward(self, x, past_kv=None):
        # x: [batch, seq_len, hidden]
        bsz, seq_len, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seq_len, self.local_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.local_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.local_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        context = context.transpose(1, 2).contiguous()
        context = context.view(bsz, seq_len, self.local_heads * self.head_dim)

        partial_out = self.out_proj(context)

        # merge: 8 张卡把各自的 partial_out 求和
        torch.distributed.all_reduce(partial_out, op=torch.distributed.ReduceOp.SUM)

        return partial_out, (k, v)
```

这里的 `all_reduce` 就是一次重要的 merge。

为什么是求和？因为原始的输出投影可以写成：

```text
out = [context_0, context_1, ..., context_7] @ W_o
```

把 `W_o` 按输入维度切开后：

```text
out = context_0 @ W_o_0
    + context_1 @ W_o_1
    + ...
    + context_7 @ W_o_7
```

每张卡算一项，最后相加，结果等价于完整模型。

### 5.2 MLP 的切法

GPT-2 MLP：

```python
h = gelu(x @ W_up + b_up)
out = h @ W_down + b_down
```

tensor parallel 后：

```text
W_up 按输出维度切开
每张卡得到一段 local intermediate
W_down 按输入维度切开
每张卡算 partial output
最后 all_reduce 求和
```

示意代码：

```python
class TensorParallelMLP(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, tp_size):
        super().__init__()
        assert intermediate_size % tp_size == 0
        self.local_intermediate = intermediate_size // tp_size

        self.up = torch.nn.Linear(hidden_size, self.local_intermediate)
        self.down = torch.nn.Linear(self.local_intermediate, hidden_size, bias=False)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.up(x))
        partial_out = self.down(h)

        # merge: MLP partial output 合并
        torch.distributed.all_reduce(partial_out, op=torch.distributed.ReduceOp.SUM)
        return partial_out
```

## 6. logits 怎么 merge

最后一层得到 hidden state 后，要经过 `lm_head` 得到词表 logits：

```python
logits = hidden @ W_vocab.T
```

有两种做法。

### 6.1 复制 lm_head

每张卡都有完整 `lm_head`，每张卡都能得到完整 logits：

```text
优点：实现简单
缺点：词表很大时浪费显存和计算
```

GPT-2 词表约 5 万，这种做法还能接受。

### 6.2 vocab parallel

把词表切到 8 张卡：

```text
GPU 0: token id 0       ~  6280
GPU 1: token id 6281    ~ 12560
...
GPU 7: token id 43967   ~ 50256
```

每张卡只算自己那段词表的 logits：

```python
local_logits = hidden @ local_lm_head.T
```

如果为了讲清楚，可以直接 `all_gather`：

```python
def gather_vocab_logits(local_logits):
    world_size = torch.distributed.get_world_size()
    parts = [torch.empty_like(local_logits) for _ in range(world_size)]
    torch.distributed.all_gather(parts, local_logits)
    return torch.cat(parts, dim=-1)
```

然后 rank 0 采样，再广播 token：

```python
def sample_and_broadcast_next_token(local_logits):
    rank = torch.distributed.get_rank()
    logits = gather_vocab_logits(local_logits)

    if rank == 0:
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    else:
        next_token = torch.empty(
            (local_logits.shape[0], 1),
            dtype=torch.long,
            device=local_logits.device,
        )

    torch.distributed.broadcast(next_token, src=0)
    return next_token
```

生产系统里不一定真的 gather 完整 logits。更高效的方式是做分布式 top-k/top-p，只通信候选 token，但逻辑上仍然是在把 8 张卡的局部概率结果合并成一次全局采样决策。

## 7. prefill/decode 在 8 卡中的 KV cache

tensor parallel 后，每张卡只保存自己 heads 的 KV cache。

假设：

```text
batch = 1
prompt_len = 220
num_layers = 12
num_heads = 16
head_dim = 64
tp_size = 8
local_heads = 2
```

那么每张卡每层保存：

```text
K: [1, 2, 220, 64]
V: [1, 2, 220, 64]
```

8 张卡合起来才是完整的：

```text
K: [1, 16, 220, 64]
V: [1, 16, 220, 64]
```

这点很关键：KV cache 不需要每卡保存完整 heads，否则显存会浪费 8 倍。decode 时，每张卡只用自己的 local K/V 算 local heads 的 attention，再通过输出投影后的 `all_reduce` 合并。

## 8. prefill/decode 分离是什么意思

有两种“分离”。

第一种是代码逻辑分离：

```text
prefill 函数负责完整 prompt
decode 函数负责逐 token 生成
```

这种在单机单进程也可以做。

第二种是服务架构分离：

```text
prefill worker：专门处理长 prompt，建立 KV cache
decode worker：专门接收 KV cache，做低延迟逐 token 生成
```

为什么要分离？

```text
prefill 是大块矩阵乘法，适合追求吞吐
decode 是小步串行计算，适合追求低延迟和高并发调度
两者混在一起时，长 prompt 可能阻塞短 decode
```

一机 8 卡上可以有几种部署方式：

```text
方式 A：8 卡都跑同一个 TP group，prefill 和 decode 顺序执行
方式 B：4 卡做 prefill TP group，4 卡做 decode TP group
方式 C：8 卡 prefill group + 8 卡 decode group，但这需要两套机器或时间复用
```

GPT-2 很小，通常没必要做复杂的 disaggregated serving。但对大模型，比如几十亿、上百亿参数，prefill/decode 分离就很有意义。

## 9. KV cache 从 prefill worker 交给 decode worker

如果 prefill 和 decode 是不同 worker，需要传 KV cache。

逻辑上：

```python
prefill_next_token, kv_cache = prefill_worker.run(prompt_ids)
decode_worker.load_kv_cache(request_id, kv_cache)
decode_worker.decode(request_id, prefill_next_token)
```

但 tensor parallel 下，KV cache 是分片的：

```text
prefill rank 0 的 local heads -> decode rank 0
prefill rank 1 的 local heads -> decode rank 1
...
prefill rank 7 的 local heads -> decode rank 7
```

如果 prefill TP size 和 decode TP size 相同，传输最简单：

```text
8 -> 8，同 rank 对同 rank
```

如果 prefill 用 8 卡，decode 用 4 卡，就要重新分片：

```text
prefill 每卡 2 heads
decode 每卡 4 heads

decode rank 0 接收 prefill rank 0 + rank 1 的 KV
decode rank 1 接收 prefill rank 2 + rank 3 的 KV
...
```

示意代码：

```python
def remap_kv_8_to_4(prefill_kv_by_rank):
    decode_kv_by_rank = []

    for decode_rank in range(4):
        src0 = 2 * decode_rank
        src1 = 2 * decode_rank + 1

        k0, v0 = prefill_kv_by_rank[src0]
        k1, v1 = prefill_kv_by_rank[src1]

        # head 维度是 dim=1: [batch, local_heads, seq, head_dim]
        k = torch.cat([k0, k1], dim=1)
        v = torch.cat([v0, v1], dim=1)
        decode_kv_by_rank.append((k, v))

    return decode_kv_by_rank
```

真实系统里会用 NCCL、CUDA IPC、RDMA 或者专门的 KV cache manager。核心原则不变：decode worker 拿到的 KV 分片必须和它负责的 attention heads 对齐。

## 10. 把整条推理链串起来

对这段补给舰 prompt，一次推理可以这样理解：

```text
1. tokenizer 把中文文本切成 token ids
2. prefill 阶段一次性读入所有 token
3. 每一层 attention 让模型建立“270米 vs 240米”“6.5万吨 vs 4.8万吨”“山东舰参照”等上下文联系
4. 每张 GPU 只负责一部分 attention heads 和 MLP intermediate
5. attention output 和 MLP output 通过 all_reduce merge
6. 最后一层 hidden state 经过 lm_head 得到下一个 token 的 logits
7. 如果词表被切分，需要 gather 或分布式 top-k 来 merge logits
8. rank 0 或采样模块选择下一个 token，并广播给所有 GPU
9. decode 阶段把这个 token 追加进 KV cache
10. 重复 decode，逐 token 生成完整回答
```

用一句话概括：

```text
prefill 负责把整段材料读懂并缓存上下文；decode 负责基于这个缓存逐字往后写；8 卡 tensor parallel 负责把每层计算拆开；merge 负责把拆开的局部结果重新合成一个等价于完整模型的结果。
```

## 11. 一个更完整的伪代码骨架

下面是单机 8 卡推理骨架，省略了权重加载和 GPT-2 block 的完整实现。

```python
import os
import torch
import torch.distributed as dist


def init_dist():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


@torch.no_grad()
def tp_prefill(model, input_ids):
    # input_ids 每张卡相同
    hidden, kv_cache = model.forward_prefill(input_ids)

    # 如果 lm_head 是 vocab parallel，local_logits 只是局部词表
    local_logits = model.local_lm_head(hidden)

    # merge logits，得到全局 next token
    next_token = sample_and_broadcast_next_token(local_logits)
    return next_token, kv_cache


@torch.no_grad()
def tp_decode(model, token, kv_cache):
    # token 每张卡相同
    hidden, kv_cache = model.forward_decode(token, kv_cache)
    local_logits = model.local_lm_head(hidden)
    next_token = sample_and_broadcast_next_token(local_logits)
    return next_token, kv_cache


def main():
    rank, world_size, local_rank = init_dist()
    assert world_size == 8

    model = build_tensor_parallel_gpt2(
        tp_rank=rank,
        tp_size=world_size,
    ).cuda()
    model.eval()

    prompt = """今年4月初，英国《简氏防务周刊》首飞放出清晰卫星图像和分析称，东大正在建造一艘尺寸巨大的新型补给舰，舰长达到了约270米，远大于现役901综合补给舰的240米。而若是以901综合补给舰满排4.8万吨的吨位作为对比，在建的这艘新型补给舰满排可能会达到6.5万吨甚至7万吨，达到与航母“山东舰”（约6.5万吨）一个级别。"""

    if rank == 0:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    else:
        input_ids = torch.empty((1, 0), dtype=torch.long, device="cuda")

    # 广播 shape 和 token。实际工程会封装得更干净。
    input_ids = broadcast_input_ids_from_rank0(input_ids)

    token, kv_cache = tp_prefill(model, input_ids)
    output_tokens = [token]

    for _ in range(128 - 1):
        token, kv_cache = tp_decode(model, token, kv_cache)
        output_tokens.append(token)

    output_ids = torch.cat(output_tokens, dim=1)

    if rank == 0:
        print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
```

## 12. 容易混淆的点

### merge 不是把 8 份文本拼起来

8 卡 tensor parallel 下，每张卡不是各自生成一段文字。它们是在共同生成同一个 next token。

```text
错误理解：GPU0 写第一句，GPU1 写第二句
正确理解：8 张 GPU 共同算出同一个位置的下一个 token
```

### prefill 不是生成，decode 才是逐 token 生成

prefill 也会产生第一个 next token 的 logits，但它的主要价值是建立 KV cache。后续大量 token 都是在 decode 阶段生成的。

### KV cache 是上下文记忆，不是自然语言摘要

KV cache 不是“模型总结出来的一段话”，而是每层 attention 的 key/value 张量。它保存的是模型继续计算所需的中间状态。

### GPT-2 不是中文强模型

原版 GPT-2 主要是英文语料训练，中文能力有限。这里用 GPT-2 是为了讲架构和推理流程。真实中文分析可以换成中文 tokenizer 和中文/多语大模型，但 prefill、decode、KV cache、tensor parallel、merge 这些机制仍然类似。

## 13. 最后再用这段话直观类比一次

这段 prompt 里有很多数字比较：

```text
270米 vs 240米
6.5万吨/7万吨 vs 4.8万吨
6.5万吨/7万吨 vs 山东舰约6.5万吨
```

prefill 阶段，模型把这些数字和实体关系编码进 KV cache。

decode 阶段，如果要继续生成：

```text
因此，这艘舰如果最终吨位接近7万吨，说明其补给能力和远洋伴随能力可能明显超过901型...
```

每生成一个 token，8 张 GPU 都会一起参与：

```text
local attention -> all_reduce merge -> local MLP -> all_reduce merge -> local logits -> logits merge/sample -> broadcast token
```

所以最终生成看似是一段连续分析，底层其实是一个很规整的循环：

```text
读完整 prompt，缓存上下文；
每次生成一个 token；
每层拆开并行算；
每个关键位置把结果 merge 回来；
直到达到结束 token 或最大长度。
```
