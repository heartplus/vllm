# OffloadingConnector 实现分析

本文分析这组代码：

```text
vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py
vllm/distributed/kv_transfer/kv_connector/v1/offloading/
  common.py
  scheduler.py
  worker.py
  events.py
  metrics.py
```

这套实现是 vLLM v1 KV connector 的一种：把 GPU KV cache block 异步搬到外部介质，并在后续请求命中时再搬回 GPU。默认外部介质是 CPU KV cache，但 connector 本身不直接绑定 CPU；CPU、tiering、自定义后端都通过 `OffloadingSpec` 抽象接入。

## 1. 总体定位

`OffloadingConnector` 是 `KVConnectorBase_V1` 的实现，同时实现 `SupportsHMA`：

```text
Scheduler 侧:
  负责查 offload cache、构建 load/store job、维护 job 状态、
  处理 worker 回传的完成信息、生成 KV cache events 和 metrics。

Worker 侧:
  负责把真实 GPU KV cache tensor 规范化成 block 粒度的 byte tensor，
  调用具体 OffloadingWorker 异步执行 GPU <-> offload medium 传输。
```

入口类 `OffloadingConnector` 很薄，它根据 `KVConnectorRole` 构造不同实现：

```python
spec = OffloadingSpecFactory.create_spec(vllm_config, kv_cache_config)

if role == KVConnectorRole.SCHEDULER:
    self.connector_scheduler = OffloadingConnectorScheduler(spec)
elif role == KVConnectorRole.WORKER:
    self.connector_worker = OffloadingConnectorWorker(spec)
```

因此真正逻辑分布在：

```text
OffloadingConnectorScheduler: scheduler-side state machine
OffloadingConnectorWorker:    worker-side transfer executor
OffloadingManager:            scheduler-side offload cache manager abstraction
OffloadingWorker:             worker-side concrete transfer abstraction
```

默认 spec 来自 `OffloadingSpecFactory`：

```text
kv_connector_extra_config["spec_name"] 缺省为 "CPUOffloadingSpec"
CPUOffloadingSpec -> CPUOffloadingManager + CPUOffloadingWorker
TieringOffloadingSpec 也已注册
```

## 2. 核心抽象

### 2.1 `OffloadingSpec`

`OffloadingSpec` 是 connector 和具体 offload 后端之间的工厂/配置层。

它负责：

```text
1. 读取 vllm_config / kv_cache_config / kv_connector_extra_config。
2. 计算每个 KV group 的 gpu_block_size。
3. 读取 hash_block_size，并校验 block hash 粒度和 GPU block 粒度可对齐。
4. 解析 offloaded block size：
   offloaded_block_size = gpu_block_size * block_size_factor
5. 提供 scheduler 侧 manager：
   get_manager() -> OffloadingManager
6. 提供 worker 侧 worker：
   get_worker(kv_caches) -> OffloadingWorker
```

其中 `block_size_factor` 非常关键：

```text
GPU block:       vLLM KV cache manager 管理的 block
offloaded block: offload 介质上的 block

offloaded_block_size = gpu_block_size * block_size_factor
```

如果用户在 `kv_connector_extra_config` 里设置 `block_size`，它会作为 offloaded block 的 token 数，并要求能被 GPU block size 整除。

### 2.2 `OffloadKey`

`OffloadKey` 是 offload cache 的逻辑 key：

```text
OffloadKey = block_hash + group_idx
```

也就是说同一个 token prefix 的 block hash，在不同 KV cache group 里会形成不同 key。原因是不同 group 的 KV 数据布局、attention 类型、block size 可能不同，不能混用。

`RequestOffloadState.update_offload_keys()` 会根据 `Request.block_hashes` 和每个 group 的 `hash_block_size_factor` 增量生成 key：

```text
hash_block_size_factor = offloaded_block_size / hash_block_size

一个 offloaded block 可能覆盖多个 hash block。
这个 offloaded block 的 key 使用该 chunk 最后一个 hash block 的 hash。
```

### 2.3 `LoadStoreSpec` 和 `GPULoadStoreSpec`

`LoadStoreSpec` 是抽象传输地址。

Scheduler 不直接接触 tensor，只构建两端的 spec：

```text
load:
  src_spec: offload medium 侧地址，由 manager.prepare_load() 生成
  dst_spec: GPU block 地址，GPULoadStoreSpec

store:
  src_spec: GPU block 地址，GPULoadStoreSpec
  dst_spec: offload medium 侧地址，由 manager.prepare_store() 生成
```

`GPULoadStoreSpec` 里有：

```text
block_ids:      实际 GPU KV cache block id 列表
group_sizes:    每个 KV group 对应多少 GPU block
block_indices:  每个 group 的第一个逻辑 GPU block index
```

`group_sizes` 和 `block_indices` 用来支持多 KV group，以及 offloaded block 比 GPU block 更大的情况。

### 2.4 Metadata

`common.py` 里定义 scheduler 和 worker 之间传递的 metadata。

Scheduler -> Worker：

```python
@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    load_jobs: dict[int, TransferJob]
    store_jobs: dict[int, TransferJob]
    jobs_to_flush: set[int] | None = None
```

其中 `TransferJob` 是：

```text
req_id
src_spec
dst_spec
```

Worker -> Scheduler：

```python
@dataclass
class OffloadingWorkerMetadata(KVConnectorWorkerMetadata):
    completed_jobs: dict[int, int]
    transfer_stats: TransferStats
```

每个 worker 对完成的 job 上报 `{job_id: 1}`。多 worker 场景下，metadata 聚合时会把同一个 `job_id` 的计数相加；scheduler 只有在完成计数达到 `world_size` 后，才认为这个 job 全局完成。

## 3. `OffloadingConnector` 入口类

`offloading_connector.py` 基本是 role dispatch wrapper。

### 3.1 Worker 侧入口

```text
register_kv_caches()
  -> connector_worker.register_kv_caches()

register_cross_layers_kv_cache()
  -> connector_worker.register_cross_layers_kv_cache()

handle_preemptions()
  -> connector_worker.handle_preemptions()

start_load_kv()
  -> connector_worker.start_kv_transfers()

get_finished()
  -> connector_worker.prepare_store_kv()
  -> connector_worker.get_finished()

build_connector_worker_meta()
  -> connector_worker.build_connector_worker_meta()
```

这里有一个重要设计：`save_kv_layer()`、`wait_for_layer_load()`、`wait_for_save()` 都是空实现。

这说明 offloading connector 不像某些 per-layer connector 那样在每层 forward 时搬运 KV。它是 block 级、job 级传输：

```text
load job:
  在 scheduler 分配 GPU block 后生成，
  worker 在 step 开始时提交异步 load。

store job:
  在 scheduler step 末尾根据已计算 token 生成，
  worker 在 get_finished() 阶段先缓存起来，
  下一 step 开始时再提交。
```

`wait_for_save()` 为空的注释也说明 store defer 不放在 `wait_for_save()`，而是放在 `get_finished()`，因为有些 step 可能跳过 `wait_for_save()`。

### 3.2 Scheduler 侧入口

```text
on_new_request()
get_num_new_matched_tokens()
update_state_after_alloc()
build_connector_meta()
has_pending_push_work()
update_connector_output()
request_finished()
request_finished_all_groups()
take_events()
reset_cache()
get_kv_connector_stats()
```

这些方法几乎全部直接转给 `OffloadingConnectorScheduler`。

### 3.3 KV cache layout

`get_required_kvcache_layout()` 返回 `"HND"`。

这表示该 connector 要求 KV cache 使用 HND layout。worker 侧会进一步把具体 tensor 重新解释为 `(num_blocks, page_size_bytes)` 的 byte view，以便按照 block 搬运。

## 4. Scheduler 初始化

`OffloadingConnectorScheduler.__init__()` 主要建立以下状态：

```text
self.config
  SchedulerOffloadConfig.from_spec(spec)

self.manager
  spec.get_manager()

self._req_status
  req_id -> RequestOffloadState

self._jobs
  job_id -> TransferJobStatus

self._current_batch_load_jobs
  当前 scheduler step 要发给 worker 的 load jobs

self._current_batch_jobs_to_flush
  当前 step 需要 worker 强制 wait 的 job ids

self._current_batch_allocated_block_ids
  当前 step 新分配的 GPU block ids

self._block_id_to_pending_jobs
  block_id -> 仍在使用该 GPU block 数据的 pending store job ids

self._blocks_being_loaded
  如果启用 GPU prefix caching，用来避免重复加载同一批 offload keys
```

### 4.1 `SchedulerOffloadConfig`

`SchedulerOffloadConfig.from_spec()` 将 `OffloadingSpec` 转成 scheduler 需要的 per-group 配置：

```text
group_idx
gpu_block_size
offloaded_block_size
hash_block_size_factor
sliding_window_size_in_blocks
alignment_block_count
is_eagle_group
kv_event_group_spec
```

其中：

```text
FullAttentionSpec:
  sliding_window_size_in_blocks = None

SlidingWindowSpec:
  sliding_window_size_in_blocks = ceil(sliding_window / offloaded_block_size)

MambaSpec:
  sliding_window_size_in_blocks = 1
```

`alignment_block_count` 是 hybrid attention 优化。例如某些模型同时有 full attention group 和 SWA group，load 命中会被 full attention 的 offloaded block size 对齐约束住；SWA group 中每个对齐段前面那些永远不会被 load 命中的 block 可以不 store。

`is_eagle_group` 用来处理 EAGLE/MTP draft attention group。decode 阶段 trailing block 不稳定，所以 store/load 都需要排除这个 volatile trailing block。

### 4.2 `RequestOffloadState`

每个 request 进入 scheduler 后会创建 `RequestOffloadState`。

它保存：

```text
req
req_context
offloading_context
group_states
max_offload_tokens
num_locally_computed_tokens
transfer_jobs
deferred_lookup_start_time
```

每个 group 的 `RequestGroupState` 保存：

```text
offload_keys:
  已知 block hash 对应的 offload keys

block_ids:
  scheduler 分配给 request 的 GPU block ids

next_stored_block_idx:
  下一个需要 store 的 offloaded block index

num_hit_blocks:
  请求开始时的 offload/GPU prefix 命中 block 数
```

`max_offload_tokens` 来自 request 的 `kv_transfer_params["max_offload_tokens"]`，是实验性参数，用来限制这个请求最多 offload 多少 token。

## 5. 请求生命周期

下面按一个 request 的生命周期看 scheduler/worker 如何协作。

### 5.1 新请求：`on_new_request`

Scheduler 调用：

```python
req_context = ReqContext(
    req_id=request.request_id,
    kv_transfer_params=request.kv_transfer_params,
)
offloading_context = manager.on_new_request(req_context)
```

然后创建 `RequestOffloadState` 放到 `_req_status`。

`manager.on_new_request()` 可以返回 `RequestOffloadingContext`，其中最重要的是 `policy`：

```text
BLOCK_LEVEL:
  只 offload 新计算出来的 block。
  对于已经通过 prefix cache / offload 命中的 block，不重复 store。

REQUEST_LEVEL:
  offload 整个请求上下文，包括命中的 prefix block。
  tiering 这类后端可能需要完整 request KV。
```

### 5.2 查询 offload 命中：`get_num_new_matched_tokens`

Scheduler 在决定能否复用外部 KV 时调用：

```python
get_num_new_matched_tokens(request, num_computed_tokens)
```

这个方法返回：

```text
(num_hit_tokens, will_load_async)

num_hit_tokens:
  可以从 offload cache 加载的 token 数。
  None 表示后端需要更多时间判断，scheduler 之后再问。

will_load_async:
  True 表示后面会创建异步 load job。
```

流程：

```text
1. 如果 request 还有 transfer_jobs 未完成，返回 (None, False) 延迟调度。
2. 更新 offload_keys。
3. 记录 num_locally_computed_tokens。
4. 如果 request.skip_reading_prefix_cache，命中数为 0。
5. 否则调用 _lookup() 查询 offload manager。
6. 更新 num_hit_blocks。
7. touch 命中 key，维护后端 LRU/ARC 等 recency。
```

### 5.3 `_lookup()` 的命中策略

`_lookup()` 要回答的是：

```text
在 num_locally_computed_tokens 之后，还有多少 token 可以从 offload medium 加载？
```

它不是简单按一个 group 查 prefix。原因是 vLLM 可能有多 KV group：

```text
full attention group:
  必须从当前位置开始连续 prefix 命中。

sliding window / mamba group:
  只要求末尾 sliding window 范围命中。
```

因此 `_lookup()` 的策略是：

```text
1. 先查 full attention groups，再查 sliding window groups。
2. full attention 使用 _maximal_prefix_lookup():
   从前往后找连续命中的 offloaded blocks。
3. sliding window 使用 _sliding_window_lookup():
   从后往前找最后一段满足 window size 的连续命中。
4. 多 group 会互相收紧 max_hit_size_tokens。
   如果某个 group 缩小了命中边界，可能需要重新验证已经查过的 SWA group。
5. 如果 manager.lookup() 返回 RETRY 或 HIT_PENDING，可能返回 None，
   告诉 scheduler 延迟请求。
6. 如果开启 GPU prefix caching，且同一 key 正在被别的 load job 加载，
   当前请求也会被延迟，避免重复 load。
```

`LookupResult` 的语义：

```text
HIT:
  block 在 offload cache 中且可读。

HIT_PENDING:
  block 已存在但还没 ready。对连续命中来说算 hit，但需要等待。

RETRY:
  后端状态暂时不确定，需要之后重试；不算确定 hit。

MISS:
  未命中。
```

EAGLE/MTP group 的特殊点：

```text
decode 阶段 trailing block volatile，没有稳定 hash。
store 时 storable_blocks() 会少算最后一个 block。
load 查询时也会额外验证并扣掉 trailing block。
```

### 5.4 分配 GPU block 后创建 load job：`update_state_after_alloc`

当 scheduler 已经为将要加载的 KV 分配 GPU blocks 后，调用：

```python
update_state_after_alloc(request, blocks, num_external_tokens)
```

如果 `num_external_tokens == 0`，直接返回。

否则它会为每个 KV group：

```text
1. 统计 num_cached_tokens = local tokens + external tokens。
2. 找出哪些 GPU blocks 已经本地计算/已有 hash。
3. 找出还缺哪些 GPU blocks，需要从 offload medium 加载。
4. 生成 keys_to_load。
5. 收集 dst_block_ids、group_sizes、block_indices。
```

然后：

```python
src_spec = manager.prepare_load(keys_to_load, req_context)
dst_spec = GPULoadStoreSpec(dst_block_ids, group_sizes, block_indices)
```

再分配一个 `job_id`，把 job 放进：

```text
_current_batch_load_jobs[job_id]
_jobs[job_id]
req_status.transfer_jobs
```

load job 的 scheduler-side invariant 是：

```text
一个 request 只有在没有任何 pending transfer job 时才能发 load。
```

如果开启 `_blocks_being_loaded`，还会把本次 `keys_to_load` 加进去，用来阻止其他请求重复加载同一 key。

### 5.5 step 末尾构建 metadata：`build_connector_meta`

`build_connector_meta(scheduler_output)` 是 scheduler 每个 step 末尾构建发给 worker 的控制消息。

它做几件事：

```text
1. _update_req_states(scheduler_output)
   更新每个 request 的 offload_keys 和 GPU block_ids。

2. manager.on_schedule_end()
   通知后端一个 scheduler step 结束。

3. 处理 preemption:
   如果 preempted request 有 pending store jobs，需要 flush。

4. 处理 GPU block 被复用:
   如果某个 pending store job 还在读一个 GPU block，
   但这个 block 在当前 step 又被分配给别的用途，需要 flush。

5. 调用 _build_store_jobs() 生成本 step 要 store 的 jobs。

6. 返回 OffloadingConnectorMetadata:
   load_jobs
   store_jobs
   jobs_to_flush
```

`jobs_to_flush` 是安全栅栏。它告诉 worker 对指定 job 调用 `wait()`，确保相关 GPU block 数据在被覆盖或释放前已经完成传输。

### 5.6 构建 store jobs：`_build_store_jobs`

`_build_store_jobs()` 依据本 step scheduler 输出，决定哪些新算出的 KV block 应该 offload。

核心计算：

```text
num_tokens_after_batch = req.num_computed_tokens + num_scheduled_tokens
num_offloadable_tokens = min(num_tokens_after_batch, req.num_tokens)
```

然后依次施加约束：

```text
max_offload_tokens:
  如果 request kv_transfer_params 指定，则最多 offload 到这个 token。

offload_prompt_only:
  默认 True，只 offload prefill/prompt block，不 offload decode 生成的 block。

storable_blocks():
  按 offloaded_block_size 取整。
  EAGLE/MTP decode 阶段排除 volatile trailing block。

sliding window / mamba skip:
  block_id 为 0 表示 skip/null/stale，不 store。

alignment_block_count:
  SWA group 中永远无法服务 load hit 的 block 不 store。
```

确定候选 keys 后，调用：

```python
store_output = manager.prepare_store(new_offload_keys, req_context)
```

manager 可能：

```text
1. 返回 None:
   offload 介质空间不足或无法驱逐，scheduler 记录 allocation failure。

2. 返回 keys_to_store 为空:
   这些 block 已经在 offload cache 中，不需要 store。

3. 返回 keys_to_store + store_spec:
   scheduler 继续构建 GPU src spec。
```

对每个实际要 store 的 key，scheduler 会收集源 GPU block ids：

```text
一个 offloaded block 可能由 block_size_factor 个 GPU block 组成。
因此 src_block_ids 会收集这个 offloaded block 覆盖的多个 GPU block。
```

然后构造：

```python
src_spec = GPULoadStoreSpec(src_block_ids, group_sizes, block_indices)
dst_spec = store_output.store_spec
```

store job 的 scheduler-side invariant 是：

```text
一个 request 可以同时有多个 store jobs。
但如果这个 request 已经有 pending job，它们必须都是 store。
load 和 store 不会在同一个 request 上并发混杂。
```

为了保护还没完成 store 的 GPU block 数据：

```text
sliding window block ids:
  store job 创建时就登记到 _block_id_to_pending_jobs。
  因为 SWA/Mamba block 可能在 request finish 前就被 KV manager 回收。

non-sliding-window block ids:
  request 运行期间由 ref_cnt 保护，不会被提前释放。
  只有 request_finished() 时才登记。
```

### 5.7 Worker 提交 load/store

Worker 侧接收 `OffloadingConnectorMetadata` 后分两类处理。

`start_kv_transfers()`：

```text
1. 先提交上一 step 延迟的 store jobs。
2. 再提交当前 metadata.load_jobs。
```

`prepare_store_kv()`：

```text
不立刻提交 store。
只把 metadata.store_jobs 缓存到 _unsubmitted_store_jobs。
```

为什么 store 要延迟？

代码注释说明：store 被推迟到下一 engine step 开始，目的是让 offloading 在 token sampling 相关传输之后启动，避免拖慢 token generation。

整体时序可以简化为：

```text
step N scheduler:
  build load jobs for current loads
  build store jobs for newly computed KV

step N worker:
  submit load jobs now
  queue store jobs, do not submit yet

step N+1 worker start:
  submit queued store jobs
  submit new load jobs
```

`handle_preemptions()` 也会先提交未提交的 store jobs，然后根据 `jobs_to_flush` wait，保证 preemption/block reuse 前的数据一致性。

### 5.8 Worker 完成回报：`get_finished` 和 `build_connector_worker_meta`

Worker 调用底层 `OffloadingWorker.get_finished()` 获取已完成 transfer。

对每个 `TransferResult`：

```text
1. assert success。
2. 判断 job_id 是否在 _load_jobs：
   是 load，则把 req_id 加到 finished_recving。
   store 不返回 finished_sending。
3. 记录 transfer_size / transfer_time 到 load 或 store stats。
4. 在 OffloadingWorkerMetadata.completed_jobs 里 mark_completed(job_id)。
```

`OffloadingConnector.get_finished()` 返回：

```text
(finished_sending, finished_recving)

finished_sending:
  对 store 永远为空。

finished_recving:
  load 完成的 req_ids。
  base scheduler 用它恢复等待 remote KV 的 request，
  或释放 load 中 abort 的 request。
```

随后 `build_connector_worker_meta()` 把从上次调用以来完成的 job ids 和传输统计一次性返回给 scheduler，并清空 worker 本地累计。

### 5.9 Scheduler 处理 worker 完成：`update_connector_output`

Scheduler 收到聚合后的 worker metadata 后：

```text
1. 将 transfer_stats 转成 OffloadingConnectorStats。
2. 遍历 completed_jobs。
3. 对每个 job_id，减小 TransferJobStatus.pending_count。
4. pending_count 归零时，这个 job 才算所有 worker 都完成。
```

job 全局完成后：

```text
store job:
  manager.complete_store(keys, req_context)
  之后这些 key 才真正变成可 load。

load job:
  manager.complete_load(keys, req_context)
  释放 manager 对这些 key 的防驱逐保护。
  如果启用 _blocks_being_loaded，也从集合里移除。
```

然后清理：

```text
_block_id_to_pending_jobs 中的登记
_jobs[job_id]
req_status.transfer_jobs
```

如果 request 已经 finished，并且没有任何 pending transfer job，就从 `_req_status` 删除。

## 6. Worker 如何操作真实 KV cache

`OffloadingConnectorWorker` 的难点不在 job 状态，而在把 vLLM 的各种 KV cache layout 规范化。

### 6.1 `register_kv_caches`

输入是：

```text
layer_name -> torch.Tensor 或 list[torch.Tensor]
```

worker 会根据 `kv_cache_config.kv_cache_groups` 遍历每个 layer。

对 attention layer：

```text
1. 取原始 layer_kv_cache tensor。
2. 计算 page_size_bytes 和 real_page_size_bytes。
3. 用 torch.tensor([], dtype=int8).set_(...) 基于同一 storage 创建 byte view。
4. byte view 形状是 (num_blocks, page_size_bytes)。
```

对 Mamba layer：

```text
1. kv_caches[layer_name] 是 state tensor 列表。
2. 从第一个 state tensor 的 storage 重建原始 byte tensor。
3. view 成 (num_blocks, page_size_bytes)。
```

最终构建：

```python
CanonicalKVCaches(
    tensors=[CanonicalKVCacheTensor(...), ...],
    group_data_refs=[[CanonicalKVCacheRef(...), ...], ...],
)
```

其中：

```text
tensors:
  去重后的真实 byte tensor 列表。

group_data_refs:
  每个 KV group 要搬运哪些 tensor 中的哪些 page。
```

这层 canonicalization 的目的：

```text
让具体 OffloadingWorker 只关心：
  第几个 GPU block
  每个 block 的 byte page
  每个 group 对应哪些 tensor ref

而不用关心原始 attention backend 的 KV layout。
```

### 6.2 packed layout

有些模型的 KV cache tensor 使用 packed layout，`kv_cache_tensor.block_stride > 0`。

代码分支会：

```text
1. 根据 tensor.stride(0) 拿 manager-block stride。
2. 用 as_strided 创建 (num_blocks, block_stride) 的 packed tensor。
3. 所有 group 都引用这个 packed tensor。
```

这里 page stride 不再简单等于 `page_size_bytes`，而是使用底层 packed tensor 的真实 block stride。

### 6.3 cross-layers KV cache

`register_cross_layers_kv_cache()` 支持跨 layer 合并的 KV cache tensor。

它先通过 attention backend 验证：

```text
num_blocks 必须在物理维度 0。
```

然后把整个 storage 解释为：

```text
(num_blocks, page_size_bytes * num_layers)
```

因为 cross-layers layout 当前只支持单个 KV group，所以 `group_data_refs` 里只有一个 group 和一个 ref。

## 7. Manager 的职责

Scheduler 只维护 request/job 状态，不直接管理 offload medium 的空间。空间分配、驱逐、ready 状态都由 `OffloadingManager` 实现。

以默认 `CPUOffloadingManager` 为例：

```text
lookup(key):
  查询 key 是否在 CPU cache 中。
  已存且 ready -> HIT
  已分配但 store 未完成 -> HIT_PENDING
  不存在 -> MISS

prepare_load(keys):
  找到对应 CPU blocks。
  增加 ref_cnt，把 block 标成 non-evictable。
  返回 CPULoadStoreSpec。

complete_load(keys):
  减 ref_cnt。
  ref_cnt 归零后重新变成 evictable。

prepare_store(keys):
  过滤已经存在的 key。
  如启用 store_threshold，过滤低复用 key。
  必要时驱逐 evictable blocks。
  分配 CPU blocks，插入 policy，但尚未 ready。
  返回 keys_to_store + CPULoadStoreSpec。

complete_store(keys):
  store 成功后将 block 标成 ready/evictable。
  之后 lookup 才会返回可读 hit。
```

CPU manager 还支持 LRU/ARC policy、KV events、cache usage metrics、store threshold metrics。

## 8. 事件系统

`events.py` 负责把 offloading manager 的原始事件转成 vLLM `KVCacheEvent`。

manager 原始事件只有：

```text
keys
medium
removed
```

但外部 KV-aware consumer 可能需要更完整的 block 信息，例如：

```text
block_hashes
parent_block_hash
token_ids
block_size
lora_id / lora_name
group_idx
kv_cache_spec_kind
kv_cache_spec_sliding_window
```

因此 `OffloadingEventsTracker` 在 `_build_store_jobs()` 阶段、request 还活着时调用 `record_store()`，为每个 offload key 保存 `BlockStored` 所需 payload。

之后 `take_events()` 做转换：

```text
store event:
  如果有完整 metadata，输出自描述 BlockStored。
  如果没有，输出 placeholder BlockStored。

remove event:
  如果有 metadata，把一个 offloaded chunk fan out 成 constituent block hashes。
  如果没有，输出 placeholder BlockRemoved。
```

限制：

```text
self_describing_kv_events 只有在 enable_kv_cache_events 也开启时才生效。
完整 metadata 当前只支持 full attention group。
sliding window / SSM group 仍使用 placeholder payload。
```

## 9. Metrics

metrics 分两层。

### 9.1 worker transfer stats

Worker 完成 transfer 时记录：

```text
load bytes
load time
load size histogram samples
store bytes
store time
store size histogram samples
```

这些先进入 `OffloadingWorkerMetadata.transfer_stats`，scheduler 收到后转换为 `OffloadingConnectorStats`。

### 9.2 scheduler/manager stats

Scheduler 自己记录：

```text
vllm:kv_offload_lookup_sync_delay_seconds
vllm:kv_offload_lookup_async_delay_seconds
vllm:kv_offload_allocation_failure
```

具体 spec/manager 也可以返回自己的 stats。例如 CPU spec 定义：

```text
vllm:kv_offload_cpu_cache_usage_perc
vllm:kv_offload_cpu_allocation_size
vllm:kv_offload_stores_skipped
```

`OffloadPromMetrics` 会根据 metric metadata 动态创建 Prometheus counter/gauge/histogram。

为了兼容旧 CPU offload metrics，它还会在 CPU spec 下同步写 deprecated metrics：

```text
vllm:kv_offload_total_bytes{transfer_type=...}
vllm:kv_offload_total_time{transfer_type=...}
vllm:kv_offload_size{transfer_type=...}
```

## 10. 一致性和并发控制

这套实现有几个重要 invariant。

### 10.1 job 完成必须等待所有 workers

`TransferJobStatus.pending_count` 初始化为 `world_size`。

每个 worker 完成后上报 `{job_id: 1}`。scheduler 聚合后递减：

```text
pending_count > 0:
  job 仍未全局完成。

pending_count == 0:
  才调用 manager.complete_load/store。
```

这样保证 tensor parallel / data parallel 相关 worker 都完成了自己的 shard 后，manager 状态才切换。

### 10.2 load/store 不在同一 request 上混跑

代码通过 assert 维护：

```text
发 load 前:
  req_status.transfer_jobs 必须为空。

发 store 前:
  如果已有 transfer_jobs，它们必须都是 store。
```

也就是说：

```text
一个 request 可以并发多个 store。
一个 request 同一时间最多一个 load。
load 不会和 store 在同一 request 上同时 pending。
```

### 10.3 store 延迟提交

store job 在 scheduler step 末尾生成，但 worker 不立即提交，而是在下一 step 开始提交。

好处：

```text
减少对当前 step token sampling / generation 关键路径的干扰。
```

代价：

```text
需要额外处理 preemption、block reuse、reset_cache 时的 flush。
```

### 10.4 GPU block 复用保护

pending store job 需要从 GPU block 读数据。如果这些 GPU block 被提前复用，store 可能读到错误数据。

实现用 `_block_id_to_pending_jobs` 做保护：

```text
sliding window / mamba:
  block 可能在 request finish 前被释放，所以 store job 创建时立刻登记。

full attention:
  request 运行期间 block ref_cnt 保护它不被释放。
  request_finished() 后如果 store 还没完成，再登记。
```

每个 step 末尾，如果当前新分配的 block id 和 pending jobs 使用的 block id 相交，就把这些 job 放入 `jobs_to_flush`，worker 会 wait。

### 10.5 reset_cache 的 stale job 处理

`reset_cache()` 会：

```text
1. assert 当前不在 schedule step 中。
2. 把所有 in-flight jobs 加入 _current_batch_jobs_to_flush。
3. 删除已 finished request 的状态。
4. manager.reset_cache() 清空 offload cache。
5. 活跃 request 的 next_stored_block_idx 归零，之后可重新 offload。
6. 清空 jobs 和 block_id tracking。
7. 设置 _stale_job_threshold = _job_counter。
8. reset event tracker。
```

之后 worker 可能仍会上报 reset 之前的 job completion。scheduler 用：

```text
if job_id < _stale_job_threshold:
    skip
```

丢弃旧 job 的完成回调，避免对已经 reset 的 manager 状态调用 `complete_load/store`。

## 11. 端到端时序图

### 11.1 命中并 load

```text
Scheduler.on_new_request
  -> manager.on_new_request
  -> create RequestOffloadState

Scheduler.get_num_new_matched_tokens
  -> RequestOffloadState.update_offload_keys
  -> _lookup
     -> manager.lookup(key)
  -> return (num_hit_tokens, True)

Scheduler.update_state_after_alloc
  -> manager.prepare_load(keys_to_load)
  -> create GPULoadStoreSpec(dst GPU blocks)
  -> create TransferJob(load)

Scheduler.build_connector_meta
  -> metadata.load_jobs contains job

Worker.start_load_kv
  -> OffloadingConnectorWorker.start_kv_transfers
     -> worker.submit_load(job_id, src_spec, dst_spec)

Worker.get_finished
  -> worker.get_finished()
  -> finished_recving.add(req_id)
  -> OffloadingWorkerMetadata.completed_jobs[job_id] = 1

Scheduler.update_connector_output
  -> pending_count -= worker_count
  -> manager.complete_load(keys)
  -> remove job
```

### 11.2 计算后 store

```text
Scheduler.build_connector_meta
  -> _update_req_states
  -> _build_store_jobs
     -> manager.prepare_store(keys)
     -> create GPULoadStoreSpec(src GPU blocks)
     -> create TransferJob(store)
  -> metadata.store_jobs contains job

Worker.get_finished
  -> prepare_store_kv(metadata)
     -> queue store job in _unsubmitted_store_jobs

Next Worker.start_load_kv / handle_preemptions
  -> submit queued store jobs first
  -> worker.submit_store(job_id, src_spec, dst_spec)

Worker.get_finished
  -> OffloadingWorkerMetadata.completed_jobs[job_id] = 1

Scheduler.update_connector_output
  -> pending_count reaches 0
  -> manager.complete_store(keys)
  -> key becomes loadable
```

## 12. 代码阅读要点

建议按这个顺序读：

```text
1. offloading_connector.py
   先看入口类如何按 role 转发。

2. common.py
   理解 scheduler <-> worker metadata 和 job completion 聚合。

3. scheduler.py
   重点看：
     SchedulerOffloadConfig.from_spec()
     RequestOffloadState.update_offload_keys()
     get_num_new_matched_tokens()
     _lookup()
     update_state_after_alloc()
     build_connector_meta()
     _build_store_jobs()
     update_connector_output()
     request_finished()
     reset_cache()

4. worker.py
   重点看：
     register_kv_caches()
     register_cross_layers_kv_cache()
     start_kv_transfers()
     prepare_store_kv()
     get_finished()

5. vllm/v1/kv_offload/base.py
   理解 OffloadingManager / OffloadingWorker 的接口契约。

6. vllm/v1/kv_offload/cpu/*
   看默认 CPU 后端如何实现 manager 和 worker。
```

## 13. 总结

`OffloadingConnector` 的核心设计是把 KV offload 拆成两条控制面：

```text
Scheduler 控制面:
  只操作 request、block hash、offload key、LoadStoreSpec 和 job 状态。
  负责决定何时 load/store、哪些 key 可命中、哪些 job 完成后能更新 manager。

Worker 数据面:
  只操作真实 GPU KV cache tensor 和具体 offload medium tensor/storage。
  负责异步提交 transfer、轮询完成、上报 job_id 和统计。
```

它通过 `OffloadingManager` 把 offload cache 的空间管理、驱逐、ready 状态和事件封装起来；通过 `OffloadingWorker` 把真实传输封装起来。connector 自己则主要维护 request 生命周期、load/store job 生命周期、跨 worker completion 聚合、GPU block reuse 安全栅栏、以及 metrics/events 的 glue logic。

