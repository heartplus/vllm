# 基于本地内存与本地 SSD 的分层 KV Cache 设计

## 1. 背景与目标

本文分析如何在 vLLM 中实现基于本机 DRAM 和本机 SSD 的分层 KV
Cache。目标架构是：

```text
GPU KV Cache
    ⇅ 异步 DMA
本机 DRAM / pinned memory
    ⇅ 异步文件 I/O
本机 NVMe SSD
```

该设计希望在 GPU KV Cache 容量有限时，通过更大但更慢的本机存储层
保存可复用的 prefix KV Cache，从而降低重复 prefill 的计算开销和请求
TTFT。

这里需要区分三类资源：

- GPU KV Cache：attention kernel 直接访问的 paged KV Cache。
- DRAM tier：GPU 和慢速存储之间的 staging buffer，同时缓存近期热点。
- SSD tier：容量更大的持久或半持久 prefix KV Cache。

SSD 数据不能直接供普通 attention kernel 使用，因此常规数据路径必须经过
DRAM：

```text
保存：GPU block → pinned DRAM block → SSD
加载：SSD → pinned DRAM block → GPU block
```

## 2. vLLM 当前已有能力

当前代码已经具备较完整的三级缓存框架。推荐在现有
`OffloadingConnector`、`TieringOffloadingSpec` 和 filesystem secondary
tier 的基础上扩展，而不是重新实现一套 KV Connector。

主要代码入口如下：

- [`vllm/config/cache.py`](../vllm/config/cache.py)：定义
  `kv_offloading_size` 和 `kv_offloading_backend`。
- [`vllm/config/vllm.py`](../vllm/config/vllm.py)：根据顶层 offloading
  配置自动选择 `OffloadingConnector`。
- [`vllm/v1/kv_offload/tiering/spec.py`](../vllm/v1/kv_offload/tiering/spec.py)：
  创建 DRAM primary tier 和 secondary tiers。
- [`vllm/v1/kv_offload/tiering/manager.py`](../vllm/v1/kv_offload/tiering/manager.py)：
  协调 GPU、DRAM 和 secondary tiers 之间的 promotion/cascade。
- [`vllm/v1/kv_offload/tiering/base.py`](../vllm/v1/kv_offload/tiering/base.py)：
  定义 secondary tier 接口。
- [`vllm/v1/kv_offload/tiering/fs/manager.py`](../vllm/v1/kv_offload/tiering/fs/manager.py)：
  现有本地文件系统 backend。
- [`docs/features/kv_offloading_usage.md`](../docs/features/kv_offloading_usage.md)：
  用户配置和调优说明。

现有 filesystem tier 已经实现：

- 按 prefix block hash 查询；
- 异步 lookup；
- 多线程异步读取和写入；
- GPU → DRAM → filesystem cascade；
- filesystem → DRAM → GPU promotion；
- 临时文件写入后通过 `os.replace` 原子发布；
- 按 TP rank 和 hash 前缀划分目录；
- 同配置 vLLM 实例之间复用 SSD 缓存；
- KV cache event 上报。

因此，若目标只是实现单机 GPU、DRAM、SSD 三级缓存，现有框架原则上已经
满足功能需求。

## 3. 推荐配置

可以使用以下配置启动：

```bash
PYTHONHASHSEED=0 vllm serve <model> \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "spec_name": "TieringOffloadingSpec",
      "cpu_bytes_to_use": 34359738368,
      "block_size": 64,
      "eviction_policy": "lru",
      "secondary_tiers": [
        {
          "type": "fs",
          "root_dir": "/mnt/nvme/vllm-kv",
          "n_read_threads": 32,
          "n_write_threads": 16
        }
      ]
    }
  }'
```

该配置表示：

- DRAM primary tier 总容量为 32 GiB；
- SSD 路径为 `/mnt/nvme/vllm-kv`；
- 每个 offloaded chunk 包含 64 tokens；
- DRAM 使用 LRU 驱逐；
- SSD load 使用更多读线程，因为读取直接影响 TTFT。

跨进程共享同一个 SSD 目录时，需要为所有进程设置一致的
`PYTHONHASHSEED`。否则不同进程可能为相同 token prefix 产生不同文件名。

## 4. 请求命中与数据传输流程

### 4.1 查询流程

调度器首先根据 token prefix、模型配置和 KV cache group 等信息生成
`OffloadKey`。随后按照以下顺序查询：

```text
查询 GPU prefix cache
        │ miss
        ▼
查询 DRAM primary tier
        │ miss
        ▼
异步查询 SSD secondary tier
        │ hit
        ▼
在 DRAM 分配并锁定目标 block
        │
        ▼
异步执行 SSD → DRAM
        │
        ▼
将 DRAM block 标记为 ready
        │
        ▼
异步执行 DRAM → GPU
        │
        ▼
继续剩余 prefill/decode
```

SSD lookup 或数据加载尚未完成时，scheduler 不能使用对应 KV Cache：

- `HIT`：数据已经在 DRAM，可以进入 GPU load。
- `HIT_PENDING`：数据已存在，但仍有写入操作未完成。
- `RETRY`：secondary tier 命中，promotion 已启动或 lookup 尚未完成。
- `MISS`：各层均未命中，或者 DRAM 无法为 promotion 分配空间。

### 4.2 保存流程

完整的 GPU KV block 生成后：

1. 在 DRAM primary tier 分配 block。
2. 异步复制 GPU KV Cache 到 pinned DRAM。
3. DRAM 写入完成后，将 block 提交给 SSD tier。
4. SSD tier 异步写入数据。
5. SSD 写入完成后，解除 DRAM block 的引用。
6. 若 DRAM 需要空间，可按照 LRU/ARC 进行驱逐。

建议默认只保存 prompt/prefill KV Cache。decode 阶段频繁生成小 block，
通常会造成较高写放大，而这些 block 的跨请求复用概率较低。

## 5. Secondary tier 扩展接口

正式扩展 SSD backend 时，建议继承
`SecondaryTierManager`，而不是修改 scheduler、attention backend 或 GPU KV
Cache manager。

主要接口包括：

```python
class SecondaryTierManager:
    def lookup(self, key, req_context):
        ...

    def submit_store(self, job_metadata):
        ...

    def submit_load(self, job_metadata):
        ...

    def get_finished_jobs(self):
        ...

    def on_new_request(self, req_context):
        ...

    def on_schedule_end(self, context):
        ...

    def on_request_finished(self, req_context):
        ...

    def shutdown(self):
        ...
```

其中：

- `lookup` 查询 block 是否存在。
- `submit_store` 提交 DRAM → SSD 异步写入任务。
- `submit_load` 提交 SSD → DRAM 异步读取任务。
- `get_finished_jobs` 返回已经完成的任务。
- request lifecycle 方法负责清理异步查询和请求级状态。

所有 scheduler-facing 方法都必须轻量且非阻塞。真实的 SSD lookup 和数据
传输应提交给线程池、io_uring 或独立 I/O 服务。

## 6. 可选实现方案

### 6.1 方案 A：直接使用现有 filesystem tier

这是改动最少、最适合首轮验证的方案。

优点：

- 已具备完整三级缓存语义；
- 与 scheduler 生命周期和 block hash 机制对齐；
- 已实现异步查询和异步读写；
- 不需要外部缓存服务；
- 易于调试和验证正确性。

局限：

- 通常一个 KV chunk 对应一个文件；
- 大规模缓存会产生大量小文件；
- `stat`、`open` 和目录 metadata I/O 成本较高；
- 缺少严格的 SSD 容量管理和磁盘 LRU；
- 线程池难以完全发挥高端 NVMe 的 IOPS；
- 小块随机 I/O 容易成为瓶颈。

适用场景：

- 功能验证和性能基线；
- 单机 prefix cache；
- 缓存对象主要为长 prompt；
- SSD 缓存规模较小。

### 6.2 方案 B：生产级 segment-based NVMe tier

保留 `TieringOffloadingManager`，新增例如
`NvmeSecondaryTierManager`：

```text
vllm/v1/kv_offload/tiering/nvme/
├── __init__.py
├── manager.py
├── index.py
├── io.py
└── allocator.py
```

在 `SecondaryTierFactory` 中注册：

```python
SecondaryTierFactory.register_tier(
    "nvme",
    "vllm.v1.kv_offload.tiering.nvme.manager",
    "NvmeSecondaryTierManager",
)
```

配置形式可以为：

```json
{
  "type": "nvme",
  "root_dir": "/mnt/nvme/vllm-kv",
  "capacity_bytes": 1099511627776,
  "segment_size_bytes": 1073741824,
  "n_read_threads": 32,
  "n_write_threads": 16,
  "eviction_policy": "clock"
}
```

建议不再采用一个 block 一个文件，而是使用若干预分配的大 segment
文件。内存索引维护：

```text
block hash
    ├── segment_id
    ├── offset
    ├── length
    ├── checksum
    ├── last_access
    ├── frequency
    └── state: writing / ready / deleting
```

可以选择固定槽位或 append-only 形式：

- 固定槽位：分配和读取简单，适合固定 KV block size。
- append-only：写路径顺序性好，但需要 compaction。

建议同时提供：

- SSD 容量上限；
- LRU、CLOCK 或 segmented LRU；
- manifest/journal；
- 启动索引恢复；
- checksum；
- 后台 compaction；
- 磁盘满时的 fail-open 行为；
- load/store 并发和带宽限制。

I/O 实现可以逐步演进：

1. `preadv/pwritev` 加线程池；
2. io_uring；
3. SPDK 或 GPUDirect Storage。

第一阶段应优先保证生命周期、失败恢复和数据正确性，而不是立即引入复杂的
异步 I/O 框架。

### 6.3 方案 C：mmap 大文件

可以把 SSD 上的大文件映射到虚拟地址空间，让操作系统 page cache 自动管理
DRAM 和 SSD 之间的数据移动。

优点：

- 实现简单；
- 数据可以通过普通内存访问；
- 内核自动进行 page cache 驱逐；
- 启动和恢复相对容易。

缺点：

- major page fault 延迟不可控；
- 关键线程可能同步阻塞在缺页上；
- 无法精确控制热点页面和 I/O 调度；
- dirty page 回写时机不确定；
- mmap 地址不能自动等价于适合 GPU DMA 的 pinned buffer。

该方案适合实验，但不适合对 P99 TTFT 有严格要求的服务。即使使用 mmap，
也建议保留显式 pinned DRAM primary tier。

### 6.4 方案 D：LMCache

vLLM 当前支持 `kv_offloading_backend="lmcache"`。LMCache 更适合：

- 希望缓存管理与 vLLM 进程解耦；
- 需要独立缓存服务；
- 后续需要跨进程或跨节点共享；
- 需要更丰富的存储 backend；
- 可以接受额外组件和部署复杂度。

若只需要单机 NVMe 实验，native tiering 更直接；若目标是生产级共享缓存
服务，可以进一步评估 LMCache。

### 6.5 方案 E：独立 LocalSSDConnector

仓库中存在教学用实现：

[`vllm/distributed/kv_transfer/kv_connector/v1/local_ssd/local_ssd_connector.py`](../vllm/distributed/kv_transfer/kv_connector/v1/local_ssd/local_ssd_connector.py)

它直接实现：

```text
GPU gather → 普通文件
普通文件 → GPU scatter
```

这份实现适合学习 KV Connector 的 scheduler/worker 调用链，但不建议作为
生产实现起点，原因包括：

- 不使用后台 I/O；
- 没有容量和 LRU 管理；
- 只处理第一个 KV cache group；
- SSD 逻辑与 connector metadata 耦合；
- 会重复实现 tiering framework 已解决的生命周期问题。

## 7. 缓存和驱逐策略

### 7.1 DRAM tier

DRAM 的主要职责是：

- GPU DMA staging；
- 吸收 SSD promotion；
- 缓存近期使用的 SSD blocks；
- 减少重复 SSD 读取。

建议：

- 使用 pinned memory；
- 先使用 LRU；
- 工作负载冷热切换明显时测试 ARC；
- 容量至少能容纳若干个并发长 prompt；
- 对 in-flight block 进行引用计数；
- 禁止驱逐正在执行 SSD/GPU I/O 的 block。

如果 DRAM tier 小于 GPU KV Cache 总容量，立即 offload 通常只是在 DRAM
中复制 GPU 仍然持有的数据，未必能增加有效缓存容量。因此应根据 GPU
容量、prompt 长度和并发量合理设置 DRAM 大小。

### 7.2 SSD tier

SSD 应提供独立容量控制，并根据实际负载选择：

- LRU；
- CLOCK；
- segmented LRU；
- TinyLFU admission 加 LRU eviction；
- TTL；
- 按租户或模型配额。

SSD cache key/namespace 不能只包含 token hash。至少需要隔离：

```text
model identity and revision
KV dtype
block/chunk size
TP/PP layout
attention/KV layout
KV cache group
LoRA identity
multimodal identity
token prefix chain hash
```

现有 `FileMapper` 已经处理了其中许多运行配置和 rank 目录隔离，新 backend
应尽量复用该逻辑，避免不同配置之间错误复用 KV Cache。

## 8. Block size 选择

GPU KV block 通常比较小，但 SSD 更适合较大的连续 I/O。因此 offload block
可以由多个 GPU blocks 组成。

假设 GPU block size 为 16 tokens：

| Offload block | 特点 |
|---|---|
| 16 tokens | 命中粒度细，但小 I/O 和 metadata 开销最大 |
| 32 tokens | 保留较细粒度，同时适当降低管理开销 |
| 64 tokens | 推荐起点，粒度和 I/O 效率比较平衡 |
| 128/256 tokens | SSD 吞吐更高，但非对齐和部分命中浪费更大 |

offload block size 必须是 GPU block size 的整数倍。

建议至少测试 32、64、128 tokens，并观察：

- 平均 SSD I/O 大小；
- TTFT；
- SSD promotion 延迟；
- prefix 命中率；
- DRAM 命中率；
- 因不完整命中而重新计算的 token 数；
- SSD IOPS 和吞吐；
- PCIe 和 CPU memory bandwidth。

## 9. 性能与正确性风险

### 9.1 SSD 命中不一定比重新计算更快

KV Cache 数据量可能很大。对于短 prefix，从 SSD 读取并经过
SSD → DRAM → GPU 两次传输，可能比重新执行 prefill 更慢。

需要基于以下因素建立 admission policy：

- prefix token 数；
- KV bytes；
- 历史复用次数；
- 预计重算时间；
- SSD load 带宽；
- GPU 当前负载；
- DRAM 是否已经命中。

可以将存储判定抽象为：

```text
expected_recompute_cost × reuse_probability
    >
expected_store_cost + expected_future_load_cost
```

### 9.2 禁止阻塞 scheduler

`lookup` 不应同步执行 `stat/open/read`。理想情况下：

- 热索引查询只访问内存；
- 冷索引恢复或磁盘 lookup 异步执行；
- scheduler 收到 `RETRY` 后在后续 step 检查结果；
- I/O 完成通过 finished job queue 上报。

### 9.3 读取应优先于后台写入

SSD load 直接影响用户请求 TTFT，而 store 通常不位于当前请求的关键路径。
因此应该：

- 设置独立读写队列；
- 为读请求预留并发；
- 在队列拥塞时丢弃低价值 store；
- 避免后台 compaction 阻塞 promotion。

### 9.4 限制 promotion 并发

大量 SSD hits 同时发生时可能占满：

- NVMe IOPS 和带宽；
- CPU memory bandwidth；
- pinned DRAM slots；
- PCIe 带宽；
- GPU copy engines。

应按 bytes 而不只是 job 数限制并发，并提供：

- promotion queue depth；
- in-flight load bytes；
- in-flight store bytes；
- 每请求 promotion 上限；
- 全局和租户级带宽限制。

### 9.5 避免写放大

建议：

- 默认只 offload prompt；
- 只写完整 chunk；
- 同一 key 已存在时跳过写入；
- 合并连续小 block；
- 对低复用概率数据执行 admission rejection；
- 队列拥塞时优先丢弃 store，而不是延迟 load。

### 9.6 失败时必须 fail open

以下情况应当退化为 cache miss 并重新计算：

- 文件不存在；
- 短读；
- checksum 不一致；
- index 与数据不一致；
- 磁盘满；
- segment 损坏；
- I/O 超时；
- 进程重启后发现未完成写入。

不得将部分加载或校验失败的 KV 数据交给 attention kernel。

## 10. 可观测性

建议至少提供以下指标：

```text
GPU prefix cache hit rate
DRAM tier hit rate
SSD tier hit rate
SSD lookup latency
SSD load/store latency
SSD load/store bytes
DRAM↔GPU transfer bytes
promotion queue depth
cascade queue depth
in-flight pinned bytes
SSD capacity and utilization
SSD eviction count
admission rejection count
checksum/read failure count
recompute fallback count
```

评估效果时不能只看 hit rate，还要比较：

- P50/P95/P99 TTFT；
- 请求吞吐；
- GPU prefill 利用率；
- SSD 带宽和 IOPS；
- CPU 利用率；
- pinned memory 使用量；
- cache hit 后节省的实际计算 token 数。

## 11. 测试设计

在添加测试之前，需要明确：

1. 模块用途：为 Tiering Offload 提供容量受控的本机 NVMe secondary tier。
2. I/O contract：输入 `OffloadKey` 和 primary-tier block slots，异步完成
   store/load 并通过 `JobResult` 上报。
3. 防护的失败：错误命中、短读、损坏数据、in-flight block 被驱逐、任务
   完成状态丢失、重启恢复错误。
4. 最便宜的测试层级：优先 manager/index/I/O unit tests，再增加少量
   scheduler integration tests。

建议测试：

- store 后 lookup 命中；
- store/load round trip 数据一致；
- 不存在 key 返回 miss；
- in-flight store/load 状态正确；
- short read 和 checksum failure 返回失败；
- 失败后 scheduler 可以重新计算；
- 达到容量上限后正确驱逐；
- 不驱逐有引用或 in-flight 的 block；
- 重启后索引恢复；
- 临时写入不会被 lookup 当成 ready；
- shutdown/drain 不丢失任务；
- TP ranks 和不同运行配置不会互相覆盖；
- 同一 key 的并发 store 去重；
- load 优先级高于后台 store；
- 请求结束后不泄漏 lookup/request state。

## 12. 推荐实施路线

### 第一阶段：验证现有能力

1. 使用 `TieringOffloadingSpec + fs`。
2. DRAM 从 32–64 GiB 起步。
3. 使用独占 NVMe 目录。
4. `block_size` 从 64 tokens 起测。
5. 默认只缓存 prompt。
6. 建立 GPU/DRAM/SSD 命中率和端到端 TTFT 基线。

### 第二阶段：增加容量和策略控制

1. 为 filesystem tier 增加可配置容量。
2. 增加 SSD eviction policy。
3. 增加 checksum 和磁盘满处理。
4. 增加 load/store 并发及字节带宽限制。
5. 增加 admission policy。

### 第三阶段：实现生产级 NVMe backend

1. 新增 `NvmeSecondaryTierManager`。
2. 使用 segment 文件替代每 block 一个文件。
3. 建立内存 hash index。
4. 增加 journal/manifest 和重启恢复。
5. 增加后台 compaction。
6. 先使用 `preadv/pwritev + thread pool`。

### 第四阶段：I/O 性能优化

1. 使用 io_uring 批量提交；
2. 合并相邻 block I/O；
3. 优化 direct I/O 对齐；
4. 评估 GDS/SPDK；
5. 根据真实模型和硬件数据决定是否引入压缩或 KV 量化。

## 13. 最终建议

对于当前代码库，最佳扩展边界是：

```text
保留：
  OffloadingConnector
  TieringOffloadingSpec
  TieringOffloadingManager
  CPUPrimaryTierOffloadingManager

新增或增强：
  SecondaryTierManager
      └── NvmeSecondaryTierManager
```

不建议从以下位置重新开始：

- scheduler 的 KV block 分配逻辑；
- attention backend；
- GPU paged KV Cache layout；
- 独立实现一套完整 LocalSSDConnector。

这样可以复用 vLLM 已有的 block hash、request lifecycle、GPU↔DRAM DMA、
promotion/cascade、引用计数和 scheduler retry 机制，将主要工程工作集中在
SSD 容量管理、索引、异步 I/O、恢复和性能优化上。
