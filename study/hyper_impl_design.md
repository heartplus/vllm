# 独立进程式分层 KV Cache Connector 实现设计

## 1. 文档目标

本文给出一个可交付给其他工程师实现的详细方案：新增一个独立的 vLLM
KV Connector，由 Connector 对接独立的 modern C++ cache daemon。daemon
统一管理本机 DRAM 和本机 SSD 两级外部 KV Cache。

本文参考
[`hyper_extern_cache_design.md`](./hyper_extern_cache_design.md)，但不要求机械
复用其中的 Python `TieringOffloadingSpec`。本方案选择新的独立 Connector，
原因是：

- vLLM 只需要知道外部 KV block 是否命中以及何时传输完成；
- DRAM、SSD、索引、驱逐和恢复都是 cache daemon 的内部实现细节；
- C++ daemon 可以独立压测、重启和演进；
- Python Connector 保持较薄，只处理 vLLM lifecycle 和 GPU
  gather/scatter；
- daemon 将来可以支持多个 vLLM 实例，而不必侵入 vLLM scheduler。

结论：**Connector 可以实现分层缓存**。分层不要求由 vLLM 自身感知。
只要 Connector 向 vLLM 提供统一的外部 cache 语义，daemon 内部可以实现：

```text
vLLM GPU KV Cache
        ⇅
共享内存传输槽
        ⇅
C++ cache daemon
   ├── DRAM cache
   └── NVMe SSD cache
```

## 2. 范围与非目标

### 2.1 首版目标

- 单机部署；
- 一个 daemon 服务一个或多个本机 vLLM engine；
- 支持 TP 多 rank；
- 支持完整 prefix block 连续命中；
- DRAM 命中和 SSD 命中对 vLLM 使用同一个 Connector 接口；
- 使用共享内存传输 KV 数据；
- 使用 Unix domain socket 或等价管道传输控制消息；
- SSD 使用预分配 segment 文件和随机 block I/O；
- 支持容量限制、驱逐、校验、异常降级和重启恢复；
- 所有 I/O 异步执行，不阻塞 vLLM scheduler；
- load 失败时 fail open，由 vLLM 重新计算。

### 2.2 首版非目标

- 跨节点共享；
- RDMA；
- GPUDirect Storage；
- daemon 直接访问 GPU memory；
- 多副本强一致；
- SSD 数据永久存储保证；
- 在线压缩或 KV dtype 转换；
- 不同模型配置之间的数据转换；
- 首版支持所有 hybrid KV cache/HMA 模型。

首版建议先支持 uniform full-attention KV layout。HMA、MLA、Mamba 和
sliding-window 应在 layout contract 明确后逐项开放，不能默认宣称支持。

## 3. 总体架构

```text
┌──────────────────── vLLM engine ────────────────────┐
│                                                     │
│  Scheduler process                                  │
│    HyperCacheConnector(SCHEDULER)                    │
│      ├── prefix lookup RPC ───────────────┐          │
│      ├── request state                    │          │
│      └── build worker metadata            │          │
│                                           │          │
│  Worker rank 0..N                         │          │
│    HyperCacheConnector(WORKER)             │          │
│      ├── GPU block gather/scatter         │          │
│      ├── shared-memory slot client        │          │
│      └── async completion tracking        │          │
└──────────────────────────┬────────────────┘          │
                           │ UDS control + shared memory
┌──────────────────────────▼───────────────────────────┐
│                 hyper-cache-daemon                   │
│                                                     │
│  Session/IPC ─ Request Coordinator ─ Metrics         │
│                       │                              │
│             Logical Block Index                      │
│        sign_key -> BlockSetEntry                     │
│                       │                              │
│       ┌───────────────┴────────────────┐             │
│       │                                │             │
│  DRAM cache                       SSD store           │
│  SLRU + pin/ref                  segment allocator    │
│  BlockId -> buffer              BlockId -> DiskLoc   │
│                                       │              │
│                               io_uring/thread pool    │
└──────────────────────────────────────────────────────┘
```

Connector 和 daemon 的职责边界：

| 组件 | 职责 |
|---|---|
| Scheduler Connector | 生成规范化 key、查询连续 prefix、维护请求状态、把 load/save 计划传给 worker |
| Worker Connector | GPU KV gather/scatter、共享内存 slot 生命周期、异步完成上报 |
| daemon IPC 层 | 鉴权、session、协议校验、请求路由、背压 |
| daemon logical index | key 到所有 rank block 的一致性和可见性 |
| daemon DRAM tier | 热 block、传输 staging、引用计数、驱逐 |
| daemon SSD tier | segment 分配、随机 I/O、checksum、容量与回收 |
| daemon journal | 崩溃恢复和元数据一致性 |

## 4. 对原始索引想法的调整

原始设计是：

```text
sign_of(tokens) -> [uint64_t per rank]
uint64_t -> [file_id, offset]
```

这个方向可行，但需要四项重要修正。

### 4.1 `sign_key` 必须是完整命名空间 key

不能只对当前 block 的 token 做弱 hash。否则可能发生：

- hash 碰撞导致错误 KV 命中；
- 相同 token 在不同模型或不同 LoRA 下错误复用；
- 相同 block token 出现在不同 prefix 位置时错误复用；
- 不同 KV layout、dtype 或 TP size 间错误复用；
- multimodal prompt 错误复用。

建议定义：

```cpp
struct CacheNamespace {
  Digest128 model_id;
  Digest128 model_revision;
  Digest128 tokenizer_id;
  Digest128 lora_id;
  Digest128 layout_fingerprint;
  uint32_t block_size_tokens;
  uint16_t tp_size;
  uint16_t pp_size;
  uint16_t kv_group_count;
  uint8_t kv_dtype;
  uint8_t hash_version;
};

struct SignKey {
  Digest128 namespace_digest;
  Digest256 prefix_chain_digest;
  uint32_t block_ordinal;
  uint16_t kv_group;
};
```

`prefix_chain_digest` 应采用链式 hash：

```text
H0 = H(namespace, NONE)
Hn = H(Hn-1, tokens_of_block_n, multimodal_identity)
```

这样相同 token block 出现在不同前缀上下文中不会被错误复用。

生产正确性建议使用 256-bit digest。内存 hashmap 可以保存一个 64-bit
fingerprint 加速筛选，但最终必须比较完整 key。不能把单个 `uint64_t`
hash 当作唯一正确性依据。

### 4.2 `[uint64_t per rank]` 应改成带状态的 BlockSet

一个逻辑 prefix block 对应多个 rank 的物理数据。只有所有必需 rank 均成功
提交后，逻辑 block 才能对 lookup 可见。

```cpp
using PhysicalBlockId = uint64_t;

enum class BlockSetState : uint8_t {
  kAllocating,
  kWriting,
  kCommitted,
  kDeleting,
  kCorrupt,
};

struct RankReplica {
  uint16_t rank;
  PhysicalBlockId block_id;
  uint32_t payload_size;
  uint64_t checksum;
};

struct BlockSetEntry {
  SignKey key;
  std::vector<RankReplica> replicas;
  BlockSetState state;
  uint64_t generation;
  uint64_t create_epoch;
  uint64_t last_access_epoch;
  uint32_t pin_count;
};
```

不能依赖 vector 的位置隐式表示 rank，应该显式保存 rank。这样可检测：

- rank 缺失；
- TP size 不匹配；
- rank 重复提交；
- 部分写成功；
- daemon 重启后的残缺记录。

`lookup(sign_key)` 只有在以下条件全部满足时返回 hit：

- entry 为 `kCommitted`；
- rank 集合完整；
- namespace/layout 匹配；
- 每个 physical block 处于 DRAM ready 或 SSD committed；
- entry 未被删除或判定损坏。

### 4.3 `uint64_t` 是稳定物理 ID，不是磁盘地址

建议保留 `PhysicalBlockId` 间接层：

```text
SignKey
   -> BlockSetEntry
      -> rank -> PhysicalBlockId
         -> PhysicalBlockEntry
            ├── DRAM location, optional
            └── DiskLocation, optional
```

定义：

```cpp
struct DiskLocation {
  uint32_t file_id;
  uint64_t offset;
  uint32_t length;
  uint32_t aligned_length;
  uint64_t generation;
};

enum class PhysicalState : uint8_t {
  kAllocating,
  kInDram,
  kWritingDisk,
  kOnDisk,
  kLoadingDram,
  kEvicting,
  kCorrupt,
};

struct PhysicalBlockEntry {
  PhysicalBlockId id;
  uint16_t rank;
  PhysicalState state;
  std::optional<uint32_t> dram_slot;
  std::optional<DiskLocation> disk;
  uint32_t payload_size;
  uint64_t checksum;
  uint32_t pin_count;
  uint64_t last_access_epoch;
};
```

间接层的收益：

- compaction 时只修改 `PhysicalBlockId -> DiskLocation`；
- logical index 不需要更新；
- 可以在 DRAM 和 SSD 间迁移；
- 可以通过 generation 防止 ABA 和旧完成事件覆盖新状态；
- 日志记录更紧凑。

ID 建议由 daemon 单点分配：

```text
high bits: daemon/store incarnation
low bits: monotonically increasing sequence
```

ID 不能在释放后立即复用。首版直接使用单调递增 ID，接近溢出时拒绝启动，
比处理 ABA 更安全。

### 4.4 logical commit 必须是原子的

一个 sign key 的所有 ranks 采用两阶段状态：

```text
ALLOCATE BlockSet
    ↓
为每个 rank 分配 PhysicalBlockId
    ↓
各 rank 写入共享内存并提交
    ↓
daemon 校验并写入 DRAM/SSD
    ↓
所有 rank 成功
    ↓
原子发布 BlockSetState::kCommitted
```

任何 rank 失败或超时：

- 整个 BlockSet 不可见；
- 已完成的 physical blocks 进入回收；
- Connector 将本次 save 视为失败；
- 不影响当前请求已经产生的输出。

## 5. vLLM Connector 设计

新增：

```text
vllm/distributed/kv_transfer/kv_connector/v1/hyper_cache/
├── __init__.py
├── connector.py
├── protocol.py
├── scheduler.py
├── worker.py
├── shared_memory.py
└── metrics.py
```

类名暂定为 `HyperCacheConnector`，继承
`KVConnectorBase_V1`。通过外部模块路径加载也可以避免首版直接修改内置
registry：

```json
{
  "kv_connector": "HyperCacheConnector",
  "kv_connector_module_path": "hyper_cache_vllm.connector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "control_socket": "/run/hyper-cache/control.sock",
    "namespace": "production-a",
    "shm_slots_per_rank": 16,
    "request_timeout_ms": 5000
  }
}
```

当实现稳定并准备 upstream 时，再在
[`factory.py`](../vllm/distributed/kv_transfer/kv_connector/factory.py) 注册。

### 5.1 Scheduler 侧

需要实现的关键方法：

- `get_num_new_matched_tokens`
- `update_state_after_alloc`
- `build_connector_meta`
- `update_connector_output`
- `request_finished`
- HMA 场景下的 `request_finished_all_groups`

Scheduler Connector 不接触大块 KV 数据，只发送批量 lookup：

```text
LookupPrefixRequest:
  namespace_digest
  ordered SignKey[]
  expected_ranks
  deadline

LookupPrefixResponse:
  matched_prefix_block_count
  status
```

查询必须 side-effect free，因为 vLLM 可能多次调用
`get_num_new_matched_tokens`。touch/提升热度应单独发送，或使用幂等 request
token 去重，不能让重复 lookup 改变正确性。

prefix 命中必须从前往后连续。中间一个 block miss 后不能越过它报告后续
命中。

### 5.2 Worker 侧

需要实现的关键方法：

- `register_kv_caches` 或 `register_cross_layers_kv_cache`
- `set_host_xfer_buffer_ops`
- `start_load_kv`
- `wait_for_layer_load`
- `save_kv_layer`
- `wait_for_save`
- `get_finished`
- `handle_preemptions`

建议优先使用 vLLM 已提供的 host transfer buffer copy operation，而不是
在 Connector 中自行维护不同设备的复制 kernel。Worker 流程为：

```text
load:
  acquire shared-memory slots
  -> daemon load blocks into slots
  -> wait completion
  -> host-to-device scatter
  -> release slots

save:
  acquire shared-memory slots
  -> device-to-host gather
  -> notify daemon
  -> daemon copies/adopts data
  -> release slots after completion
```

如果共享内存不是 CUDA pinned memory，GPU 与共享内存之间的复制性能可能
较差。首版必须做实际验证。更稳妥的实现是：

```text
GPU ↔ Connector-owned pinned buffer ↔ daemon shared memory
```

但这会多一次内存复制。优化版可以让 Python worker 创建 pinned host
buffer，并通过 CUDA IPC/host registration 能力共享；在明确各平台支持和
生命周期之前，不应假设普通 POSIX shm 自动具备 pinned 属性。

首版推荐两个可切换模式：

- `copy_shm`：普通共享内存，额外 pinned staging copy，正确性优先；
- `registered_shm`：worker 对共享内存做 host registration，作为实验优化。

### 5.3 Connector 能否真正隐藏分层

可以。vLLM 的 scheduler 只看到：

```text
MISS / external prefix hit / pending / completed / failed
```

daemon 内部执行：

```text
lookup:
  logical index miss       -> MISS
  DRAM resident            -> copy DRAM to shm
  SSD resident             -> SSD to DRAM/shm
  load already in flight   -> attach waiter

store:
  admit to DRAM
  asynchronously persist to SSD if policy requires
```

但有一个约束：Connector 必须正确映射 vLLM 的异步 lifecycle，不能在
`get_num_new_matched_tokens` 中同步等待 SSD 读取。lookup 只确认可用性；
真正数据传输在 worker load 阶段执行。

## 6. IPC 与共享内存协议

### 6.1 控制面

推荐 Linux 下使用 Unix domain `SOCK_SEQPACKET`：

- 保留消息边界；
- 支持双向通信；
- 支持 `SCM_RIGHTS` 传递 memfd/eventfd；
- 能获得 peer credentials；
- 比自定义双管道更容易管理断线和并发。

如果已有指定的“管道库”，其能力至少要覆盖：

- 消息 framing；
- request ID；
- timeout/cancel；
- backpressure；
- 断线检测；
- FD 或共享内存句柄交换；
- 多线程安全；
- 最大消息大小限制。

控制面禁止承载 KV payload。

### 6.2 数据面

每个 worker rank 建立独立 shared-memory arena：

```cpp
struct ShmArenaHeader {
  uint32_t magic;
  uint16_t protocol_version;
  uint16_t rank;
  uint32_t slot_count;
  uint32_t slot_stride;
  uint64_t session_id;
  uint64_t arena_generation;
};

enum class SlotState : uint32_t {
  kFree,
  kClientWriting,
  kServerReading,
  kServerWriting,
  kClientReading,
  kError,
};

struct SlotHeader {
  std::atomic<uint32_t> state;
  uint32_t payload_size;
  uint64_t operation_id;
  uint64_t generation;
  uint64_t checksum;
};
```

每个 slot 使用固定大小，大小等于该 rank 一个完整 offload block 的最大
payload，并按页或 direct-I/O alignment 对齐。

同步推荐：

- UDS 完成消息作为首版通知机制；
- eventfd 作为批量 completion 优化；
- slot 状态使用 acquire/release memory ordering；
- 不能用轮询加普通非原子字段；
- operation ID 和 generation 必须同时校验。

### 6.3 会话与故障

每个 Connector 实例建立 session：

```text
HELLO
  protocol version
  engine id
  rank
  namespace/layout fingerprint
  expected rank count

WELCOME
  session id
  daemon incarnation
  accepted feature flags
  limits
```

daemon 重启后 incarnation 改变。Connector 发现旧 session 失效时：

- 取消所有 pending operations；
- 将正在 load 的请求报告失败；
- 释放或重建 shm arena；
- 重新握手；
- 不复用旧 operation ID。

客户端进程退出时，daemon 应通过 socket EOF 清理：

- session；
- waiters；
- shm mappings；
- 未提交的 BlockSet；
- pin/reference。

## 7. DRAM Cache 设计

### 7.1 LRU 是否足够

纯 LRU 可作为 MVP，但不建议作为最终默认策略。KV prefix workload 常见：

- 一次性长 prompt 会扫描大量 block，污染 DRAM；
- 热 system prompt 会被大量冷数据挤出；
- 同一 prefix 的前部 block 比尾部更容易复用；
- SSD promotion 本身会把数据带入 DRAM，如果全部无条件 admission，会
  产生 cache pollution。

推荐最终使用：

```text
TinyLFU admission + Segmented LRU eviction
```

如果希望降低首版复杂度，演进顺序是：

1. MVP：sharded LRU；
2. 第二版：SLRU；
3. 第三版：TinyLFU admission + SLRU。

不建议首版直接实现 ARC：ARC 在容量按 bytes、对象 pinning、异步状态和
多 shard 场景下实现及审核成本更高。SLRU 更容易解释、测试和调参。

### 7.2 SLRU 结构

DRAM cache 分为：

- probation：新进入的 block；
- protected：再次命中的 block。

规则：

```text
new block/promotion -> probation MRU
probation hit       -> protected MRU
protected hit       -> protected MRU
protected overflow  -> demote to probation MRU
eviction            -> probation LRU first
```

建议 protected 占 DRAM 容量的 70%–80%，按 bytes 计算，不按 block 数。

### 7.3 TinyLFU admission

使用 Count-Min Sketch 近似记录访问频率，并使用周期衰减。候选 block 进入
DRAM 时，与 probation LRU victim 比较：

```text
frequency(candidate) > frequency(victim) -> admit candidate
otherwise                                -> bypass DRAM
```

load 必须有 staging 空间，但不代表 load 完成后一定长期驻留 DRAM。
被 admission 拒绝的 SSD block 可以：

1. SSD 直接读入临时 shared-memory slot；或者
2. 使用 transient DRAM slot，传输完成后立即回收。

首版允许 promotion 无条件进入 LRU；实现 TinyLFU 后再增加 bypass。

### 7.4 pinning 与并发状态

以下 block 不能驱逐：

- 正在写入；
- 正在从 SSD 加载；
- 正在复制到 shm；
- 正在从 shm 接收；
- 被一个或多个请求使用；
- logical commit 尚未完成。

每个 physical block 维护 `pin_count`。所有异步路径必须使用 RAII pin：

```cpp
class BlockPin {
 public:
  explicit BlockPin(PhysicalBlockEntry&);
  ~BlockPin();
  BlockPin(BlockPin&&);
  BlockPin(const BlockPin&) = delete;
};
```

### 7.5 分片

logical index、physical index 和 DRAM LRU 都建议 sharding，例如 64
shards。key hash 决定 shard。

避免：

- 一个全局 mutex；
- 持有 index lock 时等待 I/O；
- 持有两个 shard lock 时不定义顺序；
- completion callback 获取与提交路径相反顺序的锁。

每个异步操作先在锁内改变状态并捕获 immutable operation context，然后
释放锁再提交 I/O。

## 8. SSD Store 设计

### 8.1 Segment 文件

不建议一个 block 一个文件。使用预分配 segment：

```text
<root>/
  superblock
  manifest.snapshot
  journal/
  data/
    segment-000001.dat
    segment-000002.dat
    ...
```

每个 segment 建议 1–8 GiB，可配置。每个 physical block 占用一个对齐
extent：

```text
offset = allocator.allocate(align_up(payload_size, io_alignment))
```

首版可以使用普通 buffered I/O；direct I/O 必须作为可选项，因为它要求：

- buffer 地址对齐；
- offset 对齐；
- length 对齐；
- 尾部 padding；
- 不同文件系统行为验证。

### 8.2 分配器

首版使用固定 size class 或 bitmap allocator：

- KV layout 固定时，同一 rank payload 通常固定；
- 每个 segment 可以按固定 slot 划分；
- allocation/free 为 O(1)；
- 恢复容易；
- 外部碎片低。

如果不同模型/layout 共用 daemon，应按 namespace/layout 建立不同 storage
class，避免复杂的可变长 allocator。

### 8.3 I/O pipeline

首版：

```text
bounded priority queue
  high: user load
  normal: foreground store commit
  low: eviction writeback / compaction
        ↓
preadv/pwritev worker pool
```

第二阶段再替换为 io_uring。接口应预先抽象：

```cpp
class AsyncBlockIo {
 public:
  virtual OperationHandle Read(ReadRequest, Completion) = 0;
  virtual OperationHandle Write(WriteRequest, Completion) = 0;
  virtual void Cancel(OperationHandle) = 0;
};
```

队列必须按 in-flight bytes 限流，而不仅是请求数量。

### 8.4 数据格式

每个磁盘 extent 包含 header 和 payload：

```cpp
struct DiskBlockHeader {
  uint32_t magic;
  uint16_t format_version;
  uint16_t header_size;
  PhysicalBlockId block_id;
  uint64_t generation;
  uint32_t payload_size;
  uint16_t rank;
  uint16_t flags;
  Digest128 namespace_digest;
  uint64_t payload_checksum;
  uint64_t header_checksum;
};
```

读取时依次检查：

- magic/version；
- block ID；
- generation；
- namespace；
- payload size；
- header checksum；
- payload checksum。

任何不一致都标记 block corrupt，并让 Connector 按 miss/failure 处理。

### 8.5 写入和发布

推荐顺序：

```text
allocate ID and extent
  -> append ALLOC journal
  -> write block data
  -> verify completion
  -> append PHYSICAL_COMMIT journal
  -> all ranks committed
  -> append LOGICAL_COMMIT journal
  -> publish logical index entry
```

是否每次 `fsync` 取决于持久性模式：

- `ephemeral`：允许进程或机器崩溃后丢缓存；重点避免错误命中；
- `recoverable`：批量 group commit journal；
- `durable`：每个事务严格持久化，通常不适合 KV cache 性能目标。

默认建议 `recoverable`，但正确性原则是：可以丢 cache，不能返回错误 cache。

### 8.6 SSD 驱逐

SSD 容量管理与 DRAM 分开。首版 SSD 可使用 CLOCK 或近似 LRU：

- logical BlockSet 是驱逐单位；
- 一次驱逐所有 rank replicas；
- pinned/in-flight BlockSet 不可驱逐；
- 先从 logical index 取消可见；
- 等待 active readers 结束；
- 释放 physical extents；
- 写入 delete journal。

不要只驱逐某个 rank 的 physical block，否则会留下永远无法命中的残缺
BlockSet。

## 9. 核心操作状态机

### 9.1 Lookup

```text
Absent ────────────────────────────────> MISS
Committed + DRAM resident ─────────────> HIT_DRAM
Committed + SSD resident ──────────────> HIT_SSD
Writing/Allocating ────────────────────> MISS or RETRY
Deleting/Corrupt ──────────────────────> MISS
```

建议 prefix metadata lookup 把 `Committed + SSD resident` 也报告为 hit，
随后 worker load 阶段才发起实际 SSD 读取。

### 9.2 Load

```text
LOAD request
  -> validate committed BlockSet
  -> pin all required physical blocks
  -> reserve shm slots
  -> DRAM copy or SSD async read
  -> verify checksum
  -> publish slot completion
  -> worker H2D scatter
  -> RELEASE slots
  -> unpin blocks
```

多个客户端并发 load 同一 SSD block 时应合并底层 SSD read：

```text
PhysicalBlockEntry::kLoadingDram
  -> attach waiter
  -> one SSD read
  -> fan out completion
```

### 9.3 Store

```text
STORE_BEGIN(sign_key, ranks)
  -> existing committed key: deduplicate
  -> create BlockSet kAllocating
  -> allocate physical IDs
  -> workers fill shm slots
  -> STORE_RANK for each rank
  -> daemon validates/copies
  -> all ranks ready
  -> logical commit
  -> completion
```

同一个 sign key 并发 store 时只允许一个 owner。其他 store：

- 附着到相同 in-flight transaction；或
- 返回 `ALREADY_IN_PROGRESS`。

### 9.4 Eviction

```text
select unpinned victim
  -> mark deleting under lock
  -> remove logical visibility
  -> wait/readers already protected by pins
  -> release DRAM/SSD resources
  -> journal delete
  -> erase physical/logical entries
```

## 10. 协议草案

所有消息都包含：

```cpp
struct MessageHeader {
  uint32_t magic;
  uint16_t protocol_version;
  uint16_t message_type;
  uint32_t message_size;
  uint64_t request_id;
  uint64_t session_id;
  uint64_t deadline_ns;
};
```

主要消息：

```text
HELLO / WELCOME
REGISTER_ARENA / REGISTER_ARENA_ACK
LOOKUP_PREFIX / LOOKUP_PREFIX_REPLY
STORE_BEGIN / STORE_BEGIN_REPLY
STORE_RANK_READY / STORE_COMPLETE
LOAD_BEGIN / LOAD_BEGIN_REPLY
LOAD_COMPLETE
RELEASE_SLOTS
CANCEL
TOUCH
HEARTBEAT
ERROR
```

协议要求：

- 明确 little-endian 或使用稳定序列化库；
- 字段有版本；
- 未知字段可跳过或拒绝；
- 所有长度在分配前校验；
- request ID 在 session 内唯一；
- daemon 对相同 request ID 返回幂等结果；
- timeout 不代表底层操作已经取消，必须有明确 cancel/completion；
- 禁止直接发送 C++ struct 原始内存作为长期协议。

序列化可以选择 protobuf、FlatBuffers 或 Cap'n Proto。首版推荐 protobuf
控制消息加固定共享内存 ABI；避免自制可变长二进制协议。

## 11. 配置建议

```yaml
daemon:
  socket_path: /run/hyper-cache/control.sock
  max_sessions: 64
  request_timeout_ms: 5000

dram:
  capacity_bytes: 68719476736
  shards: 64
  policy: slru
  protected_ratio: 0.8
  transient_bytes: 8589934592

ssd:
  root_dir: /mnt/nvme/hyper-cache
  capacity_bytes: 2199023255552
  segment_size_bytes: 4294967296
  io_alignment: 4096
  direct_io: false
  read_threads: 32
  write_threads: 16
  max_inflight_read_bytes: 8589934592
  max_inflight_write_bytes: 4294967296
  persistence: recoverable

cache:
  store_prompt_only: true
  min_tokens_to_store: 64
  checksum: xxh3_64
  incomplete_transaction_timeout_ms: 30000
```

安全或多租户环境中，cache key 应使用密码学 digest；payload checksum
可以使用更快的非密码学校验，因为 namespace key 已负责身份隔离。

## 12. 可观测性

daemon 至少暴露：

```text
lookup_hit_dram_total
lookup_hit_ssd_total
lookup_miss_total
load_latency_seconds{source=dram|ssd}
store_latency_seconds{target=dram|ssd}
dram_bytes_used
dram_pinned_bytes
dram_evictions_total
ssd_bytes_used
ssd_evictions_total
ssd_read_bytes_total
ssd_write_bytes_total
io_queue_bytes{direction=read|write}
shm_slots_in_use
deduplicated_load_total
deduplicated_store_total
checksum_failures_total
partial_rank_transactions_total
recovery_discarded_blocks_total
ipc_errors_total
```

Connector 至少暴露：

```text
external_prefix_hit_tokens
load_wait_seconds
gpu_gather_seconds
gpu_scatter_seconds
shm_wait_seconds
daemon_reconnect_total
load_fallback_total
```

日志必须包含 session ID、request ID、operation ID、sign key 的短
fingerprint 和 rank，但不能打印完整 token。

## 13. 正确性不变量

实现和审核时必须逐条保持：

1. 只有 `kCommitted` logical BlockSet 可以命中。
2. BlockSet 命中时必须包含全部 expected ranks。
3. 完整 key 比较通过后才能命中，64-bit fingerprint 只能加速。
4. namespace/layout 不同的数据永不复用。
5. in-flight 或 pinned block 永不驱逐。
6. physical ID 在 daemon incarnation 内不发生 ABA 复用。
7. completion 必须同时匹配 session、operation ID 和 generation。
8. checksum 失败的数据永不传给 GPU。
9. 任意错误最多导致 cache miss，不得导致错误 KV hit。
10. scheduler-facing lookup 不执行阻塞磁盘 I/O。
11. rank 部分成功不能发布 logical entry。
12. daemon 重启后旧共享内存 completion 不再有效。
13. cancel/timeout 后仍到达的 completion 不得修改新 operation。
14. SSD 驱逐以完整 BlockSet 为单位。
15. Connector 只有在异步 save 完成后才能允许 vLLM 复用/释放相关源
    block，或者必须先复制到独立 staging buffer。

## 14. 详细开发步骤

每一步都应形成独立 PR/commit，能够单独编译、测试和审核。除非步骤明确
声明，否则不得同时引入下一步的性能优化。

### Step 0：需求冻结与兼容矩阵

交付：

- 一页支持矩阵：GPU backend、MHA/MLA、TP/PP、HMA、dtype、block size；
- KV block payload layout 文档；
- namespace fingerprint 字段清单；
- 性能和容量基线目标；
- fail-open 语义。

测试/验证：

- 用现有 vLLM tensor shape 输出验证每种拟支持模型的 bytes per block；
- 明确一个 MVP 模型配置作为 golden case。

审核重点：

- 不存在“默认支持但没有 layout contract”的模型；
- key namespace 覆盖所有影响 KV 内容和布局的配置；
- 不修改任何生产路径。

### Step 1：C++ 工程骨架

交付：

- `hyper-cache-daemon` 可执行程序；
- CMake/build 配置；
- config parser；
- structured logging；
- graceful shutdown；
- 单元测试框架和 sanitizer 配置。

测试：

- config 合法/非法输入；
- SIGTERM 正常退出；
- ASan/UBSan 基础测试。

审核重点：

- 无存储和 IPC 业务逻辑；
- RAII 管理 FD、mmap 和线程；
- 编译器警告视为错误。

### Step 2：稳定 key 和 namespace 库

交付：

- `CacheNamespace`、`SignKey`；
- token prefix chain hash；
- layout fingerprint；
- 稳定序列化；
- Python 和 C++ 两端相同的 key 生成规则。

测试：

- Python/C++ golden vectors；
- token、顺序、模型、dtype、rank layout 任一变化都会改变 key；
- 相同输入跨进程、跨重启得到相同 key；
- 完整 key equality 测试。

审核重点：

- 不使用 `std::hash` 作为持久 key；
- 不依赖 Python randomized hash；
- 明确 hash/version migration。

### Step 3：纯内存 logical/physical index

交付：

- `SignKey -> BlockSetEntry`；
- `PhysicalBlockId -> PhysicalBlockEntry`；
- 单调 ID allocator；
- 多 rank transaction；
- commit/abort/deduplicate；
- 先使用简单 mutex，暂不分片。

测试：

- 全 rank commit 后可见；
- 缺一个 rank 不可见；
- 重复 store 去重；
- abort 回收所有 physical entry；
- generation 和状态转换；
- 并发 lookup/store/delete。

审核重点：

- 状态机和不变量；
- logical commit 原子性；
- 不包含 DRAM/SSD/IPC。

### Step 4：DRAM allocator 与基础 LRU

交付：

- 固定大小 DRAM slots；
- bytes capacity；
- LRU；
- pin/ref count；
- RAII `BlockPin`；
- transient slots。

测试：

- 容量边界；
- LRU 顺序；
- pinned block 不驱逐；
- 所有 block pinned 时返回 backpressure；
- 异常路径不泄漏 slot。

审核重点：

- MVP 只实现 LRU；
- 驱逐不跨越 physical state；
- 锁内不分配大内存、不等待。

### Step 5：本地 in-process API

交付：

- 不经过 IPC 的 `Lookup/Store/Load/Delete` service API；
- DRAM-only end-to-end；
- async future/callback 抽象；
- timeout 和 cancel contract。

测试：

- store/load byte-for-byte；
- 并发相同 key；
- load 合并；
- cancel/late completion；
- daemon shutdown 时完成所有 promise。

审核重点：

- API 与具体 IPC 解耦；
- 后续 SSD 和 IPC 都复用同一 service。

### Step 6：控制面协议

交付：

- protobuf schema；
- UDS `SOCK_SEQPACKET` server/client；
- HELLO/WELCOME；
- session、request ID、deadline；
- peer credential 检查；
- framing 和消息大小限制。

测试：

- 协议版本不匹配；
- malformed/oversized message；
- client disconnect；
- request ID 重放；
- daemon restart/incarnation；
- fuzz parser。

审核重点：

- 此步骤仍不传输 KV payload；
- 不信任客户端长度和 rank；
- 协议向后演进规则明确。

### Step 7：共享内存 arena

交付：

- memfd/shm arena；
- FD passing；
- slot allocator 和状态机；
- UDS completion；
- session cleanup。

测试：

- client→server 和 server→client round trip；
- slot generation/ABA；
- client crash；
- daemon crash；
- arena 越界和错误 payload size；
- ThreadSanitizer 状态机测试。

审核重点：

- acquire/release 内存序；
- FD/mmap 生命周期；
- 普通 shm 不被错误宣称为 pinned。

### Step 8：DRAM-only C++ daemon

交付：

- IPC、index、DRAM cache 集成；
- 批量 prefix lookup；
- multi-rank store/load；
- metrics 和管理接口。

测试：

- 独立 C++ client e2e；
- 多 client、多 rank；
- 部分 rank timeout；
- 容量压力和驱逐；
- daemon 重启后客户端重连。

审核重点：

- 暂无 SSD；
- 完整故障语义；
- scheduler lookup 路径无阻塞操作。

### Step 9：Python 协议客户端

交付：

- Python UDS client；
- shared-memory wrapper；
- asyncio/thread-safe completion bridge；
- Python/C++ protocol compatibility tests。

测试：

- Python→daemon DRAM round trip；
- reconnect；
- timeout/cancel；
- FD 泄漏检查；
- fork/spawn 行为。

审核重点：

- 尚未接入 vLLM Connector；
- Python client API 足够小；
- GIL 下不做大 payload copy。

### Step 10：最小 vLLM Scheduler Connector

交付：

- 外部模块形式的 `HyperCacheConnector`；
- scheduler role；
- prefix key 生成；
- `get_num_new_matched_tokens`；
- request state；
- worker metadata 类型。

测试：

- 使用 fake daemon/client 的 unit tests；
- 连续 prefix 命中；
- 中间 miss 截断；
- 重复 lookup side-effect free；
- request finish 清理。

审核重点：

- 不改 vLLM scheduler；
- 不实现 GPU copy；
- 明确暂时禁用 HMA 或只支持单一 KV group。

### Step 11：Worker Connector 与 DRAM-only e2e

交付：

- worker role；
- KV cache registration；
- GPU gather/scatter；
- shm slot 使用；
- load/save lifecycle；
- `get_finished` 和 error reporting。

测试：

- 单 rank 小模型 e2e；
- 保存后第二次请求命中且输出一致；
- daemon load 失败后重新计算；
- preemption；
- async save 时 GPU block 生命周期；
- CUDA stream synchronization。

审核重点：

- 任何数据失败都 fail open；
- 不读取已释放 GPU block；
- 不把未完成 slot scatter 到 GPU；
- 对照无 cache 输出逐 token 一致。

### Step 12：TP 多 rank 原子提交

交付：

- expected ranks；
- rank worker session mapping；
- BlockSet transaction coordinator；
- 部分提交超时和回收。

测试：

- TP=2/4；
- 一个 rank 慢、断开或写失败；
- rank 重复；
- rank layout 不匹配；
- 只有全 rank commit 才命中。

审核重点：

- logical atomic visibility；
- scheduler/worker metadata rank 对齐；
- 不以 vector 下标隐式信任 rank。

### Step 13：SSD segment 与 allocator

交付：

- superblock；
- segment 创建/预分配；
- fixed-slot/bitmap allocator；
- `PhysicalBlockId -> DiskLocation`；
- buffered `preadv/pwritev`。

测试：

- allocation/free/reuse；
- segment 边界；
- short read/write；
- ENOSPC；
- 文件截断；
- payload/header checksum。

审核重点：

- 暂不接入 daemon service；
- offset/length 溢出检查；
- 磁盘格式有 version。

### Step 14：journal 与恢复

交付：

- WAL records；
- group commit；
- snapshot；
- replay；
- incomplete transaction 清理。

测试：

- 在每个写入阶段模拟 crash；
- torn/corrupt journal tail；
- snapshot+journal replay；
- 不发布缺 rank 或缺 physical block 的 entry；
- recovery idempotent。

审核重点：

- 可以丢数据但不能错误命中；
- 恢复后 generation/ID 不倒退；
- 明确 fsync 边界。

### Step 15：SSD 接入 daemon

交付：

- DRAM miss 时 SSD load；
- DRAM eviction 时 SSD persistence；
- load priority queue；
- in-flight bytes 限流；
- checksum failure quarantine。

测试：

- SSD hit e2e；
- 并发相同 block 只读一次 SSD；
- queue saturation；
- load 优先于 store；
- corrupt block 回退重算；
- SSD 满时服务继续工作。

审核重点：

- SSD I/O 不持有 index lock；
- completion generation 校验；
- pin 生命周期完整。

### Step 16：SSD 容量和完整 BlockSet 驱逐

交付：

- capacity accounting；
- CLOCK/近似 LRU；
- logical BlockSet eviction；
- delete journal；
- space reclamation。

测试：

- 容量稳定；
- 所有 rank 一起驱逐；
- pinned/in-flight 不驱逐；
- 重启后已删除 entry 不复活；
- 长时间 churn。

审核重点：

- 不留下可见但缺 rank 的 entry；
- 空间统计与 allocator 一致。

### Step 17：SLRU

交付：

- probation/protected；
- bytes-based protected ratio；
- promotion/demotion；
- 指标。

测试：

- 单次 scan 不立即淘汰所有热点；
- protected overflow；
- pinned victim 跳过；
- 与 LRU trace 对比。

审核重点：

- 独立策略接口；
- 不同时加入 TinyLFU，便于审核收益。

### Step 18：TinyLFU admission

交付：

- Count-Min Sketch；
- doorkeeper，可选；
- 周期衰减；
- candidate-victim admission；
- transient/bypass path。

测试：

- 频率估计和衰减；
- scan-resistant trace；
- bypass 后 slot 回收；
- hash collision 只影响策略、不影响正确性。

审核重点：

- TinyLFU 只决定 admission；
- 完整 SignKey equality 仍决定 cache 正确性。

### Step 19：性能优化

按独立 PR 逐项执行，不要合并：

1. batching；
2. io_uring backend；
3. direct I/O；
4. registered shared memory；
5. layer-wise pipeline；
6. segment compaction。

每项必须提供：

- 优化前后 benchmark；
- P50/P95/P99；
- CPU、内存和 I/O 数据；
- fallback 开关；
- correctness regression tests。

### Step 20：生产化与 vLLM upstream 准备

交付：

- 部署和容量规划文档；
- systemd/container 配置；
- rolling restart 行为；
- dashboard/alerts；
- compatibility matrix；
- model evaluation；
- vLLM 内置 Connector 注册（若决定 upstream）。

审核重点：

- 按项目要求先检查重复 issue/PR；
- 人工审查全部 AI 辅助代码；
- PR 描述披露 AI assistance；
- 列出全部测试和模型评估结果。

## 15. 每一步统一审核模板

每个开发步骤提交时必须回答：

```text
1. 本步骤只解决什么问题？
2. 明确不解决什么？
3. 新增或改变了哪些状态和不变量？
4. 失败时系统如何退化？
5. 有哪些资源需要释放？
6. 并发与锁顺序是什么？
7. 单元测试覆盖哪些状态转换？
8. 是否有跨进程/重启 golden test？
9. 是否改变磁盘或 IPC 格式？
10. 如何回滚或禁用？
```

每个步骤应满足：

- diff 较小且职责单一；
- 不依赖未合入的后续步骤才能测试；
- 有明确 acceptance criteria；
- sanitizer 或相应 Python tests 通过；
- 不用 benchmark 代替 correctness test；
- 不把暂未支持的模型默认为支持。

## 16. 推荐的 MVP 截止点

第一个可用 MVP 建议做到 Step 12：

```text
独立 Connector
  + C++ daemon
  + DRAM-only cache
  + shared memory
  + 单机 TP 多 rank
  + fail-open
```

先验证：

- Connector lifecycle 正确；
- GPU gather/scatter 正确；
- shared-memory protocol 稳定；
- multi-rank commit 正确；
- 命中输出与无 cache 完全一致。

然后 Step 13–16 增加 SSD。这样 SSD 问题不会和 Connector、GPU copy、
IPC、TP 一次性交织，审核和定位风险最低。

## 17. 最终推荐

采用新的独立 `HyperCacheConnector` 是可行且推荐的，但 Connector 自己
不应直接实现复杂的 DRAM/SSD 策略。正确的边界是：

```text
HyperCacheConnector:
  vLLM lifecycle
  prefix lookup orchestration
  GPU gather/scatter
  shm/IPC client

hyper-cache-daemon:
  key/index
  multi-rank atomicity
  DRAM SLRU/TinyLFU
  SSD segment store
  async I/O
  recovery
  capacity/eviction
```

内存策略方面，MVP 使用 LRU 足够；正式版本推荐
`TinyLFU admission + SLRU eviction`。SSD 索引保留
`SignKey -> rank replicas -> PhysicalBlockId -> DiskLocation` 两级间接结构，
但必须加入完整 namespace key、显式 rank、状态、generation、原子
multi-rank commit 和 checksum。

这个方案既保留了原始设计中简单高效的 `uint64_t` 物理 ID，又避免 hash
碰撞、rank 部分可见、磁盘 compaction 更新全索引和异步 ABA 等关键正确性
问题。
