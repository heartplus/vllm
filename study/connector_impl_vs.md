# v1 KV Connector 实现对比

本文对比 `vllm/distributed/kv_transfer/kv_connector/v1` 下注册到
`KVConnectorFactory` 的 v1 connector。核心观察维度是：调度侧如何判断外部
KV 命中、worker 侧如何搬运 KV、是否异步、是否支持 HMA、多 KV group/混合
attention，以及适合的使用场景。

## 总览

v1 connector 的接口被拆成 scheduler 和 worker 两侧：

- scheduler 侧负责命中判断、分配后更新状态、构造 metadata、请求结束时是否
  延迟释放 block，以及消费 worker 回传结果。
- worker 侧负责注册 KV cache tensor、根据 metadata 发起 load/save、等待或轮询
  异步传输完成，并上报 finished sending/recving。

不同 connector 的差异主要体现在三点：

- 外部介质：文件系统、CPU 内存、分布式 KV pool、RDMA/P2P、外部缓存系统。
- 命中模型：prefix hash 命中、router 提供的 P/D transfer params、外部系统
  lookup、或者 benchmark 人造命中。
- 搬运粒度：整 request/block、所有层合并 page、layerwise hook、跨层 packed
  block、NIXL/Mooncake 的内存描述符。

## 快速对比表

| Connector | 外部介质/后端 | 主要用途 | load 命中来源 | worker 搬运方式 | 异步模型 | HMA/多组支持 | 关键限制 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ExampleConnector` | 共享目录 `safetensors` | 调试/示例 KV 存取 | token/mm hash 文件夹是否存在 | 逐层读写 safetensors，scatter/gather 到 paged KV | 基本同步 | 非 HMA | 会覆盖已有 prefix cache；不是生产实现 |
| `ExampleHiddenStatesConnector` | 共享目录 `safetensors` | `extract_hidden_states` speculative 输出 | 不 load，仅 store hidden states | TP rank 0 从 hidden-state cache 抽取，异步 D2H + 线程写盘 | 异步 store | `SupportsHMA` | 仅 hidden states；要求特定 speculative 配置 |
| `DecodeBenchConnector` | 无外部后端 | decode 性能 benchmark | 人造“外部命中” | 直接填充 KV cache 为常量/随机值 | 同步 fill | `SupportsHMA` | 不是真实缓存；不 save |
| `SimpleCPUOffloadConnector` | CPU 内存 | 简单 CPU prefix offload | vLLM prefix caching + CPU manager | custom kernel 在 GPU/CPU 间拷贝，BlockPool LRU | load/save 都由 `get_finished` 驱动 | `SupportsHMA` | 需要 prefix caching；容量按 rank 配置 |
| `OffloadingConnector` | `vllm.v1.kv_offload` manager | 内建通用 KV offload | OffloadingManager lookup | Canonical KV block tensor，load/store job | worker metadata 汇报 job 完成 | `SupportsHMA` | 要求 HND；复杂策略在 kv_offload 层 |
| `LocalSSDConnector` | 本地 SSD 普通文件 | 教学/实验版 SSD cache | 每个 rank 文件是否存在 | 所有层合成一个 page，文件同步读写 | 同步 | 单 KV group | 无 LRU/后台线程/metadata server；教学实现 |
| `HF3FSKVConnector` | HF3FS + metadata server | 分布式文件系统 KV cache | metadata server batch key lookup | gather/scatter page，经 HF3FS client batch read/write | 后台 save/load 线程 + IO pool | 非 HMA/单组路径 | hash 较简单；依赖 HF3FS/metadata server |
| `LMCacheConnectorV1` | LMCache | 接入 LMCache 缓存生态 | LMCache engine lookup | 由 LMCache adapter 处理 layerwise 或非 layerwise 搬运 | 取决于 LMCache | 非 HMA wrapper | 主要逻辑在外部包；layerwise 需 PIECEWISE cudagraph |
| `LMCacheMPConnector` | LMCache 多进程 server | LMCache 独立服务模式 | ZMQ adapter 异步 lookup | batched retrieve/store 到 LMCache MP server | lookup/retrieve/store 异步 | 单组，要求禁用 hybrid manager | 不支持多 KV group；MLA rank 映射特殊 |
| `FlexKVConnectorV1` | FlexKV | 接入 FlexKV 多级缓存 | FlexKV adapter lookup | 由 FlexKV adapter 直接调度任务 | adapter 内部异步 | 非 HMA wrapper | vLLM 侧是薄封装，需安装 FlexKV |
| `NixlPullConnector` | NIXL READ P2P | P/D 分离，D 从 P 拉 KV | router 的 `kv_transfer_params` | NIXL 内存注册、handshake、READ descriptors | 异步 xfer + notif | `SupportsHMA` | 配置/布局兼容要求高；默认 HND |
| `NixlPushConnector` | NIXL WRITE P2P | P/D 分离，P 向 D 推 KV | D 注册目标 block，P 完成后匹配 | writer 线程处理 registration、WRITE、notif | 后台 writer 线程 | `SupportsHMA` | 不支持 bidirectional；PP/HMA 组合有限制 |
| `MooncakeConnector` | Mooncake TransferEngine P2P | P/D 直接传输 | router 的 `kv_transfer_params` | bootstrap + ZMQ side channel + Mooncake write | 后台 asyncio/线程池 | `SupportsHMA` | 依赖 Mooncake；P/D 协调复杂 |
| `MooncakeStoreConnector` | MooncakeDistributedStore | 共享 KV pool + prefix cache | store lookup key | store put/get，独立 sending/recving 线程 | `get_finished` 驱动线程队列 | `SupportsHMA` | 不支持 CrossAttention；部分 hybrid + PCP/DCP 限制 |
| `MoRIIOConnector` | MoRI IOEngine RDMA/XGMI | P/D 传输，READ 或 WRITE | router 的 transfer params | READ 拉取或 WRITE layerwise 写入 | READ/P2P async；WRITE 有后台 writer | 非 HMA 标记 | 强依赖 transfer_id/ZMQ 通知；异构 TP 限制多 |
| `MultiConnector` | 多个子 connector | 同时读写多后端 | 按配置顺序选第一个命中 | 转发到所有子 connector | 聚合子 connector 状态 | 仅当所有子 connector 支持 | 多个 async save 需额外计数；layout 必须一致 |

## 实现分组

### 1. 示例、教学和 benchmark

`ExampleConnector` 是最直接的接口示例：scheduler 根据 prompt token 和多模态
identifier 生成目录 hash，worker 对每层 attention KV cache 读写
`safetensors`。它把 load/store 写进同一个 metadata 列表，适合理解
`get_num_new_matched_tokens -> update_state_after_alloc -> build_connector_meta ->
start_load_kv/save_kv_layer` 的调用关系。缺点也很明显：同步文件 IO、没有驱逐、
没有并发控制，并且注释中说明会覆盖已有 GPU prefix cache。

`LocalSSDConnector` 是更完整的教学版本地 SSD connector。它把一个 block 的所有
layer 聚合成一个 page 文件，文件 key 是链式 prefix hash，并按 rank 分目录保存。
相比 `ExampleConnector`，它更接近真实 block-cache：按完整 block 命中、只保存新增
block、原子写临时文件再 rename，并记录 load error block。但它故意不做后台线程、
LRU、metadata server，也只看第 0 个 KV group。

`DecodeBenchConnector` 不访问外部缓存。它在 scheduler 侧把“除最后一个 token 外的
上下文”声明为外部命中，worker 侧直接把对应 KV blocks 填成常量或随机值。用途是让
decode 实例在没有真实 prefill 传输时也能模拟长 ISL 的 KV 占用和性能压力。

`ExampleHiddenStatesConnector` 是 store-only。它服务于
`extract_hidden_states` speculative decoding：请求结束时延迟释放 hidden-state
cache block，worker 在 `get_finished` 中异步 D2H 拷贝并写 `safetensors`。它通过
`.lock` 文件让客户端避免读到半写文件，只让 TP rank 0 写盘，并要求 hidden-state
cache group 可被唯一定位。

### 2. 单机/存储型 prefix offload

`SimpleCPUOffloadConnector` 面向 CPU 内存 offload。它要求启用 prefix caching，按
server 总容量或 per-rank 容量初始化 CPU manager。它不在 layer hook 里搬运，而是把
load/save 都交给 `get_finished` 和内部 worker handler 异步推进，scheduler 侧还可
绑定 GPU block pool 以做 LRU/引用计数管理。适合单机或每 rank CPU 内存扩展，不适合
跨节点共享。

`OffloadingConnector` 是更通用的内建 offload 框架入口。它通过
`OffloadingSpecFactory` 生成 `OffloadingSpec`，scheduler 侧用
`OffloadingManager` 做 lookup、store/load job 编排和 KV events，worker 侧把各层
KV cache 规整成 `CanonicalKVCaches` 后提交 load/store。它偏好 cross-layer blocks，
要求 HND layout，并通过 `OffloadingWorkerMetadata.completed_jobs` 聚合多个 worker
的完成状态。相对 `SimpleCPUOffloadConnector`，它更系统化，也处理 FullAttention、
SlidingWindow、Mamba、EAGLE trailing block 等细节。

`HF3FSKVConnector` 可以看作 LocalSSD 的生产化方向：scheduler 不扫描本地文件，而是
问 HF3FS metadata server key 是否存在；worker 使用 HF3FS client 预分配 page、确认
写入、批量读写 offset。它有 `AsyncOperationManager`，包含 save/load CUDA stream、
device buffer allocator、后台 worker thread 和 IO thread pool。它适合分布式文件
系统后端，但相比 LocalSSD 也多了 metadata server、HF3FS FUSE/client、page 分配和
失败恢复复杂度。

### 3. 外部 KV 缓存系统适配

`LMCacheConnectorV1` 是 LMCache 的 v1 adapter wrapper。vLLM 侧主要负责选择 native
adapter 或外部最新 adapter、转发 lifecycle 方法、把 LMCache KV events 转成 vLLM
`BlockStored` events。若启用 `use_layerwise`，它要求 PIECEWISE CUDA graph，因为
`wait_for_layer_load` 和 `save_kv_layer` 里可能有实际 Python 同步逻辑。

`LMCacheMPConnector` 是 LMCache 多进程模式。scheduler 侧为每个 request 建
`LMCacheMPRequestTracker`，先异步 lookup，再按 LMCache chunk 粒度生成 RETRIEVE 或
STORE metadata；worker 侧通过 ZMQ adapter 批量提交 retrieve/store 请求，并用 CUDA
event 标记 stream 依赖。它显式禁止多 KV group，提示要
`--disable-hybrid-kv-cache-manager`；MLA 下还会把 TP rank/world size 映射成 KV rank。

`FlexKVConnectorV1` 与 LMCache 类似，vLLM 内部只是薄封装。它把所有 scheduler/worker
方法转发给 `flexkv.integration.vllm.vllm_v1_adapter.FlexKVConnectorV1Impl`。注释说明
当前 FlexKV 主要在 scheduler 侧管理 transfer task，直接在 FlexKV server 和 vLLM GPU
内存间搬运，worker layer hooks 目前多是兼容接口。

### 4. P/D 直接传输型 connector

`NixlPullConnector` 和 `NixlPushConnector` 共享 `NixlBaseConnector`，区别是传输方向：

- Pull/READ：decode 侧拿到 remote block 信息后，worker 与 prefill worker 做
  handshake，注册远端 agent，再用 NIXL READ 把 KV 拉到本地 block。
- Push/WRITE：decode 侧先把目标 block registration 通过 NIXL notif 发给 prefill；
  prefill 请求完成后 worker writer 线程匹配 registration 与 finished blocks，然后
  发起 WRITE。

NIXL 的复杂度主要在 worker：内存注册、agent metadata compatibility hash、异构 TP
mapping、HND/NHD 和 kernel block size 映射、MLA/SSM descriptor 生成、host transfer
buffer 兜底、lease/heartbeat、失败 block 上报。它适合高性能 P/D 分离，但要求两端
模型、dtype、KV cache 布局、backend 等兼容。

`MooncakeConnector` 也是 P/D 直接传输，但协调方式不同。prefill worker 通过 bootstrap
server 注册自身地址，decode worker 通过请求中的 `remote_bootstrap_addr` 查询远端
worker，然后用 ZMQ side channel 发送 `MooncakeXferMetadata`。真实传输由 prefill 侧
调用 Mooncake `batch_transfer_sync_write` 写入 decode 的 KV 地址。它支持 TP/PP 对齐、
异构 TP 传输计划、MLA/GDN/Mamba 等 block 处理；成功 transfer metrics 主要记录在
producer worker。

`MoRIIOConnector` 支持 READ 和 WRITE 两种模式。READ 模式类似 decode 从 producer 拉；
WRITE 模式下 decode 侧先通知 producer 目标 block，producer 的 `save_kv_layer` 按层
排 `WriteTask`，后台 `MoRIIOWriter` 等待远端 block ready 和 CUDA event 后执行写入。
它依赖 `transfer_id` 做 request 映射，并有 ZMQ notify、handshake port、defer timeout
等机制。相比 NIXL/Mooncake classic，MoRIIO 的 WRITE 路径更明显使用 layerwise hook。

### 5. 共享 KV pool 型 connector

`MooncakeStoreConnector` 不同于 `MooncakeConnector` 的 P2P 直接传输。它使用
MooncakeDistributedStore 作为共享 KV pool：producer 和 consumer 独立 put/get，scheduler
通过 key lookup 判断 prefix 命中，worker 通过 sending/recving transfer thread 与 store
交互。key 中包含 model、TP/PCP/DCP/PP rank、KV group 和 block hash，可设置
`cache_prefix` 避免多部署冲突。

它比 P2P connector 更像 LMCache/Offloading：支持 prefix hash 去重和 KV events，也可以
reset store。限制包括不支持 CrossAttention、Mamba 必须 align mode、hybrid attention 与
PCP/DCP 同时开启时受限。它可以开启 cross-layer blocks，但 hybrid 模型下仍有约束。

### 6. 组合型 connector

`MultiConnector` 是 wrapper：load 时按配置顺序选择第一个返回命中的子 connector，
save 时转发给所有子 connector。它会聚合 metadata、worker metadata、stats、events 和
finished 状态；如果多个子 connector 对同一 request 异步 save，会用
`extra_async_saves` 计数，避免第一个子 connector 完成后过早释放 blocks。

它的两个重要约束是：

- 所有子 connector 要求的 KV cache layout 必须一致，否则初始化时报错。
- HMA 只有在所有子 connector 都支持时才允许，否则 hybrid KV cache manager 必须关闭。

## 关键设计差异

### 命中判断

文件/存储型 connector 依赖 prefix hash：

- `LocalSSDConnector` 用 model、block size、MLA 标记、上一 block hash、多模态 hash 和
  token block 生成 SHA-256。
- `HF3FSKVConnector` 用 token block 和 previous hash 生成 MD5，并通过 metadata server
  查 key。
- `MooncakeStoreConnector` 复用 vLLM block hashes，并按 store block size 压缩 chunk hash。
- `LMCache`、`FlexKV` 由外部系统决定命中。

P/D 传输型 connector 通常不做通用 prefix lookup，而是依赖 router 或上游传入的
`kv_transfer_params`，例如 `do_remote_prefill`、`do_remote_decode`、`transfer_id`、
remote host/port、remote block ids 等。

### 数据搬运粒度

- `ExampleConnector`：逐层 safetensors 文件。
- `LocalSSDConnector`、`HF3FSKVConnector`：一个 block 的所有 layer 合成一个 page。
- `OffloadingConnector`：把不同 cache layout 规整成 canonical block tensors。
- `NIXL`、`MooncakeConnector`：注册连续内存区域和 descriptor，按 block/group/region 计算
  src/dst 地址。
- `MoRIIO WRITE`：按 layer hook 排写任务，最终按 layer transfer plan 执行。
- `MooncakeStoreConnector`：按 store key 对应的 GPU memory addr/size 做 put/get。

### 异步与释放责任

`request_finished` 返回 `True` 的 connector 会接管 block 释放，直到 worker 后续通过
`get_finished` 报告 finished sending/recving。典型包括：

- `ExampleHiddenStatesConnector`：等 hidden states D2H copy 完成。
- `HF3FSKVConnector`：等后台 save futures 完成。
- `OffloadingConnector`：scheduler 通过 completed job 计数确认。
- `NIXL`、`MooncakeConnector`、`MoRIIOConnector`：等远端读取/写入通知或超时。
- `MooncakeStoreConnector`：producer save 异步完成后释放。

同步或无外部 save 的实现通常返回 `False`，例如 `LocalSSDConnector`、`DecodeBenchConnector`
和部分 consumer-only 路径。

### KV layout 和 HMA

- 明确要求 HND：`OffloadingConnector`，NIXL 非 MLA 默认要求 HND，Mooncake 非 MLA 也设置
  HND。
- 明确要求 NHD：`ExampleHiddenStatesConnector` 的 hidden-state cache。
- 可选 cross-layer：`OffloadingConnector` 默认偏好；`NIXL` 和 `MooncakeStoreConnector`
  由 extra config 控制且受 backend/model 限制；`MultiConnector` 要求所有子 connector
  都偏好才偏好。
- 单组限制明显：`LocalSSDConnector`、`LMCacheMPConnector`。
- HMA 支持需要显式继承 `SupportsHMA`；`MultiConnector` 还会检查所有子 connector。

## 选型建议

- 想学习 v1 connector 生命周期：先看 `LocalSSDConnector`，再对照 `ExampleConnector`。
- 想做 decode benchmark：用 `DecodeBenchConnector`，不要把它当缓存方案。
- 想单机扩展 KV 容量：优先看 `SimpleCPUOffloadConnector` 或 `OffloadingConnector`。
- 想接已有缓存系统：看 `LMCacheConnectorV1`、`LMCacheMPConnector`、`FlexKVConnectorV1`。
- 想做共享 KV pool/prefix cache：看 `MooncakeStoreConnector`。
- 想做 P/D 直接传输：看 `NixlPullConnector`/`NixlPushConnector`、`MooncakeConnector`、
  `MoRIIOConnector`，重点关注 router 传参、handshake、layout 和 TP/PP 兼容性。
- 想多级写入或迁移方案：用 `MultiConnector` 组合，但要先确认 layout、HMA 和 async save
  行为兼容。

## 阅读顺序建议

1. `base.py`：理解 v1 scheduler/worker 生命周期。
2. `local_ssd/local_ssd_connector.py`：以最少外部依赖看完整 prefix cache 链路。
3. `offloading_connector.py` + `offloading/`：看内建 production-style job 编排。
4. `nixl/connector.py` + `nixl/base_scheduler.py` + `nixl/base_worker.py`：看 P/D 直接传输的
   handshake、lease、descriptor 映射。
5. `mooncake/store/connector.py` + `mooncake/store/{scheduler,worker,data}.py`：看共享 store
   模型。
6. `mooncake/mooncake_connector.py` 和 `moriio/moriio_connector.py`：看另两种 P/D 传输协议
   的调度和通知方式。
