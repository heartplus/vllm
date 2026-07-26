# `vllm/v1/worker/gpu/model_runner.py` 实现逻辑分析

本文分析 `GPUModelRunner` 的实现。这个文件是 vLLM v1 GPU worker 侧的核心执行器：
它不负责调度决策，而是把 scheduler 下发的 `SchedulerOutput` 转成模型 forward 需要的
GPU 输入，执行模型，随后完成采样/池化、状态更新、KV connector 后处理和异步输出拷贝。

文件开头的注释给出一个重要约束：这里是所有模型共用路径，应该只放通用逻辑。模型特有
行为应下沉到 `model_state`、模型类、attention backend、speculator、pooling runner 或
KV connector 等模块里。

## 总体职责

`GPUModelRunner` 可以理解为 worker 内的“单步执行编排器”：

1. 初始化公共运行状态：模型配置、并行信息、请求状态表、输入 buffer、LoRA/spec/MM/pooling
   组件、KV connector。
2. 加载模型并初始化依赖模型结构的组件：`model_state`、sampler、pooling runner、speculator、
   PP intermediate tensor。
3. 初始化 KV cache 和 attention backend：block tables、KV tensors、CUDA graph manager、
   KV connector。
4. 每个 engine step 执行：
   - 同步 scheduler output 到本地 request/block 状态。
   - 构建 `InputBatch`、block table、slot mapping、attention metadata。
   - 准备 multimodal embedding、LoRA、model-specific inputs。
   - 按 CUDA graph 模式运行模型。
   - 保存中间结果到 `execute_model_state`。
   - 由 `sample_tokens()` 或 `pool()` 完成输出后处理。

核心入口按调用阶段分成三类：

- 初始化阶段：`__init__`、`load_model()`、`initialize_kv_cache()`、`capture_model()`。
- forward 阶段：`execute_model()`。
- forward 后阶段：生成模型走 `sample_tokens()`，pooling 模型走 `pool()`。

## 主要数据结构

### `RequestState`

`self.req_states` 是 request 级运行状态的集中存储，初始化于 `__init__`。它维护：

- `req_id_to_index/index_to_req_id`：把字符串 request id 映射成紧凑 slot index。
- `all_token_ids`：每个 request 的 token 序列，使用 staged/UVA tensor 降低 GPU 显存压力。
- `prompt_len/prefill_len/total_len`：区分用户 prompt、实际 prefill 输入和总长度。
- `num_computed_tokens` 与 CPU mirror `num_computed_tokens_np`。
- `last_sampled_tokens`、`draft_tokens`、`next_prefill_tokens`。

`GPUModelRunner` 不直接把 `req_id` 传进大多数 GPU kernel，而是先映射到 request slot，
再通过 `idx_mapping` 操作这些状态 tensor。

### `InputBuffers` 和 `InputBatch`

`InputBuffers` 是预分配 GPU buffer，包括 `input_ids`、`positions`、`query_start_loc`、
`seq_lens`、DCP local seq lens、padding mask 等。

`prepare_inputs()` 每步返回 `InputBatch`，它是一次 forward 的完整批描述，包含：

- request 顺序、request slot 映射。
- 每个 request 的 scheduled token 数。
- query start loc、seq lens、DCP local seq lens。
- input ids、positions、padding mask。
- logits indices 与 spec decode 的 expanded mapping。
- prefill/decode 判定相关 CPU 数组。

`InputBatch` 是后续 attention metadata、model-specific inputs、采样、prompt logprobs、
speculator 的共同输入。

### `BlockTables`

`self.block_tables` 在 `initialize_kv_cache()` 中创建。它维护 request slot 到 KV cache block
ids 的映射，并负责在每步生成：

- attention backend 需要的 block tables。
- 每个 token 对应的 KV slot mapping。

新增 request 或 cached request 获得新 block 时，`add_requests()` / `update_requests()` 会先
把 block ids staged 到 `BlockTables`，随后 `execute_model()` 调用
`self.block_tables.apply_staged_writes()` 提交。

### `model_state`

`model_state` 由 `init_model_state()` 创建。它隐藏模型类别差异：

- 模型自定义 state 优先。
- encoder-decoder/cross-attention 走 `EncoderDecoderModelState`。
- hybrid/Mamba 走 `MambaHybridModelState`。
- 普通模型走 `DefaultModelState`。

`GPUModelRunner` 通过统一接口调用它：

- `add_request/remove_request` 管 request 级模型状态。
- `prepare_attn` 构造 attention metadata。
- `prepare_inputs` 补充模型 forward kwargs。
- `preprocess_state/postprocess_state` 处理 Mamba/hybrid 等特殊状态。
- multimodal embedding 也通过它间接完成。

这个设计让 `model_runner.py` 保持通用。

## 初始化逻辑

### `__init__`

`__init__` 只做轻量初始化，不加载模型权重，也不分配 KV cache。主要工作如下：

- 缓存 `vllm_config` 中的 model/cache/compilation/parallel/scheduler/speculative 等配置。
- 确定 dtype 和 KV cache dtype。如果 `cache_dtype != "auto"`，把字符串 dtype 转为 torch dtype。
- 初始化最大 request/token 数、vocab size、max model len。
- 创建 `output_copy_stream`，后续异步 D2H 输出拷贝会用它。
- 记录 PP/DP/DCP 信息：
  - PP：是否 first/last rank，是否需要 `PPHandler`。
  - DP：dp size/rank。
  - DCP：decode context parallel rank、interleave 参数。
- 初始化 multimodal 支持和 `EncoderCache`。只有 first PP rank 负责 MM encoder/cache。
- 初始化 speculative decoding：
  - last PP rank 创建 `speculator`。
  - eagle3/dflash/dspark 需要 target model 输出辅助 hidden states，且不支持 PP。
- 初始化 `RequestState` 和 `InputBuffers`。
- 如果启用 PP，创建 `PPHandler`。
- 初始化 LoRA state、capture cases。
- KV connector 初始为 no-op，等 KV cache 初始化后才真正构造。
- 初始化 EPLB controller。

这里大量对象只是“壳”或状态容器，真正依赖模型结构的对象在 `load_model()` 后才创建。

### `load_model()`

`load_model()` 负责加载权重和初始化依赖模型实例的组件：

1. 可选把 load format 改成 dummy，用于 profiling/capture。
2. 调 `get_model_loader(...).load_model(...)` 加载模型。
3. 如果有 LoRA，调用 mixin 包装模型。
4. 如果 spec decode 需要辅助 hidden states，调用 `set_eagle3_aux_hidden_state_layers()`。
5. 如果 speculator 是 draft model speculator，让它加载 draft model，并可注册到 EPLB。
6. 记录模型加载显存和耗时。
7. 非 dummy 权重时，为模型和 speculator 准备通信 buffer。
8. 调 `init_model_state()` 创建 `self.model_state`。
9. 根据 `model_state.num_new_sampled_tokens_per_step` 计算 `decode_query_len`。
   - 普通 decode 通常是 1。
   - spec decode 时是 speculative tokens + bonus sampled tokens。
10. last PP rank 且非 pooling 模型时创建 sampler、rejection sampler、prompt logprobs worker、
    structured outputs worker。
11. pooling 模型 last PP rank 创建 `PoolingRunner`。
12. 注册模型到 EPLB 并启动异步 loop。
13. 非 first PP rank 创建持久 `intermediate_tensors`，用于 PP 接收上游 rank 的中间激活。

关键点：sampler 只在 last PP rank 存在，因为只有最后一段 pipeline 产生 logits/hidden states。

### `initialize_kv_cache()`

`initialize_kv_cache()` 在已知 `KVCacheConfig` 后执行，负责 attention/KV 运行环境：

1. deepcopy `kv_cache_config`，保存到 runner。
2. 根据模型长度、encoder-decoder 情况、DCP 和每个 KV group 的 block size，计算
   `max_num_blocks_per_group`。
3. Mamba group 如果启用 prefix caching 会按最大长度分配，否则至少 1 个 block，并额外加
   speculative blocks。
4. 调 `init_attn_backend()` 初始化 attention groups、CUDA graph 支持信息、kernel block sizes。
5. 创建 `BlockTables`。
6. 初始化 Mamba SSU backend。
7. 根据 attention backend 支持能力、配置、decode query len、TP size、KV cache config 等解析
   cudagraph mode 和 capture sizes。
8. 创建 `ModelCudaGraphManager`，并把同样模式传给 speculator。
9. 检查 attention context parallel 兼容性。
10. 如果是 draft model speculator，把 attention/KV cache 信息设置进去。
11. 调 `init_kv_cache()` 真正分配 KV cache tensors。
12. 基于 KV cache dict 创建 `self.kv_connector`。没有 KV transfer group 时为 no-op。

这里是 KV connector 与 model runner 接上的位置：connector 需要 worker 侧 KV tensor 指针，
所以必须等 KV cache 初始化后才能创建。

### `_init_kv_zero_meta()`

这个方法由 gpu worker 在需要时调用，创建 `KVBlockZeroer`。当 scheduler 分配新 block 并要求
zero 时，`update_requests()` 会用它清零新 KV block，避免 stale NaN 或旧数据污染 attention/SSM。

### profiling 与 CUDA graph capture

`profile_run()` 通过 `_dummy_run(max_num_tokens, skip_attn=True, is_profile=True)` 模拟最大批次，
用于初始显存 profiling。last PP rank 还会额外跑 dummy sampler 或 pooler，以覆盖输出阶段显存。

`capture_model()` 负责 CUDA graph capture：

- 如果 cudagraph manager 不需要 capture，直接跳过。
- 清理 Python GC 和 torch cache，记录 capture 前显存。
- 在 dummy LoRA 上下文里调用 `cudagraph_manager.capture(...)`。
- 如果有 speculator，也 capture speculator。
- 返回 CUDA graph 占用显存。

`_dummy_run()` 是 profile/capture 的公共执行路径。它构造 dummy `SchedulerOutput`，临时禁用
KV connector，调用 `execute_model(..., dummy_run=True)`，最后可在 last PP rank 上跑 speculator
dummy propose。禁用 KV connector 很重要：dummy run 不应该触发真实 KV load/save。

## request 状态更新

`execute_model()` 在真实运行且非 dummy 时，首先同步 scheduler 输出到本地状态：

1. `update_pp_decode_requests()`：非 last PP rank 接收上一步 last rank broadcast 的采样结果，
   用于补齐本地 request state。
2. `finish_requests()`：删除 finished 和 preempted request 的本地状态。
3. `free_states()`：释放 scheduler 指定的 encoder multimodal cache。
4. `add_requests()`：加入本步新调度 request。
5. `update_requests()`：更新已有 request 的 computed token 数和新增 block ids。
6. `block_tables.apply_staged_writes()`：提交 block table 更新。
7. 如果本步没有 token，要走 `kv_connector.no_forward()`，让 connector 仍能处理无 forward 的
   metadata/完成事件。

### `add_requests()`

对 `scheduled_new_reqs` 中每个请求：

- 如果 request 已存在，先 `_remove_request()`，支持 streaming input 更新。
- 向 `req_states` 添加 prompt/prefill tokens、computed tokens、max tokens。
- 如有 encoder cache，登记 request 的 multimodal features。
- 调 `model_state.add_request()`。
- 把新 request 的 block ids 写入 `block_tables`，`overwrite=True`。
- 登记 LoRA request。
- last PP rank 且是生成请求时，向 sampler 和 prompt logprobs worker 添加采样参数。

最后调用 staged write apply，确保 request state、model state、sampler state 可被本步 GPU
逻辑读取。

### `update_requests()`

对 `scheduled_cached_reqs`：

- 更新 `num_computed_tokens_np`。
- 如果有 `new_block_ids`，追加到 block table。
- 用 `min(num_computed_tokens, prefill_len)` 更新 CPU 侧
  `num_computed_prefill_tokens`。
- 如有 `new_block_ids_to_zero`，调用 `kv_block_zeroer.zero_block_ids()`。
- 如有 `kv_cache_block_copies`，调用 `copy_kv_cache_blocks_inplace()` 完成 COW block 拷贝。

顺序很关键：先 zero 新 block，再执行 copy-on-write，最后 forward 才会读取 KV。

## 输入准备

### batch descriptor 与 DP/CUDA graph 同步

`execute_model()` 根据本步 request 数、token 数、最大 query len 计算：

- `num_reqs`
- `num_toks`
- `max_query_len`
- `uniform_tok_count`

然后调用 `dispatch_cg_and_sync_dp()`。它会结合 cudagraph manager、DP rank、LoRA active 数、
profile/eager 强制条件，返回 `BatchExecutionDescriptor` 和跨 DP 的 token 数。这个
descriptor 决定：

- 实际执行 token 数是否需要 padding。
- request 数是否需要 padding。
- 使用 FULL、PIECEWISE 还是 NONE cudagraph mode。

如果所有 DP rank 都无 token，仍通过 `kv_connector.no_forward()` 处理 connector 事件。

### `prepare_inputs()`

`prepare_inputs()` 是文件中最关键的数据整形函数之一，输入是 scheduler output 和 batch desc，
输出 `InputBatch`。

主要步骤：

1. 根据 `num_scheduled_tokens` 对 request 排序。`sort_batch_req_ids()` 让 query len 等于
   `decode_query_len` 的 decode 请求排在前面，再按 token 数排序。这满足部分 attention
   工具对 decode/prefill 分段的假设。
2. 构造 `num_scheduled_tokens` CPU 数组和 request slot `idx_mapping`，并异步拷到 GPU。
3. 处理 spec decode draft tokens：
   - 没有 draft token 时，每 request 只产生一个 logits 位置。
   - 有 draft token 时，计算每 request logits 数、`cu_num_logits`，并扩展
     `idx_mapping/expanded_local_pos`。
4. 构造 `query_start_loc`，并按 FULL CUDA graph 要求填充后续位置。
5. 判断哪些 request 仍处于 prefill。若有 prefill，调用 Triton kernel
   `prepare_prefill_inputs()` 从 `all_token_ids` 写入本步 `input_ids`，同时更新
   `next_prefill_tokens`。
6. 调 `prepare_pos_seq_lens()` 生成 positions 和 seq lens。
7. 如果启用 DCP，生成每个 request 在当前 DCP rank 的 local seq lens。
8. 调 `combine_sampled_and_draft_tokens()`：
   - decode token 直接来自 `last_sampled_tokens`。
   - draft token 来自 `req_states.draft_tokens`。
   - 生成本步用于采样的 `logits_indices`。
9. 生成 `seq_lens_cpu_upper_bound`，供 attention metadata 使用。
10. PP 场景准备 `max_seq_len_np`；R-SWA 场景准备 prompt lens。

结果 `InputBatch` 同时服务 forward、attention、sampler、prompt logprobs、speculator。

### `prepare_attn()` 和 `model_state.prepare_attn()`

`prepare_attn()` 只做两件事：

- 从 `BlockTables` gather 每个 request 的 block table。
- 根据 positions/query_start_loc 计算 slot mappings。

随后 `execute_model()` 调 `build_slot_mappings_by_layer()`，再把 block tables、slot mappings、
attn groups 和 KV config 交给 `model_state.prepare_attn()`。真正的 attention metadata 结构
由 model_state/attention backend 决定。

dummy run 使用 `prepare_dummy_attn()` 构造 dummy block tables 和 slot mappings。

## forward 主路径：`execute_model()`

`execute_model()` 是整个文件的中心。真实执行路径如下：

1. 同步本地 request/block 状态。
2. 处理空 step：若无 scheduled token，走 `kv_connector.no_forward()`。
3. 计算 batch descriptor，并和 DP ranks 同步 CUDA graph 模式。
4. 构造 `InputBatch`、block tables、slot mappings。
5. 调 `model_state.preprocess_state()`。Mamba align 等状态迁移会在 forward 前发生。
6. 如果有 LoRA，按本 batch request 激活 LoRA。
7. 准备 attention metadata。
8. first PP rank 如支持 MM，准备 multimodal embeddings：
   - dummy run 用 `dummy_inputs_embeds()`。
   - 真实 run 先设置 active MM LoRA，再通过 `model_state.get_mm_embeddings()` 获取 embedding。
   - 如果模型不要求 raw input tokens，则 `input_ids=None`。
9. 构造 `model_inputs`：
   - 通用字段：`input_ids`、`positions`、`inputs_embeds`、`intermediate_tensors`。
   - 模型特有字段来自 `model_state.prepare_inputs(input_batch, req_states)`。
10. 非 first PP rank 用接收到的 `intermediate_tensors` 填充持久 buffer，并把 input ids/embeds
    置空。
11. EPLB 准备 forward metadata。
12. 根据 CUDA graph mode 执行模型：
    - FULL：调用 `kv_connector.pre_forward()` 后，直接 replay full graph。
    - PIECEWISE：在 `set_forward_context(...)` 中调用 `kv_connector.pre_forward()`，再
      `cudagraph_manager.run_pw_graph()`。
    - NONE：同样在 forward context 中调用 `kv_connector.pre_forward()`，再直接
      `self.model(**model_inputs)`。
13. last PP rank 解析模型输出为 `hidden_states` 和可选 `aux_hidden_states`。
    非 last PP rank 则得到 `IntermediateTensors`。
14. 把 `InputBatch`、attention metadata、slot mappings、hidden states、finished req ids 等存入
    `self.execute_model_state`。
15. 非 last PP rank 返回 intermediate tensors，last PP rank 返回 `None`，后续由
    `sample_tokens()` 或 `pool()` 消费状态。

### CUDA graph 相关细节

FULL graph 模式不显式传 `model_inputs`，因为输入 tensor 已复制到 capture 时固定的 graph
input buffers。PIECEWISE 和 NONE 都需要正常构造 forward context。

`skip_compiled` 的典型场景是 encoder-decoder 模型本步有 encoder inputs：cross-attention cache
会动态更新，因此强制走 eager/non-compiled。

### forward context

`set_forward_context()` 注入的信息包括：

- attention metadata。
- vLLM config。
- batch token 数。
- cudagraph runtime mode。
- DP across-rank token 数。
- batch descriptor。
- slot mapping by layer。
- 是否跳过 compiled。
- padding mask。

attention layer、KV connector layer hooks、部分模型组件都会从 forward context 读取这些信息。

## KV connector 插入点

`GPUModelRunner` 自己只持有 `self.kv_connector` wrapper。真实 connector 在
`initialize_kv_cache()` 中由 `get_kv_connector(vllm_config, kv_caches_dict)` 创建。

worker wrapper 的行为是：

- `pre_forward(scheduler_output)`：
  - 从 scheduler output 取 `kv_connector_metadata`。
  - 先 `handle_preemptions()`。
  - `bind_connector_metadata()`。
  - 调 `start_load_kv(forward_context)`。
- `post_forward(finished_req_ids)`：
  - `wait_for_save()`。
  - `get_finished()`，得到 finished sending/recving。
  - 读取 invalid block ids、stats、KV cache events、worker metadata。
  - `clear_connector_metadata()`。
- `no_forward(scheduler_output)`：
  - 没有模型 forward 时仍执行 pre/post，但跳过 wait_for_save。

在 `execute_model()` 中，`kv_connector.pre_forward()` 总是在模型执行前调用：

- FULL graph：graph replay 前调用。
- PIECEWISE/NONE：在 forward context 中调用。

在 `sample_tokens()` 和 `pool()` 中，`kv_connector.post_forward()` 在 forward 后被调用。这样
KV connector 的 load/save 生命周期覆盖整步执行。

## 采样路径：`sample_tokens()`

`execute_model()` 不采样，只把 hidden states 存入 `execute_model_state`。
生成模型后续调用 `sample_tokens(grammar_output)`：

1. 取出并清空 `execute_model_state`。
2. 非 last PP rank：
   - 从 `PPHandler.receive()` 接收 last rank broadcast 的采样结果。
   - 乐观更新 `num_computed_tokens`。
   - 如并非全部 decode next，调用 `model_state.postprocess_state()`。
   - 执行 `kv_connector.post_forward()`。
   - 返回只包含 KV connector output 的 `ModelRunnerOutput`。
3. last PP rank：
   - 调 `sample(hidden_states, input_batch, grammar_output)`。
   - 如启用 PP，把 sampled token、num sampled、num rejected broadcast 给非 last ranks。
   - 计算 prompt logprobs。
   - 构造 `ModelRunnerOutput`。
   - 立即创建 `AsyncOutput`，把 sampler output 异步拷到 CPU，和后续 speculator 计算重叠。
   - 调 `postprocess_sampled()` 更新 request 状态和 model state。
   - 如有 speculator，调用 `speculator.propose()` 生成下一步 draft tokens，并写入
     `req_states.draft_tokens`。
   - 如果配置了 speculative steps，把 draft tokens 暂存到 `DraftTokensHandler`，供外部取走。
   - 调 `kv_connector.post_forward()` 并写入 output。
   - 返回 `AsyncOutput`。

### `sample()`

`sample()` 先按 `input_batch.logits_indices` 取需要算 logits 的 hidden states，然后调用
`model.compute_logits()`。如果有 structured output grammar bitmask，先原地应用到 logits。

之后分两种采样：

- 无 draft tokens 或无 rejection sampler：走普通 sampler。
- 有 draft tokens 且有 rejection sampler：走 rejection sampling，并传入 speculator 预先保存的
  draft logits。

返回 sampler output、每 request sampled 数、rejected 数。

### `postprocess_sampled()`

采样后更新：

- `req_states.num_computed_tokens`。
- `req_states.last_sampled_tokens`。
- sampler penalties 的 output bin counts。
- `all_token_ids` 和 `total_len`。
- `model_state.postprocess_state()`，让模型状态跟 token 接受/拒绝结果对齐。

## pooling 路径：`pool()`

pooling 模型不走 sampler。`pool()` 逻辑更短：

1. 取出 `execute_model_state`。
2. 先执行 `kv_connector.post_forward()`。
3. 非 last PP rank 只更新 computed tokens，并返回 KV connector output。
4. last PP rank 调 `pooling_runner.pool(hidden_states, input_batch, req_states)`。
5. 构造 `ModelRunnerOutput` 和 `AsyncPoolingOutput`。
6. 调 `postprocess_num_computed_tokens()`。

pooling 输出同样用异步 copy stream 搬到 CPU。

## Pipeline Parallelism

PP 对执行路径影响很大：

- first PP rank 负责 input ids、MM embeddings。
- 中间/非 first PP rank 不接收 input ids/embeds，而是接收上游 `IntermediateTensors`。
- 非 last PP rank 的模型输出不是 hidden states，而是下一段需要的 intermediate tensors。
- last PP rank 负责 logits、采样、prompt logprobs、pooling。
- 采样结果通过 `PPHandler.broadcast()` 从 last rank 传给其它 ranks。
- 非 last rank 在下一步 `update_pp_decode_requests()` 或 `sample_tokens()` 中接收结果并更新本地状态。

因此 `execute_model()` 和 `sample_tokens()` 被拆开很自然：PP 非 last rank forward 后没有 logits，
但仍必须保存/更新状态并参与 KV connector 后处理。

## Speculative Decoding

spec decode 在 runner 中有三层影响：

- 初始化：last PP rank 创建 speculator；某些方法要求 target 输出 auxiliary hidden states。
- 输入准备：`prepare_inputs()` 处理 scheduler 下发的 draft tokens，扩展 logits 数和 mapping。
- 采样后：last rank 用 target hidden states 调 `speculator.propose()` 生成下一轮 draft tokens。

`decode_query_len = num_speculative_steps + model_state.num_new_sampled_tokens_per_step`。这会影响：

- dummy run 的 uniform decode shape。
- request 排序。
- sampler logits 数。
- CUDA graph capture/call shape。

如果模型提供 `get_mtp_target_hidden_states()`，runner 会优先把该 hidden state 传给 drafter，而不是
直接用 target model 最终 hidden states。

## Multimodal

multimodal 只在 first PP rank 准备：

- `EncoderCache` 保存 encoder 侧 MM cache。
- `add_requests()` 把新 request 的 `mm_features` 登记到 encoder cache。
- `execute_model()` 中如本步有 scheduled encoder inputs，会通过 `model_state.get_mm_embeddings()`
  取得 embeddings。
- LoRA + MM 时，先调用 `set_active_mm_loras()`，保证 encoder 侧 LoRA 与 request 对齐。
- `reset_mm_cache()` 和 `reset_encoder_cache()` 提供显式清理入口。

对于 speculator，如果 drafter 支持 MM，`sample_tokens()` 会在 postprocess 前 gather cached
MM embeddings，传给 `speculator.propose()`。

## LoRA

LoRA 相关逻辑分布在三处：

- `__init__` 创建 `LoraState` 和 capture cases。
- `load_model()` 用 mixin 包装 LoRA 模型。
- `add_requests()` 把 request 的 LoRA 信息登记到 `LoraState`。
- `execute_model()` 根据本 batch request 激活 LoRA adapters。
- CUDA graph dispatch 前会计算 `num_active_loras`，让 graph 选择匹配的 capture case。

dummy/capture 路径还通过 mixin 的上下文创建 dummy LoRA，确保 capture 覆盖 LoRA case。

## DCP/CP

DCP 主要影响 KV block table 和 seq lens：

- `initialize_kv_cache()` 计算每 rank block 上限时，用 `spec.block_size * dcp_size`，因为当前 rank
  的一个 KV block 覆盖全局序列中的更多 token。
- `BlockTables` 初始化时带入 cp size/rank/interleave。
- `prepare_inputs()` 调 `prepare_dcp_local_seq_lens()` 生成当前 rank 的 local seq lens。

`initialize_kv_cache()` 末尾还调用 `check_attention_cp_compatibility()` 做配置校验。

## EPLB

EPLB 是 expert parallelism load balancer。model runner 只提供钩子：

- load 前 `prepare_load()`。
- load 后注册模型/speculator 并启动 async loop。
- forward 前 `prepare_forward()`。
- `_dummy_run()`、`sample_tokens()`、`pool()` 等通过 `@step_eplb_after()` 装饰器在 step 后推进。
- 文件末尾暴露 `eplb_state`、`eep_eplb_suppressed` 和 `setup_eplb_from_mapping()`。

## shutdown

`shutdown()` 做显存清理：

- 同步 accelerator。
- 清空 KV cache list 和 attention groups。
- 删除 KV cache config、model_state、model、speculator。
- 调 `free_before_shutdown()` 释放 workspace。
- Python GC + torch empty cache。

它的目标是同进程重启/销毁 worker 时尽量回收模型权重、KV cache 和 workspace。

## 单步执行时序总结

生成模型的一步可以概括为：

```text
SchedulerOutput
  -> execute_model()
      -> update_pp_decode_requests()
      -> finish_requests/free_states/add_requests/update_requests()
      -> BlockTables.apply_staged_writes()
      -> dispatch_cg_and_sync_dp()
      -> prepare_inputs()
      -> prepare_attn()
      -> model_state.preprocess_state()
      -> activate LoRA / prepare MM embeddings / prepare model inputs
      -> kv_connector.pre_forward()
      -> model forward or CUDA graph replay
      -> save ExecuteModelState
  -> sample_tokens()
      -> sample logits
      -> PP broadcast if needed
      -> prompt logprobs
      -> AsyncOutput starts D2H copy
      -> postprocess_sampled()
      -> speculator.propose()
      -> kv_connector.post_forward()
      -> return AsyncOutput / ModelRunnerOutput
```

pooling 模型把 `sample_tokens()` 替换为 `pool()`，其余 forward 前半段基本相同。

## 设计取舍

- `model_runner.py` 是编排层，不直接承载模型特化逻辑。模型差异下沉到 `model_state` 和模型类。
- request 状态以固定 slot + staged tensor 的方式管理，避免每步频繁分配 GPU tensor。
- `execute_model()` 和 `sample_tokens()/pool()` 分离，使 PP、异步输出 copy、speculator proposal 可以更好地重叠。
- CUDA graph 逻辑通过 `BatchExecutionDescriptor` 和 `ModelCudaGraphManager` 隔离，runner 只选择 FULL/PIECEWISE/NONE 路径。
- KV connector 被设计成 forward 前后 hook，不侵入模型主逻辑，但仍能拿到 forward context 和 finished req ids。
- PP/DCP/MM/LoRA/spec decode 都在 runner 中露出最小公共钩子，真正复杂逻辑在对应 helper 模块中。

