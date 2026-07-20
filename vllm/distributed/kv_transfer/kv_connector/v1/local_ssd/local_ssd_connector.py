# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""一个教学用的 Local SSD KV connector。

这个 connector 故意写得比生产级 connector 更直白：

* scheduler 侧用 token prefix hash 判断本地 SSD 目录里是否已有 KV block；
* worker 侧把 GPU KV cache gather 成一个 block/page，再写入普通文件；
* load 时从普通文件读出 page，再 scatter 回 vLLM 的 paged KV cache；
* 不做后台线程、不做 LRU 驱逐、不做 metadata server；
* 用较多中文注释解释函数、分支和循环，便于逐行学习。

注意：这份实现面向学习和实验，
不建议直接作为生产 connector 使用。
"""

import hashlib
import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils import (
    gather_scatter_helper,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class LocalSSDLoadInfo:
    """记录一次 load 需要从本地 SSD 读取多少个 block。"""

    # 已经由 GPU prefix cache 或本地计算覆盖的 block 数。
    num_computed_blocks: int
    # 需要从本地 SSD 读取的新 block 数。
    num_blocks_to_load: int
    # scheduler 分配给这些外部 KV blocks 的 GPU block id。
    need_fetch_block_ids: list[int]


@dataclass
class LocalSSDSaveInfo:
    """记录一次 save 需要跳过多少个已经保存过的前缀 block。"""

    # 前面这些 block 已经命中过或保存过；
    # 本次只写后面的新增 block。
    skip_leading_blocks: int


@dataclass
class LocalSSDRequestState:
    """scheduler 侧为单个 request 保存的教学版状态机。"""

    # request_id 是贯穿 scheduler 和 worker metadata 的唯一标识。
    request_id: str
    # 原始 Request 对象只在 scheduler 侧使用；
    # 用来取 token 和多模态标识。
    request: "Request | None" = None
    # 已经调度到 worker 的 token；save/load 都只处理完整 block。
    token_ids: list[int] = field(default_factory=list)
    # request 当前关联的 GPU block id；本实现只支持第 0 个 KV group。
    allocated_block_ids: list[int] = field(default_factory=list)
    # 已经生成过 save metadata 的 block 数，避免每步重复写旧 block。
    num_saved_blocks: int = 0
    # 多模态输入的稳定标识，参与 hash；
    # 避免不同图片/音频共用同一路径。
    mm_hashes: list[str] = field(default_factory=list)
    # 如果本 request 有外部 KV 命中，这里描述要 load 的范围。
    load_op: LocalSSDLoadInfo | None = None
    # 简单三段状态：NEW -> WAITING_TO_LOAD -> ACTIVE。
    phase: str = "NEW"

    def needs_loading(self) -> bool:
        """判断这个 request 是否还有 SSD blocks 需要 load。"""
        return self.load_op is not None and self.load_op.num_blocks_to_load > 0

    def is_ready_to_load(self) -> bool:
        """判断 scheduler 是否已经拿到了 load 目标 GPU blocks。"""
        return self.phase == "WAITING_TO_LOAD" and self.needs_loading()

    def update_tokens_and_blocks(
        self,
        new_token_ids: list[int],
        new_block_ids: tuple[list[int], ...] | None,
    ) -> None:
        """追加 cached request 在本步新增的 token/block。"""
        # 追加新 token：decode 或继续 prefill 时会逐步增长。
        if new_token_ids:
            self.token_ids.extend(new_token_ids)

        # 追加新 block：本教学 connector 只使用第 0 个 KV cache group。
        if new_block_ids is not None:
            self.allocated_block_ids.extend(_first_group_block_ids(new_block_ids))


@dataclass
class LocalSSDRequestMetadata:
    """发送给 worker 的单个 request metadata。"""

    # 当前 request 的 ID。
    request_id: str
    # worker 用 token_ids 重新计算 block hash，从而找到文件名。
    token_ids: list[int]
    # worker 用 block_ids 定位 GPU paged KV cache 中的源/目的 block。
    block_ids: list[int]
    # 多模态标识参与 hash，保证文件名和 scheduler 侧一致。
    mm_hashes: list[str]
    # load_op 不为空时，worker 会先从 SSD 读取这些 blocks。
    load_op: LocalSSDLoadInfo | None = None
    # save_op 不为空时，worker 会把新增 blocks 写入 SSD。
    save_op: LocalSSDSaveInfo | None = None

    @staticmethod
    def from_state(
        state: LocalSSDRequestState,
        block_size: int,
        load_op: LocalSSDLoadInfo | None = None,
        skip_leading_blocks: int | None = None,
    ) -> "LocalSSDRequestMetadata | None":
        """把 scheduler 内部状态转换成 worker 可消费的 metadata。"""
        # 只处理完整 block，最后一个未满 block 不写入 SSD。
        total_blocks = len(state.token_ids) // block_size

        # load 命中时要跳过已命中的前缀；
        # 普通 save 则跳过已保存部分。
        skip_blocks = (
            state.num_saved_blocks
            if skip_leading_blocks is None
            else skip_leading_blocks
        )

        # 如果没有新增完整 block，且也没有 load 任务，
        # 就不生成 metadata。
        new_blocks_to_save = total_blocks - state.num_saved_blocks
        if new_blocks_to_save <= 0 and load_op is None:
            return None

        # 记录已经发给 worker 保存的 block 数，下一步避免重复保存。
        state.num_saved_blocks = total_blocks
        return LocalSSDRequestMetadata(
            request_id=state.request_id,
            token_ids=state.token_ids.copy(),
            block_ids=state.allocated_block_ids.copy(),
            mm_hashes=state.mm_hashes.copy(),
            load_op=load_op,
            save_op=LocalSSDSaveInfo(skip_leading_blocks=skip_blocks),
        )


@dataclass
class LocalSSDConnectorMetadata(KVConnectorMetadata):
    """一个 engine step 内，scheduler 发给 worker 的全部 metadata。"""

    # 每个 request 对应一个独立的 metadata 条目。
    requests: list[LocalSSDRequestMetadata] = field(default_factory=list)

    def add_request(self, request_metadata: LocalSSDRequestMetadata) -> None:
        """追加一个 request metadata。"""
        self.requests.append(request_metadata)


def _first_group_block_ids(block_ids: tuple[list[int], ...] | list[int]) -> list[int]:
    """取第 0 个 KV cache group 的 block ids。

    这个教学 connector 不实现 HMA/多 KV group，因此统一只看第一组。
    """
    # 某些路径已经传入 list[int]，直接 copy，避免修改调用方数据。
    if isinstance(block_ids, list):
        return block_ids.copy()

    # tuple 为空说明没有可用 block；
    # 正常 vLLM 路径一般不会走到这里。
    if not block_ids:
        return []

    # 只使用第 0 组，便于学习单组 attention KV cache 的完整链路。
    return block_ids[0].copy()


def _extract_mm_hashes(request_or_data: Any) -> list[str]:
    """从 Request 或 NewRequestData 里抽取多模态特征标识。"""
    # 没有 mm_features 时返回空列表，让纯文本 prompt 的 hash 更简单。
    mm_features = getattr(request_or_data, "mm_features", [])

    # 核心循环：逐个取 identifier；缺少 identifier 时退化为 repr。
    return [
        str(getattr(feature, "identifier", repr(feature)))
        for feature in mm_features
    ]


def _align_to_block_size(num_tokens: int, block_size: int) -> int:
    """把 token 数向下对齐到 block size。"""
    # 小于等于 0 时直接返回 0，避免负数参与后续 range。
    if num_tokens <= 0:
        return 0

    # 使用整除再乘回去；
    # 得到不超过输入值的最大完整 block token 数。
    return (num_tokens // block_size) * block_size


class LocalSSDConnector(KVConnectorBase_V1):
    """基于本地 SSD 目录和普通文件的教学版 KV connector。"""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        """初始化 scheduler 或 worker 侧 connector。"""
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )

        # block_size 决定 token 序列如何切成 KV cache blocks。
        self._block_size = vllm_config.cache_config.block_size
        # MLA 和 MHA 的 KV cache tensor shape 不同，gather/scatter 时要区分。
        self._use_mla = vllm_config.model_config.use_mla
        # 本地 SSD 根目录可通过 kv_connector_extra_config 配置。
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "local_ssd_storage_path",
            "/tmp/vllm_local_ssd_kv",
        )
        # fsync 默认关闭，学习和调试更快；
        # 需要崩溃一致性时可打开。
        self._fsync = bool(
            self._kv_transfer_config.get_from_extra_config(
                "local_ssd_fsync",
                False,
            )
        )
        # scheduler 用 expected_ranks 判断一个 key 是否所有 rank 都写完。
        self._expected_ranks = int(
            self._kv_transfer_config.get_from_extra_config(
                "local_ssd_expected_ranks",
                vllm_config.parallel_config.tensor_parallel_size,
            )
        )

        # scheduler 侧维护 request 状态；worker 侧不需要这张表。
        if role == KVConnectorRole.SCHEDULER:
            self._states: dict[str, LocalSSDRequestState] = {}

        logger.info(
            "LocalSSDConnector initialized: role=%s path=%s expected_ranks=%d",
            role.name,
            self._storage_path,
            self._expected_ranks,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """worker 侧注册 vLLM 的 paged KV cache tensors。"""
        # 保存原始 KV cache tensors；
        # 后续 gather/scatter 直接通过指针访问。
        self._kv_caches = kv_caches
        # 解析 tensor shape，得到单个 SSD page 的 shape 和字节数。
        self._setup_kv_cache_shape()
        # worker rank 决定当前进程读写 rank_N 目录。
        self._rank = get_tensor_model_parallel_rank()
        # 每个 worker 只创建自己的 rank 目录。
        os.makedirs(self._rank_dir(self._rank), exist_ok=True)
        # load 失败的 GPU block id 会记录下来；
        # 便于 scheduler 后续失效处理。
        self._load_error_blocks: set[int] = set()

    def _setup_kv_cache_shape(self) -> None:
        """根据第一层 KV cache 推导本 connector 的 page 布局。"""
        # 取第一层作为代表；vLLM 同一模型的各层 KV shape 通常一致。
        first_cache = next(iter(self._kv_caches.values()))
        self._device = first_cache.device
        self._dtype = first_cache.dtype
        element_size = first_cache.element_size()

        # MLA: 每层是 [num_blocks, block_size, head_size]。
        if self._use_mla:
            assert len(first_cache.shape) == 3
            num_blocks, block_size, head_size = first_cache.shape
            layer_block_size = block_size * head_size * element_size
            self._shape_per_page = [
                len(self._kv_caches),
                block_size,
                head_size,
            ]
        else:
            # MHA: 每层是 [2, num_blocks, block_size, num_heads, head_size]。
            assert len(first_cache.shape) == 5
            _, num_blocks, block_size, num_heads, head_size = first_cache.shape
            layer_block_size = 2 * block_size * num_heads * head_size * element_size
            self._shape_per_page = [
                len(self._kv_caches),
                2,
                block_size,
                num_heads * head_size,
            ]

        # 这两个字段用于把 block id 转成 token indices。
        self._local_block_size = block_size
        self._local_total_tokens = num_blocks * block_size
        # 一个文件就是一个完整 KV page；
        # 包含所有 layers 的该 block 数据。
        self._bytes_per_page = layer_block_size * len(self._kv_caches)
        # gather/scatter helper 需要所有 layer tensor 的 data_ptr。
        self._kvcache_ptrs = torch.tensor(
            [cache.data_ptr() for cache in self._kv_caches.values()],
            dtype=torch.int64,
            device=self._device,
        )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """worker 侧在 forward 前同步从本地 SSD 加载 KV。"""
        # 每个 step 开始时清空旧错误，避免误报之前 step 的失败。
        self._load_error_blocks.clear()

        # 从 base class 取 scheduler 绑定过来的 metadata。
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, LocalSSDConnectorMetadata)

        # 核心循环：一个 request 一个 request 地处理 load。
        for request in metadata.requests:
            if request.load_op is None:
                continue

            # load block ids 是 scheduler 刚分配给外部 KV 的目标 GPU blocks。
            block_ids = request.block_ids[: request.load_op.num_blocks_to_load]
            block_hashes = self._generate_block_hashes(
                request.token_ids,
                request.mm_hashes,
                request.load_op.num_computed_blocks,
                len(block_ids),
            )

            # 内层循环：一个 block 对应一个本地 SSD 文件。
            for block_id, block_hash in zip(block_ids, block_hashes):
                if not self._load_one_block(block_id, block_hash):
                    self._load_error_blocks.add(block_id)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """本实现不是 layer-by-layer load，因此这里无需等待。"""
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """本实现等到 wait_for_save 再一次性保存所有 layers。"""
        return

    def wait_for_save(self) -> None:
        """worker 侧在 forward 后同步把新增 KV blocks 写入本地 SSD。"""
        # 从 base class 取 scheduler 绑定过来的 metadata。
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, LocalSSDConnectorMetadata)

        # 核心循环：一个 request 一个 request 地处理 save。
        for request in metadata.requests:
            if request.save_op is None:
                continue

            # skip_blocks 表示前缀部分已经保存过；
            # 本步只保存新增完整 blocks。
            skip_blocks = request.save_op.skip_leading_blocks
            block_hashes = self._generate_block_hashes(
                request.token_ids,
                request.mm_hashes,
                skip_blocks,
            )
            block_ids = request.block_ids[skip_blocks : skip_blocks + len(block_hashes)]

            # 内层循环：一个 GPU block gather 后写成一个文件。
            for block_id, block_hash in zip(block_ids, block_hashes):
                self._save_one_block(block_id, block_hash)

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """同步读写版本没有异步完成的 request 需要上报。"""
        return None, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        """返回本 step 中 load 失败的 GPU block ids。"""
        # 返回 copy，避免调用方无意修改 connector 内部集合。
        return set(getattr(self, "_load_error_blocks", set()))

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """scheduler 侧 request 结束回调。

        返回 False 表示本 connector 没有接管 GPU block 的异步释放。
        """
        # 同步 connector 不需要延迟释放 request blocks。
        return False, None

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """scheduler 侧检查本地 SSD 中可复用的连续 prefix token 数。"""
        # 保存 request，用于 build_connector_meta 阶段继续生成 metadata。
        state = self._get_or_create_state(request.request_id)
        state.request = request
        state.mm_hashes = _extract_mm_hashes(request)
        assert request.prompt_token_ids is not None

        # vLLM 不希望外部 connector 加载最后一个未稳定 token。
        num_tokens_to_check = _align_to_block_size(
            len(request.prompt_token_ids) - 1,
            self._block_size,
        )

        # 如果本地已经计算的 token 覆盖了检查范围，
        # 就不需要外部加载。
        if num_tokens_to_check <= num_computed_tokens:
            state.load_op = LocalSSDLoadInfo(
                num_computed_blocks=num_computed_tokens // self._block_size,
                num_blocks_to_load=0,
                need_fetch_block_ids=[],
            )
            return 0, False

        # 为完整 prefix blocks 生成 hash；
        # 再检查每个 rank 目录里的文件。
        token_ids_to_check = request.prompt_token_ids[:num_tokens_to_check]
        block_hashes = self._generate_block_hashes(
            token_ids_to_check,
            state.mm_hashes,
            start_block_id=0,
        )

        matched_blocks = 0
        # 核心循环：必须从前往后连续命中；
        # 遇到第一个 miss 就停止。
        for block_hash in block_hashes:
            if not self._block_exists_for_all_expected_ranks(block_hash):
                break
            matched_blocks += 1

        # 只把超过 num_computed_tokens 的部分报告给 scheduler。
        matched_tokens = matched_blocks * self._block_size
        new_hit_tokens = max(0, matched_tokens - num_computed_tokens)
        state.load_op = LocalSSDLoadInfo(
            num_computed_blocks=num_computed_tokens // self._block_size,
            num_blocks_to_load=new_hit_tokens // self._block_size,
            need_fetch_block_ids=[],
        )

        logger.info(
            "LocalSSD token match: req=%s matched_blocks=%d new_hit_tokens=%d",
            request.request_id,
            matched_blocks,
            new_hit_tokens,
        )
        return new_hit_tokens, new_hit_tokens > 0

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """scheduler 侧在外部 KV 目标 GPU blocks 分配后更新状态。"""
        state = self._get_or_create_state(request.request_id)
        state.request = request

        # 没有外部 token 时直接返回；
        # 避免误把普通 blocks 当成 load 目标。
        if num_external_tokens <= 0 or not state.needs_loading():
            return

        assert state.load_op is not None
        expected_blocks = state.load_op.num_blocks_to_load
        actual_blocks = num_external_tokens // self._block_size
        assert actual_blocks == expected_blocks

        # 只支持单 KV group；
        # 所以用 get_unhashed_block_ids 取外部加载目标。
        state.load_op.need_fetch_block_ids.extend(blocks.get_unhashed_block_ids())
        state.phase = "WAITING_TO_LOAD"

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """scheduler 侧为当前 engine step 构造 worker metadata。"""
        metadata = LocalSSDConnectorMetadata()

        # 清理已经结束的 request 状态，避免状态表无限增长。
        for request_id in scheduler_output.finished_req_ids:
            self._states.pop(request_id, None)

        # 先处理已经等到目标 GPU blocks 的 load request。
        self._process_waiting_load_requests(metadata)
        # 再处理本步首次进入 scheduler output 的新 request。
        self._process_new_requests(scheduler_output, metadata)
        # 最后处理已经在运行、
        # 且这一步又新增 token/block 的 cached request。
        self._process_cached_requests(scheduler_output, metadata)
        return metadata

    def _process_waiting_load_requests(
        self,
        metadata: LocalSSDConnectorMetadata,
    ) -> None:
        """把 WAITING_TO_LOAD request 转成 load metadata。"""
        # 遍历 copy 后的 values，允许循环中修改 state.phase。
        for state in list(self._states.values()):
            if not state.is_ready_to_load():
                continue
            assert state.load_op is not None
            assert state.request is not None
            assert state.request.prompt_token_ids is not None

            # load metadata 只需要覆盖命中的 prefix blocks。
            num_cached_blocks = (
                state.load_op.num_computed_blocks + state.load_op.num_blocks_to_load
            )
            num_tokens_to_compute = num_cached_blocks * self._block_size
            state.token_ids = state.request.prompt_token_ids[
                :num_tokens_to_compute
            ].copy()
            state.allocated_block_ids = state.load_op.need_fetch_block_ids.copy()

            request_metadata = LocalSSDRequestMetadata.from_state(
                state,
                self._block_size,
                state.load_op,
                num_cached_blocks,
            )
            if request_metadata is not None:
                metadata.add_request(request_metadata)
                state.phase = "ACTIVE"

    def _process_new_requests(
        self,
        scheduler_output: SchedulerOutput,
        metadata: LocalSSDConnectorMetadata,
    ) -> None:
        """把本步新调度的 request 转成 save metadata。"""
        # 核心循环：逐个读取 NewRequestData，初始化本地状态。
        for request in scheduler_output.scheduled_new_reqs:
            state = self._get_or_create_state(request.req_id)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )

            # 新 request 的 block ids 来自 scheduler output。
            state.token_ids = (request.prompt_token_ids or [])[
                :num_tokens_to_compute
            ].copy()
            state.allocated_block_ids = _first_group_block_ids(request.block_ids)
            state.mm_hashes = _extract_mm_hashes(request)
            state.num_saved_blocks = 0

            # 如果前面已经有 load 命中，
            # save 时跳过已命中的前缀 blocks。
            num_cached_blocks = None
            if state.load_op is not None:
                num_cached_blocks = (
                    state.load_op.num_computed_blocks + state.load_op.num_blocks_to_load
                )

            request_metadata = LocalSSDRequestMetadata.from_state(
                state,
                self._block_size,
                None,
                num_cached_blocks,
            )
            if request_metadata is not None:
                metadata.add_request(request_metadata)
                state.phase = "ACTIVE"

    def _process_cached_requests(
        self,
        scheduler_output: SchedulerOutput,
        metadata: LocalSSDConnectorMetadata,
    ) -> None:
        """把运行中 request 的新增 tokens/blocks 转成 save metadata。"""
        cached_reqs = scheduler_output.scheduled_cached_reqs

        # 核心循环：cached_reqs 的多个数组
        # 用相同下标描述同一个 request。
        for i, request_id in enumerate(cached_reqs.req_ids):
            state = self._get_or_create_state(request_id)
            if state.request is None:
                continue

            # 优先使用 all_token_ids；
            # 如果没有，就使用 new_token_ids 增量。
            num_new_tokens = scheduler_output.num_scheduled_tokens[request_id]
            num_current_tokens = len(state.token_ids)
            all_token_ids = cached_reqs.all_token_ids.get(request_id)
            new_block_ids = cached_reqs.new_block_ids[i]

            # preemption 恢复时，scheduler 给的是整条 request 的 blocks；
            # 这时应替换旧 block 列表，而不是追加。
            if request_id in cached_reqs.resumed_req_ids:
                total_tokens = cached_reqs.num_computed_tokens[i] + num_new_tokens
                if all_token_ids is not None:
                    state.token_ids = all_token_ids[:total_tokens].copy()
                else:
                    state.token_ids = state.request.all_token_ids[
                        :total_tokens
                    ].copy()
                if new_block_ids is not None:
                    state.allocated_block_ids = _first_group_block_ids(new_block_ids)
            else:
                # 普通 cached request 只追加本步新 token 和新 blocks。
                if all_token_ids is not None:
                    new_token_ids = all_token_ids[
                        num_current_tokens : num_current_tokens + num_new_tokens
                    ]
                else:
                    new_token_ids = cached_reqs.new_token_ids[i]
                state.update_tokens_and_blocks(new_token_ids, new_block_ids)

            request_metadata = LocalSSDRequestMetadata.from_state(
                state,
                self._block_size,
                None,
            )
            if request_metadata is not None:
                metadata.add_request(request_metadata)

    def _get_or_create_state(self, request_id: str) -> LocalSSDRequestState:
        """获取已有 request 状态；如果没有就创建一个。"""
        # scheduler 侧一定有 _states；worker 不会调用这个函数。
        if request_id not in self._states:
            self._states[request_id] = LocalSSDRequestState(request_id=request_id)
        return self._states[request_id]

    def _generate_block_hashes(
        self,
        token_ids: list[int],
        mm_hashes: list[str],
        start_block_id: int,
        max_blocks_count: int | None = None,
    ) -> list[str]:
        """根据 token blocks 生成稳定的链式 prefix hashes。"""
        block_hashes: list[str] = []
        previous_hash = ""

        # 核心循环：每个完整 token block 生成一个 hash。
        for start_idx in range(0, len(token_ids), self._block_size):
            if start_idx + self._block_size > len(token_ids):
                break

            block_index = start_idx // self._block_size
            current_tokens = token_ids[start_idx : start_idx + self._block_size]
            current_hash = self._compute_prefix_hash(
                current_tokens,
                mm_hashes,
                previous_hash,
            )

            # start_block_id 用于跳过已经命中或已经保存的前缀 blocks。
            if block_index >= start_block_id:
                block_hashes.append(current_hash)

            # max_blocks_count 用于 load；
            # 只生成 scheduler 分配的目标数量。
            if max_blocks_count is not None and len(block_hashes) >= max_blocks_count:
                break

            previous_hash = current_hash

        return block_hashes

    def _compute_prefix_hash(
        self,
        token_block: list[int],
        mm_hashes: list[str],
        previous_hash: str,
    ) -> str:
        """计算单个 block 的文件 key。"""
        # 用简单文本拼接保持可读性；
        # sha256 只用于稳定命名，不是安全边界。
        payload = (
            f"model={self._vllm_config.model_config.model}|"
            f"block_size={self._block_size}|"
            f"use_mla={self._use_mla}|"
            f"prev={previous_hash}|"
            f"mm={','.join(mm_hashes)}|"
            f"tokens={token_block}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _rank_dir(self, rank: int) -> str:
        """返回某个 tensor-parallel rank 的根目录。"""
        return os.path.join(self._storage_path, f"rank_{rank}")

    def _block_file_path(self, rank: int, block_hash: str) -> str:
        """把 block hash 映射成本地 SSD 文件路径。"""
        # 两级 hash 分片目录可以避免单目录下文件过多。
        return os.path.join(
            self._rank_dir(rank),
            block_hash[:2],
            block_hash[2:4],
            f"{block_hash}.bin",
        )

    def _block_exists_for_all_expected_ranks(self, block_hash: str) -> bool:
        """检查一个 block 是否已经由所有预期 ranks 写入。"""
        # scheduler 侧没有注册 KV tensors，因此通常不知道 page 字节数。
        expected_size = getattr(self, "_bytes_per_page", None)

        # 核心循环：只要有任意 rank 的文件缺失或大小不对，
        # 就视为 miss。
        for rank in range(self._expected_ranks):
            path = self._block_file_path(rank, block_hash)
            if not os.path.exists(path):
                return False
            if expected_size is not None and os.path.getsize(path) != expected_size:
                return False
        return True

    def _save_one_block(self, block_id: int, block_hash: str) -> None:
        """把一个 GPU KV block gather 后写成本地 SSD 文件。"""
        path = self._block_file_path(self._rank, block_hash)

        # 文件已存在且大小正确时跳过，
        # 避免重复写同一个 prefix block。
        if os.path.exists(path) and os.path.getsize(path) == self._bytes_per_page:
            return

        # gather 得到一个包含所有 layers 的连续 page tensor。
        page_tensor = torch.empty(
            self._shape_per_page,
            dtype=self._dtype,
            device=self._device,
        )
        self._gather_or_scatter_one_block(block_id, page_tensor, "gather")
        data = self._tensor_to_bytes(page_tensor)

        # 先写 tmp 再 os.replace，避免 scheduler 看到半个文件。
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp_path, "wb") as f:
            f.write(data)
            if self._fsync:
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp_path, path)

    def _load_one_block(self, block_id: int, block_hash: str) -> bool:
        """从本地 SSD 读取一个 block，并 scatter 到 GPU KV cache。"""
        path = self._block_file_path(self._rank, block_hash)

        # 文件不存在说明 scheduler 判断后到 worker 读取前
        # 发生了不一致。
        if not os.path.exists(path):
            logger.warning("LocalSSD block file missing: %s", path)
            return False

        # 文件大小不正确时拒绝加载，
        # 避免把损坏数据写回 GPU KV cache。
        if os.path.getsize(path) != self._bytes_per_page:
            logger.warning("LocalSSD block file has wrong size: %s", path)
            return False

        # 一次读完整 page，再转成目标 dtype/device 的 tensor。
        with open(path, "rb") as f:
            data = f.read()
        page_tensor = self._bytes_to_tensor(data)
        self._gather_or_scatter_one_block(block_id, page_tensor, "scatter")
        return True

    def _gather_or_scatter_one_block(
        self,
        block_id: int,
        page_tensor: torch.Tensor,
        operation: str,
    ) -> None:
        """在 GPU paged KV cache 和单个 page tensor 之间搬运数据。"""
        start_idx = block_id * self._local_block_size
        token_indices = list(range(start_idx, start_idx + self._local_block_size))

        # 根据 operation 选择 gather 或 scatter；
        # 便于读写复用同一套寻址逻辑。
        if operation == "gather":
            gather_scatter_helper.gather_kv_caches(
                self._kvcache_ptrs,
                self._local_total_tokens,
                page_tensor,
                token_indices,
                is_mla=self._use_mla,
            )
        else:
            gather_scatter_helper.scatter_kv_caches(
                self._kvcache_ptrs,
                self._local_total_tokens,
                page_tensor,
                token_indices,
                is_mla=self._use_mla,
            )

    def _tensor_to_bytes(self, tensor: torch.Tensor) -> bytes:
        """把 GPU/CPU tensor 转成可写入文件的原始 bytes。"""
        # cpu() 会把 GPU 数据同步拷贝到 CPU；
        # contiguous 保证 bytes 顺序稳定。
        cpu_tensor = tensor.detach().contiguous().cpu()

        # bfloat16 在 numpy 中支持有限，转成 uint16 bytes 更稳。
        if cpu_tensor.dtype == torch.bfloat16:
            return cpu_tensor.view(dtype=torch.uint16).numpy().tobytes()

        return cpu_tensor.numpy().tobytes()

    def _bytes_to_tensor(self, data: bytes) -> torch.Tensor:
        """把文件 bytes 还原成目标 dtype/device 上的 page tensor。"""
        # bytearray 提供可写 buffer；
        # 避免 torch.frombuffer 对只读 bytes 报警。
        mutable_data = bytearray(data)

        # bfloat16 先按 uint16 读入，再 view 回 bfloat16。
        if self._dtype == torch.bfloat16:
            cpu_tensor = torch.frombuffer(mutable_data, dtype=torch.uint16)
            cpu_tensor = cpu_tensor.view(dtype=torch.bfloat16)
        else:
            cpu_tensor = torch.frombuffer(mutable_data, dtype=self._dtype)

        # reshape 恢复 page shape，再复制到 worker 的目标 GPU device。
        return cpu_tensor.reshape(self._shape_per_page).to(
            self._device,
            non_blocking=True,
        )
