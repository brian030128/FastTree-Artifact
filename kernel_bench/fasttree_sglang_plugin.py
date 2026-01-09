"""
FastTree SGLang-style plugin for paged KV cache.

This module provides:
- FastTreeMetadata: Pre-allocated buffers for FastTree scheduling
- fasttree_decode: Kernel that supports indirect KV indexing via page table

The key difference from the basic fasttree.py is that this version
uses a page table (vnode_to_kv_entries / req_to_token) for indirect
indexing into the KV buffer, avoiding any data copy.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


class FastTreeMetadata:
    """
    Pre-allocated metadata buffers for FastTree with paged KV cache.
    
    This follows SGLang's pattern of pre-allocating GPU buffers to avoid
    allocation overhead during inference.
    """
    
    def __init__(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        device: str = "cuda",
        alpha: float = 0.6,
        beta: float = 0.3,
        gamma: float = 0.1,
        TSQs: Tuple[int, int] = None,
        TSKs: Tuple[int, int] = (32, 32),
        kv_split_sizes: Tuple[int, int] = (1024, 128),
        para_threshs1: Tuple[int, int] = (132, 132),
        para_threshs2: Tuple[int, int] = (132, 528),
        max_vnodes: int = 16384,
        max_vnode_to_q_entries: int = 81920,
        max_batch_size: int = 8192,
    ):
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_group_num = num_qo_heads // num_kv_heads
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.TSQs = TSQs
        if self.TSQs is None:
            self.TSQs = (64 // self.kv_group_num, 16 // self.kv_group_num)
        self.TSKs = TSKs
        self.kv_split_sizes = kv_split_sizes
        self.para_threshs1 = para_threshs1
        self.para_threshs2 = para_threshs2

        with torch.device(device):
            self.vnode_to_q_entries = torch.empty(max_vnode_to_q_entries, dtype=torch.int32)
            self.vnode_to_q_offs = torch.empty(max_vnodes, dtype=torch.int32)
            self.vnode_to_q_lens = torch.empty(max_vnodes, dtype=torch.int32)
            self.vnode_to_kv_offs = torch.empty(max_vnodes, dtype=torch.int32)
            self.vnode_to_kv_lens = torch.empty(max_vnodes, dtype=torch.int32)
            self.req_to_vnode_entries = torch.empty(max_vnode_to_q_entries, dtype=torch.int32)
            self.req_to_vnode_offs = torch.empty(max_batch_size, dtype=torch.int32)
            self.req_to_vnode_lens = torch.empty(max_batch_size, dtype=torch.int32)
            self.mid_o = torch.empty((max_vnode_to_q_entries, num_qo_heads, head_dim), dtype=torch.float32)
            self.mid_lse = torch.empty((max_vnode_to_q_entries, num_qo_heads), dtype=torch.float32)

        self.phase_node_nums: Tuple[int, int] = None
        self.phase_node_offsets: Tuple[int, int] = None
        self.phase_q_tile_sizes: Tuple[int, int] = None
        self.phase_kv_tile_sizes: Tuple[int, int] = None


@triton.jit
def _fwd_fasttree_decode_stage1_paged(
    Q,
    K,
    V,
    stride_q0,
    stride_q1,
    stride_k0,
    stride_k1,
    stride_v0,
    stride_v1,
    stride_o0,
    stride_o1,
    stride_lse0,
    # Page table for indirect KV indexing
    vnode_to_kv_entries,  # This is the page table (req_to_token)
    vnode_to_kv_offs,
    vnode_to_kv_lens,
    vnode_to_q_entries,
    vnode_to_q_offs,
    vnode_to_q_lens,
    node_offset,
    sm_scale,
    MidO,
    MidLSE,
    head_dim: tl.constexpr,
    kv_group_num: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    KV_TILE_SIZE: tl.constexpr,
):
    """
    Stage 1 kernel with paged KV cache support.
    
    Uses double indirection:
    1. vnode_to_kv_offs[vnode] gives offset into vnode_to_kv_entries (page table)
    2. vnode_to_kv_entries[offset + i] gives actual slot index in K/V buffer
    """
    cur_vnode = tl.program_id(0) + node_offset
    cur_kv_head = tl.program_id(1)
    Q_BLOCK_SIZE: tl.constexpr = Q_TILE_SIZE * kv_group_num

    offs_d = tl.arange(0, head_dim)
    offs_g = tl.arange(0, kv_group_num)

    cur_q_entry_start = tl.load(vnode_to_q_offs + cur_vnode)
    cur_q_len = tl.load(vnode_to_q_lens + cur_vnode)

    cur_kv_entry_start = tl.load(vnode_to_kv_offs + cur_vnode)
    cur_kv_len = tl.load(vnode_to_kv_lens + cur_vnode)
    cur_qo_head = cur_kv_head * kv_group_num

    # Load queries
    offs_q_tile = tl.arange(0, Q_TILE_SIZE)
    q_entries = tl.load(
        vnode_to_q_entries + cur_q_entry_start + offs_q_tile,
        mask=offs_q_tile < cur_q_len,
        other=-1,
    )
    off_q = (
        q_entries[:, None, None] * stride_q0
        + (cur_qo_head + offs_g[None, :, None]) * stride_q1
        + offs_d[None, None, :]
    )
    q = tl.load(Q + off_q, mask=q_entries[:, None, None] >= 0, other=0.0)
    q = q.reshape([Q_BLOCK_SIZE, head_dim])

    acc = tl.zeros([Q_BLOCK_SIZE, head_dim], dtype=tl.float32)
    sum_exp = tl.zeros([Q_BLOCK_SIZE, 1], dtype=tl.float32)
    max_logics = tl.full([Q_BLOCK_SIZE, 1], float("-inf"), dtype=tl.float32)
    
    offs_kv_tile = tl.arange(0, KV_TILE_SIZE)
    for kv_tile_start in range(0, cur_kv_len, KV_TILE_SIZE):
        offs_kv_tile_new = offs_kv_tile + kv_tile_start
        
        # Load slot indices from page table (indirect indexing)
        kv_entries = tl.load(vnode_to_kv_entries + cur_kv_entry_start + offs_kv_tile_new)
        kv_entries = tl.where(offs_kv_tile_new < cur_kv_len, kv_entries, -1)
        
        # Load K/V using slot indices
        off_k = kv_entries[None, :] * stride_k0 + cur_kv_head * stride_k1 + offs_d[:, None]
        k = tl.load(K + off_k, mask=kv_entries[None, :] >= 0, other=0.0)
        
        qk = tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_kv_tile_new[None, :] < cur_kv_len, qk, -1.0e6)

        off_v = kv_entries[:, None] * stride_v0 + cur_kv_head * stride_v1 + offs_d[None, :]
        v = tl.load(V + off_v, mask=kv_entries[:, None] >= 0, other=0.0)

        cur_max_logics = tl.max(qk, axis=1)[:, None]
        new_max_logics = tl.maximum(cur_max_logics, max_logics)
        exp_logics = tl.exp(qk - new_max_logics)
        logic_scale = tl.exp(max_logics - new_max_logics)

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logics, axis=1)[:, None]
        acc = acc * logic_scale
        acc = tl.dot(exp_logics.to(v.type), v, acc)
        max_logics = new_max_logics

    off_mid_o = (
        (cur_q_entry_start + offs_q_tile[:, None, None]) * stride_o0
        + (cur_qo_head + offs_g[None, :, None]) * stride_o1
        + offs_d[None, None, :]
    )
    off_mid_lse = (
        (cur_q_entry_start + offs_q_tile[:, None]) * stride_lse0
        + cur_qo_head
        + offs_g[None, :]
    )

    mid_o = (acc / sum_exp).reshape([Q_TILE_SIZE, kv_group_num, head_dim])
    tl.store(MidO + off_mid_o, mid_o, mask=offs_q_tile[:, None, None] < cur_q_len)

    mid_lse = (tl.log(sum_exp) + max_logics).reshape([Q_TILE_SIZE, kv_group_num])
    tl.store(MidLSE + off_mid_lse, mid_lse, mask=offs_q_tile[:, None] < cur_q_len)


def _fasttree_decode_stage1(
    Q,
    K,
    V,
    vnode_to_kv_entries,
    vnode_to_kv_offs,
    vnode_to_kv_lens,
    vnode_to_q_entries,
    vnode_to_q_offs,
    vnode_to_q_lens,
    sm_scale,
    MidO,
    MidLSE,
    phase_node_nums,
    phase_node_offsets,
    head_num,
    head_dim,
    kv_head_num,
    phase_q_tile_sizes,
    phase_kv_tile_sizes,
):
    Lq, Lk = Q.shape[-1], K.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128, 256}

    kv_group_num = head_num // kv_head_num

    for node_num, node_offset, Q_TILE_SIZE, KV_TILE_SIZE in zip(
        phase_node_nums, phase_node_offsets, phase_q_tile_sizes, phase_kv_tile_sizes
    ):
        if node_num == 0:
            continue
        grid = (node_num, kv_head_num)
        _fwd_fasttree_decode_stage1_paged[grid](
            Q,
            K,
            V,
            Q.stride(0),
            Q.stride(1),
            K.stride(0),
            K.stride(1),
            V.stride(0),
            V.stride(1),
            MidO.stride(0),
            MidO.stride(1),
            MidLSE.stride(0),
            vnode_to_kv_entries,
            vnode_to_kv_offs,
            vnode_to_kv_lens,
            vnode_to_q_entries,
            vnode_to_q_offs,
            vnode_to_q_lens,
            node_offset,
            sm_scale,
            MidO,
            MidLSE,
            head_dim,
            kv_group_num,
            Q_TILE_SIZE,
            KV_TILE_SIZE,
        )


@triton.jit
def _fwd_fasttree_decoding_stage2(
    req_to_vnode_entries,
    req_to_vnode_offs,
    req_to_vnode_lens,
    MidO,
    MidLSE,
    O,
    stride_mid_o0,
    stride_mid_o1,
    stride_mid_lse0,
    stride_o0,
    stride_o1,
    head_dim: tl.constexpr,
    NODE_TILE_SIZE: tl.constexpr,
):
    cur_req = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_vnode_entry_start = tl.load(req_to_vnode_offs + cur_req)
    cur_vnode_len = tl.load(req_to_vnode_lens + cur_req)

    sum_exp = tl.zeros([1], dtype=tl.float32)
    max_lse = tl.full([1], float("-inf"), dtype=tl.float32)
    acc = tl.zeros([1, head_dim], dtype=tl.float32)

    offs_d = tl.arange(0, head_dim)
    offs_n = tl.arange(0, NODE_TILE_SIZE)
    for tile_start in range(0, cur_vnode_len, NODE_TILE_SIZE):
        offs_n_new = offs_n + tile_start
        vnode_entries = tl.load(req_to_vnode_entries + cur_vnode_entry_start + offs_n_new)
        vnode_entries = tl.where(offs_n_new < cur_vnode_len, vnode_entries, -1)

        off_mid_o = (
            vnode_entries[:, None] * stride_mid_o0
            + cur_head * stride_mid_o1
            + offs_d[None, :]
        )
        mid_o = tl.load(MidO + off_mid_o, mask=vnode_entries[:, None] >= 0, other=0.0)

        off_mid_lse = vnode_entries * stride_mid_lse0 + cur_head
        mid_lse = tl.load(MidLSE + off_mid_lse, mask=vnode_entries >= 0, other=float("-inf"))[None, :]

        tile_max_lse = tl.max(mid_lse, axis=1)
        new_max_lse = tl.maximum(tile_max_lse, max_lse)

        exp_lse = tl.exp(mid_lse - new_max_lse)
        old_scale = tl.exp(max_lse - new_max_lse)
        sum_exp = sum_exp * old_scale + tl.sum(exp_lse, axis=1)
        acc = acc * old_scale + tl.sum(exp_lse[:, :, None] * mid_o[None, :, :], axis=1)
        max_lse = new_max_lse

    tl.store(
        O + cur_req * stride_o0 + cur_head * stride_o1 + offs_d,
        tl.ravel(acc / sum_exp).to(O.type.element_ty),
    )


def _fasttree_decode_stage2(
    req_to_vnode_entries,
    req_to_vnode_offs,
    req_to_vnode_lens,
    MidO,
    MidLSE,
    O,
    batch_size,
    head_num,
    head_dim,
):
    NODE_TILE_SIZE = 4
    grid = (batch_size, head_num)
    _fwd_fasttree_decoding_stage2[grid](
        req_to_vnode_entries,
        req_to_vnode_offs,
        req_to_vnode_lens,
        MidO,
        MidLSE,
        O,
        MidO.stride(0),
        MidO.stride(1),
        MidLSE.stride(0),
        O.stride(0),
        O.stride(1),
        head_dim,
        NODE_TILE_SIZE,
    )


def fasttree_decode(
    q,
    k_buffer,
    v_buffer,
    o,
    vnode_to_kv_entries,  # Page table (req_to_token) for indirect KV access
    vnode_to_kv_offs,
    vnode_to_kv_lens,
    vnode_to_q_entries,
    vnode_to_q_offs,
    vnode_to_q_lens,
    req_to_vnode_entries,
    req_to_vnode_offs,
    req_to_vnode_lens,
    mid_o,
    mid_lse,
    phase_node_nums,
    phase_node_offsets,
    phase_q_tile_sizes,
    phase_kv_tile_sizes,
    sm_scale,
    logit_cap=-1,
):
    """
    FastTree decode with paged KV cache support.
    
    Unlike the basic version, this uses indirect indexing through
    vnode_to_kv_entries (the page table) to access K/V values.
    
    Args:
        q: Query tensor (batch_size, num_qo_heads, head_dim)
        k_buffer: Key buffer (total_slots, num_kv_heads, head_dim)
        v_buffer: Value buffer (total_slots, num_kv_heads, head_dim)
        o: Output tensor (batch_size, num_qo_heads, head_dim)
        vnode_to_kv_entries: Page table mapping positions to slot indices
        ... (other metadata tensors)
    """
    head_num = q.shape[1]
    head_dim = q.shape[-1]
    kv_head_num = k_buffer.shape[1]

    _fasttree_decode_stage1(
        q,
        k_buffer,
        v_buffer,
        vnode_to_kv_entries,
        vnode_to_kv_offs,
        vnode_to_kv_lens,
        vnode_to_q_entries,
        vnode_to_q_offs,
        vnode_to_q_lens,
        sm_scale,
        mid_o,
        mid_lse,
        phase_node_nums,
        phase_node_offsets,
        head_num,
        head_dim,
        kv_head_num,
        phase_q_tile_sizes,
        phase_kv_tile_sizes,
    )

    batch_size = q.shape[0]
    _fasttree_decode_stage2(
        req_to_vnode_entries,
        req_to_vnode_offs,
        req_to_vnode_lens,
        mid_o,
        mid_lse,
        o,
        batch_size,
        head_num,
        head_dim,
    )
