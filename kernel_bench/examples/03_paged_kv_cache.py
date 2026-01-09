"""
FastTree with Paged KV Cache (SGLang-style)

This example demonstrates how to use FastTree with paged KV cache,
following the same pattern as SGLang's integration. The key insight is
that FastTree uses indirect indexing via a page table (req_to_token),
so there's NO need to gather/copy KV values.

The kernel accesses KV via double indirection:
    1. vnode_to_kv_offs gives offset into req_to_token (page table)
    2. req_to_token[offset] gives the actual slot index in the KV buffer
    3. K[slot_index], V[slot_index] are the actual KV values

Paged KV cache format (SGLang-style):
    kv_cache[layer]: (max_num_pages, 2, page_size, num_kv_heads, head_dim)
    Flattened to: (total_slots, num_kv_heads, head_dim) for kernel access

Usage:
    python 03_paged_kv_cache.py --batch_size 8
    python 03_paged_kv_cache.py --batch_size 16 --page_size 16 --num_qo_heads 32 --num_kv_heads 8
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import argparse
from typing import List, Tuple, Dict
from dataclasses import dataclass
from kv_tree_simple import KVTreeNode

# Import the SGLang-style kernel that supports indirect indexing
from fasttree_sglang_plugin import FastTreeMetadata, fasttree_decode


@dataclass
class PageInfo:
    """Tracks page allocation for a tree node."""
    page_indices: List[int]  # Which pages this node uses
    start_slot: int          # First slot index in the flattened buffer
    num_tokens: int          # Number of tokens in this node


class PagedKVCacheManager:
    """
    Manages paged KV cache with page table for indirect access.
    
    This follows SGLang's approach where:
    - KV data is stored in pages: (max_pages, 2, page_size, num_heads, head_dim)
    - A page table (req_to_token) maps logical positions to physical slots
    - The kernel uses indirect indexing through the page table
    """
    
    def __init__(
        self,
        max_num_pages: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.max_num_pages = max_num_pages
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.total_slots = max_num_pages * page_size
        
        # Paged KV cache: (total_slots, num_kv_heads, head_dim)
        # This is the flattened view that the kernel accesses
        self.k_buffer = torch.zeros(
            (self.total_slots, num_kv_heads, head_dim),
            dtype=dtype, device=device
        )
        self.v_buffer = torch.zeros(
            (self.total_slots, num_kv_heads, head_dim),
            dtype=dtype, device=device
        )
        
        # Page allocation
        self.free_pages: List[int] = list(range(max_num_pages))
        self.node_pages: Dict[int, PageInfo] = {}
        
        # Page table: maps (request_idx, token_position) -> slot_index
        # In SGLang this is req_to_token_pool.req_to_token
        # Shape: (max_requests, max_seq_len) - we'll use a simpler version
        self.max_seq_len = max_num_pages * page_size
        
    def allocate_for_node(self, node_id: int, num_tokens: int) -> PageInfo:
        """Allocate pages for a tree node and return page info."""
        num_pages = (num_tokens + self.page_size - 1) // self.page_size
        
        if len(self.free_pages) < num_pages:
            raise RuntimeError(f"Out of pages: need {num_pages}, have {len(self.free_pages)}")
        
        pages = [self.free_pages.pop(0) for _ in range(num_pages)]
        start_slot = pages[0] * self.page_size
        
        info = PageInfo(
            page_indices=pages,
            start_slot=start_slot,
            num_tokens=num_tokens
        )
        self.node_pages[node_id] = info
        return info
    
    def store_kv(self, node_id: int, k: torch.Tensor, v: torch.Tensor):
        """
        Store K/V data for a node.
        
        Args:
            k, v: (num_tokens, num_kv_heads, head_dim)
        """
        info = self.node_pages.get(node_id)
        if info is None:
            info = self.allocate_for_node(node_id, k.shape[0])
        
        # Store tokens into their page slots
        token_idx = 0
        for page_idx in info.page_indices:
            slot_start = page_idx * self.page_size
            tokens_to_store = min(self.page_size, info.num_tokens - token_idx)
            
            self.k_buffer[slot_start:slot_start + tokens_to_store] = k[token_idx:token_idx + tokens_to_store]
            self.v_buffer[slot_start:slot_start + tokens_to_store] = v[token_idx:token_idx + tokens_to_store]
            token_idx += tokens_to_store
    
    def build_req_to_token(self, tree_info: List[KVTreeNode], batch_size: int) -> torch.Tensor:
        """
        Build the page table (req_to_token) for all requests.
        
        This maps each (request, token_position) to its slot in the KV buffer.
        Following SGLang's pattern where req_to_token[req_idx * stride + pos] = slot_idx.
        """
        # Find max sequence length (path from root to leaf)
        max_seqlen = 0
        for node in tree_info:
            if node.num_children == 0:  # Leaf
                seqlen = 0
                curr = node.id
                while curr != -1:
                    seqlen += tree_info[curr].seqlen
                    curr = tree_info[curr].parent
                max_seqlen = max(max_seqlen, seqlen)
        
        # Build page table: (batch_size, max_seqlen)
        req_to_token = torch.zeros(
            (batch_size, max_seqlen), dtype=torch.int32, device=self.device
        )
        
        # For each request, trace path from root to leaf and fill in slot indices
        for node in tree_info:
            if node.num_children == 0 and node.requests:  # Leaf with requests
                req_id = node.requests[0]
                
                # Collect path from root to this leaf
                path = []
                curr = node.id
                while curr != -1:
                    path.append(curr)
                    curr = tree_info[curr].parent
                path = path[::-1]  # Root to leaf
                
                # Fill in slot indices for each token position
                pos = 0
                for node_id in path:
                    info = self.node_pages[node_id]
                    for i in range(info.num_tokens):
                        # Calculate slot index for this token
                        page_idx = info.page_indices[i // self.page_size]
                        slot_within_page = i % self.page_size
                        slot_idx = page_idx * self.page_size + slot_within_page
                        req_to_token[req_id, pos] = slot_idx
                        pos += 1
        
        return req_to_token


def create_random_tree(batch_size: int, min_tokens: int = 20, max_tokens: int = 100, seed: int = 42):
    """Create a random tree structure for testing."""
    import random
    random.seed(seed)
    
    tree_info = []
    node_id = 0
    
    # Root node
    root = KVTreeNode()
    root.parent = -1
    root.id = node_id
    root.seqlen = random.randint(min_tokens, max_tokens)
    root.num_children = 0
    root.requests = list(range(batch_size))  # All requests go through root
    tree_info.append(root)
    node_id += 1
    
    # Build tree
    potential_parents = [0]
    leaves_created = 0
    
    while leaves_created < batch_size and potential_parents:
        parent_id = potential_parents.pop(0)
        remaining = batch_size - leaves_created
        
        if remaining == 1:
            num_children = 1
        else:
            num_children = min(random.randint(2, 4), remaining)
        
        tree_info[parent_id].num_children = num_children
        
        for _ in range(num_children):
            child = KVTreeNode()
            child.parent = parent_id
            child.id = node_id
            child.seqlen = random.randint(min_tokens, max_tokens)
            child.num_children = 0
            child.requests = []
            tree_info.append(child)
            
            if leaves_created + len(potential_parents) + 1 < batch_size and random.random() > 0.5:
                potential_parents.append(node_id)
            else:
                leaves_created += 1
            
            node_id += 1
            if leaves_created >= batch_size:
                break
    
    # Assign requests to leaves and propagate up
    leaves = [n for n in tree_info if n.num_children == 0]
    for req_id, leaf in enumerate(leaves[:batch_size]):
        leaf.requests = [req_id]
        
        # Propagate request assignment up
        curr = leaf.parent
        while curr != -1:
            if req_id not in tree_info[curr].requests:
                tree_info[curr].requests.append(req_id)
            curr = tree_info[curr].parent
    
    # Sort requests for consistency
    for node in tree_info:
        node.requests.sort()
    
    return tree_info


def prepare_fasttree_metadata_for_paged_cache(
    tree_info: List[KVTreeNode],
    req_to_token: torch.Tensor,
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: str = "cuda",
) -> FastTreeMetadata:
    """
    Prepare FastTree metadata using paged KV cache (page table approach).
    
    The key difference from contiguous KV:
    - vnode_to_kv_offs are offsets into req_to_token (page table)
    - The kernel uses req_to_token[offset] to get actual KV slot indices
    """
    import queue
    
    metadata = FastTreeMetadata(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
    )
    
    # Cost model parameters
    alpha, beta, gamma = metadata.alpha, metadata.beta, metadata.gamma
    kv_group_num = num_qo_heads // num_kv_heads
    phase_q_tile_sizes = list(metadata.TSQs)
    phase_kv_tile_sizes = list(metadata.TSKs)
    phase_kv_split_sizes = [metadata.kv_split_sizes[0], metadata.kv_split_sizes[0]]
    
    def CpadQ(TS, N):
        return TS - ((N - 1) % TS + 1)
    
    def CpadK(TS, N):
        return max(0, TS - N)
    
    def Cmm(nQ, nK):
        phase = 0 if nQ > phase_q_tile_sizes[1] else 1
        TSQ = phase_q_tile_sizes[phase]
        TSK = phase_kv_tile_sizes[phase]
        return alpha * CpadQ(TSQ, nQ) * kv_group_num * nK + beta * CpadK(TSK, nK) * nQ * kv_group_num
    
    def SplitQCost(nQcurr, nQl, lenv, lenl):
        return Cmm(nQcurr - nQl, lenv) + Cmm(nQl, lenl + lenv)
    
    def SplitKCost(nQcurr, nQl, lenl, lenv):
        return Cmm(nQcurr, lenv) + Cmm(nQl, lenl) + gamma * nQl
    
    # BFS traversal to compute node assignments
    node_num = len(tree_info)
    edges = [[] for _ in range(node_num)]
    for i in range(node_num):
        if tree_info[i].parent != -1:
            edges[tree_info[i].parent].append(i)
    
    L = [tree_info[i].seqlen for i in range(node_num)]
    node_assignments = [0] * node_num
    
    # Heuristic: decide split strategy per edge
    que = queue.Queue()
    que.put(0)
    while not que.empty():
        node = que.get()
        nQcurr = len(tree_info[node].requests)
        lenv = L[node]
        
        for child in edges[node]:
            nQl = len(tree_info[child].requests)
            lenl = L[child]
            C0 = SplitKCost(nQcurr, nQl, lenl, lenv)
            C1 = SplitQCost(nQcurr, nQl, lenv, lenl)
            if C0 > C1:
                node_assignments[child] = 1  # Merge with parent
                nQcurr -= nQl
                L[child] = lenl + lenv
            else:
                node_assignments[child] = 0  # Split
            que.put(child)
    
    # Compute which requests each node handles after merging
    node_to_reqs = [[] for _ in range(node_num)]
    que = queue.Queue()
    for i in range(node_num):
        if tree_info[i].num_children == 0:
            que.put(i)
            node_to_reqs[i] = tree_info[i].requests.copy()
    
    virtual_children = [tree_info[n].num_children for n in range(node_num)]
    while not que.empty():
        node = que.get()
        if node_assignments[node] == 0 and node != 0:
            node_to_reqs[tree_info[node].parent] += tree_info[node].requests
        virtual_children[tree_info[node].parent] -= 1
        if tree_info[node].parent >= 0 and virtual_children[tree_info[node].parent] == 0:
            que.put(tree_info[node].parent)
    
    # Build vnode metadata
    # Key: vnode_to_kv_offs are offsets into req_to_token (the page table)
    vnode_to_kv_offs = []
    vnode_to_kv_lens = []
    vnode_to_q_entries = []
    vnode_to_q_offs = []
    vnode_to_q_lens = []
    req_to_vnode_entries = [[] for _ in range(batch_size)]
    
    req_to_token_stride = req_to_token.stride(0)
    
    # Compute token offsets for each node (position in the sequence)
    node_token_offsets = [0] * node_num
    for i in range(1, node_num):
        parent = tree_info[i].parent
        # Token offset = parent's offset + parent's tokens
        node_token_offsets[i] = node_token_offsets[parent] + tree_info[parent].seqlen
    
    for i in range(node_num):
        req_num = len(node_to_reqs[i])
        if req_num == 0:
            continue
        
        # Compute merged KV length
        kv_len = tree_info[i].seqlen
        node = i
        while node_assignments[node] == 1:
            node = tree_info[node].parent
            kv_len += tree_info[node].seqlen
        
        phase = 0 if req_num > phase_q_tile_sizes[1] else 1
        kv_split_size = phase_kv_split_sizes[phase]
        q_split_size = phase_q_tile_sizes[phase]
        
        # Get first request to find the page table row
        first_req = node_to_reqs[i][0]
        token_offset = node_token_offsets[node]  # Use merged node's offset
        
        # KV offset is into the page table (req_to_token)
        kv_offset_start = req_to_token_stride * first_req + token_offset
        
        kv_split_count = (kv_len - 1) // kv_split_size + 1
        q_split_count = (req_num - 1) // q_split_size + 1
        
        for kv_split_id in range(kv_split_count):
            q_offset_start = len(vnode_to_q_entries)
            for req in node_to_reqs[i]:
                vnode_to_q_entries.append(req)
            
            split_kv_off = kv_split_id * kv_split_size
            vnode_kv_len = min(split_kv_off + kv_split_size, kv_len) - split_kv_off
            
            for q_split_id in range(q_split_count):
                split_q_off = q_split_id * q_split_size
                vnode_q_len = min(split_q_off + q_split_size, req_num) - split_q_off
                
                vnode_to_kv_offs.append(kv_offset_start + split_kv_off)
                vnode_to_kv_lens.append(vnode_kv_len)
                vnode_to_q_offs.append(q_offset_start + split_q_off)
                vnode_to_q_lens.append(vnode_q_len)
    
    # Build req_to_vnode mapping
    for i, req in enumerate(vnode_to_q_entries):
        req_to_vnode_entries[req].append(i)
    
    req_to_vnode_offs = []
    req_to_vnode_lens = []
    offset = 0
    for i in range(batch_size):
        req_to_vnode_offs.append(offset)
        offset += len(req_to_vnode_entries[i])
        req_to_vnode_lens.append(len(req_to_vnode_entries[i]))
    
    req_to_vnode_entries_flat = [item for sublist in req_to_vnode_entries for item in sublist]
    
    # Reorder vnodes by phase
    threshold = phase_q_tile_sizes[1]
    above_indices = [i for i, val in enumerate(vnode_to_q_lens) if val > threshold]
    below_indices = [i for i, val in enumerate(vnode_to_q_lens) if val <= threshold]
    new_order = above_indices + below_indices
    
    metadata.phase_node_nums = (len(above_indices), len(below_indices))
    metadata.phase_node_offsets = (0, len(above_indices))
    metadata.phase_q_tile_sizes = tuple(phase_q_tile_sizes)
    metadata.phase_kv_tile_sizes = tuple(phase_kv_tile_sizes)
    
    vnode_to_q_lens = [vnode_to_q_lens[i] for i in new_order]
    vnode_to_q_offs = [vnode_to_q_offs[i] for i in new_order]
    vnode_to_kv_lens = [vnode_to_kv_lens[i] for i in new_order]
    vnode_to_kv_offs = [vnode_to_kv_offs[i] for i in new_order]
    
    # Copy to GPU
    def to_gpu(preallocated, data):
        t = torch.tensor(data, dtype=torch.int32, device="cpu")
        preallocated[:len(data)].copy_(t, non_blocking=True)
    
    to_gpu(metadata.vnode_to_q_entries, vnode_to_q_entries)
    to_gpu(metadata.vnode_to_q_offs, vnode_to_q_offs)
    to_gpu(metadata.vnode_to_q_lens, vnode_to_q_lens)
    to_gpu(metadata.vnode_to_kv_offs, vnode_to_kv_offs)
    to_gpu(metadata.vnode_to_kv_lens, vnode_to_kv_lens)
    to_gpu(metadata.req_to_vnode_entries, req_to_vnode_entries_flat)
    to_gpu(metadata.req_to_vnode_offs, req_to_vnode_offs)
    to_gpu(metadata.req_to_vnode_lens, req_to_vnode_lens)
    
    return metadata


def pytorch_attention_reference(Q, K_list, V_list, sm_scale, num_qo_heads, num_kv_heads):
    """Reference implementation for validation."""
    batch_size = Q.shape[0]
    head_dim = Q.shape[-1]
    gqa_ratio = num_qo_heads // num_kv_heads
    
    O = torch.zeros_like(Q)
    
    for b in range(batch_size):
        K = K_list[b]
        V = V_list[b]
        q = Q[b]
        
        for h in range(num_qo_heads):
            kv_h = h // gqa_ratio
            scores = torch.matmul(q[h].unsqueeze(0), K[:, kv_h].T) * sm_scale
            attn_weights = F.softmax(scores, dim=-1)
            O[b, h] = torch.matmul(attn_weights, V[:, kv_h]).squeeze(0)
    
    return O


def main():
    parser = argparse.ArgumentParser(description='FastTree with Paged KV Cache (SGLang-style)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--page_size', type=int, default=16)
    parser.add_argument('--max_num_pages', type=int, default=1024)
    parser.add_argument('--num_qo_heads', type=int, default=32)
    parser.add_argument('--num_kv_heads', type=int, default=32)
    parser.add_argument('--head_dim', type=int, default=128)
    parser.add_argument('--min_tokens', type=int, default=20)
    parser.add_argument('--max_tokens', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    
    print("=" * 70)
    print("FastTree with Paged KV Cache (SGLang-style, No Copy)")
    print("=" * 70)
    
    device = "cuda"
    dtype = torch.float16
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Page size: {args.page_size}")
    print(f"  Max pages: {args.max_num_pages}")
    print(f"  Heads: Q/O={args.num_qo_heads}, K/V={args.num_kv_heads}")
    print(f"  GQA ratio: {args.num_qo_heads // args.num_kv_heads}")
    
    # Create tree
    print(f"\nCreating tree structure...")
    tree_info = create_random_tree(args.batch_size, args.min_tokens, args.max_tokens, args.seed)
    
    num_nodes = len(tree_info)
    total_tokens = sum(n.seqlen for n in tree_info)
    print(f"  Nodes: {num_nodes}, Tokens: {total_tokens}")
    
    # Initialize paged KV cache
    print(f"\nInitializing paged KV cache...")
    cache = PagedKVCacheManager(
        args.max_num_pages, args.page_size,
        args.num_kv_heads, args.head_dim,
        device, dtype
    )
    
    # Store KV data for each node (for validation we keep references)
    K_node_data = []
    V_node_data = []
    for node in tree_info:
        k = torch.randn(node.seqlen, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)
        v = torch.randn(node.seqlen, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)
        cache.store_kv(node.id, k, v)
        K_node_data.append(k)
        V_node_data.append(v)
    
    # Build page table
    print(f"  Building page table (req_to_token)...")
    req_to_token = cache.build_req_to_token(tree_info, args.batch_size)
    print(f"  Page table shape: {req_to_token.shape}")
    
    # Prepare FastTree metadata
    print(f"\nPreparing FastTree metadata...")
    metadata = prepare_fasttree_metadata_for_paged_cache(
        tree_info, req_to_token, args.batch_size,
        args.num_qo_heads, args.num_kv_heads, args.head_dim, device
    )
    
    # Create query
    Q = torch.randn(args.batch_size, args.num_qo_heads, args.head_dim, device=device, dtype=dtype)
    O = torch.empty_like(Q)
    sm_scale = 1.0 / (args.head_dim ** 0.5)
    
    # Run FastTree with paged KV (no copy!)
    print(f"\n{'='*70}")
    print("Running FastTree with paged KV cache (indirect indexing)...")
    print(f"{'='*70}")
    
    # Warmup
    for _ in range(3):
        fasttree_decode(
            Q, cache.k_buffer, cache.v_buffer, O,
            req_to_token,  # Page table for indirect indexing
            metadata.vnode_to_kv_offs,
            metadata.vnode_to_kv_lens,
            metadata.vnode_to_q_entries,
            metadata.vnode_to_q_offs,
            metadata.vnode_to_q_lens,
            metadata.req_to_vnode_entries,
            metadata.req_to_vnode_offs,
            metadata.req_to_vnode_lens,
            metadata.mid_o,
            metadata.mid_lse,
            metadata.phase_node_nums,
            metadata.phase_node_offsets,
            metadata.phase_q_tile_sizes,
            metadata.phase_kv_tile_sizes,
            sm_scale,
        )
    
    # Benchmark
    torch.cuda.synchronize()
    import time
    start = time.perf_counter()
    num_iters = 20
    for _ in range(num_iters):
        fasttree_decode(
            Q, cache.k_buffer, cache.v_buffer, O,
            req_to_token,
            metadata.vnode_to_kv_offs,
            metadata.vnode_to_kv_lens,
            metadata.vnode_to_q_entries,
            metadata.vnode_to_q_offs,
            metadata.vnode_to_q_lens,
            metadata.req_to_vnode_entries,
            metadata.req_to_vnode_offs,
            metadata.req_to_vnode_lens,
            metadata.mid_o,
            metadata.mid_lse,
            metadata.phase_node_nums,
            metadata.phase_node_offsets,
            metadata.phase_q_tile_sizes,
            metadata.phase_kv_tile_sizes,
            sm_scale,
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / num_iters
    print(f"FastTree decode: {elapsed:.3f} ms")
    
    # Validation
    if args.validate:
        print(f"\n{'='*70}")
        print("Validating...")
        print(f"{'='*70}")
        
        K_per_req, V_per_req = [], []
        for node in tree_info:
            if node.num_children == 0 and node.requests:
                req_id = node.requests[0]
                path = []
                curr = node.id
                while curr != -1:
                    path.append(curr)
                    curr = tree_info[curr].parent
                path = path[::-1]
                
                K_full = torch.cat([K_node_data[n] for n in path], dim=0)
                V_full = torch.cat([V_node_data[n] for n in path], dim=0)
                K_per_req.append(K_full)
                V_per_req.append(V_full)
        
        O_ref = pytorch_attention_reference(Q, K_per_req, V_per_req, sm_scale, args.num_qo_heads, args.num_kv_heads)
        
        max_diff = (O - O_ref).abs().max().item()
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Result: {'✓ PASS' if max_diff < 0.01 else '✗ FAIL'}")
    
    print(f"\n{'='*70}")
    print("Summary: This example demonstrates efficient paged KV cache usage")
    print("with FastTree using indirect indexing (no KV copy needed).")
    print(f"{'='*70}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)
    main()
