"""
FastTree with Paged KV Cache

This example demonstrates how to use FastTree with paged KV cache,
which is the typical format used in production LLM inference systems
like SGLang, vLLM, etc.

Paged KV cache format:
    kv_cache[layer]: (max_num_pages, 2, page_size, num_kv_heads, head_dim)
    - Dimension 0: page index
    - Dimension 1: K=0, V=1
    - Dimension 2: token position within page
    - Dimension 3: KV head index
    - Dimension 4: head dimension

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
from fasttree import FastTreeParams, fasttree_preparation, fasttree_decode
from kv_tree_simple import KVTreeNode


class PagedKVCache:
    """
    Manages paged KV cache for tree-structured attention.
    
    Each tree node's KV data is stored in one or more pages.
    Page table tracks which pages belong to which nodes.
    """
    
    def __init__(
        self,
        max_num_pages: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        layer_num: int = 1,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.max_num_pages = max_num_pages
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.device = device
        self.dtype = dtype
        
        # Paged KV cache: (max_num_pages, 2, page_size, num_kv_heads, head_dim)
        # 2 = [K, V]
        self.kv_cache_at_layer: List[torch.Tensor] = [
            torch.zeros(
                (max_num_pages, 2, page_size, num_kv_heads, head_dim),
                dtype=dtype,
                device=device,
            )
            for _ in range(layer_num)
        ]
        
        # Page allocation tracking
        self.free_pages: List[int] = list(range(max_num_pages))
        self.allocated_pages: Dict[int, List[int]] = {}  # node_id -> list of page indices
        
        # For each node, track (start_page_idx, num_pages, tokens_in_last_page)
        self.node_page_info: Dict[int, Tuple[int, int, int]] = {}
        
    def allocate_pages_for_node(self, node_id: int, num_tokens: int) -> List[int]:
        """Allocate pages for a tree node."""
        num_pages_needed = (num_tokens + self.page_size - 1) // self.page_size
        
        if len(self.free_pages) < num_pages_needed:
            raise RuntimeError(f"Not enough free pages. Need {num_pages_needed}, have {len(self.free_pages)}")
        
        pages = [self.free_pages.pop(0) for _ in range(num_pages_needed)]
        self.allocated_pages[node_id] = pages
        
        tokens_in_last_page = num_tokens % self.page_size
        if tokens_in_last_page == 0:
            tokens_in_last_page = self.page_size
        self.node_page_info[node_id] = (pages[0], len(pages), tokens_in_last_page)
        
        return pages
    
    def store_kv(self, node_id: int, k_data: torch.Tensor, v_data: torch.Tensor, layer: int = 0):
        """
        Store K and V data for a node into paged cache.
        
        Args:
            node_id: Tree node ID
            k_data: Key tensor of shape (seqlen, num_kv_heads, head_dim)
            v_data: Value tensor of shape (seqlen, num_kv_heads, head_dim)
            layer: Layer index
        """
        seqlen = k_data.shape[0]
        pages = self.allocated_pages.get(node_id)
        
        if pages is None:
            pages = self.allocate_pages_for_node(node_id, seqlen)
        
        # Store tokens into pages
        token_idx = 0
        for page_idx in pages:
            tokens_in_page = min(self.page_size, seqlen - token_idx)
            self.kv_cache_at_layer[layer][page_idx, 0, :tokens_in_page] = k_data[token_idx:token_idx + tokens_in_page]
            self.kv_cache_at_layer[layer][page_idx, 1, :tokens_in_page] = v_data[token_idx:token_idx + tokens_in_page]
            token_idx += tokens_in_page
    
    def get_kv_indices_for_node(self, node_id: int) -> List[int]:
        """
        Get flat token indices for a node to use with FastTree.
        
        Returns indices into a flattened view of the paged cache.
        """
        pages = self.allocated_pages.get(node_id, [])
        if not pages:
            return []
        
        indices = []
        start_page, num_pages, tokens_in_last = self.node_page_info[node_id]
        
        for i, page_idx in enumerate(pages):
            if i < num_pages - 1:
                # Full page
                for t in range(self.page_size):
                    indices.append(page_idx * self.page_size + t)
            else:
                # Last page (may be partial)
                for t in range(tokens_in_last):
                    indices.append(page_idx * self.page_size + t)
        
        return indices
    
    def get_flattened_kv(self, layer: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get flattened K and V tensors for FastTree kernel.
        
        Returns:
            K: (total_tokens, num_kv_heads, head_dim)
            V: (total_tokens, num_kv_heads, head_dim)
        """
        cache = self.kv_cache_at_layer[layer]
        # Reshape from (pages, 2, page_size, heads, dim) to (pages * page_size, 2, heads, dim)
        total_slots = self.max_num_pages * self.page_size
        K = cache[:, 0].reshape(total_slots, self.num_kv_heads, self.head_dim)
        V = cache[:, 1].reshape(total_slots, self.num_kv_heads, self.head_dim)
        return K, V


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
    root.requests = []
    tree_info.append(root)
    node_id += 1
    
    # Build tree to get batch_size leaves
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
            
            # Decide if this becomes an internal node or leaf
            if leaves_created + len(potential_parents) + 1 < batch_size and random.random() > 0.5:
                potential_parents.append(node_id)
            else:
                leaves_created += 1
            
            node_id += 1
            
            if leaves_created >= batch_size:
                break
    
    # Assign requests to leaf nodes
    leaves = [n for n in tree_info if n.num_children == 0]
    for req_id, leaf in enumerate(leaves[:batch_size]):
        leaf.requests = [req_id]
        
        # Propagate up
        curr = leaf.parent
        while curr != -1:
            if req_id not in tree_info[curr].requests:
                tree_info[curr].requests.append(req_id)
            curr = tree_info[curr].parent
    
    return tree_info


def prepare_paged_kv_for_tree(
    tree_info: List[KVTreeNode],
    paged_cache: PagedKVCache,
    layer: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Prepare paged KV cache for a tree structure.
    
    Allocates pages and fills with random data for each node.
    Returns flattened K, V tensors and KV_ptrs for FastTree.
    """
    device = paged_cache.device
    dtype = paged_cache.dtype
    
    # Allocate pages and store random KV data for each node
    for node in tree_info:
        seqlen = node.seqlen
        k_data = torch.randn(seqlen, paged_cache.num_kv_heads, paged_cache.head_dim, 
                            device=device, dtype=dtype)
        v_data = torch.randn(seqlen, paged_cache.num_kv_heads, paged_cache.head_dim,
                            device=device, dtype=dtype)
        paged_cache.store_kv(node.id, k_data, v_data, layer)
    
    # Build KV_ptrs - cumulative token counts per node
    KV_ptrs = [0]
    for node in tree_info:
        KV_ptrs.append(KV_ptrs[-1] + node.seqlen)
    
    # Get flattened KV for the kernel
    K, V = paged_cache.get_flattened_kv(layer)
    
    return K, V, KV_ptrs


def build_kv_indices_for_fasttree(
    tree_info: List[KVTreeNode],
    paged_cache: PagedKVCache,
) -> torch.Tensor:
    """
    Build the KV index mapping for FastTree to access paged cache.
    
    FastTree needs indices into the flattened KV buffer.
    With paged cache, we need to map node tokens to their page locations.
    """
    all_indices = []
    for node in tree_info:
        indices = paged_cache.get_kv_indices_for_node(node.id)
        all_indices.extend(indices)
    
    return torch.tensor(all_indices, dtype=torch.int32, device=paged_cache.device)


def pytorch_attention_reference(Q, K_list, V_list, sm_scale, num_qo_heads, num_kv_heads):
    """Reference implementation using PyTorch for validation."""
    batch_size = Q.shape[0]
    head_dim = Q.shape[-1]
    gqa_ratio = num_qo_heads // num_kv_heads
    
    O = torch.zeros_like(Q)
    
    for b in range(batch_size):
        K = K_list[b]  # (seqlen, num_kv_heads, head_dim)
        V = V_list[b]
        q = Q[b]  # (num_qo_heads, head_dim)
        
        seqlen = K.shape[0]
        
        for h in range(num_qo_heads):
            kv_h = h // gqa_ratio
            q_h = q[h]  # (head_dim,)
            k_h = K[:, kv_h]  # (seqlen, head_dim)
            v_h = V[:, kv_h]  # (seqlen, head_dim)
            
            # Attention scores
            scores = torch.matmul(q_h.unsqueeze(0), k_h.T) * sm_scale  # (1, seqlen)
            attn_weights = F.softmax(scores, dim=-1)  # (1, seqlen)
            out = torch.matmul(attn_weights, v_h)  # (1, head_dim)
            O[b, h] = out.squeeze(0)
    
    return O


def main():
    parser = argparse.ArgumentParser(description='FastTree with Paged KV Cache Example')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of requests (leaf nodes)')
    parser.add_argument('--page_size', type=int, default=16, help='Tokens per page')
    parser.add_argument('--max_num_pages', type=int, default=1024, help='Maximum number of pages')
    parser.add_argument('--num_qo_heads', type=int, default=32, help='Number of Q/O heads')
    parser.add_argument('--num_kv_heads', type=int, default=32, help='Number of K/V heads')
    parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
    parser.add_argument('--min_tokens', type=int, default=20, help='Min tokens per node')
    parser.add_argument('--max_tokens', type=int, default=100, help='Max tokens per node')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--validate', action='store_true', help='Validate against PyTorch')
    args = parser.parse_args()
    
    print("=" * 70)
    print("FastTree with Paged KV Cache Example")
    print("=" * 70)
    
    device = "cuda"
    dtype = torch.float16
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Page size: {args.page_size}")
    print(f"  Max pages: {args.max_num_pages}")
    print(f"  Q/O heads: {args.num_qo_heads}")
    print(f"  K/V heads: {args.num_kv_heads}")
    print(f"  Head dim: {args.head_dim}")
    print(f"  GQA ratio: {args.num_qo_heads // args.num_kv_heads}")
    
    # ============================================================
    # Create Tree Structure
    # ============================================================
    print(f"\nCreating random tree with {args.batch_size} requests...")
    tree_info = create_random_tree(
        args.batch_size, args.min_tokens, args.max_tokens, args.seed
    )
    
    num_nodes = len(tree_info)
    total_tokens = sum(n.seqlen for n in tree_info)
    num_leaves = sum(1 for n in tree_info if n.num_children == 0)
    
    print(f"  Total nodes: {num_nodes}")
    print(f"  Leaf nodes: {num_leaves}")
    print(f"  Total tokens: {total_tokens}")
    
    # ============================================================
    # Initialize Paged KV Cache
    # ============================================================
    print(f"\nInitializing paged KV cache...")
    paged_cache = PagedKVCache(
        max_num_pages=args.max_num_pages,
        page_size=args.page_size,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        layer_num=1,
        device=device,
        dtype=dtype,
    )
    
    pages_needed = (total_tokens + args.page_size - 1) // args.page_size
    print(f"  Pages needed: ~{pages_needed}")
    print(f"  Cache size: {paged_cache.kv_cache_at_layer[0].numel() * 2 / 1e6:.1f} MB")
    
    # ============================================================
    # Prepare KV Data in Paged Cache
    # ============================================================
    print(f"\nPreparing KV data in paged cache...")
    
    # Store random K/V data for each tree node
    K_node_list = []  # For validation
    V_node_list = []
    
    for node in tree_info:
        seqlen = node.seqlen
        k_data = torch.randn(seqlen, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)
        v_data = torch.randn(seqlen, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)
        paged_cache.store_kv(node.id, k_data, v_data, layer=0)
        K_node_list.append(k_data)
        V_node_list.append(v_data)
    
    # Get flattened KV tensors
    K_flat, V_flat = paged_cache.get_flattened_kv(layer=0)
    print(f"  Flattened K shape: {K_flat.shape}")
    print(f"  Flattened V shape: {V_flat.shape}")
    
    # Build KV_ptrs using paged indices
    # FastTree expects contiguous indices, but with paged cache we need to remap
    # Solution: Build a contiguous K/V buffer with proper indexing
    
    # For paged cache, we need to gather tokens in order for each node
    kv_entries = []
    KV_ptrs = [0]
    for node in tree_info:
        indices = paged_cache.get_kv_indices_for_node(node.id)
        kv_entries.extend(indices)
        KV_ptrs.append(len(kv_entries))
    
    kv_entries_tensor = torch.tensor(kv_entries, dtype=torch.int64, device=device)
    
    # Gather K/V using the paged indices
    K_tree = K_flat[kv_entries_tensor]  # (total_tokens, num_kv_heads, head_dim)
    V_tree = V_flat[kv_entries_tensor]
    
    print(f"  K_tree shape: {K_tree.shape}")
    print(f"  V_tree shape: {V_tree.shape}")
    
    # ============================================================
    # Create Query Tensor
    # ============================================================
    Q = torch.randn(args.batch_size, args.num_qo_heads, args.head_dim, device=device, dtype=dtype)
    
    # ============================================================
    # Run FastTree
    # ============================================================
    print(f"\n{'='*70}")
    print("Running FastTree...")
    print(f"{'='*70}")
    
    params = FastTreeParams()
    params.set_kv_group_num(args.num_qo_heads // args.num_kv_heads)
    
    metadata, node_assignments = fasttree_preparation(
        tree_info, KV_ptrs, args.batch_size,
        args.num_qo_heads, args.num_kv_heads, args.head_dim,
        [1024, 128], [132, 528], [132, 132], params
    )
    
    O_fasttree = torch.empty(args.batch_size, args.num_qo_heads, args.head_dim, 
                             device=device, dtype=dtype)
    sm_scale = 1.0 / (args.head_dim ** 0.5)
    
    # Warmup
    for _ in range(3):
        fasttree_decode(
            Q, K_tree, V_tree, O_fasttree, *metadata,
            params.TSQs, params.TSKs, sm_scale
        )
    
    # Benchmark
    torch.cuda.synchronize()
    import time
    start = time.perf_counter()
    num_iters = 20
    for _ in range(num_iters):
        fasttree_decode(
            Q, K_tree, V_tree, O_fasttree, *metadata,
            params.TSQs, params.TSKs, sm_scale
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / num_iters
    
    print(f"FastTree decode: {elapsed:.3f} ms")
    
    # ============================================================
    # Validation (Optional)
    # ============================================================
    if args.validate:
        print(f"\n{'='*70}")
        print("Validating against PyTorch reference...")
        print(f"{'='*70}")
        
        # Build full K/V for each request
        K_per_request = []
        V_per_request = []
        
        for req_id in range(args.batch_size):
            # Find path from leaf to root
            leaf_id = None
            for node in tree_info:
                if node.num_children == 0 and req_id in node.requests:
                    leaf_id = node.id
                    break
            
            if leaf_id is None:
                raise RuntimeError(f"Could not find leaf for request {req_id}")
            
            # Collect K/V along path
            path = []
            curr = leaf_id
            while curr != -1:
                path.append(curr)
                curr = tree_info[curr].parent
            path = path[::-1]  # Root to leaf order
            
            K_full = torch.cat([K_node_list[n] for n in path], dim=0)
            V_full = torch.cat([V_node_list[n] for n in path], dim=0)
            K_per_request.append(K_full)
            V_per_request.append(V_full)
        
        # Run reference
        O_pytorch = pytorch_attention_reference(
            Q, K_per_request, V_per_request, sm_scale,
            args.num_qo_heads, args.num_kv_heads
        )
        
        # Compare
        max_abs_diff = (O_fasttree - O_pytorch).abs().max().item()
        mean_abs_diff = (O_fasttree - O_pytorch).abs().mean().item()
        
        print(f"  Max absolute difference: {max_abs_diff:.6f}")
        print(f"  Mean absolute difference: {mean_abs_diff:.6f}")
        
        rtol, atol = 1e-2, 1e-3
        is_close = torch.allclose(O_fasttree, O_pytorch, rtol=rtol, atol=atol)
        
        if is_close:
            print(f"  Result: ✓ PASS - Outputs match within tolerance")
        else:
            print(f"  Result: ✗ FAIL - Outputs differ beyond tolerance")
            print(f"  Sample (request 0, head 0):")
            print(f"    FastTree: {O_fasttree[0, 0, :5]}")
            print(f"    PyTorch:  {O_pytorch[0, 0, :5]}")
    
    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"  Tree: {num_nodes} nodes, {total_tokens} tokens, {args.batch_size} requests")
    print(f"  Paged cache: {args.max_num_pages} pages × {args.page_size} tokens/page")
    print(f"  FastTree decode latency: {elapsed:.3f} ms")
    print(f"\nThis example demonstrates using paged KV cache with FastTree,")
    print(f"which is the standard format used in production LLM serving systems.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        sys.exit(1)
    
    main()
