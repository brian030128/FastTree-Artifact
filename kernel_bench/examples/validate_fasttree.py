"""
Validate FastTree Against PyTorch Scaled Dot Product Attention

Compares FastTree output with PyTorch's standard attention implementation
to verify correctness using a 4-level binary tree structure.

Usage:
    python validate_fasttree.py
    python validate_fasttree.py --batch_size 8 --tokens_per_level 100,80,60,40
    python validate_fasttree.py --batch_size 16 --tokens_per_level 150,100,75,50
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import argparse
from fasttree import FastTreeParams, fasttree_preparation, fasttree_decode
from kv_tree_simple import KVTreeNode


def create_simple_tree(batch_size=8, tokens_per_level=[100, 80, 60, 40]):
    """
    Create a balanced 4-level tree structure.

    Structure:
        Level 0: Root (tokens_per_level[0])
                    |
        Level 1: 2 nodes (tokens_per_level[1] each)
                /        \
        Level 2: 4 nodes (tokens_per_level[2] each, 2 children of each L1 node)
              /  |  |  \
        Level 3: 8 leaves (tokens_per_level[3] each, 2 children of each L2 node)

    This creates a binary tree with depth=4 and 8 leaf nodes (requests).

    Args:
        batch_size: Number of leaf nodes (requests). Must be a power of 2.
        tokens_per_level: List of token counts for each level [L0, L1, L2, L3]
    """
    import math

    # Verify batch_size is power of 2
    depth = int(math.log2(batch_size)) + 1
    if 2 ** (depth - 1) != batch_size:
        raise ValueError(f"batch_size must be a power of 2, got {batch_size}")

    # Ensure we have enough levels
    while len(tokens_per_level) < depth:
        tokens_per_level.append(tokens_per_level[-1])

    tree_info = []
    node_id = 0

    # Level 0: Root
    root = KVTreeNode()
    root.parent = -1
    root.id = node_id
    root.seqlen = tokens_per_level[0]
    root.num_children = 2
    root.requests = list(range(batch_size))
    tree_info.append(root)
    node_id += 1

    # Level 1: 2 nodes
    level1_nodes = []
    for i in range(2):
        node = KVTreeNode()
        node.parent = 0
        node.id = node_id
        node.seqlen = tokens_per_level[1]
        node.num_children = 2
        # Assign half the requests to each L1 node
        start_req = i * (batch_size // 2)
        end_req = (i + 1) * (batch_size // 2)
        node.requests = list(range(start_req, end_req))
        tree_info.append(node)
        level1_nodes.append(node_id)
        node_id += 1

    # Level 2: 4 nodes (2 children for each L1 node)
    level2_nodes = []
    for l1_idx, l1_id in enumerate(level1_nodes):
        for i in range(2):
            node = KVTreeNode()
            node.parent = l1_id
            node.id = node_id
            node.seqlen = tokens_per_level[2]
            node.num_children = 2
            # Each L2 node gets quarter of requests
            reqs_per_l2 = batch_size // 4
            start_req = l1_idx * (batch_size // 2) + i * reqs_per_l2
            end_req = start_req + reqs_per_l2
            node.requests = list(range(start_req, end_req))
            tree_info.append(node)
            level2_nodes.append(node_id)
            node_id += 1

    # Level 3: 8 leaf nodes (2 children for each L2 node)
    request_id = 0
    for l2_idx, l2_id in enumerate(level2_nodes):
        for i in range(2):
            node = KVTreeNode()
            node.parent = l2_id
            node.id = node_id
            node.seqlen = tokens_per_level[3]
            node.num_children = 0  # Leaf
            node.requests = [request_id]
            tree_info.append(node)
            node_id += 1
            request_id += 1

    return tree_info


def prepare_kv_data_for_tree(tree_info, num_kv_heads, head_dim, device, dtype):
    """Prepare K/V tensors for tree structure"""
    K_tree_list = []
    V_tree_list = []
    KV_ptrs = [0]

    for node in tree_info:
        seqlen = node.seqlen
        K_node = torch.randn(seqlen, num_kv_heads, head_dim, device=device, dtype=dtype)
        V_node = torch.randn(seqlen, num_kv_heads, head_dim, device=device, dtype=dtype)
        K_tree_list.append(K_node)
        V_tree_list.append(V_node)
        KV_ptrs.append(KV_ptrs[-1] + seqlen)

    K_tree_tensor = torch.cat(K_tree_list, dim=0)
    V_tree_tensor = torch.cat(V_tree_list, dim=0)

    return K_tree_tensor, V_tree_tensor, KV_ptrs, K_tree_list, V_tree_list


def build_full_kv_for_request(tree_info, K_tree_list, V_tree_list, request_id):
    """Build full K/V sequence for a specific request by following tree path"""
    # Find leaf node for this request
    leaf_id = None
    for node in tree_info:
        if node.num_children == 0 and request_id in node.requests:
            leaf_id = node.id
            break

    # Traverse from leaf to root, collecting nodes
    path = []
    node_id = leaf_id
    while node_id != -1:
        path.append(node_id)
        node_id = tree_info[node_id].parent
    path.reverse()  # Root to leaf order

    # Concatenate K/V along path
    K_list = [K_tree_list[nid] for nid in path]
    V_list = [V_tree_list[nid] for nid in path]

    K_full = torch.cat(K_list, dim=0)  # [total_seqlen, num_kv_heads, head_dim]
    V_full = torch.cat(V_list, dim=0)

    return K_full, V_full


def pytorch_attention(Q, K, V, sm_scale, num_qo_heads, num_kv_heads):
    """
    Compute attention using PyTorch's scaled_dot_product_attention.

    Args:
        Q: [batch, num_qo_heads, head_dim]
        K: [batch, seqlen, num_kv_heads, head_dim]
        V: [batch, seqlen, num_kv_heads, head_dim]

    Returns:
        O: [batch, num_qo_heads, head_dim]
    """
    batch_size = Q.shape[0]
    head_dim = Q.shape[2]
    seqlen = K.shape[1]

    # Reshape for attention: need [batch, heads, 1, head_dim] for Q
    #                         and [batch, heads, seqlen, head_dim] for K, V
    Q_reshaped = Q.unsqueeze(2)  # [batch, num_qo_heads, 1, head_dim]

    # Handle GQA: repeat K/V heads if needed
    if num_qo_heads != num_kv_heads:
        kv_group_num = num_qo_heads // num_kv_heads
        # K: [batch, seqlen, num_kv_heads, head_dim] -> [batch, num_kv_heads, seqlen, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        # Repeat each KV head kv_group_num times
        K = K.repeat_interleave(kv_group_num, dim=1)  # [batch, num_qo_heads, seqlen, head_dim]
        V = V.repeat_interleave(kv_group_num, dim=1)
    else:
        # K: [batch, seqlen, num_kv_heads, head_dim] -> [batch, num_kv_heads, seqlen, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

    # Compute attention
    # PyTorch's scaled_dot_product_attention expects:
    # query: [batch, heads, query_len, head_dim]
    # key: [batch, heads, seqlen, head_dim]
    # value: [batch, heads, seqlen, head_dim]
    O = F.scaled_dot_product_attention(
        Q_reshaped, K, V, attn_mask=None, dropout_p=0.0, scale=sm_scale
    )  # [batch, num_qo_heads, 1, head_dim]

    # Remove query dimension
    O = O.squeeze(2)  # [batch, num_qo_heads, head_dim]

    return O


def compare_outputs(fasttree_out, pytorch_out, rtol=1e-2, atol=1e-3):
    """Compare FastTree and PyTorch outputs"""
    # Compute differences
    abs_diff = torch.abs(fasttree_out - pytorch_out)
    rel_diff = abs_diff / (torch.abs(pytorch_out) + 1e-8)

    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    # Check if within tolerance
    close = torch.allclose(fasttree_out, pytorch_out, rtol=rtol, atol=atol)

    return {
        'close': close,
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'max_rel_diff': max_rel_diff,
        'mean_rel_diff': mean_rel_diff,
    }


def main():
    parser = argparse.ArgumentParser(description='Validate FastTree against PyTorch')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of requests (must be power of 2 for 4-level tree)')
    parser.add_argument('--tokens_per_level', type=str, default='100,80,60,40',
                        help='Comma-separated tokens per level [L0,L1,L2,L3]')
    parser.add_argument('--num_qo_heads', type=int, default=32, help='Number of Q/O heads')
    parser.add_argument('--num_kv_heads', type=int, default=32, help='Number of K/V heads')
    parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
    parser.add_argument('--rtol', type=float, default=1e-2, help='Relative tolerance')
    parser.add_argument('--atol', type=float, default=1e-3, help='Absolute tolerance')
    args = parser.parse_args()

    print("=" * 70)
    print("FastTree Validation Against PyTorch (4-Level Tree)")
    print("=" * 70)

    # Configuration
    batch_size = args.batch_size
    tokens_per_level = list(map(int, args.tokens_per_level.split(',')))
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim = args.head_dim
    device = "cuda"
    dtype = torch.float16

    # Calculate total sequence length (sum of all levels)
    total_seqlen = sum(tokens_per_level)

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size} requests")
    print(f"  Tree structure: 4-level binary tree")
    print(f"    Level 0 (root): {tokens_per_level[0]} tokens")
    print(f"    Level 1: {tokens_per_level[1]} tokens (2 nodes)")
    print(f"    Level 2: {tokens_per_level[2]} tokens (4 nodes)")
    print(f"    Level 3 (leaves): {tokens_per_level[3]} tokens (8 nodes)")
    print(f"  Total sequence length per request: {total_seqlen}")
    print(f"  Q/O heads: {num_qo_heads}")
    print(f"  K/V heads: {num_kv_heads}")
    print(f"  GQA ratio: {num_qo_heads // num_kv_heads}")
    print(f"  Head dim: {head_dim}")

    # Create tree
    tree_info = create_simple_tree(batch_size, tokens_per_level)
    print(f"\nTree: {len(tree_info)} nodes, {batch_size} requests")
    print(f"  Level 0: 1 node (root)")
    print(f"  Level 1: 2 nodes")
    print(f"  Level 2: 4 nodes")
    print(f"  Level 3: 8 leaf nodes")

    # Prepare data
    K_tree, V_tree, KV_ptrs, K_tree_list, V_tree_list = prepare_kv_data_for_tree(
        tree_info, num_kv_heads, head_dim, device, dtype
    )

    # Create queries
    Q = torch.randn(batch_size, num_qo_heads, head_dim, device=device, dtype=dtype)

    # ============================================================
    # FastTree Attention
    # ============================================================
    print(f"\n{'='*70}")
    print("Computing attention with FastTree...")
    print(f"{'='*70}")

    params = FastTreeParams()
    params.set_kv_group_num(num_qo_heads // num_kv_heads)

    metadata, _ = fasttree_preparation(
        tree_info, KV_ptrs, batch_size, num_qo_heads, num_kv_heads, head_dim,
        [1024, 128], [132, 528], [132, 132], params
    )

    O_fasttree = torch.empty(batch_size, num_qo_heads, head_dim, device=device, dtype=dtype)
    sm_scale = 1.0 / (head_dim ** 0.5)

    fasttree_decode(
        Q, K_tree, V_tree, O_fasttree, *metadata,
        params.TSQs, params.TSKs, sm_scale
    )

    print("FastTree output computed")

    # ============================================================
    # PyTorch Attention (per request)
    # ============================================================
    print(f"\n{'='*70}")
    print("Computing attention with PyTorch...")
    print(f"{'='*70}")

    O_pytorch = torch.empty(batch_size, num_qo_heads, head_dim, device=device, dtype=dtype)

    for req_id in range(batch_size):
        # Build full K/V for this request
        K_full, V_full = build_full_kv_for_request(
            tree_info, K_tree_list, V_tree_list, req_id
        )

        # Reshape for PyTorch attention
        # K_full: [seqlen, num_kv_heads, head_dim] -> [1, seqlen, num_kv_heads, head_dim]
        K_full = K_full.unsqueeze(0)
        V_full = V_full.unsqueeze(0)

        # Get query for this request
        Q_single = Q[req_id:req_id+1]  # [1, num_qo_heads, head_dim]

        # Compute attention
        O_single = pytorch_attention(
            Q_single, K_full, V_full, sm_scale, num_qo_heads, num_kv_heads
        )

        O_pytorch[req_id] = O_single[0]

    print("PyTorch output computed")

    # ============================================================
    # Compare Outputs
    # ============================================================
    print(f"\n{'='*70}")
    print("Comparing Outputs...")
    print(f"{'='*70}")

    results = compare_outputs(O_fasttree, O_pytorch, rtol=args.rtol, atol=args.atol)

    print(f"\nNumerical Comparison:")
    print(f"  Max absolute difference: {results['max_abs_diff']:.6f}")
    print(f"  Mean absolute difference: {results['mean_abs_diff']:.6f}")
    print(f"  Max relative difference: {results['max_rel_diff']:.6f}")
    print(f"  Mean relative difference: {results['mean_rel_diff']:.6f}")

    print(f"\nTolerance:")
    print(f"  Relative tolerance (rtol): {args.rtol}")
    print(f"  Absolute tolerance (atol): {args.atol}")

    print(f"\nResult: ", end='')
    if results['close']:
        print("✓ PASS - Outputs match within tolerance!")
    else:
        print("✗ FAIL - Outputs differ beyond tolerance")
        print("\nNote: Small differences are expected due to:")
        print("  - FP16 precision limitations")
        print("  - Different computation order in FastTree vs PyTorch")
        print("  - Triton kernel optimizations")

    # Show sample values
    print(f"\nSample Values (Request 0, Head 0, first 5 dims):")
    print(f"  FastTree: {O_fasttree[0, 0, :5]}")
    print(f"  PyTorch:  {O_pytorch[0, 0, :5]}")
    print(f"  Diff:     {(O_fasttree - O_pytorch)[0, 0, :5]}")

    print("\n" + "=" * 70)
    if results['close']:
        print("Validation PASSED!")
    else:
        print("Validation FAILED - outputs differ")
    print("=" * 70)

    return 0 if results['close'] else 1


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        sys.exit(1)

    sys.exit(main())
