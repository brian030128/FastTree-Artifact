"""
Validate FastTree Against PyTorch Scaled Dot Product Attention

Compares FastTree output with PyTorch's standard attention implementation
to verify correctness using tree-structured KV caches.

Supports two tree types:
1. Random imbalanced trees (default) - realistic structure with varying depths
2. Balanced binary trees - symmetric structure for systematic testing

Usage:
    # Random imbalanced tree (default)
    python validate_fasttree.py --batch_size 10
    python validate_fasttree.py --batch_size 15 --min_tokens 50 --max_tokens 200

    # Balanced binary tree (batch_size must be power of 2)
    python validate_fasttree.py --tree_type balanced --batch_size 8
    python validate_fasttree.py --tree_type balanced --batch_size 16 --tokens_per_level 150,100,75,50

    # With GQA
    python validate_fasttree.py --batch_size 12 --num_qo_heads 32 --num_kv_heads 8

    # Different random seeds
    python validate_fasttree.py --batch_size 10 --seed 123
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import argparse
from fasttree import FastTreeParams, fasttree_preparation, fasttree_decode
from kv_tree_simple import KVTreeNode


def create_random_imbalanced_tree(batch_size=8, min_tokens=20, max_tokens=150,
                                   min_children=1, max_children=4, seed=None):
    """
    Create a random imbalanced tree structure.

    This creates a more realistic tree where:
    - Different branches have different depths
    - Nodes can have varying numbers of children (1 to max_children)
    - Leaf nodes can be at different levels
    - Token counts vary randomly between min_tokens and max_tokens

    Args:
        batch_size: Number of leaf nodes (requests)
        min_tokens: Minimum tokens per node
        max_tokens: Maximum tokens per node
        min_children: Minimum children per non-leaf node
        max_children: Maximum children per non-leaf node
        seed: Random seed for reproducibility

    Returns:
        tree_info: List of KVTreeNode objects
    """
    import random

    if seed is not None:
        random.seed(seed)

    tree_info = []
    node_id = 0

    # Create root
    root = KVTreeNode()
    root.parent = -1
    root.id = node_id
    root.seqlen = random.randint(min_tokens, max_tokens)
    root.num_children = 0  # Will be updated
    root.requests = []  # Will be filled later
    tree_info.append(root)
    node_id += 1

    # Build tree top-down, ensuring we get exactly batch_size leaves
    # Start with root as the only "potential parent"
    potential_parents = [0]  # IDs of nodes that can have children
    leaves_needed = batch_size

    while leaves_needed > 0 and potential_parents:
        # Pick a random parent to expand
        parent_idx = random.randint(0, len(potential_parents) - 1)
        parent_id = potential_parents.pop(parent_idx)

        # Decide how many children (at least enough to meet quota)
        if leaves_needed == 1:
            num_children = 1
        else:
            # Random, but ensure we don't create too few or too many
            max_children_now = min(max_children, leaves_needed)
            num_children = random.randint(min_children, max_children_now)

        tree_info[parent_id].num_children = num_children

        # Create children
        for i in range(num_children):
            child = KVTreeNode()
            child.parent = parent_id
            child.id = node_id
            child.seqlen = random.randint(min_tokens, max_tokens)
            child.num_children = 0  # May be updated later
            child.requests = []  # Will be filled later
            tree_info.append(child)

            # Decide if this should be a leaf or can be expanded further
            # If this is the last child and we still need leaves, make some expandable
            if leaves_needed > 1 and random.random() > 0.4:  # 60% chance to expand further
                potential_parents.append(node_id)
            # else it stays as a leaf

            node_id += 1
            leaves_needed -= 1

            if leaves_needed == 0:
                break

    # Find all actual leaf nodes (num_children == 0)
    actual_leaves = [node.id for node in tree_info if node.num_children == 0]

    # Verify we have exactly batch_size leaves
    if len(actual_leaves) != batch_size:
        print(f"Warning: Created {len(actual_leaves)} leaves, expected {batch_size}")
        # Adjust if needed
        if len(actual_leaves) < batch_size:
            # Need to split some leaves into internal nodes with new children
            while len(actual_leaves) < batch_size:
                # Pick a leaf to expand
                leaf_to_expand = actual_leaves[random.randint(0, len(actual_leaves) - 1)]
                actual_leaves.remove(leaf_to_expand)

                # Make it internal with 2 children (creates 1 net new leaf)
                tree_info[leaf_to_expand].num_children = 2
                for i in range(2):
                    child = KVTreeNode()
                    child.parent = leaf_to_expand
                    child.id = node_id
                    child.seqlen = random.randint(min_tokens, max_tokens)
                    child.num_children = 0
                    child.requests = []
                    tree_info.append(child)
                    actual_leaves.append(node_id)
                    node_id += 1
        else:
            # Too many leaves, just take first batch_size
            actual_leaves = actual_leaves[:batch_size]

    # Assign requests to leaf nodes
    for request_id, leaf_id in enumerate(actual_leaves):
        if leaf_id < len(tree_info):
            tree_info[leaf_id].requests = [request_id]

    # Propagate requests up the tree
    for leaf_id in actual_leaves:
        if leaf_id >= len(tree_info):
            continue
        curr_id = leaf_id
        request_id = tree_info[leaf_id].requests[0] if tree_info[leaf_id].requests else None
        if request_id is None:
            continue

        visited = set()
        while curr_id != -1 and curr_id is not None:
            if curr_id in visited:
                break
            visited.add(curr_id)

            parent_id = tree_info[curr_id].parent
            if parent_id != -1 and parent_id < len(tree_info):
                # Add this request to parent
                if request_id not in tree_info[parent_id].requests:
                    tree_info[parent_id].requests.append(request_id)
            curr_id = parent_id

    # Sort requests for consistency
    for node in tree_info:
        node.requests.sort()

    return tree_info


def create_balanced_tree(batch_size=8, tokens_per_level=[100, 80, 60, 40]):
    """
    Create a balanced binary tree structure.

    The tree depth is automatically determined by batch_size:
    - batch_size must be a power of 2
    - depth = log2(batch_size) + 1
    - Number of leaf nodes = batch_size

    Example for batch_size=8 (depth=4):
        Level 0: Root (1 node)
        Level 1: 2 nodes
        Level 2: 4 nodes
        Level 3: 8 leaves

    Example for batch_size=16 (depth=5):
        Level 0: Root (1 node)
        Level 1: 2 nodes
        Level 2: 4 nodes
        Level 3: 8 nodes
        Level 4: 16 leaves

    Args:
        batch_size: Number of leaf nodes (requests). Must be a power of 2.
        tokens_per_level: List of token counts for each level
    """
    import math

    # Verify batch_size is power of 2
    if batch_size & (batch_size - 1) != 0 or batch_size == 0:
        raise ValueError(f"batch_size must be a power of 2, got {batch_size}")

    # Calculate depth: depth = log2(batch_size) + 1
    # e.g., batch_size=8 -> depth=4, batch_size=16 -> depth=5
    depth = int(math.log2(batch_size)) + 1

    # Extend tokens_per_level if needed
    while len(tokens_per_level) < depth:
        tokens_per_level.append(tokens_per_level[-1])

    tree_info = []
    node_id = 0

    # Build tree level by level
    current_level_nodes = []  # Stores node IDs for current level

    # Level 0: Root
    root = KVTreeNode()
    root.parent = -1
    root.id = node_id
    root.seqlen = tokens_per_level[0]
    root.num_children = 2 if depth > 1 else 0
    root.requests = list(range(batch_size))
    tree_info.append(root)
    current_level_nodes = [node_id]
    node_id += 1

    # Build intermediate levels (level 1 to depth-2)
    for level in range(1, depth - 1):
        next_level_nodes = []
        nodes_at_this_level = 2 ** level
        reqs_per_node = batch_size // nodes_at_this_level

        for idx, parent_id in enumerate(current_level_nodes):
            # Each node at previous level has 2 children
            for child_idx in range(2):
                node = KVTreeNode()
                node.parent = parent_id
                node.id = node_id
                node.seqlen = tokens_per_level[level]
                node.num_children = 2  # Not a leaf yet

                # Calculate which requests pass through this node
                node_index = idx * 2 + child_idx
                start_req = node_index * reqs_per_node
                end_req = start_req + reqs_per_node
                node.requests = list(range(start_req, end_req))

                tree_info.append(node)
                next_level_nodes.append(node_id)
                node_id += 1

        current_level_nodes = next_level_nodes

    # Build leaf level (level depth-1)
    if depth > 1:
        request_id = 0
        for parent_id in current_level_nodes:
            for child_idx in range(2):
                node = KVTreeNode()
                node.parent = parent_id
                node.id = node_id
                node.seqlen = tokens_per_level[depth - 1]
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

    if leaf_id is None:
        raise ValueError(f"Could not find leaf node for request {request_id}. "
                        f"Available leaves: {[n.id for n in tree_info if n.num_children == 0]}")

    # Traverse from leaf to root, collecting nodes
    path = []
    node_id = leaf_id
    visited = set()
    while node_id is not None and node_id != -1:
        if node_id in visited:
            raise ValueError(f"Cycle detected in tree at node {node_id}")
        visited.add(node_id)
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
                        help='Number of requests (leaf nodes)')
    parser.add_argument('--tree_type', type=str, default='random', choices=['random', 'balanced'],
                        help='Tree structure type: random (imbalanced) or balanced')

    # For balanced trees
    parser.add_argument('--tokens_per_level', type=str, default='100,80,60,40',
                        help='Comma-separated tokens per level for balanced trees')

    # For random trees
    parser.add_argument('--min_tokens', type=int, default=20,
                        help='Minimum tokens per node for random trees')
    parser.add_argument('--max_tokens', type=int, default=150,
                        help='Maximum tokens per node for random trees')
    parser.add_argument('--min_children', type=int, default=1,
                        help='Minimum children per node for random trees')
    parser.add_argument('--max_children', type=int, default=4,
                        help='Maximum children per node for random trees')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Model configuration
    parser.add_argument('--num_qo_heads', type=int, default=32, help='Number of Q/O heads')
    parser.add_argument('--num_kv_heads', type=int, default=32, help='Number of K/V heads')
    parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')

    # Validation
    parser.add_argument('--rtol', type=float, default=1e-2, help='Relative tolerance')
    parser.add_argument('--atol', type=float, default=1e-3, help='Absolute tolerance')
    args = parser.parse_args()

    print("=" * 70)
    print(f"FastTree Validation Against PyTorch ({args.tree_type.capitalize()} Tree)")
    print("=" * 70)

    # Configuration
    batch_size = args.batch_size
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim = args.head_dim
    device = "cuda"
    dtype = torch.float16

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size} requests")
    print(f"  Tree type: {args.tree_type}")
    print(f"  Q/O heads: {num_qo_heads}")
    print(f"  K/V heads: {num_kv_heads}")
    print(f"  GQA ratio: {num_qo_heads // num_kv_heads}")
    print(f"  Head dim: {head_dim}")

    # Create tree based on type
    if args.tree_type == 'balanced':
        tokens_per_level = list(map(int, args.tokens_per_level.split(',')))
        print(f"  Tokens per level: {tokens_per_level}")
        tree_info = create_balanced_tree(batch_size, tokens_per_level)
    else:  # random
        print(f"  Token range: [{args.min_tokens}, {args.max_tokens}]")
        print(f"  Children range: [{args.min_children}, {args.max_children}]")
        print(f"  Random seed: {args.seed}")
        tree_info = create_random_imbalanced_tree(
            batch_size, args.min_tokens, args.max_tokens,
            args.min_children, args.max_children, args.seed
        )

    # Debug: Print tree structure (first few nodes)
    print("\nDebug: First 5 nodes:")
    for i, node in enumerate(tree_info[:5]):
        print(f"  Node {node.id}: parent={node.parent}, seqlen={node.seqlen}, "
              f"children={node.num_children}, requests={node.requests}")
    print(f"  ... ({len(tree_info)} total nodes)")

    # Analyze tree structure
    def analyze_tree(tree_info):
        """Analyze tree characteristics"""
        num_nodes = len(tree_info)
        num_leaves = sum(1 for node in tree_info if node.num_children == 0)

        # Verify node IDs match list indices
        for i, node in enumerate(tree_info):
            if node.id != i:
                print(f"Warning: Node ID mismatch at index {i}: node.id={node.id}")

        # Calculate depths for each leaf
        leaf_depths = []
        for node in tree_info:
            if node.num_children == 0:
                depth = 0
                curr_id = node.id
                visited = set()
                while curr_id is not None and curr_id != -1 and tree_info[curr_id].parent != -1:
                    if curr_id in visited:
                        print(f"Warning: Cycle detected at node {curr_id}")
                        break
                    visited.add(curr_id)
                    depth += 1
                    curr_id = tree_info[curr_id].parent
                leaf_depths.append(depth)

        if not leaf_depths:
            leaf_depths = [0]

        # Calculate path lengths (tokens) for each request
        path_lengths = []
        for req_id in range(num_leaves):
            # Find leaf for this request
            leaf_id = None
            for node in tree_info:
                if node.num_children == 0 and req_id in node.requests:
                    leaf_id = node.id
                    break

            if leaf_id is None:
                print(f"Warning: Could not find leaf for request {req_id}")
                continue

            # Sum tokens along path
            total_tokens = 0
            curr_id = leaf_id
            while curr_id is not None and curr_id != -1:
                total_tokens += tree_info[curr_id].seqlen
                curr_id = tree_info[curr_id].parent
            path_lengths.append(total_tokens)

        if not path_lengths:
            # Fallback if no paths found
            path_lengths = [0]

        return {
            'num_nodes': num_nodes,
            'num_leaves': num_leaves,
            'min_depth': min(leaf_depths) if leaf_depths else 0,
            'max_depth': max(leaf_depths) if leaf_depths else 0,
            'avg_depth': sum(leaf_depths) / len(leaf_depths) if leaf_depths else 0,
            'min_path_len': min(path_lengths) if path_lengths else 0,
            'max_path_len': max(path_lengths) if path_lengths else 0,
            'avg_path_len': sum(path_lengths) / len(path_lengths) if path_lengths else 0,
        }

    stats = analyze_tree(tree_info)

    print(f"\nTree Statistics:")
    print(f"  Total nodes: {stats['num_nodes']}")
    print(f"  Leaf nodes: {stats['num_leaves']}")
    print(f"  Tree depth: min={stats['min_depth']}, max={stats['max_depth']}, avg={stats['avg_depth']:.1f}")
    print(f"  Path length (tokens): min={stats['min_path_len']}, max={stats['max_path_len']}, avg={stats['avg_path_len']:.1f}")

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
