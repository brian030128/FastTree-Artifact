"""
Minimal FastTree Example

This is the simplest possible FastTree usage example.
Creates a small tree, runs preparation, and computes attention.

Requirements:
    pip install torch triton

Usage:
    python 01_minimal_example.py

Expected Output:
    Tree created with 3 nodes
    Preparation completed
    Decode completed
    Output shape: torch.Size([2, 32, 128])
"""

import sys
import os

# Add parent directory to path to import fasttree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from fasttree import FastTreeParams, fasttree_preparation, fasttree_decode
from kv_tree_simple import KVTreeNode


def create_simple_tree():
    """
    Create a simple 2-level tree structure:

            Root (128 tokens)
           /                \
    Child 1 (64 tokens)   Child 2 (64 tokens)
       [Request 0]         [Request 1]

    This creates a tree where two requests share a common prefix (root)
    but have different continuations (child1 and child2).
    """
    # Root node - shared by both requests
    root = KVTreeNode()
    root.parent = -1        # Root has no parent
    root.id = 0
    root.seqlen = 128       # 128 tokens in root
    root.num_children = 2   # Has 2 children
    root.requests = []      # Will be filled automatically

    # Child 1 - used by request 0
    child1 = KVTreeNode()
    child1.parent = 0       # Parent is root
    child1.id = 1
    child1.seqlen = 64      # 64 tokens in child1
    child1.num_children = 0 # Leaf node
    child1.requests = [0]   # Request 0 uses this path

    # Child 2 - used by request 1
    child2 = KVTreeNode()
    child2.parent = 0       # Parent is root
    child2.id = 2
    child2.seqlen = 64      # 64 tokens in child2
    child2.num_children = 0 # Leaf node
    child2.requests = [1]   # Request 1 uses this path

    # Assign requests to root (both requests use root)
    root.requests = [0, 1]

    return [root, child1, child2]


def prepare_kv_data(tree_info, num_kv_heads, head_dim, device, dtype):
    """
    Prepare K and V tensors for the tree structure.

    For each node, create random K and V tensors representing the tokens.
    Also build KV_ptrs array containing cumulative token counts.
    """
    node_num = len(tree_info)
    K_tree_list = []
    V_tree_list = []
    KV_ptrs = [0]  # Start with 0

    # Create K and V tensors for each node
    for n in range(node_num):
        seqlen = tree_info[n].seqlen
        # Create random K tensor: [seqlen, num_kv_heads, head_dim]
        K_node = torch.randn(seqlen, num_kv_heads, head_dim, device=device, dtype=dtype)
        V_node = torch.randn(seqlen, num_kv_heads, head_dim, device=device, dtype=dtype)
        K_tree_list.append(K_node)
        V_tree_list.append(V_node)
        # Update cumulative token count
        KV_ptrs.append(KV_ptrs[-1] + seqlen)

    # Concatenate all K and V tensors
    K_tree_tensor = torch.cat(K_tree_list, dim=0)  # [total_tokens, num_kv_heads, head_dim]
    V_tree_tensor = torch.cat(V_tree_list, dim=0)  # [total_tokens, num_kv_heads, head_dim]

    return K_tree_tensor, V_tree_tensor, KV_ptrs


def main():
    print("=" * 60)
    print("FastTree Minimal Example")
    print("=" * 60)

    # ============================================================
    # Step 1: Configuration
    # ============================================================
    num_qo_heads = 32      # Number of query/output heads
    num_kv_heads = 32      # Number of key/value heads (MHA, not GQA)
    head_dim = 128         # Dimension per head
    dtype = torch.float16  # Use FP16 for efficiency
    device = "cuda"        # FastTree requires CUDA

    print(f"\nConfiguration:")
    print(f"  Heads (Q/O): {num_qo_heads}")
    print(f"  Heads (K/V): {num_kv_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Dtype: {dtype}")

    # ============================================================
    # Step 2: Create Tree Structure
    # ============================================================
    tree_info = create_simple_tree()
    batch_size = 2  # 2 requests (one for each leaf)
    print(f"\nTree created with {len(tree_info)} nodes")
    print(f"Batch size: {batch_size} requests")

    # ============================================================
    # Step 3: Prepare K/V Data
    # ============================================================
    K_tree_tensor, V_tree_tensor, KV_ptrs = prepare_kv_data(
        tree_info, num_kv_heads, head_dim, device, dtype
    )
    total_tokens = KV_ptrs[-1]
    print(f"Total tokens in tree: {total_tokens}")
    print(f"KV_ptrs: {KV_ptrs}")

    # Create query tensor (one query per request)
    Q = torch.randn(batch_size, num_qo_heads, head_dim, device=device, dtype=dtype)

    # Create output tensor (will be filled by fasttree_decode)
    O = torch.empty(batch_size, num_qo_heads, head_dim, device=device, dtype=dtype)

    # ============================================================
    # Step 4: Configure FastTree Parameters
    # ============================================================
    params = FastTreeParams()
    # Using default parameters:
    # - alpha=0.66, beta=0.33, gamma=0.1 (cost model weights)
    # - TSQs=[64, 16] (query tile sizes)
    # - TSKs=[32, 128] (KV tile sizes)
    # - kv_group_num=1 (for MHA)
    print(f"\nFastTree Parameters:")
    print(f"  alpha={params.alpha}, beta={params.beta}, gamma={params.gamma}")
    print(f"  TSQs={params.TSQs}, TSKs={params.TSKs}")

    # ============================================================
    # Step 5: Preparation (analyze tree and create metadata)
    # ============================================================
    print(f"\nRunning preparation...")
    metadata, node_assignments = fasttree_preparation(
        tree_info=tree_info,
        KV_ptrs=KV_ptrs,
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        KV_SPLIT_SIZES=[1024, 128],    # Thresholds for KV splitting
        para_threshs1=[132, 528],       # First-level parallelism thresholds
        para_threshs2=[132, 132],       # Second-level parallelism thresholds
        params=params,
    )
    print("Preparation completed")
    print(f"Node assignments (0=split K, 1=split Q): {node_assignments}")

    # ============================================================
    # Step 6: Decode (compute attention)
    # ============================================================
    print(f"\nRunning decode...")
    sm_scale = 1.0 / (head_dim ** 0.5)  # Softmax scaling factor

    fasttree_decode(
        q=Q,
        k_buffer=K_tree_tensor,
        v_buffer=V_tree_tensor,
        o=O,
        *metadata,  # Unpack 13 metadata tensors from preparation
        phase_q_tile_sizes=params.TSQs,
        phase_kv_tile_sizes=params.TSKs,
        sm_scale=sm_scale,
    )
    print("Decode completed")

    # ============================================================
    # Step 7: Validate Output
    # ============================================================
    print(f"\nOutput Validation:")
    print(f"  Shape: {O.shape}")
    print(f"  Dtype: {O.dtype}")
    print(f"  Device: {O.device}")
    print(f"  Value range: [{O.min().item():.3f}, {O.max().item():.3f}]")
    print(f"  Contains NaN: {torch.isnan(O).any().item()}")
    print(f"  Contains Inf: {torch.isinf(O).any().item()}")

    # ============================================================
    # Success!
    # ============================================================
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Try 02_basic_usage.py to load trees from files")
    print("  - Try 03_custom_tree.py to create different tree structures")
    print("  - See docs/API_REFERENCE.md for detailed API documentation")


if __name__ == "__main__":
    # Check for CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. FastTree requires a CUDA GPU.")
        sys.exit(1)

    main()
