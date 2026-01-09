"""
Basic FastTree Usage

Complete example showing typical usage pattern:
- Load tree from file
- Configure parameters
- Run preparation and decode
- Measure performance
- Validate results

Usage:
    python 02_basic_usage.py --tree_file ../data/tree_example.txt
    python 02_basic_usage.py --tree_file ../data/tree_example.txt --num_qo_heads 32 --num_kv_heads 8

Arguments:
    --tree_file: Path to tree file (required)
    --num_qo_heads: Number of query/output heads (default: 32)
    --num_kv_heads: Number of key/value heads (default: 32)
    --head_dim: Head dimension (default: 128)
    --alpha: Cost model alpha parameter (default: 0.66)
    --beta: Cost model beta parameter (default: 0.33)
    --gamma: Cost model gamma parameter (default: 0.1)
    --profile: Enable profiling (default: False)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
import time
from fasttree import FastTreeParams, fasttree_preparation, fasttree_decode
from kv_tree_simple import retrive_from_file
from flash_attn_wrap import qkv_preparation


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='FastTree Basic Usage Example')

    parser.add_argument('--tree_file', type=str, required=True,
                        help='Path to tree file')
    parser.add_argument('--num_qo_heads', type=int, default=32,
                        help='Number of query/output heads')
    parser.add_argument('--num_kv_heads', type=int, default=32,
                        help='Number of key/value heads')
    parser.add_argument('--head_dim', type=int, default=128,
                        help='Head dimension (must be in {16, 32, 64, 128, 256})')
    parser.add_argument('--alpha', type=float, default=0.66,
                        help='Cost model alpha (Q-padding weight)')
    parser.add_argument('--beta', type=float, default=0.33,
                        help='Cost model beta (K-padding weight)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Cost model gamma (reduction weight)')
    parser.add_argument('--q_tile_sizes', type=str, default='64,16',
                        help='Query tile sizes per phase (comma-separated)')
    parser.add_argument('--kv_tile_sizes', type=str, default='32,128',
                        help='KV tile sizes per phase (comma-separated)')
    parser.add_argument('--profile', action='store_true',
                        help='Enable detailed profiling')

    return parser.parse_args()


def print_tree_stats(tree_info):
    """Print statistics about the tree structure"""
    num_nodes = len(tree_info)
    total_tokens = sum(node.seqlen for node in tree_info)

    # Count leaf nodes
    num_leaves = sum(1 for node in tree_info if node.num_children == 0)

    # Find max depth
    def get_depth(node_id):
        if tree_info[node_id].parent == -1:
            return 1
        return 1 + get_depth(tree_info[node_id].parent)

    max_depth = max(get_depth(node.id) for node in tree_info)

    # Get root
    root_id = next(i for i, node in enumerate(tree_info) if node.parent == -1)

    print("\nTree Statistics:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Leaf nodes (requests): {num_leaves}")
    print(f"  Max depth: {max_depth}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Root node: {root_id} (seqlen={tree_info[root_id].seqlen})")


def benchmark_function(func, warmup=5, repeat=20):
    """
    Benchmark a function with warmup and multiple repetitions.

    Returns:
        median_ms: Median latency in milliseconds
        p20_ms: 20th percentile latency
        p80_ms: 80th percentile latency
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    times.sort()
    median = times[len(times) // 2]
    p20 = times[int(len(times) * 0.2)]
    p80 = times[int(len(times) * 0.8)]

    return median, p20, p80


def validate_output(output, name="Output"):
    """Validate output tensor and print statistics"""
    print(f"\n{name} Validation:")
    print(f"  Shape: {output.shape}")
    print(f"  Dtype: {output.dtype}")
    print(f"  Device: {output.device}")
    print(f"  Value range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")

    # Check for NaN or Inf
    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()

    if has_nan:
        print(f"  WARNING: Output contains NaN values!")
    if has_inf:
        print(f"  WARNING: Output contains Inf values!")

    return not (has_nan or has_inf)


def main():
    args = parse_args()

    print("=" * 70)
    print("FastTree Basic Usage Example")
    print("=" * 70)

    # ============================================================
    # Configuration
    # ============================================================
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim = args.head_dim
    dtype = torch.float16
    device = "cuda"

    # Parse tile sizes
    q_tile_sizes = list(map(int, args.q_tile_sizes.split(',')))
    kv_tile_sizes = list(map(int, args.kv_tile_sizes.split(',')))

    print(f"\nConfiguration:")
    print(f"  Tree file: {args.tree_file}")
    print(f"  Heads (Q/O): {num_qo_heads}")
    print(f"  Heads (K/V): {num_kv_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  GQA ratio: {num_qo_heads // num_kv_heads}")
    print(f"  Dtype: {dtype}")

    # ============================================================
    # Load Tree
    # ============================================================
    print(f"\nLoading tree from {args.tree_file}...")
    try:
        tree_info = retrive_from_file(args.tree_file)
        print(f"Tree loaded successfully!")
        print_tree_stats(tree_info)
    except FileNotFoundError:
        print(f"ERROR: Tree file not found: {args.tree_file}")
        print("Please provide a valid tree file path.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load tree: {e}")
        sys.exit(1)

    # ============================================================
    # Prepare QKV Data
    # ============================================================
    print(f"\nPreparing Q/K/V tensors...")
    Q, K_cache, V_cache, cache_seqlens, K_tree_tensor, V_tree_tensor, KV_ptrs = \
        qkv_preparation(tree_info, num_qo_heads, num_kv_heads, head_dim, device, dtype)

    batch_size = Q.shape[0]
    total_tokens = KV_ptrs[-1]

    print(f"Data prepared:")
    print(f"  Batch size: {batch_size}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Q shape: {Q.shape}")
    print(f"  K tree shape: {K_tree_tensor.shape}")
    print(f"  V tree shape: {V_tree_tensor.shape}")

    # ============================================================
    # Configure FastTree Parameters
    # ============================================================
    params = FastTreeParams()
    params.set_values(args.alpha, args.beta, args.gamma)
    params.set_q_tile_sizes(q_tile_sizes)
    params.set_kv_tile_sizes(kv_tile_sizes)
    params.set_kv_group_num(num_qo_heads // num_kv_heads)

    print(f"\nFastTree Parameters:")
    print(f"  Cost model: alpha={params.alpha}, beta={params.beta}, gamma={params.gamma}")
    print(f"  Query tile sizes: {params.TSQs}")
    print(f"  KV tile sizes: {params.TSKs}")
    print(f"  KV group num (GQA): {params.kv_group_num}")

    # ============================================================
    # Preparation
    # ============================================================
    print(f"\n{'='*70}")
    print("Running Preparation...")
    print(f"{'='*70}")

    prep_start = time.perf_counter()
    metadata, node_assignments = fasttree_preparation(
        tree_info=tree_info,
        KV_ptrs=KV_ptrs,
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        KV_SPLIT_SIZES=[1024, 128],
        para_threshs1=[132, 528],
        para_threshs2=[132, 132],
        params=params,
    )
    torch.cuda.synchronize()
    prep_time = (time.perf_counter() - prep_start) * 1000

    print(f"Preparation completed in {prep_time:.2f} ms")
    if args.profile:
        print(f"Node assignments: {node_assignments}")

    # ============================================================
    # Decode
    # ============================================================
    print(f"\n{'='*70}")
    print("Running Decode...")
    print(f"{'='*70}")

    # Create output tensor
    O = torch.empty(batch_size, num_qo_heads, head_dim, device=device, dtype=dtype)
    sm_scale = 1.0 / (head_dim ** 0.5)

    # Define decode function
    def decode_func():
        fasttree_decode(
            q=Q,
            k_buffer=K_tree_tensor,
            v_buffer=V_tree_tensor,
            o=O,
            *metadata,
            phase_q_tile_sizes=params.TSQs,
            phase_kv_tile_sizes=params.TSKs,
            sm_scale=sm_scale,
        )

    # Benchmark
    if args.profile:
        # Detailed benchmarking
        median_ms, p20_ms, p80_ms = benchmark_function(decode_func, warmup=10, repeat=50)
        print(f"Decode performance:")
        print(f"  Median: {median_ms:.3f} ms")
        print(f"  P20: {p20_ms:.3f} ms")
        print(f"  P80: {p80_ms:.3f} ms")
    else:
        # Single run with timing
        torch.cuda.synchronize()
        start = time.perf_counter()
        decode_func()
        torch.cuda.synchronize()
        decode_time = (time.perf_counter() - start) * 1000
        print(f"Decode completed in {decode_time:.2f} ms")

    # ============================================================
    # Validate Output
    # ============================================================
    is_valid = validate_output(O, "FastTree Output")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Tree: {len(tree_info)} nodes, {total_tokens} tokens, {batch_size} requests")
    print(f"Preparation: {prep_time:.2f} ms")
    if args.profile:
        print(f"Decode: {median_ms:.3f} ms (median)")
    else:
        print(f"Decode: {decode_time:.2f} ms")
    print(f"Output valid: {is_valid}")

    if is_valid:
        print("\n" + "=" * 70)
        print("Example completed successfully!")
        print("=" * 70)
        print("\nNext steps:")
        print("  - Try different tree files to see performance variation")
        print("  - Experiment with --alpha, --beta, --gamma parameters")
        print("  - Use --profile for detailed performance analysis")
        print("  - See docs/CONFIGURATION_GUIDE.md for parameter tuning")
    else:
        print("\nWARNING: Output validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    # Check for CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. FastTree requires a CUDA GPU.")
        sys.exit(1)

    main()
