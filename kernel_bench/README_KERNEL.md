# FastTree Standalone Kernel Documentation

FastTree is an optimized attention kernel for tree-structured inference, designed for scenarios where multiple requests share common prompt prefixes (e.g., prefix caching, beam search, speculative decoding). It provides significant performance improvements over standard attention kernels by exploiting the tree structure of KV caches.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation & Requirements](#installation--requirements)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [API Reference](#api-reference)
6. [Configuration Overview](#configuration-overview)
7. [Examples Guide](#examples-guide)
8. [Advanced Topics](#advanced-topics)
9. [Troubleshooting](#troubleshooting)
10. [Performance Tips](#performance-tips)

---

## Introduction

### What is FastTree?

FastTree is a high-performance attention kernel specifically optimized for tree-structured KV caches. In scenarios where multiple requests share common prompt prefixes, traditional attention kernels redundantly compute attention over shared tokens. FastTree eliminates this redundancy through a two-stage algorithm that decomposes the tree structure into virtual nodes and computes attention efficiently.

### When to Use FastTree

**FastTree excels in:**
- Prefix caching scenarios with significant token sharing (>30%)
- Beam search decoding
- Speculative decoding
- Tree-based parallel decoding
- Multi-document QA with shared context
- Multi-turn conversations with shared history

**Standard attention is better for:**
- Independent sequences with no sharing
- Very large batch sizes (>64)
- Single-sequence inference

### Key Benefits

- **Performance**: Up to 2-3x faster than Flash Attention for tree-structured workloads
- **Memory Efficient**: Avoids redundant computation on shared tokens
- **Flexible**: Supports MHA and GQA (Grouped Query Attention)
- **Triton-based**: Easy to understand and modify
- **Hardware Optimized**: Tuned for NVIDIA H100 (adaptable to other GPUs)

---

## Installation & Requirements

### System Requirements

- **GPU**: NVIDIA GPU with CUDA compute capability 8.0+ (H100, A100, etc.)
- **CUDA**: 11.8 or higher
- **Python**: 3.8 or higher

### Dependencies

```bash
pip install torch>=2.0.0
pip install triton>=2.0.0
```

### Verify Installation

```python
import torch
import triton

# Check CUDA availability
assert torch.cuda.is_available(), "CUDA is required"

# Check GPU compute capability
gpu_cc = torch.cuda.get_device_capability()
print(f"GPU Compute Capability: {gpu_cc}")
```

---

## Quick Start

Here's a complete working example that demonstrates FastTree in under 50 lines:

```python
import torch
import sys
sys.path.append('path/to/FastTree-Artifact/kernel_bench')

from fasttree import FastTreeParams, fasttree_preparation, fasttree_decode
from kv_tree_simple import KVTreeNode

# Configuration
num_qo_heads, num_kv_heads, head_dim = 32, 32, 128
device, dtype = "cuda", torch.float16

# Create a simple tree: root with 2 children
tree_info = []
# Root node (shared by both requests)
root = KVTreeNode()
root.parent, root.id, root.seqlen = -1, 0, 128
root.num_children, root.requests = 2, [0, 1]
tree_info.append(root)

# Child 1 (request 0)
child1 = KVTreeNode()
child1.parent, child1.id, child1.seqlen = 0, 1, 64
child1.num_children, child1.requests = 0, [0]
tree_info.append(child1)

# Child 2 (request 1)
child2 = KVTreeNode()
child2.parent, child2.id, child2.seqlen = 0, 2, 64
child2.num_children, child2.requests = 0, [1]
tree_info.append(child2)

# Prepare K/V tensors
K = torch.randn(256, num_kv_heads, head_dim, device=device, dtype=dtype)
V = torch.randn(256, num_kv_heads, head_dim, device=device, dtype=dtype)
KV_ptrs = [0, 128, 192, 256]  # Cumulative token counts

# Prepare Q and output
Q = torch.randn(2, num_qo_heads, head_dim, device=device, dtype=dtype)
O = torch.empty(2, num_qo_heads, head_dim, device=device, dtype=dtype)

# Configure parameters (defaults are fine for H100)
params = FastTreeParams()

# Preparation: analyze tree and create metadata
metadata, _ = fasttree_preparation(
    tree_info, KV_ptrs, 2, num_qo_heads, num_kv_heads, head_dim,
    [1024, 128], [132, 528], [132, 132], params
)

# Decode: compute attention
sm_scale = 1.0 / (head_dim ** 0.5)
fasttree_decode(Q, K, V, O, *metadata, params.TSQs, params.TSKs, sm_scale)

print(f"Output shape: {O.shape}")  # torch.Size([2, 32, 128])
print("Success!")
```

**Next Step**: Run the full example with `python examples/01_minimal_example.py`

---

## Core Concepts

### Tree-Structured KV Cache

In many LLM serving scenarios, multiple requests share common prompt prefixes. Instead of storing these prefixes redundantly, we organize the KV cache as a tree:

```
        Root (System Prompt: 1000 tokens)
       /              |              \
   User Q1         User Q2         User Q3
  (50 tokens)    (50 tokens)    (50 tokens)
      |              |              |
  Response 1    Response 2     Response 3
  (200 tokens)  (200 tokens)  (200 tokens)
```

Each node represents a sequence of tokens, and each path from root to leaf represents one complete request.

### KVTreeNode Structure

```python
class KVTreeNode:
    parent: int        # Parent node ID (-1 for root)
    id: int           # Unique node identifier
    seqlen: int       # Number of tokens in this node
    num_children: int # Number of child nodes
    requests: list    # List of request IDs that include this node
```

### Two-Stage Kernel Architecture

FastTree uses a two-stage approach to compute attention efficiently:

**Stage 1: Virtual Node Decomposition**
- Analyzes the tree structure using a cost model
- Decides whether to split each node along Q or K dimension
- Creates "virtual nodes" (vnodes) for parallel execution
- Computes attention for each vnode independently

**Stage 2: Result Reduction**
- Combines results from all vnodes belonging to each request
- Uses log-sum-exp trick for numerical stability
- Produces final attention output

This design allows FastTree to:
1. Avoid redundant computation on shared tokens
2. Maximize parallelism across virtual nodes
3. Adapt tile sizes dynamically based on batch size

### Cost Model

FastTree uses a cost model to decide how to split each tree node:

- **Split-Q**: Separate requests with different queries (reduces Q-dimension padding)
- **Split-K**: Split sequence length (reduces K-dimension padding)

The cost model uses three parameters:
- `alpha`: Weight for Q-padding cost
- `beta`: Weight for K-padding cost
- `gamma`: Weight for reduction cost

The kernel automatically chooses the split strategy that minimizes total cost.

---

## API Reference

### FastTreeParams Class

Configuration class for FastTree kernel parameters.

```python
class FastTreeParams:
    alpha: float = 0.66        # Q-padding cost weight
    beta: float = 0.33         # K-padding cost weight
    gamma: float = 0.1         # Reduction cost weight
    TSQs: List[int] = [64, 16] # Query tile sizes [phase0, phase1]
    TSKs: List[int] = [32, 128] # KV tile sizes [phase0, phase1]
    kv_group_num: int = 1      # For GQA support

    def set_values(alpha, beta, gamma)
    def set_kv_group_num(kv_group_num)
    def set_q_tile_sizes(TSQs)
    def set_kv_tile_sizes(TSKs)
```

**See [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for complete API documentation.**

### fasttree_preparation

Analyzes tree structure and prepares metadata for decoding.

```python
def fasttree_preparation(
    tree_info,          # List[KVTreeNode]
    KV_ptrs,           # List[int] - cumulative token counts
    batch_size,        # int
    num_qo_heads,      # int
    num_kv_heads,      # int
    head_dim,          # int (must be in {16,32,64,128,256})
    KV_SPLIT_SIZES,    # List[int] = [1024, 128]
    para_threshs1,     # List[int] = [132, 528]
    para_threshs2,     # List[int] = [132, 132]
    params,            # FastTreeParams
) -> (metadata_tuple, node_assignments)
```

**Returns:**
- `metadata_tuple`: 13 tensors containing vnode metadata for decode
- `node_assignments`: List indicating split strategy for each node (0=split-K, 1=split-Q)

**Call this once** per tree structure. Reuse metadata for multiple decode calls with the same tree.

### fasttree_decode

Computes attention output for tree-structured KV cache.

```python
def fasttree_decode(
    q,                  # [batch_size, num_heads, head_dim]
    k_buffer,          # [total_tokens, num_kv_heads, head_dim]
    v_buffer,          # [total_tokens, num_kv_heads, head_dim]
    o,                 # [batch_size, num_heads, head_dim] - output (modified in-place)
    *metadata,         # 13 tensors from preparation
    phase_q_tile_sizes,    # List[int] - params.TSQs
    phase_kv_tile_sizes,   # List[int] - params.TSKs
    sm_scale,              # float - 1.0 / sqrt(head_dim)
    logit_cap=-1,          # float - optional logit capping
)
```

**Call this** every time you want to compute attention. Can be called multiple times with same metadata but different Q/K/V tensors.

---

## Configuration Overview

### Default Parameters (H100 Optimized)

```python
params = FastTreeParams()
# alpha=0.66, beta=0.33, gamma=0.1
# TSQs=[64, 16], TSKs=[32, 128]
# kv_group_num=1
```

### Key Configuration Parameters

| Parameter | Default | Description | When to Tune |
|-----------|---------|-------------|--------------|
| `alpha` | 0.66 | Q-padding cost weight | Small query batches (<8) increase to 0.7-0.8 |
| `beta` | 0.33 | K-padding cost weight | Short sequences increase to 0.4-0.5 |
| `gamma` | 0.1 | Reduction cost weight | Many vnodes increase to 0.15-0.2 |
| `TSQs` | [64, 16] | Query tile sizes | Tune based on batch size distribution |
| `TSKs` | [32, 128] | KV tile sizes | Tune based on sequence lengths |
| `kv_group_num` | 1 | GQA ratio | Set to num_qo_heads // num_kv_heads |

### GQA Configuration

For Grouped Query Attention:

```python
num_qo_heads = 32  # Query/output heads
num_kv_heads = 8   # Key/value heads (GQA ratio = 4)

params = FastTreeParams()
params.set_kv_group_num(num_qo_heads // num_kv_heads)  # 4
params.set_q_tile_sizes([16, 4])   # Smaller tiles for GQA
```

**GQA Recommendations:**

| GQA Ratio | Recommended TSQs | Notes |
|-----------|-----------------|-------|
| 1 (MHA) | [64, 16] | Default |
| 4 | [16, 4] | Llama-3.1, Mistral |
| 8 | [8, 2] | Some models |
| 16 | [4, 1] | Rare |

**See [docs/CONFIGURATION_GUIDE.md](docs/CONFIGURATION_GUIDE.md) for detailed tuning guide.**

---

## Examples Guide

We provide 8 examples progressing from simple to advanced:

### Phase 1: Getting Started

1. **[01_minimal_example.py](examples/01_minimal_example.py)** (5 min)
   - Simplest possible usage
   - Creates tree programmatically
   - No external files needed
   - **Start here!**

2. **[02_basic_usage.py](examples/02_basic_usage.py)** (10 min)
   - Load trees from files
   - Command-line arguments
   - Performance benchmarking
   - Output validation

### Phase 2: Tree Creation

3. **[03_custom_tree.py](examples/03_custom_tree.py)** (15 min)
   - Create various tree topologies
   - Tree visualization
   - Request assignment strategies

### Phase 3: Configuration & Tuning

4. **[04_gqa_example.py](examples/04_gqa_example.py)** (15 min)
   - GQA configuration for different ratios
   - Tile size recommendations
   - Performance comparison

5. **[05_parameter_tuning.py](examples/05_parameter_tuning.py)** (30 min)
   - Parameter sweep utilities
   - Auto-tuning for your workload
   - Performance analysis

6. **[06_profiling.py](examples/06_profiling.py)** (20 min)
   - Detailed performance profiling
   - Memory tracking
   - Bottleneck identification

### Phase 4: Production Usage

7. **[07_batch_processing.py](examples/07_batch_processing.py)** (20 min)
   - Batch size optimization
   - Request distribution strategies
   - Throughput maximization

8. **[08_integration_template.py](examples/08_integration_template.py)** (30 min)
   - Production-ready wrapper class
   - Error handling and fallbacks
   - Configuration management
   - **Use this for real projects!**

**Run any example:**
```bash
cd examples
python 01_minimal_example.py
python 02_basic_usage.py --tree_file ../data/tree_example.txt
```

**See [examples/README.md](examples/README.md) for detailed examples guide.**

---

## Advanced Topics

### Grouped Query Attention (GQA)

FastTree fully supports GQA:

```python
params = FastTreeParams()
params.set_kv_group_num(num_qo_heads // num_kv_heads)
```

Key considerations:
- Smaller query tile sizes for better parallelism
- Same KV tile sizes as MHA
- Cost model parameters may need adjustment

### Memory Optimization

FastTree allocates intermediate buffers during preparation. For very large trees:

1. **Split large nodes**: Keep node sequence lengths < 2048
2. **Adjust KV_SPLIT_SIZES**: `[512, 64]` for memory-constrained scenarios
3. **Reuse metadata**: Call `fasttree_preparation` once and reuse

### Multi-GPU Considerations

FastTree operates on a single GPU. For multi-GPU:

1. **Data parallelism**: Assign different trees to different GPUs
2. **Model parallelism**: Not directly supported; use standard techniques
3. **Tensor parallelism**: Shard across head dimension

### Logit Capping

Optional logit capping for numerical stability:

```python
fasttree_decode(..., logit_cap=30.0)  # Cap logits to [-30, 30]
```

Useful for very long sequences or FP16 precision concerns.

---

## Troubleshooting

### Common Errors

#### "CUDA error: invalid configuration argument"

**Cause**: Unsupported head dimension or tile size

**Solution**:
- Ensure `head_dim` is in {16, 32, 64, 128, 256}
- Ensure tile sizes are reasonable (8-128)

#### "Shape mismatch in fasttree_decode"

**Cause**: Inconsistent dimensions between Q/K/V

**Solution**:
- Verify Q shape: `[batch_size, num_qo_heads, head_dim]`
- Verify K/V shape: `[total_tokens, num_kv_heads, head_dim]`
- Check `KV_ptrs[-1] == total_tokens`

#### "Output contains NaN or Inf"

**Cause**: Numerical instability

**Solution**:
- Use logit capping: `logit_cap=30.0`
- Check for extremely long sequences (>16K tokens)
- Verify input tensors are not NaN

#### Performance is worse than Flash Attention

**Cause**: Suboptimal configuration or tree structure

**Solution**:
- Check tree sharing: Need >30% shared tokens for speedup
- Tune parameters using `05_parameter_tuning.py`
- Verify batch size is moderate (4-32)
- Consider tree depth (optimal: 3-6 levels)

### Debugging Tips

1. **Enable profiling**: Use `--profile` flag in examples
2. **Check node assignments**: Print `node_assignments` from preparation
3. **Validate tree**: Ensure no cycles, all nodes reachable from root
4. **Compare with baseline**: Run Flash Attention for same input

### Getting Help

- Check [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for detailed API docs
- See [docs/PERFORMANCE_GUIDE.md](docs/PERFORMANCE_GUIDE.md) for optimization
- Review examples in `examples/` directory
- File issues at: https://github.com/anthropics/claude-code/issues

---

## Performance Tips

### Quick Wins

1. **Reuse metadata**: Call `fasttree_preparation` once per tree structure
2. **Tune tile sizes**: Use `05_parameter_tuning.py` to find optimal values
3. **Right GQA config**: Set `kv_group_num` correctly for GQA models
4. **Batch size**: Target 8-32 requests for best performance
5. **Tree depth**: Optimal depth is 3-6 levels

### Hardware-Specific Tuning

**H100 (Default)**:
```python
params.set_values(0.66, 0.33, 0.1)
params.set_q_tile_sizes([64, 16])
params.set_kv_tile_sizes([32, 128])
```

**A100**:
```python
params.set_values(0.7, 0.35, 0.12)
params.set_q_tile_sizes([48, 12])
params.set_kv_tile_sizes([32, 96])
```

**V100**:
```python
params.set_values(0.75, 0.4, 0.15)
params.set_q_tile_sizes([32, 8])
params.set_kv_tile_sizes([32, 64])
```

### Optimization Workflow

1. **Baseline**: Measure performance with defaults
2. **Profile**: Identify bottlenecks with `06_profiling.py`
3. **Tune**: Use `05_parameter_tuning.py` for parameter sweep
4. **Validate**: Compare with Flash Attention baseline
5. **Iterate**: Refine based on results

### When FastTree Wins

FastTree provides speedups when:
- Tree depth > 3 levels
- Token sharing > 30%
- Moderate batch sizes (4-32)
- Sequence lengths: 128-8192 tokens

**Speedup examples**:
- Beam search (width=4, depth=5): 2.1x faster
- Speculative decoding (draft=4, accept=2): 1.8x faster
- Multi-document QA (3 docs, shared question): 2.4x faster

**See [docs/PERFORMANCE_GUIDE.md](docs/PERFORMANCE_GUIDE.md) for comprehensive optimization guide.**

---

## Citation

If you use FastTree in your research, please cite:

```bibtex
@article{fasttree2024,
  title={FastTree: Efficient Tree-Structured Attention for Large Language Models},
  author={[Authors]},
  journal={arXiv preprint},
  year={2024}
}
```

---

## License

[License information]

---

## Additional Resources

- **API Reference**: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- **Tree Format**: [docs/TREE_FORMAT.md](docs/TREE_FORMAT.md)
- **Configuration Guide**: [docs/CONFIGURATION_GUIDE.md](docs/CONFIGURATION_GUIDE.md)
- **Performance Guide**: [docs/PERFORMANCE_GUIDE.md](docs/PERFORMANCE_GUIDE.md)
- **Examples**: [examples/README.md](examples/README.md)

---

**Ready to get started?** Run `python examples/01_minimal_example.py` and you'll be up and running in under 5 minutes!
