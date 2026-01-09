# FastTree API Reference

Complete technical reference for the FastTree standalone kernel API.

## Table of Contents

1. [FastTreeParams Class](#fasttreeparams-class)
2. [fasttree_preparation Function](#fasttree_preparation-function)
3. [fasttree_decode Function](#fasttree_decode-function)
4. [KVTreeNode Class](#kvtreenode-class)
5. [Helper Functions](#helper-functions)
6. [Type Definitions](#type-definitions)

---

## FastTreeParams Class

**Location**: `fasttree.py:7-27`

Configuration class for FastTree kernel parameters. Controls cost model weights and tile sizes.

### Class Definition

```python
class FastTreeParams:
    alpha: float = 0.66
    beta: float = 0.33
    gamma: float = 0.1
    TSQs: List[int] = [64, 16]
    TSKs: List[int] = [32, 128]
    kv_group_num: int = 1
```

### Attributes

#### alpha

**Type**: `float`
**Default**: `0.66`
**Range**: `[0.0, 1.0]`

Cost weight for Q-padding in matrix multiplication operations. Higher values penalize small query batch sizes more heavily, favoring Split-K decisions in the tree heuristic.

**When to tune:**
- Increase to 0.7-0.8 if you have very small query batches (< 8 requests)
- Decrease to 0.5-0.6 if you have large query batches (> 32 requests)

**Example:**
```python
params = FastTreeParams()
params.alpha = 0.7  # Favor larger Q tiles
```

---

#### beta

**Type**: `float`
**Default**: `0.33`
**Range**: `[0.0, 1.0]`

Cost weight for K-padding in matrix multiplication operations. Higher values penalize small key sequence lengths more heavily, favoring Split-Q decisions.

**When to tune:**
- Increase to 0.4-0.5 for short sequence lengths (< 256 tokens)
- Decrease to 0.2-0.3 for long sequence lengths (> 2048 tokens)

**Example:**
```python
params = FastTreeParams()
params.beta = 0.4  # More penalty for K-padding
```

---

#### gamma

**Type**: `float`
**Default**: `0.1`
**Range**: `[0.0, 0.3]`

Cost weight for reduction operations in stage 2. Higher values penalize creating many virtual nodes (Split-K strategy).

**When to tune:**
- Increase to 0.15-0.2 if stage 2 becomes a bottleneck (many vnodes)
- Keep low (0.05-0.1) if stage 1 dominates runtime

**Example:**
```python
params = FastTreeParams()
params.gamma = 0.15  # Reduce vnode count
```

---

#### TSQs

**Type**: `List[int]`
**Default**: `[64, 16]`
**Format**: `[phase0_tile_size, phase1_tile_size]`

Query tile sizes for each phase. Phase 0 handles large query batches (> TSQs[1]), Phase 1 handles small batches.

**Constraints:**
- Must be positive integers
- Typically powers of 2 or multiples of 4
- phase0_tile_size >= phase1_tile_size
- Recommended range: 4-128

**Hardware recommendations:**

| GPU | MHA (GQA=1) | GQA=4 | GQA=8 |
|-----|-------------|-------|-------|
| H100 | [64, 16] | [16, 4] | [8, 2] |
| A100 | [48, 12] | [16, 4] | [8, 2] |
| V100 | [32, 8] | [8, 2] | [4, 1] |

**Example:**
```python
params = FastTreeParams()
params.TSQs = [32, 8]  # Smaller tiles for moderate batches
```

---

#### TSKs

**Type**: `List[int]`
**Default**: `[32, 128]`
**Format**: `[phase0_tile_size, phase1_tile_size]`

KV tile sizes for each phase. Controls how sequence length is tiled.

**Constraints:**
- Must be positive integers
- Typically 16-256
- Affects memory access patterns and parallelism

**When to tune:**
- Increase for longer sequences (better memory coalescing)
- Decrease for shorter sequences (better parallelism)

**Example:**
```python
params = FastTreeParams()
params.TSKs = [32, 96]  # Moderate KV tiles
```

---

#### kv_group_num

**Type**: `int`
**Default**: `1`
**Range**: `[1, num_qo_heads]`

Number of query heads per key/value head. Used for Grouped Query Attention (GQA).

**Calculation:**
```python
kv_group_num = num_qo_heads // num_kv_heads
```

**Common values:**
- `1`: Multi-Head Attention (MHA) - all models before GQA
- `4`: Llama-3.1-8B, Mistral-7B
- `8`: Some GQA variants
- `16`: Rare, very aggressive grouping

**Example:**
```python
num_qo_heads = 32
num_kv_heads = 8  # GQA with ratio 4

params = FastTreeParams()
params.set_kv_group_num(num_qo_heads // num_kv_heads)  # 4
```

---

### Methods

#### set_values

**Signature:**
```python
def set_values(self, alpha: float, beta: float, gamma: float) -> None
```

Set all three cost model parameters at once.

**Parameters:**
- `alpha` (float): Q-padding cost weight [0.0, 1.0]
- `beta` (float): K-padding cost weight [0.0, 1.0]
- `gamma` (float): Reduction cost weight [0.0, 0.3]

**Returns:** None

**Example:**
```python
params = FastTreeParams()
params.set_values(alpha=0.7, beta=0.35, gamma=0.12)
```

---

#### set_kv_group_num

**Signature:**
```python
def set_kv_group_num(self, kv_group_num: int) -> None
```

Set the GQA group number.

**Parameters:**
- `kv_group_num` (int): Number of Q heads per KV head [1, num_qo_heads]

**Returns:** None

**Example:**
```python
params = FastTreeParams()
params.set_kv_group_num(4)  # GQA with ratio 4
```

---

#### set_q_tile_sizes

**Signature:**
```python
def set_q_tile_sizes(self, TSQs: List[int]) -> None
```

Set query tile sizes for both phases.

**Parameters:**
- `TSQs` (List[int]): Query tile sizes [phase0, phase1]

**Returns:** None

**Example:**
```python
params = FastTreeParams()
params.set_q_tile_sizes([32, 8])
```

---

#### set_kv_tile_sizes

**Signature:**
```python
def set_kv_tile_sizes(self, TSKs: List[int]) -> None
```

Set KV tile sizes for both phases.

**Parameters:**
- `TSKs` (List[int]): KV tile sizes [phase0, phase1]

**Returns:** None

**Example:**
```python
params = FastTreeParams()
params.set_kv_tile_sizes([32, 96])
```

---

## fasttree_preparation Function

**Location**: `fasttree.py:99-296`

Analyzes tree structure and prepares metadata for decoding. This function implements the tree heuristic algorithm and creates virtual node (vnode) decomposition.

### Function Signature

```python
def fasttree_preparation(
    tree_info: List[KVTreeNode],
    KV_ptrs: List[int],
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    KV_SPLIT_SIZES: List[int],
    para_threshs1: List[int],
    para_threshs2: List[int],
    params: FastTreeParams,
) -> Tuple[Tuple[Tensor, ...], List[int]]
```

### Parameters

#### tree_info

**Type**: `List[KVTreeNode]`
**Description**: Tree structure representing KV cache organization

List of tree nodes where each node contains:
- `parent`: Parent node ID (-1 for root)
- `id`: Unique node identifier
- `seqlen`: Number of tokens in this node
- `num_children`: Number of child nodes
- `requests`: List of request IDs that use this node

**Constraints:**
- Must form a valid tree (no cycles, single root)
- Root node must have `parent = -1`
- Node IDs should be consecutive [0, num_nodes-1]
- Each leaf should have at least one request

**Example:**
```python
from kv_tree_simple import retrive_from_file
tree_info = retrive_from_file("tree.txt")
```

---

#### KV_ptrs

**Type**: `List[int]`
**Description**: Cumulative token counts for each node

Array where `KV_ptrs[i]` is the starting index of node `i`'s tokens in the K/V buffers.

**Format:**
```python
KV_ptrs = [0, node0_seqlen, node0_seqlen + node1_seqlen, ...]
# Length: len(tree_info) + 1
# Last element: total number of tokens
```

**Example:**
```python
# Tree with 3 nodes: seqlens [128, 64, 64]
KV_ptrs = [0, 128, 192, 256]
total_tokens = KV_ptrs[-1]  # 256
```

---

#### batch_size

**Type**: `int`
**Description**: Number of requests (batch size)

Typically equals the number of leaf nodes in the tree, since each leaf corresponds to one request.

**Constraints:**
- Must be positive
- Should match number of unique request IDs in tree

**Example:**
```python
batch_size = 8  # 8 concurrent requests
```

---

#### num_qo_heads

**Type**: `int`
**Description**: Number of query/output attention heads

**Constraints:**
- Must be positive
- Must be divisible by `num_kv_heads` for GQA

**Common values:**
- 32: Llama-7B/13B, GPT-3
- 40: Llama-65B
- 64: GPT-3-175B

**Example:**
```python
num_qo_heads = 32
```

---

#### num_kv_heads

**Type**: `int`
**Description**: Number of key/value attention heads

For MHA (Multi-Head Attention): `num_kv_heads = num_qo_heads`
For GQA (Grouped Query Attention): `num_kv_heads < num_qo_heads`

**Constraints:**
- Must be positive
- `num_qo_heads` must be divisible by `num_kv_heads`

**Example:**
```python
num_qo_heads = 32
num_kv_heads = 8  # GQA ratio = 4
```

---

#### head_dim

**Type**: `int`
**Description**: Dimension of each attention head

**Constraints:**
- **Must be in {16, 32, 64, 128, 256}** (enforced by Triton kernel)

**Common values:**
- 64: Some older models
- 128: Most modern LLMs (Llama, GPT, etc.)
- 256: Some large models

**Example:**
```python
head_dim = 128
```

---

#### KV_SPLIT_SIZES

**Type**: `List[int]`
**Default**: `[1024, 128]`
**Format**: `[initial_split_size, reduced_split_size]`

Thresholds for splitting sequence length into chunks. The preparation function may adaptively reduce split sizes based on parallelism analysis.

**When to tune:**
- Increase for longer average sequences (better memory efficiency)
- Decrease for shorter sequences or memory constraints

**Example:**
```python
KV_SPLIT_SIZES = [1024, 128]  # Default for H100
KV_SPLIT_SIZES = [768, 96]    # For memory-constrained scenarios
```

---

#### para_threshs1

**Type**: `List[int]`
**Default**: `[132, 528]`
**Format**: `[phase0_threshold, phase1_threshold]`

First-level parallelism thresholds. If computed parallelism falls below these thresholds, the preparation function reduces KV split sizes to increase parallelism.

**Hardware-specific values:**
- H100: `[132, 528]`
- A100: `[100, 400]`
- V100: `[80, 320]`

**Example:**
```python
para_threshs1 = [132, 528]  # H100 defaults
```

---

#### para_threshs2

**Type**: `List[int]`
**Default**: `[132, 132]`
**Format**: `[phase0_threshold, phase1_threshold]`

Second-level parallelism thresholds. If parallelism still falls below these after KV split adjustment, the function adjusts tile sizes.

**Hardware-specific values:**
- H100: `[132, 132]`
- A100: `[100, 100]`
- V100: `[80, 80]`

**Example:**
```python
para_threshs2 = [132, 132]  # H100 defaults
```

---

#### params

**Type**: `FastTreeParams`
**Description**: FastTree configuration parameters

See [FastTreeParams Class](#fasttreeparams-class) for details.

**Example:**
```python
params = FastTreeParams()
params.set_values(0.66, 0.33, 0.1)
params.set_kv_group_num(4)
```

---

### Return Values

**Type**: `Tuple[Tuple[Tensor, ...], List[int]]`

Returns a tuple containing:
1. **metadata_tuple** (13 tensors): Metadata for `fasttree_decode`
2. **node_assignments** (list): Split strategy for each node

#### metadata_tuple

A tuple of 13 tensors (all on CUDA):

1. **vnode_to_kv_entries** (`Tensor[int32, N]`): KV cache indices for all vnodes
2. **vnode_to_kv_offs** (`Tensor[int32, V]`): Starting offset into vnode_to_kv_entries for each vnode
3. **vnode_to_kv_lens** (`Tensor[int32, V]`): Number of tokens for each vnode
4. **vnode_to_q_entries** (`Tensor[int32, Q]`): Request indices for all vnodes
5. **vnode_to_q_offs** (`Tensor[int32, V]`): Starting offset into vnode_to_q_entries for each vnode
6. **vnode_to_q_lens** (`Tensor[int32, V]`): Number of queries for each vnode
7. **req_to_vnode_entries** (`Tensor[int32, R]`): Vnode indices for all requests
8. **req_to_vnode_offs** (`Tensor[int32, B]`): Starting offset into req_to_vnode_entries for each request
9. **req_to_vnode_lens** (`Tensor[int32, B]`): Number of vnodes for each request
10. **mid_o** (`Tensor[float32, (Q, H, D)]`): Intermediate output buffer
11. **mid_lse** (`Tensor[float32, (Q, H)]`): Intermediate log-sum-exp buffer
12. **phase_node_nums** (`List[int, 2]`): Number of vnodes in each phase
13. **phase_node_offsets** (`List[int, 2]`): Starting vnode index for each phase

Where:
- `V` = total number of vnodes
- `N` = total number of KV indices across all vnodes
- `Q` = total number of query entries across all vnodes
- `R` = total number of vnode indices across all requests
- `B` = batch_size
- `H` = num_qo_heads
- `D` = head_dim

#### node_assignments

**Type**: `List[int]`
**Length**: `len(tree_info)`
**Values**: 0 or 1

Split strategy for each tree node:
- **0**: Split-K (split sequence length dimension)
- **1**: Split-Q (split query batch dimension)

**Example:**
```python
node_assignments = [0, 1, 1, 0, ...]
# Node 0: Split-K
# Node 1: Split-Q
# Node 2: Split-Q
# Node 3: Split-K
```

---

### Usage Example

```python
from fasttree import FastTreeParams, fasttree_preparation
from kv_tree_simple import retrive_from_file

# Load tree
tree_info = retrive_from_file("tree.txt")

# Build KV_ptrs
KV_ptrs = [0]
for node in tree_info:
    KV_ptrs.append(KV_ptrs[-1] + node.seqlen)

# Configuration
batch_size = 8
num_qo_heads = 32
num_kv_heads = 8
head_dim = 128

# Prepare parameters
params = FastTreeParams()
params.set_kv_group_num(num_qo_heads // num_kv_heads)

# Run preparation
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

print(f"Created {len(node_assignments)} node assignments")
print(f"Metadata contains {len(metadata)} tensors")
```

---

### Performance Notes

- **Call once** per tree structure; reuse metadata for multiple decodes
- Preparation typically takes 1-10ms depending on tree size
- Allocates GPU memory for intermediate buffers (mid_o, mid_lse)
- Prints parallelism statistics and adjusted parameters to stdout

---

## fasttree_decode Function

**Location**: `fasttree.py:582-643`

Computes attention output for tree-structured KV cache using two-stage algorithm.

### Function Signature

```python
def fasttree_decode(
    q: Tensor,
    k_buffer: Tensor,
    v_buffer: Tensor,
    o: Tensor,
    vnode_to_kv_entries: Tensor,
    vnode_to_kv_offs: Tensor,
    vnode_to_kv_lens: Tensor,
    vnode_to_q_entries: Tensor,
    vnode_to_q_offs: Tensor,
    vnode_to_q_lens: Tensor,
    req_to_vnode_entries: Tensor,
    req_to_vnode_offs: Tensor,
    req_to_vnode_lens: Tensor,
    mid_o: Tensor,
    mid_lse: Tensor,
    phase_node_nums: List[int],
    phase_node_offsets: List[int],
    phase_q_tile_sizes: List[int],
    phase_kv_tile_sizes: List[int],
    sm_scale: float,
    logit_cap: float = -1,
) -> None
```

### Parameters

#### q

**Type**: `Tensor[dtype, (batch_size, num_qo_heads, head_dim)]`
**Device**: CUDA
**Dtype**: float16 or bfloat16

Query tensor. One query vector per request.

**Shape constraints:**
- `batch_size`: Must match preparation
- `num_qo_heads`: Must match preparation
- `head_dim`: Must match preparation and be in {16, 32, 64, 128, 256}

**Example:**
```python
q = torch.randn(8, 32, 128, device='cuda', dtype=torch.float16)
```

---

#### k_buffer

**Type**: `Tensor[dtype, (total_tokens, num_kv_heads, head_dim)]`
**Device**: CUDA
**Dtype**: float16 or bfloat16

Key tensor containing all tokens from tree nodes.

**Shape constraints:**
- `total_tokens`: Must equal `KV_ptrs[-1]` from preparation
- Tokens organized according to tree structure and KV_ptrs

**Example:**
```python
total_tokens = 2048
k_buffer = torch.randn(total_tokens, 8, 128, device='cuda', dtype=torch.float16)
```

---

#### v_buffer

**Type**: `Tensor[dtype, (total_tokens, num_kv_heads, head_dim)]`
**Device**: CUDA
**Dtype**: float16 or bfloat16

Value tensor containing all tokens from tree nodes.

**Shape constraints:** Same as `k_buffer`

**Example:**
```python
v_buffer = torch.randn(total_tokens, 8, 128, device='cuda', dtype=torch.float16)
```

---

#### o

**Type**: `Tensor[dtype, (batch_size, num_qo_heads, head_dim)]`
**Device**: CUDA
**Dtype**: float16 or bfloat16

Output tensor. **Modified in-place** by the function.

**Important:** Must be pre-allocated with correct shape. Contents will be overwritten.

**Example:**
```python
o = torch.empty(8, 32, 128, device='cuda', dtype=torch.float16)
fasttree_decode(..., o=o, ...)  # o is modified
print(o)  # Now contains attention output
```

---

#### Metadata Parameters (13 tensors)

**vnode_to_kv_entries through phase_node_offsets**

These 13 parameters are the metadata tensors returned by `fasttree_preparation`. Pass them directly using unpacking:

```python
metadata, _ = fasttree_preparation(...)
fasttree_decode(q, k, v, o, *metadata, ...)  # Unpack 13 tensors
```

See [fasttree_preparation Return Values](#return-values) for details on each tensor.

---

#### phase_q_tile_sizes

**Type**: `List[int]`
**Format**: `[phase0_tile, phase1_tile]`
**Description**: Query tile sizes for each phase

Typically pass `params.TSQs` directly.

**Example:**
```python
phase_q_tile_sizes = params.TSQs  # [64, 16]
```

---

#### phase_kv_tile_sizes

**Type**: `List[int]`
**Format**: `[phase0_tile, phase1_tile]`
**Description**: KV tile sizes for each phase

Typically pass `params.TSKs` directly.

**Example:**
```python
phase_kv_tile_sizes = params.TSKs  # [32, 128]
```

---

#### sm_scale

**Type**: `float`
**Description**: Softmax scaling factor

Standard attention scaling factor: `1.0 / sqrt(head_dim)`

**Calculation:**
```python
sm_scale = 1.0 / (head_dim ** 0.5)
# For head_dim=128: sm_scale = 0.08838834764831845
```

**Example:**
```python
head_dim = 128
sm_scale = 1.0 / (head_dim ** 0.5)
fasttree_decode(..., sm_scale=sm_scale, ...)
```

---

#### logit_cap

**Type**: `float`
**Default**: `-1` (disabled)
**Description**: Optional logit capping for numerical stability

If positive, caps attention logits to `[-logit_cap, logit_cap]` before softmax.

**When to use:**
- Very long sequences (>8K tokens)
- FP16 precision concerns
- Numerical stability issues

**Recommended value**: 30.0-50.0

**Example:**
```python
fasttree_decode(..., logit_cap=30.0)  # Cap logits to [-30, 30]
```

---

### Return Value

**None** (modifies `o` in-place)

---

### Usage Example

```python
import torch
from fasttree import FastTreeParams, fasttree_preparation, fasttree_decode

# Assume preparation is done
metadata, _ = fasttree_preparation(...)

# Create input tensors
batch_size = 8
num_qo_heads = 32
num_kv_heads = 8
head_dim = 128
total_tokens = 2048

q = torch.randn(batch_size, num_qo_heads, head_dim,
                device='cuda', dtype=torch.float16)
k = torch.randn(total_tokens, num_kv_heads, head_dim,
                device='cuda', dtype=torch.float16)
v = torch.randn(total_tokens, num_kv_heads, head_dim,
                device='cuda', dtype=torch.float16)
o = torch.empty(batch_size, num_qo_heads, head_dim,
                device='cuda', dtype=torch.float16)

# Compute attention
params = FastTreeParams()
sm_scale = 1.0 / (head_dim ** 0.5)

fasttree_decode(
    q=q,
    k_buffer=k,
    v_buffer=v,
    o=o,
    *metadata,  # Unpack 13 metadata tensors
    phase_q_tile_sizes=params.TSQs,
    phase_kv_tile_sizes=params.TSKs,
    sm_scale=sm_scale,
    logit_cap=-1,  # Disabled
)

print(f"Output shape: {o.shape}")  # (8, 32, 128)
```

---

### Performance Notes

- **Fast**: Typically 1-5ms for moderate trees (1K-4K tokens, 8-16 requests)
- **Reusable**: Can call multiple times with same metadata but different Q/K/V
- **In-place**: Modifies output tensor `o` directly
- **Two-stage**: Stage 1 computes vnode attention, Stage 2 reduces to final output

---

## KVTreeNode Class

**Location**: `kv_tree_simple.py:4-10`

Data structure representing a node in the tree-structured KV cache.

### Class Definition

```python
class KVTreeNode:
    def __init__(self):
        self.parent = -1
        self.id = -1
        self.seqlen = 0
        self.num_children = 0
        self.requests = []
```

### Attributes

- **parent** (int): Parent node ID (-1 for root node)
- **id** (int): Unique node identifier (typically 0 to num_nodes-1)
- **seqlen** (int): Number of tokens stored in this node
- **num_children** (int): Number of child nodes
- **requests** (List[int]): List of request IDs that include this node in their path

### Usage Example

```python
from kv_tree_simple import KVTreeNode

# Create root node
root = KVTreeNode()
root.parent = -1
root.id = 0
root.seqlen = 128
root.num_children = 2
root.requests = [0, 1]  # Both requests use root

# Create child node
child = KVTreeNode()
child.parent = 0  # Parent is root
child.id = 1
child.seqlen = 64
child.num_children = 0  # Leaf node
child.requests = [0]  # Only request 0 uses this child

tree_info = [root, child]
```

---

## Helper Functions

### retrive_from_file

**Location**: `kv_tree_simple.py:50-64`

Load tree structure from file.

**Signature:**
```python
def retrive_from_file(filepath: str) -> List[KVTreeNode]
```

**File Format:**
```
<num_nodes>
<parent> <id> <seqlen> <num_children>
<parent> <id> <seqlen> <num_children>
...
```

**Example:**
```python
from kv_tree_simple import retrive_from_file

tree_info = retrive_from_file("tree.txt")
print(f"Loaded {len(tree_info)} nodes")
```

---

### qkv_preparation

**Location**: `flash_attn_wrap.py:5-79`

Helper function to prepare Q/K/V tensors from tree structure. Creates both tree-structured tensors (for FastTree) and cache tensors (for Flash Attention comparison).

**Signature:**
```python
def qkv_preparation(
    tree_info: List[KVTreeNode],
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[int]]
```

**Returns:**
- Q: Query tensor [batch_size, num_qo_heads, head_dim]
- K_cache: Key cache for Flash Attention [batch_size, max_seqlen, num_kv_heads, head_dim]
- V_cache: Value cache for Flash Attention [batch_size, max_seqlen, num_kv_heads, head_dim]
- cache_seqlens: Sequence lengths [batch_size]
- K_tree_tensor: Key tensor for FastTree [total_tokens, num_kv_heads, head_dim]
- V_tree_tensor: Value tensor for FastTree [total_tokens, num_kv_heads, head_dim]
- KV_ptrs: Cumulative token counts [num_nodes + 1]

**Example:**
```python
from flash_attn_wrap import qkv_preparation

Q, K_cache, V_cache, cache_seqlens, K_tree, V_tree, KV_ptrs = \
    qkv_preparation(tree_info, 32, 32, 128, "cuda", torch.float16)
```

---

## Type Definitions

### Common Types

```python
# Tensors
Tensor = torch.Tensor

# Tree structure
TreeInfo = List[KVTreeNode]

# Metadata tuple (13 tensors returned by fasttree_preparation)
MetadataTuple = Tuple[
    Tensor,  # vnode_to_kv_entries
    Tensor,  # vnode_to_kv_offs
    Tensor,  # vnode_to_kv_lens
    Tensor,  # vnode_to_q_entries
    Tensor,  # vnode_to_q_offs
    Tensor,  # vnode_to_q_lens
    Tensor,  # req_to_vnode_entries
    Tensor,  # req_to_vnode_offs
    Tensor,  # req_to_vnode_lens
    Tensor,  # mid_o
    Tensor,  # mid_lse
    List[int],  # phase_node_nums
    List[int],  # phase_node_offsets
]
```

---

## Complete Workflow Example

```python
import torch
from fasttree import FastTreeParams, fasttree_preparation, fasttree_decode
from kv_tree_simple import retrive_from_file
from flash_attn_wrap import qkv_preparation

# Configuration
num_qo_heads = 32
num_kv_heads = 8
head_dim = 128
device = "cuda"
dtype = torch.float16

# Step 1: Load tree
tree_info = retrive_from_file("tree.txt")

# Step 2: Prepare Q/K/V data
Q, _, _, _, K_tree, V_tree, KV_ptrs = qkv_preparation(
    tree_info, num_qo_heads, num_kv_heads, head_dim, device, dtype
)
batch_size = Q.shape[0]

# Step 3: Configure parameters
params = FastTreeParams()
params.set_values(0.66, 0.33, 0.1)
params.set_kv_group_num(num_qo_heads // num_kv_heads)
params.set_q_tile_sizes([64, 16])
params.set_kv_tile_sizes([32, 128])

# Step 4: Preparation
metadata, node_assignments = fasttree_preparation(
    tree_info, KV_ptrs, batch_size, num_qo_heads, num_kv_heads, head_dim,
    [1024, 128], [132, 528], [132, 132], params
)

# Step 5: Decode
O = torch.empty(batch_size, num_qo_heads, head_dim, device=device, dtype=dtype)
sm_scale = 1.0 / (head_dim ** 0.5)

fasttree_decode(
    Q, K_tree, V_tree, O, *metadata,
    params.TSQs, params.TSKs, sm_scale
)

print(f"Output shape: {O.shape}")
print("Success!")
```

---

## See Also

- [README_KERNEL.md](../README_KERNEL.md) - Main documentation
- [TREE_FORMAT.md](TREE_FORMAT.md) - Tree structure specification
- [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) - Parameter tuning guide
- [PERFORMANCE_GUIDE.md](PERFORMANCE_GUIDE.md) - Optimization strategies
