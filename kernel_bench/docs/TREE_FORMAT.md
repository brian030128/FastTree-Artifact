# Tree Structure Format

Complete specification for FastTree's tree-structured KV cache format.

## Table of Contents

1. [Overview](#overview)
2. [KVTreeNode Structure](#kvtreenode-structure)
3. [File Format Specification](#file-format-specification)
4. [Programmatic Tree Creation](#programmatic-tree-creation)
5. [Request Assignment](#request-assignment)
6. [KV_ptrs Array](#kv_ptrs-array)
7. [Common Tree Patterns](#common-tree-patterns)
8. [Tree Validation](#tree-validation)

---

## Overview

FastTree represents KV caches as tree structures where:
- Each **node** represents a sequence of tokens
- **Edges** connect parent nodes to child nodes
- Each **path** from root to leaf represents one complete request
- Nodes can be **shared** by multiple requests

This representation enables efficient attention computation when multiple requests share common prompt prefixes.

### Visual Example

```
         Root (System prompt: 100 tokens)
        /                |                \
    Doc 1             Doc 2             Doc 3
  (500 tokens)      (500 tokens)      (500 tokens)
      |                 |                 |
 Question 1        Question 2        Question 3
 (20 tokens)       (20 tokens)       (20 tokens)
```

**Request paths:**
- Request 0: Root → Doc 1 → Question 1 (total: 620 tokens)
- Request 1: Root → Doc 2 → Question 2 (total: 620 tokens)
- Request 2: Root → Doc 3 → Question 3 (total: 620 tokens)

All three requests share the root node's 100 tokens.

---

## KVTreeNode Structure

**Location**: `kv_tree_simple.py:4-10`

### Class Definition

```python
class KVTreeNode:
    def __init__(self):
        self.parent = -1       # Parent node ID (-1 for root)
        self.id = -1          # Unique node identifier
        self.seqlen = 0       # Number of tokens in this node
        self.num_children = 0 # Number of child nodes
        self.requests = []    # List of request IDs using this node
```

### Attribute Descriptions

#### parent

**Type**: `int`
**Constraints**: `-1` for root, otherwise valid node ID in range `[0, num_nodes-1]`

The ID of this node's parent. The root node (typically ID 0) has `parent = -1`.

**Example:**
```python
root.parent = -1     # Root has no parent
child.parent = 0     # Child's parent is node 0
```

---

#### id

**Type**: `int`
**Constraints**: Unique in range `[0, num_nodes-1]`

Unique identifier for this node. Typically assigned consecutively starting from 0.

**Recommended:** Use breadth-first ordering (root=0, then level 1, then level 2, etc.)

**Example:**
```python
root.id = 0
child1.id = 1
child2.id = 2
```

---

#### seqlen

**Type**: `int`
**Constraints**: `> 0` (must have at least 1 token)

Number of tokens stored in this node. For a complete request, sum the `seqlen` of all nodes along the path from root to leaf.

**Example:**
```python
root.seqlen = 128     # System prompt
child1.seqlen = 64    # User query
child2.seqlen = 256   # Response tokens
```

---

#### num_children

**Type**: `int`
**Constraints**: `>= 0`

Number of children this node has. Leaf nodes have `num_children = 0`.

**Note:** This attribute is informational and used by tree traversal algorithms.

**Example:**
```python
root.num_children = 3     # Root has 3 children
leaf.num_children = 0     # Leaf node
```

---

#### requests

**Type**: `List[int]`

List of request IDs that include this node in their path. Request IDs are typically `[0, batch_size-1]`.

**Key insight:**
- **Leaf nodes** should have exactly one request (the request ending at that leaf)
- **Internal nodes** should have all requests that pass through them

**Example:**
```python
root.requests = [0, 1, 2]     # All three requests use root
child1.requests = [0]         # Only request 0 uses child1
child2.requests = [1]         # Only request 1 uses child2
```

---

## File Format Specification

### Text File Format

```
<num_nodes>
<parent> <id> <seqlen> <num_children>
<parent> <id> <seqlen> <num_children>
...
```

### Format Rules

1. **First line**: Single integer indicating total number of nodes
2. **Subsequent lines**: One node per line with 4 space-separated integers
   - `parent`: Parent node ID (-1 for root)
   - `id`: Node ID
   - `seqlen`: Sequence length
   - `num_children`: Number of children
3. **Request assignment**: NOT stored in file; computed when loading

### Example Files

#### Example 1: Simple Binary Tree

```
3
-1 0 128 2
0 1 64 0
0 2 64 0
```

**Structure:**
```
    Node 0 (128 tokens)
    /              \
Node 1 (64)    Node 2 (64)
```

**Loading:**
```python
from kv_tree_simple import retrive_from_file

tree_info = retrive_from_file("simple_tree.txt")
# tree_info[0]: root (parent=-1, id=0, seqlen=128, num_children=2)
# tree_info[1]: child1 (parent=0, id=1, seqlen=64, num_children=0)
# tree_info[2]: child2 (parent=0, id=2, seqlen=64, num_children=0)
```

---

#### Example 2: Linear Chain

```
4
-1 0 100 1
0 1 200 1
1 2 300 1
2 3 400 0
```

**Structure:**
```
Node 0 (100) → Node 1 (200) → Node 2 (300) → Node 3 (400)
```

This represents a single request with total sequence length: 100 + 200 + 300 + 400 = 1000 tokens.

---

#### Example 3: Three-Level Tree

```
7
-1 0 50 2
0 1 100 2
0 2 100 0
1 3 150 0
1 4 150 0
```

**Structure:**
```
        Node 0 (50)
       /           \
   Node 1 (100)   Node 2 (100)
   /         \
Node 3 (150) Node 4 (150)
```

**Paths:**
- Request 0: 0 → 2 (total: 150 tokens)
- Request 1: 0 → 1 → 3 (total: 300 tokens)
- Request 2: 0 → 1 → 4 (total: 300 tokens)

---

### Loading from File

```python
from kv_tree_simple import retrive_from_file

tree_info = retrive_from_file("tree.txt")
print(f"Loaded {len(tree_info)} nodes")

# Access nodes
for node in tree_info:
    print(f"Node {node.id}: parent={node.parent}, seqlen={node.seqlen}")
```

**Note:** The `requests` attribute is computed automatically by `qkv_preparation()` based on leaf nodes.

---

## Programmatic Tree Creation

### Simple Binary Tree

```python
from kv_tree_simple import KVTreeNode

def create_binary_tree():
    """Create a simple binary tree: root with 2 children"""
    tree_info = []

    # Root node
    root = KVTreeNode()
    root.parent = -1
    root.id = 0
    root.seqlen = 128
    root.num_children = 2
    root.requests = [0, 1]  # Both requests use root
    tree_info.append(root)

    # Left child (request 0)
    left = KVTreeNode()
    left.parent = 0
    left.id = 1
    left.seqlen = 64
    left.num_children = 0
    left.requests = [0]
    tree_info.append(left)

    # Right child (request 1)
    right = KVTreeNode()
    right.parent = 0
    right.id = 2
    right.seqlen = 64
    right.num_children = 0
    right.requests = [1]
    tree_info.append(right)

    return tree_info
```

---

### Linear Chain

```python
def create_chain(num_nodes, tokens_per_node=100):
    """Create a linear chain of nodes"""
    tree_info = []

    for i in range(num_nodes):
        node = KVTreeNode()
        node.parent = i - 1 if i > 0 else -1
        node.id = i
        node.seqlen = tokens_per_node
        node.num_children = 1 if i < num_nodes - 1 else 0
        node.requests = [0]  # Single request
        tree_info.append(node)

    return tree_info

# Create chain with 5 nodes, 100 tokens each
tree_info = create_chain(5, 100)  # Total: 500 tokens
```

---

### Balanced Binary Tree

```python
def create_balanced_binary_tree(depth, tokens_per_node=100):
    """Create a balanced binary tree with given depth"""
    tree_info = []
    node_id = 0
    request_id = 0

    # Queue: (node_id, parent_id, current_depth)
    queue = [(0, -1, 0)]

    while queue:
        curr_id, parent_id, curr_depth = queue.pop(0)

        node = KVTreeNode()
        node.parent = parent_id
        node.id = curr_id
        node.seqlen = tokens_per_node

        # Determine if this is a leaf
        is_leaf = (curr_depth == depth - 1)
        node.num_children = 0 if is_leaf else 2

        if is_leaf:
            node.requests = [request_id]
            request_id += 1
        else:
            node.requests = []  # Will be filled later

        tree_info.append(node)

        # Add children to queue
        if not is_leaf:
            queue.append((curr_id * 2 + 1, curr_id, curr_depth + 1))
            queue.append((curr_id * 2 + 2, curr_id, curr_depth + 1))

        node_id += 1

    # Fill parent requests
    for node in reversed(tree_info):
        if node.num_children == 0:  # Leaf
            parent = node.parent
            while parent != -1:
                tree_info[parent].requests.extend(node.requests)
                parent = tree_info[parent].parent

    # Remove duplicates and sort
    for node in tree_info:
        node.requests = sorted(list(set(node.requests)))

    return tree_info

# Create balanced binary tree with depth 3
tree_info = create_balanced_binary_tree(3, 100)
# Nodes: 1 + 2 + 4 = 7 nodes
# Leaves (requests): 4 requests
```

---

## Request Assignment

### Automatic Assignment by qkv_preparation

The `qkv_preparation()` function in `flash_attn_wrap.py` automatically assigns requests to tree nodes:

```python
# flash_attn_wrap.py:24-33
for n in range(node_num):
    if tree_info[n].num_children == 0:  # Leaf node
        node = n
        num_requests += 1
        # Traverse up to root
        while node != -1:
            tree_info[node].requests.append(num_requests - 1)
            node = tree_info[node].parent
```

**Algorithm:**
1. For each leaf node, assign a new request ID
2. Traverse from leaf to root
3. Add the request ID to every node along the path

**Result:** Each node's `requests` list contains all request IDs that pass through that node.

---

### Manual Assignment

For manual control:

```python
def assign_requests_manually(tree_info, leaf_to_request):
    """
    Manually assign requests to tree nodes.

    Args:
        tree_info: List of KVTreeNode
        leaf_to_request: Dict mapping leaf node ID to request ID
    """
    # Clear existing requests
    for node in tree_info:
        node.requests = []

    # Assign based on leaf_to_request
    for leaf_id, request_id in leaf_to_request.items():
        node_id = leaf_id
        while node_id != -1:
            if request_id not in tree_info[node_id].requests:
                tree_info[node_id].requests.append(request_id)
            node_id = tree_info[node_id].parent

    # Sort request lists
    for node in tree_info:
        node.requests.sort()

# Example usage
leaf_to_request = {
    2: 0,  # Leaf 2 → Request 0
    3: 1,  # Leaf 3 → Request 1
    4: 2,  # Leaf 4 → Request 2
}
assign_requests_manually(tree_info, leaf_to_request)
```

---

## KV_ptrs Array

The `KV_ptrs` array stores cumulative token counts for efficient indexing into K/V buffers.

### Construction

```python
KV_ptrs = [0]
for node in tree_info:
    KV_ptrs.append(KV_ptrs[-1] + node.seqlen)

total_tokens = KV_ptrs[-1]
```

### Format

**Length:** `len(tree_info) + 1`
**Values:** Cumulative token counts

```python
# Example: tree with 3 nodes, seqlens [128, 64, 64]
KV_ptrs = [0, 128, 192, 256]
# Node 0 tokens: [0:128]
# Node 1 tokens: [128:192]
# Node 2 tokens: [192:256]
```

### Usage

```python
# Get token range for node i
start_idx = KV_ptrs[node.id]
end_idx = KV_ptrs[node.id + 1]
node_tokens = K_buffer[start_idx:end_idx]
```

### Complete Example

```python
from kv_tree_simple import KVTreeNode

# Create tree
tree_info = []
root = KVTreeNode()
root.parent, root.id, root.seqlen = -1, 0, 100
root.num_children, root.requests = 2, [0, 1]
tree_info.append(root)

child1 = KVTreeNode()
child1.parent, child1.id, child1.seqlen = 0, 1, 50
child1.num_children, child1.requests = 0, [0]
tree_info.append(child1)

child2 = KVTreeNode()
child2.parent, child2.id, child2.seqlen = 0, 2, 75
child2.num_children, child2.requests = 0, [1]
tree_info.append(child2)

# Build KV_ptrs
KV_ptrs = [0]
for node in tree_info:
    KV_ptrs.append(KV_ptrs[-1] + node.seqlen)

print(KV_ptrs)  # [0, 100, 150, 225]
print(f"Total tokens: {KV_ptrs[-1]}")  # 225
```

---

## Common Tree Patterns

### Pattern 1: Beam Search

**Use case:** Beam search decoding with width=4

```
        Prompt (1000 tokens)
       /      |      |      \
   Beam 1  Beam 2  Beam 3  Beam 4
   (10)    (10)    (10)    (10)
```

**Code:**
```python
def create_beam_search_tree(prompt_len, beam_width, beam_len):
    tree_info = []

    # Root: shared prompt
    root = KVTreeNode()
    root.parent, root.id, root.seqlen = -1, 0, prompt_len
    root.num_children = beam_width
    root.requests = list(range(beam_width))
    tree_info.append(root)

    # Beams
    for i in range(beam_width):
        beam = KVTreeNode()
        beam.parent, beam.id, beam.seqlen = 0, i + 1, beam_len
        beam.num_children, beam.requests = 0, [i]
        tree_info.append(beam)

    return tree_info

tree = create_beam_search_tree(1000, 4, 10)
# 5 nodes, 4 requests, total: 4*1010 = 4040 token-attentions
# Shared: 1000 tokens computed once
```

---

### Pattern 2: Speculative Decoding

**Use case:** Speculative decoding with 4 draft tokens, 2 accepted

```
    Prefix (500 tokens)
           |
    Draft attempts (4 tokens)
       /          \
   Accepted    Rejected
   (2 tokens)  (0 tokens)
```

**Code:**
```python
def create_speculative_tree(prefix_len, draft_len, accepted_len):
    tree_info = []

    # Prefix
    prefix = KVTreeNode()
    prefix.parent, prefix.id, prefix.seqlen = -1, 0, prefix_len
    prefix.num_children, prefix.requests = 1, [0, 1]
    tree_info.append(prefix)

    # Draft
    draft = KVTreeNode()
    draft.parent, draft.id, draft.seqlen = 0, 1, draft_len
    draft.num_children, draft.requests = 2, [0, 1]
    tree_info.append(draft)

    # Accepted path
    accepted = KVTreeNode()
    accepted.parent, accepted.id, accepted.seqlen = 1, 2, accepted_len
    accepted.num_children, accepted.requests = 0, [0]
    tree_info.append(accepted)

    # Rejected path (empty continuation)
    rejected = KVTreeNode()
    rejected.parent, rejected.id, rejected.seqlen = 1, 3, 1
    rejected.num_children, rejected.requests = 0, [1]
    tree_info.append(rejected)

    return tree_info
```

---

### Pattern 3: Multi-Document QA

**Use case:** Same question, 3 different documents

```
    Question (20 tokens)
    /      |      \
  Doc1   Doc2   Doc3
  (500)  (500)  (500)
```

**Code:**
```python
def create_multi_doc_qa(question_len, num_docs, doc_len):
    tree_info = []

    # Shared question
    question = KVTreeNode()
    question.parent, question.id, question.seqlen = -1, 0, question_len
    question.num_children = num_docs
    question.requests = list(range(num_docs))
    tree_info.append(question)

    # Documents
    for i in range(num_docs):
        doc = KVTreeNode()
        doc.parent, doc.id, doc.seqlen = 0, i + 1, doc_len
        doc.num_children, doc.requests = 0, [i]
        tree_info.append(doc)

    return tree_info

tree = create_multi_doc_qa(20, 3, 500)
```

---

## Tree Validation

### Validation Rules

A valid tree must satisfy:
1. **Single root**: Exactly one node with `parent = -1`
2. **Acyclic**: No node is its own ancestor
3. **Connected**: All nodes reachable from root
4. **Valid parent references**: All `parent` IDs in range `[-1, num_nodes-1]`
5. **Positive seqlens**: All `seqlen > 0`
6. **Consistent requests**: Leaf nodes have requests, internal nodes inherit from children

### Validation Function

```python
def validate_tree(tree_info):
    """Validate tree structure"""
    num_nodes = len(tree_info)

    # Check 1: Single root
    roots = [n for n in tree_info if n.parent == -1]
    assert len(roots) == 1, f"Expected 1 root, found {len(roots)}"

    # Check 2: Valid parent references
    for node in tree_info:
        if node.parent != -1:
            assert 0 <= node.parent < num_nodes, \
                f"Node {node.id} has invalid parent {node.parent}"

    # Check 3: No cycles (using DFS)
    def has_cycle(node_id, visited):
        if node_id in visited:
            return True
        visited.add(node_id)
        parent = tree_info[node_id].parent
        if parent != -1:
            return has_cycle(parent, visited)
        return False

    for node in tree_info:
        assert not has_cycle(node.id, set()), \
            f"Cycle detected involving node {node.id}"

    # Check 4: Positive seqlens
    for node in tree_info:
        assert node.seqlen > 0, \
            f"Node {node.id} has invalid seqlen {node.seqlen}"

    # Check 5: All nodes reachable from root
    reachable = set([roots[0].id])
    queue = [roots[0].id]
    while queue:
        curr = queue.pop(0)
        for node in tree_info:
            if node.parent == curr and node.id not in reachable:
                reachable.add(node.id)
                queue.append(node.id)

    assert len(reachable) == num_nodes, \
        f"Only {len(reachable)}/{num_nodes} nodes reachable from root"

    print(f"✓ Tree validation passed ({num_nodes} nodes)")
    return True

# Usage
try:
    validate_tree(tree_info)
except AssertionError as e:
    print(f"Validation failed: {e}")
```

---

## See Also

- [README_KERNEL.md](../README_KERNEL.md) - Main documentation
- [API_REFERENCE.md](API_REFERENCE.md) - Complete API reference
- [examples/03_custom_tree.py](../examples/03_custom_tree.py) - Tree creation examples
- [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) - Parameter tuning
