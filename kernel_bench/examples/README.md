# FastTree Examples Guide

Comprehensive collection of examples demonstrating FastTree standalone kernel usage, from basic to advanced.

## Quick Start

**New to FastTree?** Start here:
```bash
cd examples
python 01_minimal_example.py
```

This will run a simple example that completes in under 5 minutes.

---

## Examples Overview

### Difficulty Levels
- 🟢 **Beginner**: No prior knowledge needed
- 🟡 **Intermediate**: Requires understanding basic usage
- 🔴 **Advanced**: Requires understanding of internals

### Estimated Time
- ⚡ Quick (< 10 min)
- 🕐 Moderate (10-30 min)
- 🕐🕐 Extended (> 30 min)

---

## Phase 1: Getting Started

### 01_minimal_example.py
**Level**: 🟢 Beginner | **Time**: ⚡ Quick

The absolute simplest FastTree example. Creates a small tree programmatically and runs attention.

**What you'll learn:**
- Basic FastTree workflow (params → preparation → decode)
- How to create tree structures in code
- How to prepare K/V data
- How to validate output

**Prerequisites:** None

**Run:**
```bash
python 01_minimal_example.py
```

**Expected output:**
```
==========================================
FastTree Minimal Example
==========================================
Configuration:
  Heads (Q/O): 32
  Heads (K/V): 32
  ...
Output shape: torch.Size([2, 32, 128])
Example completed successfully!
==========================================
```

---

### 02_basic_usage.py
**Level**: 🟢 Beginner | **Time**: ⚡ Quick

Standard usage pattern with file I/O, command-line arguments, and performance benchmarking.

**What you'll learn:**
- How to load trees from files
- Command-line argument handling
- Performance measurement
- Output validation
- Using the `qkv_preparation` helper

**Prerequisites:** 01_minimal_example.py

**Run:**
```bash
# Basic usage
python 02_basic_usage.py --tree_file ../data/tree_example.txt

# With custom parameters
python 02_basic_usage.py --tree_file ../data/tree_example.txt \
    --num_qo_heads 32 --num_kv_heads 8 --alpha 0.7

# With profiling
python 02_basic_usage.py --tree_file ../data/tree_example.txt --profile
```

**Arguments:**
- `--tree_file`: Path to tree file (required)
- `--num_qo_heads`: Number of query heads (default: 32)
- `--num_kv_heads`: Number of KV heads (default: 32)
- `--head_dim`: Head dimension (default: 128)
- `--alpha`, `--beta`, `--gamma`: Cost model parameters
- `--q_tile_sizes`, `--kv_tile_sizes`: Tile sizes
- `--profile`: Enable detailed benchmarking

---

## Phase 2: Tree Creation

### 03_custom_tree.py
**Level**: 🟡 Intermediate | **Time**: 🕐 Moderate

Demonstrates creating various tree topologies programmatically.

**What you'll learn:**
- How to create different tree structures (chain, binary, n-ary)
- Tree visualization techniques
- Request assignment strategies
- Tree validation

**Prerequisites:** 01_minimal_example.py

**Run:**
```bash
# Create and visualize a binary tree
python 03_custom_tree.py --topology binary --depth 4

# Create a chain
python 03_custom_tree.py --topology chain --nodes 5

# Create an n-ary tree
python 03_custom_tree.py --topology nary --branching 3 --depth 3
```

**Tree topologies:**
- `chain`: Linear sequence of nodes
- `binary`: Balanced binary tree
- `nary`: N-ary tree with specified branching factor
- `custom`: User-defined structure

---

## Phase 3: Configuration & Tuning

### 04_gqa_example.py
**Level**: 🟡 Intermediate | **Time**: 🕐 Moderate

Configuration guide for Grouped Query Attention (GQA) with different ratios.

**What you'll learn:**
- How to configure FastTree for GQA
- Recommended tile sizes for different GQA ratios
- Performance comparison: MHA vs GQA
- When to use which tile sizes

**Prerequisites:** 02_basic_usage.py

**Run:**
```bash
# Test GQA ratio = 4 (e.g., Llama-3.1-8B, Mistral)
python 04_gqa_example.py --gqa_ratio 4

# Test GQA ratio = 8
python 04_gqa_example.py --gqa_ratio 8

# Compare all GQA ratios
python 04_gqa_example.py --compare_all
```

**GQA ratios covered:**
- 1 (MHA): Standard multi-head attention
- 4: Llama-3.1, Mistral-7B
- 8: Some GQA models
- 16: Rare configurations

---

### 05_parameter_tuning.py
**Level**: 🔴 Advanced | **Time**: 🕐🕐 Extended

Interactive parameter tuning tool to find optimal configuration for your workload.

**What you'll learn:**
- How to sweep over parameter space
- How tree characteristics affect optimal parameters
- How to analyze performance results
- Automatic recommendation strategies

**Prerequisites:** 02_basic_usage.py, understanding of cost model

**Run:**
```bash
# Auto-tune for your tree
python 05_parameter_tuning.py --tree_file ../data/tree_example.txt --mode auto

# Manual parameter sweep
python 05_parameter_tuning.py --tree_file ../data/tree_example.txt --mode sweep \
    --alpha_range 0.5,0.8 --beta_range 0.2,0.5

# Analyze tree and get recommendations
python 05_parameter_tuning.py --tree_file ../data/tree_example.txt --mode analyze
```

**Modes:**
- `auto`: Automatic tuning (recommended)
- `sweep`: Grid search over specified ranges
- `analyze`: Analyze tree and suggest parameters

**Warning:** Parameter sweeps can take 30+ minutes for large trees.

---

### 06_profiling.py
**Level**: 🔴 Advanced | **Time**: 🕐 Moderate

Performance profiling tools for detailed analysis.

**What you'll learn:**
- How to profile with Triton
- How to profile with CUDA profiler
- Memory usage tracking
- Bottleneck identification
- How to interpret profiling results

**Prerequisites:** 02_basic_usage.py

**Run:**
```bash
# Quick profiling
python 06_profiling.py --tree_file ../data/tree_example.txt

# Detailed profiling with CUDA profiler
python 06_profiling.py --tree_file ../data/tree_example.txt --cuda_profile

# Memory tracking
python 06_profiling.py --tree_file ../data/tree_example.txt --track_memory
```

**Profiling modes:**
- Triton benchmarking (default)
- CUDA profiler integration
- Memory tracking
- Bottleneck analysis

---

## Phase 4: Production Usage

### 07_batch_processing.py
**Level**: 🟡 Intermediate | **Time**: 🕐 Moderate

Efficient batch handling and throughput optimization.

**What you'll learn:**
- Request distribution strategies
- Batch size optimization
- Load balancing across tree paths
- Throughput maximization

**Prerequisites:** 02_basic_usage.py

**Run:**
```bash
# Test with 16 requests
python 07_batch_processing.py --tree_file ../data/tree_example.txt --num_requests 16

# Find optimal batch size
python 07_batch_processing.py --tree_file ../data/tree_example.txt --optimize_batch

# Test different distribution strategies
python 07_batch_processing.py --tree_file ../data/tree_example.txt \
    --distribution uniform  # or: random, weighted
```

**Distribution strategies:**
- `uniform`: Equal requests per path
- `random`: Random assignment
- `weighted`: Based on path cost

---

### 08_integration_template.py
**Level**: 🟡 Intermediate | **Time**: 🕐 Moderate

Production-ready wrapper class for integrating FastTree into your project.

**What you'll learn:**
- How to wrap FastTree in a clean API
- Error handling and fallback strategies
- Configuration management
- Unit testing

**Prerequisites:** Understanding of all previous examples

**Run:**
```bash
# Run the integration template with tests
python 08_integration_template.py

# Run with custom config
python 08_integration_template.py --config config.json
```

**Features:**
- `FastTreeAttention` wrapper class
- Automatic fallback to standard attention
- Configuration from dictionary/JSON
- Comprehensive error handling
- Unit tests included
- **Use this as a starting point for your project!**

---

## Utility Scripts

### utils/tree_builder.py

Helper classes for building trees programmatically.

**Classes:**
- `TreeBuilder`: Fluent API for tree construction
- `TreeValidator`: Validation utilities
- `TreeVisualizer`: ASCII visualization

**Usage:**
```python
from utils.tree_builder import TreeBuilder

tree = TreeBuilder() \
    .add_node(seqlen=100, parent=-1) \
    .add_child(parent_id=0, seqlen=50) \
    .add_child(parent_id=0, seqlen=50) \
    .build()
```

---

### utils/config_helper.py

Configuration management utilities.

**Features:**
- Hardware-specific presets (H100, A100, V100)
- Automatic hardware detection
- Parameter recommendations based on tree characteristics
- Configuration validation

**Usage:**
```python
from utils.config_helper import ConfigPresets, recommend_config

# Get H100 preset
config = ConfigPresets.H100_DEFAULT

# Get recommendations for your tree
config = recommend_config(tree_info, hardware="H100")
```

---

### utils/benchmark_utils.py

Benchmarking and performance analysis utilities.

**Features:**
- `Timer` context manager
- Statistical benchmarking with quantiles
- Memory tracking
- Multi-implementation comparison
- Result formatting

**Usage:**
```python
from utils.benchmark_utils import Timer, benchmark_function

# Simple timing
with Timer() as t:
    fasttree_decode(...)
print(f"Elapsed: {t.elapsed:.3f}s")

# Statistical benchmarking
median_ms = benchmark_function(lambda: fasttree_decode(...))
```

---

## Testing

### test_all_examples.py

Automated testing for all examples.

**Run:**
```bash
python test_all_examples.py
```

This will run all 8 examples and report pass/fail status.

---

## Common Workflows

### Workflow 1: First-Time User

1. Run `01_minimal_example.py` to understand basics
2. Run `02_basic_usage.py` with a sample tree
3. Read [../README_KERNEL.md](../README_KERNEL.md) for concepts
4. Try `03_custom_tree.py` to create your own trees

**Time:** ~30 minutes

---

### Workflow 2: Integrating into Project

1. Understand API with `01_minimal_example.py`
2. Review `08_integration_template.py` wrapper class
3. Copy template and adapt to your needs
4. Use `utils/` helpers for tree building and config

**Time:** ~1 hour

---

### Workflow 3: Performance Optimization

1. Run `02_basic_usage.py --profile` to get baseline
2. Use `05_parameter_tuning.py --mode auto` to find optimal params
3. Run `06_profiling.py` to identify bottlenecks
4. Iterate with different configurations
5. Compare with Flash Attention baseline

**Time:** ~2-3 hours

---

### Workflow 4: GQA Model Deployment

1. Determine your GQA ratio (num_qo_heads // num_kv_heads)
2. Run `04_gqa_example.py --gqa_ratio N` to get recommended config
3. Test with your tree using recommended tile sizes
4. Fine-tune with `05_parameter_tuning.py` if needed

**Time:** ~1 hour

---

## Prerequisites

### System Requirements
- NVIDIA GPU with CUDA (H100, A100, V100, or similar)
- CUDA 11.8+
- Python 3.8+

### Python Packages
```bash
pip install torch>=2.0.0 triton>=2.0.0
```

### Optional (for some examples)
```bash
pip install matplotlib  # For visualization in 05_parameter_tuning.py
```

---

## File Structure

```
examples/
├── README.md                 (this file)
├── 01_minimal_example.py     (~200 lines)
├── 02_basic_usage.py         (~290 lines)
├── 03_custom_tree.py         (~200 lines)
├── 04_gqa_example.py         (~150 lines)
├── 05_parameter_tuning.py    (~250 lines)
├── 06_profiling.py           (~180 lines)
├── 07_batch_processing.py    (~200 lines)
├── 08_integration_template.py (~300 lines)
├── test_all_examples.py      (~100 lines)
└── utils/
    ├── tree_builder.py       (~200 lines)
    ├── config_helper.py      (~150 lines)
    └── benchmark_utils.py    (~200 lines)
```

---

## Troubleshooting

### "File not found" errors
Make sure you're running examples from the `examples/` directory:
```bash
cd examples
python 01_minimal_example.py
```

### "CUDA out of memory"
- Try smaller trees or reduce batch size
- Adjust `KV_SPLIT_SIZES` to `[512, 64]`
- Check GPU memory with `nvidia-smi`

### "No module named 'fasttree'"
Examples automatically add parent directory to path. If this fails:
```bash
export PYTHONPATH=/path/to/FastTree-Artifact/kernel_bench:$PYTHONPATH
```

### Performance worse than expected
- Check that tree has sufficient sharing (>30% of tokens)
- Run `05_parameter_tuning.py` to find optimal config
- Verify batch size is in optimal range (4-32)
- Check GPU utilization with `nvidia-smi`

---

## Additional Resources

- **Main Documentation**: [../README_KERNEL.md](../README_KERNEL.md)
- **API Reference**: [../docs/API_REFERENCE.md](../docs/API_REFERENCE.md)
- **Tree Format**: [../docs/TREE_FORMAT.md](../docs/TREE_FORMAT.md)
- **Configuration Guide**: [../docs/CONFIGURATION_GUIDE.md](../docs/CONFIGURATION_GUIDE.md)
- **Performance Guide**: [../docs/PERFORMANCE_GUIDE.md](../docs/PERFORMANCE_GUIDE.md)

---

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Review the relevant documentation
3. Run `test_all_examples.py` to verify setup
4. File an issue at: https://github.com/anthropics/claude-code/issues

---

**Ready to start?** Run `python 01_minimal_example.py` and you'll be up and running in under 5 minutes!
