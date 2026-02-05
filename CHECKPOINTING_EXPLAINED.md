# Checkpointing Strategies in GPipe

This document explains the two different checkpointing strategies available in torchgpipe: **micro-batch-based checkpointing** and **partition-based checkpointing**.

## Table of Contents

- [What is Checkpointing?](#what-is-checkpointing)
- [Micro-batch-based Checkpointing](#micro-batch-based-checkpointing)
- [Partition-based Checkpointing](#partition-based-checkpointing)
- [Comparison](#comparison)
- [When to Use Each Strategy](#when-to-use-each-strategy)
- [Examples](#examples)

---

## What is Checkpointing?

Checkpointing is a memory-optimization technique used in deep learning to reduce peak memory usage during training. Instead of storing all intermediate activations during the forward pass, checkpointing discards them and recomputes them during the backward pass when needed.

In the context of GPipe:
- **Without checkpointing**: All intermediate activations are stored in memory throughout training
- **With checkpointing**: Selected intermediate activations are discarded and recomputed during backpropagation

This trade-off exchanges **memory** for **computation time**: you save memory but spend extra time recomputing activations.

---

## Micro-batch-based Checkpointing

### Overview

**Micro-batch-based checkpointing** controls checkpointing based on which **micro-batch** is being processed. This is the original behavior in torchgpipe and is controlled by string modes.

### How It Works

When you split a mini-batch into multiple micro-batches (controlled by the `chunks` parameter), GPipe processes them sequentially through the pipeline. Micro-batch-based checkpointing decides whether to checkpoint based on the micro-batch index.

### String Modes

The checkpoint parameter accepts three string values:

- **`'always'`**: Checkpoint **all** micro-batches
- **`'except_last'`** (default): Checkpoint all micro-batches **except the last one**
- **`'never'`**: **Don't checkpoint** any micro-batches

### Visualization

```
Model: [Layer0 | Layer1 | Layer2 | Layer3]
Chunks: 4 micro-batches
Checkpoint mode: 'except_last'

Processing timeline:
┌─────────────────────────────────────────┐
│ Micro-batch 0 → Checkpointed ✓         │
│ Micro-batch 1 → Checkpointed ✓         │
│ Micro-batch 2 → Checkpointed ✓         │
│ Micro-batch 3 → NOT checkpointed ✗     │
└─────────────────────────────────────────┘
```

### Example Usage

```python
from torchgpipe import GPipe
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
)

# Checkpoint all micro-batches except the last
gpipe = GPipe(
    model,
    balance=[1, 1, 1, 1],
    devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
    chunks=8,
    checkpoint='except_last'  # String mode
)
```

### When It's Applied

- Decision is made per micro-batch
- Applied uniformly across **all partitions**
- Same checkpointing behavior for every layer in every partition

---

## Partition-based Checkpointing

### Overview

**Partition-based checkpointing** (also called **layer-based checkpointing**) controls checkpointing based on which **partition** or **layer** is being processed. This gives you fine-grained control over which parts of your model should use checkpointing.

### How It Works

Instead of controlling checkpointing by micro-batch, you specify exactly which layers (or partitions) in your model should be checkpointed. The checkpoint parameter accepts a list, tuple, or set of layer indices.

### Layer Indices

You specify layer indices from the **original sequential module**:

```python
model = nn.Sequential(
    layer0,  # index 0
    layer1,  # index 1
    layer2,  # index 2
    layer3,  # index 3
    # ... etc
)
```

These layer indices are automatically converted to partition indices based on your `balance` configuration.

### Visualization

```
Model: 10 layers [0,1,2,3,4,5,6,7,8,9]
Balance: [3, 3, 4]

Partition mapping:
┌──────────────────────────────────────────┐
│ Partition 0: layers [0, 1, 2]           │
│ Partition 1: layers [3, 4, 5]           │
│ Partition 2: layers [6, 7, 8, 9]        │
└──────────────────────────────────────────┘

Checkpoint specification: [3, 5, 7]

Result:
┌──────────────────────────────────────────┐
│ Layer 3 → in Partition 1 → Checkpoint ✓ │
│ Layer 5 → in Partition 1 → Checkpoint ✓ │
│ Layer 7 → in Partition 2 → Checkpoint ✓ │
│                                          │
│ Partitions checkpointed: {0, 1, 2}      │
│ (includes borders 0 and 2)               │
└──────────────────────────────────────────┘
```

### Example Usage

```python
from torchgpipe import GPipe
import torch.nn as nn

# 10-layer model
model = nn.Sequential(*[nn.Linear(128, 128) for _ in range(10)])

# Split into 3 partitions
# Partition 0: layers 0,1,2
# Partition 1: layers 3,4,5  
# Partition 2: layers 6,7,8,9

# Checkpoint specific layers
gpipe = GPipe(
    model,
    balance=[3, 3, 4],
    devices=['cuda:0', 'cuda:1', 'cuda:2'],
    chunks=4,
    checkpoint=[3, 5, 7]  # List of layer indices
)
```

### Border Partitions

**Important**: The first and last partitions (borders) are **always checkpointed** automatically, regardless of what you specify. This ensures pipeline correctness.

### When It's Applied

- Decision is made per partition
- Applied to **all micro-batches** passing through that partition
- Different layers can have different checkpointing behavior

---

## Comparison

| Feature | Micro-batch-based | Partition-based |
|---------|-------------------|-----------------|
| **Control dimension** | Which micro-batch | Which layer/partition |
| **Specified by** | String: 'always', 'except_last', 'never' | List/tuple/set of layer indices: [3, 5, 7] |
| **Applies to** | All partitions uniformly | Specific partitions only |
| **Granularity** | Coarse (all or none per micro-batch) | Fine (specific layers) |
| **Use case** | General memory reduction | Selective checkpointing of expensive layers |
| **Original GPipe** | ✓ Yes | ✗ No (extension) |
| **Backward compatible** | ✓ Always | ✓ When using strings |

---

## When to Use Each Strategy

### Use Micro-batch-based Checkpointing When:

1. **Simple memory reduction**: You want a straightforward way to reduce memory
2. **Uniform model**: All layers have similar computational cost
3. **Default behavior**: The 'except_last' mode works well for most cases
4. **Original GPipe semantics**: You want to match the paper's behavior

**Example scenario**: Training a standard ResNet or transformer where all layers are roughly equivalent.

### Use Partition-based Checkpointing When:

1. **Heterogeneous model**: Some layers are much more expensive than others
2. **Selective optimization**: You want to checkpoint only specific expensive operations
3. **Fine-tuned control**: You need precise control over the memory/compute trade-off
4. **Known bottlenecks**: You've identified specific layers that cause memory issues

**Example scenario**: A model with expensive attention layers at specific positions, where you want to checkpoint only those layers.

---

## Examples

### Example 1: Micro-batch-based (Simple)

```python
from torchgpipe import GPipe
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
)

# Use default micro-batch-based checkpointing
gpipe = GPipe(
    model,
    balance=[1, 1, 1, 1],
    chunks=8,
    checkpoint='except_last'  # Checkpoint 7 out of 8 micro-batches
)
```

### Example 2: Partition-based (Selective)

```python
from torchgpipe import GPipe
import torch.nn as nn

# Model with 16 layers
model = nn.Sequential(*[
    nn.Linear(256, 256) if i % 4 != 3 else nn.TransformerEncoderLayer(256, 8)
    for i in range(16)
])

# Only checkpoint the expensive TransformerEncoderLayers at positions 3, 7, 11, 15
gpipe = GPipe(
    model,
    balance=[4, 4, 4, 4],
    chunks=4,
    checkpoint=[3, 7, 11, 15]  # Checkpoint specific expensive layers
)
```

### Example 3: Mixed Architecture

```python
from torchgpipe import GPipe
import torch.nn as nn

# 10 layers with varying complexity
model = nn.Sequential(
    nn.Linear(128, 256),      # 0: Simple
    nn.Linear(256, 512),      # 1: Simple
    nn.Linear(512, 512),      # 2: Simple
    nn.TransformerBlock(...), # 3: Expensive ← Checkpoint this
    nn.Linear(512, 512),      # 4: Simple
    nn.TransformerBlock(...), # 5: Expensive ← Checkpoint this
    nn.Linear(512, 512),      # 6: Simple
    nn.Linear(512, 256),      # 7: Simple
    nn.Linear(256, 128),      # 8: Simple
    nn.Linear(128, 10),       # 9: Simple
)

# Balance: [3, 3, 4] creates 3 partitions
# Partition 0: [0,1,2]
# Partition 1: [3,4,5]  ← Contains both expensive layers
# Partition 2: [6,7,8,9]

gpipe = GPipe(
    model,
    balance=[3, 3, 4],
    chunks=4,
    checkpoint=[3, 5]  # Only checkpoint the expensive transformer blocks
)
```

---

## Technical Details

### Implementation

From the code perspective:

**Micro-batch-based** (`checkpoint_stop`):
```python
# In pipeline.py
checkpoint = (i < checkpoint_stop)  # i = micro-batch index
```

**Partition-based** (`checkpoint_partitions`):
```python
# In pipeline.py  
checkpoint = (j in checkpoint_partitions)  # j = partition index
```

### Conversion Process

When you specify layer indices, they are automatically converted to partition indices:

```python
# User specifies
checkpoint = [3, 5, 7]

# Internally converts to partition indices based on balance
# balance = [3, 3, 4]
# → Partitions: {1, 2} (plus borders 0, 2)
```

See `layer_indices_to_partitions()` in `torchgpipe/gpipe.py` for implementation details.

---

## References

- Original GPipe paper: [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- PyTorch checkpointing: [torch.utils.checkpoint](https://pytorch.org/docs/stable/checkpoint.html)
- Implementation notes: [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)
- Custom checkpoint examples: [EXAMPLE_CUSTOM_CHECKPOINT.md](EXAMPLE_CUSTOM_CHECKPOINT.md)
