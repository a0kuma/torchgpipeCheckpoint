# Checkpointing Strategies in GPipe

This document explains the two different checkpointing strategies available in torchgpipe: **micro-batch-based checkpointing** and **partition-based checkpointing**.

## Table of Contents

- [Understanding the Relationship Between Checkpointing and Micro-batches](#understanding-the-relationship-between-checkpointing-and-micro-batches)
- [What is Checkpointing?](#what-is-checkpointing)
- [Micro-batch-based Checkpointing](#micro-batch-based-checkpointing)
- [Partition-based Checkpointing](#partition-based-checkpointing)
- [Comparison](#comparison)
- [When to Use Each Strategy](#when-to-use-each-strategy)
- [Examples](#examples)

---

## Understanding the Relationship Between Checkpointing and Micro-batches

**Important Clarification**: Checkpointing and micro-batches **ARE** related in GPipe's design. This is not arbitrary!

### Why They're Related

In GPipe's pipeline parallelism:

1. **Multiple micro-batches exist simultaneously** in different pipeline stages
2. **Each occupies memory** for its activations
3. **They progress through the pipeline at different times**
4. **Some micro-batches finish forward pass much earlier than others**

This creates a **memory pressure pattern over time**:

```
Forward Pass Timeline (4 micro-batches, 3 partitions):
─────────────────────────────────────────────────────
Time  MB0     MB1     MB2     MB3     Memory Status
─────────────────────────────────────────────────────
T0    →P0                             MB0 activations in memory
T1    →P1     →P0                     MB0+MB1 activations
T2    →P2     →P1     →P0             MB0+MB1+MB2 activations (PEAK!)
T3    DONE    →P2     →P1     →P0     MB0 done, MB1+MB2+MB3 activations
T4            DONE    →P2     →P1     MB1 done, MB2+MB3 activations
T5                    DONE    →P2     MB2 done, MB3 activations
T6                            DONE    MB3 done

Backward Pass Begins:
T7                            ←P2     MB3 needs activations NOW
T8                    ←P1     ←P2     MB2+MB3 need activations
...
```

**Key Insight**: MB0-MB2 finish forward pass long before backward starts. **Checkpointing them saves memory during the gap**. MB3 finishes right before backward - checkpointing it provides **no benefit** since activations are needed immediately.

### The Core Principle

Micro-batch-based checkpointing leverages the **temporal gap** between when a micro-batch finishes forward propagation and when it needs activations for backward propagation. Earlier micro-batches have longer gaps → more memory savings from checkpointing.

This is why the strategies are defined by micro-batch index:
- **`'except_last'`**: Checkpoint MB0 through MB(n-2), skip MB(n-1) - optimal memory/compute trade-off
- **`'always'`**: Checkpoint all including MB(n-1) - wastes computation on the last one
- **`'never'`**: Don't checkpoint any - uses maximum memory

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

**Important**: Checkpointing and micro-batches ARE related in GPipe's pipeline parallelism design. This is not arbitrary - it's a fundamental part of how GPipe optimizes memory usage during pipelined execution.

### How It Works

When you split a mini-batch into multiple micro-batches (controlled by the `chunks` parameter), GPipe processes them sequentially through the pipeline. Micro-batch-based checkpointing decides whether to checkpoint based on the micro-batch index.

**Key Insight**: Different micro-batches are at different stages in the pipeline at any given time. This creates opportunities to optimize which ones need checkpointing.

### Code Implementation

The decision happens in **two locations**:

#### 1. Configuration Stage (`torchgpipe/gpipe.py`, lines 442-448)

```python
if isinstance(self.checkpoint, str):
    # Original micro-batch-based checkpointing for backward compatibility
    checkpoint_stop = {
        'always': m,           # m = number of micro-batches
        'except_last': m - 1,  # Checkpoint all except the last
        'never': 0,            # Don't checkpoint any
    }[self.checkpoint]
    checkpoint_partitions = None  # Use micro-batch-based checkpointing
```

The `checkpoint_stop` variable is set to a number indicating how many micro-batches (counting from 0) should be checkpointed.

#### 2. Execution Stage (`torchgpipe/pipeline.py`, lines 189-203)

```python
for i, j in schedule:
    # i = micro-batch index (0, 1, 2, ...)
    # j = partition index (0, 1, 2, ...)
    
    batch = batches[i]
    partition = partitions[j]
    
    # Determine whether checkpointing or not
    if checkpoint_partitions is not None:
        checkpoint = (j in checkpoint_partitions)  # Partition-based
    else:
        checkpoint = (i < checkpoint_stop)          # Micro-batch-based ← HERE
    
    if checkpoint:
        # Use checkpointing (save memory, recompute during backward)
        chk = Checkpointing(function, batch)
        task = Task(streams[j], compute=chk.checkpoint, finalize=chk.recompute)
    else:
        # Don't checkpoint (keep activations in memory)
        task = Task(streams[j], compute=compute, finalize=None)
```

The condition `i < checkpoint_stop` determines if micro-batch `i` should be checkpointed.

### Why Not Checkpoint All Micro-batches?

You might wonder: "Why not checkpoint all micro-batches if it saves memory?"

The answer lies in **pipeline parallelism dynamics**:

1. **The last micro-batch is already at the end of the pipeline** when backpropagation starts
2. **No other micro-batches are waiting behind it** in the forward pass
3. **Its activations will be needed immediately** for backward pass
4. **Checkpointing would waste computation** since there's no memory pressure benefit

#### Detailed Explanation

In pipeline parallelism with `chunks=4`:

```
Time →
T0:  MB0→P0
T1:  MB1→P0   MB0→P1
T2:  MB2→P0   MB1→P1   MB0→P2
T3:  MB3→P0   MB2→P1   MB1→P2   MB0→P3
T4:           MB3→P1   MB2→P2   MB1→P3
T5:                    MB3→P2   MB2→P3
T6:                             MB3→P3
     ↓ Forward pass ends, backward starts here
T7:                             MB3←P3  (need activations NOW)
T8:                    MB3←P2   MB2←P3
T9:           MB3←P1   MB2←P2   MB1←P3
...
```

**MB3 (last micro-batch)**:
- Finishes forward pass at T6
- Needs activations immediately at T7 for backward
- **Checkpointing it would require immediate recomputation** (no benefit!)
- Memory saved = 0 (activations needed right away)

**MB0-MB2 (earlier micro-batches)**:
- Finish forward earlier (T3, T4, T5)
- Don't need activations until later (T10+)
- **Checkpointing saves memory during the gap**
- Other micro-batches can use that memory

This is why `'except_last'` is the **optimal default** for pipeline parallelism.

### String Modes

The checkpoint parameter accepts three string values:

- **`'always'`**: Checkpoint **all** micro-batches (maximum memory saving, most recomputation)
- **`'except_last'`** (default): Checkpoint all micro-batches **except the last one** (optimal for pipelines)
- **`'never'`**: **Don't checkpoint** any micro-batches (maximum memory usage, no recomputation)

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

Why?
- MB 0-2: Finish early, don't need activations until much later → checkpoint saves memory
- MB 3: Finishes last, needs activations immediately → checkpointing wastes computation
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
