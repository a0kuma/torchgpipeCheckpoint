# How Neural Networks are Partitioned in torchgpipe

This document explains how torchgpipe "cuts" or partitions a neural network into multiple segments for pipeline parallelism.

## Overview

torchgpipe splits a `nn.Sequential` module into multiple partitions, where each partition:
- Contains one or more consecutive layers from the original model
- Runs on a specific device (GPU)
- Processes micro-batches in a pipelined fashion

## The Partitioning Process

### 1. Entry Point: `GPipe.__init__()` 

Located in `torchgpipe/gpipe.py`, the GPipe class constructor takes:

```python
from torchgpipe import GPipe

model = nn.Sequential(a, b, c, d)
model = GPipe(model, balance=[1, 1, 1, 1], chunks=8)
```

**Parameters:**
- `model`: An `nn.Sequential` module containing the layers
- `balance`: A list specifying how many layers go in each partition
- `chunks`: Number of micro-batches for pipeline parallelism

### 2. Core Partitioning Function: `split_module()`

The actual "cutting" happens in the `split_module()` function (lines 71-127 in `gpipe.py`):

```python
def split_module(module: nn.Sequential,
                 balance: Iterable[int],
                 devices: List[torch.device],
                 ) -> Tuple[List[nn.Sequential], List[int], List[torch.device]]:
    """Splits a module into multiple partitions."""
```

#### How it works:

1. **Validates the balance**: Ensures the sum of balance equals the number of layers
   ```python
   if len(module) != sum(balance):
       raise BalanceError('module and sum of balance have different length')
   ```

2. **Iterates through layers**: Groups consecutive layers according to the balance
   ```python
   j = 0
   partitions = []
   layers: NamedModules = OrderedDict()
   
   for name, layer in module.named_children():
       layers[name] = layer
       
       if len(layers) == balance[j]:
           # Group buffered layers as a partition
           partition = nn.Sequential(layers)
           device = devices[j]
           partition.to(device)
           partitions.append(partition)
           
           # Prepare for the next partition
           layers.clear()
           j += 1
   ```

3. **Assigns devices**: Each partition is moved to its designated device

### 3. Example: Cutting a 4-layer Model

Given:
```python
model = nn.Sequential(
    layer_a,  # Layer 0
    layer_b,  # Layer 1
    layer_c,  # Layer 2
    layer_d,  # Layer 3
)

# Split into 4 partitions with 1 layer each
gpipe_model = GPipe(model, balance=[1, 1, 1, 1])
```

Result:
```
Partition 0 (Device 0): [layer_a]
Partition 1 (Device 1): [layer_b]
Partition 2 (Device 2): [layer_c]
Partition 3 (Device 3): [layer_d]
```

Another example with different balance:
```python
# Split into 2 partitions with 2 layers each
gpipe_model = GPipe(model, balance=[2, 2])
```

Result:
```
Partition 0 (Device 0): [layer_a, layer_b]
Partition 1 (Device 1): [layer_c, layer_d]
```

## Automatic Balancing

Instead of manually specifying the balance, torchgpipe provides automatic balancing utilities in `torchgpipe/balance/`:

### 1. Balance by Time (`balance_by_time`)

Located in `torchgpipe/balance/__init__.py` (lines 38-77):

```python
from torchgpipe.balance import balance_by_time

sample = torch.empty(128, 3, 224, 224)
balance = balance_by_time(
    partitions=4,        # Number of GPUs
    module=model,
    sample=sample,
    timeout=1.0
)
```

**How it works:**
1. Profiles the execution time of each layer
2. Uses the block partition algorithm to distribute layers evenly by time

### 2. Balance by Size (`balance_by_size`)

Located in `torchgpipe/balance/__init__.py` (lines 80-156):

```python
from torchgpipe.balance import balance_by_size

balance = balance_by_size(
    partitions=4,
    module=model,
    input=torch.empty(1024, 3, 224, 224),
    chunks=8,
    param_scale=4.0  # For Adam optimizer
)
```

**How it works:**
1. Profiles the memory usage of each layer
2. Uses the block partition algorithm to distribute layers evenly by memory

### 3. Block Partition Algorithm

Located in `torchgpipe/balance/blockpartition.py`:

This implements the "Block Partitions of Sequences" algorithm by Imre Bárány et al. to minimize variance across partitions.

**The algorithm:**
1. Takes a sequence of costs (time or memory per layer)
2. Splits into `k` partitions to minimize the difference between the largest and smallest partition
3. Runs in O(kn³) time complexity

```python
def solve(sequence: List[int], partitions: int = 1) -> List[List[int]]:
    """Splits a sequence into several partitions to minimize variance."""
```

## Pipeline Execution

After partitioning, the `Pipeline` class (in `torchgpipe/pipeline.py`) orchestrates execution:

1. **Micro-batch splitting**: The mini-batch is split into micro-batches
2. **Clock cycles**: Each partition processes micro-batches in a pipelined fashion
3. **Device synchronization**: CUDA streams handle data transfers between devices

### Pipeline Schedule

For `m` micro-batches and `n` partitions, the schedule looks like:

```
Clock | Partition 0 | Partition 1 | Partition 2
------|-------------|-------------|-------------
  0   | batch_0     |             |
  1   | batch_1     | batch_0     |
  2   | batch_2     | batch_1     | batch_0
  3   |             | batch_2     | batch_1
  4   |             |             | batch_2
```

## Key Files Reference

1. **`torchgpipe/gpipe.py`**:
   - `split_module()`: The core partitioning function (lines 71-127)
   - `GPipe.__init__()`: Entry point that calls split_module

2. **`torchgpipe/balance/__init__.py`**:
   - `balance_by_time()`: Automatic balancing by execution time
   - `balance_by_size()`: Automatic balancing by memory usage
   - `balance_cost()`: Helper that calls the block partition solver

3. **`torchgpipe/balance/blockpartition.py`**:
   - `solve()`: Block partition algorithm implementation

4. **`torchgpipe/pipeline.py`**:
   - `Pipeline.run()`: Executes the partitioned model
   - `clock_cycles()`: Generates the pipeline schedule

## Summary

The "cut" or partitioning in torchgpipe happens through:

1. **Manual approach**: User specifies `balance=[...]` parameter
2. **Automatic approach**: Use `balance_by_time()` or `balance_by_size()`
3. **Core mechanism**: The `split_module()` function iterates through layers and groups them into partitions based on the balance
4. **Algorithm**: The block partition algorithm optimally distributes layers to minimize variance

The partitioning is **static** (happens at initialization) and **consecutive** (layers are not reordered, only grouped sequentially).
