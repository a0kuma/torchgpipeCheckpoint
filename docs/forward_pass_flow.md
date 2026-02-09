# Understanding the Forward Pass in Pipeline.compute()

This document explains where and how the forward pass through individual layers happens in the `Pipeline.compute()` function.

## The Question

**"Where is the forward pass code? I'm expecting something like `for xyz_layer in xyz_partition: do_fwd()` but I don't see it!"**

## The Answer

The forward pass through individual layers happens **implicitly** through PyTorch's `nn.Sequential` mechanism. Here's the detailed flow:

## Code Flow

### 1. Task Creation (in `compute()`)

```python
# In pipeline.py, compute() method around line 252-273:

if checkpoint:
    def function(input, partition=partition, ...):
        return partition(input)  # <-- Forward pass starts here!
else:
    def compute(batch, partition=partition, ...):
        return batch.call(partition)  # <-- Or here!
```

### 2. What Happens When `partition(input)` is Called?

When you call `partition(input)`, here's what happens under the hood:

```python
# partition is an nn.Sequential object containing multiple layers
# For example: nn.Sequential(Linear(100, 100), ReLU(), Linear(100, 10))

# When you call partition(input), PyTorch's nn.Sequential does this:
class Sequential(nn.Module):
    def forward(self, input):
        # THIS IS THE IMPLICIT LOOP YOU'RE LOOKING FOR!
        for module in self:
            input = module(input)  # Calls each layer's forward()
        return input
```

So `partition(input)` automatically:
1. Iterates through each layer in the partition
2. Calls `layer.forward(input)` for each layer
3. Passes the output of one layer as input to the next
4. Returns the final output

### 3. Where Layers Execute (Worker Thread)

The actual execution happens in the worker thread:

```python
# In worker.py, worker() function around line 82:

def worker(in_queue, out_queue, device, grad_mode):
    while True:
        task = in_queue.get()
        try:
            batch = task.compute()  # <-- Executes the partition here!
            # task.compute() calls the function we defined above
            # which calls partition(input)
            # which loops through all layers
        except Exception:
            # handle error
```

## Visual Flow Diagram

```
compute() creates task
    ↓
task sent to worker thread via in_queue
    ↓
worker thread calls task.compute()
    ↓
task.compute() calls partition(input)
    ↓
nn.Sequential.__call__() invoked
    ↓
FOR EACH LAYER IN PARTITION:
    ├─ layer_0.forward(input)  → output_0
    ├─ layer_1.forward(output_0) → output_1
    ├─ layer_2.forward(output_1) → output_2
    └─ ...
    ↓
result returned through out_queue
```

## Example with Actual Layers

Let's say you have a model with 5 layers split into 2 partitions:

```python
model = nn.Sequential(
    nn.Linear(100, 100),  # Layer 0
    nn.ReLU(),            # Layer 1
    nn.Linear(100, 100),  # Layer 2
    nn.ReLU(),            # Layer 3
    nn.Linear(100, 10)    # Layer 4
)

# Split into partitions with balance=[2, 3]
# Partition 0: layers 0-1 (Linear, ReLU)
# Partition 1: layers 2-4 (Linear, ReLU, Linear)
model = GPipe(model, balance=[2, 3], chunks=4)
```

When `compute()` processes partition 0:

```python
# This happens in compute() around line 258 or 270:
partition_0(input)  # partition_0 = nn.Sequential(layer_0, layer_1)

# PyTorch's nn.Sequential automatically does:
output = layer_0.forward(input)   # Linear forward pass
output = layer_1.forward(output)  # ReLU forward pass
return output
```

## Why You Don't See an Explicit Loop

The loop `for layer in partition: layer.forward()` is **inside PyTorch's nn.Sequential implementation**, not in torchgpipe code. When you call a Sequential object like a function, PyTorch handles the iteration automatically.

## How to See Individual Layer Execution

If you want to see which individual layers are executing, use the `register_layer_logging_hooks()` function:

```python
from torchgpipe import GPipe, register_layer_logging_hooks
import logging

logging.basicConfig(level=logging.DEBUG)

model = GPipe(model, balance=[2, 3], chunks=4)
hooks = register_layer_logging_hooks(model.partitions)

# Now when you run the model, you'll see DEBUG logs like:
# DEBUG: Executing layer '0' in partition 0: Linear
# DEBUG: Executing layer '1' in partition 0: ReLU
# DEBUG: Executing layer '2' in partition 1: Linear
# etc.

output = model(input)

for hook in hooks:
    hook.remove()
```

## Summary

**Q: Where is `for layer in partition` code?**

**A: It's in PyTorch's `nn.Sequential.__call__()` method, not in torchgpipe code.**

The forward pass flow:
1. `compute()` creates a task that will call `partition(input)`
2. Worker thread executes the task
3. `partition(input)` invokes PyTorch's `nn.Sequential`
4. `nn.Sequential` loops through its layers automatically
5. Each layer's `forward()` is called in sequence

The key insight: **`partition(input)` is the implicit loop** - it's just hidden inside PyTorch's Sequential implementation!
