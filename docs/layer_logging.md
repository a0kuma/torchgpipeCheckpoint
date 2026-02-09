# How to Log Which Layer is Currently Running in the Compute Function

This guide explains how to track and log which individual **layer** (not partition) is currently being executed in the `Pipeline.compute()` function of torchgpipe.

## Understanding Layers vs Partitions

**Important distinction:**
- A **layer** is a single neural network module (e.g., `nn.Linear`, `nn.ReLU`, `nn.Conv2d`)
- A **partition** is a group of consecutive layers that run together on the same device
- When you split a model with `balance=[2, 3, 1]`, you're creating 3 partitions:
  - Partition 0 contains the first 2 layers
  - Partition 1 contains the next 3 layers
  - Partition 2 contains the last 1 layer

The `compute` function processes micro-batches through partitions, but each partition executes multiple layers sequentially.

## The Compute Function

The `compute` function in `torchgpipe/pipeline.py` is responsible for executing micro-batches across different partitions of your model in parallel. Each partition executes as a unit, but internally it runs multiple layers sequentially.

### Key Parameters

- **schedule**: A list of tuples `(i, j)` where:
  - `i` is the micro-batch index (which micro-batch is being processed)
  - `j` is the partition index (which partition is being used)
- **partitions**: A list of `nn.Sequential` modules, each containing multiple layers

## Built-in Partition-Level Logging

The `compute` function already includes partition-level logging functionality. By default, logging is configured at the INFO level in `pipeline.py`:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

### Existing Partition-Level Log Messages

The function logs information about:
- Which micro-batch (`i`) is being processed
- Which partition (`j`) is executing it
- Whether checkpointing is enabled for that micro-batch

**This only tells you which partition is running, not which individual layer.**

## How to Enable Layer-Level Logging

To track individual layer execution within partitions, use the `register_layer_logging_hooks` function:

### Method 1: Using register_layer_logging_hooks (Recommended for Layer Tracking)

```python
import logging
from torchgpipe import GPipe, register_layer_logging_hooks

# Set logging level to DEBUG to see individual layer executions
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create and wrap your model
model = nn.Sequential(layer1, layer2, layer3, layer4, layer5)
model = GPipe(model, balance=[2, 3], chunks=4)

# Register hooks to log individual layer execution
hooks = register_layer_logging_hooks(model.partitions)

# Training/inference - logs will show individual layers
for input in data_loader:
    output = model(input)

# Clean up hooks when done (optional but recommended)
for hook in hooks:
    hook.remove()
```

**Example output:**
```
INFO - Micro-batch 0 (partition 0): checkpoint_stop=0, using_checkpointing=False
DEBUG - Executing layer '0' in partition 0: Linear
DEBUG - Executing layer '1' in partition 0: ReLU
INFO - Micro-batch 0 (partition 1): checkpoint_stop=0, using_checkpointing=False
DEBUG - Executing layer '2' in partition 1: Linear
DEBUG - Executing layer '3' in partition 1: ReLU
DEBUG - Executing layer '4' in partition 1: Linear
```

### Method 2: Partition-Level Logging Only (No Layer Details)

If you only need to know which partition is running (not individual layers), simply enable logging:

```python
import logging

# Set logging level to INFO to see partition execution
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from torchgpipe import GPipe

model = nn.Sequential(layer1, layer2, layer3, layer4)
model = GPipe(model, balance=[2, 2], chunks=4)

# Logs will automatically appear showing partition execution
output = model(input)
```

**Example output (partition-level only):**
```
INFO - Micro-batch 0 (partition 0): checkpoint_stop=0, using_checkpointing=False
INFO - Micro-batch 1 (partition 0): checkpoint_stop=0, using_checkpointing=False
INFO - Micro-batch 0 (partition 1): checkpoint_stop=0, using_checkpointing=False
```

## Understanding the Output

### Partition-Level Logs

When you see:
```
Micro-batch 0 (partition 1): checkpoint_stop=2, using_checkpointing=False
```

This tells you:
- **Micro-batch 0**: The first micro-batch is being processed
- **partition 1**: It's being processed by the second partition (0-indexed)
- **checkpoint_stop=2**: Checkpointing is enabled for micro-batches 0 and 1
- **using_checkpointing=False**: This specific micro-batch is NOT using checkpointing

### Layer-Level Logs

When you see:
```
Executing layer '2' in partition 1: Linear
```

This tells you:
- **layer '2'**: The layer's index/name within the original model
- **partition 1**: This layer is part of the second partition
- **Linear**: The type of layer being executed (`nn.Linear`, `nn.ReLU`, etc.)

## Log Levels

torchgpipe uses different log levels:

- **DEBUG**: Individual layer executions (only when using `register_layer_logging_hooks`)
- **INFO**: Partition-level execution logs
- **WARNING/ERROR**: Issues or exceptions

To see layer execution, use `logging.DEBUG`. For partition-level info only, use `logging.INFO`.

## Advanced: Logging to File

To save logs to a file for later analysis:

```python
import logging

# Create a file handler
file_handler = logging.FileHandler('layer_execution.log')
file_handler.setLevel(logging.DEBUG)

# Create a console handler for errors only
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

# Create a formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Get the torchgpipe logger and add handlers
logger = logging.getLogger('torchgpipe')
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
```

## Example: Complete Training Script with Layer Logging

```python
import torch
import torch.nn as nn
import logging
from torchgpipe import GPipe, register_layer_logging_hooks

# Configure logging first - DEBUG level to see individual layers
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Define your model with clear layer structure
model = nn.Sequential(
    nn.Linear(100, 100),  # Layer 0
    nn.ReLU(),            # Layer 1
    nn.Linear(100, 100),  # Layer 2
    nn.ReLU(),            # Layer 3
    nn.Linear(100, 10)    # Layer 4
)

# Create GPipe model - balance=[2, 2, 1] means:
# - Partition 0: layers 0-1 (Linear + ReLU)
# - Partition 1: layers 2-3 (Linear + ReLU)
# - Partition 2: layer 4 (Linear)
model = GPipe(model, balance=[2, 2, 1], chunks=4)

# Register layer logging hooks
hooks = register_layer_logging_hooks(model.partitions)

# Training loop
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    # Your training data
    inputs = torch.randn(32, 100)
    targets = torch.randint(0, 10, (32,))
    
    optimizer.zero_grad()
    outputs = model(inputs)  # Layer-by-layer logs will appear here
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item()}")

# Clean up hooks
for hook in hooks:
    hook.remove()
```

## Summary

To know which **individual layer** is currently running in the `compute` function:

1. **Use `register_layer_logging_hooks(model.partitions)`** - This registers forward hooks on each layer
2. **Set logging to DEBUG level** - `logging.basicConfig(level=logging.DEBUG)`
3. **Run your model** - You'll see logs for both partitions and individual layers
4. **Clean up when done** - Remove hooks with `hook.remove()` for each hook

### Key Points:

- **Partitions** are groups of layers, not individual layers
- The partition index `j` tells you which group of layers is executing
- To see **individual layers**, you must use `register_layer_logging_hooks`
- Layer names/indices correspond to the original model's layer structure
- Each partition executes its layers sequentially, even though partitions run in parallel

By using layer-level logging, you can track exactly which layer is processing which micro-batch at any given time, giving you fine-grained visibility into pipeline execution.
