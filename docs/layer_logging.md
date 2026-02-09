# How to Log Which Layer is Currently Running in the Compute Function

This guide explains how to track and log which layer (partition) is currently being executed in the `Pipeline.compute()` function of torchgpipe.

## Understanding the Compute Function

The `compute` function in `torchgpipe/pipeline.py` is responsible for executing micro-batches across different partitions (layers) of your model in parallel. Each partition represents a group of consecutive layers that run on a single device.

### Key Parameters

- **schedule**: A list of tuples `(i, j)` where:
  - `i` is the micro-batch index (which micro-batch is being processed)
  - `j` is the partition index (which layer/partition is being used)
- **partitions**: The actual neural network layers/modules being executed

## Built-in Logging

The `compute` function already includes logging functionality. By default, logging is configured at the INFO level in `pipeline.py`:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

### Existing Log Messages

The function logs information about:
- Which micro-batch (`i`) is being processed
- Which partition/layer (`j`) is executing it
- Whether checkpointing is enabled for that micro-batch

## How to Enable Layer Logging

### Method 1: Use Existing Logging (Recommended)

The easiest way to see which layer is running is to enable Python logging at the appropriate level:

```python
import logging

# Set logging level to INFO to see checkpoint information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Or set logging level to DEBUG to see all layer executions
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

**Example output:**
```
2024-02-09 12:00:00 - torchgpipe.pipeline - INFO - Micro-batch 0 (partition 0): checkpoint_stop=2, using_checkpointing=False
2024-02-09 12:00:01 - torchgpipe.pipeline - DEBUG - Micro-batch 1 (partition 1): checkpoint_stop=2, using_checkpointing=True
```

### Method 2: Add Custom Logging

If you need more detailed logging, you can modify the `compute` function to add additional log statements. Here's what the relevant section looks like:

```python
for i, j in schedule:
    batch = batches[i]
    partition = partitions[j]
    
    # Log which layer is starting
    logger.info(f"Starting execution: micro-batch {i}, partition {j}")
    
    # ... rest of the computation logic ...
```

### Method 3: Add a Custom Logger to Your Training Script

You can also add logging in your own training code:

```python
import logging
from torchgpipe import GPipe

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

# Create your model
model = nn.Sequential(layer1, layer2, layer3, layer4)
model = GPipe(model, balance=[1, 1, 1, 1], chunks=8)

# Training loop - logs will automatically appear
for input in data_loader:
    output = model(input)
```

## Understanding the Output

When you enable logging, you'll see messages like:

```
Micro-batch 0 (partition 0): checkpoint_stop=2, using_checkpointing=True
```

This tells you:
- **Micro-batch 0**: The first micro-batch is being processed
- **partition 0**: It's being processed by the first partition (layer group)
- **checkpoint_stop=2**: Checkpointing is enabled for micro-batches 0 and 1
- **using_checkpointing=True**: This specific micro-batch is using checkpointing

## Log Levels

torchgpipe uses different log levels for different information:

- **DEBUG**: Logs all micro-batch/partition executions that use checkpointing
- **INFO**: Logs micro-batch/partition executions that do NOT use checkpointing
- **WARNING/ERROR**: Used for issues or exceptions

To see all layer executions, use `logging.DEBUG`. For production use, `logging.INFO` provides important information without being too verbose.

## Advanced: Logging to File

To save logs to a file for later analysis:

```python
import logging

# Create a file handler
file_handler = logging.FileHandler('torchgpipe_execution.log')
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

## Example: Complete Training Script with Logging

```python
import torch
import torch.nn as nn
import logging
from torchgpipe import GPipe

# Configure logging first
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Define your model
class MyModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

# Create GPipe model
model = MyModel()
model = GPipe(model, balance=[2, 2, 1], chunks=4)

# Training loop
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    # Your training data
    inputs = torch.randn(32, 100)
    targets = torch.randint(0, 10, (32,))
    
    optimizer.zero_grad()
    outputs = model(inputs)  # Logs will appear here showing layer execution
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item()}")
```

## Summary

To know which layer is currently running in the `compute` function:

1. **Simplest approach**: Enable logging at the INFO or DEBUG level
2. **The logging already exists**: The code in `pipeline.py` already logs partition/layer execution
3. **Key information logged**: Micro-batch index, partition index, and checkpointing status
4. **Customize as needed**: You can add more detailed logging if the built-in logs aren't sufficient

The partition index `j` directly corresponds to which group of layers is being executed. If you split your model with `balance=[2, 2, 1]`, then:
- Partition 0 = first 2 layers
- Partition 1 = next 2 layers  
- Partition 2 = last 1 layer

By monitoring the log output, you can track exactly which partition (layer group) is processing which micro-batch at any given time.
