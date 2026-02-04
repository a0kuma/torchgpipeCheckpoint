# Custom Checkpoint Example

This example demonstrates how to use the new layer-based checkpointing feature in GPipe.

## Original String-based Modes (Backward Compatible)

The traditional way to specify checkpointing:

```python
from torchgpipe import GPipe

model = nn.Sequential(...)

# Checkpoint all micro-batches
gpipe = GPipe(model, balance=[...], checkpoint='always')

# Checkpoint all micro-batches except the last (default)
gpipe = GPipe(model, balance=[...], checkpoint='except_last')

# Disable checkpointing
gpipe = GPipe(model, balance=[...], checkpoint='never')
```

## New Layer-based Checkpointing

Specify which layers in the original sequential module should be checkpointed:

```python
from torchgpipe import GPipe

# Create a model with 10 layers
model = nn.Sequential(*[nn.Linear(10, 10) for _ in range(10)])

# Split into 3 partitions: [3, 3, 4]
# - Partition 0: layers 0, 1, 2
# - Partition 1: layers 3, 4, 5
# - Partition 2: layers 6, 7, 8, 9

# Checkpoint layers 3, 5, and 7
# This will checkpoint partitions 1 and 2 (which contain these layers)
# Border partitions (0 and 2) are automatically checkpointed
gpipe = GPipe(
    model,
    balance=[3, 3, 4],
    devices=['cpu']*3,
    chunks=4,
    checkpoint=[3, 5, 7]  # Layer indices, not partition indices
)

# Forward pass
output = gpipe(input_data)
```

## Key Features

1. **Backward Compatible**: String modes ('always', 'except_last', 'never') work as before
2. **Flexible Input**: Accept list, tuple, or set of layer indices from the original sequential module
3. **Automatic Border Protection**: First and last partitions are always checkpointed
4. **Layer-to-Partition Conversion**: Layer indices are automatically converted to partition indices based on balance
