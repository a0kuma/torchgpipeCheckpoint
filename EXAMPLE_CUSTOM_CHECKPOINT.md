# Custom Checkpoint Example

This example demonstrates how to use the new partition-based checkpointing feature in GPipe.

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

## New Partition-based Checkpointing

Specify which partitions should be checkpointed:

```python
from torchgpipe import GPipe

# Create a model with 8 partitions
model = nn.Sequential(*[nn.Linear(10, 10) for _ in range(8)])

# Checkpoint only partitions 2, 4, and 6
# Border partitions (0 and 7) are automatically checkpointed
gpipe = GPipe(
    model,
    balance=[1, 1, 1, 1, 1, 1, 1, 1],
    devices=['cpu']*8,
    chunks=4,
    checkpoint=[2, 4, 6]  # List, tuple, or set of partition indices
)

# Forward pass
output = gpipe(input_data)
```

## Key Features

1. **Backward Compatible**: String modes ('always', 'except_last', 'never') work as before
2. **Flexible Input**: Accept list, tuple, or set of partition indices
3. **Automatic Border Protection**: First and last partitions are always checkpointed
4. **Out-of-range Handling**: Invalid partition indices are automatically filtered
