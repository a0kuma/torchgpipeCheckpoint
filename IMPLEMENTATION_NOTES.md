# Layer-based Checkpointing Implementation

## Problem Statement
The original request was: "what i want is iterable of Layer indices within the sequential module"

The implementation was initially using **partition indices**, but the user wanted **layer indices** from the original sequential module.

## Solution

### Key Changes

1. **Added `layer_indices_to_partitions()` helper function**
   - Converts layer indices to partition indices based on the balance
   - Issues warnings for out-of-range layer indices
   - Example: With balance=[3,3,4] and layers [3,5,7], returns partitions {1,2}

2. **Updated checkpoint behavior**
   - When checkpoint is a list/tuple/set of integers, they are treated as layer indices
   - Layer indices are converted to partition indices during forward pass
   - Border partitions (0 and n-1) are always checkpointed

3. **Documentation updates**
   - All docstrings and examples now refer to layer indices, not partition indices
   - Error messages clarify "layer indices"

## Examples

### Example 1: Simple case (one layer per partition)

```python
# Model with 16 layers, one layer per partition
model = nn.Sequential(*[nn.Linear(10, 10) for _ in range(16)])

# Checkpoint layers 3, 5, 12, 15
gpipe = GPipe(
    model,
    balance=[1] * 16,
    checkpoint=[3, 5, 12, 15]
)
```

With balance=[1]*16, layer indices map directly to partition indices, so this checkpoints partitions 3, 5, 12, 15 (plus borders 0 and 15).

### Example 2: Complex case (multiple layers per partition)

```python
# Model with 10 layers
model = nn.Sequential(*[nn.Linear(10, 10) for _ in range(10)])

# Split into 3 partitions:
# - Partition 0: layers 0, 1, 2
# - Partition 1: layers 3, 4, 5
# - Partition 2: layers 6, 7, 8, 9

# Checkpoint layers 3, 5, 7
gpipe = GPipe(
    model,
    balance=[3, 3, 4],
    checkpoint=[3, 5, 7]
)
```

Layers 3 and 5 are in partition 1, layer 7 is in partition 2, so partitions 1 and 2 are checkpointed (plus borders 0 and 2).

## Backward Compatibility

String modes ('always', 'except_last', 'never') continue to work exactly as before:

```python
gpipe = GPipe(model, balance=[...], checkpoint='except_last')
```

## Warning System

Out-of-range layer indices trigger warnings during forward pass:

```python
# Model has only layers 0, 1, 2
gpipe = GPipe(model, balance=[1, 1, 1], checkpoint=[1, 5, 10])

# During forward pass, warnings are issued:
# "Layer index 5 is out of range for model with 3 layers (valid range: 0-2)"
# "Layer index 10 is out of range for model with 3 layers (valid range: 0-2)"
```

## Testing

All tests updated to use layer indices and verify:
- Layer-to-partition conversion
- Out-of-range warning behavior
- Backward compatibility with string modes
- Various balance configurations

## Implementation Details

The `layer_indices_to_partitions()` function:
1. Builds a mapping from layer index to partition index based on balance
2. Converts each layer index to its corresponding partition index
3. Warns about out-of-range indices
4. Returns a set of partition indices to checkpoint
