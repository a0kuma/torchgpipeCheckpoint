# Understanding checkpoint_stop in torchgpipe

## Overview

`checkpoint_stop` is a parameter in torchgpipe's pipeline parallelism implementation that controls which micro-batches use checkpointing during forward propagation. It provides fine-grained control over the memory-computation trade-off in GPipe models.

## What is Checkpointing?

Checkpointing is a memory optimization technique used in GPipe that:
- **During forward propagation**: Discards intermediate activations (except at partition boundaries)
- **During backward propagation**: Recomputes the forward propagation to regenerate the activations needed for gradient calculation

This technique significantly reduces memory usage at the cost of additional computation time. According to the torchgpipe documentation, checkpointing typically incurs around 25% overhead, though this varies by model architecture and configuration.

## How checkpoint_stop Works

### Basic Mechanism

`checkpoint_stop` is an integer that represents **the micro-batch index where checkpointing stops**. Specifically:

- **Micro-batches with index `i < checkpoint_stop`**: Use checkpointing (memory-efficient mode)
- **Micro-batches with index `i >= checkpoint_stop`**: Do NOT use checkpointing (memory-intensive mode)

### Example

If you have 8 micro-batches (chunks=8) and `checkpoint_stop=6`:
- Micro-batches 0, 1, 2, 3, 4, 5: **Use checkpointing** (recompute activations during backprop)
- Micro-batches 6, 7: **Do NOT use checkpointing** (keep all activations in memory)

## How checkpoint_stop is Determined

The value of `checkpoint_stop` is automatically calculated based on the `checkpoint` parameter in the `GPipe` constructor:

```python
if self.training:
    checkpoint_stop = {
        'always': self.chunks,       # All micro-batches use checkpointing
        'except_last': self.chunks-1, # All except the last micro-batch
        'never': 0,                  # No micro-batches use checkpointing
    }[self.checkpoint]
else:
    checkpoint_stop = 0  # No checkpointing during evaluation/inference
```

### Checkpoint Modes

1. **`checkpoint='always'`**: `checkpoint_stop = chunks`
   - All micro-batches use checkpointing
   - Maximum memory savings
   - Maximum computation overhead

2. **`checkpoint='except_last'`** (default): `checkpoint_stop = chunks - 1`
   - All micro-batches except the last one use checkpointing
   - Good balance: saves memory while avoiding unnecessary recomputation for the last batch
   - Recommended for most use cases

3. **`checkpoint='never'`**: `checkpoint_stop = 0`
   - No micro-batches use checkpointing
   - Maximum memory usage
   - No computation overhead from recomputation
   - Equivalent to typical model parallelism

## Implementation Details

### In pipeline.py

The `checkpoint_stop` parameter is used in the `Pipeline` class's `compute` method:

```python
def compute(self, schedule, skip_trackers, in_queues, out_queues):
    checkpoint_stop = self.checkpoint_stop
    
    for i, j in schedule:
        # Determine whether to use checkpointing for this micro-batch
        checkpoint = (i < checkpoint_stop)
        
        if checkpoint:
            # Use Checkpointing class: forward pass discards activations
            # Recomputation will happen during backprop
            chk = Checkpointing(function, batch)
            task = Task(streams[j], compute=chk.checkpoint, finalize=chk.recompute)
        else:
            # Normal execution: keep all activations in memory
            task = Task(streams[j], compute=compute, finalize=None)
```

### Autograd Graph with Checkpointing

When checkpointing is enabled for a micro-batch, the autograd graph includes:

1. **Wait**: Synchronize with copy stream
2. **Checkpoint**: Forward pass with `torch.no_grad()` - activations are not saved
3. **Wait**: Synchronize copy stream
4. **Recompute**: During backprop, recompute forward pass to regenerate activations

```
┌─────────────┐
│    Copy     │
└──────┬──────┘   (fence)
───────┼────────────────
       │          (compute)
┌──────┴──────┐
│    Wait     │ [1] Sync with copy stream
└──────┬──────┘
┌──────┴──────┐
│ Checkpoint  │ [2] Forward pass without saving activations
└──────┬──────┘
┌──────┴──────┐
│    Wait     │ [3] Sync copy stream
└──────┬──────┘
       ├────────┐
       │ ┌──────┴──────┐
       │ │  Recompute  │ [4] Scheduled for backprop
       │ └──────┬──────┘
       ├────────┘
```

## Why "except_last" is the Default

The default `checkpoint='except_last'` mode is chosen because:

1. **The last micro-batch completes forward propagation right before backpropagation begins**
2. **Memory saved by checkpointing the last batch would be immediately needed anyway**
3. **Avoiding recomputation for the last batch reduces unnecessary overhead**

In a typical pipeline schedule with 8 micro-batches (chunks=8) and 4 partitions:
```
Clock | Micro-batch operations
------|-------------------------------------------------------
0     | (0,0)
1     | (1,0) (0,1)
2     | (2,0) (1,1) (0,2)
...   | ...
10    |             (7,2) (6,3)  <- Last micro-batch reaches last partition
11    |                   (7,3)  <- Last micro-batch completes
      | [Backpropagation begins immediately after]
```

## Memory vs Computation Trade-off

> **Note on overhead**: The overhead figures below are based on the torchgpipe documentation's estimate that checkpointing typically adds ~25% additional computation time per checkpointed micro-batch. Actual overhead varies by model architecture, configuration, and hardware.

| Mode | Memory Usage | Computation Overhead | Use Case |
|------|-------------|---------------------|----------|
| `always` | Lowest | ~25% additional computation time (all micro-batches) | When memory is extremely limited |
| `except_last` | Low | ~25% × (chunks-1)/chunks additional time | **Recommended default** - good balance |
| `never` | Highest | No overhead | When memory is not a concern, or chunks=1 |

*For `except_last`: If chunks=8, overhead is approximately 25% × 7/8 ≈ 21.9% of total forward+backward time.

## Logging

The implementation includes logging to help track which micro-batches use checkpointing:

```python
if checkpoint:
    logger.debug(f"Micro-batch {i} (partition {j}): checkpoint_stop={checkpoint_stop}, using_checkpointing=True")
else:
    logger.info(f"Micro-batch {i} (partition {j}): checkpoint_stop={checkpoint_stop}, using_checkpointing=False")
```

Note: Micro-batches **NOT** using checkpointing are logged at INFO level for visibility.

## Best Practices

1. **Use the default `checkpoint='except_last'`** unless you have a specific reason not to
2. **For model parallelism without micro-batching**, use `chunks=1, checkpoint='never'`
3. **If memory is critical**, use `checkpoint='always'` 
4. **Monitor training speed** - if checkpointing overhead is too high, consider using `except_last` or `never`
5. **Remember**: No checkpointing is used during evaluation (`model.eval()`) regardless of settings

## Related Components

- **`Checkpointing` class** (`checkpoint.py`): Implements the checkpoint/recompute mechanism
- **`Checkpoint` and `Recompute` autograd functions**: Handle forward/backward passes
- **`is_checkpointing()` and `is_recomputing()`**: Helper functions to detect checkpoint state

## References

- Main implementation: `torchgpipe/pipeline.py` 
  - `Pipeline.__init__`: Stores `checkpoint_stop` as an instance variable
  - `Pipeline.compute`: Uses `checkpoint_stop` to determine checkpointing behavior
- Checkpoint logic: `torchgpipe/checkpoint.py` (`Checkpointing`, `Checkpoint`, and `Recompute` classes)
- User interface: `torchgpipe/gpipe.py` (`GPipe.forward` method calculates `checkpoint_stop` value)
- Documentation: `docs/guide.rst` (Checkpointing section)
