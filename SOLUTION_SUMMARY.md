# Solution: Multiple Partitions on One GPU with Checkpointing

## Problem Statement
User asked: "can i put 2 or more partition on one gpu? (btwn these "same gpu" partition might not have comucation stuff) aka add extra ckpt in middle of gpu/partition"

## Solution
**The feature already exists!** It just needed better documentation.

### How to Use
To place multiple partitions on the same GPU, simply repeat the device in the `devices` list:

```python
from torchgpipe import GPipe
import torch.nn as nn

model = nn.Sequential(a, b, c, d, e, f)

# Example 1: 3 partitions all on GPU 0
model = GPipe(model, balance=[2, 2, 2],
              devices=['cuda:0', 'cuda:0', 'cuda:0'],
              chunks=8, checkpoint='always')

# Example 2: Mix devices (partitions 0-1 on GPU 0, 2-3 on GPU 1)
model = GPipe(model, balance=[1, 2, 2, 1],
              devices=['cuda:0', 'cuda:0', 'cuda:1', 'cuda:1'],
              chunks=8)
```

### What Happens
- Each partition is assigned to `devices[i]`
- Repeating a device puts multiple partitions on the same GPU
- Checkpointing boundaries are created between partitions
- This adds extra checkpoint points on a single GPU for memory reduction

### Benefits
1. **Memory Reduction**: Extra checkpointing boundaries reduce memory usage
2. **Flexibility**: Fine-tune partition placement for optimal performance
3. **No Communication Overhead**: Between same-device partitions (as user wanted)

## Changes Made

### 1. Documentation (torchgpipe/gpipe.py)
- Enhanced `GPipe` class docstring with clear examples
- Updated `split_module` function docstring
- Added "Multiple Partitions on Same Device" section

### 2. Tests (tests/test_multi_partition_same_device.py)
Added 7 comprehensive tests:
- `test_multiple_partitions_on_same_device`
- `test_checkpointing_between_same_device_partitions`
- `test_mixed_device_mapping`
- `test_multiple_partitions_on_single_gpu`
- `test_mixed_gpu_allocation`
- `test_checkpoint_modes_with_same_device`
- `test_large_balance_same_device`

### 3. README (README.md)
- Added "Multiple Partitions on Same Device" section
- Included usage examples

### 4. Demo (examples/multi_partition_same_device_demo.py)
- Practical demonstration script
- Shows different scenarios and checkpoint modes

## Testing
- ✅ All new tests pass (5 CPU, 2 GPU tests with skip conditions)
- ✅ All existing tests pass (34 passed)
- ✅ Code review completed (style issues fixed)
- ✅ Security scan completed (0 alerts)

## Technical Details
The existing implementation already supported this through:
1. Device mapping: partition `i` → `devices[i]`
2. Constraint: `len(balance) <= len(devices)` (not 1:1 mapping)
3. Checkpointing applied at partition boundaries

No code changes to core functionality were needed, only documentation improvements.
