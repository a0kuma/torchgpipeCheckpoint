# Partitions vs GPUs: Understanding the Relationship

## Question: Does the number of partitions have to be the same as the number of GPUs?

**Short Answer**: No, the number of partitions does **not** have to match the number of GPUs.

## Key Concepts

### Partitions
A **partition** is a group of consecutive layers from your model that are processed together. When you specify `balance=[2, 2, 3]`, you are creating 3 partitions with 2, 2, and 3 layers respectively.

### Devices (GPUs)
A **device** is the physical GPU (or CPU) where computations are performed. You can specify devices explicitly with the `devices` parameter, or let torchgpipe automatically use all available CUDA devices.

## The Relationship

The key rule is: **The number of partitions must be less than or equal to the number of devices.**

```
Number of Partitions ≤ Number of Devices
```

### Why This Flexibility?

1. **Multiple partitions can share the same device**: If you have 4 partitions but only 2 GPUs, torchgpipe will place multiple partitions on the same GPU.

2. **Not all devices need to be used**: If you have 8 GPUs but only 3 partitions, only the first 3 GPUs will be used.

## Examples

### Example 1: Partition Count Equals GPU Count (Most Common)

```python
from torchgpipe import GPipe

model = nn.Sequential(a, b, c, d)
# 4 partitions on 4 GPUs (cuda:0, cuda:1, cuda:2, cuda:3)
model = GPipe(model, balance=[1, 1, 1, 1], chunks=8)

# Result:
# Partition 0 (layer a) → cuda:0
# Partition 1 (layer b) → cuda:1
# Partition 2 (layer c) → cuda:2
# Partition 3 (layer d) → cuda:3
```

### Example 2: Fewer Partitions Than GPUs

```python
model = nn.Sequential(a, b, c, d)
# 2 partitions on 4 available GPUs
model = GPipe(model, balance=[2, 2], chunks=8)

# Result:
# Partition 0 (layers a, b) → cuda:0
# Partition 1 (layers c, d) → cuda:1
# cuda:2 and cuda:3 are not used
```

### Example 3: More Partitions Than GPUs (Using Specific Devices)

```python
model = nn.Sequential(a, b, c, d)
# 4 partitions on 2 GPUs - explicitly specify devices
model = GPipe(model, balance=[1, 1, 1, 1], devices=['cuda:0', 'cuda:1', 'cuda:0', 'cuda:1'], chunks=8)

# Result:
# Partition 0 (layer a) → cuda:0
# Partition 1 (layer b) → cuda:1
# Partition 2 (layer c) → cuda:0
# Partition 3 (layer d) → cuda:1
```

### Example 4: Single Device (CPU) with Multiple Partitions

```python
model = nn.Sequential(a, b)
# 2 partitions on 1 CPU device
model = GPipe(model, balance=[1, 1], devices=['cpu', 'cpu'], chunks=4)

# Result:
# Partition 0 (layer a) → cpu
# Partition 1 (layer b) → cpu
```

## Real-World Benchmark Example

From the ResNet-101 speed benchmark in this repository:

```python
# Pipeline-8 configuration
balance = [26, 22, 33, 44, 44, 66, 66, 69]  # 8 partitions
model = GPipe(model, balance, devices=devices, chunks=chunks)
```

If `devices` contains 8 GPU IDs, each partition gets its own GPU. If it contains only 4 GPU IDs, multiple partitions will share GPUs.

## Automatic Device Assignment

When you don't specify the `devices` parameter, torchgpipe automatically uses available CUDA devices:

```python
# From gpipe.py line 244-246
if devices is None:
    devices = range(torch.cuda.device_count())
```

This means:
- If you have 4 GPUs and 4 partitions → each partition gets one GPU
- If you have 4 GPUs and 2 partitions → only first 2 GPUs are used
- If you have 2 GPUs and 4 partitions → **ERROR!** (too few devices)

## Error Handling

If you try to create more partitions than you have devices without explicitly specifying device reuse, you'll get an error:

```python
# From gpipe.py line 100-102
if len(balance) > len(devices):
    raise IndexError('too few devices to hold given partitions '
                     f'(devices: {len(devices)}, partitions: {len(balance)})')
```

**To avoid this error**: Either reduce the number of partitions or explicitly provide a device list that reuses GPUs.

## Best Practices

1. **For maximum parallelism**: Make the number of partitions equal to the number of GPUs you have.

2. **For memory constraints**: Use fewer partitions than GPUs if you want to simplify the pipeline.

3. **For testing on limited hardware**: Explicitly specify device reuse in the `devices` parameter.

4. **For optimal performance**: Use automatic balancing tools to determine partition sizes:

```python
from torchgpipe.balance import balance_by_time

partitions = torch.cuda.device_count()  # Match GPU count
sample = torch.rand(128, 3, 224, 224)
balance = balance_by_time(partitions, model, sample)

model = GPipe(model, balance, chunks=8)
```

## Summary

- ✅ Partitions can equal GPU count (most common, optimal parallelism)
- ✅ Partitions can be less than GPU count (unused GPUs)
- ✅ Partitions can be more than GPU count (with explicit device specification)
- ❌ Partitions cannot exceed device count without explicit device reuse

The flexibility allows you to:
- Test multi-partition models on a single GPU
- Optimize memory usage and computational distribution
- Scale models across different hardware configurations
