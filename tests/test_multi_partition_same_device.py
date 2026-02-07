"""Tests for multiple partitions on the same device.

This test module demonstrates and validates that torchgpipe supports placing
multiple partitions on the same GPU/device, with checkpointing between them.
"""
import pytest
import torch
from torch import nn

from torchgpipe import GPipe


def test_multiple_partitions_on_same_device():
    """Test that multiple partitions can be placed on the same device."""
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
    )

    # Place 3 partitions all on CPU
    gpipe = GPipe(
        model,
        balance=[2, 2, 2],
        devices=['cpu', 'cpu', 'cpu'],
        chunks=2,
    )

    assert len(gpipe.partitions) == 3
    assert len(gpipe.devices) == 3
    assert all(d == torch.device('cpu') for d in gpipe.devices)

    # Verify forward pass works
    x = torch.randn(4, 10)
    y = gpipe(x)
    assert y.shape == torch.Size([4, 10])


def test_checkpointing_between_same_device_partitions():
    """Test that checkpointing works between partitions on the same device."""
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
    )

    # Place 2 partitions on same device with checkpointing enabled
    gpipe = GPipe(
        model,
        balance=[2, 2],
        devices=['cpu', 'cpu'],
        chunks=2,
        checkpoint='always',
    )

    x = torch.randn(4, 10, requires_grad=True)
    gpipe.train()
    y = gpipe(x)

    # Verify checkpoint nodes exist in computational graph
    assert y.grad_fn is not None
    
    # Count checkpoint nodes
    def count_checkpoint_nodes(grad_fn, visited=None):
        if visited is None:
            visited = set()
        if grad_fn is None or grad_fn in visited:
            return 0
        visited.add(grad_fn)
        
        count = 1 if 'Checkpoint' in grad_fn.__class__.__name__ else 0
        for next_fn, _ in grad_fn.next_functions:
            count += count_checkpoint_nodes(next_fn, visited)
        return count
    
    checkpoint_count = count_checkpoint_nodes(y.grad_fn)
    # Should have checkpoints for both chunks on each partition
    assert checkpoint_count >= 2, f"Expected at least 2 checkpoints, got {checkpoint_count}"

    # Verify backward pass works
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_mixed_device_mapping():
    """Test partitions distributed across multiple devices with some sharing."""
    model = nn.Sequential(
        nn.Linear(10, 10),  # Partition 0
        nn.Linear(10, 10),  # Partition 1
        nn.Linear(10, 10),  # Partition 2
        nn.Linear(10, 10),  # Partition 3
    )

    # Place: partition 0 on cpu, partitions 1-2 on cpu (same), partition 3 on cpu
    # This simulates: GPU 0, GPU 1 (shared), GPU 1 (shared), GPU 2
    gpipe = GPipe(
        model,
        balance=[1, 1, 1, 1],
        devices=['cpu', 'cpu', 'cpu', 'cpu'],
        chunks=2,
        checkpoint='always',
    )

    assert len(gpipe.partitions) == 4
    assert len(gpipe.devices) == 4

    # Verify forward and backward
    x = torch.randn(4, 10, requires_grad=True)
    gpipe.train()
    y = gpipe(x)
    loss = y.sum()
    loss.backward()
    
    assert y.shape == torch.Size([4, 10])
    assert x.grad is not None


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason='CUDA device required')
def test_multiple_partitions_on_single_gpu():
    """Test multiple partitions on a single GPU device."""
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
    )

    # Place 3 partitions on GPU 0
    gpipe = GPipe(
        model,
        balance=[1, 1, 1],
        devices=['cuda:0', 'cuda:0', 'cuda:0'],
        chunks=2,
        checkpoint='always',
    )

    assert len(gpipe.partitions) == 3
    assert all(d == torch.device('cuda:0') for d in gpipe.devices)

    # Verify it works with GPU tensors
    x = torch.randn(4, 10, device='cuda:0', requires_grad=True)
    gpipe.train()
    y = gpipe(x)
    loss = y.sum()
    loss.backward()
    
    assert y.device == torch.device('cuda:0')
    assert x.grad is not None


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='2 CUDA devices required')
def test_mixed_gpu_allocation():
    """Test partitions on mixed GPUs with some sharing."""
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.Linear(10, 10),
        nn.Linear(10, 10),
        nn.Linear(10, 10),
    )

    # Partitions: GPU 0, GPU 0 (shared), GPU 1, GPU 1 (shared)
    gpipe = GPipe(
        model,
        balance=[1, 1, 1, 1],
        devices=['cuda:0', 'cuda:0', 'cuda:1', 'cuda:1'],
        chunks=2,
        checkpoint='always',
    )

    assert gpipe.devices[0] == torch.device('cuda:0')
    assert gpipe.devices[1] == torch.device('cuda:0')
    assert gpipe.devices[2] == torch.device('cuda:1')
    assert gpipe.devices[3] == torch.device('cuda:1')

    x = torch.randn(4, 10, device='cuda:0', requires_grad=True)
    gpipe.train()
    y = gpipe(x)
    loss = y.sum()
    loss.backward()
    
    assert y.device == torch.device('cuda:1')
    assert x.grad is not None


def test_checkpoint_modes_with_same_device():
    """Test different checkpoint modes with multiple partitions on same device."""
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
    )

    for checkpoint_mode in ['always', 'except_last', 'never']:
        gpipe = GPipe(
            model,
            balance=[2, 2],
            devices=['cpu', 'cpu'],
            chunks=2,
            checkpoint=checkpoint_mode,
        )

        x = torch.randn(4, 10, requires_grad=True)
        gpipe.train()
        y = gpipe(x)
        loss = y.sum()
        loss.backward()
        
        assert y.shape == torch.Size([4, 10])
        assert x.grad is not None


def test_large_balance_same_device():
    """Test many partitions on the same device."""
    # Create a model with 8 layers
    layers = []
    for i in range(8):
        layers.append(nn.Linear(10, 10))
        if i < 7:
            layers.append(nn.ReLU())
    
    model = nn.Sequential(*layers)

    # Split into 8 partitions, all on the same CPU
    balance = [2] * 7 + [1]  # [2, 2, 2, 2, 2, 2, 2, 1] = 15 layers
    devices = ['cpu'] * 8
    
    gpipe = GPipe(
        model,
        balance=balance,
        devices=devices,
        chunks=4,
        checkpoint='except_last',
    )

    assert len(gpipe.partitions) == 8
    assert all(d == torch.device('cpu') for d in gpipe.devices)

    x = torch.randn(8, 10, requires_grad=True)
    gpipe.train()
    y = gpipe(x)
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None
