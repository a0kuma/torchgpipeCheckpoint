#!/usr/bin/env python3
"""
Demonstration: Multiple Partitions on Same Device with Checkpointing

This script demonstrates how to place multiple partitions on the same GPU/device
with checkpointing between them, which can help reduce memory usage.
"""

import torch
from torch import nn
from torchgpipe import GPipe


def main():
    print("=" * 70)
    print("Multi-Partition Same-Device Demo")
    print("=" * 70)
    
    # Create a simple sequential model
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )
    
    print("\n1. Original model:")
    print(f"   Layers: {len(model)}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example 1: Multiple partitions on CPU
    print("\n2. Example 1: 3 partitions on same CPU device")
    gpipe1 = GPipe(
        model,
        balance=[2, 3, 2],  # 3 partitions: [2 layers, 3 layers, 2 layers]
        devices=['cpu', 'cpu', 'cpu'],  # All on CPU
        chunks=4,
        checkpoint='always'
    )
    
    print(f"   Partitions: {len(gpipe1.partitions)}")
    print(f"   Balance: {gpipe1.balance}")
    print(f"   Devices: {[str(d) for d in gpipe1.devices]}")
    
    # Test forward pass
    x = torch.randn(16, 128)
    y = gpipe1(x)
    print(f"   ✓ Forward pass: {x.shape} -> {y.shape}")
    
    # Test backward pass
    loss = y.sum()
    loss.backward()
    print(f"   ✓ Backward pass successful")
    
    # Example 2: Mixed device allocation (if CUDA available)
    if torch.cuda.is_available():
        print("\n3. Example 2: Mixed GPU allocation")
        gpipe2 = GPipe(
            model,
            balance=[2, 2, 2, 1],  # 4 partitions
            devices=['cuda:0', 'cuda:0', 'cuda:0', 'cuda:0'],  # All on GPU 0
            chunks=4,
            checkpoint='except_last'
        )
        
        print(f"   Partitions: {len(gpipe2.partitions)}")
        print(f"   Balance: {gpipe2.balance}")
        print(f"   Devices: {[str(d) for d in gpipe2.devices]}")
        
        # Test with GPU tensors
        x_gpu = torch.randn(16, 128, device='cuda:0')
        y_gpu = gpipe2(x_gpu)
        print(f"   ✓ Forward pass on GPU: {x_gpu.shape} -> {y_gpu.shape}")
        
        # Test backward
        loss_gpu = y_gpu.sum()
        loss_gpu.backward()
        print(f"   ✓ Backward pass on GPU successful")
        
        # Show memory benefit
        if hasattr(torch.cuda, 'max_memory_allocated'):
            mem_mb = torch.cuda.max_memory_allocated(0) / 1024 / 1024
            print(f"   Memory used: {mem_mb:.2f} MB")
    else:
        print("\n3. Example 2: CUDA not available, skipping GPU demo")
    
    # Example 3: Different checkpoint modes
    print("\n4. Example 3: Different checkpoint modes with same device")
    for checkpoint_mode in ['always', 'except_last', 'never']:
        gpipe3 = GPipe(
            model,
            balance=[3, 4],  # 2 partitions on same device
            devices=['cpu', 'cpu'],
            chunks=2,
            checkpoint=checkpoint_mode
        )
        
        x = torch.randn(8, 128, requires_grad=True)
        y = gpipe3(x)
        loss = y.sum()
        loss.backward()
        
        print(f"   ✓ Checkpoint mode '{checkpoint_mode}': "
              f"{len(gpipe3.partitions)} partitions, forward+backward OK")
    
    print("\n" + "=" * 70)
    print("All demonstrations completed successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Multiple partitions can share the same GPU/device")
    print("  • Repeat device in devices list: devices=['cuda:0', 'cuda:0', ...]")
    print("  • Checkpointing works between partitions on same device")
    print("  • This adds extra checkpoint boundaries for memory savings")
    print("=" * 70)


if __name__ == '__main__':
    main()
