"""
Demo script showing the new partition-based checkpoint functionality.
"""
import torch
from torch import nn
from torchgpipe import GPipe


def demo_original_modes():
    """Demonstrate the original string-based checkpoint modes."""
    print("=" * 60)
    print("Original String-based Checkpoint Modes")
    print("=" * 60)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 5)
    )
    
    # Test 'always' mode
    gpipe_always = GPipe(model, balance=[2, 2, 2, 1], devices=['cpu']*4, chunks=4, checkpoint='always')
    print(f"\ncheckpoint='always': {gpipe_always.checkpoint}")
    
    # Test 'except_last' mode (default)
    gpipe_except = GPipe(model, balance=[2, 2, 2, 1], devices=['cpu']*4, chunks=4, checkpoint='except_last')
    print(f"checkpoint='except_last': {gpipe_except.checkpoint}")
    
    # Test 'never' mode
    gpipe_never = GPipe(model, balance=[2, 2, 2, 1], devices=['cpu']*4, chunks=4, checkpoint='never')
    print(f"checkpoint='never': {gpipe_never.checkpoint}")
    
    # Test forward pass
    input_data = torch.randn(16, 10)
    output = gpipe_always(input_data)
    print(f"\nForward pass successful! Output shape: {output.shape}")


def demo_partition_based_checkpointing():
    """Demonstrate the new partition-based checkpoint functionality."""
    print("\n" + "=" * 60)
    print("New Partition-based Checkpointing")
    print("=" * 60)
    
    # Create a model with 8 partitions
    model = nn.Sequential(*[nn.Linear(10, 10) for _ in range(8)])
    
    # Checkpoint only partitions 2, 4, and 6
    # Note: Borders (partition 0 and 7) will be automatically checkpointed
    gpipe = GPipe(
        model,
        balance=[1, 1, 1, 1, 1, 1, 1, 1],
        devices=['cpu']*8,
        chunks=4,
        checkpoint=[2, 4, 6]
    )
    
    print(f"\nSpecified checkpoint partitions: [2, 4, 6]")
    print(f"Stored checkpoint: {gpipe.checkpoint}")
    print("Note: Borders (0 and 7) are automatically included during forward pass")
    
    # Test forward pass
    input_data = torch.randn(16, 10)
    output = gpipe(input_data)
    print(f"\nForward pass successful! Output shape: {output.shape}")


def demo_flexible_specification():
    """Show different ways to specify checkpoint partitions."""
    print("\n" + "=" * 60)
    print("Flexible Checkpoint Specification")
    print("=" * 60)
    
    model = nn.Sequential(*[nn.Linear(10, 10) for _ in range(6)])
    
    # Using a list
    gpipe1 = GPipe(model, balance=[1]*6, devices=['cpu']*6, chunks=2, checkpoint=[1, 3, 5])
    print(f"\nUsing list [1, 3, 5]: {gpipe1.checkpoint}")
    
    # Using a tuple
    gpipe2 = GPipe(model, balance=[1]*6, devices=['cpu']*6, chunks=2, checkpoint=(1, 3, 5))
    print(f"Using tuple (1, 3, 5): {gpipe2.checkpoint}")
    
    # Using a set
    gpipe3 = GPipe(model, balance=[1]*6, devices=['cpu']*6, chunks=2, checkpoint={1, 3, 5})
    print(f"Using set {{1, 3, 5}}: {gpipe3.checkpoint}")
    
    # Empty list (only borders will be checkpointed)
    gpipe4 = GPipe(model, balance=[1]*6, devices=['cpu']*6, chunks=2, checkpoint=[])
    print(f"Using empty list []: {gpipe4.checkpoint}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("GPipe Checkpoint Customization Demo")
    print("=" * 60)
    
    demo_original_modes()
    demo_partition_based_checkpointing()
    demo_flexible_specification()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
