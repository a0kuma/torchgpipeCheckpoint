"""
Example script demonstrating layer-level logging in torchgpipe.

This script shows how to enable and use logging to track which individual
layers (not just partitions) are currently being executed in the 
Pipeline.compute() function.
"""

import torch
import torch.nn as nn
import logging
from torchgpipe import GPipe, register_layer_logging_hooks

# Configure logging to see both partition and layer execution
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG to see individual layer executions
    format='%(levelname)s:%(name)s: %(message)s'
)

# Optional: Save logs to a file as well
file_handler = logging.FileHandler('layer_execution.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logging.getLogger('torchgpipe').addHandler(file_handler)


def main():
    print("=" * 70)
    print("torchgpipe Layer-Level Logging Example")
    print("=" * 70)
    print("\nThis example demonstrates how to log which INDIVIDUAL LAYER")
    print("(not just partition) is running in the Pipeline.compute() function.")
    print("\nIMPORTANT: Layers and partitions are different things!")
    print("  - A LAYER is a single module (e.g., Linear, ReLU, Conv2d)")
    print("  - A PARTITION is a group of consecutive layers\n")
    
    # Create a simple model with 5 layers
    model = nn.Sequential(
        nn.Linear(100, 100),  # Layer 0
        nn.ReLU(),            # Layer 1
        nn.Linear(100, 100),  # Layer 2
        nn.ReLU(),            # Layer 3
        nn.Linear(100, 10)    # Layer 4
    )
    
    print("Original model structure (5 layers):")
    for i, (name, layer) in enumerate(model.named_children()):
        print(f"  Layer {name}: {layer.__class__.__name__}")
    
    # Split the model into 2 partitions
    # balance=[2, 3] means:
    #   Partition 0: layers 0-1 (Linear + ReLU)
    #   Partition 1: layers 2-4 (Linear + ReLU + Linear)
    balance = [2, 3]
    chunks = 4  # Split mini-batch into 4 micro-batches
    
    print(f"\nGPipe configuration:")
    print(f"  - Partitions: {len(balance)}")
    print(f"  - Balance: {balance}")
    print(f"  - Micro-batches (chunks): {chunks}")
    print(f"\nPartition layout:")
    print(f"  - Partition 0: layers 0-1 (Linear + ReLU)")
    print(f"  - Partition 1: layers 2-4 (Linear + ReLU + Linear)")
    
    # Wrap with GPipe
    # Use CPU devices since GPUs might not be available in all environments
    devices = [torch.device('cpu')] * len(balance)
    model = GPipe(model, balance=balance, chunks=chunks, devices=devices, checkpoint='never')
    
    # Register hooks to log individual layer execution
    print("\n" + "=" * 70)
    print("Registering layer logging hooks...")
    print("=" * 70)
    hooks = register_layer_logging_hooks(model.partitions)
    print(f"Hooks registered: {len(hooks)} (one per layer)")
    
    # Create sample data
    batch_size = 8
    inputs = torch.randn(batch_size, 100)
    targets = torch.randint(0, 10, (batch_size,))
    
    # Training setup
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("\n" + "=" * 70)
    print("Starting training - watch for log messages below:")
    print("=" * 70)
    print("Legend:")
    print("  INFO:torchgpipe.pipeline  = Partition-level logs")
    print("  DEBUG:torchgpipe.pipeline = Layer-level logs (which specific layer)")
    print("=" * 70 + "\n")
    
    # Flush output before running model
    import sys
    sys.stdout.flush()
    
    # Single training step
    # You will see:
    # 1. INFO logs showing which partition is processing which micro-batch
    # 2. DEBUG logs showing which individual LAYER within that partition is executing
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    print("\n" + "=" * 70)
    print(f"Training step completed. Loss: {loss.item():.4f}")
    print("=" * 70)
    print("\nLog interpretation:")
    print("  Partition-level log:")
    print("    'Micro-batch X (partition Y)' = Which partition is processing")
    print("  Layer-level log:")
    print("    'Executing layer Z in partition Y: Type' = Which specific layer")
    print("\nKey distinction:")
    print("  - Partition Y = a GROUP of layers executing together")
    print("  - Layer Z = a SINGLE layer (Linear, ReLU, etc.)")
    print("\nLogs have been saved to: layer_execution.log")
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()


if __name__ == "__main__":
    main()
