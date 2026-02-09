"""
Example script demonstrating layer logging in torchgpipe.

This script shows how to enable and use logging to track which layers
(partitions) are currently being executed in the Pipeline.compute() function.
"""

import torch
import torch.nn as nn
import logging
from torchgpipe import GPipe

# Configure logging to see layer execution
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG to see all layer executions
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Optional: Save logs to a file as well
file_handler = logging.FileHandler('layer_execution.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logging.getLogger('torchgpipe').addHandler(file_handler)


def main():
    print("=" * 60)
    print("torchgpipe Layer Logging Example")
    print("=" * 60)
    print("\nThis example demonstrates how to log which layer is running")
    print("in the Pipeline.compute() function.\n")
    
    # Create a simple model with 5 layers
    model = nn.Sequential(
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    
    # Split the model into 3 partitions across devices
    # Partition 0: layers 0-1 (Linear + ReLU)
    # Partition 1: layers 2-3 (Linear + ReLU)
    # Partition 2: layer 4 (Linear)
    balance = [2, 2, 1]
    chunks = 4  # Split mini-batch into 4 micro-batches
    
    print(f"Model configuration:")
    print(f"  - Total layers: 5")
    print(f"  - Partitions: {len(balance)}")
    print(f"  - Balance: {balance}")
    print(f"  - Micro-batches (chunks): {chunks}")
    print(f"\nPartition layout:")
    print(f"  - Partition 0: layers 0-1 (Linear + ReLU)")
    print(f"  - Partition 1: layers 2-3 (Linear + ReLU)")
    print(f"  - Partition 2: layer 4 (Linear)")
    print("\n" + "=" * 60)
    print("Starting training - watch for log messages below:")
    print("=" * 60 + "\n")
    
    # Wrap with GPipe
    # Use CPU devices since GPUs might not be available in all environments
    # Use checkpoint='never' to disable checkpointing and show INFO level logs
    devices = [torch.device('cpu')] * len(balance)
    model = GPipe(model, balance=balance, chunks=chunks, devices=devices, checkpoint='never')
    
    # Create sample data
    batch_size = 8
    inputs = torch.randn(batch_size, 100)
    targets = torch.randint(0, 10, (batch_size,))
    
    # Training setup
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Single training step
    # You will see log messages showing which micro-batch is being
    # processed by which partition
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    print("\n" + "=" * 60)
    print(f"Training step completed. Loss: {loss.item():.4f}")
    print("=" * 60)
    print("\nLog interpretation:")
    print("  - 'Micro-batch X (partition Y)' tells you:")
    print("    - Which micro-batch (X) is being processed")
    print("    - Which partition/layer group (Y) is executing it")
    print("    - Whether checkpointing is being used")
    print("\nLogs have been saved to: layer_execution.log")


if __name__ == "__main__":
    main()
