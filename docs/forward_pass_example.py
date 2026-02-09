"""
Example demonstrating where the forward pass happens in Pipeline.compute()

This script shows that when partition(input) is called, PyTorch's nn.Sequential
automatically iterates through all layers and calls their forward() methods.
"""

import torch
import torch.nn as nn
import logging
from torchgpipe import GPipe, register_layer_logging_hooks

# Configure logging to see execution flow
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(message)s'
)

print("=" * 70)
print("DEMONSTRATION: Where Does the Forward Pass Happen?")
print("=" * 70)
print()

# Create a model with 5 distinct layers
model = nn.Sequential(
    nn.Linear(20, 20),  # Layer 0
    nn.ReLU(),          # Layer 1
    nn.Linear(20, 20),  # Layer 2
    nn.ReLU(),          # Layer 3
    nn.Linear(20, 10)   # Layer 4
)

print("Model structure (5 layers):")
for i, (name, layer) in enumerate(model.named_children()):
    print(f"  Layer {name}: {layer.__class__.__name__}")

print()
print("=" * 70)
print("The Key Question:")
print("=" * 70)
print("Where is the code that does: for layer in partition: layer.forward()?")
print()
print("Answer: It's INSIDE PyTorch's nn.Sequential.__call__() method!")
print()

# Create GPipe model - split into 2 partitions
devices = [torch.device('cpu')] * 2
model_gpipe = GPipe(model, balance=[2, 3], chunks=2, devices=devices, checkpoint='never')

print("Created 2 partitions:")
print("  Partition 0: Layers 0-1 (Linear, ReLU)")
print("  Partition 1: Layers 2-4 (Linear, ReLU, Linear)")
print()

# Register hooks to see individual layer execution
print("=" * 70)
print("Registering layer logging hooks to see the forward pass...")
print("=" * 70)
hooks = register_layer_logging_hooks(model_gpipe.partitions)
print()

print("=" * 70)
print("Running forward pass...")
print("=" * 70)
print("Watch for DEBUG logs showing each layer executing!")
print()

# Run the model - this will show the execution flow
inp = torch.randn(4, 20)
out = model_gpipe(inp)

print()
print("=" * 70)
print("What Just Happened?")
print("=" * 70)
print()
print("1. Pipeline.compute() created tasks for each partition")
print("2. Each task contains: lambda: partition(input)")
print("3. Worker threads executed: partition(input)")
print("4. partition(input) called nn.Sequential.__call__()")
print("5. nn.Sequential looped through layers:")
print("   for layer in self:")
print("       input = layer(input)")
print("6. Each layer's forward() was called (shown in DEBUG logs)")
print()
print("The loop 'for layer in partition' IS THERE!")
print("It's just hidden inside PyTorch's nn.Sequential implementation.")
print()
print("That's why you see DEBUG logs like:")
print("  'Executing layer X in partition Y: LayerType'")
print()
print("Each of those DEBUG logs represents one iteration of the implicit loop.")
print()

# Clean up
for hook in hooks:
    hook.remove()

print("=" * 70)
print("Summary")
print("=" * 70)
print()
print("The forward pass code you're looking for is:")
print()
print("  partition(input)  <-- This line in pipeline.py")
print()
print("Which expands to:")
print()
print("  nn.Sequential.__call__(input)")
print("      for module in self:")
print("          input = module(input)")
print()
print("So partition(input) IS the loop that calls each layer's forward()!")
print("=" * 70)
