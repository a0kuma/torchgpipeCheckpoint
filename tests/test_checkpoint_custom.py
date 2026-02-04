"""Test custom checkpoint functionality with layer indices."""
import pytest
import torch
from torch import nn

from torchgpipe import GPipe


def count_grad_fn(grad_fn, name, visited=None):
    """Count occurrences of a specific grad_fn type in the computation graph."""
    if visited is None:
        visited = set()
    
    if grad_fn in visited:
        return 0
    visited.add(grad_fn)

    if grad_fn is None:
        return 0
    if grad_fn.__class__.__name__ == name:
        return 1

    counter = 0
    for next_grad_fn, _ in grad_fn.next_functions:
        counter += count_grad_fn(next_grad_fn, name, visited=visited)
    return counter


def test_checkpoint_with_list():
    """Test checkpoint with list of layer indices."""
    # Create a model with 4 layers
    model = nn.Sequential(
        nn.Linear(1, 1),
        nn.Linear(1, 1),
        nn.Linear(1, 1),
        nn.Linear(1, 1),
    )
    
    # Checkpoint layers 1 and 2
    # With balance=[1,1,1,1], layer indices = partition indices
    # Borders (partitions 0 and 3) will be added automatically
    gpipe = GPipe(model, balance=[1, 1, 1, 1], devices=['cpu']*4, chunks=2, checkpoint=[1, 2])
    
    # Check that checkpoint is stored as a set (layer indices)
    assert isinstance(gpipe.checkpoint, set)
    assert gpipe.checkpoint == {1, 2}
    
    # Run forward pass
    input_data = torch.rand(2, 1)
    output = gpipe(input_data)
    
    # With chunks=2, we expect checkpoint functions in the graph
    assert count_grad_fn(output.grad_fn, 'CheckpointBackward') > 0


def test_checkpoint_with_tuple():
    """Test checkpoint with tuple of layer indices."""
    model = nn.Sequential(
        nn.Linear(1, 1),
        nn.Linear(1, 1),
        nn.Linear(1, 1),
    )
    
    # Use tuple instead of list
    gpipe = GPipe(model, balance=[1, 1, 1], devices=['cpu']*3, chunks=2, checkpoint=(1,))
    
    assert isinstance(gpipe.checkpoint, set)
    assert 1 in gpipe.checkpoint


def test_checkpoint_with_empty_list():
    """Test checkpoint with empty list - should still checkpoint borders."""
    model = nn.Sequential(
        nn.Linear(1, 1),
        nn.Linear(1, 1),
        nn.Linear(1, 1),
    )
    
    # Empty list, but borders should be added
    gpipe = GPipe(model, balance=[1, 1, 1], devices=['cpu']*3, chunks=2, checkpoint=[])
    
    assert isinstance(gpipe.checkpoint, set)
    assert gpipe.checkpoint == set()  # Empty set stored
    
    input_data = torch.rand(2, 1)
    output = gpipe(input_data)
    
    # Borders (0 and 2) should still be checkpointed
    assert count_grad_fn(output.grad_fn, 'CheckpointBackward') > 0


def test_checkpoint_string_modes_still_work():
    """Test that original string modes still work."""
    model = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1))
    input_data = torch.rand(2, 1)
    
    # Test 'always'
    gpipe_always = GPipe(model, balance=[1, 1], devices=['cpu', 'cpu'], chunks=2, checkpoint='always')
    assert gpipe_always.checkpoint == 'always'
    output_always = gpipe_always(input_data)
    assert count_grad_fn(output_always.grad_fn, 'CheckpointBackward') > 0
    
    # Test 'except_last'
    gpipe_except = GPipe(model, balance=[1, 1], devices=['cpu', 'cpu'], chunks=2, checkpoint='except_last')
    assert gpipe_except.checkpoint == 'except_last'
    output_except = gpipe_except(input_data)
    assert count_grad_fn(output_except.grad_fn, 'CheckpointBackward') > 0
    
    # Test 'never'
    gpipe_never = GPipe(model, balance=[1, 1], devices=['cpu', 'cpu'], chunks=2, checkpoint='never')
    assert gpipe_never.checkpoint == 'never'
    output_never = gpipe_never(input_data)
    assert count_grad_fn(output_never.grad_fn, 'CheckpointBackward') == 0


def test_checkpoint_invalid_string():
    """Test that invalid string raises error."""
    model = nn.Sequential(nn.Linear(1, 1))

    with pytest.raises(ValueError, match="checkpoint is not one of 'always', 'except_last', or 'never'"):
        GPipe(model, balance=[1], devices=['cpu'], chunks=2, checkpoint='invalid')


def test_checkpoint_invalid_type():
    """Test that invalid types raise appropriate errors."""
    model = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1))
    
    # Non-integer in list should raise error
    with pytest.raises(ValueError, match="checkpoint must be a string.*or an iterable of layer indices"):
        GPipe(model, balance=[1, 1], devices=['cpu', 'cpu'], chunks=2, checkpoint=[1, 'a'])


def test_checkpoint_out_of_range_layer_indices():
    """Test that out-of-range layer indices trigger warnings."""
    model = nn.Sequential(
        nn.Linear(1, 1),
        nn.Linear(1, 1),
        nn.Linear(1, 1),
    )
    
    # Include layer indices outside valid range (model only has layers 0, 1, 2)
    # Invalid indices (5, 10, -1) should trigger warnings during forward pass
    gpipe = GPipe(model, balance=[1, 1, 1], devices=['cpu']*3, chunks=2, checkpoint=[1, 5, 10, -1])
    
    # Verify that layer index 1 is stored (out-of-range indices not validated yet)
    assert gpipe.checkpoint == {1, 5, 10, -1}, f"Expected {{1, 5, 10, -1}}, got {gpipe.checkpoint}"
    
    # Warnings should be triggered during forward pass when conversion happens
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        input_data = torch.rand(2, 1)
        output = gpipe(input_data)
        
        # Should have warnings for out-of-range indices
        warning_messages = [str(warning.message) for warning in w if 'out of range' in str(warning.message)]
        assert len(warning_messages) >= 3, f"Expected at least 3 warnings, got {len(warning_messages)}"
    
    # Should still work without errors
    assert output.shape == (2, 1)


def test_checkpoint_eval_mode():
    """Test that checkpoint is disabled in eval mode."""
    model = nn.Sequential(
        nn.Linear(1, 1),
        nn.Linear(1, 1),
    )
    
    gpipe = GPipe(model, balance=[1, 1], devices=['cpu', 'cpu'], chunks=2, checkpoint=[0, 1])
    
    # In eval mode, no checkpointing should occur
    gpipe.eval()
    input_data = torch.rand(2, 1)
    output = gpipe(input_data)
    
    assert count_grad_fn(output.grad_fn, 'CheckpointBackward') == 0


def test_checkpoint_specific_layers():
    """Test checkpointing specific layers in a model."""
    # Create a model with 6 layers
    model = nn.Sequential(*[nn.Linear(1, 1) for _ in range(6)])
    
    # Checkpoint layers 2, 3, 4
    # With balance=[1,1,1,1,1,1], layer indices = partition indices
    gpipe = GPipe(
        model,
        balance=[1, 1, 1, 1, 1, 1],
        devices=['cpu']*6,
        chunks=2,
        checkpoint=[2, 3, 4]
    )
    
    input_data = torch.rand(2, 1)
    output = gpipe(input_data)
    
    # Should have checkpoint functions
    assert count_grad_fn(output.grad_fn, 'CheckpointBackward') > 0
    
    # Output should be correct shape
    assert output.shape == (2, 1)


def test_checkpoint_layer_to_partition_conversion():
    """Test that layer indices are correctly converted to partition indices."""
    # Create a model with 10 layers
    model = nn.Sequential(*[nn.Linear(1, 1) for _ in range(10)])
    
    # Split into 3 partitions: balance=[3, 3, 4]
    # Partition 0: layers 0, 1, 2
    # Partition 1: layers 3, 4, 5
    # Partition 2: layers 6, 7, 8, 9
    
    # Checkpoint layers 3, 5, 7 (should checkpoint partitions 1 and 2)
    gpipe = GPipe(
        model,
        balance=[3, 3, 4],
        devices=['cpu']*3,
        chunks=2,
        checkpoint=[3, 5, 7]
    )
    
    # Layer indices are stored as-is
    assert gpipe.checkpoint == {3, 5, 7}
    
    input_data = torch.rand(2, 1)
    output = gpipe(input_data)
    
    # Should have checkpoint functions
    assert count_grad_fn(output.grad_fn, 'CheckpointBackward') > 0
    
    # Output should be correct shape
    assert output.shape == (2, 1)
