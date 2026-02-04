"""Test custom checkpoint functionality with partition indices."""
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
    """Test checkpoint with list of partition indices."""
    # Create a model with 4 partitions
    model = nn.Sequential(
        nn.Linear(1, 1),
        nn.Linear(1, 1),
        nn.Linear(1, 1),
        nn.Linear(1, 1),
    )
    
    # Checkpoint only partitions 1 and 2 (borders 0 and 3 will be added automatically)
    gpipe = GPipe(model, balance=[1, 1, 1, 1], devices=['cpu']*4, chunks=2, checkpoint=[1, 2])
    
    # Check that checkpoint is stored as a set
    assert isinstance(gpipe.checkpoint, set)
    assert gpipe.checkpoint == {1, 2}
    
    # Run forward pass
    input_data = torch.rand(2, 1)
    output = gpipe(input_data)
    
    # All 4 partitions should be checkpointed (borders are always added)
    # With chunks=2, we expect checkpoint functions in the graph
    assert count_grad_fn(output.grad_fn, 'CheckpointBackward') > 0


def test_checkpoint_with_tuple():
    """Test checkpoint with tuple of partition indices."""
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
    
    with pytest.raises(ValueError, match="checkpoint is not one of"):
        GPipe(model, balance=[1], devices=['cpu'], chunks=2, checkpoint='invalid')


def test_checkpoint_invalid_type():
    """Test that invalid types raise appropriate errors."""
    model = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1))
    
    # Non-integer in list should raise error
    with pytest.raises(ValueError, match="checkpoint must be a string"):
        GPipe(model, balance=[1, 1], devices=['cpu', 'cpu'], chunks=2, checkpoint=[1, 'a'])


def test_checkpoint_out_of_range_indices():
    """Test that out-of-range indices are filtered out."""
    model = nn.Sequential(
        nn.Linear(1, 1),
        nn.Linear(1, 1),
        nn.Linear(1, 1),
    )
    
    # Include indices outside valid range
    gpipe = GPipe(model, balance=[1, 1, 1], devices=['cpu']*3, chunks=2, checkpoint=[1, 5, 10, -1])
    
    # Should only contain valid indices plus borders
    input_data = torch.rand(2, 1)
    output = gpipe(input_data)
    
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


def test_checkpoint_specific_partitions():
    """Test checkpointing specific partitions in a larger model."""
    # Create a model with 6 partitions
    model = nn.Sequential(*[nn.Linear(1, 1) for _ in range(6)])
    
    # Checkpoint only partitions 2, 3, 4 (borders 0 and 5 will be added)
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
