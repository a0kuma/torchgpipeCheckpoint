Copy Function Documentation
===========================

Overview
--------

The ``Copy`` class is a custom PyTorch autograd function that implements stream-aware CUDA tensor copying. It is designed to enable efficient overlap of data copying and computation operations on GPUs by managing tensor transfers across different CUDA streams.

Class Definition
----------------

.. code-block:: python

    class Copy(torch.autograd.Function):
        """Copies tensors on specific streams."""

Located in: ``torchgpipe/copy.py``

Purpose
-------

The ``Copy`` class serves a critical role in the torchgpipe pipeline parallelism implementation:

1. **Stream-Aware Copying**: Manages tensor copies between different CUDA streams to enable concurrent execution of copy and compute operations
2. **Device Transfer**: Handles tensor transfers between different devices (CPU to CUDA, CUDA to CPU, or CUDA to CUDA)
3. **Gradient Flow**: Properly implements backward pass to ensure gradients flow correctly through the copy operation
4. **Memory Management**: Uses ``record_stream`` to ensure tensor memory is not prematurely freed while still in use on different streams

How It Works
------------

Forward Pass
~~~~~~~~~~~~

The ``forward`` method performs the following operations:

.. code-block:: python

    @staticmethod
    def forward(ctx: Context,
                prev_stream: AbstractStream,
                next_stream: AbstractStream,
                *input: Tensor,
                ) -> Tensors:

**Parameters:**

- ``ctx``: Context object to save information for backward pass
- ``prev_stream``: The stream where input tensors originate
- ``next_stream``: The stream where output tensors will be used
- ``*input``: Variable number of input tensors to copy

**Process:**

1. Saves ``prev_stream`` and ``next_stream`` in context for use during backward pass
2. Captures the current output stream for memory management
3. Activates both ``prev_stream`` and ``next_stream`` using context managers
4. For each input tensor:

   - Copies the tensor to the device associated with ``next_stream``
   - Records the input tensor on ``prev_stream`` to prevent premature memory deallocation
   - Records the output tensor on the ``output_stream`` to ensure it remains valid for subsequent operations

5. Returns tuple of copied tensors

**Key Operations:**

- ``x.to(get_device(next_stream))``: Performs the actual tensor copy/transfer
- ``record_stream(x, prev_stream)``: Marks that tensor ``x`` is still in use on ``prev_stream``
- ``record_stream(y, output_stream)``: Marks that tensor ``y`` might be used on ``output_stream``

Backward Pass
~~~~~~~~~~~~~

The ``backward`` method implements gradient propagation:

.. code-block:: python

    @staticmethod
    def backward(ctx: Context,
                 *grad_output: Tensor,
                 ) -> Tuple[Optional[Tensor], ...]:

**Parameters:**

- ``ctx``: Context object containing saved information from forward pass
- ``*grad_output``: Gradients with respect to outputs

**Process:**

1. Retrieves saved ``prev_stream`` and ``next_stream`` from context
2. Creates a deque to store gradient inputs in reverse order
3. Activates both streams using context managers
4. For each gradient output (processed in reverse order):

   - Copies the gradient back to the device of ``prev_stream``
   - Records memory dependencies to prevent premature deallocation

5. Returns ``(None, None, *grad_input)`` where:

   - First two ``None`` values correspond to non-tensor parameters (``prev_stream``, ``next_stream``)
   - Remaining values are the computed gradients for input tensors

Usage Example
-------------

The ``Copy`` class is typically used through its ``apply`` method:

.. code-block:: python

    from torchgpipe.copy import Copy
    from torchgpipe.stream import current_stream

    # Define streams
    prev_stream = current_stream(torch.device('cuda:0'))
    next_stream = current_stream(torch.device('cuda:1'))

    # Copy tensors from one stream/device to another
    input_tensor = torch.randn(10, 20, device='cuda:0')
    output_tensor, = Copy.apply(prev_stream, next_stream, input_tensor)

    # output_tensor is now on cuda:1 and managed by next_stream

In the torchgpipe pipeline, it's commonly used via the ``copy`` helper function:

.. code-block:: python

    def copy(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) -> None:
        batch[:] = Copy.apply(prev_stream, next_stream, *batch)

Why Stream-Aware Copying Matters
---------------------------------

Problem Without Stream Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In standard PyTorch, tensor operations are executed on the default CUDA stream. When training large models with pipeline parallelism:

- Data must be copied between different GPU partitions
- Without proper stream management, copy operations block computation
- This creates idle time where GPUs wait for data transfers

Solution With Copy Class
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Copy`` class enables:

1. **Concurrent Execution**: Copy operations on one stream while computation happens on another
2. **Memory Safety**: Proper tracking of tensor lifetimes across streams using ``record_stream``
3. **Gradient Correctness**: Backward pass properly handles gradient flow across streams
4. **Device Flexibility**: Works with CPU-to-CUDA, CUDA-to-CPU, and CUDA-to-CUDA transfers

Relationship to GPipe
---------------------

In the GPipe implementation:

- Each partition of the model runs on a different device
- Tensors must be copied between partitions
- The ``Copy`` class ensures these copies happen efficiently without blocking computation
- It works in conjunction with the ``Wait`` class (also in ``copy.py``) to synchronize streams when necessary

Memory Management Details
--------------------------

record_stream Calls
~~~~~~~~~~~~~~~~~~~

The ``Copy`` class makes strategic use of ``record_stream``:

**In Forward Pass:**

.. code-block:: python

    record_stream(x, prev_stream)  # Input tensor used on prev_stream
    record_stream(y, output_stream)  # Output tensor used on output_stream

**In Backward Pass:**

.. code-block:: python

    record_stream(x, next_stream)  # Gradient used on next_stream
    record_stream(y, input_stream)  # Computed gradient used on input_stream

These calls inform PyTorch's memory allocator that tensors are still in use on specific streams, preventing premature garbage collection that could cause use-after-free errors.

Integration with Other Components
----------------------------------

The ``Copy`` class integrates with several torchgpipe components:

1. **Stream Module** (``torchgpipe/stream.py``): Provides stream abstraction for both CPU and CUDA
2. **Pipeline Module** (``torchgpipe/pipeline.py``): Uses ``Copy`` to transfer batches between partitions
3. **Skip Connections** (``torchgpipe/skip/portal.py``): Uses ``Copy`` to transfer skip tensors across partitions
4. **Checkpoint Module** (``torchgpipe/checkpoint.py``): Works with ``Copy`` during recomputation

Testing
-------

The ``Copy`` class is thoroughly tested in ``tests/test_copy.py`` with test cases covering:

- CPU to CPU copying
- CPU to CUDA copying
- CUDA to CPU copying
- CUDA to CUDA copying (different streams)
- Gradient flow verification
- Multiple tensor handling

Technical Considerations
------------------------

Why Inherit from torch.autograd.Function?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inheriting from ``torch.autograd.Function`` allows:

1. Custom forward and backward implementations
2. Integration with PyTorch's autograd engine
3. Access to context for saving/retrieving information between passes

Why Use Static Methods?
~~~~~~~~~~~~~~~~~~~~~~~~

Static methods are required by PyTorch's autograd function API. The ``ctx`` parameter provides state management between forward and backward passes without needing instance methods.

AbstractStream Type
~~~~~~~~~~~~~~~~~~~

The ``AbstractStream`` type (defined in ``torchgpipe/stream.py``) is a Union type that represents either:

- ``torch.cuda.Stream`` for CUDA devices
- ``CPUStreamType`` for CPU (placeholder since CPU doesn't have streams)

This abstraction allows the same code to work seamlessly with both CPU and CUDA devices.

Related Classes
---------------

- **Wait**: Another autograd function in the same file that synchronizes streams without copying data
- **PortalCopy**: Extends ``Copy`` for use with skip connections

Summary
-------

The ``Copy`` class is a fundamental building block for efficient pipeline parallelism in torchgpipe. By managing tensor copies in a stream-aware manner, it enables:

- Overlapping of data transfer and computation
- Proper gradient flow across device boundaries
- Safe memory management across CUDA streams
- Flexible CPU/CUDA device handling

This allows torchgpipe to achieve efficient training of large models across multiple GPUs with minimal idle time.
