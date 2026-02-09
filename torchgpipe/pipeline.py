"""The pipeline parallelism of GPipe."""
import logging
from queue import Queue
from types import TracebackType
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Type, Union, cast

import torch
from torch import Tensor, nn

from torchgpipe.checkpoint import Checkpointing
from torchgpipe.copy import Copy, Wait
from torchgpipe.dependency import fork, join
from torchgpipe.microbatch import Batch
from torchgpipe.skip.layout import SkipLayout, inspect_skip_layout
from torchgpipe.skip.tracker import SkipTrackerThroughPotals, use_skip_tracker
from torchgpipe.stream import AbstractStream, current_stream, use_device
from torchgpipe.worker import Task, spawn_workers

__all__: List[str] = ['register_layer_logging_hooks']

logger = logging.getLogger(__name__)


Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]

ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]

# Queue is generic only in stubs.
# https://mypy.readthedocs.io/en/latest/common_issues.html#using-classes-that-are-generic-in-stubs-but-not-at-runtime
if TYPE_CHECKING:
    InQueue = Queue[Optional['Task']]
    OutQueue = Queue[Tuple[bool, Union[Tuple['Task', Batch], ExcInfo, None]]]
else:
    InQueue = Queue
    OutQueue = Queue


def depend(fork_from: Batch, join_to: Batch) -> None:
    fork_from[0], phony = fork(fork_from[0])
    join_to[0] = join(join_to[0], phony)


def copy(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) -> None:
    batch[:] = Copy.apply(prev_stream, next_stream, *batch)


def wait(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) -> None:
    batch[:] = Wait.apply(prev_stream, next_stream, *batch)


def clock_cycles(m: int, n: int) -> Iterable[List[Tuple[int, int]]]:
    """Generates schedules for each clock cycle."""
    # m: number of micro-batches
    # n: number of partitions
    # i: index of micro-batch
    # j: index of partition
    # k: clock number
    #
    # k (i,j) (i,j) (i,j)
    # - ----- ----- -----
    # 0 (0,0)
    # 1 (1,0) (0,1)
    # 2 (2,0) (1,1) (0,2)
    # 3       (2,1) (1,2)
    # 4             (2,2)
    for k in range(m+n-1):
        yield [(k-j, j) for j in range(max(1+k-m, 0), min(1+k, n))]


def register_layer_logging_hooks(partitions: List[nn.Sequential]) -> List:
    """Register forward hooks on individual layers within partitions for logging.
    
    This allows tracking which individual layer is executing within each partition.
    A partition contains multiple layers, and this function adds hooks to log
    each layer's execution.
    
    Args:
        partitions: List of partitions (each partition is an nn.Sequential of layers)
        
    Returns:
        List of hook handles that can be used to remove hooks later
        
    Example:
        >>> hooks = register_layer_logging_hooks(model.partitions)
        >>> # Run your model...
        >>> # Remove hooks when done
        >>> for hook in hooks:
        >>>     hook.remove()
    """
    hooks = []
    
    for partition_idx, partition in enumerate(partitions):
        for layer_name, layer in partition.named_children():
            def make_hook(part_idx, layer_id):
                def hook(module, input, output):
                    logger.debug(
                        f"Executing layer '{layer_id}' in partition {part_idx}: {module.__class__.__name__}"
                    )
                return hook
            
            handle = layer.register_forward_hook(make_hook(partition_idx, layer_name))
            hooks.append(handle)
    
    return hooks


class Pipeline:
    """The pipeline parallelism for GPipe."""

    def __init__(self,
                 batches: List[Batch],
                 partitions: List[nn.Sequential],
                 devices: Optional[List[torch.device]] = None,
                 copy_streams: Optional[List[List[AbstractStream]]] = None,
                 skip_layout: Optional[SkipLayout] = None,
                 checkpoint_stop: int = 0,
                 ) -> None:
        self.batches = batches
        self.partitions = partitions

        if devices is None:
            devices = [torch.device('cpu') for _ in partitions]
        self.devices = devices

        if copy_streams is None:
            copy_streams = [[current_stream(d)] * len(batches) for d in devices]
        self.copy_streams = copy_streams

        if skip_layout is None:
            skip_layout = inspect_skip_layout(partitions)

        self.skip_layout = skip_layout
        self.checkpoint_stop = checkpoint_stop

    def run(self) -> None:
        """Runs pipeline parallelism.

        It modifies the given batches in place.

        """
        batches = self.batches
        partitions = self.partitions
        devices = self.devices
        skip_layout = self.skip_layout

        m = len(batches)
        n = len(partitions)

        skip_trackers = [SkipTrackerThroughPotals(skip_layout) for _ in batches]

        with spawn_workers(devices) as (in_queues, out_queues):
            for schedule in clock_cycles(m, n):
                self.fence(schedule, skip_trackers)
                self.compute(schedule, skip_trackers, in_queues, out_queues)

    def fence(self,
              schedule: List[Tuple[int, int]],
              skip_trackers: List[SkipTrackerThroughPotals],
              ) -> None:
        """Copies micro-batches after computation for the previous
        micro-batches.
        """
        batches = self.batches
        copy_streams = self.copy_streams
        skip_layout = self.skip_layout

        for i, j in schedule:
            # Ensure that batches[i-1] is executed after batches[i] in
            # backpropagation by an explicit dependency.
            if i != 0:
                depend(batches[i-1], batches[i])

            next_stream = copy_streams[j][i]

            for prev_j, ns, name in skip_layout.copy_policy(j):
                prev_stream = copy_streams[prev_j][i]
                skip_trackers[i].copy(batches[i], prev_stream, next_stream, ns, name)

            if j != 0:
                prev_stream = copy_streams[j-1][i]
                copy(batches[i], prev_stream, next_stream)

    def compute(self,
                schedule: List[Tuple[int, int]],
                skip_trackers: List[SkipTrackerThroughPotals],
                in_queues: List[InQueue],
                out_queues: List[OutQueue],
                ) -> None:
        """Runs tasks with synchronization to copy streams.
        
        This function schedules the execution of partitions (groups of layers) on worker threads.
        The actual forward pass through individual layers happens implicitly when a partition
        (which is an nn.Sequential) is called.
        
        IMPORTANT: The forward pass code flow
        ======================================
        1. This function creates tasks containing partition execution closures
        2. Tasks are sent to worker threads via in_queues[j].put(task)
        3. Worker threads call task.compute() which executes the partition
        4. When partition(input) or batch.call(partition) is called:
           - PyTorch's nn.Sequential.__call__() is invoked
           - nn.Sequential automatically iterates through all layers in the partition
           - Each layer's forward() method is called sequentially
        5. The result is returned through out_queues[j].get()
        
        So while you don't see "for layer in partition: layer.forward()" explicitly,
        it happens inside PyTorch's nn.Sequential when partition(input) is called.
        
        To see which individual layers execute, use register_layer_logging_hooks().
        """
        batches = self.batches
        partitions = self.partitions
        devices = self.devices
        copy_streams = self.copy_streams
        checkpoint_stop = self.checkpoint_stop

        n = len(partitions)
        streams = [current_stream(d) for d in devices]
        exc_info: Optional[ExcInfo] = None

        # With checkpointing, the autograd graph looks like this diagram:
        # ┌─────┸──────┐
        # │    Copy    │
        # └─────┰──────┘   (fence)
        # ─ ─ ─ ╂ ─ ─ ─ ─ ─ ─ ─ ─ ─
        #       ┃          (compute)
        # ┌─────┸──────┐
        # │    Wait    │ [1] Synchronize the current stream with the copy stream.
        # └─────┰──────┘
        # ┌─────┸──────┐
        # │ Checkpoint │ [2] Compute a partition within checkpointing.
        # └─────┰──────┘
        # ┌─────┸──────┐
        # │    Wait    │ [3] Synchronize the copy stream with the current stream.
        # └─────┰──────┘
        #       ┠ ─ ─ ─ ┐
        #       ┃ ┌─────┴─────┐
        #       ┃ │ Recompute │ [4] Schedule the recomputation at backpropagation.
        #       ┃ └─────┬─────┘
        #       ┠ ─ ─ ─ ┘
        #       ┃
        # ─ ─ ─ ╂ ─ ─ ─ ─ ─ ─ ─ ─ ─
        # ┌─────┸──────┐   (fence)
        # │    Copy    │
        # └─────┰──────┘
        for i, j in schedule:
            batch = batches[i]
            partition = partitions[j]

            # Synchronize with the copied input. ([1] in the diagram)
            if j != 0:
                wait(batch, copy_streams[j][i], streams[j])

            # Determine whether checkpointing or not.
            checkpoint = (i < checkpoint_stop)

            # Log checkpoint decision for each micro-batch
            # Use INFO level to highlight micro-batches NOT using checkpointing
            if checkpoint:
                logger.debug(
                    f"Micro-batch {i} (partition {j}): "
                    f"checkpoint_stop={checkpoint_stop}, "
                    f"using_checkpointing=True"
                )
            else:
                logger.info(
                    f"Micro-batch {i} (partition {j}): "
                    f"checkpoint_stop={checkpoint_stop}, "
                    f"using_checkpointing=False"
                )

            if checkpoint:
                # WITH CHECKPOINTING:
                # Define a function that will execute the partition's forward pass.
                # When this function is called, partition(input) invokes nn.Sequential.__call__()
                # which internally loops through all layers: for layer in partition: output = layer(output)
                def function(input: TensorOrTensors,
                             partition: nn.Sequential = partition,
                             skip_tracker: SkipTrackerThroughPotals = skip_trackers[i],
                             ) -> TensorOrTensors:
                    with use_skip_tracker(skip_tracker):
                        # HERE is where the forward pass happens!
                        # partition(input) calls nn.Sequential.__call__() which:
                        # 1. Iterates through each layer in the partition
                        # 2. Calls layer.forward() for each layer sequentially
                        # 3. Passes output of one layer as input to the next
                        return partition(input)

                chk = Checkpointing(function, batch)
                task = Task(streams[j], compute=chk.checkpoint, finalize=chk.recompute)
                del function, chk

            else:
                # WITHOUT CHECKPOINTING:
                # Define a compute function that will execute the partition's forward pass.
                def compute(batch: Batch = batch,
                            partition: nn.Sequential = partition,
                            skip_tracker: SkipTrackerThroughPotals = skip_trackers[i],
                            ) -> Batch:
                    with use_skip_tracker(skip_tracker):
                        # HERE is where the forward pass happens!
                        # batch.call(partition) internally calls partition(batch.value)
                        # which invokes nn.Sequential.__call__() that:
                        # 1. Iterates through each layer in the partition
                        # 2. Calls layer.forward() for each layer sequentially  
                        # 3. Passes output of one layer as input to the next
                        return batch.call(partition)

                task = Task(streams[j], compute=compute, finalize=None)
                del compute

            # Compute tasks in parallel. ([2] in the diagram)
            in_queues[j].put(task)

        for i, j in schedule:
            ok, payload = out_queues[j].get()

            # Hold the first exception.
            if exc_info is not None:
                continue
            elif not ok:
                exc_info = cast(ExcInfo, payload)
                continue

            task, batch = cast(Tuple[Task, Batch], payload)

            # The copy stream synchronizes to copy the output. ([3] in the
            # diagram)
            if j != n-1:
                wait(batch, streams[j], copy_streams[j][i])

            # Finalize tasks. If checkpointing is enabled, here the
            # recomputation is scheduled at backpropagation. ([4] in the
            # diagram)
            with use_device(devices[j]):
                task.finalize(batch)

            batches[i] = batch

        # Fail at the first exception.
        if exc_info is not None:
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])
