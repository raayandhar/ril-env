# Notes
This repository is largely credited to the Diffusion Policy authors,
whose evaluation and demo code I am transferring to work with the XArm
robot.

## Shared Memory
In order to bypass the Python GIL, most functionality inherits from
mp.process. In order to enable efficient sharing of data between processes,
we use Python's SharedMemory and SharedMemoryManager classes, from the
multiprocessing module. SharedMemory allows you to create, access and manage
blocks of memory that can be simultaneously accessed by multiple processes,
i.e. the sharing of numpy arrays between processes with low overhead. 
SharedMemoryManager is a high-level interface that helps manage the lifecycle
of shared memory objects. We use a SharedMemoryManager to manage all of our
SharedMemory objects. There are two main SharedMemory objects:
1. SharedMemoryQueue: lock-free FIFO queue that uses shared memory to efficiently pass numpy arrays between different processes. It uses atomic counters to track read and write positions, making it lock-free.
2. SharedMemoryRingBuffer: lock-free FIFO buffer that wraps around and starts writing over itself if the buffer has not been flushed. It is optimized for
scenarios when you need access to the most recent data.
3. SharedNDArray: a warapper arround numpy arrays to make them work with 
SharedMemory objects; enables allocation of memory that is accessible from
multiple processes.
