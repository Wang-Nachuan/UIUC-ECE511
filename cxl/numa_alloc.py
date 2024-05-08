import ctypes
import numpy as np

numa_lib = ctypes.CDLL('./libnuma_alloc.so')
numa_lib.allocate_on_node.restype = ctypes.c_void_p
numa_lib.allocate_on_node.argtypes = [ctypes.c_size_t, ctypes.c_int]
numa_lib.free_on_node.argtypes = [ctypes.c_void_p]
numa_lib.get_numa_node_count.restype = ctypes.c_int

def allocate_numpy_array(shape, dtype, node):
    size = np.prod(shape) * np.dtype(dtype).itemsize
    ptr = numa_lib.allocate_on_node(size, node)
    if not ptr:
        raise MemoryError(f"Failed to allocate memory on NUMA node {node}")
    
    # Create a numpy array directly from the allocated memory
    buffer = (ctypes.c_char * size).from_address(ptr)
    array = np.frombuffer(buffer, dtype=dtype).reshape(shape)

    # Initialize array with random numbers similar to np.random.randn
    # Note: np.random.randn produces samples from the "standard normal" distribution.
    # array[:] = np.random.randn(*shape).astype(dtype)
    return array