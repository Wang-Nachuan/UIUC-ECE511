import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
import numpy as np
from skcuda import cublas
from pycuda.driver import Stream

from numa_alloc import *


def mult(matrices_A, matrices_B, size, num_matrices):
    # Initialize cuBLAS context
    cublas_handle = cublas.cublasCreate()

    # Create streams for overlapping data transfers and computation
    stream_for_data_transfer_A = Stream()
    stream_for_data_transfer_B = Stream()
    stream_for_computation = Stream()

    # Creating events for timing
    start_event = cuda.Event()
    stop_event = cuda.Event()

    # Lists to hold GPU data
    results = []

    alpha = np.float32(1.0)
    beta = np.float32(0.0)

    # Start timing just before data transfer to GPU
    start_event.record(stream_for_data_transfer_A)

    for h_A, h_B in zip(matrices_A, matrices_B):
        # Asynchronously transfer to GPU
        d_A = gpuarray.to_gpu_async(h_A, stream=stream_for_data_transfer_A)
        d_B = gpuarray.to_gpu_async(h_B, stream=stream_for_data_transfer_B)
        d_C = gpuarray.empty((size, size), np.float32)

        # Enqueue the matrix multiplication in the computation stream
        cublas.cublasSetStream(cublas_handle, stream_for_computation.handle)
        cublas.cublasSgemm(cublas_handle, 'n', 'n', size, size, size, alpha,
                           d_A.gpudata, size, d_B.gpudata, size, beta, d_C.gpudata, size)

        results.append((d_C, stream_for_computation))

    # Stop timing after the last computation has been enqueued
    stop_event.record(stream_for_computation)
    stop_event.synchronize()

    # Calculate the elapsed time
    elapsed_time_ms_A = stop_event.time_since(start_event)
    print(f"Total GPU execution time: {elapsed_time_ms_A/1000} s")

    # Clean up resources
    cublas.cublasDestroy(cublas_handle)

def copy(matrices_A, matrices_B, size, num_matrices):
    # Initialize cuBLAS context
    cublas_handle = cublas.cublasCreate()

    # Create streams for overlapping data transfers
    stream_for_data_transfer_A = Stream()
    stream_for_data_transfer_B = Stream()

    # Creating events for timing
    start_event = cuda.Event()
    stop_event = cuda.Event()

    # Start timing just before data transfer to GPU
    start_event.record(stream_for_data_transfer_A)

    for h_A, h_B in zip(matrices_A, matrices_B):
        # Asynchronously transfer to GPU
        d_A = gpuarray.to_gpu_async(h_A, stream=stream_for_data_transfer_A)
        d_B = gpuarray.to_gpu_async(h_B, stream=stream_for_data_transfer_B)

        # Wait for the last data transfer to complete
        d_A.get_async(stream=stream_for_data_transfer_A)
        d_B.get_async(stream=stream_for_data_transfer_B)

    # Stop timing after the last data transfer has completed
    stop_event.record(stream_for_data_transfer_B)
    stop_event.synchronize()

    # Calculate the elapsed time
    elapsed_time_ms = stop_event.time_since(start_event)
    print(f"Total data transfer time: {elapsed_time_ms/1000} s")

    # Clean up resources
    cublas.cublasDestroy(cublas_handle)

if __name__ == "__main__":
    # size = 32768
    size = 128
    num_mtx = 2000
    is_copy = True
    
    node_count = numa_lib.get_numa_node_count()
    print(f"Number of NUMA nodes: {node_count}")

    nDevices = cuda.Device.count()
    
    # Loop through all devices and print their properties
    for i in range(nDevices):
        device = cuda.Device(i)
        print(f"Device Number: {i}")
        print(f"  Device name: {device.name()}")
        # The attribute to check if the device supports overlapping copy and execution
        device_overlap = device.get_attribute(cuda.device_attribute.ASYNC_ENGINE_COUNT)
        print(f"  Async Engine Count (indicates number of DMA engines for overlap): {device_overlap}")

    
    print('\n-------------------- DDR --------------------')
    mtx_A1 = [allocate_numpy_array((size, size), np.float32, 1) for _ in range(num_mtx)]
    mtx_B1 = [allocate_numpy_array((size, size), np.float32, 1) for _ in range(num_mtx)]
    if is_copy:
        copy(mtx_A1, mtx_B1, size, num_mtx)
    else:
        mult(mtx_A1, mtx_B1, size, num_mtx)
    

    print('\n------------------- CXL 1 -------------------')
    mtx_A2 = [allocate_numpy_array((size, size), np.float32, 2) for _ in range(num_mtx)]
    mtx_B2 = [allocate_numpy_array((size, size), np.float32, 2) for _ in range(num_mtx)]
    if is_copy:
        copy(mtx_A1, mtx_B1, size, num_mtx)
    else:
        mult(mtx_A1, mtx_B1, size, num_mtx)

    print('\n------------------- CXL 2 -------------------')
    mtx_A3 = [allocate_numpy_array((size, size), np.float32, 3) for _ in range(num_mtx)]
    mtx_B3 = [allocate_numpy_array((size, size), np.float32, 3) for _ in range(num_mtx)]
    if is_copy:
        copy(mtx_A1, mtx_B1, size, num_mtx)
    else:
        mult(mtx_A1, mtx_B1, size, num_mtx)

    print('\n------------------ CXL 1-2 ------------------')
    mtx_A4 = [allocate_numpy_array((size, size), np.float32, 2) for _ in range(num_mtx)]
    mtx_B4 = [allocate_numpy_array((size, size), np.float32, 3) for _ in range(num_mtx)]
    if is_copy:
        copy(mtx_A1, mtx_B1, size, num_mtx)
    else:
        mult(mtx_A1, mtx_B1, size, num_mtx)

    print('')
