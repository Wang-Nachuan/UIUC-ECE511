import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from numa_alloc import allocate_numpy_array
import time

def main():
    cuda.init()

    size = 38  # GB
    row_size = size * 256 * 1024 * 64
    row_size_half = size * 128 * 1024 * 64
    col_size = 16

    stream1 = cuda.Stream()
    stream2 = cuda.Stream()

    start_event1 = cuda.Event()
    end_event1 = cuda.Event()
    start_event2 = cuda.Event()
    end_event2 = cuda.Event()

    print(f"Data transfer size: {size} GB")

    # print('\n-------------------- DDR --------------------')
    # data_ddr = np.empty((row_size, col_size), dtype=np.float32)
    # # data_ddr = cuda.register_host_memory(np.empty((row_size, col_size), dtype=np.float32))
    # gpu_buf1 = cuda.mem_alloc(data_ddr.nbytes)

    # start_event1.record(stream=stream1)
    # cuda.memcpy_htod_async(gpu_buf1, data_ddr, stream1)
    # end_event1.record(stream=stream1)
    # stream1.synchronize()
    
    # bandwidth = size / (start_event1.time_till(end_event1) * 1e-3)
    # print(f"Data transfer bandwidth: {bandwidth:.2f} GB/s")

    # # Free resources
    # del data_ddr
    # gpu_buf1.free()

    # # print('\n-------------------- DDR 1 --------------------')
    # # data_ddr1_half = np.empty((row_size_half, col_size), dtype=np.float32)
    # # data_ddr2_half = np.empty((row_size_half, col_size), dtype=np.float32)
    # # # data_ddr1_half = cuda.register_host_memory(np.empty((row_size_half, col_size), dtype=np.float32))
    # # # data_ddr2_half = cuda.register_host_memory(np.empty((row_size_half, col_size), dtype=np.float32))
    # # gpu_buf1 = cuda.mem_alloc(data_ddr1_half.nbytes)
    # # gpu_buf2 = cuda.mem_alloc(data_ddr2_half.nbytes)

    # # start_event1.record(stream=stream1)
    # # cuda.memcpy_htod_async(gpu_buf1, data_ddr1_half, stream1)
    # # end_event1.record(stream=stream1)

    # # start_event2.record(stream=stream2)
    # # cuda.memcpy_htod_async(gpu_buf2, data_ddr2_half, stream2)
    # # end_event2.record(stream=stream2)

    # # stream1.synchronize()
    # # stream2.synchronize()
    
    # # bandwidth_1 = size / 2 / (start_event1.time_till(end_event1) * 1e-3)
    # # bandwidth_2 = size / 2 / (start_event2.time_till(end_event2) * 1e-3)
    # # bandwidth_total = size / (start_event1.time_till(end_event2) * 1e-3)
    # # print(f"Data transfer bandwidth (stream1): {bandwidth_1:.2f} GB/s")
    # # print(f"Data transfer bandwidth (stream2): {bandwidth_2:.2f} GB/s")
    # # print(f"Data transfer bandwidth (total): {bandwidth_total:.2f} GB/s")

    # # # Free resources
    # # del data_ddr1_half, data_ddr2_half
    # # gpu_buf1.free()
    # # gpu_buf2.free()

    # print('\n------------------- CXL 1 -------------------')
    # data_cxl1 = allocate_numpy_array((row_size, col_size), np.float32, 2)
    # gpu_buf1 = cuda.mem_alloc(data_cxl1.nbytes)

    # start_event1.record(stream=stream1)
    # cuda.memcpy_htod_async(gpu_buf1, data_cxl1, stream1)
    # end_event1.record(stream=stream1)
    # stream1.synchronize()

    # bandwidth = size / (start_event1.time_till(end_event1) * 1e-3)
    # print(f"Data transfer bandwidth: {bandwidth:.2f} GB/s")

    # # Free resources
    # del data_cxl1
    # gpu_buf1.free()

    print('\n------------------ CXL 1-2 ------------------')
    data_cxl1_half = allocate_numpy_array((row_size_half, col_size), np.float32, 2)
    data_cxl2_half = allocate_numpy_array((row_size_half, col_size), np.float32, 3)

    gpu_buf1 = cuda.mem_alloc(data_cxl1_half.nbytes)
    gpu_buf2 = cuda.mem_alloc(data_cxl2_half.nbytes)

    start_event1.record(stream=stream1)
    cuda.memcpy_htod_async(gpu_buf1, data_cxl1_half, stream1)
    end_event1.record(stream=stream1)

    start_event2.record(stream=stream2)
    cuda.memcpy_htod_async(gpu_buf2, data_cxl2_half, stream2)
    end_event2.record(stream=stream2)

    stream1.synchronize()
    stream2.synchronize()

    transfer_time1 = start_event1.time_till(end_event1) * 1e-3
    transfer_time2 = start_event2.time_till(end_event2) * 1e-3
    transfer_time_total = start_event1.time_till(end_event2) * 1e-3

    print(f"s1-start to s1-end {transfer_time1} ms")
    print(f"s2-start to s2-end {transfer_time2} ms")
    print(f"s1-start to s2-end {transfer_time_total} ms")

    bandwidth_s1 = size / 2 / transfer_time1
    bandwidth_s2 = size / 2 / transfer_time2
    bandwidth_total = size / transfer_time_total
    print(f"Data transfer bandwidth (stream1): {bandwidth_s1:.2f} GB/s")
    print(f"Data transfer bandwidth (stream2): {bandwidth_s2:.2f} GB/s")
    print(f"Data transfer bandwidth (total): {bandwidth_total:.2f} GB/s")

    # Free resources
    del data_cxl1_half, data_cxl2_half
    gpu_buf1.free()
    gpu_buf2.free()


if __name__ == "__main__":
    main()
