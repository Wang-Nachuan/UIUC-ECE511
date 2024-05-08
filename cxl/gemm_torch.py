import numpy as np
import os
import time
import torch


def profile_bandwidth():
    b, s, h = 512, 512, 512
    number = 10
    repeat = 10
    warmup = 100
    size = b * s * h * number / (1024**3)

    # Create random matrices on CPU and immediately delete the CPU data
    t_cpu = torch.ones((b, s, h), dtype=torch.int8, pin_memory=True)
    t_gpu = torch.ones((b, s, h), dtype=torch.int8, device="cuda:0")
    t_cpu_indices = (slice(0, b), slice(0, s), slice(0, h))
    t_cpu_indices = (slice(0, b), slice(0, s), slice(0, h))

    def memcpy():
        t_gpu.copy_(t_cpu)

    # Warming up
    for _ in range(warmup):
        memcpy()

    # Start timing
    costs = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(number):
            memcpy()
        torch.cuda.synchronize()
        end_time = time.time()
        costs.append(end_time - start_time)

    total_time = np.mean(costs)
    total_bw = size / total_time

    # print(f"Time: {total_time:.3f} s")
    print(f"Bandwidth: {total_bw:.3f} GB/s")

    return total_bw

if __name__ == "__main__":
    profile_bandwidth()