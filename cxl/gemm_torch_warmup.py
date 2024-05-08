import numpy as np
import os
import time
import torch
def benchmark_func(func, number, repeat, warmup=100):
    for i in range(warmup):
        func()
    costs = [0]
    for i in range(repeat):
        torch.cuda.synchronize()
        tic = time.time()
        for i in range(number):
            func()
        torch.cuda.synchronize()
        costs.append((time.time() - tic) / number)
    return costs
def profile_bandwidth():
    s, h = 512, 512
    for b in [1, 128, 512]:
        dst_tensor = torch.ones((b, s, h), dtype=torch.int8, device="cuda:0")
        src_tensor = torch.ones((b, s, h), dtype=torch.int8, pin_memory=True)
        dst_indices = (slice(0, b), slice(0, s), slice(0, h))
        src_indices = (slice(0, b), slice(0, s), slice(0, h))
        def func():
            # if isinstance(src_tensor, str):
            #     src_tensor_ = torch.from_numpy(np.lib.format.open_memmap(src_tensor))
            # else:
            #     src_tensor_ = src_tensor
            # if isinstance(dst_tensor, str):
            #     dst_tensor_ = torch.from_numpy(np.lib.format.open_memmap(dst_tensor))
            # else:
            #     dst_tensor_ = dst_tensor
            # dst_tensor_[dst_indices].copy_(src_tensor_[src_indices])
            dst_tensor.copy_(src_tensor)
        size = np.prod([(x.stop - x.start) / (x.step or 1) for x in dst_indices])
        cost = np.mean(benchmark_func(func, number=5, repeat=3))
        bandwidth = size / cost / (1024 ** 3)
        print(f"size: {size / (1024 ** 2):6.2f} MB, bandwidth: {bandwidth:.3f} GB/s")
if __name__ == "__main__":
    profile_bandwidth()