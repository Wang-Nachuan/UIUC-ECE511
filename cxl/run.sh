# python3 gemm.py
# numactl --membind 1,2,3 python3 gemm.py

echo 'DDR:'
# numactl -m 1 python3 gemm_torch.py
numactl -m 1 python3 gemm_torch_warmup.py
# echo 'CXL 1:'
# # numactl -m 2 python3 gemm_torch.py
# numactl -m 2 python3 gemm_torch_warmup.py
# echo 'CXL 1-2:'
# # numactl -i 2,3 python3 gemm_torch.py
# numactl -i 1,2 python3 gemm_torch_warmup.py

# nsys profile -o cxl_interleave_3 -f true python3 gemm_torch.py
