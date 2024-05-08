NUM_GPU_BATCHES=1
BATCH_SIZE=1

# PERF_COMMAND="sudo -E perf stat -a -e uncore_imc_0/cas_count_read/,uncore_imc_0/cas_count_write/,uncore_imc_1/cas_count_read/,uncore_imc_1/cas_count_write/ --"
# PERF_COMMAND="sudo -E perf mem record -a --"

export TRANSFORMERS_CACHE=/storage/nachuan3/huggingface

# DDR
echo "==================== DDR ===================="
numactl -m 1 python3 -m flexgen.flex_opt --model facebook/opt-30b --cpu-cache-compute --overlap False --num-gpu-batches $NUM_GPU_BATCHES --prompt-len 32 --gen-len 32 --gpu-batch-size $BATCH_SIZE --percent 0 100 0 100 0 100

echo ""

# # CXL (single)
# echo "=============== CXL (single) ================"
# numactl -m 2 python3 -m flexgen.flex_opt --model facebook/opt-30b  --cpu-cache-compute --overlap False --num-gpu-batches $NUM_GPU_BATCHES --prompt-len 32 --gen-len 32 --gpu-batch-size $BATCH_SIZE --percent 0 100 0 100 0 100

# echo ""

# # CXL (dual, numactl interleaving)
# echo "=========== CXL (dual, numactl) ============="
# numactl -i 2,3 python3 -m flexgen.flex_opt --model facebook/opt-30b --cpu-cache-compute --overlap False --num-gpu-batches $NUM_GPU_BATCHES --prompt-len 32 --gen-len 32 --gpu-batch-size $BATCH_SIZE --percent 0 100 0 100 0 100

# echo ""

# CXL (dual)
# export LD_PRELOAD=/storage/nachuan3/SMDK/lib/smdk_allocator/lib/libcxlmalloc.so
# export CXLMALLOC_CONF=use_exmem:true,priority:exmem,exmem_size:256G,normal_size:0,maxmemory_policy:interleave,use_auto_arena_scaling:true,exmem_partition_range:all
# # export CXLMALLOC_CONF=use_exmem:true,priority:exmem,exmem_size:256G,normal_size:0G,maxmemory_policy:oom,use_auto_arena_scaling:false,exmem_partition_range:2,3,use_adaptive_interleaving:true,interleave_node:0.5-0.5
# # export CXLMALLOC_CONF=use_exmem:true,exmem_size:131072,normal_size:2048,priority:normal,maxmemory_policy:remain,use_auto_arena_scaling:false
# # export CXLMALLOC_CONF=use_exmem:true,use_adaptive_interleaving:true,adaptive_interleaving_policy:bw_saturation
# echo "============= CXL (dual, SPDK) =============="
# numactl -m 2,3 python3 -m flexgen.flex_opt --model facebook/opt-1.3b --num-gpu-batches 1 --prompt-len 1024 --gen-len 32 --gpu-batch-size $BATCH_SIZE --percent 0 100 0 100 0 100

# echo ""