# !/bin/bash
set -x
nnodes=1
nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
node_rank=0
master_addr=localhost
master_port=12477

# export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "cuda" | grep -v "cudnn" | tr '\n' ':')
export PYTHONPATH="${PYTHONPATH}:${PWD}"

torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
--master_addr=$master_addr --master_port=$master_port \
tokenizer/tokenizer_image/vq_train.py "$@"

