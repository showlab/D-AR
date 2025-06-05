# !/bin/bash
set -x

export PYTHONPATH="${PYTHONPATH}:${PWD}"
nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

torchrun \
--nnodes=1 --nproc_per_node=$nproc_per_node --node_rank=0 \
--master_port=12345 \
autoregressive/sample/sample_c2i_ddp.py \
"$@"
