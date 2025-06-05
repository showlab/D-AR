# !/bin/bash
set -x

nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
python -m torch.distributed.run \
--nnodes=1 --nproc_per_node=$nproc_per_node --node_rank=0 \
--master_port=12344 \
tokenizer/tokenizer_image/reconstruction_vq_ddp.py \
"$@"