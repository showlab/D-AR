set -x
nnodes=1
nproc_per_node=8
node_rank=0
master_addr=localhost
master_port=12002

export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port
export PYTHONPATH="${PWD}:${PYTHONPATH}"


python -m accelerate.commands.launch \
    --num_processes $nproc_per_node \
    --num_machines $nnodes \
    --main_process_ip $master_addr \
    --main_process_port $master_port \
    --machine_rank $node_rank \
    --mixed_precision bf16 \
    tokenizer/tokenizer_image/vq_train_accelerate.py \
    --cloud-save-path results \
    --no-local-save \
    --data-path /path/to/dataset/ \
    --image-size 256 \
    --entropy-loss-ratio 0.0 \
    --global-batch-size 512 \
    --lr 2e-4 \
    --weight-decay 0.0 \
    --epochs 1000 \
    --ckpt-every 10000 \
    --num-workers 4 \
    --ema \
    --warmup-steps 500 \
    --beta2 0.95 \
    --perceptual-weight 0.5 \
    --dino-weight 0.5 \
    --config configs/tokenizer_v1.yaml \
    --temp-ckpt temp/temp.pt \
