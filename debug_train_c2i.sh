export PYTHONPATH="${PWD}:${PYTHONPATH}"


python -m accelerate.commands.launch \
    --num_processes 8 \
    --num_machines 1 \
    --main_process_ip 127.0.0.1 \
    --main_process_port 12478 \
    --machine_rank 0 \
    --mixed_precision bf16 \
    autoregressive/train/train_c2i_accelerate.py \
    --data-path /dev/shm/imagenet-val/ \
    --cloud-save-path results \
    --tokenizer-config configs/tokenizer_v1.yaml \
    --tokenizer-ckpt temp/tokenizer_v1.pt \
    --gpt-model GPT-L \
    --no-local-save \
    --log-every 50 \
    --global-batch-size 256