export PYTHONPATH="${PYTHONPATH}:${PWD}"
bash scripts/autoregressive/sample_c2i.sh --gpt-model GPT-L \
    --gpt-ckpt D-AR-L-360K.pt \
    --tokenizer-config configs/tokenizer_v1.yaml \
    --tokenizer-ckpt temp/tokenizer_v1.pt \
    --cfg-scale 1.2,8.0 \
    --top-p 1.0 \
    --top-k 0 \
    --temperature 1.0 \
    --num-fid-samples 50000 \
    --per-proc-batch-size 64 \
    --sample-dir ar_samples


python evaluations/c2i/evaluator.py VIRTUAL_imagenet256_labeled.npz ../dit_github_private/ar_samples/GPT-L-D-AR-L-360K-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.2,8.0-seed-0-None.npz
 