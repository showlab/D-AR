export PYTHONPATH="${PWD}:${PYTHONPATH}"

PATH_TO=/dev/shm/
# you need to prepare the imagenet validation set in $PATH_TO/imagenet-val

bash scripts/tokenizer/val.sh --data-path $PATH_TO/imagenet-val --image-size 256 --sample-dir $PATH_TO/temp/ 
bash scripts/tokenizer/reconstruction_vq.sh --data-path $PATH_TO/imagenet-val --image-size 256 --vq-ckpt temp/tokenizer_v1.pt --config configs/tokenizer_v1.yaml --per-proc-batch-size 64 --sample-dir $PATH_TO/temp/ --use-ema

# use tensorflow environment then (adm evaluation suite)

# python evaluations/c2i/evaluator.py $PATH_TO/temp/val_imagenet.npz VQ-16-imagenet-size-256-size-256-codebook-size-16384-dim-8-seed-0.npz

# or simply use pytorch-fid: pip install pytorch-fid

python -m pytorch_fid $PATH_TO/temp/VQ-16-imagenet-size-256-size-256-codebook-size-16384-dim-8-seed-0 $PATH_TO/temp/gt --device cuda:0
