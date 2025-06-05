export PYTHONPATH="${PYTHONPATH}:${PWD}"
python autoregressive/sample/sample_c2i.py \
	--gpt-model GPT-L \
	--gpt-ckpt D-AR-L-360K.pt \
	--tokenizer-config configs/tokenizer_v1.yaml \
	--tokenizer-ckpt temp/tokenizer_v1.pt \
	--cfg-scale 4.0 \
	--top-p 1.0 \
	--top-k 0 \
	--temperature 1.0 \
	--seed 2 --preview 8
