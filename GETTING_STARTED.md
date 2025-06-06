## Code Structure
Here’s an overview at the code layout and key modules:

|module|path|
| -- | -- |
| Tokenizer train script | [vq_train_accelerate.py](tokenizer/tokenizer_image/vq_train_accelerate.py) |
| Sequential diffusion tokenizer | [vq_model.py](tokenizer/tokenizer_image/vq_model.py) |
| Sequential diffusion decoder | [diff_decoder.py](tokenizer/tokenizer_image/diff_decoder.py)|
| Tokenizer loss  | [vq_loss.py](tokenizer/tokenizer_image/vq_loss.py) |
| **---** | **---** |
| AR train script | [train_c2i_accelerate.py](autoregressive/train/train_c2i_accelerate.py) |
| AR Llama backbones | [gpt.py](autoregressive/models/gpt.py) (basically unchanged from LlamaGen) |
| AR generate logic | [generate.py](autoregressive/models/generate.py) |

## Getting Started
### Requirements
- PyTorch ≥ 2.1
- timm, accelerate, datasets
- xformers (optional)
- 80GB A100 GPUs (smaller batches for GPUs with lower VRAM)
- tensorflow for adm evaluation suite

### Preparation
```shell
mkdir -p temp # For storing temporary checkpoints
python tokenizer/tokenizer_image/utils_repa.py # Download REPA DINO and perform sanity check
```
**IMPORTANT:**
We use `accelerate` to train our sequential diffusion tokenizer and D-AR models on multi GPU nodes.
Please get yourself familiar with `accelerate` before proceeding.
While the provided demo training scripts are designed for single-node training, they can be easily configured for multi-node setups by modifying relevant parameters.
You may also need to modify `batch_size` or `global_batch_size` accordingly.

### Train or Finetune a Sequential Diffusion Tokenizer
The training script is provided as [debug_train_tokenizer.sh](debug_train_tokenizer.sh), based on `accelerate`.
```shell
bash debug_train_tokenizer.sh
```

You can simply finetune a sequential diffusion tokenizer from a checkpoint, e.g., `temp/tokenizer_v1.pt`, by appending this argument to the above script:
```
--vq-ckpt temp/tokenizer_v1.pt
```

We provide several dataset interface supports (webdataset, huggingface datasets, or simply folder). You can finetune our tokenizers with your own dataset by varying the `--data-path` argument. It can start with `wds://` or `datasets://` with remote streaming data loading (see [dar_tool.py](dar_tool.py) for details.)

### Evaluate the Sequential Diffusion Tokenizer
See [eval_recon_rfid.sh](eval_recon_rfid.sh).

### Visualize the Sequential Diffusion Tokenizer
See [sample_tokenizer.py](sample_tokenizer.py).

### Train a D-AR model
The training script is provided as [debug_train_c2i.sh](debug_train_c2i.sh), also based on `accelerate`. Note that we here tokenize images on the fly during training, which may not be optimal for training throughput.

### Evaluate D-AR models
See [eval_c2i_fid.sh](eval_c2i_fid.sh).


### Sample Images from D-AR models
See [sample_dar.sh](sample_dar.sh).