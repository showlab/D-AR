# D-AR: Diffusion via Autoregressive Models
> [Ziteng Gao](https://sebgao.github.io/), [Mike Zheng Shou](https://sites.google.com/view/showlab)
> <br>Show Lab, National University of Singapore<br>

[[arxiv]](https://arxiv.org/abs/2505.23660)

# News
- Jun 5, 2025: The initial code for training and inference is released. See [GETTING_STARTED.md](GETTING_STARTED.md) and give it a try now! Most cases have been tested but if you find bugs, feel free to open an issue.


# Overview
**Diffusion via Autoregressive models (D-AR)** recast diffusion on pixels to sequential token generation with a Llama backbone by the standard next token prediction.

The hyphen - in *D-AR* means symbolizes the bridge between diffusion and autoregressive modeling. However, feel free to refer to it simply as DAR.

*D-AR* is a long-term project. We're actively developing improved tokenizers trained on larger datasets with higher resolutions, and exploring text-to-image generation models. Stay tuned for updates!

# Pretrained Models
**Sequential diffusion tokenizer** trained on ImageNet:
| model | Params | rFID | resolution |
|-------|  ----  |------| ---------- |
| [DAR tokenizer v1](https://huggingface.co/showlab/D-AR/resolve/main/D-AR-tokenizer_v1.pt) | 300M | 1.52 | 256x256 |

**DAR models** trained on ImageNet 256x256:
| model | Params | tokenizer | gFID | IS |
|-------|  ----  | --------- | ------| -- |
| [DAR-L](https://huggingface.co/showlab/D-AR/resolve/main/D-AR-L-360K.pt) | 343M | DAR tokenizer v1|  2.44 | 262.9 |
| [DAR-XL](https://huggingface.co/showlab/D-AR/resolve/main/D-AR-XL-360K.pt) | 775M | DAR tokenizer v1 | 2.09 | 298.4 |


# Getting Started
See [GETTING_STARTED.md](GETTING_STARTED.md) for installation and script usage details.

# License
The majority of this project is licensed under MIT License. Portions of the project are available under separate license of referred projects, detailed in corresponding files.


# Acknowledgement
Our codebase is main based on [LlamaGen](https://github.com/FoundationVision/LlamaGen/tree/main/tokenizer), and incorporates components from several existing repositories. We gratefully acknowledge the contributions of the community and these codebases, without which our codebase would not have been built such clearly:
```
tokenizer/tokenizer_image/utils_repa.py from https://github.com/sihyun-yu/REPA
tokenizer/tokenizer_image/tokenizer_transformer.py rope-relevant stuff from https://github.com/black-forest-labs/flux
tokenizer/tokenizer_image/patching.py from https://github.com/NVIDIA/Cosmos-Tokenizer
```

# BibTeX
```bibtex
@article{gao25dar,
  title={D-AR: Diffusion via Autoregressive Models},
  author={Ziteng Gao and Mike Zheng Shou},
  journal={arXiv 2505.23660},
  year={2025}
}
```
