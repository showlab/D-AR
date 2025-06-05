# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from torchvision.utils import save_image

import time
import argparse
from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate


from config_util import Config
from tokenizer.tokenizer_image.vq_model import VQ_models, VQModel

def main(args):
    # Setup PyTorch:
    if args.seed == -1:
        import random
        torch.manual_seed(random.randint(0, 2048))
    else:
        torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load model
    tokenizer_config = Config(args.tokenizer_config)
    tokenizer_ckpt = torch.load(args.tokenizer_ckpt, map_location="cpu", weights_only=False)
    tokenizer_config = tokenizer_ckpt['config'] if 'config' in tokenizer_ckpt else tokenizer_config
    print(tokenizer_config)
    # create and load model
    vq_model = VQModel(tokenizer_config['model'])
    vq_model.to(device)
    vq_model.eval()

    print(vq_model.load_state_dict(tokenizer_ckpt['ema'], strict=False))
    vq_model = vq_model.to(device)

    del tokenizer_ckpt
    print(f"image tokenizer is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)
    
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=False)
    if args.from_fsdp: # fspd
        model_weight = checkpoint
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
    # if 'freqs_cis' in model_weight:
    #     model_weight.pop('freqs_cis')
    unwanted_prefix = '_orig_mod.' 
    for k,v in list(model_weight.items()): 
        if k.startswith(unwanted_prefix): 
            model_weight[k[len(unwanted_prefix):]] = model_weight.pop(k) 
    print(gpt_model.load_state_dict(model_weight, strict=False))
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 

    # Labels to condition the model with (feel free to change):
    # class_labels = [360,] * 1
    # class_labels = [207, 360, 387, 388, ] * 2
    import random
    # class_labels = [207, 951, 387, 291, 100, 130, 812, 475] + [814, 873, 355, 309, ]
    class_labels = [random.randint(0, 999) for _ in range(8)]
    if args.label != -1:
        class_labels = [args.label for _ in range(1)]
    c_indices = torch.tensor(class_labels, device=device)
    # qzshape = [len(class_labels), args.codebook_embed_dim, latent_size, latent_size]

    seq_len = 256

    t1 = time.time()
    index_sample = generate(
        gpt_model, c_indices, seq_len,
        cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
        temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, sample_logits=True, 
        )
    sampling_time = time.time() - t1
    print(f"gpt sampling takes about {sampling_time:.2f} seconds.")    
    
    print(index_sample.float().std(0))
    t2 = time.time()
    if args.preview > 0:
        clip_steps = args.preview
    else:
        clip_steps = None
    samples = []
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        torch.manual_seed(0)
        sample = vq_model.decode_code(index_sample, clip_steps=i) # output value is between [-1, 1]
        samples.append(sample)
    samples = torch.cat(samples, dim=0)
    # samples = vq_model.decode_code(index_sample, clip_steps=clip_steps)
    # if clip_steps:
    #     samples = torch.nn.functional.interpolate(samples, size=(clip_steps*32, clip_steps*32), mode='bilinear', antialias=True)
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")

    # Save and display images:
    save_image(samples, "sample_{}.png".format(args.gpt_type), nrow=9, normalize=True, value_range=(-1, 1))
    print(f"image is saved to sample_{args.gpt_type}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--tokenizer-config", type=str, required=True)
    parser.add_argument("--tokenizer-ckpt", type=str, required=True)
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=str, default="4.0")
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview", type=int, default=0)
    parser.add_argument("--label", type=int, default=-1)
    parser.add_argument("--top-k", type=int, default=2000,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    args = parser.parse_args()
    main(args)