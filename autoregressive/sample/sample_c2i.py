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
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate


from dar_tool import load_visual_tokenizer

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
    vq_model = load_visual_tokenizer(args.tokenizer_config, args.tokenizer_ckpt, device=device)

    print(f"image tokenizer is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    vocab_size = vq_model.vq_codebook_size
    seq_len = vq_model.all_queries
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=vocab_size,
        block_size=seq_len,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)
    
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.from_fsdp: # fspd
        model_weight = checkpoint
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        model_weight = checkpoint
        # raise Exception("please check model weight, maybe add --from-fsdp to run command")
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
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279] * 2
    # class_labels = [207, 360, 387, 388, ] * 2
    import random
    class_labels = [random.randint(0, 999) for i in range(8)]
    if args.label != -1:
        class_labels = [args.label for _ in range(16)]
    c_indices = torch.tensor(class_labels, device=device)
    # qzshape = [len(class_labels), args.codebook_embed_dim, latent_size, latent_size]

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
    samples = vq_model.decode_code(index_sample, clip_steps=clip_steps) # output value is between [-1, 1]
    # if clip_steps:
    #     samples = torch.nn.functional.interpolate(samples, size=(clip_steps*32, clip_steps*32), mode='bilinear', antialias=True)
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")

    # Save and display images:
    save_image(samples, "sample_{}.png".format(args.gpt_type), nrow=4, normalize=True, value_range=(-1, 1))
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
    parser.add_argument("--vq-model", type=str, default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
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