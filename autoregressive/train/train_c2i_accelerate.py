import os
import time
import random
import argparse
from glob import glob
from copy import deepcopy

import torch
import inspect
from torch.utils.data import DataLoader

from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from utils.logger import create_logger
from utils.ema import update_ema, requires_grad
from dataset.build import build_dataset
from autoregressive.models.gpt import GPT_models



from torchvision import transforms
from dataset.augmentation import center_crop_arr


from dar_tool import load_visual_tokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def create_optimizer(model, weight_decay, learning_rate, betas, logger):
    # filter trainable parameters
    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for p in param_dict.values() if p.dim() >= 2]
    nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    logger.info(f"num decayed tensors: {len(decay_params)}, params: {sum(p.numel() for p in decay_params):,}")
    logger.info(f"num non-decay tensors: {len(nodecay_params)}, params: {sum(p.numel() for p in nodecay_params):,}")

    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas,
        **({'fused': True} if fused_available else {})
    )
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer


def main():
    parser = argparse.ArgumentParser()
    # same args as original
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--cloud-save-path", type=str, required=True)
    parser.add_argument("--tokenizer-config", type=str, required=True)
    parser.add_argument("--tokenizer-ckpt", type=str, required=True)
    parser.add_argument("--no-local-save", action='store_true')
    parser.add_argument("--gpt-model", choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", choices=['c2i','t2i'], default="c2i")
    parser.add_argument("--ema", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--token-dropout-p", type=float, default=0.1)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--temp-ckpt", type=str, default="temp/ar.pt")
    parser.add_argument("--dataset", type=str, default='imagenet_code')
    parser.add_argument("--image-size", type=int, choices=[256,384,448,512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=60000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default='bf16')
    args = parser.parse_args()

    # Initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None,
        kwargs_handlers=[ddp_kwargs]
    )
    device = accelerator.device  # handles CPU/GPU automatically

    # Experiment dirs only on main process
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)
        idx = len(glob(f"{args.results_dir}/*"))
        model_name = args.gpt_model.replace('/', '-')
        exp_dir = f"{args.results_dir}/{idx:03d}-{model_name}"
        ckpt_dir = f"{exp_dir}/checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        logger = create_logger(exp_dir)
        logger.info(f"Experiment dir: {exp_dir}")
        tstamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cloud_base = f"{args.cloud_save_path}/{tstamp}/{idx:03d}-{model_name}/checkpoints"
        os.makedirs(cloud_base, exist_ok=True)
        logger.info(f"Cloud ckpt dir: {cloud_base}")
    else:
        logger = create_logger(None)

    logger.info(args)

    # seed
    torch.manual_seed(args.global_seed)

    vq_model = load_visual_tokenizer(args.tokenizer_config, args.tokenizer_ckpt, device=device)
    vocab_size = vq_model.vq_codebook_size
    seq_len = vq_model.all_queries

    # build model
    dropout = 0.0 if args.drop_path_rate > 0 else args.dropout_p
    model = GPT_models[args.gpt_model](
        vocab_size=vocab_size,
        block_size=seq_len,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout,
        ffn_dropout_p=dropout,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p
    )

    model = model.to(device)

    
    if args.ema:
        ema_model = deepcopy(model)
        requires_grad(ema_model, False)

    optimizer = create_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    # load checkpoint if specified
    start_epoch = 0
    global_steps = 0
    if args.gpt_ckpt:
        ckpt = torch.load(args.gpt_ckpt, map_location='cpu')

        model_weight = ckpt['model']
        unwanted_prefix = '_orig_mod.' 
        for k,v in list(model_weight.items()): 
            if k.startswith(unwanted_prefix): 
                model_weight[k[len(unwanted_prefix):]] = model_weight.pop(k) 
        print(model.load_state_dict(model_weight, strict=False))
        if args.ema:
            ema_model.load_state_dict(ckpt.get('ema', ckpt['model']))
        optimizer.load_state_dict(ckpt['optimizer'])
        global_steps = ckpt.get('steps', 0)
        # start_epoch = global_steps // (len(build_dataset(args)) // args.global_batch_size)
        start_epoch = 0
        del ckpt
        logger.info(f"Resumed from {args.gpt_ckpt}, at step {global_steps}, epoch {start_epoch}")

    # dataset & loader
    batch_size = args.global_batch_size // accelerator.num_processes
    crop_size = int(args.image_size * 1.1)
    transform = transforms.Compose([
        transforms.Lambda(lambda x: center_crop_arr(x, crop_size)),
        transforms.TenCrop(args.image_size),
        transforms.Lambda(lambda crops: random.choice(crops)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])


    from dataset.imagenet import build_imagenet
    dataset = build_imagenet(args, transform=transform)
    train_loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size // accelerator.num_processes,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )




    

    # prepare with accelerator
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    try:
        logger.info(f"Dataset contains  images ({len(dataset)}) batch size {batch_size} total iterations {len(train_loader)}")
    except Exception as e:
        logger.info(f"Error: {e}")

    if args.ema:
        ema_model = accelerator.prepare(ema_model)

    if not args.no_compile and hasattr(torch, 'compile'):
        model = torch.compile(model)

    # training loop
    total_steps = global_steps
    log_loss = 0.0
    log_steps = 0
    start_time = time.time()
    model.train()
    if args.ema:
        ema_model.eval()

    for epoch in range(start_epoch, args.epochs):
        for x, y in train_loader:
            x = x.to(device)
            with torch.no_grad():
                z = vq_model.extract_code(x)

            y = y.to(device)
            z = z.flatten(1)
            c = y.flatten()
            with accelerator.autocast():
                _, loss = model(cond_idx=c, idx=z[:,:-1], targets=z)
            accelerator.backward(loss)
            if args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            if args.ema:
                update_ema(ema_model, accelerator.unwrap_model(model))

            total_steps += 1
            log_loss += loss.item()
            log_steps += 1

            if total_steps % args.log_every == 0:
                accelerator.wait_for_everyone()
                elapsed = time.time() - start_time
                avg_loss = log_loss / log_steps
                if accelerator.is_main_process:
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"(step={total_steps}) loss={avg_loss:.4f}, lr={current_lr:.6f}, {log_steps/elapsed:.2f} steps/s")
                log_loss, log_steps = 0.0, 0
                start_time = time.time()

            if total_steps % args.ckpt_every == 0:
                if accelerator.is_main_process:
                    state = {
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'steps': total_steps,
                        'args': args
                    }
                    if args.ema:
                        state['ema'] = ema_model.state_dict()
                    local_path = os.path.join(ckpt_dir, f"{total_steps:07d}.pt")
                    if not args.no_local_save:
                        torch.save(state, local_path)
                        logger.info(f"Saved local ckpt to {local_path}")
                    cloud_path = os.path.join(cloud_base, f"{total_steps:07d}.pt")
                    torch.save(state, cloud_path)
                    logger.info(f"Saved cloud ckpt to {cloud_path}")
                accelerator.wait_for_everyone()

            if total_steps % 1000 == 0:
                if accelerator.is_main_process:
                    state = {
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'steps': total_steps,
                        'args': args
                    }
                    if args.ema:
                        state['ema'] = ema_model.state_dict()
                    # local_path = os.path.join(ckpt_dir, f"{total_steps:07d}.pt")
                    local_path = args.temp_ckpt
                    torch.save(state, local_path)
                    logger.info(f"Saved local ckpt to {local_path}")
                accelerator.wait_for_everyone()

            begin_steps = 300000
            end_steps = 360000
            if begin_steps <= total_steps <= end_steps:
                progress = (total_steps - begin_steps) / (end_steps - begin_steps)
                new_lr = args.lr * (1 - progress) + 1e-5 * progress
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

    # done
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Training complete.")


if __name__ == '__main__':
    main()
