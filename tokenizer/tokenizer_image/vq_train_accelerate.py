import os
import time
import math
import random
import argparse
import warnings
from glob import glob
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from config_util import Config

from dar_tool import make_generic_dataset_loader

import accelerate
from accelerate import Accelerator

from accelerate.utils import InitProcessGroupKwargs, DistributedDataParallelKwargs
from datetime import timedelta


from utils.logger import create_logger
from utils.ema import update_ema, requires_grad
from dataset.augmentation import random_crop_arr, center_crop_arr
from dataset.build import build_dataset
from tokenizer.tokenizer_image.vq_model import VQModel
from tokenizer.tokenizer_image.vq_loss import VQLoss

warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#################################################################################
#                           Helper Functions                                    #
#################################################################################
def load_state_dict_compatible(model, state_dict):
    """
    Load the state_dict into the model, only loading parameters with matching shapes.
    """
    model_dict = model.state_dict()
    loaded_params = []
    skipped_params = []

    for name, param in state_dict.items():
        if name in model_dict:
            model_param = model_dict[name]
            if model_param.shape == param.shape:
                model_dict[name].copy_(param)
                loaded_params.append(name)
            else:
                skipped_params.append((name, model_param.shape, param.shape))
        else:
            skipped_params.append((name, None, param.shape))


    # Load the updated state dict into the model
    model.load_state_dict(model_dict)

    return {
        "loaded": loaded_params,
        "skipped": skipped_params,
    }

    
#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    # Initialize accelerator (it will handle device placement, distributed, and mixed precision)
    kwargs = DistributedDataParallelKwargs()
    timeout = InitProcessGroupKwargs()
    accelerator = Accelerator(mixed_precision=args.mixed_precision, kwargs_handlers=[kwargs, timeout])
    device = accelerator.device
    rank = accelerator.process_index  # accelerator.process_index is analogous to the distributed rank

    os.makedirs("temp", exist_ok=True)
    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.vq_model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cloud_results_dir = os.path.join(args.cloud_save_path, time_record)
        cloud_checkpoint_dir = os.path.join(cloud_results_dir, f"{experiment_index:03d}-{model_string_name}", "checkpoints")
        os.makedirs(cloud_checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
    else:
        logger = create_logger(None)

    logger.info(f"{args}")
    logger.info(f"Starting process_index={rank} on device {device}")

    # Set seeds (adding rank to the global seed)
    seed = args.global_seed + rank
    torch.manual_seed(seed)
    random.seed(seed)

    config = Config(args.config)
    logger.info(config)

    vq_model = VQModel(config['model'])
    logger.info(f"VQ Model Parameters: {sum(p.numel() for p in vq_model.parameters()):,}")
    if args.ema:
        ema = deepcopy(vq_model)
        ema.to(device)
        requires_grad(ema, False)
        logger.info(f"VQ Model EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")
    vq_model = vq_model.to(device)

    vq_loss = VQLoss(
        disc_start=args.disc_start, 
        disc_weight=args.disc_weight,
        dino_weight=args.dino_weight,
        disc_type=args.disc_type,
        disc_loss=args.disc_loss,
        gen_adv_loss=args.gen_loss,
        image_size=args.image_size,
        perceptual_weight=args.perceptual_weight,
        reconstruction_weight=args.reconstruction_weight,
        reconstruction_loss=args.reconstruction_loss,
        codebook_weight=args.codebook_weight,  
    ).to(device)

    vq_loss = vq_loss.to(device)
    logger.info(f"Discriminator Parameters: {sum(p.numel() for p in vq_loss.discriminator.parameters()):,}")


    # Setup optimizer
    learnable_parameters = []
    names = []
    for name, param in vq_model.named_parameters():
        if not param.requires_grad:
            continue
        learnable_parameters.append(param)
        names.append(name)

    disc_learnable_parameters = []
    names = []
    for name, param in vq_loss.discriminator.named_parameters():
        if not param.requires_grad:
            continue
        disc_learnable_parameters.append(param)
        names.append(name)

    optimizer = torch.optim.AdamW(learnable_parameters, lr=args.lr, betas=(args.beta1, args.beta2),
                                  weight_decay=args.weight_decay)
    optimizer_disc = torch.optim.AdamW(disc_learnable_parameters, lr=args.lr, betas=(args.beta1, args.beta2),
                                       weight_decay=args.weight_decay)

    def adjust_learning_rate(step, warmup_steps=10000, lr_cosine_end_steps=0, min_coeff=0.1):
        # In our experiments, we do not use lr cosine decay
        coeff = min(1.0, step / warmup_steps)
        if lr_cosine_end_steps > 0 and step > warmup_steps:
            progress = (step - warmup_steps) / lr_cosine_end_steps
            progress = min(progress, 1.0)  # Ensure progress does not exceed 1
            cosine_coeff = 0.5 * (1 + math.cos(math.pi * progress))
            coeff *= max(cosine_coeff, min_coeff)

        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = coeff * args.lr

    def get_current_lr():
        return optimizer.param_groups[0]["lr"]

    # Setup data:
    if args.data_path.startswith("wds"):
        transform = transforms.Compose(
            [
            transforms.Lambda(lambda x: random_crop_arr(x, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        dataset, loader = make_generic_dataset_loader(args.data_path,
            transform=transform, 
            batch_size=args.global_batch_size // accelerator.num_processes, 
            num_workers=args.num_workers, 
            global_batch_size=args.global_batch_size,
            target_type="recon")
    elif args.data_path.startswith("datasets"):
        transform = transforms.Compose(
            [
            transforms.Lambda(lambda x: random_crop_arr(x, args.image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        dataset, loader = make_generic_dataset_loader(args.data_path,
            transform=transform,
            accelerator=accelerator,
            batch_size=args.global_batch_size // accelerator.num_processes,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True)
    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        dataset = build_dataset(args, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=args.global_batch_size // accelerator.num_processes,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path}) iterations {len(loader):,} batch size {args.global_batch_size // accelerator.num_processes}")

    # Prepare for loading checkpoints/resuming training:
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
        model_ckpt = checkpoint["model"] if "model" in checkpoint else checkpoint

        msg = (load_state_dict_compatible(vq_model, model_ckpt))
        logger.info("Skipped params:")
        logger.info(msg['skipped'])

        if args.ema:
            try:
                ema.load_state_dict(checkpoint["ema"])
                optimizer.load_state_dict(checkpoint["optimizer"])
            except:
                logger.info("EMA load failed; initialize EMA; this is expected when you change the architecture")
                update_ema(ema, vq_model, decay=0)
        
        try:
            vq_loss.discriminator.load_state_dict(checkpoint["discriminator"])
            optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        except:
            logger.info("load GAN model failed")
            pass

        try:
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.vq_ckpt.split('/')[-1].split('.')[0])
        except:
            train_steps = 0
        
        if hasattr(dataset, "__len__"):
            start_epoch = int(train_steps / (len(dataset) / args.global_batch_size))
        else:
            start_epoch = 0

        del checkpoint

        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")

    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, vq_model, decay=0)
    
    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        vq_model = torch.compile(vq_model)  # requires PyTorch 2.0

    # Use accelerator.prepare to handle distributed training, mixed precision, etc.
    vq_model, vq_loss, optimizer, optimizer_disc, = accelerator.prepare(
        vq_model, vq_loss, optimizer, optimizer_disc
    )

    loader = accelerator.prepare_data_loader(loader)

    vq_model.train()
    vq_loss.train()
    if args.ema:
        ema.eval()  # EMA model is kept in eval mode

    # Variables for monitoring/logging:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        try:
            # If your DataLoader uses a sampler with set_epoch, update it here.
            if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)
            # for Huggingface datasets
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
        except Exception as e:
            logger.warning(f"Failed to set epoch for DataLoader: {e}, continuing without setting epoch.")
        logger.info(f"Beginning epoch {epoch}...")
        for imgs, label in loader:
            adjust_learning_rate(train_steps, args.warmup_steps, args.lr_cosine_end_steps)
            # (Accelerate automatically puts tensors on the correct device)

            # Generator training
            optimizer.zero_grad()
            with accelerator.autocast():
                recons_imgs, codebook_loss = vq_model(imgs)
                loss_gen = vq_loss(
                    codebook_loss, imgs, recons_imgs, optimizer_idx=0,
                    global_step=train_steps+1, last_layer=None,
                    logger=logger, log_every=args.log_every, use_gan=args.use_gan
                )
            accelerator.backward(loss_gen)

            if args.max_grad_norm != 0.0:
                if accelerator.sync_gradients:
                    accelerator.unscale_gradients(optimizer)
                    grad_norm = accelerator.clip_grad_norm_(vq_model.parameters(), args.max_grad_norm)

            else:
                grad_norm = 0.0

            optimizer.step()

            if args.ema:
                # Unwrap the model to update EMA
                update_ema(ema, accelerator.unwrap_model(vq_model), decay=0.999)

            # Discriminator training (currently commented out)
            if args.use_gan and train_steps >= args.disc_start:
                optimizer_disc.zero_grad()
                with accelerator.autocast():
                    loss_disc = vq_loss(
                        codebook_loss, imgs, recons_imgs, optimizer_idx=1,
                        global_step=train_steps+1, logger=logger, log_every=args.log_every, use_gan=args.use_gan
                    )
                accelerator.backward(loss_disc)
                if args.max_grad_norm != 0.0:
                    if accelerator.sync_gradients:
                        accelerator.unscale_gradients(optimizer_disc)
                        # grad_norm_disc = accelerator.clip_grad_norm_(vq_loss.parameters(), args.max_grad_norm)
                optimizer_disc.step()

            running_loss += loss_gen.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss_tensor = torch.tensor(running_loss / log_steps, device=device)
                # Reduce the average loss over all processes:
                avg_loss = accelerator.reduce(avg_loss_tensor, reduction="mean").item()
                vmem_used = torch.cuda.max_memory_allocated(device) / 1024**3
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, lr: {get_current_lr():.8f}, grad norm: {grad_norm.item():.4f} -> {args.max_grad_norm}, mem: {vmem_used:.2f}GB")
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    model_state = accelerator.unwrap_model(vq_model).state_dict()
                    checkpoint = {
                        "model": model_state,
                        "optimizer": optimizer.state_dict(),
                        "discriminator": accelerator.unwrap_model(vq_loss).discriminator.state_dict(),
                        "optimizer_disc": optimizer_disc.state_dict(),
                        "steps": train_steps,
                        "args": args,
                        "config": config,
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    if not args.no_local_save:
                        checkpoint_path = os.path.join(checkpoint_dir, f"{train_steps:07d}.pt")
                        accelerator.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    cloud_checkpoint_path = os.path.join(cloud_checkpoint_dir, f"{train_steps:07d}.pt")
                    torch.save(checkpoint, cloud_checkpoint_path)

                    logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
                accelerator.wait_for_everyone()

            if train_steps % 500 == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    model_state = accelerator.unwrap_model(vq_model).state_dict()
                    checkpoint = {
                        "model": model_state,
                        "optimizer": optimizer.state_dict(),
                        "discriminator": accelerator.unwrap_model(vq_loss).discriminator.state_dict(),
                        "optimizer_disc": optimizer_disc.state_dict(),
                        "steps": train_steps,
                        "args": args,
                        "config": config,
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()                    
                    temp_checkpoint_path = args.temp_ckpt
                    torch.save(checkpoint, temp_checkpoint_path)
                    logger.info(f"Saved temp checkpoint to {temp_checkpoint_path}")

                accelerator.wait_for_everyone()

    # Switch to eval mode for sampling or evaluation.
    vq_model.eval()
    logger.info("Done training!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-face-path", type=str, default=None, help="face datasets to improve vq model")
    parser.add_argument("--cloud-save-path", type=str, required=True, help="please specify a cloud disk path, if not, local path")
    parser.add_argument("--no-local-save", action='store_true', help="do not save checkpoints to local path")
    parser.add_argument("--vq-model", type=str, default="VQ-16") # just a placeholder
    parser.add_argument("--vq-ckpt", type=str, default=None, help="checkpoint path for resume training")
    parser.add_argument("--temp-ckpt", type=str, default="temp.pt", help="checkpoint path for resume training")
    parser.add_argument("--ema", action='store_true', help="whether to use EMA training")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="reconstruction loss weight of image pixel")
    parser.add_argument("--reconstruction-loss", type=str, default="l2", help="reconstruction loss type of image pixel")
    parser.add_argument("--perceptual-weight", type=float, default=0.5, help="perceptual loss weight of LPIPS")
    parser.add_argument("--dino-weight", type=float, default=0.5, help="perceptual loss weight of LPIPS")
    parser.add_argument("--disc-weight", type=float, default=0.5, help="discriminator loss weight for GAN training")
    parser.add_argument("--disc-start", type=int, default=20000, help="iteration to start discriminator training and loss")
    parser.add_argument("--disc-type", type=str, choices=["patchgan", "stylegan"], default="patchgan", help="discriminator type")
    parser.add_argument("--disc-loss", type=str, choices=["hinge", "vanilla", "non-saturating"], default="hinge", help="discriminator loss")
    parser.add_argument("--gen-loss", type=str, choices=["hinge", "non-saturating"], default="hinge", help="generator loss for GAN training")
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--use-gan", action="store_true", default=False)
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout probability")
    parser.add_argument("--results-dir", type=str, default="results_tokenizer_image")
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--image-size", type=int, choices=[128, 384, 256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 parameter for Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta2 parameter for Adam optimizer.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="maximum gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--lr-cosine-end-steps", type=int, default=0)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    args = parser.parse_args()
    main(args)

