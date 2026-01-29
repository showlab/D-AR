# Modified from:
#   taming-transformers:  https://github.com/CompVis/taming-transformers
#   muse-maskgit-pytorch: https://github.com/lucidrains/muse-maskgit-pytorch/blob/main/muse_maskgit_pytorch/vqgan_vae.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer.tokenizer_image.lpips import LPIPS
from tokenizer.tokenizer_image.vq_types import VQCodebookLoss, DiffusionAux
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize


def preprocess_raw_image(x, enc_type):
    assert 'dinov2' in enc_type
    x = 0.5*x + 0.5
    x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    height, width = x.size(2), x.size(3)
    new_height = math.ceil(height/256 * 224)
    new_width = math.ceil(width/256 * 224)
    x = torch.nn.functional.interpolate(x, (new_height, new_width), mode='bicubic')

    return x

class LossTracker:
    def __init__(self, window_size=2000) -> None:
        self.tracked_loss = []
        self.ema_loss = None
        self.window_size = window_size
        self.eta = 0.999
    
    def add_loss(self, loss):
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().item()
        self.tracked_loss.append(loss)
        self.tracked_loss = self.tracked_loss[-self.window_size:]
        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = self.ema_loss * self.eta + loss * (1.0-self.eta)
        # if len(self.tracked_loss) >= 10:  # Only process if we have enough samples
        #     num_to_remove = max(1, int(len(self.tracked_loss)*0.4))  # 10% of samples
        #     sorted_loss = sorted(self.tracked_loss)
        #     tracked_loss = sorted_loss[num_to_remove:-num_to_remove]  # Remove top and bottom 10%
        # else:
        #     tracked_loss = self.tracked_loss
        tracked_loss = self.tracked_loss
        self.ema_loss = sum(tracked_loss) / len(tracked_loss)
        return self.ema_loss



class VQLoss(nn.Module):
    def __init__(self, reconstruction_loss='l2', reconstruction_weight=1.0, 
                 codebook_weight=1.0, perceptual_weight=1.0, dino_weight=1.0):
        super().__init__()
        
        # perceptual loss
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.dino_weight = dino_weight

        # reconstruction loss
        if reconstruction_loss == "l1":
            self.rec_loss = F.l1_loss
        elif reconstruction_loss == "l2":
            self.rec_loss = F.mse_loss
        else:
            raise ValueError(f"Unknown rec loss '{reconstruction_loss}'.")
        self.rec_weight = reconstruction_weight

        # codebook loss
        self.codebook_weight = codebook_weight

        from tokenizer.tokenizer_image.utils_repa import load_encoders
        encoders, encoder_types, architectures = load_encoders('dinov2-vit-b', 'cpu')
        self.repa_encoder = encoders[0]
        self.repa_encoder.eval()
        self.repa_encoder.requires_grad_(False)
        self.repa_type = encoder_types[0]

        self.tracker1 = LossTracker()
        self.tracker2 = LossTracker()
        self.tracker3 = LossTracker()
        self.tracker4 = LossTracker()

    def forward(self, codebook_loss: VQCodebookLoss, inputs: torch.Tensor, 
                diff_aux: DiffusionAux, diff_loss: torch.Tensor,
                global_step: int, logger=None, log_every=100, vq_loss_start_step=None):
        """
        Compute the total loss for VQ-VAE training.
        
        Args:
            codebook_loss: VQCodebookLoss dataclass with VQ loss components
            inputs: Original input images
            reconstructions: DiffusionAux dataclass with reconstruction outputs
            diff_loss: Diffusion feature matching loss
            global_step: Current training step
            logger: Logger for metrics
            log_every: Logging frequency
            vq_loss_start_step: Step to start VQ loss (optional warmup)
        """
        repa_input = preprocess_raw_image(inputs, self.repa_type)
        repa_features = self.repa_encoder.forward_features(repa_input)['x_norm_patchtokens']

        cosp = F.cosine_similarity(repa_features, diff_aux.h_repa, dim=-1)
        loss_repa = (1.0-cosp).mean()

        if vq_loss_start_step is not None and global_step < vq_loss_start_step:
            vq_loss_weight = 0.0
        else:
            vq_loss_weight = 1.0
        vq_loss_weight = 1.0
        
        # perceptual loss
        p_loss = self.perceptual_loss(
            inputs.contiguous().float(),
            diff_aux.predict_x1.contiguous().float())

        te = diff_aux.t.view(-1, 1, 1, 1)
        if self.perceptual_weight >= 0:
            p_weight = 1.0
        else:
            p_weight = (te*2).float()
        p_loss = torch.mean(p_loss*p_weight)

        # Total loss computation
        loss = (vq_loss_weight * (codebook_loss.vq_loss + codebook_loss.commit_loss + codebook_loss.entropy_loss) + 
                diff_loss +
                abs(self.perceptual_weight) * p_loss +
                self.dino_weight * loss_repa)

        self.tracker1.add_loss(diff_loss)
        self.tracker2.add_loss(p_loss)
        self.tracker3.add_loss(loss_repa)
        
        if global_step % log_every == 0:
            logger.info(f"vq_loss: {codebook_loss.vq_loss:.4f}, commit_loss: {codebook_loss.commit_loss:.4f}, "
                       f"entropy_loss: {codebook_loss.entropy_loss:.4f}, codebook_usage: {codebook_loss.codebook_usage:.4f}, "
                       f"dead_rate: {codebook_loss.dead_code_rate:.4f}")
            logger.info(f"fm_loss: {self.tracker1.ema_loss:.4f}, perceptual_loss: {self.tracker2.ema_loss:.4f}, repa loss: {self.tracker3.ema_loss:.4f}")
        return loss