# Modified from:
#   taming-transformers:  https://github.com/CompVis/taming-transformers
#   muse-maskgit-pytorch: https://github.com/lucidrains/muse-maskgit-pytorch/blob/main/muse_maskgit_pytorch/vqgan_vae.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer.tokenizer_image.lpips import LPIPS
from tokenizer.tokenizer_image.discriminator_patchgan import NLayerDiscriminator as PatchGANDiscriminator
from tokenizer.tokenizer_image.discriminator_stylegan import Discriminator as StyleGANDiscriminator
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize


def preprocess_raw_image(x, enc_type):
    if 'clip' in enc_type:
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = 0.5*x + 0.5
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')
    elif 'dinov1' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')

    return x
    

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def non_saturating_d_loss(logits_real, logits_fake, weight=None):
    if weight is None:
        weight = 1.0
    loss_real = torch.mean(weight * F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real)))
    loss_fake = torch.mean(weight * F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake)))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def hinge_gen_loss(logit_fake):
    return -torch.mean(logit_fake)


def non_saturating_gen_loss(logit_fake):
    return torch.mean(F.binary_cross_entropy_with_logits(logit_fake, torch.ones_like(logit_fake)))


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

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
    def __init__(self, disc_start, disc_loss="hinge", disc_dim=64, disc_type='patchgan', image_size=256,
                 disc_num_layers=3, disc_in_channels=3, disc_weight=1.0, disc_adaptive_weight = False,
                 gen_adv_loss='hinge', reconstruction_loss='l2', reconstruction_weight=1.0, 
                 codebook_weight=1.0, perceptual_weight=1.0,  dino_weight=1.0,
    ):
        super().__init__()
        # discriminator loss
        disc_loss = "non-saturating"
        gen_adv_loss = "non-saturating"
        disc_type = "patchgan"
        assert disc_type in ["patchgan", "stylegan"]
        assert disc_loss in ["hinge", "vanilla", "non-saturating"]
        if disc_type == "patchgan":
            self.discriminator = PatchGANDiscriminator(
                input_nc=3, 
                n_layers=disc_num_layers,
                ndf=128,
            )
        elif disc_type == "stylegan":
            self.discriminator = StyleGANDiscriminator(
                input_nc=3, 
                image_size=image_size,
            )
        else:
            raise ValueError(f"Unknown GAN discriminator type '{disc_type}'.")

        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "non-saturating":
            self.disc_loss = non_saturating_d_loss
        else:
            raise ValueError(f"Unknown GAN discriminator loss '{disc_loss}'.")
        self.discriminator_iter_start = disc_start
        self.disc_weight = disc_weight
        self.disc_adaptive_weight = disc_adaptive_weight

        assert gen_adv_loss in ["hinge", "non-saturating"]
        # gen_adv_loss
        if gen_adv_loss == "hinge":
            self.gen_adv_loss = hinge_gen_loss
        elif gen_adv_loss == "non-saturating":
            self.gen_adv_loss = non_saturating_gen_loss
        else:
            raise ValueError(f"Unknown GAN generator loss '{gen_adv_loss}'.")

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

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight.detach()

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx, global_step, last_layer=None, 
                logger=None, log_every=100, use_gan=False, vq_loss_start_step=None):
        predict_x1, xt, t, h_repa, fake, real = reconstructions
        repa_input = preprocess_raw_image(inputs, self.repa_type)
        repa_features = self.repa_encoder.forward_features(repa_input)['x_norm_patchtokens']

        cosp = F.cosine_similarity(repa_features, h_repa, dim=-1)
        loss_repa = (1.0-cosp).mean()


        if vq_loss_start_step is not None and global_step < vq_loss_start_step:
            vq_loss_weight = 0.0
        else:
            vq_loss_weight = 1.0
        vq_loss_weight = 1.0
        t_weight = t.view(-1, 1)
        # generator update
        if optimizer_idx == 0:
            # reconstruction loss

            p_loss = self.perceptual_loss(inputs.contiguous().float(), predict_x1.contiguous().float())
            te = t.view(-1, 1, 1, 1)
            if self.perceptual_weight >= 0:
                p_weight = 1.0
            else:
                p_weight = (te*2).float()
            p_loss = torch.mean(p_loss*p_weight)

            # discriminator loss
            if use_gan:
                fake = fake.contiguous()
                logits_fake = self.discriminator(fake)
                generator_adv_loss = self.gen_adv_loss(logits_fake)
                
                disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)
            else:
                generator_adv_loss = 0
                disc_weight = 0
            
            fm_loss = codebook_loss[-1]

            loss =  vq_loss_weight * (codebook_loss[0] + codebook_loss[1] + codebook_loss[2]) + fm_loss + abs(self.perceptual_weight) * p_loss + disc_weight * generator_adv_loss + self.dino_weight * loss_repa

            self.tracker1.add_loss(fm_loss)
            self.tracker2.add_loss(p_loss)
            self.tracker3.add_loss(loss_repa)
            
            if global_step % log_every == 0:
                logger.info(f"vq_loss: {codebook_loss[0]:.4f}, commit_loss: {codebook_loss[1]:.4f}, entropy_loss: {codebook_loss[2]:.4f}, codebook_usage: {codebook_loss[3]:.4f}, dead_rate: {codebook_loss[4]:.4f}")
                logger.info(f"fm_loss: {self.tracker1.ema_loss:.4f}, perceptual_loss: {self.tracker2.ema_loss:.4f}, repa loss: {self.tracker3.ema_loss:.4f}")
            return loss

        # discriminator update
        if use_gan and optimizer_idx == 1:
            logits_real = self.discriminator(real.detach())
            logits_fake = self.discriminator(fake.detach())

            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)
            d_adversarial_loss = (self.disc_loss(logits_real, logits_fake, weight=t_weight))
            
            if global_step % log_every == 0:
                logits_real = logits_real.detach().mean()
                logits_fake = logits_fake.detach().mean()
                logger.info(f"(Discriminator) " 
                            f"discriminator_adv_loss: {d_adversarial_loss:.4f}, disc_weight: {disc_weight:.4f}, "
                            f"logits_real: {logits_real:.4f}, logits_fake: {logits_fake:.4f}")
            return d_adversarial_loss