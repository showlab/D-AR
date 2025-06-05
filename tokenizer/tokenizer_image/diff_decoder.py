import math

import torch
import torch.nn as nn
import torchvision.transforms.functional as VF
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F
from torch.distributions import Normal

from einops import rearrange

from timm.layers.mlp import Mlp

from tokenizer.tokenizer_image.patching import Patcher, UnPatcher
from tokenizer.tokenizer_image.tokenizer_transformer import Attention, EmbedND2DMaker



class SequentialDiffusionDecoder(nn.Module):
    def __init__(self, config: dict):
        super(SequentialDiffusionDecoder, self).__init__()
        self.in_channels = config["in_dim"]

        self.net = DiffusionCore(
            in_dim=config["in_dim"],
            patch_size=config["patch_size"],
            dim=config["dim"],
            out_dim=config["out_dim"],
            cond_dim=config["cond_dim"],
            depth=config["depth"],
            rope_base_len=config.get("rope_base_len", 10000),
            grad_checkpointing=config.get("grad_checkpointing", True),
            mlp_head=config.get("mlp_head", False),
            share_time_encoding=config.get("share_time_encoding", False),
            repa_layer=config.get("repa_layer", 8)
        )
        self.std = config.get("data_std", 1.0)
        self.time_shift = config.get("time_shift", 1.0)

        self.cfg = config.get("cfg", 1.0)

    def adjust_schedule(self, t):
        return t/(t+self.time_shift*(1-t))

    def forward(self, target, z, time_steps=None):
        if time_steps is None:
            time_steps = torch.rand(z.size(0)).to(z)

        time_steps = self.adjust_schedule(time_steps)

        time_ext = time_steps[:, None, None, None]

        x0 = torch.randn_like(target)


        x1 = (target*self.std).detach()

        xt = time_ext*x1 + (1.0-time_ext)*x0
        x_target = x1 - x0

        pred, h_repa = self.net(xt, time_steps, z)
        fm_loss = F.mse_loss(pred, x_target) / (self.std ** 2)

        predict_x1 = xt + pred * (1.0-time_ext)

        fake = predict_x1/self.std
        real = x1/self.std

        unscaled_xt = xt/self.std


        return fm_loss, [fake, unscaled_xt, time_steps, h_repa, fake, real]

    def sample(self, zs, timesteps, method="adams2", clip_steps=None, height=None, width=None, start_steps=None, start_x=None):
        z = zs[0]
        assert method in ["euler", "adams2"]
        with torch.no_grad():
            batch = z.size(0)
            if height is None:
                height, width = 256, 256

            x0 = torch.randn(batch, self.in_channels, height, width).to(z)
            xt = x0

            if clip_steps is not None:
                timesteps = timesteps[:clip_steps]
            schedule = torch.Tensor(timesteps + [1.0]).to(xt)
            schedule = self.adjust_schedule(schedule)

            if start_steps is not None and start_x is not None:
                t = schedule[start_steps]
                schedule = schedule[start_steps:]
                zs = zs[start_steps:]
                xt = x0 * (1.0-t) + start_x*self.std * t


            step_size = schedule[1:] - schedule[:-1]
            if method == "euler":
                for i, cur_t in enumerate(schedule[:-1]):
                    if clip_steps is not None and i >= clip_steps:
                        break
                    cur_t = cur_t.repeat(xt.size(0))
                    v, _ = self.net.forward(xt, cur_t, zs[i])
                    xt = xt + step_size[i] * v
            
            if method == "adams2":
                v_prev = None
                for i, cur_t in enumerate(schedule[:-1]):
                    if clip_steps is not None and i >= clip_steps:
                        break
                    cur_t = cur_t.repeat(xt.size(0))

                    if v_prev is None:
                        v, _ = self.net.forward(xt, cur_t, zs[i])
                        xt = xt + step_size[i] * v
                    else:
                        v, _ = self.net.forward(xt, cur_t, zs[i])
                        xt = xt + step_size[i] * (1.5*v - 0.5*v_prev)
                    v_prev = v

        xt = xt*(1.0/self.std)
        return xt


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=None):
        super().__init__()
        if frequency_embedding_size is None:
            frequency_embedding_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, mul=1000):
        t_freq = self.timestep_embedding(t*mul, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class SiTBlock(nn.Module):
    def __init__(self, embed_dim, head_dim=64, attn_drop=0.0, proj_drop=0.0, mlp_mult=4, share_time_encoding=False):
        super().__init__()
        self.attention = Attention(embed_dim, embed_dim // head_dim,
                                   attn_drop=attn_drop, proj_drop=proj_drop)
        self.cross_attn = Attention(embed_dim, embed_dim // head_dim,
                                   attn_drop=attn_drop, proj_drop=proj_drop)

        self.mlp = Mlp(embed_dim, embed_dim * int(mlp_mult), act_layer=nn.SiLU, drop=proj_drop)
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6*embed_dim, bias=True),
        )
        
        if not share_time_encoding:
            self.time_embed = TimestepEmbedder(embed_dim)
        else:
            self.time_embed = nn.Identity()

        self.cond_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.init_parameter()

    def init_parameter(self):
        def init_linear(m):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_linear(m)

            if isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c, t, rope2d, grad_checkpointing=False):
        t = self.time_embed(t)
        t = rearrange(t, 'b c -> b 1 c')

        c = self.norm(self.cond_embed(c))
        sc, _ = self.cross_attn(q=self.norm(x), k=c, v=c, q_pe=rope2d)



        def _inner_attention_forward(x, pe):
            return self.attention(x, pe=pe)
        
        def _iner_mlp_forward(x):
            return self.mlp(x)

        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t+sc).chunk(6, dim=-1)

        
        attn_in = modulate(self.norm(x), shift_attn, scale_attn)
        if grad_checkpointing and self.training:
            attn, attn_map = checkpoint(_inner_attention_forward, attn_in, rope2d)
        else:
            attn, attn_map = _inner_attention_forward(attn_in, rope2d)
        x = x + gate_attn * attn
        
        mlp_in = modulate(self.norm(x), shift_mlp, scale_mlp)
        if grad_checkpointing and self.training:
            mlp = checkpoint(_iner_mlp_forward, mlp_in)
        else:
            mlp = self.mlp(mlp_in)
        x = x + gate_mlp * mlp
        return x


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            model_channels, elementwise_affine=False, eps=1e-6)
        self.linear =  nn.Linear(model_channels, out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True),
        )

    def initialize_weights(self):
        if isinstance(self.linear, nn.Linear):
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
        else:
            nn.init.zeros_(self.linear[-1].weight)
            nn.init.zeros_(self.linear[-1].bias)

        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, t):
        t = rearrange(t, 'b c -> b 1 c')

        shift, scale = self.adaLN_modulation(t).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x



class DiffusionCore(nn.Module):
    def __init__(
        self,
        in_dim,
        patch_size,
        dim,
        out_dim,
        cond_dim,
        depth,
        grad_checkpointing=True,
        rope_base_len=10000,
        mlp_head=False,
        share_time_encoding=False,
        repa_layer=8
    ):
        super().__init__()

        self.in_dim = in_dim
        self.dim = dim
        self.out_dim = out_dim
        self.depth = depth
        self.grad_checkpointing = grad_checkpointing

        self.share_time_encoding = share_time_encoding

        if share_time_encoding:
            self.time_embed = TimestepEmbedder(dim)
        else:
            self.time_embed = nn.Identity()
        
        self.cond_mlp = nn.Linear(cond_dim, dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)


        self.repa_layer = repa_layer
        self.patch_size = patch_size
        self.patcher = Patcher(patch_size, patch_method="rearrange")
        self.unpatcher = UnPatcher(patch_size, patch_method="rearrange")

        self.conv_in = nn.Conv2d(3 * patch_size**2, dim, 1)
        self.mlp_head = mlp_head
        if self.mlp_head:
            self.linear_out = FinalLayer(dim, 3 * patch_size**2)
        else:
            self.conv_out = nn.Conv2d(dim, 3 * patch_size**2, 1)

        self.blocks = nn.ModuleList()
        for i in range(self.depth):
            self.blocks.append(
                SiTBlock(dim, share_time_encoding=share_time_encoding)
            )

       

        self.repa_out = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.SiLU(),
            nn.Linear(2048, 2048),
            nn.SiLU(),
            nn.Linear(2048, 768),
        )

        self.rope2d = EmbedND2DMaker(dim, rope_base_len, [32, 32])


        self.initialize_weights()

    def initialize_weights(self):
        if hasattr(self, "conv_out"):
            nn.init.zeros_(self.conv_out.weight)
            nn.init.zeros_(self.conv_out.bias)
        if hasattr(self, "linear_out"):
            if hasattr(self.linear_out, "initialize_weights"):
                self.linear_out.initialize_weights()

    def forward(self, x, t, cond):
        cond = self.cond_mlp(cond)
        cond = self.norm(cond)

        
        if self.share_time_encoding:
            t = self.time_embed(t)

        p = self.patcher(x)

        height, width = p.size(2), p.size(3)
        rope2d = self.rope2d(height, width, x.device)
        p = self.conv_in(p)
        p = rearrange(p, 'b c h w -> b (h w) c')
        h_repa = None
        for i, block in enumerate(self.blocks):

            # repa logistic
            # when larger than 16x16, avg pool it
            if i == self.repa_layer:
                repa = rearrange(p, 'b (h w) c -> b c h w', h=height, w=width)
                stride = height//16
                repa = F.avg_pool2d(repa, stride, stride)
                
                repa = rearrange(repa, 'b c h w -> b (h w) c')
                h_repa = self.repa_out(repa)

            p = block(p, cond, t, rope2d, grad_checkpointing=self.grad_checkpointing)
        
        p = self.norm(p)
        if self.mlp_head:
            p = self.linear_out(p, t)
        p = rearrange(p, 'b (h w) c -> b c h w', h=height, w=width)
        if not self.mlp_head:
            p = self.conv_out(p) 
        x = self.unpatcher(p)
        
        return x, h_repa