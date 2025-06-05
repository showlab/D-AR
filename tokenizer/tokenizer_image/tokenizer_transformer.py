import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from timm.layers.mlp import Mlp
from timm.layers.drop import DropPath

from einops import rearrange


QKV_BIAS = True
QK_NORM = True


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = False

        qkv_bias = QKV_BIAS
        qk_norm = QK_NORM

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: Tensor = None,
        q: Tensor = None,
        k: Tensor = None,
        v: Tensor = None,
        attn_mask: Tensor = None,
        return_attn_map: bool = False,
        causal: bool = False,
        pe: Tensor = None,
        q_pe: Tensor = None,
    ) -> tuple[Tensor, Tensor]:

        if x is not None:
            q = k = v = x

        B, N, C = q.size()

        q = self.q(q).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(v).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q, k = self.q_norm(q).type_as(q), self.k_norm(k).type_as(k)

        if pe is not None:
            q, k = apply_rope(q, k, pe)

        if q_pe is not None:
            q = apply_rope_single(q, q_pe)

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False):
            x = nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=causal,
            )

        if return_attn_map:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if causal or attn_mask is not None:
                mask = (
                    torch.tril(torch.ones_like(attn)) if causal else attn_mask
                )
                mask = mask.logical_not().float()
                attn += torch.where(mask == 1.0, float("-inf"), mask)

            attn = attn.softmax(-1)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return (x, attn) if return_attn_map else (x, None)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        drop_path_prob: float = 0.0,
        head_dim: int = 64,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_mult: int = 4,
    ):
        super().__init__()

        self.attention = Attention(
            embed_dim, embed_dim // head_dim,
            attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.mlp = Mlp(embed_dim, int(embed_dim * mlp_mult), act_layer=nn.SiLU, drop=proj_drop)

        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        self.gate1 = nn.Parameter(torch.zeros(embed_dim))
        self.gate2 = nn.Parameter(torch.zeros(embed_dim))

        self.droppath = DropPath(drop_prob=drop_path_prob)
        self.checkpoint = True

        self.init_parameter()

    def init_parameter(self):
        def init_linear(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(init_linear)

    def forward(self, x, attn_mask=None, pe=None, causal=False):
        def _attn_fn(x_, mask_, pe_, causal_):
            return self.attention(x_, attn_mask=mask_, pe=pe_, causal=causal_)[0]

        def _mlp_fn(x_):
            return self.mlp(x_)

        if self.checkpoint and self.training:
            attn = checkpoint(_attn_fn, self.norm1(x), attn_mask, pe, causal)
            x = x + self.droppath(attn) * self.gate1.reshape(1, 1, -1)

            mlp = checkpoint(_mlp_fn, self.norm2(x))
            x = x + self.droppath(mlp) * self.gate2.reshape(1, 1, -1)
        else:
            attn, attn_map = self.attention(
                self.norm1(x), attn_mask=attn_mask, pe=pe,
                return_attn_map=True, causal=causal
            )
            x = x + self.droppath(attn) * self.gate1.reshape(1, 1, -1)
            mlp = self.mlp(self.norm2(x))
            x = x + self.droppath(mlp) * self.gate2.reshape(1, 1, -1)

        return x

# ROPE from Flux
# https://github.com/black-forest-labs/flux/blob/main/src/flux/math.py

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return (
        xq_out.reshape(*xq.shape).type_as(xq),
        xk_out.reshape(*xk.shape).type_as(xk),
    )


def apply_rope_single(xq: Tensor, freqs_cis: Tensor) -> Tensor:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq)


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.size(-1)
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


class EmbedND2DMaker(EmbedND):
    def forward(self, height: int, width: int, device=None) -> Tensor:
        ids = torch.stack(
            torch.meshgrid(
                torch.arange(height, device=device, dtype=torch.float),
                torch.arange(width, device=device, dtype=torch.float),
            ),
            dim=-1,
        )
        ids = ids.flatten(0, 1).unsqueeze(0)

        n_axes = ids.size(-1)
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)
