# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from tokenizer.tokenizer_image.tokenizer_transformer import TransformerBlock, EmbedND2DMaker
from tokenizer.tokenizer_image.diff_decoder import SequentialDiffusionDecoder

class VQModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        encoder_patch_size = config["encoder_patch_size"]

        encoder_dim = config["encoder_dim"]
        decoder_dim = config["decoder_dim"]

        self.encoder_layers = nn.ModuleList([TransformerBlock(encoder_dim) for _ in range(config["num_encoder_layers"])])
        
        num_decoder_layers = config["num_decoder_layers"]
        self.num_decoder_layers = num_decoder_layers

        if self.num_decoder_layers > 0:
            self.decoder_layers = nn.ModuleList([TransformerBlock(decoder_dim) for _ in range(num_decoder_layers)])

        self.encoder_norm = nn.LayerNorm(encoder_dim, eps=1e-6, elementwise_affine=False, bias=False)

        self.norm = nn.LayerNorm(decoder_dim, eps=1e-6, elementwise_affine=False, bias=False)

        self.patchify = nn.Conv2d(3, encoder_dim, kernel_size=encoder_patch_size, stride=encoder_patch_size)

        vq_dim = config["vq_dim"]
        self.quant_in_conv = nn.Conv2d(encoder_dim, vq_dim, kernel_size=1, stride=1, bias=False)
        self.quant_out_conv = nn.Conv2d(vq_dim, decoder_dim, kernel_size=1, stride=1, bias=False)

        if config.get("vq_enable", True):
            self.vq_codebook_size = config["vq_codebook_size"]
            self.quantize = VectorQuantizer(
                config["vq_codebook_size"], vq_dim, 
                0.25, 0.0,
                True, True)
        else:
            self.quantize = LNBottleneck(vq_dim)

        self.denoising_steps = config["num_denoising_steps"]
        self.all_queries = config["num_all_queries"]
        self.num_groups = config["num_query_groups"]
        default_queries_per_step = self.all_queries//self.num_groups
        self.queries_per_step = config.get("queries_per_step", default_queries_per_step)


        self.sampling_resolution = config.get("sampling_resolution", 256)
        self.output_resolution = config.get("output_resolution", 0)

        query_low_rank = config.get("query_low_rank", True)
        if query_low_rank:
            self.query_embedding = nn.Embedding(1024, 8)
            self.query_linear = nn.Linear(8, encoder_dim)
            nn.init.trunc_normal_(self.query_embedding.weight, std=1.0)
        else:
            self.query_embedding = nn.Embedding(1024, encoder_dim)
            self.query_linear = nn.Identity()
            nn.init.trunc_normal_(self.query_embedding.weight, std=0.02)

        self.decoder_embedding = nn.Embedding(1024, decoder_dim)

        nn.init.trunc_normal_(self.decoder_embedding.weight, std=0.02)
        self.rope2d = EmbedND2DMaker(encoder_dim, config.get("rope_base_len", 10000), [32, 32])

        self.diff_decoder = SequentialDiffusionDecoder(config["diff_decoder"])

    def encode(self, x, just_code=False):

        all_queries = self.all_queries
        len_queries_per_step = self.queries_per_step


        time_steps = torch.rand(x.shape[0]).to(x)

        num_groups = self.num_groups

        discrete_time_steps = (time_steps * num_groups+0.5).floor().div(num_groups)
        start_query_ids = (discrete_time_steps * all_queries).long()
        start_query_ids = start_query_ids.clamp(min=0, max=all_queries-len_queries_per_step)


        start_query_ids = start_query_ids.unsqueeze(1)
        query_ids = start_query_ids + torch.arange(0, len_queries_per_step).to(start_query_ids).unsqueeze(0)


        patches = self.patchify(x)
        batch, height, width = patches.shape[0], patches.shape[2], patches.shape[3]
        patches = rearrange(patches, 'b c h w -> b (h w) c')

        rope2d = self.rope2d(height, width).to(patches.device)
        eye = torch.eye(2).to(rope2d)
        pad2d = eye[None, None, None, None].repeat(1, 1, self.all_queries, rope2d.size(3), 1, 1)

        rope2d = torch.cat([rope2d, pad2d], dim=2)

        queries = self.query_embedding.weight[:self.all_queries]
        queries = repeat(queries, 'n d -> b n d', b=batch)

        queries = self.query_linear(queries)
        patches = self.encoder_norm(patches) 
        tokens = torch.cat([patches, queries], dim=1)

        attn_mask = torch.tril(
            torch.ones(1, 1, tokens.size(1), tokens.size(1), device=tokens.device, dtype=torch.bool)
        )
        attn_mask[:, :, :, : patches.size(1)] = 1

        for layer in self.encoder_layers:
            tokens = layer(tokens, attn_mask=attn_mask, pe=rope2d)

        tokens = tokens[:, -self.all_queries:]
        tokens = self.encoder_norm(tokens)
        tokens = rearrange(tokens, 'b n c -> b c 1 n', n=self.all_queries)

        tokens = self.quant_in_conv(tokens)
        quant, emb_loss, info = self.quantize(tokens, query_ids=query_ids)

        if just_code:
            return info

        cond = self.decoder_layers_forward(quant)

        if self.training:
            cond = cond[torch.arange(cond.size(0)).view(-1, 1), query_ids]

            diff_target = x
            diff_loss, aux_tuple = self.diff_decoder.forward(diff_target, cond, time_steps=time_steps)
        else:
            diff_loss = 0.0
            aux_tuple = None


        emb_loss = list(emb_loss) + [diff_loss,]
        return quant, emb_loss, info, cond, aux_tuple

    def decode(self, cond, steps=50, cfg=1.0):
        if not self.training:
            dec = self.diff_decoder.sample(cond)
        else:
            dec = None
        return dec

    def extract_code(self, x):
        info = self.encode(x, just_code=True)
        code = info[-1] if info is not None else None
        code = code.reshape(x.size(0), -1)
        return code

    def decoder_layers_forward(self, quant):
        tokens = self.quant_out_conv(quant)
        tokens = rearrange(tokens, 'b c 1 n -> b n c')

        decoder_query = self.decoder_embedding.weight[:self.all_queries]
        decoder_query = repeat(decoder_query, 'n d -> b n d', b=tokens.size(0))
        tokens = tokens + decoder_query

        t_batch, t_len, t_dim = tokens.size()
        if self.num_decoder_layers > 0:
            for layer in self.decoder_layers:
                tokens = layer(tokens, causal=True)

        cond = self.norm(tokens)
        return cond

    def decode_code(self,
                    code_b,
                    height=256, width=256,
                    channel_first=True,
                    clip_steps=None,
                    start_steps=None,
                    start_x=None):
        batch = code_b.size(0) if code_b.ndim > 1 else 1
        length = code_b.size(-1)
        quant_b = self.quantize.get_codebook_entry(code_b, [batch, -1, 1, length], channel_first)
        cond = self.decoder_layers_forward(quant_b)

        height = max(self.sampling_resolution, height)
        width = max(self.sampling_resolution, width)
        
        dec = self.decode_into_pixels(cond, height, width, clip_steps=clip_steps, start_steps=start_steps, start_x=start_x)

        output_resolution = self.output_resolution
        if output_resolution != 0 and (height > output_resolution or width > output_resolution):
            dec = F.interpolate(dec, size=(output_resolution, output_resolution), mode='bicubic', align_corners=False)
        return dec


    def straight_forward(self,
                         x,
                         clip_steps=None):
        quant, diff, misc, cond, recon = self.encode(x)

        height = max(self.sampling_resolution, x.size(2))
        width = max(self.sampling_resolution, x.size(3))

        dec = self.decode_into_pixels(cond, height, width, clip_steps=clip_steps)
        output_resolution = self.output_resolution
        if output_resolution != 0 and (height > output_resolution or width > output_resolution):
            dec = F.interpolate(dec, size=(output_resolution, output_resolution), mode='bicubic', align_corners=False)
        return dec


    def decode_into_pixels(self,
                           cond,
                           height, width,
                           clip_steps=None,
                           start_steps=None,
                           start_x=None):
        all_queries = self.all_queries
        denosing_steps = self.denoising_steps
        timesteps = [i/denosing_steps for i in range(denosing_steps)]
        timesteps = timesteps[:clip_steps] if clip_steps is not None else timesteps
        timesteps = timesteps[start_steps:] if start_steps is not None else timesteps

        conds = []
        for t in timesteps:
            discrete_t = math.floor(t * self.num_groups+0.5)/self.num_groups
            idx = int(discrete_t * all_queries)
            idx = min(self.all_queries- self.queries_per_step, idx)
            begin, end = idx, idx + self.queries_per_step
            conds.append(cond[:, begin:end])

        dec = self.diff_decoder.sample(conds, timesteps, clip_steps=clip_steps, height=height, width=width, start_steps=start_steps, start_x=start_x)
        return dec

    def forward(self, input, **kwargs):
        if not self.training:
            return self.straight_forward(input, **kwargs)
        quant, diff, _, cond, recon = self.encode(input)
        dec = self.decode(cond, **kwargs)
        return recon, diff


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        else:
            self.embedding.weight.data.normal_(0.0, 2.0)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))
    
    def forward(self, z, query_ids=None):
        batch = z.size(0)
        # reshape z -> (batch, height, width, channel) and flatten
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)

        else:
            embedding = self.embedding.weight
            

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = embedding[min_encoding_indices].view(z.shape)
        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.training:
            code = min_encoding_indices.reshape(batch, -1)
            dead_code_rate = (code.float().std(0) < 1e-5).float().mean()
        else:
            dead_code_rate = 0.0

        if self.show_usage and self.training:
            cur_len = 512
            # cur_len = min_encoding_indices.shape[0]
            indices = torch.randperm(min_encoding_indices.shape[0])[:cur_len]
            min_encoding_indices_sampled = min_encoding_indices[indices]
            
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices_sampled
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        # compute loss for embedding
        if self.training:
            # mask VQ-related loss for unaffected tokens

            # vq_loss = torch.mean((z_q - z.detach()) ** 2) 
            # commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 
            # entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)

            vq_loss = (z_q - z.detach()) ** 2
            vq_loss = vq_loss.mean(dim=-1, keepdim=True)
            commit_loss = self.beta * (z_q.detach() - z) ** 2
            commit_loss = commit_loss.mean(dim=-1, keepdim=True)

            loss_mask = torch.arange(vq_loss.size(2), device=vq_loss.device)
            loss_mask = loss_mask.view(1, 1, -1, 1)
            loss_mask = (loss_mask <= query_ids[:, -1].view(-1, 1, 1, 1)).float()

            vq_loss = (vq_loss * loss_mask).sum() / loss_mask.sum()
            commit_loss = (commit_loss * loss_mask).sum() / loss_mask.sum()
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = torch.einsum('b h w c -> b c h w', z_q)


        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage, dead_code_rate), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q


class LNBottleneck(nn.Module):
    def __init__(self, dim, ):
        super().__init__()


    def forward(self, x: torch.Tensor, **kwargs):
        x = F.normalize(x, p=2, dim=1)

        loss_kld = x.mean()
        zeros = torch.zeros_like(loss_kld)
        return x, (zeros, zeros, zeros, zeros), None 


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss

