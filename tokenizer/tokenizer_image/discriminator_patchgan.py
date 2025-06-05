# Modified from:
#   taming-transformers:  https://github.com/CompVis/taming-transformers
import functools
import torch
import torch.nn as nn

import math


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
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
    

class BatchNormWithTimeEmbedding(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.bn = nn.GroupNorm(16, num_features, affine=False)
        # self.bn = nn.SyncBatchNorm(num_features, affine=False)
        self.embedder = TimestepEmbedder(num_features*2)
        # nn.init.zeros_(self.embedder.mlp[-1].weight)
        nn.init.trunc_normal_(self.embedder.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.embedder.mlp[-1].bias)

    def forward(self, x, t):
        embed = self.embedder(t)
        embed = embed[:, :, None, None]
        gamma, beta = embed.chunk(2, dim=1)
        gamma = 1.0 + gamma
        normed = self.bn(x)
        out = normed * gamma + beta
        return out


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        # if not use_actnorm:
        #     # norm_layer = nn.BatchNorm2d
        #     norm_layer = nn.SyncBatchNorm
        # else:
        #     norm_layer = ActNorm
        # if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
        #     use_bias = norm_layer.func != nn.BatchNorm2d
        # else:
        #     use_bias = norm_layer != nn.BatchNorm2d
        #     use_bias = False
        norm_layer = BatchNormWithTimeEmbedding
        use_bias=False

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)
        # import copy
        # self.dual = copy.deepcopy(self.main)

        # self.fc2 = nn.Sequential(
        #     nn.Conv2d(2* ndf * nf_mult, ndf * nf_mult, kernel_size=1, stride=1, padding=0),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv2d(ndf * nf_mult, 1, kernel_size=1, stride=1, padding=0)

        # )

        self.apply(self._init_weights)
    
    def _init_weights(self, module):    
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            # nn.init.normal_(module.weight.data, 1.0, 0.02)
            # nn.init.constant_(module.bias.data, 0)
            pass
        
        # nn.init.zeros_(self.main[-1].weight)
        # nn.init.constant_(self.main[-1].bias, 0.0)
        # nn.init.zeros_(self.fc2[-1].weight)
        # nn.init.constant_(self.fc2[-1].bias, 0.0)
        
    def forward(self, input, t):
        """Standard forward."""
        # x1, x2 = input.chunk(2, dim=1)
        # return self.main(input)
        x1 = input
        for layer in self.main:
            if isinstance(layer, BatchNormWithTimeEmbedding):
                x1 = layer(x1, t)
            else:
                x1 = layer(x1)
        return x1

class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h