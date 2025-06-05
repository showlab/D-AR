# Modified from:
#   stylegan2-pytorch: https://github.com/lucidrains/stylegan2-pytorch/blob/master/stylegan2_pytorch/stylegan2_pytorch.py
#   stylegan2-pytorch: https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py
#   maskgit: https://github.com/google-research/maskgit/blob/main/maskgit/nets/discriminator.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# try:
#     from kornia.filters import filter2d
# except:
#     pass


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
            -math.log(max_period) * torch.arange(start=0,
                                                 end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, mul=1000):
        t_freq = self.timestep_embedding(t*mul, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TimeAdaGN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.bn = nn.GroupNorm(32, num_features, affine=False)
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


class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)),
                          int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if (self.filt_size == 1):
            a = np.array([1.,])
        elif (self.filt_size == 2):
            a = np.array([1., 1.])
        elif (self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif (self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif (self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif (self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif (self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None]*a[None, :])
        filt = filt/torch.sum(filt)
        self.register_buffer(
            'filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if (self.filt_size == 1):
            if (self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if (pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif (pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif (pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class BlurPool1D(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(BlurPool1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2),
                          int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # print('Filter size [%i]' % filt_size)
        if (self.filt_size == 1):
            a = np.array([1., ])
        elif (self.filt_size == 2):
            a = np.array([1., 1.])
        elif (self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif (self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif (self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif (self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif (self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer(
            'filt', filt[None, None, :].repeat((self.channels, 1, 1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if (self.filt_size == 1):
            if (self.pad_off == 0):
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer_1d(pad_type):
    if (pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad1d
    elif (pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad1d
    elif (pad_type == 'zero'):
        PadLayer = nn.ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, channel_multiplier=1, image_size=256):
        super().__init__()
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        log_size = int(math.log(image_size, 2))
        in_channel = channels[image_size]

        blocks = [nn.Conv2d(input_nc, in_channel, 3, padding=1), leaky_relu()]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            blocks.append(DiscriminatorBlock(in_channel, out_channel))
            in_channel = out_channel
        self.blocks = nn.ModuleList(blocks)

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channel, channels[4], 3, padding=1),
            leaky_relu(),
        )
        self.final_linear = nn.Sequential(
            nn.Linear(channels[4] * 4 * 4, channels[4]),
            leaky_relu(),
            nn.Linear(channels[4], 1)
        )

    def forward(self, x, t):
        for block in self.blocks:
            if isinstance(block, DiscriminatorBlock):
                x = block(x, t)
            else:
                x = block(x)
        x = self.final_conv(x)
        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)
        return x


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(
            input_channels, filters, 1, stride=(2 if downsample else 1))

        self.norm = TimeAdaGN(input_channels)

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Sequential(
            # Blur(),
            BlurPool(filters, filt_size=4, stride=2, pad_off=0),
            nn.Conv2d(filters, filters, 1, padding=0, stride=1)
        ) if downsample else None

    def forward(self, x, t):
        res = self.conv_res(x)
        x = self.norm(x, t)
        x = self.net(x)
        if exists(self.downsample):
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)


def exists(val):
    return val is not None
