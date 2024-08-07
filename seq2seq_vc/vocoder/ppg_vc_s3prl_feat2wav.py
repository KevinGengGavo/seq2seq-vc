import logging
import numpy as np
import pyworld
from scipy.interpolate import interp1d
from scipy.signal import firwin, get_window, lfilter

import yaml

import torch
from abc import ABC
from abc import abstractmethod
from typing import Tuple

import torch
from typeguard import typechecked

import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import torch
import torch.nn as nn
import yaml
import numpy as np
import pyworld
from scipy.interpolate import interp1d
from scipy.signal import firwin, get_window, lfilter
from typing import Dict, Tuple, List
from abc import ABC, abstractmethod
from seq2seq_vc.utils import read_hdf5
from seq2seq_vc.vocoder import Vocoder
from seq2seq_vc.vocoder.griffin_lim import Spectrogram2Waveform
from s3prl.nn import Featurizer
import s3prl_vc.models
from s3prl_vc.upstream.interface import get_upstream

from seq2seq_vc.vocoder.audio import preprocess_wav
from seq2seq_vc.vocoder.voice_encoder import SpeakerEncoder
from pathlib import Path

import os
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from seq2seq_vc.vocoder.hifigan import load_hifigan_generator


LRELU_SLOPE = 0.1

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

DEFAULT_CONFIG="/home/kevingenghaopeng/vc/seq2seq-vc/seq2seq_vc/vocoder/vctk_24k10ms/config.json"
DEFAULT_CKPT="/home/kevingenghaopeng/vc/seq2seq-vc/seq2seq_vc/vocoder/vctk_24k10ms/g_02830000"

class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class InterpolationBlock(torch.nn.Module):
    def __init__(self, scale_factor, mode='nearest', align_corners=None, downsample=False):
        super(InterpolationBlock, self).__init__()
        self.downsample = downsample
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    
    def forward(self, x):
        outputs = torch.nn.functional.interpolate(
            x,
            size=x.shape[-1] * self.scale_factor \
                if not self.downsample else x.shape[-1] // self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=False
        )
        return outputs


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        import pdb; pdb.set_trace()
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2
        self.sampling_rate = h.sampling_rate
        self.ups = nn.ModuleList()
        if h.sampling_rate == 24000:
            for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
                self.ups.append(
                    torch.nn.Sequential(
                        InterpolationBlock(u),
                        weight_norm(torch.nn.Conv1d(
                            h.upsample_initial_channel//(2**i),
                            h.upsample_initial_channel//(2**(i+1)),
                            k, padding=(k-1)//2,
                        ))
                    )
                )
        else:
            for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
                self.ups.append(weight_norm(
                    ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                    k, u, padding=(k-u)//2)))
        # for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            # self.ups.append(weight_norm(
                # ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                # k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            if self.sampling_rate == 24000:
                remove_weight_norm(l[-1])
            else:
                remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses



class AbsMelDecoder(torch.nn.Module, ABC):
    """The abstract PPG-based voice conversion class
    This "model" is one of mediator objects for "Task" class.

    """

    @abstractmethod
    def forward(
        self, 
        bottle_neck_features: torch.Tensor,
        feature_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        logf0_uv: torch.Tensor = None,
        spembs: torch.Tensor = None,
        styleembs: torch.Tensor = None,
    ) -> torch.Tensor:
        raise NotImplementedError

class MOLAttention(nn.Module):
    """ Discretized Mixture of Logistic (MOL) attention.
    C.f. Section 5 of "MelNet: A Generative Model for Audio in the Frequency Domain" and 
        GMMv2b model in "Location-relative attention mechanisms for robust long-form speech synthesis".
    """
    def __init__(
        self,
        query_dim,
        r=1,
        M=5,
    ):
        """
        Args:
            query_dim: attention_rnn_dim.
            M: number of mixtures.
        """
        super().__init__()
        if r < 1:
            self.r = float(r)
        else:
            self.r = int(r)
        self.M = M
        self.score_mask_value = 0.0 # -float("inf")
        self.eps = 1e-5
        # Position arrary for encoder time steps
        self.J = None
        # Query layer: [w, sigma,]
        self.query_layer = torch.nn.Sequential(
            nn.Linear(query_dim, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 3*M, bias=True)
        )
        self.mu_prev = None
        self.initialize_bias()

    def initialize_bias(self):
        """Initialize sigma and Delta."""
        # sigma
        torch.nn.init.constant_(self.query_layer[2].bias[self.M:2*self.M], 1.0)
        # Delta: softplus(1.8545) = 2.0; softplus(3.9815) = 4.0; softplus(0.5413) = 1.0
        # softplus(-0.432) = 0.5003
        if self.r == 2:
            torch.nn.init.constant_(self.query_layer[2].bias[2*self.M:3*self.M], 1.8545)
        elif self.r == 4:
            torch.nn.init.constant_(self.query_layer[2].bias[2*self.M:3*self.M], 3.9815)
        elif self.r == 1:
            torch.nn.init.constant_(self.query_layer[2].bias[2*self.M:3*self.M], 0.5413)
        else:
            torch.nn.init.constant_(self.query_layer[2].bias[2*self.M:3*self.M], -0.432)

    
    def init_states(self, memory):
        """Initialize mu_prev and J.
            This function should be called by the decoder before decoding one batch.
        Args:
            memory: (B, T, D_enc) encoder output.
        """
        B, T_enc, _ = memory.size()
        device = memory.device
        self.J = torch.arange(0, T_enc + 2.0).to(device) + 0.5  # NOTE: for discretize usage
        # self.J = memory.new_tensor(np.arange(T_enc), dtype=torch.float)
        self.mu_prev = torch.zeros(B, self.M).to(device)

    def forward(self, att_rnn_h, memory, memory_pitch=None, mask=None):
        """
        att_rnn_h: attetion rnn hidden state.
        memory: encoder outputs (B, T_enc, D).
        mask: binary mask for padded data (B, T_enc).
        """
        # [B, 3M]
        mixture_params = self.query_layer(att_rnn_h)
        
        # [B, M]
        w_hat = mixture_params[:, :self.M]
        sigma_hat = mixture_params[:, self.M:2*self.M]
        Delta_hat = mixture_params[:, 2*self.M:3*self.M]
        
        # print("w_hat: ", w_hat)

# from .basic_layers import Linear, Conv1d
        # print("sigma_hat: ", sigma_hat)
        # print("Delta_hat: ", Delta_hat)

        # Dropout to de-correlate attention heads
        w_hat = F.dropout(w_hat, p=0.5, training=self.training) # NOTE(sx): needed?
        
        # Mixture parameters
        w = torch.softmax(w_hat, dim=-1) + self.eps
        sigma = F.softplus(sigma_hat) + self.eps
        Delta = F.softplus(Delta_hat)
        mu_cur = self.mu_prev + Delta
        # print("w:", w)
        j = self.J[:memory.size(1) + 1]

        # Attention weights
        # CDF of logistic distribution
        phi_t = w.unsqueeze(-1) * (1 / (1 + torch.sigmoid(
            (mu_cur.unsqueeze(-1) - j) / sigma.unsqueeze(-1))))
        # print("phi_t:", phi_t)
        
        # Discretize attention weights
        # (B, T_enc + 1)
        alpha_t = torch.sum(phi_t, dim=1)
        alpha_t = alpha_t[:, 1:] - alpha_t[:, :-1]
        alpha_t[alpha_t == 0] = self.eps
        # print("alpha_t: ", alpha_t.size())
        # Apply masking
        if mask is not None:
            alpha_t.data.masked_fill_(mask, self.score_mask_value)

        context = torch.bmm(alpha_t.unsqueeze(1), memory).squeeze(1)
        if memory_pitch is not None:
            context_pitch = torch.bmm(alpha_t.unsqueeze(1), memory_pitch).squeeze(1)

        self.mu_prev = mu_cur
        
        if memory_pitch is not None:
            return context, context_pitch, alpha_t
        return context, alpha_t

class Linear(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(Linear, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)
    
class Conv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear', param=None):
        super(Conv1d, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1)/2)
        
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain, param=param))

    def forward(self, x):
        # x: BxDxT
        return self.conv(x)

class DecoderPrenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super().__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [Linear(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x

def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

class Decoder(nn.Module):
    """Mixture of Logistic (MoL) attention-based RNN Decoder."""
    def __init__(
        self,
        enc_dim,
        num_mels,
        frames_per_step,
        attention_rnn_dim,
        decoder_rnn_dim,
        prenet_dims,
        num_mixtures,
        encoder_down_factor=1,
        num_decoder_rnn_layer=1,
        use_stop_tokens=False,
        concat_context_to_last=False,
    ):
        super().__init__()
        self.enc_dim = enc_dim
        self.encoder_down_factor = encoder_down_factor
        self.num_mels = num_mels
        self.frames_per_step = frames_per_step
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dims = prenet_dims
        self.use_stop_tokens = use_stop_tokens
        self.num_decoder_rnn_layer = num_decoder_rnn_layer
        self.concat_context_to_last = concat_context_to_last

        # Mel prenet
        self.prenet = DecoderPrenet(num_mels, prenet_dims)
        self.prenet_pitch = DecoderPrenet(num_mels, prenet_dims)

        # Attention RNN
        self.attention_rnn = nn.LSTMCell(
            prenet_dims[-1] + enc_dim,
            attention_rnn_dim
        )
        
        # Attention
        self.attention_layer = MOLAttention(
            attention_rnn_dim,
            r=frames_per_step/encoder_down_factor,
            M=num_mixtures,
        )

        # Decoder RNN
        self.decoder_rnn_layers = nn.ModuleList()
        for i in range(num_decoder_rnn_layer):
            if i == 0:
                self.decoder_rnn_layers.append(
                    nn.LSTMCell(
                        enc_dim + attention_rnn_dim,
                        decoder_rnn_dim))
            else:
                self.decoder_rnn_layers.append(
                    nn.LSTMCell(
                        decoder_rnn_dim,
                        decoder_rnn_dim))
        # self.decoder_rnn = nn.LSTMCell(
            # 2 * enc_dim + attention_rnn_dim,
            # decoder_rnn_dim
        # )
        if concat_context_to_last:
            self.linear_projection = Linear(
                enc_dim + decoder_rnn_dim,
                num_mels * frames_per_step
            )
        else:
            self.linear_projection = Linear(
                decoder_rnn_dim,
                num_mels * frames_per_step
            )


        # Stop-token layer
        if self.use_stop_tokens:
            if concat_context_to_last:
                self.stop_layer = Linear(
                    enc_dim + decoder_rnn_dim, 1, bias=True, w_init_gain="sigmoid"
                )
            else:
                self.stop_layer = Linear(
                    decoder_rnn_dim, 1, bias=True, w_init_gain="sigmoid"
                )
                

    def get_go_frame(self, memory):
        B = memory.size(0)
        go_frame = torch.zeros((B, self.num_mels), dtype=torch.float,
                               device=memory.device)
        return go_frame

    def initialize_decoder_states(self, memory, mask):
        device = next(self.parameters()).device
        B = memory.size(0)
        
        # attention rnn states
        self.attention_hidden = torch.zeros(
            (B, self.attention_rnn_dim), device=device)
        self.attention_cell = torch.zeros(
            (B, self.attention_rnn_dim), device=device)

        # decoder rnn states
        self.decoder_hiddens = []
        self.decoder_cells = []
        for i in range(self.num_decoder_rnn_layer):
            self.decoder_hiddens.append(
                torch.zeros((B, self.decoder_rnn_dim),
                            device=device)
            )
            self.decoder_cells.append(
                torch.zeros((B, self.decoder_rnn_dim),
                            device=device)
            )
        # self.decoder_hidden = torch.zeros(
            # (B, self.decoder_rnn_dim), device=device)
        # self.decoder_cell = torch.zeros(
            # (B, self.decoder_rnn_dim), device=device)
        
        self.attention_context =  torch.zeros(
            (B, self.enc_dim), device=device)

        self.memory = memory
        # self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """Prepare decoder inputs, i.e. gt mel
        Args:
            decoder_inputs:(B, T_out, n_mel_channels) inputs used for teacher-forced training.
        """
        decoder_inputs = decoder_inputs.reshape(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.frames_per_step), -1)
        # (B, T_out//r, r*num_mels) -> (T_out//r, B, r*num_mels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        # (T_out//r, B, num_mels)
        decoder_inputs = decoder_inputs[:,:,-self.num_mels:]
        return decoder_inputs
        
    def parse_decoder_outputs(self, mel_outputs, alignments, stop_outputs):
        """ Prepares decoder outputs for output
        Args:
            mel_outputs:
            alignments:
        """
        # (T_out//r, B, T_enc) -> (B, T_out//r, T_enc)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out//r, B) -> (B, T_out//r)
        if stop_outputs is not None:
            if alignments.size(0) == 1:
                stop_outputs = torch.stack(stop_outputs).unsqueeze(0)
            else:
                stop_outputs = torch.stack(stop_outputs).transpose(0, 1)
            stop_outputs = stop_outputs.contiguous()
        # (T_out//r, B, num_mels*r) -> (B, T_out//r, num_mels*r)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        # (B, T_out, num_mels)
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.num_mels)
        return mel_outputs, alignments, stop_outputs     
    
    def attend(self, decoder_input):
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_context, attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, None, self.mask)
        
        decoder_rnn_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)

        return decoder_rnn_input, self.attention_context, attention_weights

    def decode(self, decoder_input):
        for i in range(self.num_decoder_rnn_layer):
            if i == 0:
                self.decoder_hiddens[i], self.decoder_cells[i] = self.decoder_rnn_layers[i](
                    decoder_input, (self.decoder_hiddens[i], self.decoder_cells[i]))
            else:
                self.decoder_hiddens[i], self.decoder_cells[i] = self.decoder_rnn_layers[i](
                    self.decoder_hiddens[i-1], (self.decoder_hiddens[i], self.decoder_cells[i]))
        return self.decoder_hiddens[-1]
    
    def forward(self, memory, mel_inputs, memory_lengths):
        """ Decoder forward pass for training
        Args:
            memory: (B, T_enc, enc_dim) Encoder outputs
            decoder_inputs: (B, T, num_mels) Decoder inputs for teacher forcing.
            memory_lengths: (B, ) Encoder output lengths for attention masking.
        Returns:
            mel_outputs: (B, T, num_mels) mel outputs from the decoder
            alignments: (B, T//r, T_enc) attention weights.
        """
        # [1, B, num_mels]
        go_frame = self.get_go_frame(memory).unsqueeze(0)
        # [T//r, B, num_mels]
        mel_inputs = self.parse_decoder_inputs(mel_inputs)
        # [T//r + 1, B, num_mels]
        mel_inputs = torch.cat((go_frame, mel_inputs), dim=0)
        # [T//r + 1, B, prenet_dim]
        decoder_inputs = self.prenet(mel_inputs) 
        # decoder_inputs_pitch = self.prenet_pitch(decoder_inputs__)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths),
        )
        
        self.attention_layer.init_states(memory)
        # self.attention_layer_pitch.init_states(memory_pitch)

        mel_outputs, alignments = [], []
        if self.use_stop_tokens:
            stop_outputs = []
        else:
            stop_outputs = None
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            # decoder_input_pitch = decoder_inputs_pitch[len(mel_outputs)]

            decoder_rnn_input, context, attention_weights = self.attend(decoder_input)

            decoder_rnn_output = self.decode(decoder_rnn_input)
            if self.concat_context_to_last:    
                decoder_rnn_output = torch.cat(
                    (decoder_rnn_output, context), dim=1)
                   
            mel_output = self.linear_projection(decoder_rnn_output)
            if self.use_stop_tokens:
                stop_output = self.stop_layer(decoder_rnn_output)
                stop_outputs += [stop_output.squeeze()]
            mel_outputs += [mel_output.squeeze(1)] #? perhaps don't need squeeze
            alignments += [attention_weights]
            # alignments_pitch += [attention_weights_pitch]   

        mel_outputs, alignments, stop_outputs = self.parse_decoder_outputs(
            mel_outputs, alignments, stop_outputs)
        if stop_outputs is None:
            return mel_outputs, alignments
        else:
            return mel_outputs, stop_outputs, alignments

    def inference(self, memory, stop_threshold=0.5):
        """ Decoder inference
        Args:
            memory: (1, T_enc, D_enc) Encoder outputs
        Returns:
            mel_outputs: mel outputs from the decoder
            alignments: sequence of attention weights from the decoder
        """
        # [1, num_mels]
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        self.attention_layer.init_states(memory)
        
        mel_outputs, alignments = [], []
        # NOTE(sx): heuristic 
        max_decoder_step = memory.size(1)*self.encoder_down_factor//self.frames_per_step 
        min_decoder_step = memory.size(1)*self.encoder_down_factor // self.frames_per_step - 5
        while True:
            decoder_input = self.prenet(decoder_input)

            decoder_input_final, context, alignment = self.attend(decoder_input)

            #mel_output, stop_output, alignment = self.decode(decoder_input)
            decoder_rnn_output = self.decode(decoder_input_final)
            if self.concat_context_to_last:    
                decoder_rnn_output = torch.cat(
                    (decoder_rnn_output, context), dim=1)
            
            mel_output = self.linear_projection(decoder_rnn_output)
            stop_output = self.stop_layer(decoder_rnn_output)
            
            mel_outputs += [mel_output.squeeze(1)]
            alignments += [alignment]
            
            if torch.sigmoid(stop_output.data) > stop_threshold and len(mel_outputs) >= min_decoder_step:
                break
            if len(mel_outputs) >= max_decoder_step:
                print("Warning! Decoding steps reaches max decoder steps.")
                break

            decoder_input = mel_output[:,-self.num_mels:]


        mel_outputs, alignments, _  = self.parse_decoder_outputs(
            mel_outputs, alignments, None)

        return mel_outputs, alignments

    def inference_batched(self, memory, stop_threshold=0.5):
        """ Decoder inference
        Args:
            memory: (B, T_enc, D_enc) Encoder outputs
        Returns:
            mel_outputs: mel outputs from the decoder
            alignments: sequence of attention weights from the decoder
        """
        # [1, num_mels]
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        self.attention_layer.init_states(memory)
        
        mel_outputs, alignments = [], []
        stop_outputs = []
        # NOTE(sx): heuristic 
        max_decoder_step = memory.size(1)*self.encoder_down_factor//self.frames_per_step 
        min_decoder_step = memory.size(1)*self.encoder_down_factor // self.frames_per_step - 5
        while True:
            decoder_input = self.prenet(decoder_input)

            decoder_input_final, context, alignment = self.attend(decoder_input)

            #mel_output, stop_output, alignment = self.decode(decoder_input)
            decoder_rnn_output = self.decode(decoder_input_final)
            if self.concat_context_to_last:    
                decoder_rnn_output = torch.cat(
                    (decoder_rnn_output, context), dim=1)
            
            mel_output = self.linear_projection(decoder_rnn_output)
            # (B, 1)
            stop_output = self.stop_layer(decoder_rnn_output)
            stop_outputs += [stop_output.squeeze()]
            # stop_outputs.append(stop_output) 

            mel_outputs += [mel_output.squeeze(1)]
            alignments += [alignment]
            # print(stop_output.shape)
            if torch.all(torch.sigmoid(stop_output.squeeze().data) > stop_threshold) \
                    and len(mel_outputs) >= min_decoder_step:
                break
            if len(mel_outputs) >= max_decoder_step:
                print("Warning! Decoding steps reaches max decoder steps.")
                break

            decoder_input = mel_output[:,-self.num_mels:]


        mel_outputs, alignments, stop_outputs = self.parse_decoder_outputs(
            mel_outputs, alignments, stop_outputs)
        mel_outputs_stacked = []
        for mel, stop_logit in zip(mel_outputs, stop_outputs):
            idx = np.argwhere(torch.sigmoid(stop_logit.cpu()) > stop_threshold)[0][0].item()
            mel_outputs_stacked.append(mel[:idx,:])
        mel_outputs = torch.cat(mel_outputs_stacked, dim=0).unsqueeze(0)
        return mel_outputs, alignments

class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """
    def __init__(self, num_mels=80,
                 num_layers=5,
                 hidden_dim=512,
                 kernel_size=5):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                Conv1d(
                    num_mels, hidden_dim,
                    kernel_size=kernel_size, stride=1,
                    padding=int((kernel_size - 1) / 2),
                    dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hidden_dim)))

        for i in range(1, num_layers - 1):
            self.convolutions.append(
                nn.Sequential(
                    Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size, stride=1,
                        padding=int((kernel_size - 1) / 2),
                        dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hidden_dim)))

        self.convolutions.append(
            nn.Sequential(
                Conv1d(
                    hidden_dim, num_mels,
                    kernel_size=kernel_size, stride=1,
                    padding=int((kernel_size - 1) / 2),
                    dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(num_mels)))

    def forward(self, x):
        # x: (B, num_mels, T_dec)
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x

class MelDecoderMOLv2(AbsMelDecoder):
    """Use an encoder to preprocess ppg."""
    def __init__(
        self,
        num_speakers: int,
        spk_embed_dim: int,
        bottle_neck_feature_dim: int,
        encoder_dim: int = 256,
        encoder_downsample_rates: List = [2, 2],
        attention_rnn_dim: int = 512,
        attention_dim: int = 512,
        decoder_rnn_dim: int = 512,
        num_decoder_rnn_layer: int = 1,
        concat_context_to_last: bool = True,
        prenet_dims: List = [256, 128],
        prenet_dropout: float = 0.5,
        num_mixtures: int = 5,
        frames_per_step: int = 2,
        postnet_num_layers: int = 5,
        postnet_hidden_dim: int = 512,
        mask_padding: bool = True,
        use_spk_dvec: bool = False,
    ):
        super().__init__()
        
        self.mask_padding = mask_padding
        self.bottle_neck_feature_dim = bottle_neck_feature_dim
        self.num_mels = 80
        self.encoder_down_factor=np.cumprod(encoder_downsample_rates)[-1]
        self.frames_per_step = frames_per_step
        self.multi_speaker = True if num_speakers > 1 or self.use_spk_dvec else False
        self.use_spk_dvec = use_spk_dvec

        input_dim = bottle_neck_feature_dim
        
        # Downsampling convolution
        self.bnf_prenet = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim, encoder_dim, kernel_size=1, bias=False),
            torch.nn.LeakyReLU(0.1),

            torch.nn.InstanceNorm1d(encoder_dim, affine=False),
            torch.nn.Conv1d(
                encoder_dim, encoder_dim, 
                kernel_size=2*encoder_downsample_rates[0], 
                stride=encoder_downsample_rates[0], 
                padding=encoder_downsample_rates[0]//2,
            ),
            torch.nn.LeakyReLU(0.1),
            
            torch.nn.InstanceNorm1d(encoder_dim, affine=False),
            torch.nn.Conv1d(
                encoder_dim, encoder_dim, 
                kernel_size=2*encoder_downsample_rates[1], 
                stride=encoder_downsample_rates[1], 
                padding=encoder_downsample_rates[1]//2,
            ),
            torch.nn.LeakyReLU(0.1),

            torch.nn.InstanceNorm1d(encoder_dim, affine=False),
        )
        decoder_enc_dim = encoder_dim
        self.pitch_convs = torch.nn.Sequential(
            torch.nn.Conv1d(2, encoder_dim, kernel_size=1, bias=False),
            torch.nn.LeakyReLU(0.1),

            torch.nn.InstanceNorm1d(encoder_dim, affine=False),
            torch.nn.Conv1d(
                encoder_dim, encoder_dim, 
                kernel_size=2*encoder_downsample_rates[0], 
                stride=encoder_downsample_rates[0], 
                padding=encoder_downsample_rates[0]//2,
            ),
            torch.nn.LeakyReLU(0.1),
            
            torch.nn.InstanceNorm1d(encoder_dim, affine=False),
            torch.nn.Conv1d(
                encoder_dim, encoder_dim, 
                kernel_size=2*encoder_downsample_rates[1], 
                stride=encoder_downsample_rates[1], 
                padding=encoder_downsample_rates[1]//2,
            ),
            torch.nn.LeakyReLU(0.1),

            torch.nn.InstanceNorm1d(encoder_dim, affine=False),
        )
        
        if self.multi_speaker:
            if not self.use_spk_dvec:
                self.speaker_embedding_table = nn.Embedding(num_speakers, spk_embed_dim)
            self.reduce_proj = torch.nn.Linear(encoder_dim + spk_embed_dim, encoder_dim)

        # Decoder
        self.decoder = Decoder(
            enc_dim=decoder_enc_dim,
            num_mels=self.num_mels,
            frames_per_step=frames_per_step,
            attention_rnn_dim=attention_rnn_dim,
            decoder_rnn_dim=decoder_rnn_dim,
            num_decoder_rnn_layer=num_decoder_rnn_layer,
            prenet_dims=prenet_dims,
            num_mixtures=num_mixtures,
            use_stop_tokens=True,
            concat_context_to_last=concat_context_to_last,
            encoder_down_factor=self.encoder_down_factor,
        )

        # Mel-Spec Postnet: some residual CNN layers
        self.postnet = Postnet()
    
    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths, outputs[0].size(1))
            mask = mask.unsqueeze(2).expand(mask.size(0), mask.size(1), self.num_mels)
            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
        return outputs

    def forward(
        self,
        bottle_neck_features: torch.Tensor,
        feature_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        logf0_uv: torch.Tensor = None,
        spembs: torch.Tensor = None,
        styleembs: torch.Tensor = None,
        output_att_ws: bool = False,
    ):
        decoder_inputs = self.bnf_prenet(
            bottle_neck_features.transpose(1, 2)
        ).transpose(1, 2)
        logf0_uv = self.pitch_convs(logf0_uv.transpose(1, 2)).transpose(1, 2)
        decoder_inputs = decoder_inputs + logf0_uv
            
        if self.multi_speaker:
            assert spembs is not None
            if not self.use_spk_dvec:
                spk_embeds = self.speaker_embedding_table(spembs)
                spk_embeds = F.normalize(
                    spk_embeds).unsqueeze(1).expand(-1, decoder_inputs.size(1), -1)
            else:
                spk_embeds = F.normalize(
                    spembs).unsqueeze(1).expand(-1, decoder_inputs.size(1), -1)
            decoder_inputs = torch.cat([decoder_inputs, spk_embeds], dim=-1)
            decoder_inputs = self.reduce_proj(decoder_inputs)
        
        # (B, num_mels, T_dec)
        mel_outputs, predicted_stop, alignments = self.decoder(
            decoder_inputs, speech, feature_lengths//int(self.encoder_down_factor))
        ## Post-processing
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(1, 2)).transpose(1, 2)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        if output_att_ws: 
            return self.parse_output(
                [mel_outputs, mel_outputs_postnet, predicted_stop, alignments], speech_lengths)
        else:
            return self.parse_output(
                [mel_outputs, mel_outputs_postnet, predicted_stop], speech_lengths)

        # return mel_outputs, mel_outputs_postnet

    def inference(
        self,
        bottle_neck_features: torch.Tensor,
        logf0_uv: torch.Tensor = None,
        spembs: torch.Tensor = None,
        use_stop_tokens: bool = True,
    ):
        decoder_inputs = self.bnf_prenet(bottle_neck_features.transpose(1, 2)).transpose(1, 2)
        logf0_uv = self.pitch_convs(logf0_uv.transpose(1, 2)).transpose(1, 2)
        decoder_inputs = decoder_inputs + logf0_uv
        if self.multi_speaker:
            assert spembs is not None
            # spk_embeds = self.speaker_embedding_table(spembs)
            # spk_embeds = F.normalize(
                # spk_embeds).unsqueeze(1).expand(-1, bottle_neck_features.size(1), -1)
            if not self.use_spk_dvec:
                spk_embeds = self.speaker_embedding_table(spembs)
                spk_embeds = F.normalize(
                    spk_embeds).unsqueeze(1).expand(-1, decoder_inputs.size(1), -1)
            else:
                spk_embeds = F.normalize(
                    spembs).unsqueeze(1).expand(-1, decoder_inputs.size(1), -1)
            bottle_neck_features = torch.cat([decoder_inputs, spk_embeds], dim=-1)
            bottle_neck_features = self.reduce_proj(bottle_neck_features)

        ## Decoder
        if bottle_neck_features.size(0) > 1:
            mel_outputs, alignments = self.decoder.inference_batched(bottle_neck_features)
        else:
            mel_outputs, alignments = self.decoder.inference(bottle_neck_features,)
        ## Post-processing
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(1, 2)).transpose(1, 2)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        # outputs = mel_outputs_postnet[0]
        
        return mel_outputs[0], mel_outputs_postnet[0], alignments[0]

class AbsMelDecoder(torch.nn.Module, ABC):
    """The abstract PPG-based voice conversion class
    This "model" is one of mediator objects for "Task" class.

    """
    
    @abstractmethod
    def forward(
        self, 
        bottle_neck_features: torch.Tensor,
        feature_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        logf0_uv: torch.Tensor = None,
        spembs: torch.Tensor = None,
        styleembs: torch.Tensor = None,
    ) -> torch.Tensor:
        raise NotImplementedError

def load_hparams(filename):
    stream = open(filename, 'r')
    docs = yaml.safe_load_all(stream)
    hparams_dict = dict()
    for doc in docs:
        for k, v in doc.items():
            hparams_dict[k] = v
    return hparams_dict

def merge_dict(user, default):
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                user[k] = merge_dict(user[k], v)
    return user

class Dotdict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value

class HpsYaml(Dotdict):
    def __init__(self, yaml_file):
        super(Dotdict, self).__init__()
        hps = load_hparams(yaml_file)
        hp_dict = Dotdict(hps)
        for k, v in hp_dict.items():
            setattr(self, k, v)

    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__
    
def compute_f0(wav, sr=16000, frame_period=10.0):
    """Compute f0 from wav using pyworld harvest algorithm."""
    wav = wav.astype(np.float64)
    f0, _ = pyworld.harvest(
        wav, sr, frame_period=frame_period, f0_floor=20.0, f0_ceil=600.0)
    return f0.astype(np.float32)

def low_pass_filter(x, fs, cutoff=70, padding=True):
    """FUNCTION TO APPLY LOW PASS FILTER

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low pass filter

    Return:
        (ndarray): Low pass filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    numtaps = 255
    fil = firwin(numtaps, norm_cutoff)
    x_pad = np.pad(x, (numtaps, numtaps), 'edge')
    lpf_x = lfilter(fil, 1, x_pad)
    lpf_x = lpf_x[numtaps + numtaps // 2: -numtaps // 2]

    return lpf_x

def convert_continuous_f0(f0):
    """CONVERT F0 TO CONTINUOUS F0

    Args:
        f0 (ndarray): original f0 sequence with the shape (T)

    Return:
        (ndarray): continuous f0 with the shape (T)
    """
    # get uv information as binary
    uv = np.float32(f0 != 0)

    # get start and end of f0
    if (f0 == 0).all():
        logging.warn("all of the f0 values are 0.")
        return uv, f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]

    # padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nz_frames = np.where(f0 != 0)[0]

    # perform linear interpolation
    f = interp1d(nz_frames, f0[nz_frames])
    cont_f0 = f(np.arange(0, f0.shape[0]))

    return uv, cont_f0

def get_cont_lf0(f0, frame_period=10.0, lpf=False):
    uv, cont_f0 = convert_continuous_f0(f0)
    if lpf:
        cont_f0_lpf = low_pass_filter(cont_f0, int(1.0 / (frame_period * 0.001)), cutoff=20)
        cont_lf0_lpf = cont_f0_lpf.copy()
        nonzero_indices = np.nonzero(cont_lf0_lpf)
        cont_lf0_lpf[nonzero_indices] = np.log(cont_f0_lpf[nonzero_indices])
        # cont_lf0_lpf = np.log(cont_f0_lpf)
        return uv, cont_lf0_lpf 
    else:
        nonzero_indices = np.nonzero(cont_f0)
        cont_lf0 = cont_f0.copy()
        cont_lf0[cont_f0>0] = np.log(cont_f0[cont_f0>0])
        return uv, cont_lf0

def build_ppg2mel_model(model_config, model_file, device):
    # model_class = BiRnnPpg2MelModel

    ppg2mel_model = MelDecoderMOLv2(**model_config["model"]).to(device)
    ckpt = torch.load(model_file, map_location=device)
    ppg2mel_model.load_state_dict(ckpt["model"])
    ppg2mel_model.eval()
    return ppg2mel_model

def compute_f0(wav, sr=16000, frame_period=10.0):
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(
        wav, sr, frame_period=frame_period, f0_floor=20.0, f0_ceil=600.0)
    return f0

def compute_mean_std(lf0):
    nonzero_indices = np.nonzero(lf0)
    mean = np.mean(lf0[nonzero_indices])
    std = np.std(lf0[nonzero_indices])
    return mean, std 

def f02lf0(f0):
    lf0 = f0.copy()
    nonzero_indices = np.nonzero(f0)
    lf0[nonzero_indices] = np.log(f0[nonzero_indices])
    return lf0

def get_converted_lf0uv(
    wav, 
    lf0_mean_trg, 
    lf0_std_trg,
    convert=True,
):
    f0_src = compute_f0(wav)
    if not convert:
        uv, cont_lf0 = get_cont_lf0(f0_src)
        lf0_uv = np.concatenate([cont_lf0[:, np.newaxis], uv[:, np.newaxis]], axis=1)
        return lf0_uv

    lf0_src = f02lf0(f0_src)
    lf0_mean_src, lf0_std_src = compute_mean_std(lf0_src)
    
    lf0_vc = lf0_src.copy()
    lf0_vc[lf0_src > 0.0] = (lf0_src[lf0_src > 0.0] - lf0_mean_src) / lf0_std_src * lf0_std_trg + lf0_mean_trg
    f0_vc = lf0_vc.copy()
    f0_vc[lf0_src > 0.0] = np.exp(lf0_vc[lf0_src > 0.0])
    
    uv, cont_lf0_vc = get_cont_lf0(f0_vc)
    lf0_uv = np.concatenate([cont_lf0_vc[:, np.newaxis], uv[:, np.newaxis]], axis=1)
    return lf0_uv

def compute_spk_dvec(
    wav_path, weights_fpath="/home/kevingenghaopeng/vc/seq2seq-vc/seq2seq_vc/vocoder/speaker_encoder.pt",
):
    fpath = Path(wav_path)
    wav = preprocess_wav(fpath)
    encoder = SpeakerEncoder(weights_fpath)
    spk_dvec = encoder.embed_utterance(wav)
    return spk_dvec

@typechecked
class BiRnnPpg2MelModel(AbsMelDecoder):
    """ Bidirectional RNN-based PPG-to-Mel Model for voice conversion tasks.
        RNN could be LSTM-based or GRU-based.
    """
    def __init__(
        self,
        input_size: int, 
        multi_spk: bool = False,    
        num_speakers: int = 1,
        spk_embed_dim: int = 256,
        use_spk_dvec: bool = False,
        multi_styles: bool =  False,
        num_styles: int = 3,
        style_embed_dim: int = 256,
        dense_layer_size: int = 256,
        num_layers: int = 4,
        bidirectional: bool = True,
        hidden_dim: int = 256,
        dropout_rate: float = 0.5,
        output_size: int = 80,
        rnn_type: str = "lstm"
    ):

        super().__init__()

        self.multi_spk = multi_spk
        self.spk_embed_dim = spk_embed_dim
        self.use_spk_dvec= use_spk_dvec
        self.multi_styles = multi_styles
        self.style_embed_dim = style_embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        
        self.ppg_dense_layer = nn.Linear(input_size - 2, hidden_dim)
        self.logf0_uv_layer = nn.Linear(2, hidden_dim)

        projection_input_size = hidden_dim
        if self.multi_spk:
            if not self.use_spk_dvec:         
                self.spk_embedding = nn.Embedding(num_speakers, spk_embed_dim)
            projection_input_size += self.spk_embed_dim
        if self.multi_styles:
            self.style_embedding = nn.Embedding(num_styles, style_embed_dim)
            projection_input_size += self.style_embed_dim

        self.reduce_proj = nn.Sequential(
            nn.Linear(projection_input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        rnn_type = rnn_type.upper()
        if rnn_type in ["LSTM", "GRU"]:
            rnn_class = getattr(nn, rnn_type)
            self.rnn = rnn_class(
                hidden_dim, hidden_dim, num_layers, 
                bidirectional=bidirectional,
                dropout=dropout_rate,
                batch_first=True)
        else:
            # Default: use BiLSTM
            self.rnn = nn.LSTM(
                hidden_dim, hidden_dim, num_layers, 
                bidirectional=bidirectional,
                dropout_rate=dropout_rate,
                batch_first=True)
        # Fully connected layers
        self.hidden2out_layers = nn.Sequential(
            nn.Linear(self.num_direction * hidden_dim, dense_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_layer_size, output_size)
        )
    
    def forward(
        self, 
        ppg: torch.Tensor,
        ppg_lengths: torch.Tensor,
        logf0_uv: torch.Tensor,
        spembs: torch.Tensor = None,
        styleembs: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            ppg (tensor): [B, T, D_ppg]
            ppg_lengths (tensor): [B,]
            logf0_uv (tensor): [B, T, 2], concatented logf0 and u/v flags.
            spembs (tensor): [B,] index-represented speaker.
            styleembs (tensor): [B,] index-repreented speaking style (e.g. emotion). 
        """
        ppg = self.ppg_dense_layer(ppg)
        logf0_uv = self.logf0_uv_layer(logf0_uv)

        ## Concatenate/add ppg and logf0_uv
        x = ppg + logf0_uv
        B, T, _ = x.size()

        if self.multi_spk:
            assert spembs is not None
            # spk_embs = self.spk_embedding(torch.LongTensor([0,]*ppg.size(0)).to(ppg.device))
            if not self.use_spk_dvec:
                spk_embs = self.spk_embedding(spembs)
                spk_embs = torch.nn.functional.normalize(
                    spk_embs).unsqueeze(1).expand(-1, T, -1)
            else:
                spk_embs = torch.nn.functional.normalize(
                    spembs).unsqueeze(1).expand(-1, T, -1)
            x = torch.cat([x, spk_embs], dim=2)
        
        if self.multi_styles and styleembs is not None:
            style_embs = self.style_embedding(styleembs)
            style_embs = torch.nn.functional.normalize(
                style_embs).unsqueeze(1).expand(-1, T, -1)
            x = torch.cat([x, style_embs], dim=2)
        ## FC projection
        x = self.reduce_proj(x)

        if ppg_lengths is not None:
            x = torch.nn.utils.rnn.pack_padded_sequence(x, ppg_lengths,
                                                       batch_first=True,
                                                       enforce_sorted=False)
        x, _ = self.rnn(x)
        if ppg_lengths is not None:
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.hidden2out_layers(x)
        
        return x

class PPG_Feat2Wav(object):
    def __init__(self, checkpoint, config, stats, trg_stats, device):

        self.device = device

        # upstream stats, used to denormalize the converted feature
        self.trg_stats = {
            "mean": torch.tensor(trg_stats["mean"], dtype=torch.float).to(self.device),
            "scale": torch.tensor(trg_stats["scale"], dtype=torch.float).to(
                self.device
            ),
        }

        # mel stats of the downstream model, only used in the autoregressive downstream model
        downstream_mel_stats = {
            "mean": torch.tensor(read_hdf5(stats, "mean"), dtype=torch.float).to(
                self.device
            ),
            "scale": torch.tensor(read_hdf5(stats, "scale"), dtype=torch.float).to(
                self.device
            ),
        }
        # get model and load parameters
        # load the model.keys(), remove unnecessary keys from ppg2mel_config
        ppg2mel_config = HpsYaml(config) 
        ppg2mel_model_file = checkpoint
        ppg2mel_model = build_ppg2mel_model(ppg2mel_config, ppg2mel_model_file, self.device) 

        self.ppg2mel_model = ppg2mel_model
        
        # Data related
        logging.info(f"Loaded S3PRL model parameters from {checkpoint}.")

        # Speaker encoder, usage:
        # compute_spk_dvec(wav_path)
        self.speaker_embedding_model = SpeakerEncoder(weights_fpath="/home/kevingenghaopeng/vc/seq2seq-vc/seq2seq_vc/vocoder/speaker_encoder.pt")

        # compute lf0uv
        # self.f0_function = f02lf0(compute_f0())
        
        self.vocoder = load_hifigan_generator(device)
        # Vocoder
        # if self.config.get("vocoder", None) is not None:
        #     self.vocoder = Vocoder(
        #         self.config["vocoder"]["checkpoint"],
        #         self.config["vocoder"]["config"],
        #         self.config["vocoder"]["stats"],
        #         device,
        #         take_norm_feat=False,
        #     )
        # else:
        #     self.vocoder = Spectrogram2Waveform(
        #         n_fft=self.config["fft_size"],
        #         n_shift=self.config["hop_size"],
        #         fs=self.config["sampling_rate"],
        #         n_mels=self.config["num_mels"],
        #         fmin=self.config["fmin"],
        #         fmax=self.config["fmax"],
        #         griffin_lim_iters=64,
        #         take_norm_feat=False,
        #     )

    def decode(self, c):
        # denormalize with target stats
        # here c is the converted PPG-like feature
        # ref_wav_path = args.ref_wav_path
        # ref_fid = os.path.basename(ref_wav_path)[:-4]
        # ref_spk_dvec = compute_spk_dvec(ref_wav_path)
        # ref_spk_dvec = torch.from_numpy(ref_spk_dvec).unsqueeze(0).to(device)
        # ref_wav, _ = librosa.load(ref_wav_path, sr=16000)
        # ref_lf0_mean, ref_lf0_std = compute_mean_std(f02lf0(compute_f0(ref_wav)))
        
        # source_file_list = sorted(glob.glob(f"{args.src_wav_dir}/*.wav"))
        # print(f"Number of source utterances: {len(source_file_list)}.")
        # for src_wav_path in tqdm(source_file_list):
        #     # Load the audio to a numpy array:
        #     src_wav, _ = librosa.load(src_wav_path, sr=16000)
        #     src_wav_tensor = torch.from_numpy(src_wav).unsqueeze(0).float().to(device)
        #     src_wav_lengths = torch.LongTensor([len(src_wav)]).to(device)
        #     ppg = ppg_model(src_wav_tensor, src_wav_lengths)
        #     lf0_uv = get_converted_lf0uv(src_wav, ref_lf0_mean, ref_lf0_std, convert=True)
        #     min_len = min(ppg.shape[1], len(lf0_uv))

        #     ppg = ppg[:, :min_len]
        #     # import pdb; pdb.set_trace()
        #     lf0_uv = lf0_uv[:min_len]
            
        #     start = time.time()
        #     import pdb; pdb.set_trace()
        #     if isinstance(ppg2mel_model, BiRnnPpg2MelModel):
        #         ppg_length = torch.LongTensor([ppg.shape[1]]).to(device)
        #         logf0_uv=torch.from_numpy(lf0_uv).unsqueeze(0).float().to(device)
        #         mel_pred = ppg2mel_model(ppg, ppg_length, logf0_uv, ref_spk_dvec)
        #     else:
        #         _, mel_pred, att_ws = ppg2mel_model.inference(
        #             ppg,
        #             logf0_uv=torch.from_numpy(lf0_uv).unsqueeze(0).float().to(device),
        #             spembs=ref_spk_dvec,
        #             use_stop_tokens=True,
        #     )
        
        c = c * self.trg_stats["scale"] + self.trg_stats["mean"]
        lens = torch.LongTensor([c.shape[0]]).to(self.device)
        c = c.unsqueeze(0).float()

        start = time.time()
        # outs, _ , _ = self.model(c, lens, spk_embs=None) # outs: if use diffusion / TACO2 model, return 3 variables
        # import pdb; pdb.set_trace()
        # out = outs[0]
        # y, sr = self.vocoder.decode(out)
        # import pdb; pdb.set_trace()
        # Todo
        # spk_v_emb = self.speaker_embedding_model.embed_utterance(c).unsqueeze(0).to(self.device)
        # create a [1, 256] tensor as dummy speaker embedding, range from 0 to 1
        spk_v_emb = torch.rand(1, 256).to(self.device)
        # create a dummy lf0_uv tensor, length is 
        import pdb; pdb.set_trace()
        __, mel_pred, att_ws = self.ppg2mel_model.inference(c,
                                                            logf0_uv=None,
                                                            spembs=spk_v_emb,
                                                            use_stop_tokens=True,
                                                            ).to(self.device)

        pdb.set_trace()
        mel_pred = mel_pred.squeeze(0)
        y = self.vocoder(mel_pred)
        rtf = (time.time() - start) / (len(y) / self.config["sampling_rate"])
        logging.info(f"Finished waveform generation. (RTF = {rtf:.03f}).")
        return y, self.config["sampling_rate"]
        # mel_pred = mel_pred * self.trg_stats["scale"] + self.trg_stats["mean"]
        