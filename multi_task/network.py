#@markdown - `SinusoidalPosEmb` Positional encoding for the diffusion iteration k
#@markdown - `Downsample1d` Strided convolution to reduce temporal resolution
#@markdown - `Upsample1d` Transposed convolution to increase temporal resolution
#@markdown - `Conv1dBlock` Conv1d --> GroupNorm --> Mish
#@markdown - `ConditionalResidualBlock1D` Takes two inputs `x` and `cond`. \
#@markdown `x` is passed through 2 `Conv1dBlock` stacked together with residual connection.
#@markdown `cond` is applied to `x` with [FiLM](https://arxiv.org/abs/1709.07871) conditioning.

import math
from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import vae.network

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # PE_2i = sin(pos/(10000^(2i/dim)))
        # PE_2i+1 = cos(pos/(10000^(2i/dim)))
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1) # 2*ln(10000)/dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb) # 10000^(2i/dim)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

class ConvEncoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, horizon=11, kernel_size=3, n_groups=8, noise_fingering=0, noise_ft=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.noise_fingering = noise_fingering
        self.noise_ft = noise_ft
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, mid_channels),
            nn.Mish(),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
        )
        self.output_dim = self.get_conv_output_dim(in_channels, horizon)
        self.latent_dim = self.output_dim
        self.kl = 0

    def get_conv_output_dim(self, in_channels, horizon):
        x = torch.randn(1, in_channels, horizon)
        x = self.block(x)
        x = self.block2(x)
        return x.flatten(start_dim=1).shape[-1]

    def forward(self, x):
        x = x.reshape(x.shape[0], self.in_channels, -1)
        if self.noise_ft > 0:
            x_nn = x.clone()
            x_nn[:,16:52,:] = x[:,16:52,:] + torch.randn_like(x[:,16:52,:])*self.noise_ft-self.noise_ft/2
            x = x_nn
        x = self.block(x)
        x = self.block2(x)
        return x
    
    def forward_without_sampling(self, x):
        x = x.reshape(x.shape[0], self.in_channels, -1)
        # Add gaussian noise
        if self.noise_fingering > 0:
            x_new = x.clone()
            x_new[:,16:26,:] = x[:,16:26,:] + torch.randn_like(x[:,16:26,:])*self.noise_fingering-self.noise_fingering/2
        if self.noise_ft > 0:
            x_nn = x_new.clone()
            x_nn[:,26:62,:] = x_new[:,26:62,:] + torch.randn_like(x[:,26:62,:])*self.noise_ft-self.noise_ft/2
            x = x_nn
        x = self.block(x)
        x = self.block2(x)
        return x

class BetVariationalConvMlpEncoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bet_horizon=16, horizon=11, cond_dim=36, latent_dim=16, kernel_size=3, n_groups=8, device='cuda', beta=1e-6, noise=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bet_horizon = bet_horizon
        self.latent_dim = latent_dim
        self.noise = noise
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, mid_channels),
            nn.Mish(),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
        )
        # self.block3 = nn.Sequential(
        #     nn.Conv1d(out_channels, out_channels*2, kernel_size, padding=kernel_size // 2),
        #     nn.GroupNorm(n_groups, out_channels*2),
        # )
        self.output_dim = self.get_conv_output_dim(in_channels, horizon)
        # self.mlp = nn.Sequential(
        #         nn.Linear(self.output_dim+cond_dim, latent_dim*4),
        #         nn.Mish(),
        #         nn.Linear(latent_dim*4, latent_dim),
        # )
        self.mlp = nn.Sequential(
            nn.Linear(self.output_dim+cond_dim, latent_dim),
        )
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0
        self.beta = beta

    def get_conv_output_dim(self, in_channels, horizon):
        x = torch.randn(1, in_channels, horizon)
        x = self.block(x)
        x = self.block2(x)
        # x = self.block3(x)
        return x.flatten(start_dim=1).shape[-1]

    def forward(self, x, cond):
        x = x.reshape(x.shape[0], self.bet_horizon, self.in_channels, -1)
        x = self.block(x)
        x = self.block2(x)
        # x = self.block3(x)
        # Add gaussian noise to cond
        cond = cond + torch.randn_like(cond)*self.noise-self.noise/2
        x = self.mlp(torch.cat([x.flatten(start_dim=1), cond], dim=-1))
        mu = self.fc_mu(x)
        # print("mu", mu.shape)
        sigma = torch.exp(self.fc_var(x))
        # print("sigma", sigma.shape)
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = self.beta*(sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

    def forward_without_sampling(self, x, cond):
        batch_size = x.shape[0]
        x = x.reshape(x.shape[0]*self.bet_horizon, self.in_channels, -1)
        x = self.block(x)
        x = self.block2(x)
        cond = cond + torch.randn_like(cond)*self.noise-self.noise/2
        x = x.reshape(batch_size, self.bet_horizon, -1)
        x = self.mlp(torch.cat([x.flatten(start_dim=2), cond], dim=-1))
        mu = self.fc_mu(x)
        self.kl = 0
        return mu

class VariationalConvMlpEncoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, horizon=11, cond_dim=36, latent_dim=16, kernel_size=3, n_groups=8, device='cuda', beta=1e-6, noise=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.noise = noise
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, mid_channels),
            nn.Mish(),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
        )
        # self.block3 = nn.Sequential(
        #     nn.Conv1d(out_channels, out_channels*2, kernel_size, padding=kernel_size // 2),
        #     nn.GroupNorm(n_groups, out_channels*2),
        # )
        self.output_dim = self.get_conv_output_dim(in_channels, horizon)
        # self.mlp = nn.Sequential(
        #         nn.Linear(self.output_dim+cond_dim, latent_dim*4),
        #         nn.Mish(),
        #         nn.Linear(latent_dim*4, latent_dim),
        # )
        self.mlp = nn.Sequential(
            nn.Linear(self.output_dim+cond_dim, latent_dim),
        )
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0
        self.beta = beta

    def get_conv_output_dim(self, in_channels, horizon):
        x = torch.randn(1, in_channels, horizon)
        x = self.block(x)
        x = self.block2(x)
        # x = self.block3(x)
        return x.flatten(start_dim=1).shape[-1]

    def forward(self, x, cond):
        x = x.reshape(x.shape[0], self.in_channels, -1)
        x = self.block(x)
        x = self.block2(x)
        # x = self.block3(x)
        # Add gaussian noise to cond
        cond = cond + torch.randn_like(cond)*self.noise-self.noise/2
        x = self.mlp(torch.cat([x.flatten(start_dim=1), cond], dim=-1))
        mu = self.fc_mu(x)
        # print("mu", mu.shape)
        sigma = torch.exp(self.fc_var(x))
        # print("sigma", sigma.shape)
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = self.beta*(sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

    def forward_without_sampling(self, x, cond):
        x = x.reshape(x.shape[0], self.in_channels, -1)
        x = self.block(x)
        x = self.block2(x)
        cond = cond + torch.randn_like(cond)*self.noise-self.noise/2
        x = self.mlp(torch.cat([x.flatten(start_dim=1), cond], dim=-1))
        mu = self.fc_mu(x)
        self.kl = 0
        return mu

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8,
            midi_dim=0,
            midi_cond_dim=0,
            freeze_encoder=False,
            midi_encoder=None):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        self.midi_dim = midi_dim
        self.midi_cond_dim = midi_cond_dim
        self.kl = 0
        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        # Use conv1d for first conditioning layer
        if midi_dim > 0:
            # self.midi_encoder = midi_encoder
            self.midi_encoder = midi_encoder()
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(self.midi_encoder.latent_dim+cond_dim-midi_dim, cond_channels),
                # nn.Mish(), # Comment for high level
                # nn.Linear(cond_channels, cond_channels), # Comment for high level
                nn.Unflatten(-1, (-1, 1))
            )
            if freeze_encoder:
                for param in self.midi_encoder.parameters():
                    param.requires_grad = False
        else:
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),
                nn.Unflatten(-1, (-1, 1))
            )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
    
    def get_midi_encoder_output_dim(self, midi_dim):
        x = torch.randn(1, 88, 1)
        x = self.midi_encoder(x)
        return x.flatten(start_dim=1).shape[-1]

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        if self.midi_dim == 0:
            embed = self.cond_encoder(cond)
        else:
            # cond_1 (B, 88*n_steps)
            if self.midi_cond_dim == 0:
                midi = cond[:, :self.midi_dim]
                cond_midi = self.midi_encoder(midi)
            else:
                midi = cond[:, :self.midi_dim-self.midi_cond_dim]
                cond_midi = cond[:, self.midi_dim-self.midi_cond_dim:self.midi_dim]
                cond_midi = self.midi_encoder.forward_without_sampling(midi, cond_midi)
            cond_2 = cond[:,self.midi_dim:]
            cond_midi = cond_midi.flatten(start_dim=1)
            # cond_2 = self.cond_encoder_2(cond_2)
            embed = torch.cat([cond_midi, cond_2], dim=-1)
            embed = self.cond_encoder(embed)
        self.kl = self.midi_encoder.kl
            
        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=4,
        midi_dim=0,
        midi_cond_dim=0,
        midi_encoder=None,
        freeze_encoder=False,
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        # print(all_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups, midi_dim=midi_dim, midi_encoder=midi_encoder, freeze_encoder=freeze_encoder, midi_cond_dim=midi_cond_dim
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups, midi_dim=midi_dim, midi_encoder=midi_encoder, freeze_encoder=freeze_encoder, midi_cond_dim=midi_cond_dim
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups, midi_dim=midi_dim, midi_encoder=midi_encoder, freeze_encoder=freeze_encoder, midi_cond_dim=midi_cond_dim),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups, midi_dim=midi_dim, midi_encoder=midi_encoder, freeze_encoder=freeze_encoder, midi_cond_dim=midi_cond_dim),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups, midi_dim=midi_dim, midi_encoder=midi_encoder, freeze_encoder=freeze_encoder, midi_cond_dim=midi_cond_dim),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups, midi_dim=midi_dim, midi_encoder=midi_encoder, freeze_encoder=freeze_encoder, midi_cond_dim=midi_cond_dim),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv
        self.midi_dim = midi_dim
        self.kl = 0

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            if self.midi_dim > 0:
                global_feature = torch.cat([
                    global_cond, global_feature
                ], axis=-1)
            else:
                global_feature = torch.cat([
                    global_feature, global_cond
                ], axis=-1)
        x = sample
        h = []
        self.kl = 0
        cnt = 0
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            self.kl += resnet.kl + resnet2.kl
            cnt += 2
            h.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
            self.kl += mid_module.kl
            cnt += 1

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            # for i in h:
            #     print(i.shape)
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            self.kl += resnet.kl + resnet2.kl
            cnt += 2
            x = upsample(x)
        
        self.kl = self.kl / cnt

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, mid_dim=1024, n_layers=3, n_groups=8, noise=0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.noise = noise
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, mid_dim),
                nn.GroupNorm(n_groups, mid_dim),
                nn.Mish(),
            )
        ])
        for i in range(n_layers-2):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(mid_dim, mid_dim),
                    nn.GroupNorm(n_groups, mid_dim),
                    nn.Mish(),
                )
            )
        self.layers.append(
            nn.Sequential(
                nn.Linear(mid_dim, out_dim),
            )
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], self.in_dim)
        if self.noise > 0:
            x = x + torch.randn_like(x)*self.noise-self.noise/2
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == '__main__':
    s = SinusoidalPosEmb(dim=10)
    t = torch.tensor([2])
    print(s(t))