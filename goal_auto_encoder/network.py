import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
import math
import vae.transformer

class ConditionalConv1dBlock(nn.Module):
    '''
        Conv1d --> BatchNorm --> Mish
    '''
    def __init__(self, in_channels, out_channels, cond_dim=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, 3, 2, 1),
            nn.BatchNorm1d(out_channels),
            nn.Mish(),
        )
        # FilM layer
        cond_channels = out_channels * 2
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, cond_channels),
            nn.Mish(),
            nn.Unflatten(-1, (-1, 1)),
        )
    
    def forward(self, x, cond):
        # x: (B, C, T)
        # cond: (B, cond_dim)
        cond = self.cond_encoder(cond)
        cond = cond.reshape(cond.shape[0], 2, self.out_channels, -1)
        scale = cond[:,0,...]
        bias = cond[:,1,...]
        x = self.block(x)
        x = x * scale + bias
        return x

class VariationalEncoder(nn.Module):
    def __init__(self, 
                 input_dim=88, 
                 input_channel=1, 
                 latent_dim=16, 
                 h_channels=[2, 4, 8, 16],
                 beta=1e-5,
                 device="cuda"):
        super(VariationalEncoder, self).__init__()
        modules = []
        self.input_dim = input_dim
        self.input_channel = input_channel
        self.latent_dim = latent_dim
        h_channels = [input_channel] + h_channels
        for i in range(len(h_channels)-1):
            modules.append(nn.Sequential(
                nn.Conv1d(h_channels[i], h_channels[i+1], 3, 2, 1),
                nn.BatchNorm1d(h_channels[i+1]),
                nn.Mish(),
                )
            )
        self.encoder = nn.Sequential(*modules)
        conv_out_size, self.final_dim = self.get_conv_out_size()
        self.fc_mu = nn.Linear(conv_out_size, latent_dim)
        self.fc_var = nn.Linear(conv_out_size, latent_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0
        self.beta = beta
    
    def get_conv_out_size(self):
        x = torch.randn((1, self.input_channel, self.input_dim))
        x = self.encoder(x)
        return x.numel(), x.shape[-1]
        
    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) to (B, C, T)
        x = x.moveaxis(-1,-2)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        x = x.flatten(start_dim=1)
        # print(x.shape)
        mu = self.fc_mu(x)
        # print("mu", mu.shape)
        sigma = torch.exp(self.fc_var(x))
        # print("sigma", sigma.shape)
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = self.beta*(sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

    def forward_without_sampling(self, x):
        # (B, T, C) to (B, C, T)
        x = x.moveaxis(-1,-2)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        x = x.flatten(start_dim=1)
        # print(x.shape)
        mu = self.fc_mu(x)
        # print("mu", mu.shape)
        sigma = torch.exp(self.fc_var(x))
        # print("sigma", sigma.shape)
        z = mu
        self.kl = self.beta*(sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class Conv1dEncoder(nn.Module):
    def __init__(self, 
                 horizon=10,
                 input_dim=16,  
                 latent_dim=16, 
                 cond_dim=36,
                 h_channels=[32, 64, 128],
                 beta=1e-5,
                 device="cuda"):
        super(Conv1dEncoder, self).__init__()
        self.horizon = horizon
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        modules = []
        h_channels = [input_dim] + h_channels
        for i in range(len(h_channels)-1):
            modules.append(nn.Sequential(
                nn.Conv1d(h_channels[i], h_channels[i+1], 3, 2, 1),
                nn.GroupNorm(1, h_channels[i+1]),
                nn.Mish(),
                )
            )
        self.encoder = nn.Sequential(*modules)

        self.out_dim, _ = self.get_conv_out_size()
        self.fc_mu = nn.Linear(self.out_dim+self.cond_dim, latent_dim)
        self.fc_var = nn.Linear(self.out_dim+self.cond_dim, latent_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0
        self.beta = beta
        
    def get_conv_out_size(self):
        x = torch.randn((1, self.input_dim, self.horizon))
        x = self.encoder(x)
        return x.numel(), x.shape[-1]
    
    def forward(self, 
                x: torch.Tensor,
                cond=None) -> torch.Tensor:
        x = self.encoder(x)

        x = x.flatten(start_dim=1)

        if cond is not None:
            x = torch.cat((x, cond), dim=-1)

        # print(x.shape)
        mu = self.fc_mu(x)
        # print("mu", mu.shape)
        sigma = torch.exp(self.fc_var(x))
        # print("sigma", sigma.shape)
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = self.beta*(sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class TransformerEncoder(nn.Module):
    def __init__(self, 
                 horizon=10,
                 input_dim=16,  
                 latent_dim=16, 
                 cond_dim=36,
                 beta=1e-5,
                 device="cuda"):
        super(TransformerEncoder, self).__init__()
        self.horizon = horizon
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.encoder = vae.transformer.Decoder(
                        embedding=vae.transformer.Embeddings(input_dim, 16),
                        positional_encoding=vae.transformer.PositionalEncoding(16, 0.1),
                        layer=vae.transformer.DecoderLayer(16, 
                                                 vae.transformer.MultiHeadedAttention(4, 16), 
                                                 vae.transformer.PositionwiseFeedForward(16, 128), 0.1), 
                        N=4)
        self.fc_mu = nn.Linear(input_dim+cond_dim, latent_dim)
        self.fc_var = nn.Linear(input_dim+cond_dim, latent_dim)
        self.out_dim = latent_dim

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0
        self.beta = beta
        
    def forward(self, 
                x: torch.Tensor,
                cond=None) -> torch.Tensor:
        # print(x.shape)
        x = self.encoder(x, mask=None)
        # Mean pooling
        x = x.mean(dim=1)
        # x = x.flatten(start_dim=1)

        if cond is not None:
            x = torch.cat((x, cond), dim=-1)

        # print(x.shape)
        mu = self.fc_mu(x)
        # print("mu", mu.shape)
        sigma = torch.exp(self.fc_var(x))
        # print("sigma", sigma.shape)
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = self.beta*(sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
    
class TransformerDecoder(nn.Module):
    def __init__(self, 
                 horizon=10,
                 latent_dim=16,
                 cond_dim=36,
                 output_dim=36,
                 mid_dims=[128, 128],
                 device="cuda"):
        super(TransformerDecoder, self).__init__()
        self.horizon = horizon
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        modules = []
        mid_dims = [latent_dim] + mid_dims
        for i in range(len(mid_dims)-1):
            modules.append(nn.Sequential(
                nn.Linear(mid_dims[i], mid_dims[i+1]),
                nn.LayerNorm(mid_dims[i+1]),
                nn.Mish(),
                )
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.Linear(mid_dims[-1], output_dim),
        )
    def forward(self, 
                z: torch.Tensor,
                cond=None) -> torch.Tensor:
        # Concatenate z and cond
        if cond is not None:
            z = torch.cat((z, cond), dim=-1)
        # print(x.shape)
        x = self.decoder(z)
        # print(x.shape)
        x = self.final_layer(x)
        # print(x.shape)
        return x
    
class TransformerAutoencoder(nn.Module):
    def __init__(self, 
                 horizon=10,
                 input_dim=16,  
                 latent_dim=16, 
                 output_dim=36,
                 device="cuda"):
        super(TransformerAutoencoder, self).__init__()
        self.horizon = horizon
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.encoder = TransformerEncoder(horizon=horizon, input_dim=input_dim, latent_dim=latent_dim, device=device)
        self.encoder = Conv1dEncoder(horizon=horizon, input_dim=input_dim, latent_dim=latent_dim, device=device)
        self.decoder = TransformerDecoder(horizon=horizon, latent_dim=latent_dim, output_dim=output_dim, device=device)
        self.device = device
        
    def forward(self, 
                x: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        x = self.encoder(x, cond)
        # print(x.shape)
        x = self.decoder(x)
        # print(x.shape)
        return x
    
    def forward_without_sampling(self, 
                x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        x = self.encoder.forward_without_sampling(x)
        # print(x.shape)
        x = self.decoder(x)
        # print(x.shape)
        return x


class Encoder(nn.Module):
    def __init__(self, 
                 input_dim=88, 
                 input_channel=1, 
                 latent_dim=16, 
                 h_channels=[2, 4, 8, 16],
                 device="cuda"):
        super(Encoder, self).__init__()
        modules = []
        self.input_dim = input_dim
        self.input_channel = input_channel
        self.latent_dim = latent_dim
        h_channels = [input_channel] + h_channels
        for i in range(len(h_channels)-1):
            modules.append(nn.Sequential(
                nn.Conv1d(h_channels[i], h_channels[i+1], 3, 2, 1),
                nn.BatchNorm1d(h_channels[i+1]),
                nn.Mish(),
                )
            )
        self.encoder = nn.Sequential(*modules)
        conv_out_size, self.final_dim = self.get_conv_out_size()
        self.fc = nn.Linear(conv_out_size, latent_dim)
        self.reg = 0
    
    def get_conv_out_size(self):
        x = torch.randn((1, self.input_channel, self.input_dim))
        x = self.encoder(x)
        return x.numel(), x.shape[-1]
        
    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) to (B, C, T)
        x = x.moveaxis(-1,-2)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        x = x.flatten(start_dim=1)
        # print(x.shape)
        z = self.fc(x)
        self.reg = (z**2).sum()
        return z
    
    def forward_without_sampling(self, 
                                 x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) to (B, C, T)
        x = x.moveaxis(-1,-2)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        x = x.flatten(start_dim=1)
        # print(x.shape)
        z = self.fc(x)
        self.reg = (z**2).sum()
        return z

class ConditionalDecoder(nn.Module):
    def __init__(self, final_dim, output_dim=1, output_channel=1, latent_dim=16, h_channels=[8, 4, 2], cond_dim=64):
        super(ConditionalDecoder, self).__init__()
        modules = []
        self.output_dim = output_dim
        self.output_channel = output_channel
        self.latent_dims = latent_dim
        self.final_dim = final_dim
        h_channels = h_channels + [output_channel]
        self.h_channels = h_channels
        self.h_channels = h_channels
        self.cond_dim = cond_dim
        self.decoder = nn.ModuleList([])
        for i in range(len(h_channels)-1):
            self.decoder.append(ConditionalConv1dBlock(h_channels[i], h_channels[i+1], cond_dim=cond_dim))

        deconv_out_size = self.get_deconv_out_size()
        self.final_layer = nn.Sequential(
            nn.Linear(deconv_out_size, output_dim),
            nn.BatchNorm1d(h_channels[-1]),
            nn.Sigmoid(),
        )

        # Use Mlp
        modules.append(nn.Sequential(
            nn.Linear(latent_dim+cond_dim, 16),
            nn.BatchNorm1d(16),
            nn.Mish(),
            )
        )
        
        modules.append(nn.Sequential(
            nn.Linear(16, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
            )
        )
        self.decoder = nn.Sequential(*modules)

        self.decoder_fc = nn.Linear(latent_dim, h_channels[0]*final_dim)
    
    def get_deconv_out_size(self):
        x = torch.randn((1, self.h_channels[0], self.final_dim))
        cond = torch.randn((1, 1, self.cond_dim))
        cond = cond.squeeze(1)
        for module in self.decoder:
            x = module(x, cond)
        return x.numel()

    def forward(self, z, cond):
        cond = cond.squeeze(1)
        # z = self.decoder_fc(z)
        # z = z.view(z.shape[0], self.h_channels[0], self.final_dim)
        # # z = self.decoder(z, cond)
        # for module in self.decoder:
        #     z = module(z, cond)
        # z = self.final_layer(z)

        z = torch.cat((z, cond), dim=-1)
        for module in self.decoder:
            z = module(z)

        # z = z.moveaxis(-2,-1)
        return z

class Decoder(nn.Module):
    def __init__(self, final_dim, output_dim=88, output_channel=1, latent_dim=16, h_channels=[16, 8, 4, 2]):
        super(Decoder, self).__init__()
        modules = []
        self.output_dim = output_dim
        self.output_channel = output_channel
        self.latent_dims = latent_dim
        self.final_dim = final_dim
        h_channels = h_channels + [output_channel]
        self.h_channels = h_channels
        self.h_channels = h_channels
        for i in range(len(h_channels)-1):
            modules.append(nn.Sequential(
                nn.ConvTranspose1d(h_channels[i], h_channels[i+1], 3, 2, 1),
                nn.BatchNorm1d(h_channels[i+1]), 
                nn.Mish(),
                )
            )
        self.decoder = nn.Sequential(*modules)
        deconv_out_size = self.get_deconv_out_size()
        self.final_layer = nn.Sequential(
            nn.Linear(deconv_out_size, output_dim),
            nn.BatchNorm1d(h_channels[-1]),
            nn.Sigmoid(),  
        )
        self.decoder_fc = nn.Linear(latent_dim, h_channels[0]*final_dim)
    
    def get_deconv_out_size(self):
        x = torch.randn((1, self.h_channels[0], self.final_dim))
        x = self.decoder(x)
        return x.numel()

    def forward(self, z):
        z = self.decoder_fc(z)
        z = z.view(z.shape[0], self.h_channels[0], self.final_dim)
        z = self.decoder(z)
        z = self.final_layer(z)
        z = z.moveaxis(-2,-1)
        return z

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # PE_2i = sin(pos/(10000^(2i/dim)))
        # PE_2i+1 = cos(pos/(10000^(2i/dim)))
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1) # 2*ln(10000)/dim
        emb = torch.exp(torch.arange(half_dim, device='cuda') * -emb) # 10000^(2i/dim)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Autoencoder(nn.Module):
    def __init__(self, latent_dim, cond_dim=64, device="cuda"):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.encoder = Encoder(latent_dim=latent_dim, device=device)
        # self.decoder = Decoder(latent_dim=latent_dim, final_dim=self.encoder.final_dim)
        self.conditional_decoder = ConditionalDecoder(latent_dim=latent_dim, final_dim=self.encoder.final_dim, cond_dim=cond_dim)
        self.cond_encoder = SinusoidalPosEmb(cond_dim)
        self.device = device

    def forward(self, x, q=None):
        x = x.to(self.device)
        z = self.encoder(x)
        if self.cond_dim > 0:
            cond = self.cond_encoder(q)
            sdf = self.conditional_decoder(z, cond)
            # out = self.decoder(z)
            return 0, sdf
        else:
            out = self.decoder(z)
            return out
    
    def forward_without_sampling(self, x, q=None):
        # Same as forward
        x = x.to(self.device)
        z = self.encoder.forward_without_sampling(x)
        if self.cond_dim > 0:
            cond = self.cond_encoder(q)
            sdf = self.conditional_decoder(z, cond)
            # out = self.decoder(z)
            return 0, sdf
        else:
            out = self.decoder(z)
            return out
    



class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim, cond_dim=64, beta=1e-5, device="cuda"):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.encoder = VariationalEncoder(latent_dim=latent_dim, beta=beta, device=device)
        # self.decoder = Decoder(latent_dim=latent_dim, final_dim=self.encoder.final_dim)
        self.conditional_decoder = ConditionalDecoder(latent_dim=latent_dim, final_dim=self.encoder.final_dim, cond_dim=cond_dim)
        self.cond_encoder = SinusoidalPosEmb(cond_dim)
        self.device = device

    def forward(self, x, q=None):
        x = x.to(self.device)
        z = self.encoder(x)
        if self.cond_dim > 0:
            cond = self.cond_encoder(q)
            sdf = self.conditional_decoder(z, cond)
            # out = self.decoder(z)
            return 0, sdf
        else:
            out = self.decoder(z)
            return out
    
    def forward_without_sampling(self, x, q=None):
        x = x.to(self.device)
        z = self.encoder.forward_without_sampling(x)
        if self.cond_dim > 0:
            cond = self.cond_encoder(q)
            sdf = self.conditional_decoder(z, cond)
            # out = self.decoder(z)
            return 0, sdf
        else:
            out = self.decoder(z)
            return out

if __name__ == "__main__":
    model = VariationalAutoencoder(latent_dim=16).to("cuda")
    x = torch.randn((1, 88, 1)).to("cuda")