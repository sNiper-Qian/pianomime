import sys
directory = 'pianomime'
if directory not in sys.path:
    sys.path.append(directory)
from network import ConditionalUnet1D, EMAModel, ConvEncoder, VariationalConvMlpEncoder
import torch
import math
from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import torch
import torch.nn as nn
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from dataset import RoboPianistDataset, read_dataset
import sys
import time
import wandb
import sys

if __name__ == '__main__':
    pred_horizon = 1
    action_horizon = 1
    obs_horizon = 1

    obs_dim = 212
    action_dim = 36
    # seed = sys.argv[1]
    num_songs = 50
    dataset_path = sys.argv[1]
    device = torch.device('cuda')

    # create dataloader
    dataloader, stats = read_dataset(pred_horizon=pred_horizon,
                              obs_horizon=obs_horizon,
                              action_horizon=action_horizon,
                              dataset_path=dataset_path,
                              normalization=True)
  
    def create_midi_encoder(device='cuda'):
        # TCN for midi encoding
        midi_encoder = VariationalConvMlpEncoder(
            in_channels=16,
            mid_channels=32,
            out_channels=64,
            latent_dim=32,
            noise=0.08,
        ).to(device)
        return midi_encoder
    
    # Conditional UNet for noise prediction
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon,
        midi_dim=obs_dim,
        midi_encoder=create_midi_encoder,
        midi_cond_dim=36,
        freeze_encoder=False,
    ).to(device)

    num_epochs = 1200

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        model=noise_pred_net,
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader) * num_epochs
    )

    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )
    # wandb.login()
    run_name = f"DF-HL-{dataset_path.split('.')[0]}"
    # wandb.init(
    #     project="robopianist",
    #     name=run_name,
    #     config={},
    #     sync_tensorboard=True,
    # )

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nobs = nbatch['obs'].to(device)
                    naction = nbatch['action'].to(device)
                    
                    naction = naction.reshape(naction.shape[0], 4, -1)

                    # Exclude fingering, the last 10 elements is fingering, the first 36 is fingertip position
                    naction = naction[:, :, :36]
                    B = nobs.shape[0]

                    # observation as FiLM conditioning
                    # (B, obs_horizon, obs_dim)
                    obs_cond = nobs[:,:obs_horizon,:]

                    # (B, obs_horizon * obs_dim)
                    obs_cond = obs_cond.flatten(start_dim=1)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    l = noise_pred-noise

                    loss = nn.functional.mse_loss(noise_pred, noise) + noise_pred_net.kl

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(noise_pred_net)

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))
            # wandb.log({"loss": np.mean(epoch_loss)})
            # wandb.log({"learning rate": lr_scheduler.get_last_lr()[0]})
            # wandb.log({"epoch": epoch_idx})
            if epoch_idx % 400 == 0:
                # Weights of the EMA model
                # is used for inference
                ema_noise_pred_net = ema.averaged_model

                # Assuming ema_model is your EMA model
                ema_model_state_dict = ema_noise_pred_net.state_dict()

                # Specify the path to save the EMA model's weights
                ema_model_weights_path = 'diffusion/ckpts/checkpoint_{}_without_fingering.ckpt'.format(run_name)

                # Save the EMA model's weights to the specified path
                torch.save(ema_model_state_dict, ema_model_weights_path)
                print("Saved checkpoint at epoch {}".format(epoch_idx))


    # Weights of the EMA model
    # is used for inference
    ema_noise_pred_net = ema.averaged_model

    # Assuming ema_model is your EMA model
    ema_model_state_dict = ema_noise_pred_net.state_dict()

    # Specify the path to save the EMA model's weights
    ema_model_weights_path = 'diffusion/ckpts/checkpoint_{}_without_fingering.ckpt'.format(run_name)

    # Save the EMA model's weights to the specified path
    torch.save(ema_model_state_dict, ema_model_weights_path)

    print("Done!")