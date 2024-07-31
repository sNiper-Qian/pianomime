from network import ConditionalUnet1D, EMAModel, ConvEncoder, VariationalConvMlpEncoder, MLP
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
from torchviz import make_dot
import time
import wandb
import vae.network
import sys

if __name__ == '__main__':
    pred_horizon = 1
    action_horizon = 1
    obs_horizon = 1

    obs_dim = 260
    action_dim = 64

    dataset_path = sys.argv[1]
    device = torch.device('cuda')

    # create dataloader
    dataloader, stats = read_dataset(pred_horizon=pred_horizon,
                              obs_horizon=obs_horizon,
                              action_horizon=action_horizon,
                              dataset_path=dataset_path,
                              normalization=True)
  
    mlp = MLP(
        in_dim = obs_dim,
        out_dim = action_dim,
    ).to(device)

    num_epochs = 4800

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        model=mlp,
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=mlp.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader) * num_epochs
    )

    wandb.login()
    # run_name = f"DF-HL-{dataset_path.split('.')[0]}-{time.time()}"
    run_name = f"DF-LL-{dataset_path.split('.')[0]}"
    wandb.init(
        project="robopianist",
        name=run_name,
        config={},
        sync_tensorboard=True,
    )

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
                    # print(naction.shape)
                    # Pad naction to 64

                    B = nobs.shape[0]

                    # (B, obs_horizon, obs_dim)
                    obs_cond = nobs[:,:obs_horizon,:]

                    # (B, obs_horizon * obs_dim)
                    obs_cond = obs_cond.flatten(start_dim=1)

                    action_pred = mlp(obs_cond)
                    
                    # L2 loss
                    naction = naction.squeeze()
                    # print(action_pred.shape, naction.shape)
                    l = naction-action_pred

                    loss = nn.functional.mse_loss(action_pred, naction) 

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(mlp)

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))
            wandb.log({"loss": np.mean(epoch_loss)})
            wandb.log({"learning rate": lr_scheduler.get_last_lr()[0]})
            wandb.log({"epoch": epoch_idx})
            if epoch_idx % 400 == 0:
                # Weights of the EMA model
                # is used for inference
                ema_noise_pred_net = ema.averaged_model

                # Assuming ema_model is your EMA model
                ema_model_state_dict = ema_noise_pred_net.state_dict()

                # Specify the path to save the EMA model's weights
                ema_model_weights_path = 'diffusion/ckpts/checkpoint_mlp_{}.ckpt'.format(run_name)

                # Save the EMA model's weights to the specified path
                torch.save(ema_model_state_dict, ema_model_weights_path)
                print("Saved checkpoint at epoch {}".format(epoch_idx))

    # Weights of the EMA model
    # is used for inference
    ema_noise_pred_net = ema.averaged_model

    # Assuming ema_model is your EMA model
    ema_model_state_dict = ema_noise_pred_net.state_dict()

    # Specify the path to save the EMA model's weights
    ema_model_weights_path = 'diffusion/ckpts/checkpoint_mlp_{}.ckpt'.format(run_name)

    # Save the EMA model's weights to the specified path
    torch.save(ema_model_state_dict, ema_model_weights_path)

    print("Done!")
    # Kill