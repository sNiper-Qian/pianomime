import sys
directory = 'pianomime'
if directory not in sys.path:
    sys.path.append(directory)
from diffusion_policy.policy.bet_lowdim_policy import BETLowdimPolicy
from diffusion_policy.model.bet.action_ae.discretizers.k_means import KMeansDiscretizer
from diffusion_policy.model.bet.latent_generators.mingpt import MinGPT
from diffusion_policy.model.bet.utils import eval_mode
from network import ConditionalUnet1D, EMAModel, ConvEncoder, BetVariationalConvMlpEncoder
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
    pred_horizon = 4
    action_horizon = 1
    obs_horizon = 4

    obs_dim = 450
    action_dim = 46
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
    
    action_ae = KMeansDiscretizer(action_dim=action_dim,
                                  predict_offsets=True,
                                  num_bins=64).to(device)

    obs_encoding_net = torch.nn.Identity().to(device)
    state_prior = MinGPT(input_dim=obs_dim,
                         vocab_size=64,
                         n_layer=8,
                         n_head=8,
                         n_embd=120,
                         predict_offsets=True,
                         action_dim=action_dim).to(device)
    bet_policy = BETLowdimPolicy(action_ae=action_ae,
                                obs_encoding_net=obs_encoding_net,
                                state_prior=state_prior,
                                horizon=obs_horizon,
                                n_action_steps=pred_horizon,
                                n_obs_steps=obs_horizon).to(device)
    # Get all data in dataloader
    action_data = []
    for nbatch in dataloader:
        action_data.append(nbatch['action'].numpy())

    action_data = np.concatenate(action_data, axis=0).squeeze()
    # print(action_data.shape)
    bet_policy.fit_action_ae(torch.from_numpy(action_data))

    num_epochs = 800

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        model=bet_policy,
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=bet_policy.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader) * num_epochs
    )
    run_name = f"BET-HL-{dataset_path.split('.')[0]}"
    wandb.login()
    # run_name = f"DF-HL-{dataset_path.split('.')[0]}-{time.time()}"
    # run_name = f"DF-HL-{dataset_path.split('.')[0]}"
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

                    # L2 loss
                    nbatch = {k: v.to(device) for k, v in nbatch.items()}
                    loss, _ = bet_policy.compute_loss(nbatch)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(bet_policy)

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))
            wandb.log({"loss": np.mean(epoch_loss)})
            wandb.log({"learning rate": lr_scheduler.get_last_lr()[0]})
            wandb.log({"epoch": epoch_idx})
            if epoch_idx % 100 == 0:
                # Weights of the EMA model
                # is used for inference
                ema_noise_pred_net = ema.averaged_model

                # Assuming ema_model is your EMA model
                ema_model_state_dict = ema_noise_pred_net.state_dict()

                # Specify the path to save the EMA model's weights
                ema_model_weights_path = 'diffusion/ckpts/checkpoint_{}_bet.ckpt'.format(run_name)
                

                # Save the EMA model's weights to the specified path
                torch.save(ema_model_state_dict, ema_model_weights_path)
                print("Saved checkpoint at epoch {}".format(epoch_idx))

                # midi_encoder_state_dict = midi_encoder.state_dict()
                # midi_encoder_weights_path = 'diffusion/ckpts/midi_encoder_{}.ckpt'.format(run_name)
                # torch.save(midi_encoder_state_dict, midi_encoder_weights_path)
            

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