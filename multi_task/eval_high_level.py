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
from dataset import RoboPianistDataset, read_dataset, normalize_data, unnormalize_data, read_dataset_split
import sys
import time
import wandb
import vae.network
from handtracking.utils import draw_point_on_image, preprocess_frame
import cv2
import network
from utils import get_env_hl, get_diffusion_obs, get_flattend_obs, get_goal_only_obs, adjust_ft_fingering
import os

CTRL_TIMESTEP = 0.05

if __name__ == '__main__':
    pred_horizon = 4
    action_horizon = 1
    obs_horizon = 1

    obs_dim = 212
    action_dim = 36

    midi_channel = 16
    midi_dim = 212

    noise = 0.01
    write_video = False # Plot the fingertip trajectory on the video
    num_seeds = 5


    for seed in range(num_seeds):
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        dataset_path = "/root/pianomime/pianomime/dataset_hl.zarr"

        device = torch.device('cuda')

        # create dataloader
        dataloader, stats = read_dataset(pred_horizon=pred_horizon,
                                obs_horizon=obs_horizon,
                                action_horizon=action_horizon,
                                dataset_path=dataset_path,
                                normalization=True)

        ae = vae.network.Autoencoder(
            latent_dim=16,
            cond_dim=64,
        ).to('cuda')

        ckpt_path = "vae/ckpts/checkpoint_AE-dataset_h0_midi_large-1707164820.8369377.ckpt"
        state_dict = torch.load(ckpt_path, map_location='cuda')
        ae.load_state_dict(state_dict)
        encoder = ae.encoder

        def create_midi_encoder(device='cuda'):
            midi_encoder = VariationalConvMlpEncoder(
                in_channels=16,
                mid_channels=32,
                out_channels=64,
                latent_dim=32,
                noise=0.08,
            ).to(device)
            return midi_encoder

        # create network object
        noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim*obs_horizon,
            midi_dim=obs_dim,
            midi_cond_dim=36,
            midi_encoder=create_midi_encoder,
        ).to(device)

        ckpt_path = "diffusion/ckpts/checkpoint_DF-HL-dataset_h10_h3_high_level_plan_fingering_v3_without_fingering.ckpt"
        state_dict = torch.load(ckpt_path, map_location='cuda')
        ema_noise_pred_net = noise_pred_net
        ema_noise_pred_net.load_state_dict(state_dict)
        num_epochs = 3200

        # Exponential Moving Average
        # accelerates training and improves stability
        # holds a copy of the model weights
        ema = EMAModel(
            model=noise_pred_net,
            power=0.75)

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
        # task_names = os.listdir('trained_songs')
        num_songs = 1
        losses = []
        for i in range(num_songs):
            task_name = "NoTimeToDie_{}".format(i+1)
            print(task_name)
            # left_hand_action_list = np.load('handtracking/trajectory/{}_left_hand_action_list.npy'.format(task_name))
            # max_steps = left_hand_action_list.shape[0] 

            # Load homography matrix
            H = np.load('handtracking/H_matrices/PianoX.npy')

            env, max_steps = get_env_hl(task_name, lookahead=10)
            trajectory_lh = np.zeros((max_steps, 3, 6))
            trajectory_rh = np.zeros((max_steps, 3, 6))
            trajectory = []
            fingerings = []
            timestep = env.reset()
            # Record last fingertip position
            lh_current, rh_current = env.task.get_fingertip_pos(env.physics)
            last_fingertip_pos = np.concatenate((lh_current, rh_current), axis=0).flatten()     

            last_timestamp = float('-inf') # Let the first frame be processed -inf
            timestamp = 0
            first_timestamp = None
            step = 0
            B = 1
            loss_ft = 0
            loss_fingering = 0
            last_lh_ft = None
            last_rh_ft = None
            last_keys = None
            last_fingering = None
            with tqdm(total=max_steps, desc="Eval Env") as pbar:
                while not timestep.last():
                    cont = np.zeros((4, midi_channel+action_dim))
                    goal = get_flattend_obs(timestep, 
                                    lookahead=10,
                                    exclude_keys=[
                                                'fingering', 
                                                'hand', 
                                                'fingering', 
                                                'demo', 
                                                'prior_action', 
                                                'q_piano',
                                                ], 
                                    encoder=encoder, 
                                    sampling=False)
                    cont[:, :midi_channel] = goal[:4*midi_channel].reshape((4, -1))
                    goal = torch.from_numpy(goal)
                    current = last_fingertip_pos
                    cond = torch.from_numpy(current)
                    obs = torch.cat((goal, cond), dim=-1).float()
                    obs = normalize_data(obs, stats['obs']).to(device)
                    with torch.no_grad():
                        obs = obs.unsqueeze(0)
                        # initialize action from Guassian noise
                        noisy_action = torch.randn(
                            (B, pred_horizon, action_dim), device=device)
                        naction = noisy_action

                        # init scheduler
                        noise_scheduler.set_timesteps(num_diffusion_iters)

                        for k in noise_scheduler.timesteps:
                            # predict noise
                            noise_pred = ema_noise_pred_net(
                                sample=naction,
                                timestep=k,
                                global_cond=obs
                            )

                            # inverse diffusion step (remove noise)
                            naction = noise_scheduler.step(
                                model_output=noise_pred,
                                timestep=k,
                                sample=naction
                            ).prev_sample
                    # naction = naction.detach().to('cpu').numpy().flatten()

                    naction = naction.detach().to('cpu').numpy()
                    # Append 10 dimensions for fingering
                    naction = np.concatenate((naction, np.zeros((1, 4, 10))), axis=2).flatten()
                    # print(naction.shape)
                    naction = unnormalize_data(naction, stats['action'])
                    naction = naction.reshape(B, 4, -1)
                    action = naction[0][0]
                    # Get fingertip position from the action
                    nft = naction[0, :, :36]
                    ft = action[:36]
                    goal = timestep.observation['goal'][:88]
                    keys = np.nonzero(goal)

                    lh_ft, rh_ft, fingering = adjust_ft_fingering(env, keys, 
                                                                nft[0][:18].reshape(6, 3).T,
                                                                nft[0][18:].reshape(6, 3).T,
                                                                last_keys, last_lh_ft, last_rh_ft, 
                                                                last_fingering)
                    last_lh_ft = lh_ft
                    last_rh_ft = rh_ft
                    last_keys = keys
                    last_fingering = fingering
                    ft = np.concatenate((lh_ft.T.flatten(), rh_ft.T.flatten()))
                    trajectory_lh[step] = lh_ft
                    trajectory_rh[step] = rh_ft
                    last_fingertip_pos = ft
                    step += 1
                    timestep = env.step(np.zeros(47))
                    # trajectory.append(cont)
                    pbar.update(1)
            # trajectory = np.array(trajectory)

            # Save the trajectory
            np.save("diffusion/trajectories/{}_trajectory.npy".format(task_name), trajectory)
            np.save("diffusion/trajectories/{}_left_hand_action_list.npy".format(task_name), trajectory_lh)
            np.save("diffusion/trajectories/{}_right_hand_action_list.npy".format(task_name), trajectory_rh)

            # Release the VideoCapture and VideoWriter objects
    print("Average loss: ", np.mean(losses))