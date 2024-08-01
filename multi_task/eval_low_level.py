import sys
directory = 'pianomime'
if directory not in sys.path:
    sys.path.append(directory)
from robopianist import suite
import dm_env_wrappers as wrappers
import robopianist.wrappers as robopianist_wrappers

from pathlib import Path
from typing import Optional, Tuple
import tyro
from dataclasses import dataclass, asdict
import time
import collections

from tqdm import tqdm
import numpy as np
import torch
import os

from IPython.display import HTML
from base64 import b64encode

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from network import ConditionalUnet1D, VariationalConvMlpEncoder, ConvEncoder
from dataset import normalize_data, read_dataset, unnormalize_data
from utils import get_diffusion_obs, Args, get_env_ll, get_flattend_obs
import goal_auto_encoder.network

def play_video(filename: str):
    mp4 = open(filename, "rb").read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

    return HTML(
        """
  <video controls>
        <source src="%s" type="video/mp4">
  </video>
  """
        % data_url
    )

def main() -> None:
    pred_horizon = 4
    action_horizon = 4
    obs_horizon = 1

    obs_dim = 404
    action_dim = 46
    task_name = sys.argv[1]
    n_songs = 1

    hl_model_num_songs = None
    ll_model_num_songs = None
    use_gt = False

    precisions = []
    recalls = []
    f1s = []

    dataset_path = "pianomime/dataset_ll.zarr"
    dataloader, stats = read_dataset(pred_horizon=pred_horizon,
                            obs_horizon=obs_horizon,
                            action_horizon=action_horizon,
                            dataset_path=dataset_path,
                            normalization=True)
    # ckpt_path = "diffusion/ckpts/checkpoint_dataset_h3.ckpt"

    device = torch.device('cuda')

    # SDF
    ae = goal_auto_encoder.network.Autoencoder(
        latent_dim=16,
        cond_dim=64,
        ).to('cuda')

    ckpt_path = "checkpoint_ae.ckpt"
    state_dict = torch.load(ckpt_path, map_location='cuda')
    ae.load_state_dict(state_dict)
    encoder = ae.encoder

    # # create diffusion network object
    def create_midi_encoder(device='cuda'):
        midi_encoder = ConvEncoder(
                        in_channels=52,
                        mid_channels=64,
                        out_channels=128,
                        horizon=4,
                        noise_fingering=0,
                        noise_ft=0,
                    ).to(device)
        return midi_encoder
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon,
        midi_dim=208,
        midi_cond_dim=0,
        midi_encoder=create_midi_encoder,
        freeze_encoder=False,
    ).to(device)

    ckpt_path = "checkpoint_low_level.ckpt"

    state_dict = torch.load(ckpt_path, map_location='cuda')
    ema_noise_pred_net = noise_pred_net
    ema_noise_pred_net.load_state_dict(state_dict)

    for i in range(1):
        left_hand_action_list = np.load('pianomime/multi_task/trajectories/{}_left_hand_action_list.npy'.format(task_name))
        max_steps = left_hand_action_list.shape[0] 
        env = get_env_ll(task_name=task_name, enable_ik=False, lookahead = 10, 
                        record_dir=".", use_fingering_emb=False, 
                        use_midi=False)
        
        timestep = env.reset()
        lh_current, rh_current = env.get_fingertip_pos()

        step_idx = 0
        current_fingertip = np.concatenate((lh_current, rh_current), axis=0).flatten()
        obs = get_flattend_obs(timestep, 
                                lookahead=3, 
                                exclude_keys=['fingering', 'prior_action'], 
                                encoder=encoder, sampling=False,
                                concatenate_keys=['goal', 'demo']
                                )
        
        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)

        num_diffusion_iters = 50
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

        with tqdm(total=max_steps, desc="Eval Env") as pbar:
            while not timestep.last():
                B = 1
                # stack the last obs_horizon number of observations
                nobs = np.stack(obs_deque)
                # normalize observation
                nobs = normalize_data(nobs, stats['obs'])
                # device transfer
                nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

                # infer action
                with torch.no_grad():
                    # reshape observation to (B,obs_horizon*obs_dim)
                    obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

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
                            global_cond=obs_cond
                        )

                        # inverse diffusion step (remove noise)
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample

                # unnormalize action
                naction = naction.detach().to('cpu').numpy()

                naction = naction[0]
                action_pred = naction # Unnormalize

                # only take action_horizon number of actions
                start = obs_horizon - 1
                end = start + action_horizon
                action = action_pred[start:end,:]
                # (action_horizon, action_dim)

                # execute action_horizon number of steps
                # without replanning
                for i in range(len(action)):
                    # stepping env
                    action[i] = unnormalize_data(action[i], stats=stats["action"])
                    demo = timestep.observation['demo']
                    demo_lh, demo_rh = np.split(demo, 2)
                    demo_lh = demo_lh[:18]
                    demo_rh = demo_rh[:18]
                    demo = np.concatenate((demo_lh, demo_rh), axis=0).flatten()

                    timestep = env.step(np.append(action[i], 0))
                    if timestep.last():
                        break
                    # save observations
                    lh_current, rh_current = env.get_fingertip_pos()
                    current_fingertip = np.concatenate((lh_current, rh_current), axis=0).flatten()
                    step_idx += 1
                    if step_idx < left_hand_action_list.shape[0]:
                        # hl_command = traj[step_idx].flatten()
                        obs = get_flattend_obs(timestep, 
                            lookahead=3, 
                            exclude_keys=['fingering', 'prior_action'], 
                            encoder=encoder, sampling=False,
                            concatenate_keys=['goal', 'demo']
                            )
                    obs_deque.append(obs)
                    # update progress bar
                    pbar.update(1)
        print(task_name)
        metric = env.get_musical_metrics()
        precision = metric['precision']
        recall = metric['recall']
        f1 = metric['f1']
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))

if __name__ == '__main__':
    main()