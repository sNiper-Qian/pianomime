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
import vae.network
from dataset import normalize_data, read_dataset, unnormalize_data
from utils import get_diffusion_obs, Args, get_env, get_flattend_obs

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

def replace_obs(obs, hl_command):
    obs_replace = obs[:248].reshape(4, -1)
    # print(obs_replace[0, 16:])
    # print(hl_command.reshape(4, -1)[0, 16:])
    obs_replace[:, 16:] = hl_command.reshape(4, -1)[:, 16:]
    # obs_replace[:, 26:] = hl_command.reshape(4, -1)[:, 26:]
    obs[:248] = obs_replace.flatten()
    # Add noise to the first 248 elements
    # obs[64:248] += np.random.normal(0, 0.0001, 184)
    return obs

def main(args: Args) -> None:
    pred_horizon = 4
    action_horizon = 4
    obs_horizon = 1

    obs_dim = 450
    action_dim = 47
    task_names = os.listdir('trained_songs')
    task_name = "C"
    n_songs = 1

    hl_model_num_songs = None
    ll_model_num_songs = None
    use_gt = False

    precisions = []
    recalls = []
    f1s = []

    num_seeds = 5 if ll_model_num_songs is not None else 1              

    for seed in range(num_seeds):
        # dataset_path = "dataset_h3_low_level_cont.zarr"
        # dataset_path = "dataset_h3_low_level_cont_v2.zarr"
        # dataset_path = "dataset_h3_low_level_cont_tendon.zarr"
        if ll_model_num_songs is None:
            dataset_path = "dataset_h3_low_level_cont_tendon_res_v2_without_fingering.zarr"
        else:
            dataset_path = "dataset_h3_low_level_cont_tendon_res_num_{}_seed_{}.zarr".format(ll_model_num_songs, seed)
        dataloader, stats = read_dataset(pred_horizon=pred_horizon,
                                obs_horizon=obs_horizon,
                                action_horizon=action_horizon,
                                dataset_path=dataset_path,
                                normalization=True)
        # ckpt_path = "diffusion/ckpts/checkpoint_dataset_h3.ckpt"

        device = torch.device('cuda')

        # SDF
        ae = vae.network.Autoencoder(
            latent_dim=16,
            cond_dim=64,
            ).to('cuda')

        ckpt_path = "vae/ckpts/checkpoint_AE-dataset_h0_midi_large-1707164820.8369377.ckpt"
        state_dict = torch.load(ckpt_path, map_location='cuda')
        ae.load_state_dict(state_dict)
        encoder = ae.encoder

        # # create diffusion network object
        # midi_encoder = ConvEncoder(
        #             in_channels=62,
        #             mid_channels=128,
        #             out_channels=256,
        #             horizon=4,
        #             noise_fingering=0.04,
        #             noise_ft=0.08,
        #         ).to(device)
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

        # ckpt_path = "diffusion/ckpts/checkpoint_DF-LL-dataset_h3_low_level_cont_v2-1709496683.714291.ckpt"
        # ckpt_path = "diffusion/ckpts/checkpoint_DF-LL-dataset_h3_low_level_cont_tendon_res-1710935698.0606441.ckpt"
        if ll_model_num_songs is None:
            ckpt_path = "diffusion/ckpts/checkpoint_DF-LL-dataset_h3_low_level_cont_tendon_res_v2_without_fingering_900.ckpt"
        else:
            ckpt_path = "diffusion/ckpts/checkpoint_DF-LL-dataset_h3_low_level_cont_tendon_res_num_{}_seed_{}.ckpt".format(ll_model_num_songs, seed)
        # ckpt_path = "diffusion/ckpts/checkpoint_DF-LL-dataset_h3_low_level_cont-1707239779.8681304.ckpt"
        # ckpt_path = "diffusion/ckpts/checkpoint_DF-LL-dataset_h3_low_level_cont_tendon_num_20_seed_0-1710784171.425198.ckpt"
        state_dict = torch.load(ckpt_path, map_location='cuda')
        ema_noise_pred_net = noise_pred_net
        ema_noise_pred_net.load_state_dict(state_dict)

        if args.name:
            run_name = args.name
        else:
            run_name = f"DF-{task_name}-{args.seed}-{time.time()}-eval"

        # Create experiment directory.
        experiment_dir = Path(args.root_dir) / run_name
        experiment_dir.mkdir(parents=True)

        for i in range(1):
            task_name = "{}_{}".format(task_name, i+1)
            # traj = np.load('diffusion/trajectories/{}_trajectory_num_{}.npy'.format(task_name, hl_model_num_songs))
            # task_name = task_names[i]
            left_hand_action_list = np.load('handtracking/trajectory/{}_left_hand_action_list.npy'.format(task_name))
            fingering = np.load('diffusion/trajectories/{}_fingerings.npy'.format(task_name))
            max_steps = left_hand_action_list.shape[0] 
            env = get_env(task_name=task_name, enable_ik=True, lookahead = 10, 
                          record_dir=experiment_dir / "eval", use_fingering_emb=True, 
                          use_gt_hand_actions=use_gt, external_fingering=fingering,
                          num_songs=hl_model_num_songs, seed=seed)
            # get first observation
            timestep = env.reset()
            # obs = get_diffusion_obs(timestep)
            lh_current, rh_current = env.get_fingertip_pos()

            step_idx = 0
            current_fingertip = np.concatenate((lh_current, rh_current), axis=0).flatten()
            obs = get_flattend_obs(timestep, 
                                    lookahead=3, 
                                    exclude_keys=['fingering'], 
                                    encoder=encoder, sampling=False,
                                    concatenate_keys=['goal', 'demo']
                                    )
            
            # hl_command = traj[step_idx].flatten()
            # print(obs[0:248].reshape(4, -1)[1])
            # obs = replace_obs(obs, hl_command)
            # print(obs[0:248].reshape(4, -1)[1])

            # keep a queue of last 2 steps of observations
            # obs[64:248] += np.random.normal(0, 0.0001, 184)
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
                    # stack the last obs_horizon (2) number of observations
                    nobs = np.stack(obs_deque)
                    # raise Exception
                    # normalize observation
                    nobs = normalize_data(nobs, stats['obs'])
                    # device transfer
                    nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

                    # infer action
                    # t_start = time.time()
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
                    # t_end = time.time()
                    # unnormalize action
                    naction = naction.detach().to('cpu').numpy()
                    # naction = naction.reshape(1, 1, 64)
                    # Discard the last 10 dimensions
                    # naction = naction[:,:,:-17]
                    # (B, pred_horizon, action_dim)

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
                        # print("Time taken for one step: ", t_end-t_start)
                        # timestep = env.step(np.append(action[i], 0))
                        # timestep = env.step(np.zeros(47))
                        timestep = env.step(action[i])
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
                                exclude_keys=['fingering'], 
                                encoder=encoder, sampling=False,
                                concatenate_keys=['goal', 'demo']
                                )
                        obs_deque.append(obs)
                        # and reward/vis
                        # update progress bar
                        pbar.update(1)
                        # pbar.set_postfix(reward=reward)
            print(task_name)
            print(seed)
            metric = env.get_musical_metrics()
            precision = metric['precision']
            recall = metric['recall']
            f1 = metric['f1']
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            print(metric)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)
    precision_list = np.array(precisions).reshape(-1, 6).T
    recall_list = np.array(recalls).reshape(-1, 6).T
    f1_list = np.array(f1s).reshape(-1, 6).T
    # print("Precision list: ", precision_list)
    std_precision = np.std(precision_list, axis=1)
    std_recall = np.std(recall_list, axis=1)
    std_f1 = np.std(f1_list, axis=1)

    print("Average precision: {}".format(avg_precision))
    print("Average recall: {}".format(avg_recall))
    print("Average f1: {}".format(avg_f1))
    print("Standard deviation precision: {}".format(std_precision))
    print("Standard deviation recall: {}".format(std_recall))
    print("Standard deviation f1: {}".format(std_f1))
    print("Average sde precision: {}".format(np.mean(std_precision)))
    print("Average sde recall: {}".format(np.mean(std_recall)))
    print("Average sde f1: {}".format(np.mean(std_f1)))
    print("ll_model_num_songs: ", ll_model_num_songs)
    print("hl_model_num_songs: ", hl_model_num_songs)
    print("use_gt: ", use_gt)

if __name__ == '__main__':
    main(tyro.cli(Args, description=__doc__))