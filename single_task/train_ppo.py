import sys
directory = 'pianomime'
if directory not in sys.path:
    sys.path.append(directory)
from pathlib import Path
from typing import Optional, Tuple
import tyro
from dataclasses import dataclass, asdict
import wandb
import time
import random
import numpy as np
from tqdm import tqdm
import torch
from copy import copy
from dataclasses import dataclass, replace

import logging_callback
import lr_scheduler

import os
from mujoco_utils import composer_utils
import gymnasium as gym
from utils import get_env, make_envs

from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

import pickle
import shutil


@dataclass(frozen=True)
class Args:
    root_dir: str = "/tmp/robopianist"
    seed: int = 42
    max_steps: int = 1_000_000
    warmstart_steps: int = 5_000
    log_interval: int = 1_000
    eval_interval: int = 10_000
    eval_episodes: int = 1
    batch_size: int = 256
    discount: float = 0.99
    tqdm_bar: bool = False
    replay_capacity: int = 1_000_000
    project: str = "robopianist"
    entity: str = ""
    name: str = ""
    tags: str = ""
    notes: str = ""
    mode: str = "disabled"
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"
    n_steps_lookahead: int = 10
    trim_silence: bool = False
    gravity_compensation: bool = False
    reduced_action_space: bool = False
    control_timestep: float = 0.05
    stretch_factor: float = 1.0
    shift_factor: int = 0
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = False
    disable_forearm_reward: bool = False
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    primitive_fingertip_collisions: bool = False
    frame_stack: int = 1
    clip: bool = True
    record_dir: Optional[Path] = None
    record_every: int = 1
    record_resolution: Tuple[int, int] = (480, 640)
    camera_id: Optional[str | int] = "piano/back"
    action_reward_observation: bool = False
    deepmimic: bool = False
    mimic_task: str = "TwinkleTwinkleRousseau"
    midi_start_from: int = 0    
    residual_action: bool = False
    num_envs: int = 16
    pretrained: Optional[Path] = None
    initial_lr: float = 3e-4
    lr_decay_rate: float = 0.99
    residual_factor: float = 0.02
    n_steps: int = 512
    use_note_trajectory: bool = False
    mimic_z_axis: bool = False
    disable_hand_collisions: bool = True
    rsi: bool = False
    curriculum: bool = False
    total_iters: int = 1000

def prefix_dict(prefix: str, d: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in d.items()}

def main(args: Args) -> None:
    if args.name:
        run_name = args.name
    else:
        run_name = f"PPO-{args.environment_name}-{args.seed}-{time.time()}"

    # Create experiment directory.
    experiment_dir = Path(args.root_dir) / run_name
    experiment_dir.mkdir(parents=True)

    # Seed RNGs.
    random.seed(args.seed)
    np.random.seed(args.seed)

    # wandb.login()

    # wandb.init(
    #     project=args.project,
    #     config=asdict(args),
    #     name=run_name,
    #     sync_tensorboard=True,
    # )
    eval_args = copy(args)
    eval_args = replace(eval_args, rsi=False)
    eval_env = get_env(eval_args, record_dir=experiment_dir / "eval")
    def make_env():
        env = get_env(args)
        return Monitor(env)
    # Parallel environments
    vec_env = SubprocVecEnv([make_envs(make_env, i) for i in range(args.num_envs)], start_method="fork")

    lr_scheduler_instance = lr_scheduler.LR_Scheduler(initial_lr=args.initial_lr,
                                                      decay_rate=args.lr_decay_rate,)

    policy_kwargs = dict(activation_fn=torch.nn.GELU,
                     net_arch=dict(pi=[1024, 256], vf=[1024, 256]))
    model = PPO("MlpPolicy", 
                vec_env, 
                n_epochs=10,
                n_steps=args.n_steps,
                batch_size=1024,
                learning_rate=lr_scheduler_instance.lr_schedule,
                policy_kwargs=policy_kwargs, 
                verbose=2,
                tensorboard_log="./robopianist_rl/tensorboard/{}".format(run_name),
                )
    if args.pretrained is not None:
        # Reload learning rate scheduler
        custom_objects = { 'learning_rate': lr_scheduler_instance.lr_schedule}
        model = PPO.load(args.pretrained, env=vec_env, custom_objects=custom_objects)
    best_f1 = -np.inf
    # last_extending_curriculum_step = 0
    try:
        for i in range(args.total_iters):
            # Training
            model.learn(total_timesteps=args.n_steps*args.num_envs, 
                        progress_bar=True,
                        reset_num_timesteps=False,
                        callback= None)
            # Evaluation
            obs, _ = eval_env.reset()
            while True:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = eval_env.step(action)
                if done == True:
                    break
            log_dict = prefix_dict("eval", eval_env.env.get_statistics())
            music_dict = prefix_dict("eval", eval_env.env.get_musical_metrics())
            # wandb.log(log_dict | music_dict, step=i)
            # if args.deepmimic:
                # wandb.log(prefix_dict("eval", eval_env.env.get_deepmimic_rews()), step=i)
            f1 = eval_env.env.get_musical_metrics()["f1"]
            if f1 > best_f1:
                print("best_f1:{}->{}".format(best_f1, eval_env.env.get_musical_metrics()["f1"]))
                best_f1 = eval_env.env.get_musical_metrics()["f1"]
                model.save("./robopianist_rl/ckpts/{}_best".format(run_name))
                # video = wandb.Video(str(eval_env.env.latest_filename), fps=4, format="mp4")
                # wandb.log({"video": video, "global_step": i})
            
            eval_env.env.latest_filename.unlink()  
    except KeyboardInterrupt:
        pass

    # model.save("./robopianist_rl/ckpts/{}".format(run_name))

    # Evaluate the trained model
    model = PPO.load("./robopianist_rl/ckpts/{}_best".format(run_name), env=vec_env)
    os.makedirs("./trained_songs", exist_ok=True)
    os.makedirs("./trained_songs/{}".format(args.mimic_task), exist_ok=True)
    left_hand_action_list = np.load(
    f"dataset/high_level_trajectories/{args.mimic_task}_left_hand_action_list.npy"
    )
    right_hand_action_list = np.load(
    f"dataset/high_level_trajectories/{args.mimic_task}_right_hand_action_list.npy"
    )
    np.save("./trained_songs/{}/left_hand_action_list".format(args.mimic_task), left_hand_action_list)
    np.save("./trained_songs/{}/right_hand_action_list".format(args.mimic_task), right_hand_action_list)
    # shutil.copy("./handtracking/midis/{}.mid".format(args.mimic_task), "./trained_songs/{}".format(args.mimic_task))
    shutil.copy("./handtracking/notes/{}.pkl".format(args.mimic_task), "./trained_songs/{}".format(args.mimic_task))
    # shutil.copy(args.pretrained, "./trained_songs/{}".format(args.mimic_task))

    obs, _ = eval_env.reset()
    actions = []
    rewards = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, reward, done, _, info = eval_env.step(action)
        rewards += reward
        if done:
            break
    print(f"Total reward: {rewards}")
    print(eval_env.env.latest_filename)
    print(eval_env.env.get_musical_metrics())
    # play_video(eval_env.env.latest_filename)
    actions = np.array(actions)
    # np.save("./handtracking/trained/actions_{}".format(args.mimic_task), actions)
    np.save("./trained_songs/{}/actions_{}".format(args.mimic_task, args.mimic_task), actions)

    del model # remove to demonstrate saving and loading

if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))