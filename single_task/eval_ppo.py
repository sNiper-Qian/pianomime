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
import shutil

import sac
import specs
import replay
import logging_callback
import lr_scheduler

from IPython.display import HTML
from base64 import b64encode

import os
import pickle

import orbax.checkpoint
from flax.training import orbax_utils

from robopianist import suite
import dm_env_wrappers as wrappers
import robopianist.wrappers as robopianist_wrappers
from robopianist.suite.tasks import piano_with_shadow_hands
from robopianist.suite.tasks import piano_with_shadow_hands_res
from robopianist import music
from mujoco_utils import composer_utils
import gymnasium as gym
from utils import get_env

from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

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
    agent_config: sac.SACConfig = sac.SACConfig()
    deepmimic: bool = False
    mimic_task: str = "TwinkleTwinkleRousseau"
    midi_start_from: int = 0    
    residual_action: bool = False
    num_envs: int = 16
    pretrained: Optional[Path] = None
    initial_lr: float = 3e-4
    lr_decay_rate: float = 0.99
    residual_factor: float = 0.02
    save_actions: bool = False
    use_note_trajectory: bool = False
    mimic_z_axis: bool = False
    rsi: bool = False
    curriculum: bool = False

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

    eval_env = get_env(args, record_dir=".")

    def make_env():
        env = get_env(args)
        return Monitor(env)
    # Parallel environments
    vec_env = DummyVecEnv([make_env for _ in range(args.num_envs)])

    lr_scheduler_instance = lr_scheduler.LR_Scheduler(initial_lr=args.initial_lr,
                                                      decay_rate=args.lr_decay_rate,)

    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[1024, 256], vf=[1024, 256]))
    model = PPO("MlpPolicy", 
                vec_env, 
                n_steps=256,
                batch_size=1024,
                learning_rate=lr_scheduler_instance.lr_schedule,
                policy_kwargs=policy_kwargs, 
                verbose=1,
                tensorboard_log="./robopianist_rl/tensorboard/{}".format(run_name),
                )
    if args.pretrained is not None:
        model = PPO.load(args.pretrained, env=vec_env)
    
    if args.save_actions:
        os.makedirs("./trained_songs", exist_ok=True)
        os.makedirs("./trained_songs/{}".format(args.mimic_task), exist_ok=True)
        left_hand_action_list = np.load(
        f"handtracking/trajectory/{args.mimic_task}_left_hand_action_list.npy"
        )
        right_hand_action_list = np.load(
        f"handtracking/trajectory/{args.mimic_task}_right_hand_action_list.npy"
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
    print(eval_env.env.get_deepmimic_rews())
    play_video(eval_env.env.latest_filename)
    actions = np.array(actions)
    if args.save_actions:
        # np.save("./handtracking/trained/actions_{}".format(args.mimic_task), actions)
        np.save("./trained_songs/{}/actions_{}".format(args.mimic_task, args.mimic_task), actions)

if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))