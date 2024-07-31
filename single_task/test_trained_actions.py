from robopianist.controller.ik_controller import move_finger_to_key, move_fingers_to_keys, move_fingers_to_pos_qp
from IPython.display import HTML
from base64 import b64encode
import numpy as np
from robopianist.models.piano import piano_constants
from robopianist.suite.tasks import self_actuated_piano
from robopianist.suite.tasks import piano_with_shadow_hands
from robopianist.suite.tasks import piano_with_one_shadow_hand
from robopianist.suite.tasks import piano_with_shadow_hands_res
from dm_env_wrappers import CanonicalSpecWrapper
from robopianist.wrappers import PianoSoundVideoWrapper
from robopianist.wrappers.pixels import PixelWrapper
from robopianist.wrappers.deep_mimic import DeepMimicWrapper
from robopianist.wrappers.residual import ResidualWrapper
from robopianist.wrappers.fingering_emb import FingeringEmbWrapper
from robopianist.wrappers.dm2gym import Dm2GymWrapper
from dm_env_wrappers import SinglePrecisionWrapper
from dm_env_wrappers import DmControlWrapper
from robopianist.wrappers.evaluation import MidiEvaluationWrapper
from robopianist import music
from mujoco_utils import composer_utils
import dm_env
from robopianist.models.hands import HandSide
import matplotlib.pyplot as plt
from dm_control.mujoco.wrapper import mjbindings
import handtracking.start_from_dict as start_from_dict
import os
from collections import namedtuple
import time
import pickle

mjlib = mjbindings.mjlib

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

# for folders in os.listdir('trained_songs'):
#     if folders.startswith('.'):
#         continue
#     for files in os.listdir('trained_songs/{}'.format(folders)):
#         if files.startswith('.'):
#             continue
#         if files.endswith('.npy') and files.startswith('actions'):
#             actions = np.load('trained_songs/{}/{}'.format(folders, files))
#             task_name = files.split('.')[0].split('_')[1]
# raise Exception
# task_name = "ArabesqueNo1"
task_name = "NeverGonnaGiveYouUp_1"

# start_from = start_from_dict.START_FROM[task_name]
with open('handtracking/notes/{}.pkl'.format(task_name), 'rb') as f:
    note_traj = pickle.load(f)


task = piano_with_shadow_hands_res.PianoWithShadowHandsResidual(
    # hand_side=HandSide.LEFT,
    note_trajectory=note_traj,
    # midi=music.load(task_name),
    change_color_on_activation=True,
    trim_silence=True,
    control_timestep=0.05,
    disable_hand_collisions=True,
    disable_forearm_reward=True,
    disable_fingering_reward=False,
    midi_start_from=0,
    n_steps_lookahead=10,
    gravity_compensation=True,
    residual_factor=0.03, # 0.03 for after NeverGonnaGiveYouUp
    shift=0,
)

# Load hand action trajectory
left_hand_action_list = np.load('handtracking/trajectory/{}_left_hand_action_list.npy'.format(task_name))
right_hand_action_list = np.load('handtracking/trajectory/{}_right_hand_action_list.npy'.format(task_name))

# Load trained actions
# actions = np.load('handtracking/trained/actions_{}.npy'.format(task_name))
actions = np.load('trained_songs/{}/actions_{}.npy'.format(task_name, task_name))

env = composer_utils.Environment(
    recompile_physics=False, task=task, strip_singleton_obs_buffer_dim=True
)

env = PianoSoundVideoWrapper(
    env,
    record_every=1,
    camera_id="piano/top",
    record_dir=".",
)
env = DeepMimicWrapper(env,
                      demonstrations_lh=left_hand_action_list,
                      demonstrations_rh=right_hand_action_list,
                      remove_goal_observation=False,
                      mimic_z_axis=False,)
env = ResidualWrapper(env, 
                      demonstrations_lh=left_hand_action_list,
                      demonstrations_rh=right_hand_action_list,
                      demo_ctrl_timestep=0.05,)
env = MidiEvaluationWrapper(
    environment=env, deque_size=1
)
env = CanonicalSpecWrapper(env, clip=True)

env = SinglePrecisionWrapper(env)
env = DmControlWrapper(env)

env = Dm2GymWrapper(env)
step = 0
err_poses = list()

demos = []
env = env.env
timestep = env.reset()
reward = 0
while not timestep.last():
    action = actions[step]
    # print(timestep.observation)
    # print(action)
    # timestep = env.step_offset(action, offset = piano_constants.WHITE_KEY_WIDTH)
    timestep = env.step(action)
    step += 1
    reward += timestep.reward
print(f"Total steps: {step}")
print(f"Total reward: {reward}")
print(env.get_musical_metrics())

# err_poses = np.array(err_poses)

play_video(env.latest_filename)

# Rename 00001.mp4 as "./demos/{}.mp4".format(task_name)
os.rename("./00001.mp4", "./demos/{}.mp4".format(task_name))

# plot_err_poses(err_poses, 4)