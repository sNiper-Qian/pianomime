import numpy as np
from typing import Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

from robopianist import suite
import dm_env_wrappers as wrappers
import robopianist.wrappers as robopianist_wrappers
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
from mujoco_utils import composer_utils
from dm_env_wrappers import CanonicalSpecWrapper
from robopianist.wrappers import PianoSoundVideoWrapper
import pickle
import torch
from robopianist.models.piano import piano_constants as consts
from robopianist import music

WHITE_KEY_INDICES = [
        0,
        2,
        3,
        5,
        7,
        8,
        10,
        12,
        14,
        15,
        17,
        19,
        20,
        22,
        24,
        26,
        27,
        29,
        31,
        32,
        34,
        36,
        38,
        39,
        41,
        43,
        44,
        46,
        48,
        50,
        51,
        53,
        55,
        56,
        58,
        60,
        62,
        63,
        65,
        67,
        68,
        70,
        72,
        74,
        75,
        77,
        79,
        80,
        82,
        84,
        86,
        87,
    ]

BLACK_TWIN_KEY_INDICES = [
        4,
        6,
        16,
        18,
        28,
        30,
        40,
        42,
        52,
        54,
        64,
        66,
        76,
        78,
    ]
BLACK_TRIPLET_KEY_INDICES = [
        1,
        9,
        11,
        13,
        21,
        23,
        25,
        33,
        35,
        37,
        45,
        47,
        49,
        57,
        59,
        61,
        69,
        71,
        73,
        81,
        83,
        85,
    ]

@dataclass(frozen=True)
class Args:
    root_dir: str = None
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
    trim_silence: bool = True
    gravity_compensation: bool = True
    reduced_action_space: bool = False
    control_timestep: float = 0.05
    stretch_factor: float = 1.0
    shift_factor: int = 0
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = False
    disable_forearm_reward: bool = False
    disable_colorization: bool = False
    disable_hand_collisions: bool = True
    primitive_fingertip_collisions: bool = False
    frame_stack: int = 1
    clip: bool = True
    record_dir: Optional[Path] = None
    record_every: int = 1
    record_resolution: Tuple[int, int] = (480, 640)
    camera_id: Optional[str | int] = "piano/back"
    action_reward_observation: bool = False

def get_diffusion_obs_1(timestep, exclude_keys=[]):
    ret = {}
    if 'fingering' not in exclude_keys:
        fingering = timestep.observation['fingering']
        ret['fingering'] = fingering
    q_piano = timestep.observation['piano/state']
    ret['q_piano'] = q_piano
    goal = timestep.observation['goal']
    ret['goal'] = goal
    q_hand_l = timestep.observation['lh_shadow_hand/joints_pos'] / (np.pi) # normalize to [-1, 1]
    q_hand_r = timestep.observation['rh_shadow_hand/joints_pos'] / (np.pi) # normalize to [-1, 1]
    q_hand = np.concatenate((q_hand_l, q_hand_r), axis=0).flatten()
    ret['q_hand'] = q_hand
    v_hand_l = timestep.observation['lh_shadow_hand/joints_vel'] / (np.pi) # normalize to [-1, 1]
    v_hand_r = timestep.observation['rh_shadow_hand/joints_vel'] / (np.pi) # normalize to [-1, 1]
    v_hand = np.concatenate((v_hand_l, v_hand_r), axis=0).flatten()
    ret['v_hand'] = v_hand
    if 'demo' not in exclude_keys:
        demo = timestep.observation['demo']
        ret['demo'] = demo
    if 'prior_action' not in exclude_keys:
        prior = timestep.observation['prior_action']
        ret['prior_action'] = prior
    # print(observation)
    return ret

def get_goal_only_obs(timestep, lookahead=0):
    goal = timestep.observation['goal']
    goal = goal.reshape(lookahead+1, 89)
    goal = goal[:, :-1]
    goal = goal.flatten()
    return goal

def get_diffusion_obs(timestep, lookahead=3, exclude_keys=[], 
                      encoder=None, plan_encoder=None, sampling=False, 
                      current_fingertip=None, concatenate_keys=[]):
    ret = {}
    goal = timestep.observation['goal'][:89*(lookahead+1)]
    goal = goal.reshape(lookahead+1, 89)
    goal = goal[:, :-1]
    # for i in range(lookahead+1):
    #     # print non-zero elements
    #     print(np.nonzero(goal[i]))
    if encoder is not None and sampling:
        # Add a dimension to goal at last axis
        goal = np.expand_dims(goal, axis=-1)
        goal = encoder(torch.from_numpy(goal))
        ret['goal'] = goal.detach().numpy().flatten()
    elif encoder is not None and not sampling:
        goal = np.expand_dims(goal, axis=-1)
        # print(goal.reshape(lookahead+1, -1)[0])
        if torch.cuda.is_available():
            goal = torch.from_numpy(goal).cuda()
            goal = encoder.forward_without_sampling(goal)
            ret['goal'] = goal.detach().cpu().numpy().flatten()
            # print(ret['goal'].reshape(lookahead+1, -1)[0])
            # raise ValueError
        else:
            goal = encoder.forward_without_sampling(torch.from_numpy(goal))
            ret['goal'] = goal.detach().numpy().flatten()
    else:
        ret['goal'] = goal.flatten()      
    if plan_encoder is not None:
        assert current_fingertip is not None
        goal = ret['goal']
        goal = torch.from_numpy(goal)
        cond = torch.from_numpy(current_fingertip)
        goal = goal.unsqueeze(0).float()
        cond = cond.unsqueeze(0).float()
        plan = plan_encoder.forward_without_sampling(goal, cond)
        ret['goal'] = plan.detach().numpy().flatten()
    elif current_fingertip is not None:
        ret['current_fingertip'] = current_fingertip.flatten()
    if 'hand' not in exclude_keys:  
        q_hand_l = timestep.observation['lh_shadow_hand/joints_pos'] / (np.pi) # normalize to [-1, 1]
        q_hand_r = timestep.observation['rh_shadow_hand/joints_pos'] / (np.pi) # normalize to [-1, 1]
        q_hand = np.concatenate((q_hand_l, q_hand_r), axis=0).flatten()
        ret['q_hand'] = q_hand
        v_hand_l = timestep.observation['lh_shadow_hand/joints_vel'] / (np.pi) # normalize to [-1, 1]
        v_hand_r = timestep.observation['rh_shadow_hand/joints_vel'] / (np.pi) # normalize to [-1, 1]
        v_hand = np.concatenate((v_hand_l, v_hand_r), axis=0).flatten()
        ret['v_hand'] = v_hand
    if 'fingering' not in exclude_keys:
        if 'fingering' in timestep.observation:
            fingering = timestep.observation['fingering']
        else:
            fingering = timestep.observation['fingering_emb'][:40]
        ret['fingering'] = fingering
    if 'demo' not in exclude_keys:
        demo = timestep.observation['demo']
        demo_lh, demo_rh = np.split(demo, 2)
        demo = np.zeros((4, 36))
        for i in range(4):
            demo[i] = np.concatenate((demo_lh[i*18:(i+1)*18], demo_rh[i*18:(i+1)*18]), axis=0)
        ret['demo'] = demo.flatten()
    if 'prior_action' not in exclude_keys:
        prior = timestep.observation['prior_action']
        ret['prior_action'] = prior
    if 'q_piano' not in exclude_keys:
        q_piano = timestep.observation['piano/state']
        ret['q_piano'] = q_piano
    if concatenate_keys != []:
        for key in concatenate_keys:
            ret[key] = ret[key].reshape(lookahead+1, -1)
        ret['cont'] = np.concatenate([ret[key] for key in concatenate_keys], axis=1).flatten()
        for key in concatenate_keys:
            del ret[key]
    return ret

def get_flattend_obs(timestep, lookahead=3, exclude_keys=[], encoder=None, sampling=False, 
                     plan_encoder=None, current_fingertip=None, concatenate_keys=[]):
    ret = get_diffusion_obs(timestep, lookahead, exclude_keys=exclude_keys, encoder=encoder, sampling=sampling, 
                            plan_encoder=plan_encoder, current_fingertip=current_fingertip, concatenate_keys=concatenate_keys)
    # Concatenate the items in ret
    items = list(ret.values())
    if 'cont' in ret:
        # put cont at the first
        items = [ret['cont']] + items[:-1]
    # for item in items:
    #     print(item.shape)
    obs = np.concatenate(items, axis=0).flatten()
    # obs = np.concatenate((fingering, q_piano, goal, q_hand, v_hand, demo, prior), axis=0).flatten()
    return obs

def get_env_test(task_name, enable_ik = True, record_dir=None, lookahead = 3,
                use_fingering_emb=False, use_note_traj=False):
    # start_from = start_from_dict.START_FROM[task_name]
    if use_note_traj:
        with open('handtracking/notes/{}.pkl'.format(task_name), 'rb') as f:
            note_traj = pickle.load(f)

    trim = True
    if use_note_traj:
        task = piano_with_shadow_hands_res.PianoWithShadowHandsResidual(
            # hand_side=HandSide.LEFT,
            note_trajectory=note_traj,
            # midi=music.load(task_name),
            change_color_on_activation=True,
            trim_silence=trim,
            control_timestep=0.05,
            disable_hand_collisions=True,
            disable_forearm_reward=True,
            disable_fingering_reward=False,
            midi_start_from=0,
            n_steps_lookahead=lookahead,
            gravity_compensation=True,
            residual_factor=0.03 if enable_ik else 1,
            shift=0,
            enable_joints_vel_obs=True,
            fingering_lookahead=use_fingering_emb,
        )
    else:
        task = piano_with_shadow_hands_res.PianoWithShadowHandsResidual(
            # hand_side=HandSide.LEFT,
            # note_trajectory=note_traj,
            midi=music.load(task_name),
            change_color_on_activation=True,
            trim_silence=trim,
            control_timestep=0.05,
            disable_hand_collisions=True,
            disable_forearm_reward=True,
            disable_fingering_reward=False,
            midi_start_from=0,
            n_steps_lookahead=lookahead,
            gravity_compensation=True,
            residual_factor=0.03 if enable_ik else 1,
            shift=0,
            enable_joints_vel_obs=True,
            fingering_lookahead=use_fingering_emb,
        )

    env = composer_utils.Environment(
        recompile_physics=False, task=task, strip_singleton_obs_buffer_dim=True
    )
    if record_dir is not None:
        env = PianoSoundVideoWrapper(
            env,
            record_every=1,
            camera_id="piano/back",
            record_dir=record_dir,
        )
    env = MidiEvaluationWrapper(
        environment=env, deque_size=1
    )
    if enable_ik:
        env = CanonicalSpecWrapper(env, clip=True)

    env = SinglePrecisionWrapper(env)
    env = DmControlWrapper(env)

    env = Dm2GymWrapper(env)
    return env.env
    

def get_env_hl(task_name, record_dir=None, lookahead = 3, use_fingering_emb=False,
                use_midi=False):
    # start_from = start_from_dict.START_FROM[task_name]
    if not use_midi:
        try:
            with open('dataset/notes/{}.pkl'.format(task_name), 'rb') as f:
                note_traj = pickle.load(f)
        except:
            with open('dataset/notes_test/{}.pkl'.format(task_name), 'rb') as f:
                note_traj = pickle.load(f)
        notes = note_traj.notes        
        length = len(notes)
        trim = False if length >=600 or length < 500 else True
    if use_midi:
        task = piano_with_shadow_hands_res.PianoWithShadowHandsResidual(
            # hand_side=HandSide.LEFT,
            # note_trajectory=note_traj, 
            midi=music.load(task_name),
            change_color_on_activation=True,
            trim_silence=True,
            control_timestep=0.05,
            disable_hand_collisions=True,
            disable_forearm_reward=True,
            disable_fingering_reward=False,
            midi_start_from=0,
            n_steps_lookahead=lookahead,
            gravity_compensation=True,
            residual_factor=1,
            shift=0,
            enable_joints_vel_obs=True,
            fingering_lookahead=use_fingering_emb,
        )
    else:
        task = piano_with_shadow_hands_res.PianoWithShadowHandsResidual(
            # hand_side=HandSide.LEFT,
            note_trajectory=note_traj,
            # midi=music.load(task_name),
            change_color_on_activation=True,
            trim_silence=trim,
            control_timestep=0.05,
            disable_hand_collisions=True,
            disable_forearm_reward=True,
            disable_fingering_reward=False,
            midi_start_from=0,
            n_steps_lookahead=lookahead,
            gravity_compensation=True,
            residual_factor=1,
            shift=0,
            enable_joints_vel_obs=True,
            fingering_lookahead=use_fingering_emb,
        )

    env = composer_utils.Environment(
        recompile_physics=False, task=task, strip_singleton_obs_buffer_dim=True
    )
    if record_dir is not None:
        env = PianoSoundVideoWrapper(
            env,
            record_every=1,
            camera_id="piano/back",
            record_dir=record_dir,
        )
    env = MidiEvaluationWrapper(
        environment=env, deque_size=1
    )
    env = CanonicalSpecWrapper(env, clip=True)

    env = SinglePrecisionWrapper(env)
    env = DmControlWrapper(env)

    env = Dm2GymWrapper(env)
    
    return env.env, length

def get_env_ll(task_name, enable_ik = True, record_dir=None, lookahead = 3, external_demo=False, use_fingering_emb=False,
            external_fingering=None, use_midi=False):
    # start_from = start_from_dict.START_FROM[task_name]
    if not use_midi:
        try:
            with open('dataset/notes/{}.pkl'.format(task_name), 'rb') as f:
                note_traj = pickle.load(f)
        except:
            with open('dataset/notes_test/{}.pkl'.format(task_name), 'rb') as f:
                note_traj = pickle.load(f)

    # Load hand action trajectory
    left_hand_action_list = np.load('pianomime/multi_task/trajectories/{}_left_hand_action_list.npy'.format(task_name))
    right_hand_action_list = np.load('pianomime/multi_task/trajectories/{}_right_hand_action_list.npy'.format(task_name))
            
    length = left_hand_action_list.shape[0]
    trim = False if length >=600 or length < 500 else True

    if use_midi:
        task = piano_with_shadow_hands_res.PianoWithShadowHandsResidual(
            midi=music.load(task_name),
            change_color_on_activation=True,
            trim_silence=True,
            control_timestep=0.05,
            disable_hand_collisions=True,
            disable_forearm_reward=True,
            disable_fingering_reward=False,
            midi_start_from=0,
            n_steps_lookahead=lookahead,
            gravity_compensation=True,
            residual_factor=0.03 if enable_ik else 1,
            shift=0,
            enable_joints_vel_obs=True,
            fingering_lookahead=use_fingering_emb,
        )
    else:
        task = piano_with_shadow_hands_res.PianoWithShadowHandsResidual(
            note_trajectory=note_traj,
            change_color_on_activation=True,
            trim_silence=trim,
            control_timestep=0.05,
            disable_hand_collisions=True,
            disable_forearm_reward=True,
            disable_fingering_reward=False,
            midi_start_from=0,
            n_steps_lookahead=lookahead,
            gravity_compensation=True,
            residual_factor=0.03 if enable_ik else 1,
            shift=0,
            enable_joints_vel_obs=True,
            fingering_lookahead=use_fingering_emb,
        )

    env = composer_utils.Environment(
        recompile_physics=False, task=task, strip_singleton_obs_buffer_dim=True
    )

    if record_dir is not None:
        env = PianoSoundVideoWrapper(
            env,
            record_every=1,
            camera_id="piano/back",
            record_dir=record_dir,
        )
    if use_fingering_emb:
        env = FingeringEmbWrapper(env, external_fingering=external_fingering)
    if left_hand_action_list is None or right_hand_action_list is None:
        length = len(env.task._notes)
        left_hand_action_list = np.zeros((length, 8, 6))
        right_hand_action_list = np.zeros((length, 8, 6))
    env = DeepMimicWrapper(env,
                        demonstrations_lh=left_hand_action_list,
                        demonstrations_rh=right_hand_action_list,
                        remove_goal_observation=False,
                        mimic_z_axis=False,
                        n_steps_lookahead=lookahead,)
    env = ResidualWrapper(env, 
                        demonstrations_lh=left_hand_action_list,
                        demonstrations_rh=right_hand_action_list,
                        demo_ctrl_timestep=0.05,
                        enable_ik=enable_ik,
                        external_demo=external_demo,)
    env = MidiEvaluationWrapper(
        environment=env, deque_size=1
    )
    if enable_ik:
        env = CanonicalSpecWrapper(env, clip=True)

    env = SinglePrecisionWrapper(env)
    env = DmControlWrapper(env)

    env = Dm2GymWrapper(env)
    
    return env.env

def adjust_ft_fingering(env, keys, lh_ft, rh_ft, last_keys=None, last_lh_ft=None, last_rh_ft=None, last_fingering=None):
    # print(lh_ft, rh_ft)
    lh_ft[2, :] = np.ones(6) * consts.WHITE_KEY_HEIGHT * 2
    rh_ft[2, :] = np.ones(6) * consts.WHITE_KEY_HEIGHT * 2
    lh_ft_old = lh_ft.copy()
    rh_ft_old = rh_ft.copy()
    lh_ft = lh_ft.T[1:] 
    rh_ft = rh_ft.T[1:]
    ft_l_to_r = np.concatenate((lh_ft[::-1], rh_ft), axis=0)
    # print(ft_l_to_r)
    if last_lh_ft is not None and last_rh_ft is not None:
        last_lh_ft = last_lh_ft.T[1:]
        last_rh_ft = last_rh_ft.T[1:]
        last_ft_l_to_r = np.concatenate((last_lh_ft[::-1], last_rh_ft), axis=0)
        last_keys = [int(key) for key in last_keys[0]]
        
    fingering = np.ones(10) * -1
    occupied_fingers = []
    keys = [int(key) for key in keys[0]]
    for key in keys:
        key_pos = env.task.piano._keys[key].pos
        # Check if the key is in the last_keys
        if last_keys is not None and key in last_keys and key in last_fingering:
            # Find the finger idx of the key
            finger_idx = last_fingering.tolist().index(key)
            fingering[finger_idx] = key
            occupied_fingers.append(finger_idx)
            ft_l_to_r[finger_idx][1] = key_pos[1]
            ft_l_to_r[finger_idx][2] = 0
        else:
            min_dist = 100
            min_dist_finger_idx = -1
            # Find the closest finger to the key
            for i, ft in enumerate(ft_l_to_r):
                if i in occupied_fingers:
                    continue
                dist = abs(ft[1] - key_pos[1])
                if dist < min_dist and ft[0] < key_pos[0] + 0.52 * consts.WHITE_KEY_LENGTH:
                    min_dist = dist
                    min_dist_finger_idx = i
            if min_dist_finger_idx == -1:
                continue
            fingering[min_dist_finger_idx] = key
            occupied_fingers.append(min_dist_finger_idx)
            ft_l_to_r[min_dist_finger_idx][1] = key_pos[1]
            ft_l_to_r[min_dist_finger_idx][2] = 0
            if key in WHITE_KEY_INDICES:
                if not(ft_l_to_r[min_dist_finger_idx][0] < key_pos[0] + consts.WHITE_KEY_LENGTH*0.5 and \
                    ft_l_to_r[min_dist_finger_idx][0] > key_pos[0] - consts.WHITE_KEY_LENGTH*0.5):
                    ft_l_to_r[min_dist_finger_idx][0] = key_pos[0] + 1/4*consts.WHITE_KEY_LENGTH
            else:
                if not(ft_l_to_r[min_dist_finger_idx][0] < key_pos[0] + consts.BLACK_KEY_LENGTH*0.5 and \
                    ft_l_to_r[min_dist_finger_idx][0] > key_pos[0] - consts.BLACK_KEY_LENGTH*0.5):
                    ft_l_to_r[min_dist_finger_idx][0] = key_pos[0] + 1/4*consts.BLACK_KEY_LENGTH
    # split the ft_l_to_r to lh_ft and rh_ft
    lh_ft_replace = ft_l_to_r[:5][::-1].T
    rh_ft_replace = ft_l_to_r[5:].T
    lh_ft = np.concatenate((lh_ft_old[:,0:1], lh_ft_replace), axis=1)
    rh_ft = np.concatenate((rh_ft_old[:,0:1], rh_ft_replace), axis=1)
    # print(lh_ft, rh_ft)
    # print(fingering)
    return lh_ft, rh_ft, fingering
                    
                
            
        
            

