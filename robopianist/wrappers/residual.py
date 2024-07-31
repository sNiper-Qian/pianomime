"""A wrapper for residual learning framework."""
from robopianist.controller.ik_controller import move_finger_to_key, move_fingers_to_keys, move_fingers_to_pos_qp
import collections
from typing import Any, Dict, Optional

import dm_env
import numpy as np
from dm_env import specs
from dm_env_wrappers import EnvironmentWrapper
import math
from dm_control import mjcf
from dm_control.utils.rewards import tolerance
from dm_control.mujoco.wrapper import mjbindings
import random
mjlib = mjbindings.mjlib

_FINGERTIP_CLOSE_ENOUGH = 0.01

class ResidualWrapper(EnvironmentWrapper):
    """Change step function."""
    def __init__(
        self,
        environment: dm_env.Environment,
        demonstrations_lh: np.ndarray,
        demonstrations_rh: np.ndarray,
        demo_ctrl_timestep: float = 0.05,
        rsi: bool = False,
        enable_ik: bool = True,
        external_demo: bool = False,
    ) -> None:
        super().__init__(environment)
        self._demonstrations_lh = demonstrations_lh
        self._demonstrations_rh = demonstrations_rh
        useful_columns = [0, 1, 2]
        self._demonstrations_lh = self._demonstrations_lh[:, useful_columns, :]
        self._demonstrations_rh = self._demonstrations_rh[:, useful_columns, :]
        self._step_scale = self._environment.task.control_timestep / demo_ctrl_timestep
        self._reference_frame_idx = -int(round(self._environment.task._initial_buffer_time/
                                        self._environment.task.control_timestep))
        self._rsi = rsi
        self._enable_ik = enable_ik
        assert self._demonstrations_lh.shape[0] == self._demonstrations_rh.shape[0]
        self._demonstrations_length = self._demonstrations_lh.shape[0]
        # Update the observation spec.
        self._wrapped_observation_spec = self._environment.observation_spec()
        self._observation_spec = collections.OrderedDict()
        self._observation_spec.update(self._wrapped_observation_spec)
        # Add the prior action observation.
        prior_action = np.zeros(self._environment.action_spec().shape[0]-1, dtype=np.float64)
        prior_action_spec = specs.Array(
            shape=prior_action.shape, dtype=prior_action.dtype, name='prior_action'
        )
        self._observation_spec['prior_action'] = prior_action_spec
        # # Add the demo observation.
        # demo_lh= self._demonstrations_lh[0:self.task._n_steps_lookahead+1].flatten()
        # demo_rh = self._demonstrations_rh[0:self.task._n_steps_lookahead+1].flatten()
        # demo = np.concatenate((demo_lh, demo_rh)).flatten()
        # demo_spec = specs.Array(
        #     shape=demo.shape, dtype=demo.dtype, name='demo'
        # )
        # self._observation_spec['demo'] = demo_spec

        # self._add_end_effector_pos_mimic_reward()
        self._prior_action = None
        self._lh_target = None
        self._rh_target = None
        self._mimic_reward = 0
        self._external_demo = external_demo
        self.current_demo_lh = None
        self.current_demo_rh = None

    def observation_spec(self):
        return self._observation_spec
    
    def _add_prior_action_observation(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        prior_qpos = self._get_prior_action()
        self._prior_action = self.qpos2ctrl(prior_qpos)
        return timestep._replace(
            observation=collections.OrderedDict(
                timestep.observation, **{"prior_action": self._prior_action}
            )
        )
    
    def _add_demo_observation(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        if self._external_demo:
            if self.current_demo_lh is not None and self.current_demo_rh is not None:
                demo_lh = self.current_demo_lh[0:self.task._n_steps_lookahead+1]
                demo_rh = self.current_demo_rh[0:self.task._n_steps_lookahead+1]
                self.current_demo_lh = None
                self.current_demo_rh = None
            else:
                raise ValueError("External demo is enabled but no demo is provided.")
        else:
            # print(self._demonstrations_lh[0])
            # print(self._demonstrations_rh[0])
            demo_lh = self._demonstrations_lh[self._reference_frame_idx:self._reference_frame_idx+self.task._n_steps_lookahead+1]
            demo_rh = self._demonstrations_rh[self._reference_frame_idx:self._reference_frame_idx+self.task._n_steps_lookahead+1]
            if self._reference_frame_idx + self.task._n_steps_lookahead >= self._demonstrations_length:
                # Fill rest with the last frame
                demo_lh = np.concatenate((demo_lh, self._demonstrations_lh[-1].reshape(1, 3, 6).repeat(self._reference_frame_idx + self.task._n_steps_lookahead - self._demonstrations_length + 1, axis=0)))
                demo_rh = np.concatenate((demo_rh, self._demonstrations_rh[-1].reshape(1, 3, 6).repeat(self._reference_frame_idx + self.task._n_steps_lookahead - self._demonstrations_length + 1, axis=0)))
        demo_lh = np.transpose(demo_lh, (0, 2, 1)).flatten()
        demo_rh = np.transpose(demo_rh, (0, 2, 1)).flatten()
        demo = np.concatenate((demo_lh, demo_rh)).flatten()
        return timestep._replace(
            observation=collections.OrderedDict(
                timestep.observation, **{"demo": demo}
            )
        ) 

    def set_current_demo(self, demonstrations_lh, demonstrations_rh):
        self.current_demo_lh = demonstrations_lh
        self.current_demo_rh = demonstrations_rh

    def _get_prior_action(self) -> np.ndarray:
        if self._external_demo:
            # print("inner:", self._demonstrations_lh[max(0, self._reference_frame_idx)])
            # print("outer:", self.current_demo_lh[0])
            # raise ValueError("External demo is enabled but no demo is provided.")
            if self.current_demo_lh is not None and self.current_demo_rh is not None:
                qvel_left, lh_dof_indices, self._lh_target = move_fingers_to_pos_qp(self,
                                            self.current_demo_lh[0],
                                            finger_names=['th', 'ff', 'mf', 'rf', 'lf'],
                                            hand_side='left',
                                            targeting_wrist=True,
                                            )
                qvel_right, rh_dof_indices, self._rh_target = move_fingers_to_pos_qp(self,
                                            self.current_demo_rh[0],
                                            finger_names=['th', 'ff', 'mf', 'rf', 'lf'],
                                            hand_side='right',
                                            targeting_wrist=True,
                                            )
            else:
                raise ValueError("External demo is enabled but no demo is provided.")
        else:
            qvel_left, lh_dof_indices, self._lh_target = move_fingers_to_pos_qp(self,
                                        self._demonstrations_lh[max(0, self._reference_frame_idx)],
                                        finger_names=['th', 'ff', 'mf', 'rf', 'lf'],
                                        hand_side='left',
                                        targeting_wrist=True,
                                        )
            qvel_right, rh_dof_indices, self._rh_target = move_fingers_to_pos_qp(self,
                                        self._demonstrations_rh[max(0, self._reference_frame_idx)],
                                        finger_names=['th', 'ff', 'mf', 'rf', 'lf'],
                                        hand_side='right',
                                        targeting_wrist=True,
                                        )
        v_full = np.zeros(self.physics.model.nv, dtype=self.physics.data.qpos.dtype)
        v_full[lh_dof_indices] = qvel_left
        v_full[rh_dof_indices] = qvel_right
        pos = self.physics.data.qpos.copy()
        mjlib.mj_integratePos(self.physics.model.ptr, pos, v_full, 0.05)
        return pos[88:]

    def qpos2ctrl(self, qpos):
        # action = np.zeros(54, dtype=np.float64)
        # action[0:24] = qpos[3:27]
        # action[24:27] = qpos[0:3]
        # action[27:51] = qpos[30:54]
        # action[51:54] = qpos[27:30]
        # Tendon is estimated by the sum of the two joint angles
        action = np.zeros(46, dtype=np.float64)
        action[0:2] = qpos[3:5]
        action[7:9] = qpos[5:7]
        action[9] = qpos[7] + qpos[8]
        action[10:12] = qpos[9:11]
        action[12] = qpos[11] + qpos[12]
        action[13:15] = qpos[13:15]
        action[15] = qpos[15] + qpos[16]
        action[16:19] = qpos[17:20]
        action[19] = qpos[20] + qpos[21]
        action[2:7] = qpos[22:27]
        action[20:23] = qpos[0:3]
        action[23:25] = qpos[30:32]
        action[30:32] = qpos[32:34]
        action[32] = qpos[34] + qpos[35]
        action[33:35] = qpos[36:38]
        action[35] = qpos[38] + qpos[39]
        action[36:38] = qpos[40:42]
        action[38] = qpos[42] + qpos[43]
        action[39:42] = qpos[44:47]
        action[42] = qpos[47] + qpos[48]
        action[25:30] = qpos[49:54]
        action[43:46] = qpos[27:30]
        return action

    def step(self, action) -> dm_env.TimeStep:
        if self._enable_ik:
            action_hand = action[:-1] + self._prior_action # Apply residual
        else:
            action_hand = action[:-1]
        self.non_residual_action = action_hand
        # self.physics.data.qpos[88:] = action_hand # Apply qpos instead of ctrl. Only sustain pedal is acted in task.before_step().
        action_sustain = action[-1]
        # Merge action_sustain into action_hand 
        action = np.append(action_hand, action_sustain)
        timestep = self._environment.step(action)
        self._reference_frame_idx = int(min(self._reference_frame_idx+self._step_scale, self._demonstrations_length-1))
        return self._add_demo_observation(self._add_prior_action_observation(timestep))
    
    def get_non_residual_action(self):
        return self.non_residual_action

    def reset(self) -> dm_env.TimeStep:
        timestep = self._environment.reset()
        self._mimic_reward = 0
        if self._rsi:
            self._reference_frame_idx = random.randint(-int(round(self._environment.task._initial_buffer_time/
                                               self._environment.task.control_timestep)), self._demonstrations_length-1)
            self._environment._reference_frame_idx = self._reference_frame_idx
            self._environment.task._t_idx = int(self._reference_frame_idx/self._step_scale)
            reference_joint_pos = self._get_prior_action()
            action = self.qpos2ctrl(reference_joint_pos)
            self._environment.task.right_hand.configure_joints(self.physics, action[:27])
            self._environment.task.left_hand.configure_joints(self.physics, action[27:])
        else: 
            self._reference_frame_idx = -int(round(self._environment.task._initial_buffer_time/
                                               self._environment.task.control_timestep))
        return self._add_demo_observation(self._add_prior_action_observation(timestep))

    
