"""A wrapper for deep mimic framework."""


import collections
from typing import Any, Dict, Optional

import dm_env
import numpy as np
from dm_env import specs
from dm_env_wrappers import EnvironmentWrapper
import math
from dm_control import mjcf
from dm_control.utils.rewards import tolerance
import warnings
import random

_FINGERTIP_CLOSE_ENOUGH = 0.01
_FINGERTIP_CLOSE_ENOUGH_Z = 0.005
_JOINTS_CLOSE_ENOUGH = 0.05

class DeepMimicWrapper_Old(EnvironmentWrapper):
    """Change observation and reward to deep mimic fashion."""

    def __init__(
        self,
        environment: dm_env.Environment,
        demonstrations: np.ndarray,
        demo_ctrl_timestep: float = 0.01,
        remove_goal_observation: bool = False,
        disable_joints_pos_mimic_reward: bool = False,
        disable_end_effector_pos_mimic_reward: bool = False,
    ) -> None:
        super().__init__(environment)
        self._demonstrations = demonstrations
        self._step_scale = self._environment.task.control_timestep / demo_ctrl_timestep
        self.joints_mimic_rew = 0
        self.end_effector_mimic_rew = 0
        self._remove_goal_obs = remove_goal_observation
        self._disable_joints_pos_mimic_reward = disable_joints_pos_mimic_reward
        self._disable_end_effector_pos_mimic_reward = disable_end_effector_pos_mimic_reward

        # Update the observation spec.
        self._wrapped_observation_spec = self._environment.observation_spec()
        self._observation_spec = collections.OrderedDict()
        self._observation_spec.update(self._wrapped_observation_spec)
        # # Remove the goal observation.
        if self._remove_goal_obs:
            del self._observation_spec['goal']
            del self._observation_spec['piano/state']
            del self._observation_spec['piano/sustain_state']
        # Add the demo observation.
        if self._disable_joints_pos_mimic_reward and self._disable_end_effector_pos_mimic_reward:
            raise Exception("At least one of the mimic reward should be enabled.")
        if self._disable_joints_pos_mimic_reward:
            self._demonstrations = self._demonstrations[:,:,27:]
        elif self._disable_end_effector_pos_mimic_reward:
            self._demonstrations = self._demonstrations[:,:,:27]
        demo = self._demonstrations[0].flatten()
        demo_spec = specs.Array(
            shape=demo.shape, dtype=demo.dtype, name='demo'
        )
        self._observation_spec['demo'] = demo_spec

        self._add_deep_mimic_rewards()
        self._reference_frame_idx = 0
        self._demonstrations_length = self._demonstrations.shape[0]
        self._action_divergence_termination = False
    
    def observation_spec(self):
        return self._observation_spec

    def step(self, action) -> dm_env.TimeStep:
        timestep = self._environment.step(action)
        self._reference_frame_idx = min(int(self.task._t_idx*self._step_scale), self._demonstrations_length-1)
        return self._remove_goal_observation(self._add_demo_observation(timestep))

    def reset(self) -> dm_env.TimeStep:
        timestep = self._environment.reset()
        self.joints_mimic_rew = 0
        self.end_effector_mimic_rew = 0
        self._reference_frame_idx = 0
        return self._remove_goal_observation(self._add_demo_observation(timestep))
    
    def _compute_end_effector_pos_mimic_reward(self, physics: mjcf.Physics) -> float:
        """Computes the reward for matching the end effector positions."""
        if self._disable_joints_pos_mimic_reward:
            lh_target = self._demonstrations[self._reference_frame_idx, 0]
            rh_target = self._demonstrations[self._reference_frame_idx, 1]
        else:
            # Last 15 dimensions are for end effector positions.
            lh_target = self._demonstrations[self._reference_frame_idx, 0, 27:]
            rh_target = self._demonstrations[self._reference_frame_idx, 1, 27:]
        lh_current = np.array(self.physics.bind(self.task.left_hand.fingertip_sites).xpos).flatten()
        rh_current = np.array(self.physics.bind(self.task.right_hand.fingertip_sites).xpos).flatten()
        diffs = []
        for i in range(5):
            diffs.append(np.linalg.norm(lh_target[i*3:(i+1)*3] - lh_current[i*3:(i+1)*3]))
            diffs.append(np.linalg.norm(rh_target[i*3:(i+1)*3] - rh_current[i*3:(i+1)*3]))
        rews = tolerance(
            np.hstack(diffs),
            bounds=(0, _FINGERTIP_CLOSE_ENOUGH),
            margin=(_FINGERTIP_CLOSE_ENOUGH * 5),
            sigmoid="gaussian",
        )
        self.end_effector_mimic_rew += float(np.mean(rews))
        return float(np.mean(rews))

    def _compute_joints_pos_mimic_reward(self, physics: mjcf.Physics) -> float:
        """Computes the reward for matching the joint positions."""
        # First 27 dimensions are for joint positions.
        lh_target = self._demonstrations[self._reference_frame_idx, 0, :27]
        rh_target = self._demonstrations[self._reference_frame_idx, 1, :27]
        lh_current = self._environment.task.left_hand.observables.joints_pos(physics)
        rh_current = self._environment.task.right_hand.observables.joints_pos(physics)
        diffs = []
        for i in range(27):
            diffs.append(np.linalg.norm(lh_target[i] - lh_current[i]))
            diffs.append(np.linalg.norm(rh_target[i] - rh_current[i]))
        rews = tolerance(
            np.hstack(diffs),
            bounds=(0, _JOINTS_CLOSE_ENOUGH),
            margin=(_JOINTS_CLOSE_ENOUGH * 5),
            sigmoid="gaussian",
        )
        self.joints_mimic_rew += float(np.mean(rews))
        return float(np.mean(rews))

    def _add_deep_mimic_rewards(self):
        if not self._disable_joints_pos_mimic_reward:
            self.task._reward_fn.add("joints_pos_mimic", self._compute_joints_pos_mimic_reward)
        if not self._disable_end_effector_pos_mimic_reward:
            self.task._reward_fn.add("end_effector_pos_mimic", self._compute_end_effector_pos_mimic_reward)
    
    def _remove_goal_observation(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        if self._remove_goal_obs:
            timestep.observation.pop("goal")
            timestep.observation.pop("piano/state")
            timestep.observation.pop("piano/sustain_state")
        return timestep

    def _add_demo_observation(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        demo = self._demonstrations[self._reference_frame_idx].flatten()
        return timestep._replace(
            observation=collections.OrderedDict(
                timestep.observation, **{"demo": demo}
            )
        )
    
    def get_deepmimic_rews(self):
        if self._disable_joints_pos_mimic_reward and self._disable_end_effector_pos_mimic_reward:
            return {}
        elif self._disable_joints_pos_mimic_reward:
            return {
                "end_effector_pos_mimic_rew": self.end_effector_mimic_rew,
            }
        elif self._disable_end_effector_pos_mimic_reward:
            return {
                "joints_pos_mimic_rew": self.joints_mimic_rew,
            }
        else:
            return {
                "joints_pos_mimic_rew": self.joints_mimic_rew,
                "end_effector_pos_mimic_rew": self.end_effector_mimic_rew,
            }


class DeepMimicWrapper(EnvironmentWrapper):
    """Change observation and reward to deep mimic fashion."""

    def __init__(
        self,
        environment: dm_env.Environment,
        demonstrations_lh: np.ndarray,
        demonstrations_rh: np.ndarray,
        demo_ctrl_timestep: float = 0.05,
        remove_goal_observation: bool = False,
        n_steps_lookahead: int = 2,
        mimic_z_axis: bool = False,    
        rsi: bool = False,
    ) -> None:
        super().__init__(environment)
        self._demonstrations_lh = demonstrations_lh
        self._demonstrations_rh = demonstrations_rh
        assert(len(self.task._notes)==len(self._demonstrations_lh))
        self._step_scale = self._environment.task.control_timestep / demo_ctrl_timestep
        self.end_effector_mimic_rew = 0
        self.end_effector_mimic_rew_z = 0
        self._remove_goal_obs = remove_goal_observation
        self._n_steps_lookahead = n_steps_lookahead
        self._mimic_z_axis = mimic_z_axis
        self._rsi = rsi # Reference state initialization

        # Select the useful demonstrations. Remove quats, key_pressed and wrist
        useful_columns = [0, 1, 2]
        self._demonstrations_lh = self._demonstrations_lh[:, useful_columns, 1:]
        self._demonstrations_rh = self._demonstrations_rh[:, useful_columns, 1:]

        # Update the observation spec.
        self._wrapped_observation_spec = self._environment.observation_spec()
        self._observation_spec = collections.OrderedDict()
        self._observation_spec.update(self._wrapped_observation_spec)
        # # Remove the goal observation.
        if self._remove_goal_obs:
            del self._observation_spec['goal']
            del self._observation_spec['piano/state']
            del self._observation_spec['piano/sustain_state']
        # Add the demo observation.
        demo_lh = np.transpose(self._demonstrations_lh[0:self._n_steps_lookahead+1], (0, 2, 1)).flatten()
        demo_lh_spec = specs.Array(
            shape=demo_lh.shape, dtype=demo_lh.dtype, name='demo'
        )
        self._observation_spec['demo_lh'] = demo_lh_spec
        demo_rh = np.transpose(self._demonstrations_rh[0:self._n_steps_lookahead+1], (0, 2, 1)).flatten()
        demo_rh_spec = specs.Array(
            shape=demo_rh.shape, dtype=demo_rh.dtype, name='demo'
        )
        self._observation_spec['demo_rh'] = demo_rh_spec
        self._add_deep_mimic_rewards()
        self._reference_frame_idx = -int(round(self._environment.task._initial_buffer_time/
                                               self._environment.task.control_timestep))
        self._demonstrations_length = self._demonstrations_lh.shape[0]
        self._action_divergence_termination = False
    
    def observation_spec(self):
        return self._observation_spec

    def step(self, action) -> dm_env.TimeStep:
        timestep = self._environment.step(action)
        self._reference_frame_idx = int(min(self._reference_frame_idx+self._step_scale, self._demonstrations_length-1))
        return self._remove_goal_observation(self._add_demo_observation(timestep))

    def reset(self) -> dm_env.TimeStep:
        timestep = self._environment.reset()
        self.end_effector_mimic_rew = 0
        self.end_effector_mimic_rew_z = 0
        self._reference_frame_idx = -int(round(self._environment.task._initial_buffer_time/
                                            self._environment.task.control_timestep))
        return self._remove_goal_observation(self._add_demo_observation(timestep))

    def get_fingertip_pos(self) -> np.ndarray:
        physics = self.physics
        lh_wrist = physics.named.data.site_xpos["lh_shadow_hand/wrist_site"].flatten()
        rh_wrist = physics.named.data.site_xpos["rh_shadow_hand/wrist_site"].flatten()
        lh_fingertips = np.array(self.physics.bind(self.task.left_hand.fingertip_sites).xpos).flatten()
        rh_fingertips = np.array(self.physics.bind(self.task.right_hand.fingertip_sites).xpos).flatten()
        lh_current = np.concatenate((lh_wrist, lh_fingertips))
        rh_current = np.concatenate((rh_wrist, rh_fingertips))
        return lh_current, rh_current
    
    def _compute_end_effector_pos_mimic_reward(self, physics: mjcf.Physics) -> float:
        """Computes the reward for matching the end effector positions."""
        # Give full reward when it is at initial buffer time.
        if self._reference_frame_idx < 0:
            return 0
        lh_target = self._demonstrations_lh[self._reference_frame_idx].T
        rh_target = self._demonstrations_rh[self._reference_frame_idx].T
        lh_current = np.array(self.physics.bind(self.task.left_hand.fingertip_sites).xpos).flatten()
        rh_current = np.array(self.physics.bind(self.task.right_hand.fingertip_sites).xpos).flatten()
        diffs = []
        diffs_z = []
        if self._mimic_z_axis:
            for i in range(5):
                diffs.append(np.linalg.norm(lh_target[i, :2] - lh_current[i*3:(i+1)*3][:2]))
                diffs.append(np.linalg.norm(rh_target[i, :2] - rh_current[i*3:(i+1)*3][:2])) 
                diffs_z.append(np.abs(lh_target[i, 2] - lh_current[i*3:(i+1)*3][2]))
                diffs_z.append(np.abs(rh_target[i, 2] - rh_current[i*3:(i+1)*3][2]))
            rews = tolerance(
                np.hstack(diffs),
                bounds=(0, _FINGERTIP_CLOSE_ENOUGH),
                margin=(_FINGERTIP_CLOSE_ENOUGH * 5),
                sigmoid="gaussian",
            )
            rews_z = tolerance(
                np.hstack(diffs_z),
                bounds=(0, _FINGERTIP_CLOSE_ENOUGH_Z),
                margin=(_FINGERTIP_CLOSE_ENOUGH_Z),
                sigmoid="gaussian",
            )
            rew = 0.5*(float(np.mean(rews)) + float(np.mean(rews_z)))
            self.end_effector_mimic_rew_z += 0.5*float(np.mean(rews_z))
        else:
            for i in range(5):
                diffs.append(np.linalg.norm(lh_target[i, :2] - lh_current[i*3:(i+1)*3][:2]))
                diffs.append(np.linalg.norm(rh_target[i, :2] - rh_current[i*3:(i+1)*3][:2])) 
            rews = tolerance(
                np.hstack(diffs),
                bounds=(0, _FINGERTIP_CLOSE_ENOUGH),
                margin=(_FINGERTIP_CLOSE_ENOUGH * 5),
                sigmoid="gaussian",
            )
            rew = float(np.mean(rews))
        self.end_effector_mimic_rew += rew
        return rew

    def _add_deep_mimic_rewards(self):
        self.task._reward_fn.add("end_effector_pos_mimic", self._compute_end_effector_pos_mimic_reward)
    
    def _remove_goal_observation(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        if self._remove_goal_obs:
            timestep.observation.pop("goal")
            timestep.observation.pop("piano/state")
            timestep.observation.pop("piano/sustain_state")
        return timestep

    def _add_demo_observation(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        if self._reference_frame_idx < 0:
            # If it is at initial buffer time, use the first frame.
            demo_lh = np.array([self._demonstrations_lh[0].T]*(self._n_steps_lookahead+1))
            demo_rh = np.array([self._demonstrations_rh[0].T]*(self._n_steps_lookahead+1))
            for i in range(0, max(0, self._reference_frame_idx+self._n_steps_lookahead+1)):
                demo_lh[i-self._reference_frame_idx] = self._demonstrations_lh[i].T
                demo_rh[i-self._reference_frame_idx] = self._demonstrations_rh[i].T
        else:
            demo_lh = np.array([self._demonstrations_lh[-1].T]*(self._n_steps_lookahead+1))
            demo_rh = np.array([self._demonstrations_rh[-1].T]*(self._n_steps_lookahead+1))
            for i in range(self._reference_frame_idx, min(self._reference_frame_idx+self._n_steps_lookahead, self._demonstrations_length)):
                demo_lh[i-self._reference_frame_idx] = self._demonstrations_lh[i].T
                demo_rh[i-self._reference_frame_idx] = self._demonstrations_rh[i].T
        return timestep._replace(
            observation=collections.OrderedDict(
                timestep.observation, **{"demo_lh": demo_lh.flatten(), 
                                        "demo_rh": demo_rh.flatten()}
            )
        )
    
    def get_deepmimic_rews(self):
        return {
            "end_effector_pos_mimic_rew": self.end_effector_mimic_rew,
            "end_effector_pos_mimic_rew_z": self.end_effector_mimic_rew_z,
        }