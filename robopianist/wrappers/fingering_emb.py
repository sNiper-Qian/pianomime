"""A wrapper for replacing goal and fingering with embedding in observation."""


import collections
from typing import Any, Dict, Optional

import dm_env
import numpy as np
from dm_env import specs
from dm_env_wrappers import EnvironmentWrapper


class FingeringEmbWrapper(EnvironmentWrapper):
    """Change observation['goal'] and observation['fingering'] to a embedding vector."""

    def __init__(
        self,
        environment: dm_env.Environment,
        observation_key: str = "fingering_emb",
        external_fingering: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(environment)
        self._observation_key = observation_key 
        # Update the observation spec.
        self._wrapped_observation_spec = self._environment.observation_spec()
        self._external_fingering = external_fingering
        # Extend the external fingering to the length of the task with the last value.
        if self._external_fingering is not None:
            self._external_fingering = np.pad(
                self._external_fingering, ((0, self.task._n_steps_lookahead+1), (0, 0)), 'edge'
            )

        # Remove unused goal and fingering observation.
        # print(self._wrapped_observation_spec)
        # del self._wrapped_observation_spec["goal"]
        del self._wrapped_observation_spec["fingering"]
        # Add fingering embedding observation.
        self._wrapped_observation_spec[self._observation_key] = specs.Array(
            shape=(10,), dtype=np.float64, name=self._observation_key
        )
        self._observation_spec = collections.OrderedDict()
        self._observation_spec.update(self._wrapped_observation_spec)

    def observation_spec(self):
        return self._observation_spec

    def reset(self) -> dm_env.TimeStep:
        timestep = self._environment.reset()
        return self._add_fingering_emb_observation(timestep)

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        timestep = self._environment.step(action)
        return self._add_fingering_emb_observation(timestep)
    
    def _add_fingering_emb_observation(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        if self._external_fingering is not None:
            fingering_emb = self._external_fingering[self.task._t_idx:self.task._t_idx+self.task._n_steps_lookahead+1].flatten()
        else:
            n_keys = 88
            n_steps_lookahead = self.task._n_steps_lookahead
            key = timestep.observation['goal'] # (n_lookahead + 1) * (n_piano_keys + 1) (with sustain pedal)
            fingering = timestep.observation['fingering']
            fingering_emb = np.zeros((n_steps_lookahead+1, 10)) # 10 fingers + sustain pedal
            for i in range(n_steps_lookahead+1):
                key_idx = np.flatnonzero(key[i*(n_keys+1):(i+1)*(n_keys+1)-1])
                fingering_l_to_r = fingering[i*10:(i+1)*10][::-1]
                finger_idx = np.flatnonzero(fingering_l_to_r)
                # print(finger_idx, key_idx)
                # fingering_emb[i][finger_idx] = (key_idx+1)/89 # normalize to [0, 1], plus one to avoid zero (zero is for unpressed)
                for idx, finger in enumerate(finger_idx):
                    fingering_emb[i][finger] = (key_idx[idx]+1)/89
                # fingering_emb[i][10] = key[(i+1)*(n_keys+1)-1]
            fingering_emb = fingering_emb.flatten()
            # del timestep.observation['goal']
        del timestep.observation['fingering']
        return timestep._replace(
            observation=collections.OrderedDict(
                timestep.observation, **{self._observation_key: fingering_emb}
            )
        )
        

