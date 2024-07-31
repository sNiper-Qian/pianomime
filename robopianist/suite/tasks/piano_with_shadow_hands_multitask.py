# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A task where two shadow hands must play a given MIDI file on a piano."""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from dm_control import mjcf
from dm_control.composer import variation as base_variation
from dm_control.composer.observation import observable
from dm_control.mjcf import commit_defaults
from dm_control.utils.rewards import tolerance
from dm_env import specs
from mujoco_utils import collision_utils, spec_utils

import robopianist.models.hands.shadow_hand_constants as hand_consts
from robopianist.models.arenas import stage
from robopianist.music import midi_file
from robopianist.suite import composite_reward
from robopianist.suite.tasks import base

# Distance thresholds for the shaping reward.
_FINGER_CLOSE_ENOUGH_TO_KEY = 0.01
_KEY_CLOSE_ENOUGH_TO_PRESSED = 0.05

# Energy penalty coefficient.
_ENERGY_PENALTY_COEF = 5e-3

# Transparency of fingertip geoms.
_FINGERTIP_ALPHA = 1.0

# Bounds for the uniform distribution from which initial hand offset is sampled.
_POSITION_OFFSET = 0.05

class PianoWithShadowHandsMultiTask(base.PianoTask):
    def __init__(
        self,
        note_trajectories: midi_file.NoteTrajectory = None,
        midis: midi_file.MidiFile = None,
        task_names = None,
        n_steps_lookahead: int = 1,
        n_seconds_lookahead: Optional[float] = None,
        trim_silence: bool = False,
        wrong_press_termination: bool = False,
        initial_buffer_time: float = 0.0,
        disable_fingering_reward: bool = False,
        disable_forearm_reward: bool = False,
        disable_colorization: bool = False,
        disable_hand_collisions: bool = False,
        augmentations: Optional[Sequence[base_variation.Variation]] = None,
        energy_penalty_coef: float = _ENERGY_PENALTY_COEF,
        randomize_hand_positions: bool = False,
        fingering_lookahead: bool=False,
        midi_start_from: int=0,
        residual_factor: float = 0.02,
        curriculum: bool = False,
        enable_joint_vel_obs: bool = False,
        enable_base_joint_ctrl_domain: bool = False,
        subgoal_length: int = 10,
        **kwargs,
    ) -> None:
        """Task constructor.

        Args:
            note_trajectory: A `NoteTrajectory` object.
            midi: A `MidiFile` object.
            n_steps_lookahead: Number of timesteps to look ahead when computing the
                goal state.
            n_seconds_lookahead: Number of seconds to look ahead when computing the
                goal state. If specified, this will override `n_steps_lookahead`.
            trim_silence: If True, shifts the MIDI file so that the first note starts
                at time 0.
            wrong_press_termination: If True, terminates the episode if the hands press
                the wrong keys at any timestep.
            initial_buffer_time: Specifies the duration of silence in seconds to add to
                the beginning of the MIDI file. A non-zero value can be useful for
                giving the agent time to place its hands near the first notes.
            disable_fingering_reward: If True, disables the shaping reward for
                fingering. This will also disable the colorization of the fingertips
                and corresponding keys. Note that if the MIDI file does not contain
                any fingering information, the fingering reward will also be disabled.
            disable_forearm_reward: If True, disables the shaping reward for the
                forearms.
            disable_colorization: If True, disables the colorization of the fingertips
                and corresponding keys.
            disable_hand_collisions: If True, disables collisions between the two hands.
            augmentations: A list of `Variation` objects that will be applied to the
                MIDI file at the beginning of each episode. If None, no augmentations
                will be applied.
            energy_penalty_coef: Coefficient for the energy penalty.
            randomize_hand_positions: If True, randomizes the initial position of the
                hands at the beginning of each episode.
            fingering_lookahead: If True, insert the ahead fingering information in the 
                observation.
            midi_start_from: The index of the note to start from.
            residual_factor: The factor to multiply the residual action.
            curriculum: If True, use curriculum learning.
        """
        super().__init__(arena=stage.Stage(), **kwargs)
        if note_trajectories is None and midis is None:
            raise ValueError("Either `note_trajectory` or `midi` must be specified.")
        if note_trajectories:
            self._note_trajs = note_trajectories
            if trim_silence:
                for note_traj in self._note_trajs:
                    note_traj = note_traj.trim_silence()
            scale = int(0.05 / self.control_timestep)
            self._scale_note_traj(scale)
            self._traj_num = len(self._note_trajs)
        else:
            self._midis = midis
            self._initial_midis = midis
            if trim_silence:
                for midi in self._midis:
                    midi = midi.trim_silence()
            self._traj_num = len(self._midis)
        assert self._traj_num > 0 
        self._task_names = task_names

        self._n_steps_lookahead = n_steps_lookahead
        if n_seconds_lookahead is not None:
            self._n_steps_lookahead = int(
                np.ceil(n_seconds_lookahead / self.control_timestep)
            )
        self._initial_buffer_time = initial_buffer_time
        self._disable_fingering_reward = disable_fingering_reward
        self._disable_forearm_reward = disable_forearm_reward
        self._wrong_press_termination = wrong_press_termination
        self._disable_colorization = disable_colorization
        self._disable_hand_collisions = disable_hand_collisions
        self._augmentations = augmentations
        self._energy_penalty_coef = energy_penalty_coef
        self._randomize_hand_positions = randomize_hand_positions
        self._fingering_lookahead = fingering_lookahead
        self._midi_start_from = midi_start_from
        self._residual_factor = residual_factor
        self._curriculum = curriculum
        self._curriculum_length = 100 # initial curriculum length (5 seconds)
        self._enable_joint_vel_obs = enable_joint_vel_obs

        if not disable_fingering_reward and not disable_colorization:
            self._colorize_fingertips()
        if disable_hand_collisions:
            self._disable_collisions_between_hands()
        self._reset_quantities_at_episode_init()
        self._reset_trajectory()  # Important: call before adding observables.
        self._add_observables()
        self._set_rewards()

    def _set_rewards(self) -> None:
        self._reward_fn = composite_reward.CompositeReward(
            key_press_reward=self._compute_key_press_reward,
            sustain_reward=self._compute_sustain_reward,
            energy_reward=self._compute_energy_reward,
        )
        if not self._disable_fingering_reward:
            self._reward_fn.add("fingering_reward", self._compute_fingering_reward)
        if not self._disable_forearm_reward:
            self._reward_fn.add("forearm_reward", self._compute_forearm_reward)

    def _reset_quantities_at_episode_init(self) -> None:
        self._t_idx: int = 0
        self._should_terminate: bool = False
        self._discount: float = 1.0

    def _maybe_change_midi(self, random_state: np.random.RandomState) -> None:
        if self._augmentations is not None:
            midi = self._initial_midi
            for var in self._augmentations:
                midi = var(initial_value=midi, random_state=random_state)
            self._midi = midi
        self._reset_trajectory()

    def _reset_trajectory(self) -> None:
        traj_idx = np.random.randint(self._traj_num)
        if hasattr(self, "_midis"):
            self.note_traj = midi_file.NoteTrajectory.from_midi(
                self._midis[traj_idx], self.control_timestep
            )
        else:
            self.note_traj = self._note_trajs[traj_idx]
        self.note_traj.add_initial_buffer_time(self._initial_buffer_time)
        self._task_name = self._task_names[traj_idx]
        self._notes = self.note_traj.notes
        self._sustains = self.note_traj.sustains

    def _scale_note_traj(self, scale):
        # For each note and sustain in note trajectory, repeat it `scale` times.
        # This is used to scale the note trajectory to the same length as the MIDI file.
        for note_traj in self._note_trajs:
            new_notes = []
            new_sustains = []
            for note, sustain in zip(note_traj.notes, note_traj.sustains):
                new_notes += [note] * scale
                new_sustains += [sustain] * scale
            note_traj.notes = new_notes
            note_traj.sustains = new_sustains
    
    def extend_curriculum(self) -> None:
        self._curriculum_length += 100

    # Composer methods.

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        self._maybe_change_midi(random_state)
        self._reset_quantities_at_episode_init()
        self._randomize_initial_hand_positions(physics, random_state)

    def before_step(
        self,
        physics: mjcf.Physics,
        action: np.ndarray,
        random_state: np.random.RandomState,
    ) -> None:
        """Applies the control to the hands and the sustain pedal to the piano."""
        action_right, action_left = np.split(action[:-1], 2)
        self.right_hand.apply_action(physics, action_right, random_state)
        self.left_hand.apply_action(physics, action_left, random_state)
        self.piano.apply_sustain(physics, action[-1], random_state)

    def after_step(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        # del random_state  # Unused.
        self._t_idx += 1
        self._should_terminate = (self._t_idx - 1) == len(self._notes) - 1
        if self._curriculum:
            self._should_terminate = self._should_terminate or self._t_idx == self._curriculum_length
        self._goal_current = self._goal_state[0]
        if not self._disable_fingering_reward:
            self._rh_keys_current = self._rh_keys
            self._lh_keys_current = self._lh_keys
            if not self._disable_colorization:
                self._colorize_keys_without_fingering(physics)
        should_not_be_pressed = np.flatnonzero(1 - self._goal_current[:-1])
        self._failure_termination = self.piano.activation[should_not_be_pressed].any()
        
    def get_reward(self, physics: mjcf.Physics) -> float:
        return self._reward_fn.compute(physics)

    def get_discount(self, physics: mjcf.Physics) -> float:
        del physics  # Unused.
        return self._discount

    def should_terminate_episode(self, physics: mjcf.Physics) -> bool:
        del physics  # Unused.
        if self._should_terminate:
            return True
        if self._wrong_press_termination and self._failure_termination:
            self._discount = 0.0
            return True
        return False

    @property
    def task_observables(self):
        return self._task_observables

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        right_spec = self.right_hand.action_spec(physics)
        left_spec = self.left_hand.action_spec(physics)
        hands_spec = spec_utils.merge_specs([right_spec, left_spec])
        sustain_spec = specs.BoundedArray(
            shape=(1,),
            dtype=hands_spec.dtype,
            minimum=[0.0],
            maximum=[1.0],
            name="sustain",
        )
        return spec_utils.merge_specs([hands_spec, sustain_spec])

    # Other.

    @property
    def midi(self) -> midi_file.MidiFile:
        return self._midi

    @property
    def reward_fn(self) -> composite_reward.CompositeReward:
        return self._reward_fn
    
    @property
    def task_name(self) -> str:
        return self._task_name

    # Helper methods.

    def _compute_forearm_reward(self, physics: mjcf.Physics) -> float:
        """Reward for not colliding the forearms."""
        if collision_utils.has_collision(
            physics,
            [g.full_identifier for g in self.right_hand.root_body.geom],
            [g.full_identifier for g in self.left_hand.root_body.geom],
        ):
            return 0.0
        return 0.5

    def _compute_sustain_reward(self, physics: mjcf.Physics) -> float:
        """Reward for pressing the sustain pedal at the right time."""
        del physics  # Unused.
        # return tolerance(
        #     self._goal_current[-1] - self.piano.sustain_activation[0],
        #     bounds=(0, _KEY_CLOSE_ENOUGH_TO_PRESSED),
        #     margin=(_KEY_CLOSE_ENOUGH_TO_PRESSED * 10),
        #     sigmoid="gaussian",
        # )
        return 0

    def _compute_energy_reward(self, physics: mjcf.Physics) -> float:
        """Reward for minimizing energy."""
        rew = 0.0
        for hand in [self.right_hand, self.left_hand]:
            power = hand.observables.actuators_power(physics).copy()
            rew -= self._energy_penalty_coef * np.sum(power)
        return rew

    def _compute_key_press_reward(self, physics: mjcf.Physics) -> float:
        """Reward for pressing the right keys at the right time."""
        del physics  # Unused.
        on = np.flatnonzero(self._goal_current[:-1])
        rew = 0.0
        # It's possible we have no keys to press at this timestep, so we need to check
        # that `on` is not empty.
        if on.size > 0:
            actual = np.array(self.piano.state / self.piano._qpos_range[:, 1])
            rews = tolerance(
                self._goal_current[:-1][on] - actual[on],
                bounds=(0, _KEY_CLOSE_ENOUGH_TO_PRESSED),
                margin=(_KEY_CLOSE_ENOUGH_TO_PRESSED * 10),
                sigmoid="gaussian",
            )
            rew += 0.5 * rews.mean()
        # If there are any false positives, the remaining 0.5 reward is lost.
        off = np.flatnonzero(1 - self._goal_current[:-1])
        rew += 0.5 * (1 - float(self.piano.activation[off].any()))
        return rew

    def _compute_fingering_reward(self, physics: mjcf.Physics) -> float:
        """Reward for minimizing the distance between the fingers and the keys."""

        def _distance_finger_to_key(
            hand_keys: List[Tuple[int, int]], hand
        ) -> List[float]:
            distances = []
            for key, mjcf_fingering in hand_keys:
                fingertip_site = hand.fingertip_sites[mjcf_fingering]
                fingertip_pos = physics.bind(fingertip_site).xpos.copy()
                key_geom = self.piano.keys[key].geom[0]
                key_geom_pos = physics.bind(key_geom).xpos.copy()
                key_geom_pos[-1] += 0.5 * physics.bind(key_geom).size[2]
                key_geom_pos[0] += 0.35 * physics.bind(key_geom).size[0]
                diff = key_geom_pos - fingertip_pos
                distances.append(float(np.linalg.norm(diff)))
            return distances

        distances = _distance_finger_to_key(self._rh_keys_current, self.right_hand)
        distances += _distance_finger_to_key(self._lh_keys_current, self.left_hand)

        # Case where there are no keys to press at this timestep.
        if not distances:
            return 0.0

        rews = tolerance(
            np.hstack(distances),
            bounds=(0, _FINGER_CLOSE_ENOUGH_TO_KEY),
            margin=(_FINGER_CLOSE_ENOUGH_TO_KEY * 10),
            sigmoid="gaussian",
        )
        # return float(np.mean(rews))
        return 0.0

    def _update_goal_state(self) -> None:
        # Observable callables get called after `after_step` but before
        # `should_terminate_episode`. Since we increment `self._t_idx` in `after_step`,
        # we need to guard against out of bounds indexing. Note that the goal state
        # does not matter at this point since we are terminating the episode and this
        # update is usually meant for the next timestep.
        if self._t_idx == len(self._notes):
            return

        self._goal_state = np.zeros(
            (self._n_steps_lookahead + 1, self.piano.n_keys + 1),
            dtype=np.float64,
        )
        t_start = self._t_idx
        t_end = min(t_start + self._n_steps_lookahead + 1, len(self._notes))
        for i, t in enumerate(range(t_start, t_end)):
            keys = [note.key for note in self._notes[t]]
            self._goal_state[i, keys] = 1.0
            self._goal_state[i, -1] = self._sustains[t]

    def _update_fingering_state(self) -> None:
        if self._t_idx == len(self._notes):
            return
        fingering = [note.fingering for note in self._notes[self._t_idx]]
        fingering_keys = [note.key for note in self._notes[self._t_idx]]
        # print("fingering", fingering) # from the rightest (right hand) to the leftest
        # print("keys", fingering_keys) # from the leftest to the rightest

        # Split fingering into right and left hand.
        self._rh_keys: List[Tuple[int, int]] = []
        self._lh_keys: List[Tuple[int, int]] = []
        for key, finger in enumerate(fingering):
            piano_key = fingering_keys[key]
            if finger < 5:
                self._rh_keys.append((piano_key, finger))
            else:
                self._lh_keys.append((piano_key, finger - 5))

        if not self._fingering_lookahead:
            # For each hand, set the finger to 1 if it is used and 0 otherwise.
            self._fingering_state = np.zeros((2, 5), dtype=np.float64)
            for hand, keys in enumerate([self._rh_keys, self._lh_keys]):
                for key, mjcf_fingering in keys:
                    self._fingering_state[hand, mjcf_fingering] = 1.0
            # print(self._fingering_state)
        else:
            self._fingering_state = np.zeros((self._n_steps_lookahead + 1, 10), dtype=np.float64)
            t_start = self._t_idx
            t_end = min(t_start + self._n_steps_lookahead + 1, len(self._notes))
            for i, t in enumerate(range(t_start, t_end)):
                fingers = [note.fingering for note in self._notes[t]]
                self._fingering_state[i, fingers] = 1.0
            # print(self._fingering_state)

    def _add_observables(self) -> None:
        # Enable hand observables.
        enabled_observables = [
            "joints_pos",
            # NOTE(kevin): This observable was previously enabled but it is redundant
            # since it is encoded in the joint positions, specifically via the forearm
            # slider joints (which are in units of meters).
            # "position",
        ]
        if self._enable_joint_vel_obs:
            enabled_observables += ["joints_vel"]
        for hand in [self.right_hand, self.left_hand]:
            for obs in enabled_observables:
                getattr(hand.observables, obs).enabled = True

        # This returns the current state of the piano keys.
        self.piano.observables.state.enabled = True
        self.piano.observables.sustain_state.enabled = True

        # This returns the goal state for the current timestep and n steps ahead.
        def _get_goal_state(physics) -> np.ndarray:
            del physics  # Unused.
            self._update_goal_state()
            return self._goal_state.ravel()

        goal_observable = observable.Generic(_get_goal_state)
        goal_observable.enabled = True
        self._task_observables = {"goal": goal_observable}

        # This adds fingering information for the current timestep.
        def _get_fingering_state(physics) -> np.ndarray:
            del physics  # Unused.
            self._update_fingering_state()
            return self._fingering_state.ravel()

        fingering_observable = observable.Generic(_get_fingering_state)
        fingering_observable.enabled = not self._disable_fingering_reward
        self._task_observables["fingering"] = fingering_observable

    def _colorize_fingertips(self) -> None:
        """Colorize the fingertips of the hands."""
        for hand in [self.right_hand, self.left_hand]:
            for i, body in enumerate(hand.fingertip_bodies):
                color = hand_consts.FINGERTIP_COLORS[i] + (_FINGERTIP_ALPHA,)
                for geom in body.find_all("geom"):
                    if geom.dclass.dclass == "plastic_visual":
                        geom.rgba = color
                # Also color the fingertip sites.
                hand.fingertip_sites[i].rgba = color

    def _colorize_keys_without_fingering(self, physics) -> None:
        """Colorize the correctly or wrongly pressed keys."""
        activation = self.piano.activation
        active_key_indices = np.where(activation == True)[0]
        key_should_pressed = []
        if len(self._rh_keys_current) > 0:
            key_should_pressed += [key for key, _ in self._rh_keys_current]
        if len(self._lh_keys_current) > 0:
            key_should_pressed += [key for key, _ in self._lh_keys_current]
        for key in active_key_indices:
            if key in key_should_pressed:
                # Correctly pressed keys
                key_geom = self.piano.keys[key].geom[0]
                # Green
                physics.bind(key_geom).rgba = (0.0, 1.0, 0.0, 1.0)
            else:
                # Wrongly pressed keys
                key_geom = self.piano.keys[key].geom[0]
                # Red
                physics.bind(key_geom).rgba = (1.0, 0.0, 0.0, 1.0)
        for key in key_should_pressed:
            if key not in active_key_indices:
                # Unpressed keys
                key_geom = self.piano.keys[key].geom[0]
                # Yellow
                physics.bind(key_geom).rgba = (1.0, 1.0, 0.0, 1.0)

    def _colorize_keys(self, physics) -> None:
        """Colorize the keys by the corresponding fingertip color."""
        for hand, keys in zip(
            [self.right_hand, self.left_hand],
            [self._rh_keys_current, self._lh_keys_current],
        ):
            for key, mjcf_fingering in keys:
                key_geom = self.piano.keys[key].geom[0]
                fingertip_site = hand.fingertip_sites[mjcf_fingering]
                if not self.piano.activation[key]:
                    physics.bind(key_geom).rgba = tuple(fingertip_site.rgba[:3]) + (
                        1.0,
                    )

    def _disable_collisions_between_hands(self) -> None:
        """Disable collisions between the hands."""
        for hand in [self.right_hand, self.left_hand]:
            for geom in hand.mjcf_model.find_all("geom"):
                # If both hands have the same contype and conaffinity, then they can't
                # collide. They can still collide with the piano since the piano has
                # contype 0 and conaffinity 1. Lastly, we make sure we're not changing
                # the contype and conaffinity of the hand geoms that are already
                # disabled (i.e., the visual geoms).
                commit_defaults(geom, ["contype", "conaffinity"])
                if geom.contype == 0 and geom.conaffinity == 0:
                    continue
                geom.conaffinity = 0
                geom.contype = 1

    def _randomize_initial_hand_positions(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        """Randomize the initial position of the hands."""
        if not self._randomize_hand_positions:
            return
        offset = random_state.uniform(low=-_POSITION_OFFSET, high=_POSITION_OFFSET)
        for hand in [self.right_hand, self.left_hand]:
            hand.shift_pose(physics, (0, offset, 0))