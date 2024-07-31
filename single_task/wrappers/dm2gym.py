import gymnasium as gym
from gymnasium import spaces

from dm_control import suite
from dm_env import specs
import dm_env

def convert_dm_control_to_gym_space(dm_control_space):
    r"""Convert dm_control space to gym space. """
    if isinstance(dm_control_space, specs.BoundedArray):
        try:
            space = spaces.Box(low=dm_control_space.minimum, 
                            high=dm_control_space.maximum, 
                            shape=dm_control_space.shape,
                            dtype=dm_control_space.dtype)
        except:
            space = spaces.Box(low=-float('inf'), 
                                high=float('inf'), 
                                shape=dm_control_space.shape,
                                dtype=dm_control_space.dtype)
        assert space.shape == dm_control_space.shape
        return space
    elif isinstance(dm_control_space, specs.Array) and not isinstance(dm_control_space, specs.BoundedArray):
        space = spaces.Box(low=-float('inf'), 
                           high=float('inf'), 
                           shape=dm_control_space.shape, 
                           dtype=dm_control_space.dtype)
        return space
    elif isinstance(dm_control_space, dict):
        space = spaces.Dict({key: convert_dm_control_to_gym_space(value)
                             for key, value in dm_control_space.items()})
        return space


class Dm2GymWrapper(gym.Env):
    def __init__(self, environment:dm_env.Environment):
        self.env = environment  
        self.metadata = {'render.modes': ['human', 'rgb_array'],
                         'video.frames_per_second': round(1.0/self.env.control_timestep())}
        self.observation_space = convert_dm_control_to_gym_space(self.env.observation_spec())
        self.action_space = convert_dm_control_to_gym_space(self.env.action_spec())
        self.viewer = None
        self.physics = self.env.physics
        self.task = self.env.task
    
    def seed(self, seed):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
    
    def step(self, action):
        timestep = self.env.step(action)
        observation = timestep.observation
        reward = timestep.reward
        done = timestep.last()
        info = {}
        return observation, reward, done, False, info
    
    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        timestep = self.env.reset()
        return timestep.observation, None
    
    def render(self, mode='human', **kwargs):
        if 'camera_id' not in kwargs:
            kwargs['camera_id'] = 0  # Tracking camera
        use_opencv_renderer = kwargs.pop('use_opencv_renderer', False)
        
        img = self.env.physics.render(**kwargs)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self.env.close()
    
    