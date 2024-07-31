from wandb.integration.sb3 import WandbCallback
from typing import Optional
from typing import Literal
import wandb
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

def prefix_dict(prefix: str, d: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in d.items()}

class LoggingCallback(WandbCallback):
    """Custom callback for logging data to wandb."""

    def __init__(
        self,
        verbose: int = 0,
        model_save_path: Optional[str] = None,
        model_save_freq: int = 0,
        gradient_save_freq: int = 0,
        log: Optional[Literal["gradients", "parameters", "all"]] = "all",
        logging_freq: int = 1000,
    ):
        super().__init__(
            verbose,
            model_save_path,
            model_save_freq,
            gradient_save_freq,
            log,
        )
        self._logging_freq = logging_freq
        self._num_episodes = None

    def on_rollout_end(self) -> None:
        if self._num_episodes is None:
            self._num_episodes = [0] * len(self.training_env.envs) # Number of episodes logged so far
        if self.num_timesteps % self._logging_freq == 0:
            for idx, env in enumerate(self.training_env.envs):
                for i in range(self._num_episodes[idx], len(env.get_episode_rewards())):
                    self._num_episodes[idx] += 1
                    wandb.log({"train/reward": env.get_episode_lengths()[i]})