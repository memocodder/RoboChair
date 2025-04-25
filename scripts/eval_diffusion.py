# %%
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
current_dir_name = os.path.basename(script_dir)

if current_dir_name == "scripts":
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
else:
    project_root = current_dir_name

print(f"project_root: {project_root}")


from configs.log_config import LoggingConfig, setup_logger

log_config = LoggingConfig(
    base_log_dir=project_root + "/results",
    experiment_name="RobochairGenesisDiffusionEval_Simple_v1",
    experiment_description="Провека результатов обучения диффузионной политики на экспертных записях PPO.",
    log_level="INFO",
    overwrite_existing=True,
)

logger = setup_logger(log_config)
experiment_path = log_config.experiment_dir


from robochair.environments import Env
from robochair.algorithms.diffusion import DiffusionRunner

from configs.genesis_config import GenesisEnvConfig
from configs.diffusion_config import DiffusionAlgoConfig, DiffusionRunnerConfig

# %%

algo_configuration = DiffusionAlgoConfig(
    runner=DiffusionRunnerConfig(
        checkpoint_dir=project_root + "/trained_models"
    )
)
env_configuration = GenesisEnvConfig()

env = Env(
    env_configuration,
    num_envs=1,
    log_level=log_config.log_level,
    show_viewer=True,
    gta_cam=True,
)

runner = DiffusionRunner(algo_configuration, logger)

runner.load_checkpoint()

# %%

import torch
from robochair.models.diffusion import DiffusionPolicy

policy: DiffusionPolicy = runner.get_inference_policy()


def split_img_and_obs(obs_img):
    image_size = 39 * 39
    image_shape = (1, 39, 39)
    batch_size = obs_img.shape[0]

    image = obs_img[:, -image_size:].view(batch_size, *image_shape)
    obs = obs_img[:, :-image_size]
    return image, obs


def to_diff_type(obs, grid):
    return {
        "observation.state": obs,
        "observation.images": grid,
    }


with torch.no_grad():
    obs, _ = env.reset()
    grid, obs_ = split_img_and_obs(obs)
    print(f"grid.shape = {grid.shape}")
    print(f"obs_.shape = {obs_.shape}")

    for i in range(10):
        actions = policy.select_action(to_diff_type(obs_, grid))
        obs, _, rews, dones, infos = env.step(actions)
        grid, obs_ = split_img_and_obs(obs)

    # %%
    obs, _ = env.reset()
    grid, obs_ = split_img_and_obs(obs)

    for i in range(1000):
        actions = policy.select_action(to_diff_type(obs_, grid))
        obs, _, rews, dones, infos = env.step(actions)
        grid, obs_ = split_img_and_obs(obs)
