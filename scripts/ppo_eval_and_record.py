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

# обязательно реализовать релоауды

# %%

from configs.log_config import LoggingConfig, setup_logger

log_config = LoggingConfig(
    base_log_dir=project_root + "/results",
    experiment_name="RobochairGenesisPPO_Eval_100_Simple_v1",
    experiment_description="Инференс и запись, следовать цели в простом широком коридоре.",
    log_level="WARNING",
    overwrite_existing=True,
)

logger = setup_logger(log_config)
experiment_path = log_config.experiment_dir


# %%

from robochair.environments import Env
from robochair.algorithms.ppo.on_policy_runner import OnPolicyRunner

from configs.genesis_config import GenesisEnvConfig
from configs.ppo_config import PPOAlgoConfig

from robochair.data_handling.recorder import EpisodeRecorder

algo_configuration = PPOAlgoConfig()
env_configuration = GenesisEnvConfig()


NUM_ENVS = 100

env = Env(
    env_configuration,
    num_envs=NUM_ENVS,
    show_viewer=True,
    log_level=log_config.log_level
)

recorders = {}
for i in range(NUM_ENVS):
    recorders[i] = EpisodeRecorder(
        root_dir=project_root + "/data", episodes_dir="episodes_ppo"
    )

# %%

runner = OnPolicyRunner(
    env,
    algo_configuration,
    log_dir=experiment_path,
)

runner.load(project_root + "/trained_models/ppo.pt")

policy = runner.get_inference_policy(device="cuda")

# %%

import torch

def split_img_and_obs(obs_img):
    image_size = 39 * 39
    image_shape = (1, 39, 39)
    batch_size = obs_img.shape[0]

    image = obs_img[:, -image_size:].view(batch_size, *image_shape)
    obs = obs_img[:, :-image_size]
    return image, obs


obs, _ = env.reset()
with torch.no_grad():
    for i in range(4 * 1000 + 20):
        actions = policy(obs)
        obs, _, rews, dones, infos = env.step(actions)

        grid, obs_ = split_img_and_obs(obs)

        for env_idx in range(NUM_ENVS):
            current_recorder = recorders[env_idx]
            current_recorder.add_step(
                obs_[env_idx], grid[env_idx], rews[env_idx], actions[env_idx]
            )
            if dones[env_idx] == 1:
                save_path = current_recorder.save_episode()
