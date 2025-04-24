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

# %%

import torch
import genesis as gs

from robochair.environments import Env
from robochair.environments.agent_control import AgentControl
from robochair.data_handling.recorder import EpisodeRecorder
from robochair.algorithms.ppo.on_policy_runner import OnPolicyRunner

from configs.experiments.genesis_ppo_simple_cfg import config


import matplotlib.pyplot as plt


# %%

gs.init(theme="light", logging_level="warning")

# %%

NUM_ENVS = 100

env = Env(
    config.env,
    num_envs=NUM_ENVS,
    show_viewer=True,
)

recorders = {}
for i in range(NUM_ENVS):
    recorders[i] = EpisodeRecorder(
        root_dir=project_root + "/data", episodes_dir="episodes_ppo"
    )

# %%

runner = OnPolicyRunner(
    env,
    config.algo,
    # log_dir=config.logging.log_dir,
    log_dir=project_root + "/results/ppo_eval",
)

runner.load(project_root + "/results/checkpoints_ppo/model_wall_great.pt")

policy = runner.get_inference_policy(device="cuda")

# %%


def split_img_and_obs(obs_img):
    image_size = 39 * 39
    image_shape = (1, 39, 39)
    batch_size = obs_img.shape[0]

    image = obs_img[:, -image_size:].view(batch_size, *image_shape)
    obs = obs_img[:, :-image_size]
    return image, obs


obs, _ = env.reset()
with torch.no_grad():
    for i in range(2 * 1000):
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
