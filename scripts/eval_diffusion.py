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
from robochair.algorithms.diffusion import (
    modeling_diffusion,
    configuration_diffusion,
)
from robochair.algorithms.diffusion.utils import load_ckpt


from configs.experiments.genesis_ppo_simple_cfg import config


# %%
checkpoint_path = project_root + "/results/checkpoints_diffusion"
device = "cuda"
cfg = configuration_diffusion.DiffusionConfig()
policy = modeling_diffusion.DiffusionPolicy(cfg).to(device)
_ = load_ckpt(
    policy,
    torch.optim.Adam(policy.get_optim_params(), lr=1e-4),
    checkpoint_path,
    device=device,
)
policy.eval()

# %%

gs.init(theme="light", logging_level="warning")

NUM_ENVS = 1

env = Env(
    config.env,
    num_envs=NUM_ENVS,
    show_viewer=True,
    gta_cam=True,
)


# %%


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


obs, _ = env.reset()
grid, obs_ = split_img_and_obs(obs)
print(f"grid.shape = {grid.shape}")
print(f"obs_.shape = {obs_.shape}")
with torch.inference_mode():
    for i in range(1 * 1000):
        actions = policy.select_action(to_diff_type(obs_, grid))
        obs, _, rews, dones, infos = env.step(actions)
        grid, obs_ = split_img_and_obs(obs)
