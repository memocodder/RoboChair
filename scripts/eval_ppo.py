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
from robochair.algorithms.ppo.on_policy_runner import OnPolicyRunner

from configs.experiments.genesis_ppo_simple_cfg import config


import matplotlib.pyplot as plt


# %%

gs.init(theme="light", logging_level="warning")

# %%

NUM_ENVS = 1

env = Env(
    config.env,
    num_envs=NUM_ENVS,
    show_viewer=True,
    # gta_cam=True,
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



obs, _ = env.reset()
with torch.no_grad():
    for i in range(2 * 1000):
        actions = policy(obs)
        obs, _, rews, dones, infos = env.step(actions)
