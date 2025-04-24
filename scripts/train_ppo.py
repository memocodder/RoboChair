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


# %%
import genesis as gs

from robochair.environments import Env
from robochair.algorithms.ppo.on_policy_runner import OnPolicyRunner

from configs.experiments.genesis_ppo_simple_cfg import config


# %%

gs.init(
    theme="light", logging_level="warning"
)  # logging_level=config.logging.level


env = Env(config.env, num_envs=1024 * 1)

# %%

runner = OnPolicyRunner(
    env,
    config.algo,
    # log_dir=config.logging.log_dir,
    log_dir=project_root + "/results/ppo_train",
)

# %%

runner.load(path=project_root + "/results/checkpoints_ppo/model_wall_great.pt")

# %%
runner.learn(num_learning_iterations=200, init_at_random_ep_len=True)

# %%
