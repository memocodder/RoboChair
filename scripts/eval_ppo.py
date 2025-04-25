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
    experiment_name="RobochairGenesisPPOEval_Simple_v1",
    experiment_description="Инференс, следовать цели в простом широком коридоре.",
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

algo_configuration = PPOAlgoConfig()
env_configuration = GenesisEnvConfig()

env = Env(
    env_configuration,
    num_envs=1,
    log_level=log_config.log_level,
    show_viewer=True,
    gta_cam=True,
)

runner = OnPolicyRunner(
    env,
    algo_configuration,
    log_dir=experiment_path,
)


runner.load(project_root + "/trained_models/ppo.pt")

policy = runner.get_inference_policy(device="cuda")

# %%
import torch

obs, _ = env.reset()
with torch.no_grad():
    for i in range(20 * 1000):
        actions = policy(obs)
        obs, _, rews, dones, infos = env.step(actions)
