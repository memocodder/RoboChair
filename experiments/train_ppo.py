# %%

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
current_dir_name = os.path.basename(script_dir)

if current_dir_name == "experiments":
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
    experiment_name="RobochairGenesisPPO_Simple_v1",
    experiment_description="Обучение следовать цели в простом широком коридоре.",
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

# algo_configuration = PPOAlgoConfig(
#     # --- Переопределение гиперпараметров PPO ---
#     learning_rate=5e-4,  # Пример: немного снизили LR
#     entropy_coef=0.005,  # Пример: уменьшили энтропию
#     # --- Переопределение параметров Политики ---
#     policy=PPOPolicyConfig(
#         # Например, хотим другую активацию или размер сети
#         activation="relu",
#         actor_hidden_dims=[512, 256],  # Сделали сеть чуть больше/другой
#         critic_hidden_dims=[512, 256],
#         # init_noise_std и энкодеры остаются по умолчанию
#     ),
# )

env = Env(env_configuration, num_envs=1024 * 1, log_level=log_config.log_level)

runner = OnPolicyRunner(
    env,
    algo_configuration,
    log_dir=experiment_path,
)

# %%

runner.load(path=project_root + "/trained_models/ppo.pt")

# %%
runner.learn(num_learning_iterations=200, init_at_random_ep_len=True)
