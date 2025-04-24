# configs/experiments/robochair_genesis_ppo_baseline.py
from .base_experiment_cfg import ExperimentConfig, LoggingConfig

from ..environments.genesis_simple_cfg import GenesisEnvConfig
from ..algorithms.ppo_algo_cfg import PPOAlgoConfig


# --- Конфигурация Среды ---
# Используем дефолтную конфигурацию простой среды Genesis
env_configuration = GenesisEnvConfig()

# --- Конфигурация Алгоритма PPO (включая политику и раннер) ---
# Используем дефолтную конфигурацию PPO, но можем переопределить нужное
algo_configuration = PPOAlgoConfig()
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
#     # --- Переопределение параметров Раннера ---
#     runner=PPORunnerConfig(
#         seed=123,  # Используем другой сид для этого запуска
#         total_timesteps=30_000_000,  # Обучаем дольше
#         log_interval_iters=10,  # Логируем чаще
#         save_interval_iters=250,  # Сохраняем реже
#     ),
# )

# --- Конфигурация Логирования ---
logging_configuration = LoggingConfig(log_to_wandb=False, log_to_file=False)

# --- Финальный объект конфигурации эксперимента ---
config = ExperimentConfig(
    experiment_name="RobochairGenesisPPO_Simple_v1",
    seed=123,
    env=env_configuration,
    algo=algo_configuration,
    logging=logging_configuration,
)
