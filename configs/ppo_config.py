# configs/algorithm/ppo.py
from dataclasses import dataclass, field
from typing import List, Optional, Literal

from .device_config import DeviceConfig

# --- Вспомогательные Конфиги для PPO ---


@dataclass
class EncoderConfig:
    """Конфигурация для одного энкодера (Vision или Observation)."""

    type: str = "Default"
    output_dim: int = 128


@dataclass
class PPOPolicyConfig:
    """Конфигурация нейронной сети (политики) для PPO."""

    activation: Literal["elu", "relu", "tanh", "leaky_relu"] = "leaky_relu"
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    init_noise_std: float = 1.0
    vision_encoder: EncoderConfig = field(
        default_factory=lambda: EncoderConfig(type="VisEncoder")
    )
    observation_encoder: EncoderConfig = field(
        default_factory=lambda: EncoderConfig(type="ObsEncoder")
    )


@dataclass
class PPORunnerConfig:
    """Параметры запуска и процесса обучения, специфичные для PPO."""

    seed: int = 42
    total_timesteps: int = 20_000_000
    num_steps_per_env: int = 48  # Шагов на среду за итерацию сбора данных
    # buffer_size вычисляется = num_envs * num_steps_per_env
    # max_iterations вычисляется = total_timesteps / buffer_size
    max_iterations = 1000

    # Интервалы в ИТЕРАЦИЯХ PPO
    log_interval_iters: int = 20
    save_interval_iters: int = 100
    eval_interval_iters: int = 200  # -1 если не нужно
    eval_episodes: int = 10
    checkpoint_dir: str = "checkpoints"  # Относительная папка для чекпоинтов

    # Загрузка чекпоинта
    load_run_id: Optional[str] = None
    load_checkpoint_iter: int = -1
    resume_training: bool = False


# --- Основной Конфиг PPO (Обертка) ---


@dataclass
class PPOAlgoConfig:
    """
    Полная конфигурация для алгоритма PPO.
    Включает гиперпараметры, конфигурацию политики и параметры запуска/обучения (раннер).
    """

    name: str = "PPO"

    device_settings: DeviceConfig = field(default_factory=DeviceConfig)

    # --- Гиперпараметры PPO ---
    learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_loss_coef: float = 1.0
    max_grad_norm: float = 1.0
    clip_param: float = 0.2
    ppo_epoch: int = 5
    num_mini_batches: int = 16
    use_clipped_value_loss: bool = True
    schedule: Literal["adaptive", "fixed", "linear"] = "adaptive"
    desired_kl: Optional[float] = 0.01

    # --- Вложенные Конфигурации ---
    policy: PPOPolicyConfig = field(default_factory=PPOPolicyConfig)
    runner: PPORunnerConfig = field(default_factory=PPORunnerConfig)

    inference_weights_path: Optional[str] = field(
        default=None,
        metadata=dict(
            help="Путь к файлу .pt или .safetensors с весами модели для загрузки перед началом (инференс, оценка, старт дообучения). Имеет приоритет над runner.load_run_id для загрузки весов."
        ),
    )

    @property
    def device(self) -> str:
        return self.device_settings.device
