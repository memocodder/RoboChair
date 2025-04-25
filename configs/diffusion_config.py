# configs/algorithm/ppo.py
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Literal
from .device_config import DeviceConfig


@dataclass
class VisionEncoderConfig:
    """Конфигурация визуального энкодера."""

    output_dim: int = 128


@dataclass
class UNetConfig:
    """Конфигурация U-Net."""

    # Размерности признаков на этапах downsampling
    down_dims: Tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5  # Размер ядра свертки
    n_groups: int = 8  # Кол-во групп для GroupNorm в сверточных блоках
    # Размерность эмбеддинга для временного шага диффузии
    diffusion_step_embed_dim: int = 128
    # Использовать ли модуляцию масштаба в FiLM (в дополнение к смещению)
    use_film_scale_modulation: bool = True


@dataclass
class NoiseSchedulerConfig:
    """Конфигурация планировщика шума диффузии."""

    scheduler_type: Literal["DDPM", "DDIM"] = "DDPM"  # Тип планировщика
    num_train_timesteps: int = (
        100  # Кол-во шагов диффузии для обучения (прямой процесс)
    )
    beta_schedule: str = (
        "squaredcos_cap_v2"  # Тип расписания беты (из diffusers)
    )
    beta_start: float = 0.0001  # Начальное значение беты
    beta_end: float = 0.02  # Конечное значение беты
    prediction_type: Literal["epsilon", "sample"] = (
        "epsilon"  # Что предсказывает U-Net ('epsilon' обычно лучше)
    )
    clip_sample: bool = True  # Обрезать ли предсказания на каждом шаге вывода
    clip_sample_range: float = 1.0  # Диапазон обрезки [-range, +range]


@dataclass
class TrainingConfig:
    """Конфигурация оптимизатора и планировщика скорости обучения."""

    # Оптимизатор AdamW
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.95, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-6
    # Планировщик скорости обучения
    scheduler_name: Literal["cosine", "linear", "constant"] = (
        "cosine"  # Тип планировщика
    )
    warmup_steps: int = 500  # Кол-во шагов "прогрева" для планировщика


@dataclass
class DiffusionPolicyConfig:
    """
    Конфигурация для DiffusionPolicy.
    Группирует параметры по назначению.
    """

    name: str = "DiffusionPolicy"  # Просто имя для идентификации

    # --- Структура входов/выходов и временные параметры ---
    n_obs_steps: int = (
        2  # Кол-во шагов наблюдений (включая текущий) подаваемых на вход
    )
    horizon: int = 16  # Горизонт предсказания действий диффузионной модели
    n_action_steps: int = (
        8  # Кол-во шагов действий, выполняемых за один вызов политики
    )

    num_obs = 20
    num_actions = 4
    # input_shapes: Dict[str, List[int]] = field(
    #     default_factory=lambda: {
    #         "observation.image_primary": [3, 96, 96],  # Пример: одна камера
    #         "observation.state": [10],  # Пример: проприоцептивные данные
    #     }
    # )
    # output_shapes: Dict[str, List[int]] = field(
    #     default_factory=lambda: {"action": [7]}  # Пример: 7-мерное действие
    # )

    # --- Параметры архитектуры ---
    vision_encoder: VisionEncoderConfig = field(
        default_factory=VisionEncoderConfig
    )
    unet: UNetConfig = field(default_factory=UNetConfig)
    noise_scheduler: NoiseSchedulerConfig = field(
        default_factory=NoiseSchedulerConfig
    )

    # --- Параметры Вывода (Inference) ---
    num_inference_steps: Optional[int] = (
        None  # Кол-во шагов обратной диффузии при выводе (None = num_train_timesteps)
    )

    # --- Параметры Обучения ---
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # Нужно ли маскировать лосс для действий, добавленных копированием (padding)
    do_mask_loss_for_padding: bool = False
    # Пропускать последние N кадров при сборе траекторий для обучения
    # (Оригинальная реализация DiffusionPolicy делает так для избежания излишнего padding)
    drop_n_last_frames: int = (
        7  # Рассчитывается как: horizon - n_action_steps - n_obs_steps + 1
    )


@dataclass
class DiffusionRunnerConfig:
    """Параметры запуска и процесса обучения/оценки для Diffusion Policy."""

    seed: int = 42
    # Определение длительности обучения (выберите один или несколько)
    total_train_steps: Optional[int] = 1_000_000  # Общее кол-во шагов градиента
    total_train_epochs: Optional[int] = None  # Или общее кол-во эпох
    # Размер батча для обучения
    batch_size: int = 256

    # Параметры логирования
    log_interval_steps: int = 100  # Интервал логирования (в шагах градиента)

    # Параметры сохранения чекпоинтов
    save_interval_steps: int = 10000
    checkpoint_dir: str = "checkpoints_diffusion"
    max_checkpoints_to_keep: int = (
        3  # Сколько последних чекпоинтов хранить (-1 = все)
    )

    # Параметры оценки (evaluation)
    eval_interval_steps: Optional[int] = (
        5000  # Интервал оценки (-1 или None = не оценивать)
    )
    eval_episodes: int = 10  # Количество эпизодов для оценки

    # Загрузка чекпоинта для возобновления или старта
    load_checkpoint_path: Optional[str] = (
        None  # Путь к конкретному файлу чекпоинта
    )
    resume_training: bool = (
        False  # Возобновлять ли обучение с чекпоинта (включая состояние оптимизатора)
    )


@dataclass
class DiffusionAlgoConfig:
    """
    Полная конфигурация для алгоритма Diffusion Policy.
    Включает конфигурацию политики, параметры запуска/обучения и устройства.
    """

    algo_name: str = "Diffusion"

    device_settings: DeviceConfig = field(default_factory=DeviceConfig)

    policy: DiffusionPolicyConfig = field(default_factory=DiffusionPolicyConfig)
    runner: DiffusionRunnerConfig = field(default_factory=DiffusionRunnerConfig)

    inference_weights_path: Optional[str] = field(
        default=None,
        metadata=dict(
            help="Путь к файлу .pt или .safetensors с весами модели для загрузки перед началом (инференс, оценка, старт дообучения)."
        ),
    )

    @property
    def device(self) -> str:
        """Удобный доступ к строке устройства."""
        return self.device_settings.device
