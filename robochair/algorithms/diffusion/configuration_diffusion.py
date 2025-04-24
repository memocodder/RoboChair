from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamConfig
from lerobot.common.optim.schedulers import DiffuserSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("diffusion-robo-chair")
@dataclass
class DiffusionConfig(PreTrainedConfig):
    """Класс конфигурации для DiffusionPolicy.

    Значения по умолчанию настроены для обучения с PushT, предоставляющим проприоцептивные наблюдения и наблюдения с одной камеры.

    Параметры, которые вам, скорее всего, потребуется изменить, — это те, которые зависят от среды / сенсоров.
    Это: `input_shapes` и `output_shapes`.

    Примечания к входам и выходам:
        - "observation.state" требуется как ключ входа.
        - Либо:
            - Требуется хотя бы один ключ, начинающийся с "observation.image", как вход.
              И/ИЛИ
            - Ключ "observation.environment_state" требуется как вход.
        - Если есть несколько ключей, начинающихся с "observation.image", они обрабатываются как несколько
          видов с камер. В настоящее время поддерживаются только изображения одинаковой формы.
        - "action" требуется как ключ выхода.

    Args:
        n_obs_steps: Количество шагов среды, наблюдения за которые передаются политике (включает
            текущий шаг и предыдущие шаги).
        horizon: Размер предсказания действий диффузионной модели, как описано в `DiffusionPolicy.select_action`.
        n_action_steps: Количество шагов действий, выполняемых в среде за один вызов политики.
            См. `DiffusionPolicy.select_action` для получения дополнительной информации.
        input_shapes: Словарь, определяющий формы входных данных для политики. Ключ представляет
            имя входных данных, а значение — это список, указывающий размерности соответствующих данных.
            Например, "observation.image" относится к входу с камеры с размерностями [3, 96, 96],
            указывая, что он имеет три цветовых канала и разрешение 96x96. Важно отметить, что `input_shapes` не
            включает измерение батча или временное измерение.
        output_shapes: Словарь, определяющий формы выходных данных для политики. Ключ представляет
            имя выходных данных, а значение — это список, указывающий размерности соответствующих данных.
            Например, "action" относится к выходной форме [14], указывая на 14-мерные действия.
            Важно отметить, что `output_shapes` не включает измерение батча или временное измерение.
        input_normalization_modes: Словарь, где ключ представляет модальность (например, "observation.state"),
            а значение указывает режим нормализации, который нужно применить. Два доступных режима: "mean_std",
            который вычитает среднее и делит на стандартное отклонение, и "min_max", который масштабирует данные
            в диапазон [-1, 1].
        output_normalization_modes: Аналогичный словарь, как `normalize_input_modes`, но для денормализации
            к исходному масштабу. Обратите внимание, что это также используется для нормализации целей обучения.
        vision_backbone: Название базовой архитектуры (backbone) resnet из torchvision, используемой для кодирования изображений.
        crop_shape: Форма (H, W), до которой обрезаются изображения на этапе предварительной обработки для
            визуальной базовой архитектуры. Должна умещаться в размер изображения. Если None, обрезка не выполняется.
        crop_is_random: Определяет, должна ли обрезка быть случайной во время обучения (в режиме оценки это
            всегда обрезка по центру).
        pretrained_backbone_weights: Предварительно обученные веса из torchvision для инициализации базовой
            архитектуры. `None` означает отсутствие предварительно обученных весов.
        use_group_norm: Заменять ли батч-нормализацию (batch normalization) групповой нормализацией (group normalization) в базовой
            архитектуре. Размеры групп устанавливаются примерно равными 16 (точнее, feature_dim // 16).
        spatial_softmax_num_keypoints: Количество ключевых точек для SpatialSoftmax.
        use_separate_rgb_encoders_per_camera: Использовать ли отдельный RGB-кодировщик для каждого вида с камеры.
        down_dims: Размерность признаков для каждого этапа временного понижения разрешения (downsampling) в U-Net
            диффузионной модели. Вы можете предоставить переменное количество размерностей, тем самым
            контролируя степень понижения разрешения.
        kernel_size: Размер сверточного ядра U-Net диффузионной модели.
        n_groups: Количество групп, используемых в групповой нормализации (group norm) сверточных блоков U-Net.
        diffusion_step_embed_dim: U-Net обусловлена временным шагом диффузии через небольшую нелинейную
            сеть. Это выходная размерность этой сети, т. е. размерность вложения (embedding dimension).
        use_film_scale_modulation: FiLM (https://arxiv.org/abs/1709.07871) используется для обусловливания U-Net.
            Модуляция смещения (bias modulation) используется по умолчанию, а этот параметр указывает, следует ли
            также использовать модуляцию масштаба (scale modulation).
        noise_scheduler_type: Название планировщика шума для использования. Поддерживаемые опции: ["DDPM", "DDIM"].
        num_train_timesteps: Количество шагов диффузии для прямого расписания диффузии (forward diffusion schedule).
        beta_schedule: Название расписания бета диффузии согласно DDPMScheduler из Hugging Face diffusers.
        beta_start: Значение бета для первого шага прямой диффузии.
        beta_end: Значение бета для последнего шага прямой диффузии.
        prediction_type: Тип предсказания, которое делает U-Net диффузионной модели. Выберите из "epsilon"
            или "sample". С точки зрения моделирования латентных переменных они приводят к эквивалентным
            результатам, но показано, что "epsilon" лучше работает во многих настройках глубоких нейронных сетей.
        clip_sample: Обрезать ли сэмпл до [-`clip_sample_range`, +`clip_sample_range`] для каждого шага
            шумоподавления (denoising) во время инференса. ВНИМАНИЕ: вам нужно будет убедиться, что ваше
            пространство действий нормализовано, чтобы соответствовать этому диапазону.
        clip_sample_range: Величина диапазона обрезки, как описано выше.
        num_inference_steps: Количество шагов обратной диффузии, используемых во время инференса (шаги
            равномерно распределены). Если не указано, по умолчанию совпадает с `num_train_timesteps`.
        do_mask_loss_for_padding: Маскировать ли потери, когда есть действия, дополненные копированием (copy-padded).
            См. `LeRobotDataset` и `load_previous_and_future_frames` для получения дополнительной информации.
            Обратите внимание, что по умолчанию установлено значение False, так как исходная реализация Diffusion Policy делает то же самое.
    """

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Оригинальная реализация не сэмплирует кадры для последних 7 шагов,
    # что позволяет избежать избыточного дополнения (padding) и приводит к улучшению результатов обучения.
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1


    num_actions: int = 4
    num_obs: int = 20

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    # Unet.
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    # Noise scheduler.
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Inference
    num_inference_steps: int | None = None

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )

        # Check that the horizon size and U-Net downsampling is compatible.
        # U-Net downsamples by 2 with each stage.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for "
                        f"`{key}`."
                    )

        # Check that all input images have the same shape.
        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
