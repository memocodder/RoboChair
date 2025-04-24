# configs/environment/robochair_genesis_simple.py
from dataclasses import dataclass, field
from typing import Tuple, List, Literal, Optional, Dict
import math

from .base_environment_cfg import EnvBaseConfig, GridBaseConfig


@dataclass
class RobotInitStateConfig:
    """Конфигурация начального состояния робота."""

    base_pos: Tuple[float, float, float] = (0.0, 0.0, 0.1)
    base_quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    dof_pos: Optional[List[float]] = None
    randomize_yaw: bool = True
    random_yaw_range: Tuple[float, float] = (-math.pi, math.pi)


@dataclass
class RobotControlConfig:
    """Конфигурация параметров управления робота."""

    wheel_scale: Tuple[float, float, float, float] = (30.0, -30.0, -30.0, 30.0)
    turn_scale: Tuple[float, float] = (1.0, 1.0)
    head_control_scale: Tuple[float] = (3.1,)
    eye_control_scale: Tuple[float] = (1.0,)
    clip_actions: float = 1.0


@dataclass
class RobotPhysicsConfig:
    """Конфигурация физических параметров и терминирования робота."""

    # Списки Kp/Kd для каждого DoF (порядок важен!)
    kp_values: List[float] = field(
        default_factory=lambda: [
            80.0,
            80.0,
            80.0,
            80.0,  # Колеса
            50.0,
            50.0,  # Поворот
            10.0,  # Глаз (индекс 6)
            20.0,  # Голова (индекс 7)
        ]
    )
    kd_values: List[float] = field(
        default_factory=lambda: [
            2.0,
            2.0,
            2.0,
            2.0,  # Колеса
            1.0,
            1.0,  # Поворот
            0.2,  # Глаз
            0.5,  # Голова
        ]
    )
    termination_roll_deg: float = 45.0
    termination_pitch_deg: float = 45.0


@dataclass
class RobotInstanceConfig:
    """
    Конфигурация экземпляра робота для использования в конкретной среде.
    Определяет начальное состояние, управление и физику, ссылаясь
    на структурное определение робота (RobotDefinitionConfig) по имени.
    (Предполагается, что RobotDefinitionConfig загружается отдельно по имени).
    """

    definition_name: str = (
        "RobochairV1"  # Имя определения робота из configs/robots/
    )

    init_state: RobotInitStateConfig = field(
        default_factory=RobotInitStateConfig
    )
    control: RobotControlConfig = field(default_factory=RobotControlConfig)
    physics: RobotPhysicsConfig = field(default_factory=RobotPhysicsConfig)


@dataclass
class GenesisObservationConfig:
    """Конфигурация масштабирования компонентов наблюдений."""

    base_ang_vel_scale: float = 0.1
    base_lin_vel_scale: float = 0.1
    projected_gravity_scale: float = 0.8
    wheel_vel_scale: float = 0.01


@dataclass
class GenesisCorridorConfig:
    arches_using: bool = False


@dataclass
class GoalConfig:
    """Конфигурация для компонента цели (Goal)."""

    # Начальная позиция цели [X, Y, Z] (используется до первого сброса)
    init_pos: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    init_quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # --- Параметры для сброса/перегенерации позиции цели ---
    # Минимальная дистанция от центра (0,0) по X/Y при сбросе
    reset_min_distance: float = 15.0
    # Максимальная дистанция от центра (0,0) по X/Y при сбросе
    reset_max_distance: float = 25.0
    # Высота Z, на которую устанавливается цель при сбросе.
    reset_height_z: float = 1.0

    def __post_init__(self):
        """Проверка корректности параметров."""
        if self.reset_min_distance < 0 or self.reset_max_distance <= 0:
            raise ValueError(
                "GoalConfig Error: reset_min_distance и reset_max_distance должны быть положительными."
            )
        if self.reset_min_distance >= self.reset_max_distance:
            raise ValueError(
                "GoalConfig Error: reset_min_distance должен быть меньше reset_max_distance."
            )


# --- Основной Датакласс Конфигурации Среды Genesis ---


@dataclass
class GenesisEnvConfig(EnvBaseConfig):
    """Конфигурация для среды Robochair, использующей бэкенд Genesis."""

    backend: str = "genesis"

    episode_length_s: float = 20.0
    resampling_time_s: float = 20.0
    goal_resample_distance_threshold: float = 1.3
    simulate_action_latency: bool = True

    robot: RobotInstanceConfig = field(default_factory=RobotInstanceConfig)
    corridor: GenesisCorridorConfig = field(
        default_factory=GenesisCorridorConfig
    )
    goal: GoalConfig = field(default_factory=GoalConfig)

    main_grid: GridBaseConfig = field(
        default_factory=lambda: GridBaseConfig(
            enabled=True,
            cells=(39, 39, 1),
            channels=["occupancy"],
        )
    )

    reward_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "action_rate": -0.0001,
            "goal_dist": -0.1,
            "relative_angle": -100.0,
            "speed_towards_goal": 100.0,
            "collision": -500.0,
        }
    )

    observation_config: GenesisObservationConfig = field(
        default_factory=GenesisObservationConfig
    )

    control_frequency: int = 50
    simulation_frequency: int = 100

    def __post_init__(self):
        super().__post_init__()

        if "main_grid" not in self.grids:
            self.grids["main_grid"] = self.main_grid
        if hasattr(self, "main_grid"):
            delattr(self, "main_grid")
