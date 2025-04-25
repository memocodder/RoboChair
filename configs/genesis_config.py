# configs/environment/robochair_genesis_simple.py
from dataclasses import dataclass, field
from typing import Tuple, List, Literal, Optional, Dict
import math

# from .grid_config import GridBaseConfig
from .device_config import DeviceConfig



@dataclass
class GenesisEnvConfig:
    """Конфигурация для среды Robochair, использующей бэкенд Genesis."""

    backend: str = "genesis"

    device_settings: DeviceConfig = field(default_factory=DeviceConfig)

    # надо подумать
    # episode_length_s: float = 20.0
    # resampling_time_s: float = 20.0
    # goal_resample_distance_threshold: float = 1.3
    # simulate_action_latency: bool = True

    # robot: RobotInstanceConfig = 
    # corridor: GenesisCorridorConfig = 
    # goal: GoalConfig = 

    # main_grid: GridBaseConfig = field(
    #     default_factory=lambda: GridBaseConfig(
    #         enabled=True,
    #         cells=(39, 39, 1),
    #         channels=["occupancy"],
    #     )
    # )

    # reward_weights: Dict[str, float] = field(
    #     default_factory=lambda: {
    #         "action_rate": -0.0001,
    #         "goal_dist": -0.1,
    #         "relative_angle": -100.0,
    #         "speed_towards_goal": 100.0,
    #         "collision": -500.0,
    #     }
    # )

    # control_frequency: int = 50
    # simulation_frequency: int = 100

    # def __post_init__(self):
    #     super().__post_init__()

    #     if "main_grid" not in self.grids:
    #         self.grids["main_grid"] = self.main_grid
    #     if hasattr(self, "main_grid"):
    #         delattr(self, "main_grid")

    @property
    def device(self) -> str:
        """Возвращает строку устройства из вложенных настроек для удобства."""
        return self.device_settings.device
