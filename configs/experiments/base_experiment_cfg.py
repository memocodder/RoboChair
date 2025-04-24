# configs/experiments/base.py
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import time

from ..environments.base_environment_cfg import EnvBaseConfig
from ..algorithms.base_algo_cfg import BaseAlgoConfig


@dataclass
class LoggingConfig:
    """Параметры логирования (универсальные)."""

    level: str = "INFO"
    log_to_file: bool = True
    # Шаблон имени файла лога ({experiment_name}, {timestamp} будут заменены)
    # Каталог logs/ должен существовать или создаваться скриптом
    log_file_pattern: str = "logs/{experiment_name}_{timestamp}.log"
    log_dir: str = "results/logs/{experiment_name}"
    log_to_wandb: bool = False
    wandb_project: Optional[str] = "RobochairRL"
    wandb_entity: Optional[str] = None


@dataclass
class ExperimentConfig:
    """
    Базовая структура для конфигурации одного эксперимента.
    Объединяет среду, алгоритм (который содержит свои параметры запуска, если применимо)
    и настройки логирования/сида.
    """

    env: EnvBaseConfig
    algo: BaseAlgoConfig
    
    # Уникальное имя для этого запуска (для логов, сохранений и т.д.)
    experiment_name: str
    # Метка времени для уникальности (генерируется автоматически)
    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y%m%d_%H%M%S")
    )
    # Глобальный сид для воспроизводимости эксперимента
    seed: int = 42


    logging: LoggingConfig = field(default_factory=LoggingConfig)
