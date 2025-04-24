# configs/algorithm/base.py
from dataclasses import dataclass, field


@dataclass
class BaseAlgoConfig:
    """Базовый класс для конфигураций алгоритмов (локальное определение)."""

    name: str = "BaseAlgorithm"
