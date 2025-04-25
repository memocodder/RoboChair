# configs/base/device_config.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DeviceConfig:
    """
    Базовая и максимально простая конфигурация устройства.
    """
    device: str = "cuda"

    # В дальнейшем можно будет подумать над мапингом