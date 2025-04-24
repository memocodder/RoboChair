# configs/robots/robochair_v1.py
from dataclasses import dataclass, field
from typing import List


@dataclass
class RobotDefinitionConfig:
    """
    Определение структуры и базовых свойств робота Robochair v1.
    Переиспользуется в разных средах.
    """

    name: str = "RobochairV1"
    urdf_path: str = "urdf/car/car.urdf"

    # Канонический список имен управляемых DoF в ожидаемом порядке
    dof_names: List[str] = field(
        default_factory=lambda: [
            "Component3_motor_tl",  # wheel 0 (индекс 0)
            "Component2_motor_tr",  # wheel 1 (индекс 1)
            "Component1_motor_r",  # wheel 2 (индекс 2)
            "Component1_motor_l",  # wheel 3 (индекс 3)
            "Component1_turn_l",  # turn 0  (индекс 4)
            "Component1_turn_r",  # turn 1  (индекс 5)
            "Component8_eye",  # eye 0   (индекс 6)
            "Component1_head",  # head 0  (индекс 7)
        ]
    )
    # default_collision_group: int = 1


# Создаем экземпляр по умолчанию для удобного импорта
ROBOCHAIR_V1_DEFINITION = RobotDefinitionConfig()
