# configs/environment/base.py

from dataclasses import dataclass, field
from typing import Tuple, List, Literal, Optional, Dict
import numpy as np


@dataclass
class GridBaseConfig:
    """
    Универсальная конфигурация для одного пространственного грида восприятия.

    Рассчитана на гибкую настройку различных гридов: от крупных обзорных
    (например, для основы робота) до небольших и точных (например, для
    манипуляторов или "щупалец"), используя параметры размера, смещения,
    разрешения и разбиения ячеек. Среда может использовать словарь таких
    конфигураций для определения нескольких именованных гридов одновременно.
    """

    enabled: bool = True

    # --- Физические параметры грида ---
    # Размеры грида в метрах: (Глубина X, Ширина Y, Высота Z)
    size_meters: Tuple[float, float, float] = (4.0, 4.0, 1.2)
    # Смещение начала координат грида [0,0,0] относительно
    # базовой точки отсчета робота/сенсора (X, Y, Z) в метрах.
    # Например, (0.1, -2.0, 0.0) означает на 0.1м вперед,
    # на 2м влево (т.е. центр грида по ширине на оси Y робота),
    # и на 0.6 метра вверх, сетка лежит на уровне оси Z робота (0.0).
    origin_offset: Tuple[float, float, float] = (0.1, -2.0, 0.6)

    # --- Параметры дискретизации (ячейки) ---
    # Количество ячеек по каждому измерению: (Глубина X, Ширина Y, Высота Z)
    cells: Tuple[int, int, int] = (20, 20, 1)  # Пример: 20x20 по X/Y, 1 по Z

    # --- Содержимое ячеек (Каналы) ---
    # Список имен каналов. Порядок определяет индекс канала в тензоре.
    # Например: ["obstacle", "goal"] -> тензор будет NxMxKx2
    channels: List[str] = field(default_factory=lambda: ["occupancy"])

    # --- Настройка разбиения по осям (по умолчанию - равномерное) ---
    depth_spacing_type: Literal["uniform", "custom_boundaries"] = "uniform"
    # Список границ ячеек по оси X (глубина) в метрах, относительно origin_offset[0].
    # Длина списка должна быть cells[0] + 1.
    # Например, для 2х ячеек и размера 4м: [0.0, 1.5, 4.0] (ячейки 0-1.5м и 1.5-4м)
    depth_boundaries_meters: Optional[List[float]] = None

    width_spacing_type: Literal["uniform", "custom_boundaries"] = "uniform"
    # Список границ ячеек по оси Y (ширина) в метрах, относительно origin_offset[1].
    # Длина списка должна быть cells[1] + 1.
    width_boundaries_meters: Optional[List[float]] = None

    height_spacing_type: Literal["uniform", "custom_boundaries"] = "uniform"
    # Список границ ячеек по оси Z (высота) в метрах, относительно origin_offset[2].
    # Длина списка должна быть cells[2] + 1.
    height_boundaries_meters: Optional[List[float]] = None

    def __post_init__(self):
        """Проверка корректности параметров после инициализации."""
        self._validate_boundaries(
            "depth",
            self.depth_spacing_type,
            self.depth_boundaries_meters,
            self.cells[0],
            self.size_meters[0],
        )
        self._validate_boundaries(
            "width",
            self.width_spacing_type,
            self.width_boundaries_meters,
            self.cells[1],
            self.size_meters[1],
        )
        self._validate_boundaries(
            "height",
            self.height_spacing_type,
            self.height_boundaries_meters,
            self.cells[2],
            self.size_meters[2],
        )

    def _validate_boundaries(
        self,
        dim_name: str,
        spacing_type: str,
        boundaries: Optional[List[float]],
        cell_count: int,
        dim_size: float,
    ):
        """Вспомогательная функция валидации границ."""
        if spacing_type == "custom_boundaries":
            if boundaries is None:
                raise ValueError(
                    f"GridConfig Error ({dim_name}): boundaries_meters must be provided when spacing_type is 'custom_boundaries'."
                )
            if len(boundaries) != cell_count + 1:
                raise ValueError(
                    f"GridConfig Error ({dim_name}): boundaries_meters length must be {cell_count + 1} for {cell_count} cells, but got {len(boundaries)}."
                )
            # Проверяем, что границы начинаются с 0 и заканчиваются размером грида по этой оси
            # (границы указываются относительно начала грида по этой оси)
            if not np.isclose(boundaries[0], 0.0):
                raise ValueError(
                    f"GridConfig Error ({dim_name}): First boundary in boundaries_meters must be 0.0, but got {boundaries[0]}."
                )
            if not np.isclose(boundaries[-1], dim_size):
                raise ValueError(
                    f"GridConfig Error ({dim_name}): Last boundary in boundaries_meters must match size_meters ({dim_size}), but got {boundaries[-1]}."
                )
            # Проверяем монотонность границ
            if not np.all(np.diff(boundaries) > 0):
                raise ValueError(
                    f"GridConfig Error ({dim_name}): boundaries_meters must be monotonically increasing."
                )

    def get_output_shape(self) -> Tuple[int, ...]:
        """Возвращает форму тензора для этого грида: (Глубина, Ширина, Высота, Каналы)."""
        if not self.enabled:
            return (0,)  # Или какой-то другой индикатор отключенного грида
        return (self.cells[0], self.cells[1], self.cells[2], len(self.channels))
