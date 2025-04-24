import numpy as np
import torch  # Нужен torch для векторизации
from typing import Tuple

import genesis as gs


class Arch:
    """
    Арка из ДВУХ боксов ФИКСИРОВАННОГО размера для N сред (векторизовано).
    reset_idx рандомизирует проем (gap) и смещение (shift) для выбранных сред.
    """

    def __init__(
        self,
        scene: gs.Scene,
        center_pos: Tuple[float, float, float],
        orientation: str,
        pillar_width: float,
        min_gap_width: float = 0.8,
        max_gap_width: float = 1.2,
        min_shift: float = -0.5,
        max_shift: float = 2.0,
        thickness: float = 0.5,
        height: float = 2.0,
        device: torch.device = torch.device("cuda"),
    ):
        self.scene = scene
        self.device = device
        # Базовый центр (можно сделать тензором num_envs x 3, если он разный для сред)
        self.center_pos = torch.tensor(
            center_pos, dtype=gs.tc_float, device=self.device
        )  # shape [3]

        self.orientation = orientation
        self.pillar_width = pillar_width
        self.thickness = thickness
        self.height = height

        # Диапазоны рандомизации
        self.min_gap = min_gap_width
        self.max_gap = max_gap_width
        self.min_shift = min_shift
        self.max_shift = max_shift
        if not (0 <= self.min_gap <= self.max_gap):
            raise ValueError("Check gap ranges")
        if not (self.min_shift <= self.max_shift):
            raise ValueError("Check shift ranges")

        # Фиксированный размер бокса (стойки)
        self.box_size_tuple = (
            (self.pillar_width, self.thickness, self.height)
            if orientation == "y"
            else (self.thickness, self.pillar_width, self.height)
        )

        box_params = {
            "size": self.box_size_tuple,
            "fixed": True,
            "pos": (0, 0, -1000),
        }
        self.box1 = self.scene.add_entity(gs.morphs.Box(**box_params))  # Правая стойка
        self.box2 = self.scene.add_entity(gs.morphs.Box(**box_params))  # Левая стойка

    def _calculate_target_positions(
        self, gap_widths: torch.Tensor, shifts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Векторно вычисляет тензоры позиций для box1 и box2."""
        num_targets = len(gap_widths)
        half_gaps = gap_widths / 2.0
        half_pillar = self.pillar_width / 2.0

        offsets1 = shifts + half_gaps + half_pillar
        offsets2 = shifts - half_gaps - half_pillar

        # Расширяем базовый center_pos до нужного размера
        base_centers = self.center_pos.unsqueeze(0).expand(num_targets, -1)

        pos1 = base_centers.clone()
        pos2 = base_centers.clone()

        if self.orientation == "y":  # Смещение по X (index 0)
            pos1[:, 0] += offsets1
            pos2[:, 0] += offsets2
        else:  # Смещение по Y (index 1)
            pos1[:, 1] += offsets1
            pos2[:, 1] += offsets2
        # Z остается из base_centers

        return pos1, pos2

    def reset_idx(self, envs_idx):
        """Рандомизирует gap и shift для envs_idx и обновляет позиции боксов."""
        if len(envs_idx) == 0:
            return

        num_resets = len(envs_idx)

        # Генерируем случайные параметры
        rand_gaps = (
            torch.rand(num_resets, device=self.device, dtype=gs.tc_float)
            * (self.max_gap - self.min_gap)
            + self.min_gap
        )
        rand_shifts = (
            torch.rand(num_resets, device=self.device, dtype=gs.tc_float)
            * (self.max_shift - self.min_shift)
            + self.min_shift
        )

        # Вычисляем целевые позиции
        pos1_tensor, pos2_tensor = self._calculate_target_positions(
            rand_gaps, rand_shifts
        )

        self.box1.set_pos(pos1_tensor, envs_idx=envs_idx)
        self.box2.set_pos(pos2_tensor, envs_idx=envs_idx)

    def __repr__(self):
        return (
            f"Arch(num_envs={self.num_envs}, pillar_w={self.pillar_width}, "
            f"gap_range=[{self.min_gap},{self.max_gap}], shift_range=[{self.min_shift},{self.max_shift}])"
        )
