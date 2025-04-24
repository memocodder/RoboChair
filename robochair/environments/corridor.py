import torch
import numpy as np
import genesis as gs
from typing import Tuple, List, Optional  # Добавим типизацию для ясности


# Класс для одной арки
class Arch:
    def __init__(
        self,
        scene: gs.Scene,
        center_pos: Tuple[float, float, float],  # Центр *проема* арки
        orientation: str,  # 'x' или 'y' - вдоль какой оси идет коридор, который арка перекрывает
        corridor_width: float,  # Общая ширина, которую должна перекрыть арка (gap_between_boxes)
        gap_width: float,  # Ширина проема в арке
        thickness: float = 0.5,  # Толщина самой арки (вдоль направления коридора)
        height: float = 2.0,  # Высота арки
        min_side_width: float = 0.1,  # Минимальная ширина боковой стойки арки
    ):
        self.scene = scene
        self.center_pos = np.array(center_pos)
        self.orientation = orientation
        self.corridor_width = corridor_width
        self.gap_width = gap_width
        self.thickness = thickness
        self.height = height
        self.min_side_width = min_side_width

        self.box_center_z = center_pos[2]  # Используем Z из центра проема

        # Расчет размеров и позиций боксов арки
        self.half_gap = self.gap_width / 2.0

        params = self._calculate_params()

        self.box1 = self.scene.add_entity(
            gs.morphs.Box(pos=params['pos1'], size=params['size1'], fixed=True)
        )
        self.box2 = self.scene.add_entity(
            gs.morphs.Box(pos=params['pos2'], size=params['size2'], fixed=True)
        )

        # Сохраняем информацию об индексах геометрии для удобства
        self.geom_start = self.box1.geom_start
        self.geom_end = self.box2.geom_end  # Предполагаем, что индексы идут подряд

    def _box_size(self, width):
        return (
            (
                width,
                self.thickness,
            )
            if self.orientation == "y"
            else (
                self.thickness,
                width,
            )
        ) + (self.height,)
    
    def _box_pos(self, offset):
        return (
            (
                self.center_pos[0] + offset,
                self.center_pos[1],
            )
            if self.orientation == "y"
            else (
                self.center_pos[0],
                self.center_pos[1] + offset,
            )
        ) + (self.box_center_z,)

    def _calculate_params(self):

        total_side_width = self.corridor_width - self.gap_width
        flexible_width = total_side_width - 2 * self.min_side_width

        rand_fraction = np.random.rand()
        width1 = self.min_side_width + rand_fraction * flexible_width
        width2 = total_side_width - width1

        size1 = self._box_size(width1)
        size2 = self._box_size(width2)
        
        offset1 = self.half_gap + width2 / 2.0
        offset2 = -(self.half_gap + width1 / 2.0)

        pos1 = self._box_pos(offset1)
        pos2 = self._box_pos(offset2)

        return {"size1": size1, "pos1": pos1, "size2": size2, "pos2": pos2}

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # num_resets = envs_idx.numel()
        # offset, upper = 15, 20  # Требуется 0 < offset <= upper

        # base_pos = torch.zeros((num_resets, 3), device=self.device, dtype=gs.tc_float)

        # # Генерируем случайную *величину* в диапазоне [offset, upper)
        # magnitude = offset + torch.rand((num_resets, 2), device=self.device) * (
        #     upper - offset
        # )
        # # Умножаем на случайно выбранный знак (+1 или -1)
        # base_pos[:, :2] = magnitude * torch.where(
        #     torch.rand((num_resets, 2), device=self.device) < 0.5, 1.0, -1.0
        # )

        # self.goal.set_pos(base_pos, zero_velocity=False, envs_idx=envs_idx)

    def __repr__(self):
        return (
            f"Arch(center={self.center_pos}, orientation='{self.orientation}', "
            f"gap={self.gap_width}, thickness={self.thickness})"
        )
from .arch import Arch

# Обновленный класс коридора
class Corridor:
    def __init__(
        self,
        scene: gs.Scene,
        num_envs: int,
        arches_using: bool,
        device: str = "cuda",
    ):
        self.arches_using = arches_using

        # Параметры основных стен
        box_width = 10.0
        box_length = 10.0
        box_height = 2.0
        gap_between_boxes = 3.0

        # Параметры арок
        arch_thickness = 0.5  # Толщина арок
        min_arch_side_width = 0.2  # Мин. ширина стойки арки

        arch_distances_x = [
            4.0,
            7.0,
            10.0,
        ]  # Добавим арки в X-коридорах
        arch_distances_y = [
            4.0,
            7.0,
            10.0,
        ]
        arch_gap_width = 2.0  # Проем в арках шириной 1.0

        self.device = torch.device(device)
        self.scene = scene
        self.num_envs = num_envs
        self.entities = (
            []
        )  # Список для хранения всех созданных сущностей (стены и арки)

        # --- Создание основных стен ---
        box_center_z = box_height / 2.0
        half_size_x = box_width / 2.0
        half_size_y = box_length / 2.0
        half_gap = gap_between_boxes / 2.0
        center_offset_x = half_size_x + half_gap
        center_offset_y = half_size_y + half_gap

        pos_top_right = (center_offset_x, center_offset_y, box_center_z)
        pos_top_left = (-center_offset_x, center_offset_y, box_center_z)
        pos_bottom_left = (-center_offset_x, -center_offset_y, box_center_z)
        pos_bottom_right = (center_offset_x, -center_offset_y, box_center_z)

        box_size = (box_width, box_length, box_height)

        self.box_wall_1 = self.scene.add_entity(
            gs.morphs.Box(pos=pos_top_right, size=box_size, fixed=True)
        )
        self.box_wall_2 = self.scene.add_entity(
            gs.morphs.Box(pos=pos_top_left, size=box_size, fixed=True)
        )
        self.box_wall_3 = self.scene.add_entity(
            gs.morphs.Box(pos=pos_bottom_left, size=box_size, fixed=True)
        )
        self.box_wall_4 = self.scene.add_entity(
            gs.morphs.Box(pos=pos_bottom_right, size=box_size, fixed=True)
        )
        self.entities.extend(
            [self.box_wall_1, self.box_wall_2, self.box_wall_3, self.box_wall_4]
        )
        if self.arches_using:
            # --- Создание арок ---
            self.arches = []

            # Арки вдоль оси X (в коридорах, идущих по Y)
            if arch_distances_x:
                for dist in arch_distances_x:
                    # Положительное направление X
                    center_pos_px = (dist, 0.0, box_center_z)
                    arch_px = Arch(
                        scene=scene,
                        device=device,
                        center_pos=center_pos_px,
                        orientation="x",
                        pillar_width=gap_between_boxes,
                    )
                    self.arches.append(arch_px)
                    self.entities.extend([arch_px.box1, arch_px.box2])

                    # Отрицательное направление X
                    center_pos_nx = (-dist, 0.0, box_center_z)
                    arch_nx = Arch(
                        scene=scene,
                        device=device,
                        center_pos=center_pos_nx,
                        orientation="x",
                        pillar_width=gap_between_boxes,
                    )
                    self.arches.append(arch_nx)
                    self.entities.extend([arch_nx.box1, arch_nx.box2])

            # Арки вдоль оси Y (в коридорах, идущих по X)
            if arch_distances_y:
                for dist in arch_distances_y:
                    # Положительное направление Y
                    center_pos_py = (0.0, dist, box_center_z)
                    arch_py = Arch(
                        scene=scene,
                        device=device,
                        center_pos=center_pos_py,
                        orientation="y",
                        pillar_width=gap_between_boxes,
                    )
                    self.arches.append(arch_py)
                    self.entities.extend([arch_py.box1, arch_py.box2])

                    # Отрицательное направление Y
                    center_pos_ny = (0.0, -dist, box_center_z)
                    arch_ny = Arch(
                        scene=scene,
                        device=device,
                        center_pos=center_pos_ny,
                        orientation="y",
                        pillar_width=gap_between_boxes,
                    )
                    self.arches.append(arch_ny)
                    self.entities.extend([arch_ny.box1, arch_ny.box2])
        # if arches_using:
        #     # --- Создание арок ---
        #     self.arches = []
        #     arch_height = box_height  # Используем ту же высоту, что и у стен

        #     # Арки вдоль оси X (в коридорах, идущих по Y)
        #     if arch_distances_x:
        #         for dist in arch_distances_x:
        #             # Положительное направление X
        #             center_pos_px = (dist, 0.0, box_center_z)
        #             arch_px = Arch(
        #                 scene=scene,
        #                 center_pos=center_pos_px,
        #                 orientation="x",  # Перекрывает коридор вдоль X
        #                 corridor_width=gap_between_boxes,  # Ширина коридора = зазор между стенами
        #                 gap_width=arch_gap_width,
        #                 thickness=arch_thickness,
        #                 height=arch_height,
        #                 min_side_width=min_arch_side_width,
        #             )
        #             self.arches.append(arch_px)
        #             self.entities.extend([arch_px.box1, arch_px.box2])

        #             # Отрицательное направление X
        #             center_pos_nx = (-dist, 0.0, box_center_z)
        #             arch_nx = Arch(
        #                 scene=scene,
        #                 center_pos=center_pos_nx,
        #                 orientation="x",
        #                 corridor_width=gap_between_boxes,
        #                 gap_width=arch_gap_width,
        #                 thickness=arch_thickness,
        #                 height=arch_height,
        #                 min_side_width=min_arch_side_width,
        #             )
        #             self.arches.append(arch_nx)
        #             self.entities.extend([arch_nx.box1, arch_nx.box2])

        #     # Арки вдоль оси Y (в коридорах, идущих по X)
        #     if arch_distances_y:
        #         for dist in arch_distances_y:
        #             # Положительное направление Y
        #             center_pos_py = (0.0, dist, box_center_z)
        #             arch_py = Arch(
        #                 scene=scene,
        #                 center_pos=center_pos_py,
        #                 orientation="y",  # Перекрывает коридор вдоль Y
        #                 corridor_width=gap_between_boxes,
        #                 gap_width=arch_gap_width,
        #                 thickness=arch_thickness,
        #                 height=arch_height,
        #                 min_side_width=min_arch_side_width,
        #             )
        #             self.arches.append(arch_py)
        #             self.entities.extend([arch_py.box1, arch_py.box2])

        #             # Отрицательное направление Y
        #             center_pos_ny = (0.0, -dist, box_center_z)
        #             arch_ny = Arch(
        #                 scene=scene,
        #                 center_pos=center_pos_ny,
        #                 orientation="y",
        #                 corridor_width=gap_between_boxes,
        #                 gap_width=arch_gap_width,
        #                 thickness=arch_thickness,
        #                 height=arch_height,
        #                 min_side_width=min_arch_side_width,
        #             )
        #             self.arches.append(arch_ny)
        #             self.entities.extend([arch_ny.box1, arch_ny.box2])

        # Запоминаем последнюю добавленную сущность для метода idx()
        self.last_entity = self.entities[-1] if self.entities else None

    def reset_idx(self, envs_idx):
        # Эта функция пока не делает ничего с коридором/арками, т.к. они фиксированы
        if len(envs_idx) == 0:
            return
        
        if self.arches_using:
            for arch in self.arches:
                arch.reset_idx(envs_idx)

    def idx(self):
        # Возвращает список индексов геометрии для всех стен и арок
        # Простейший вариант, если индексы идут подряд:
        if not self.entities:
            return []
        first_geom_idx = self.entities[0].geom_start
        last_geom_idx = self.last_entity.geom_end
        return list(range(first_geom_idx, last_geom_idx))

        # Более надежный, но медленный вариант, если индексы могут быть не подряд:
        # all_indices = []
        # for entity in self.entities:
        #     all_indices.extend(list(range(entity.geom_start, entity.geom_end)))
        # return all_indices

    def get_pos(self):
        # Эта функция должна возвращать позицию цели, которой здесь нет.
        # Если вам нужна позиция какой-то части коридора, нужно это уточнить.
        # Например, позиция центра первой стены:
        # return self.box_wall_1.get_pos()
        # Пока возвращаем None или выбрасываем ошибку
        # raise NotImplementedError("get_pos is not defined for the Corridor itself. Define a goal entity.")
        return None
