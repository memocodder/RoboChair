import taichi as ti
import torch
import numpy as np


@ti.func
def quat_to_angle(qw, qx, qy, qz) -> ti.f32:  # type: ignore
    # --- Расчет Yaw (вращение вокруг Z) ---
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = ti.atan2(siny_cosp, cosy_cosp)
    return yaw


@ti.data_oriented
class Grid:
    def __init__(
        self,
        scene,
        ignore_idx,
        num_envs=1,
        x_interval_list=np.arange(-2, 2, 0.1),
        y_interval_list=np.arange(0, 4, 0.1),
        z_interval_list=np.array([0.01, 1.0]),
    ):
        self.scene = scene
        self.num_envs = num_envs

        num_aabb = self.scene.rigid_solver.geoms_state.aabb_min.shape[0]

        ignore_set = set(ignore_idx)
        valid_indices_list = [i for i in range(num_aabb) if i not in ignore_set]
        valid_indices_np = np.array(valid_indices_list, dtype=np.int32)

        self.num_valid_objects = len(valid_indices_np)
        if self.num_valid_objects > 0:
            self.valid_indices_field = ti.field(
                dtype=ti.i32, shape=self.num_valid_objects
            )
            self.valid_indices_field.from_numpy(valid_indices_np)
        else:
            raise "Нет объектов для зрения!"

        self.x_intervals = ti.field(dtype=ti.f32, shape=len(x_interval_list))
        self.y_intervals = ti.field(dtype=ti.f32, shape=len(y_interval_list))
        self.z_intervals = ti.field(dtype=ti.f32, shape=len(z_interval_list))

        self.x_intervals.from_numpy(np.array(x_interval_list, dtype=np.float32))
        self.y_intervals.from_numpy(np.array(y_interval_list, dtype=np.float32))
        self.z_intervals.from_numpy(np.array(z_interval_list, dtype=np.float32))

        self.shape = (
            self.num_envs,
            self.x_intervals.shape[0] - 1,
            self.y_intervals.shape[0] - 1,
            self.z_intervals.shape[0] - 1,
        )
        self._grid_buffer = torch.zeros(*self.shape, dtype=torch.int8, device="cuda")

    @ti.func
    def calul_colision(self, cell_world_corners, obj_aabb_min_w, obj_aabb_max_w):

        result = ti.int8(0)  # По умолчанию - ни один угол не внутри

        # Перебираем все 8 углов ячейки (строки матрицы)
        for i in ti.static(range(8)):
            # Координаты текущего угла
            px = cell_world_corners[i, 0]
            py = cell_world_corners[i, 1]
            pz = cell_world_corners[i, 2]

            # Проверяем, находится ли угол ВНУТРИ AABB объекта по каждой оси
            # (включая границы AABB)
            is_inside_x = (obj_aabb_min_w[0] <= px) and (px <= obj_aabb_max_w[0])
            is_inside_y = (obj_aabb_min_w[1] <= py) and (py <= obj_aabb_max_w[1])
            is_inside_z = (obj_aabb_min_w[2] <= pz) and (pz <= obj_aabb_max_w[2])

            # Если угол внутри по всем трем осям
            if is_inside_x and is_inside_y and is_inside_z:
                result = ti.int8(1)  # Устанавливаем флаг пересечения
                # Поскольку нам достаточно ОДНОГО угла внутри,
                # можно прервать цикл для экономии вычислений
                # break

        # Возвращаем результат: 1 если хотя бы один угол был внутри, иначе 0
        return result

    @ti.func
    def cell_points(self, x, y, angle, i, j, k):
        world_corners = ti.Matrix.zero(dt=ti.f32, n=8, m=3)

        cos_theta = ti.cos(angle)
        sin_theta = ti.sin(angle)

        cell_min_local = ti.Vector(
            [self.x_intervals[i], self.y_intervals[j], self.z_intervals[k]]
        )
        cell_max_local = ti.Vector(
            [self.x_intervals[i + 1], self.y_intervals[j + 1], self.z_intervals[k + 1]]
        )

        corner_idx = 0
        for ix in ti.static(range(2)):
            for iy in ti.static(range(2)):
                for iz in ti.static(range(2)):
                    # а) Локальные координаты угла (смещение отн. агента)
                    local_x = cell_min_local[0] if ix == 0 else cell_max_local[0]
                    local_y = cell_min_local[1] if iy == 0 else cell_max_local[1]
                    local_z = cell_min_local[2] if iz == 0 else cell_max_local[2]

                    # б) Применяем 2D вращение к локальным XY вокруг локального нуля (0,0)
                    x_rotated = local_x * cos_theta - local_y * sin_theta
                    y_rotated = local_x * sin_theta + local_y * cos_theta

                    # в) Смещаем в мировые координаты, добавляя позицию агента 'pos'
                    world_x = x_rotated + x
                    world_y = y_rotated + y
                    world_z = local_z  #  + pos[2] # Z просто смещается

                    # г) Сохраняем результат
                    world_corners[corner_idx, 0] = world_x
                    world_corners[corner_idx, 1] = world_y
                    world_corners[corner_idx, 2] = world_z
                    corner_idx += 1

        return world_corners

    @ti.kernel
    def _step(self, _grid_buffer: ti.types.ndarray(), aabb_min: ti.template(), aabb_max: ti.template(), pos: ti.types.ndarray(), rot: ti.types.ndarray()):  # type: ignore
        # for b, o, i, j, k in ti.ndrange(self.shape[0], aabb_min.shape[0], *self.shape[1:]):
        for b, o_idx, i, j, k in ti.ndrange(
            self.shape[0], self.num_valid_objects, *self.shape[1:]
        ):
            o = self.valid_indices_field[o_idx]
            x, y, yaw = (
                pos[b, 0],
                pos[b, 1],
                quat_to_angle(rot[b, 0], rot[b, 1], rot[b, 2], rot[b, 3]),
            )
            cur_points = self.cell_points(x, y, yaw, i, j, k)
            _grid_buffer[b, i, j, k] |= self.calul_colision(
                cur_points, aabb_min[o, b], aabb_max[o, b]
            )

    def update(self, agent_pos, agent_quat):
        geoms_state = self.scene.rigid_solver.geoms_state
        data = (
            geoms_state.aabb_min,
            geoms_state.aabb_max,
            agent_pos,
            agent_quat,
        )

        self._grid_buffer.fill_(0)
        self._step(self._grid_buffer, *data)
        return self._grid_buffer

    def get_val(self):
        return self._grid_buffer
