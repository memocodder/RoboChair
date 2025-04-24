import torch
import math
import genesis as gs


def relative_angle(agent_euler, agent_pos, relative_pos):
    goal_pos = relative_pos[:, :2]  # [N, 2]
    z_angle = agent_euler[:, -1]  # [N]
    agent_pos = agent_pos[:, :2]

    # Преобразуем угол из градусов в радианы
    z_angle_rad = torch.deg2rad(z_angle)  # [N]

    # Добавляем смещение, если z_angle = 90° когда робот смотрит на цель
    offset_rad = torch.deg2rad(torch.tensor(-90.0))  # Скаляр
    z_angle_rad_adjusted = z_angle_rad - offset_rad  # [N]

    # Вектор от робота к цели
    delta = goal_pos - agent_pos  # [N, 2]
    # Угол направления на цель для всего батча
    target_angle = torch.atan2(delta[:, 1], delta[:, 0])  # [N]

    # Разница углов с учётом круговой природы
    angle_diff = torch.atan2(
        torch.sin(target_angle - z_angle_rad_adjusted),
        torch.cos(target_angle - z_angle_rad_adjusted),
    )  # [N]

    # Нормализуем в диапазон [-1, 1]
    normalized_angle = angle_diff / torch.pi  # [N]

    return normalized_angle.unsqueeze(1)  # [N, 1]


def check_agent_collision(agent, threshold_idx) -> torch.Tensor:

    agent_idx = agent.idx
    contact_info = agent.get_contacts()

    valid_mask = contact_info["valid_mask"]
    geom_a = contact_info["geom_a"]
    geom_b = contact_info["geom_b"]

    # 1. Находим индекс партнера по контакту
    partner_geom_idx = torch.where(geom_a == agent_idx, geom_b, geom_a)

    # 2. Формируем маску целевых столкновений: контакт валиден И индекс партнера > порога
    is_target_collision = valid_mask & (partner_geom_idx > threshold_idx)

    # 3. Проверяем, было ли хотя бы одно такое столкновение для каждого env
    result_per_env = torch.any(is_target_collision, dim=1)

    return result_per_env


def calculate_behind_camera_pos(
    agent_pos: torch.Tensor,
    agent_euler: torch.Tensor,
    distance: float = 5,
    height_offset: float = 3,
) -> torch.Tensor:
    """
    Вычисляет позицию камеры сзади и сверху от цели по ее позе.
    Предполагает Z-up систему координат и yaw - последний элемент robot_euler.

    Args:
        robot_pos: Тензор позиции цели (формат [3], ожидается на CPU).
        robot_euler: Тензор углов Эйлера цели в градусах (формат [3], ожидается на CPU).
                     Yaw (вращение вокруг Z) - предполагается последним элементом.
        distance: Желаемое горизонтальное расстояние позади цели.
        height_offset: Желаемое вертикальное смещение (по Z) над целью.

    Returns:
        Вычисленный тензор позиции камеры (формат [3]).
    """

    yaw_degrees = agent_euler[-1]
    yaw_rad = torch.deg2rad(yaw_degrees)

    # offset_x = -distance * torch.cos(yaw_rad)
    # offset_y = -distance * torch.sin(yaw_rad)
    # offset_y = distance * torch.sin(yaw_rad)
    offset_x =  distance * torch.sin(yaw_rad)
    offset_y = -distance * torch.cos(yaw_rad)

    cam_x = agent_pos[0] + offset_x
    cam_y = agent_pos[1] + offset_y
    cam_z = agent_pos[2] + height_offset

    cam_pos_tensor = torch.stack([cam_x, cam_y, cam_z])

    return cam_pos_tensor
