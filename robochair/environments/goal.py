import torch
import numpy as np
import genesis as gs


class Goal:
    def __init__(
        self,
        scene,
        num_envs,
        device="cuda",
    ):

        self.device = torch.device(device)
        self.scene = scene
        self.num_envs = num_envs

        self.base_init_pos = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        self.goal = self.scene.add_entity(
            gs.morphs.Sphere(
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        num_resets = envs_idx.numel()
        offset, upper = 15, 20  # Требуется 0 < offset <= upper

        base_pos = torch.zeros((num_resets, 3), device=self.device, dtype=gs.tc_float)

        # Генерируем случайную *величину* в диапазоне [offset, upper)
        magnitude = offset + torch.rand((num_resets, 2), device=self.device) * (
            upper - offset
        )
        # Умножаем на случайно выбранный знак (+1 или -1)
        base_pos[:, :2] = magnitude * torch.where(
            torch.rand((num_resets, 2), device=self.device) < 0.5, 1.0, -1.0
        )

        self.goal.set_pos(base_pos, zero_velocity=True, envs_idx=envs_idx)

    def idx(self):
        return np.arange(self.goal.geom_start, self.goal.geom_end).tolist()

    def get_pos(self):
        return self.goal.get_pos()
