import torch
import numpy as np
import math
import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
    xyz_to_quat,
)
from .utils import relative_angle


class RobotBuffer:

    def __init__(
        self,
        robot_ctx,
    ):
        num_envs = robot_ctx.num_envs
        device = robot_ctx.device
        num_joints = robot_ctx.num_joints
        self.motor_dofs = sorted(robot_ctx.motor_dofs)
        self.robot = robot_ctx.robot

        # База
        self.base_lin_vel = torch.zeros((num_envs, 3), device=device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((num_envs, 3), device=device, dtype=gs.tc_float)
        self.base_pos = torch.zeros((num_envs, 3), device=device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((num_envs, 4), device=device, dtype=gs.tc_float)
        self.base_euler = torch.zeros((num_envs, 3), device=device, dtype=gs.tc_float)
        self.base_vel = torch.zeros((num_envs, 3), device=device, dtype=gs.tc_float)


        self.relative_angle = torch.zeros((num_envs, 1), device=device, dtype=gs.tc_float)

        # Гравитация
        self.projected_gravity = torch.zeros(
            (num_envs, 3), device=device, dtype=gs.tc_float
        )
        self.global_gravity = torch.tensor(
            [0.0, 0.0, -1.0], device=device, dtype=gs.tc_float
        ).repeat(num_envs, 1)

        # Суставы (DoF - Degrees of Freedom)
        self.dof_pos = torch.zeros(
            (num_envs, num_joints), device=device, dtype=gs.tc_float
        )
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        # Позиции по умолчанию (одни для всех)
        self.default_dof_pos = torch.zeros(num_joints, device=device, dtype=gs.tc_float)

    def update(self, relative_pos=None):
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler[:] = quat_to_xyz(self.base_quat)
        self.base_vel[:] = self.robot.get_vel()
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        if relative_pos is not None:
            self.relative_angle[:] = relative_angle(self.base_euler, self.base_pos, relative_pos)


class Robot:
    def __init__(
        self,
        scene,
        num_envs,
        device="cuda",
    ):

        self.device = torch.device(device)
        self.scene = scene
        self.num_envs = num_envs

        self.num_actions = 4
        self.clip_actions = 1.0
        self.termination_if_roll_greater_than = 45
        self.termination_if_pitch_greater_than = 45

        self.kp = 80.0
        self.kd = 2.0

        self.base_init_pos = torch.tensor([0.0, 0.0, 0.1], device=self.device)
        self.base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="/home/o/Downloads/URDF/car/car/car.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        self.dof_names = [
            "Component3_motor_tl",
            "Component2_motor_tr",
            "Component1_motor_r",
            "Component1_motor_l",
            "Component1_turn_l",
            "Component1_turn_r",
            "Component8_eye",
            "Component1_head",
        ]

        self.num_joints = len(self.dof_names)

        self.motor_dofs = [
            self.robot.get_joint(name).dof_idx_local for name in self.dof_names
        ]

        self.wheel_idx = sorted(self.motor_dofs[0:4])
        self.turn_idx = sorted(self.motor_dofs[4:6])
        self.head_idx = sorted(self.motor_dofs[6:7])
        self.eye_idx = sorted(self.motor_dofs[7:8])

        self.wheel_scale = torch.tensor(
            [30, -30, -30, 30], dtype=gs.tc_float, device=self.device
        )
        self.turn_scale = torch.tensor([1, 1], dtype=gs.tc_float, device=self.device)
        self.head_scale = torch.tensor([3.1], dtype=gs.tc_float, device=self.device)
        self.eye_scale = torch.tensor([1], dtype=gs.tc_float, device=self.device)

        self.buffer = RobotBuffer(self)

    def build(self):
        self.robot.set_dofs_kp([self.kp] * self.num_joints, self.motor_dofs)
        self.robot.set_dofs_kv([self.kd] * self.num_joints, self.motor_dofs)

    def dead_idx(self):
        is_dead = (
            torch.abs(self.buffer.base_euler[:, 1]) > self.termination_if_pitch_greater_than
        )
        is_dead |= (
            torch.abs(self.buffer.base_euler[:, 0]) > self.termination_if_roll_greater_than
        )
        return is_dead.nonzero(as_tuple=False).flatten()

    def step(self, actions):
        exec_actions = torch.clip(
            actions,
            -self.clip_actions,
            self.clip_actions,
        ).unsqueeze(-1)

        # exec_actions = actions.unsqueeze(-1)

        self.robot.control_dofs_force(
            self.wheel_scale * exec_actions[:, 0],
            self.wheel_idx,
        )
        self.robot.control_dofs_position(
            self.turn_scale * exec_actions[:, 1],
            self.turn_idx,
        )
        self.robot.control_dofs_position(
            self.eye_scale * exec_actions[:, 2],
            self.head_idx,
        )
        self.robot.control_dofs_position(
            self.head_scale * exec_actions[:, 3],
            self.eye_idx,
        )

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        num_resets = envs_idx.numel()

        # reset dofs
        dof_pos = torch.zeros(
            (num_resets, self.num_joints),
            device=self.device,
            dtype=gs.tc_float,
        )

        self.robot.set_dofs_position(
            position=dof_pos,
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        base_pos = torch.zeros((num_resets, 3), device=self.device, dtype=gs.tc_float)
        base_pos[:] = self.base_init_pos
        self.robot.set_pos(base_pos, zero_velocity=False, envs_idx=envs_idx)

        random_yaw_angles = (
            torch.rand(num_resets, device=self.device) * 2.0 - 1.0
        ) * torch.pi
        euler_angles_xyz = torch.zeros(num_resets, 3, device=self.device)
        euler_angles_xyz[:, 2] = random_yaw_angles
        random_quaternions = xyz_to_quat(euler_angles_xyz, rpy=True, degrees=False)

        self.robot.set_quat(random_quaternions, zero_velocity=False, envs_idx=envs_idx)
        self.robot.zero_all_dofs_velocity(envs_idx)

    def update_buffer(self, relative_pos=None):
        self.buffer.update(relative_pos=relative_pos)
            

    def idx(self):
        return np.arange(self.robot.geom_start, self.robot.geom_end).tolist()
    
    # def get_turn_pos(self):
    #     return self.buffer.dof_pos[:, self.turn_idx]
    
    # def get_wheel_vel(self):
    #     return self.buffer.dof_vel[:, self.wheel_idx]

    def get_pos(self):
        return self.buffer.base_pos

    def get_global_vel(self):
        return self.buffer.base_vel
